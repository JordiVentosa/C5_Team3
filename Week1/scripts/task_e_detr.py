import argparse
import inspect
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import albumentations as A
from PIL import Image
from tqdm import tqdm

from pycocotools.coco import COCO

from transformers import (
    AutoImageProcessor,
    AutoModelForObjectDetection,
    TrainingArguments,
    Trainer,
)

from Week1.src.models.huggingface_detr import draw_detections
from Week1.src.utils.kitti_helpers import (
    KITTI_CLASS_TO_TRAIN_ID,
    list_training_pairs,
    list_kitti_rgb_images,
    rel_from_split,
    sanitize_coco_bboxes,
    format_annotations_for_processor,
    coco_eval_bbox,
)

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

SEED = 42
ID2LABEL = {0: "car", 1: "person"}
LABEL2ID = {"car": 0, "person": 1}
EVAL_CAT_IDS = [0, 1]
TRAIN_CATEGORIES = [
    {"id": 0, "name": "car", "supercategory": "vehicle"},
    {"id": 1, "name": "person", "supercategory": "person"},
]


def get_args():
    parser = argparse.ArgumentParser(description="Task E: Fine-tune DeTR on KITTI-MOTS")
    parser.add_argument("--kitti_root", type=str, default="/home/mcv/datasets/C5/KITTI-MOTS")
    parser.add_argument("--checkpoint", type=str, default="facebook/detr-resnet-50")
    parser.add_argument("--output_dir", type=str, default="./detr_finetune_kitti")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--train_bs", type=int, default=1)
    parser.add_argument("--eval_bs", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--image_size", type=int, default=480)
    parser.add_argument("--qualitative_split", type=str, default="testing")
    parser.add_argument("--max_qual_images", type=int, default=200)
    return parser.parse_args()


def extract_bboxes_from_instance_png(mask_path: Path) -> Tuple[List[int], List[List[float]]]:
    mask = np.array(Image.open(mask_path), dtype=np.int32)
    ids = np.unique(mask)
    ids = ids[ids != 0]

    cats, bboxes = [], []
    for inst_id in ids:
        class_id = int(inst_id // 1000)
        if class_id not in KITTI_CLASS_TO_TRAIN_ID:
            continue
        train_cat = KITTI_CLASS_TO_TRAIN_ID[class_id]

        ys, xs = np.where(mask == inst_id)
        if len(xs) == 0:
            continue

        x1, y1 = float(xs.min()), float(ys.min())
        x2, y2 = float(xs.max()) + 1.0, float(ys.max()) + 1.0
        w, h = x2 - x1, y2 - y1
        if w <= 1.0 or h <= 1.0:
            continue

        cats.append(int(train_cat))
        bboxes.append([x1, y1, w, h])

    return cats, bboxes


def get_train_transform():
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.4),
            A.HueSaturationValue(p=0.2),
            A.MotionBlur(p=0.1),
            A.Perspective(p=0.1),
        ],
        bbox_params=A.BboxParams(
            format="coco", label_fields=["category"],
            clip=True, min_area=9, min_visibility=0.01,
            check_each_transform=False,
        ),
    )


def get_val_transform():
    return A.Compose(
        [A.NoOp()],
        bbox_params=A.BboxParams(
            format="coco", label_fields=["category"],
            clip=True, min_area=1, min_visibility=0.0,
            check_each_transform=False,
        ),
    )


class KittiMotsDataset(torch.utils.data.Dataset):
    def __init__(self, pairs, transform, start_image_id=1):
        self.pairs = pairs
        self.transform = transform
        self.image_ids = list(range(start_image_id, start_image_id + len(pairs)))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        seq, rgb_path, mask_path = self.pairs[idx]
        image_id = self.image_ids[idx]

        img = Image.open(rgb_path).convert("RGB")
        img_np = np.array(img)
        h, w = img_np.shape[:2]

        cats, bboxes = extract_bboxes_from_instance_png(mask_path)
        cats, bboxes = sanitize_coco_bboxes(bboxes, cats, img_w=w, img_h=h)

        out = self.transform(image=img_np, bboxes=bboxes, category=cats)
        ann = format_annotations_for_processor(image_id, out["category"], out["bboxes"])

        return {"image": out["image"], "annotations": ann}


def build_coco_gt_from_pairs(pairs):
    images, annotations = [], []
    ann_id = 1

    for img_id, (seq, rgb_path, mask_path) in enumerate(pairs, start=1):
        with Image.open(rgb_path) as im:
            w, h = im.size
        images.append({"id": img_id, "file_name": str(rgb_path), "width": w, "height": h})

        cats, bboxes = extract_bboxes_from_instance_png(mask_path)
        for cat, bb in zip(cats, bboxes):
            x, y, bw, bh = map(float, bb)
            if bw <= 1.0 or bh <= 1.0:
                continue
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": int(cat),
                "bbox": [x, y, bw, bh],
                "area": float(bw * bh),
                "iscrowd": 0,
            })
            ann_id += 1

    gt_dict = {"images": images, "annotations": annotations, "categories": TRAIN_CATEGORIES}
    coco_gt = COCO()
    coco_gt.dataset = gt_dict
    coco_gt.createIndex()
    img_ids = [img["id"] for img in images]
    return coco_gt, img_ids


@torch.no_grad()
def run_val_preds(model, processor, pairs, device, threshold):
    model.eval()
    preds = []

    for img_id, (seq, rgb_path, mask_path) in tqdm(
        list(enumerate(pairs, start=1)), desc="Val predict", total=len(pairs),
    ):
        img = Image.open(rgb_path).convert("RGB")
        inputs = processor(images=img, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)

        target_sizes = torch.tensor([img.size[::-1]], device=device)
        results = processor.post_process_object_detection(
            outputs, threshold=threshold, target_sizes=target_sizes,
        )[0]

        boxes = results["boxes"].detach().cpu().numpy()
        scores = results["scores"].detach().cpu().numpy()
        labels = results["labels"].detach().cpu().numpy()

        for (x1, y1, x2, y2), s, lab in zip(boxes, scores, labels):
            lab = int(lab)
            if lab not in EVAL_CAT_IDS:
                continue
            w, h = float(x2 - x1), float(y2 - y1)
            if w <= 1.0 or h <= 1.0:
                continue
            preds.append({
                "image_id": int(img_id),
                "category_id": lab,
                "bbox": [float(x1), float(y1), w, h],
                "score": float(s),
            })
    return preds


def run_qualitative(model, processor, args, device):
    kitti_root = Path(args.kitti_root)
    output_dir = Path(args.output_dir + "_qualitative")
    output_dir.mkdir(parents=True, exist_ok=True)

    id2label = dict(model.config.id2label)
    keep_classes = {"car", "person"}

    img_paths = list_kitti_rgb_images(kitti_root, args.qualitative_split)
    print(f"Found {len(img_paths)} images in {args.qualitative_split}/image_02")

    if args.max_qual_images > 0:
        img_paths = img_paths[:args.max_qual_images]

    model.eval()
    with torch.no_grad():
        for p in tqdm(img_paths, desc="Qualitative inference"):
            img = Image.open(p).convert("RGB")
            inputs = processor(images=img, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)

            target_sizes = torch.tensor([img.size[::-1]], device=device)
            results = processor.post_process_object_detection(
                outputs, threshold=args.threshold, target_sizes=target_sizes,
            )[0]

            vis = draw_detections(img, results, id2label=id2label, keep_labels=keep_classes)
            rel = rel_from_split(p, kitti_root, args.qualitative_split)
            save_path = output_dir / args.qualitative_split / rel
            save_path.parent.mkdir(parents=True, exist_ok=True)
            vis.save(save_path.with_suffix(".png"))

    print("Qualitative results saved to:", output_dir.resolve())


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    kitti_root = Path(args.kitti_root)
    train_seqs = [f"{i:04d}" for i in range(0, 16)]
    val_seqs = [f"{i:04d}" for i in range(16, 21)]

    # Processor
    processor = AutoImageProcessor.from_pretrained(args.checkpoint, use_fast=False)
    processor.do_resize = True
    processor.size = {"shortest_edge": args.image_size, "longest_edge": 1333}
    processor.do_pad = True

    # Model
    model = AutoModelForObjectDetection.from_pretrained(
        args.checkpoint,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    ).to(device)

    # Datasets
    train_pairs = list_training_pairs(kitti_root, train_seqs)
    val_pairs = list_training_pairs(kitti_root, val_seqs)

    if not train_pairs or not val_pairs:
        raise RuntimeError("No train/val pairs found. Check KITTI_ROOT path.")

    print(f"Train frames: {len(train_pairs)}")
    print(f"Val   frames: {len(val_pairs)}")

    train_ds = KittiMotsDataset(train_pairs, transform=get_train_transform())
    val_ds = KittiMotsDataset(val_pairs, transform=get_val_transform())

    def collate_fn(batch):
        images = [x["image"] for x in batch]
        annotations = [x["annotations"] for x in batch]
        return processor(images=images, annotations=annotations, return_tensors="pt")

    # TrainingArguments
    ta_kwargs = dict(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        weight_decay=1e-4,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_bs,
        per_device_eval_batch_size=args.eval_bs,
        dataloader_num_workers=args.num_workers,
        fp16=(device.type == "cuda"),
        logging_steps=50,
        save_strategy="epoch",
        remove_unused_columns=False,
        eval_do_concat_batches=False,
        report_to="none",
    )

    ta_params = set(inspect.signature(TrainingArguments.__init__).parameters.keys())
    if "eval_strategy" in ta_params:
        ta_kwargs["eval_strategy"] = "epoch"
    else:
        ta_kwargs["evaluation_strategy"] = "epoch"

    ta_kwargs = {k: v for k, v in ta_kwargs.items() if k in ta_params}
    training_args = TrainingArguments(**ta_kwargs)

    # Trainer
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
        processing_class=processor,
        tokenizer=processor,
    )
    tr_params = set(inspect.signature(Trainer.__init__).parameters.keys())
    trainer_kwargs = {k: v for k, v in trainer_kwargs.items() if k in tr_params}
    trainer = Trainer(**trainer_kwargs)

    # Train
    trainer.train()
    print("Training done.")

    # COCO evaluation on val
    print("\nBuilding COCO GT for val...")
    coco_gt, val_img_ids = build_coco_gt_from_pairs(val_pairs)

    print("Running val predictions...")
    val_preds = run_val_preds(model, processor, val_pairs, device, args.threshold)

    coco_eval_bbox(
        coco_gt, val_preds, val_img_ids, EVAL_CAT_IDS,
        title=f"DeTR fine-tuned (train {train_seqs}, eval {val_seqs})",
    )

    # Save model
    trainer.save_model(args.output_dir)
    print("Saved model to:", args.output_dir)

    # Qualitative evaluation
    run_qualitative(model, processor, args, device)


if __name__ == "__main__":
    main()
