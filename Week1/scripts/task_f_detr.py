import argparse
import inspect
import json
import random


import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

import albumentations as A
from datasets import load_dataset, get_dataset_config_names

from transformers import (
    AutoImageProcessor,
    AutoModelForObjectDetection,
    TrainingArguments,
    Trainer,
)

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

SEED = 42

CLASS_MAPPING = {
    "person": 1, "nude": 1, "angel": 1, "knight": 1, "monk": 1,
    "crucifixion": 1, "god the father": 1, "shepherd": 1, "saturno": 1, "judith": 1,
}

NUM_LABELS = 2
ID2LABEL = {0: "unlabeled", 1: "target_class"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}


def get_args():
    parser = argparse.ArgumentParser(description="Task F: Fine-tune DeTR on DEArt")
    parser.add_argument("--checkpoint", type=str, default="facebook/detr-resnet-50")
    parser.add_argument("--dataset_name", type=str, default="biglam/european_art")
    parser.add_argument("--preferred_config", type=str, default="coco")
    parser.add_argument("--output_dir", type=str, default="./out_task_f_detr_deart")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--train_bs", type=int, default=2)
    parser.add_argument("--eval_bs", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--eval_threshold", type=float, default=0.05)
    return parser.parse_args()


# --- Normalization helpers ---

def safe_int(x, fallback: int) -> int:
    try:
        return int(x)
    except Exception:
        return int(fallback)


def build_global_cat_map(hf_ds):
    cat_map = {}
    if "annotations" in hf_ds.features:
        ann_feat = hf_ds.features["annotations"]
        if hasattr(ann_feat, "feature") and "category_id" in ann_feat.feature:
            names = ann_feat.feature["category_id"].names
            cat_map = {i: name.lower() for i, name in enumerate(names)}
    return cat_map


def normalize_example(ex, idx, global_hf_cat_map):
    img = ex["image"].convert("RGB")

    # Case A: native HF list
    if isinstance(ex.get("annotations", None), list):
        image_id = safe_int(ex.get("image_id", idx), idx)
        anns_out = []
        for a in ex["annotations"]:
            cat_id = a.get("category_id", None)
            bbox = a.get("bbox", None)
            if bbox is None or cat_id is None:
                continue
            cat_name = global_hf_cat_map.get(int(cat_id), "")
            if cat_name not in CLASS_MAPPING:
                continue
            x, y, w, h = map(float, bbox)
            if w <= 1 or h <= 1:
                continue
            anns_out.append({
                "bbox": [x, y, w, h],
                "category_id": 1,
                "iscrowd": int(bool(a.get("iscrowd", 0))),
                "area": float(a.get("area", w * h)),
            })
        return img, int(image_id), anns_out

    # Case B: JSON string
    ann_str = ex.get("annotations", None)
    if isinstance(ann_str, str):
        coco = json.loads(ann_str)
        images_list = coco.get("images", [])
        anns_list = coco.get("annotations", [])
        cats_list = coco.get("categories", [])
        local_cat_map = {c["id"]: c.get("name", "").lower() for c in cats_list}
        image_id = safe_int(images_list[0].get("id", idx), idx) if images_list else idx

        anns_out = []
        for a in anns_list:
            cat_id = a.get("category_id", None)
            bbox = a.get("bbox", None)
            if bbox is None or cat_id is None:
                continue
            cat_name = local_cat_map.get(cat_id, "")
            if cat_name not in CLASS_MAPPING:
                continue
            x, y, w, h = map(float, bbox)
            if w <= 1 or h <= 1:
                continue
            anns_out.append({
                "bbox": [x, y, w, h],
                "category_id": 1,
                "iscrowd": int(bool(a.get("iscrowd", 0))),
                "area": float(a.get("area", w * h)),
            })
        return img, int(image_id), anns_out

    return img, idx, []


def sanitize_xywh_bboxes(bboxes, cat_ids, img_w, img_h):
    clean_b, clean_c = [], []
    for (x, y, w, h), cid in zip(bboxes, cat_ids):
        x = max(0.0, min(float(x), img_w - 1.0))
        y = max(0.0, min(float(y), img_h - 1.0))
        w = max(0.0, min(float(w), img_w - x))
        h = max(0.0, min(float(h), img_h - y))
        if w <= 1.0 or h <= 1.0:
            continue
        clean_b.append([x, y, w, h])
        clean_c.append(int(cid))
    return clean_b, clean_c


# --- Transforms ---

def get_train_transform():
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.HueSaturationValue(p=0.2),
            A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.05, rotate_limit=5, border_mode=0, p=0.2),
        ],
        bbox_params=A.BboxParams(format="coco", label_fields=["category_ids"], min_area=4, min_visibility=0.2),
    )


def get_val_transform():
    return A.Compose(
        [],
        bbox_params=A.BboxParams(format="coco", label_fields=["category_ids"], min_area=4, min_visibility=0.0),
    )


# --- Dataset ---

class DeartDetrDataset(Dataset):
    def __init__(self, hf_ds, processor, global_cat_map, transform=None):
        self.ds = hf_ds
        self.processor = processor
        self.transform = transform
        self.global_cat_map = global_cat_map

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        ex = self.ds[idx]
        img, image_id, anns = normalize_example(ex, idx, self.global_cat_map)

        bboxes, cat_ids = [], []
        for a in anns:
            x, y, w, h = map(float, a["bbox"])
            if w <= 1 or h <= 1:
                continue
            bboxes.append([x, y, w, h])
            cat_ids.append(a["category_id"])

        if self.transform is not None and len(bboxes) > 0:
            img_np = np.array(img)
            out = self.transform(image=img_np, bboxes=bboxes, category_ids=cat_ids)
            img_np = out["image"]
            img = Image.fromarray(img_np)
            bboxes = list(out["bboxes"])
            cat_ids = list(out["category_ids"])
            h_img, w_img = img_np.shape[:2]
            bboxes, cat_ids = sanitize_xywh_bboxes(bboxes, cat_ids, w_img, h_img)

        new_anns = []
        for bbox, cid in zip(bboxes, cat_ids):
            x, y, w, h = map(float, bbox)
            if w <= 1 or h <= 1:
                continue
            new_anns.append({
                "bbox": [x, y, w, h],
                "category_id": int(cid),
                "iscrowd": 0,
                "area": float(w * h),
            })

        encoding = self.processor(
            images=img,
            annotations={"image_id": int(image_id), "annotations": new_anns},
            return_tensors="pt",
        )

        lbl = encoding["labels"][0]
        if "boxes" in lbl:
            lbl["boxes"] = torch.nan_to_num(lbl["boxes"], nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

        return {
            "pixel_values": encoding["pixel_values"].squeeze(0),
            "labels": lbl,
        }


# --- Evaluation ---

def build_coco_gt_from_val(val_hf, global_cat_map):
    gt_images, gt_annotations = [], []
    gt_categories = [{"id": 1, "name": ID2LABEL[1]}]
    ann_id = 1
    idx_to_coco_imgid = {}

    for idx in range(len(val_hf)):
        ex = val_hf[idx]
        img, _, anns = normalize_example(ex, idx, global_cat_map)

        coco_img_id = idx + 1
        idx_to_coco_imgid[idx] = coco_img_id
        w, h = img.size

        gt_images.append({"id": coco_img_id, "file_name": str(coco_img_id), "width": w, "height": h})

        for a in anns:
            x, y, bw, bh = map(float, a["bbox"])
            if bw <= 1 or bh <= 1:
                continue
            gt_annotations.append({
                "id": ann_id,
                "image_id": coco_img_id,
                "category_id": int(a["category_id"]),
                "bbox": [x, y, bw, bh],
                "area": float(bw * bh),
                "iscrowd": int(a.get("iscrowd", 0)),
            })
            ann_id += 1

    gt_dict = {"images": gt_images, "annotations": gt_annotations, "categories": gt_categories}
    coco_gt = COCO()
    coco_gt.dataset = gt_dict
    coco_gt.createIndex()

    img_ids = [img["id"] for img in gt_images]
    cat_ids = [c["id"] for c in gt_categories]
    return coco_gt, img_ids, cat_ids, idx_to_coco_imgid


def run_eval(eval_model, eval_processor, val_hf, global_cat_map, device, threshold):
    coco_gt, img_ids, cat_ids, idx_to_coco_imgid = build_coco_gt_from_val(val_hf, global_cat_map)
    print(f"Val images: {len(img_ids)}, GT annotations: {len(coco_gt.dataset['annotations'])}")

    preds = []
    eval_model.eval()
    with torch.no_grad():
        for idx in tqdm(range(len(val_hf)), desc="Predict val"):
            ex = val_hf[idx]
            img, _, _ = normalize_example(ex, idx, global_cat_map)
            coco_img_id = idx_to_coco_imgid[idx]

            inputs = eval_processor(images=img, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = eval_model(**inputs)

            target_sizes = torch.tensor([img.size[::-1]], device=device)
            res = eval_processor.post_process_object_detection(
                outputs, threshold=threshold, target_sizes=target_sizes,
            )[0]

            boxes = res["boxes"].detach().cpu().numpy()
            scores = res["scores"].detach().cpu().numpy()
            labels = res["labels"].detach().cpu().numpy()

            for (x1, y1, x2, y2), s, lab in zip(boxes, scores, labels):
                lab = int(lab)
                if lab != 1:
                    continue
                w, h = float(x2 - x1), float(y2 - y1)
                if w <= 1 or h <= 1:
                    continue
                preds.append({
                    "image_id": int(coco_img_id),
                    "category_id": lab,
                    "bbox": [float(x1), float(y1), w, h],
                    "score": float(s),
                })

    print(f"Predictions: {len(preds)}")

    if len(preds) == 0:
        print("No valid predictions found.")
        return

    coco_dt = coco_gt.loadRes(preds)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.params.imgIds = img_ids
    coco_eval.params.catIds = cat_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    print("\n===== DeTR fine-tuned on DeART (COCOeval) =====")
    coco_eval.summarize()

    stats_names = [
        "AP@[.50:.95]", "AP@0.50", "AP@0.75", "AP_small", "AP_medium", "AP_large",
        "AR@1", "AR@10", "AR@100", "AR_small", "AR_medium", "AR_large",
    ]
    for n, v in zip(stats_names, coco_eval.stats):
        print(f"{n:>18}: {v:.4f}")


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, Device: {device}")

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Load HuggingFace dataset
    configs = get_dataset_config_names(args.dataset_name)
    config = args.preferred_config if args.preferred_config in configs else configs[0]
    print("Using config:", config)

    ds = load_dataset(args.dataset_name, config)
    split = ds["train"].train_test_split(test_size=args.val_ratio, seed=SEED)
    train_hf, val_hf = split["train"], split["test"]
    print(f"Train: {len(train_hf)}, Val: {len(val_hf)}")

    global_cat_map = build_global_cat_map(train_hf)

    # Processor + Model
    processor = AutoImageProcessor.from_pretrained(args.checkpoint, use_fast=False)

    model = AutoModelForObjectDetection.from_pretrained(
        args.checkpoint,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    ).to(device)

    # Datasets
    train_ds = DeartDetrDataset(train_hf, processor, global_cat_map, transform=get_train_transform())
    val_ds = DeartDetrDataset(val_hf, processor, global_cat_map, transform=get_val_transform())

    def collate_fn(batch):
        pixel_values = [b["pixel_values"] for b in batch]
        enc = processor.pad(pixel_values, return_tensors="pt")
        labels = [b["labels"] for b in batch]
        return {"pixel_values": enc["pixel_values"], "pixel_mask": enc["pixel_mask"], "labels": labels}

    # TrainingArguments
    args_kwargs = dict(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        weight_decay=1e-4,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_bs,
        per_device_eval_batch_size=args.eval_bs,
        dataloader_num_workers=args.num_workers,
        fp16=False,
        logging_steps=50,
        save_strategy="epoch",
        remove_unused_columns=False,
        report_to="none",
        max_grad_norm=0.1,
    )

    sig = inspect.signature(TrainingArguments.__init__)
    if "evaluation_strategy" in sig.parameters:
        args_kwargs["evaluation_strategy"] = "epoch"
    elif "eval_strategy" in sig.parameters:
        args_kwargs["eval_strategy"] = "epoch"
    if "eval_do_concat_batches" in sig.parameters:
        args_kwargs["eval_do_concat_batches"] = False

    training_args = TrainingArguments(**args_kwargs)

    # Trainer
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
    )
    trainer_sig = inspect.signature(Trainer.__init__)
    if "processing_class" in trainer_sig.parameters:
        trainer_kwargs["processing_class"] = processor
    trainer = Trainer(**trainer_kwargs)

    # Train
    trainer.train()
    print("Training done.")

    final_dir = os.path.join(args.output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    trainer.save_model(final_dir)
    processor.save_pretrained(final_dir)
    print("Saved model to:", final_dir)

    # Evaluation
    eval_processor = AutoImageProcessor.from_pretrained(final_dir, use_fast=False)
    eval_model = AutoModelForObjectDetection.from_pretrained(final_dir).to(device)
    run_eval(eval_model, eval_processor, val_hf, global_cat_map, device, args.eval_threshold)


if __name__ == "__main__":
    main()
