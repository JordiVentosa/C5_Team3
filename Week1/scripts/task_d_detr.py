import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

from Week1.src.models.huggingface_detr import load_detr_model
from Week1.src.utils.kitti_helpers import (
    KITTI_CLASS_TO_COCO,
    list_training_pairs,
    extract_gt_from_instance_png,
    coco_eval_bbox,
)

from pycocotools.coco import COCO

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

COCO_CATEGORIES = [
    {"id": 1, "name": "person", "supercategory": "person"},
    {"id": 3, "name": "car", "supercategory": "vehicle"},
]
EVAL_CAT_IDS = sorted(set(KITTI_CLASS_TO_COCO.values()))  # [1, 3]


def get_args():
    parser = argparse.ArgumentParser(description="Task D: Quantitative DeTR evaluation on KITTI-MOTS")
    parser.add_argument("--kitti_root", type=str, default="/home/mcv/datasets/C5/KITTI-MOTS")
    parser.add_argument("--checkpoint", type=str, default="facebook/detr-resnet-50")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seq_start", type=int, default=16)
    parser.add_argument("--seq_end", type=int, default=21)
    return parser.parse_args()


def build_coco_gt(pairs):
    gt_images, gt_annotations = [], []
    ann_id = 1

    for img_id, (seq, rgb_path, mask_path) in enumerate(pairs, start=1):
        with Image.open(rgb_path) as im:
            w, h = im.size

        gt_images.append({"id": img_id, "file_name": str(rgb_path), "width": w, "height": h})

        gt_objs = extract_gt_from_instance_png(mask_path, KITTI_CLASS_TO_COCO)
        for coco_cat, bbox_xywh in gt_objs:
            x, y, bw, bh = bbox_xywh
            gt_annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": int(coco_cat),
                "bbox": [float(x), float(y), float(bw), float(bh)],
                "area": float(bw * bh),
                "iscrowd": 0,
            })
            ann_id += 1

    gt_dict = {"images": gt_images, "annotations": gt_annotations, "categories": COCO_CATEGORIES}
    coco_gt = COCO()
    coco_gt.dataset = gt_dict
    coco_gt.createIndex()
    img_ids = [img["id"] for img in gt_images]
    return coco_gt, img_ids


def predict_detr(model, processor, pairs, img_ids, device, threshold):
    preds = []
    keep_ids = set(EVAL_CAT_IDS)

    with torch.no_grad():
        for img_id, (seq, rgb_path, mask_path) in tqdm(
            list(zip(img_ids, pairs)), desc="DeTR preds", total=len(pairs),
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
                if lab not in keep_ids:
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


def plot_metrics(stats, title="COCOeval metrics - DeTR on KITTI-MOTS"):
    labels = [
        "AP@[.50:.95]", "AP@0.50", "AP@0.75", "AP_small", "AP_medium", "AP_large",
        "AR@1", "AR@10", "AR@100", "AR_small", "AR_medium", "AR_large",
    ]
    x = np.arange(len(labels))
    plt.figure(figsize=(12, 4))
    plt.bar(x, stats)
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylim(0, 1.0)
    plt.title(title)
    plt.ylabel("score")
    plt.tight_layout()
    plt.savefig("detr_kitti_metrics_task_d.png", dpi=150)
    plt.show()


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    kitti_root = Path(args.kitti_root)
    seqs = [f"{i:04d}" for i in range(args.seq_start, args.seq_end)]
    print("Sequences:", seqs)

    pairs = list_training_pairs(kitti_root, seqs=seqs)
    if not pairs:
        raise RuntimeError("No (RGB, GT mask) pairs found.")
    print(f"Evaluating on {len(pairs)} frames")

    coco_gt, img_ids = build_coco_gt(pairs)
    print("GT annotations:", len(coco_gt.dataset["annotations"]))

    processor, model = load_detr_model(args.checkpoint, device)
    preds = predict_detr(model, processor, pairs, img_ids, device, args.threshold)

    stats = coco_eval_bbox(
        coco_gt, preds, img_ids, EVAL_CAT_IDS,
        title=f"DeTR (COCO pretrained) - seqs {seqs}",
    )

    if stats is not None:
        plot_metrics(np.array(stats, dtype=float))


if __name__ == "__main__":
    main()
