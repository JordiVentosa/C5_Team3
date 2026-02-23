"""
evaluate_coco.py
----------------
Evaluates either YOLOv8 or Faster R-CNN on the KITTI-MOTS dataset using
standard COCO metrics (AP, AP50, AP75, APs, APm, APl, AR@1, AR@10, AR@100).

Usage
-----
Set MODEL_TYPE = "yolo" | "frcnn" and RUN_SPLIT = "training" | "test" in main().

Dependencies
------------
    pip install pycocotools torchvision ultralytics opencv-python numpy
"""

import os
import glob
import cv2
import numpy as np
import torch
from torchvision import models, transforms
import torch.nn as nn
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from Week1.src.models.ultralytics_yolo import YOLOInference
from Week1.src.models.torchvision_faster_rcnn import FasterRCNNInference


# ── Dataset constants ─────────────────────────────────────────────────────────

KITTI_TO_EVAL_CAT = {1: 1, 2: 2}   # Car=1, Pedestrian=2
EVAL_CAT_INFO = [
    {"id": 1, "name": "Car",        "supercategory": "vehicle"},
    {"id": 2, "name": "Pedestrian", "supercategory": "person"},
]

# YOLO (0-indexed COCO): person=0 -> eval 2,  car=2 -> eval 1
YOLO_ALLOWED  = {0: 2, 2: 1}
# Torchvision Faster R-CNN (1-indexed COCO): person=1 -> eval 2, car=3 -> eval 1
FRCNN_ALLOWED = {1: 2, 3: 1}



def decode_rle_to_bbox(rle_str, height, width):
    rle = {"counts": rle_str.encode(), "size": [height, width]}
    binary_mask = mask_utils.decode(rle)
    rows = np.any(binary_mask, axis=1)
    cols = np.any(binary_mask, axis=0)
    if not rows.any():
        return None
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    return (int(x1), int(y1), int(x2), int(y2))


def load_gt_for_seq(instances_txt_dir, seq_str):
    ann_file = os.path.join(instances_txt_dir, f"{seq_str}.txt")
    annotations = {}
    if not os.path.exists(ann_file):
        print(f"  [WARN] GT file missing: {ann_file}")
        return annotations
    with open(ann_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            frame_id    = int(parts[0])
            object_id   = int(parts[1])
            height      = int(parts[3])
            width       = int(parts[4])
            rle_str     = parts[5]
            class_id    = object_id // 1000
            instance_id = object_id % 1000
            if class_id not in KITTI_TO_EVAL_CAT:
                continue
            bbox = decode_rle_to_bbox(rle_str, height, width)
            if bbox is None:
                continue
            annotations.setdefault(frame_id, []).append(
                (class_id, instance_id, bbox)
            )
    return annotations


# ── Build GT + collect image metadata ────────────────────────────────────────

def build_coco_gt_and_meta(base_dir, seq_range):
    """
    Build a COCO ground-truth object entirely in memory 

    Returns
    -------
    coco_gt   : COCO object with createIndex() already called
    image_meta: list of {id, path} used to drive prediction
    """
    instances_txt_dir = os.path.join(base_dir, "instances_txt")
    image_base_dir    = os.path.join(base_dir, "training", "image_02")

    images_list = []
    anns_list   = []
    image_meta  = []
    image_id    = 0
    ann_id      = 0

    for seq_id in seq_range:
        seq_str  = f"{seq_id:04d}"
        seq_path = os.path.join(image_base_dir, seq_str)
        if not os.path.isdir(seq_path):
            continue

        img_paths      = sorted(glob.glob(os.path.join(seq_path, "*.png")))
        gt_annotations = load_gt_for_seq(instances_txt_dir, seq_str)

        print(f"  seq {seq_str}: {len(img_paths)} frames, "
              f"{sum(len(v) for v in gt_annotations.values())} GT objects")

        for img_path in img_paths:
            frame_id = int(os.path.splitext(os.path.basename(img_path))[0])
            img = cv2.imread(img_path)
            h, w = img.shape[:2]

            # ── Register image ──────────────────────────────────────────────
            images_list.append({
                "id": image_id, "file_name": img_path,
                "height": h, "width": w,
            })
            image_meta.append({"id": image_id, "path": img_path})

            # ── Register GT annotations for this frame ──────────────────────
            for class_id, instance_id, (x1, y1, x2, y2) in gt_annotations.get(frame_id, []):
                bw = max(x2 - x1, 1)
                bh = max(y2 - y1, 1)
                anns_list.append({
                    "id":          ann_id,
                    "image_id":    image_id,          # same integer we just stored
                    "category_id": KITTI_TO_EVAL_CAT[class_id],
                    "bbox":        [x1, y1, bw, bh],  # COCO: [x, y, w, h]
                    "area":        float(bw * bh),
                    "iscrowd":     0,
                })
                ann_id += 1

            image_id += 1

    coco_gt = COCO()
    coco_gt.dataset = {
        "info":        {"description": "KITTI-MOTS as COCO"},
        "categories":  EVAL_CAT_INFO,
        "images":      images_list,
        "annotations": anns_list,
    }
    coco_gt.createIndex()

    print(f"\n  Total: {len(images_list)} images | {len(anns_list)} GT annotations")
    return coco_gt, image_meta


# ── Prediction runners ────────────────────────────────────────────────────────

def predict_yolo(detector, image_meta, conf_threshold=0.5):
    predictions = []
    for meta in image_meta:
        results = detector.predict(meta["path"], conf_threshold=conf_threshold,
                                   save_results=False)
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                if cls_id not in YOLO_ALLOWED:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                predictions.append({
                    "image_id":    meta["id"],
                    "category_id": YOLO_ALLOWED[cls_id],
                    "bbox":        [x1, y1, max(x2 - x1, 1), max(y2 - y1, 1)],
                    "score":       float(box.conf[0]),
                })
    return predictions


def predict_frcnn(detector, image_meta, conf_threshold=0.5):
    predictions = []
    for meta in image_meta:
        for eval_cat, score, (x1, y1, x2, y2) in detector.predict(
                meta["path"], conf_threshold=conf_threshold):
            predictions.append({
                "image_id":    meta["id"],
                "category_id": eval_cat,
                "bbox":        [x1, y1, max(x2 - x1, 1), max(y2 - y1, 1)],
                "score":       score,
            })
    return predictions


# ── COCO evaluation ───────────────────────────────────────────────────────────

def run_coco_eval(coco_gt, predictions):
    """
    Runs COCOeval and prints:
      - Standard 12-metric summary (overall)
      - Per-category table with all 12 metrics
    """
    if not predictions:
        print("[ERROR] No predictions to evaluate.")
        return None

    coco_pred = coco_gt.loadRes(predictions)

    print("\n── Overall COCO bbox metrics ───────────────────────────────")
    coco_eval = COCOeval(coco_gt, coco_pred, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # ── Per-category breakdown ──────────────────────────────────────────────
    metric_names = [
        "AP[.5:.95]", "AP@.50", "AP@.75",
        "AP@small",   "AP@med", "AP@large",
        "AR@1",       "AR@10",  "AR@100",
        "AR@small",   "AR@med", "AR@large",
    ]
    print("\n── Per-category COCO metrics ───────────────────────────────")
    header = f"{'Category':<15}" + "".join(f"{m:>12}" for m in metric_names)
    print(header)
    print("─" * len(header))

    for cat in EVAL_CAT_INFO:
        ev = COCOeval(coco_gt, coco_pred, iouType="bbox")
        ev.params.catIds = [cat["id"]]
        ev.evaluate()
        ev.accumulate()
        # Suppress the per-metric printout from summarize()
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ev.summarize()
        stats = ev.stats
        row = f"{cat['name']:<15}" + "".join(f"{v:>12.4f}" for v in stats)
        print(row)

    print("────────────────────────────────────────────────────────────\n")
    return coco_eval


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    BASE_DIR   = "/home/msiau/data/tmp/jventosa/KITTI-MOTS"
    MODEL_TYPE = "yolo"      # "yolo" | "frcnn"
    RUN_SPLIT  = "training"   # "training" | "test"
    CONF_THR   = 0.5

    SEQ_RANGES = {
        "training": range(0, 16),
        "test":     range(16, 21),
    }

    print(f"\nBuilding COCO GT  [split={RUN_SPLIT}] ...")
    coco_gt, image_meta = build_coco_gt_and_meta(BASE_DIR, SEQ_RANGES[RUN_SPLIT])

    if not image_meta:
        print("[ERROR] No images found. Check BASE_DIR and RUN_SPLIT.")
        return

    print(f"  GT categories : {sorted(coco_gt.getCatIds())}")
    print(f"  GT images     : {len(coco_gt.getImgIds())}")
    print(f"  GT annotations: {len(coco_gt.getAnnIds())}")

    if MODEL_TYPE == "yolo":
        print("\nLoading YOLOv8x ...")
        detector = YOLOInference(model_version="yolov8x.pt")
        print(f"Running YOLO predictions  [conf>={CONF_THR}] ...")
        predictions = predict_yolo(detector, image_meta, conf_threshold=CONF_THR)

    elif MODEL_TYPE == "frcnn":
        print("\nLoading Faster R-CNN (ResNet-50 FPN) ...")
        detector = FasterRCNNInference(conf_threshold=CONF_THR)
        print(f"Running Faster R-CNN predictions  [conf>={CONF_THR}] ...")
        predictions = predict_frcnn(detector, image_meta, conf_threshold=CONF_THR)

    else:
        raise ValueError(f"Unknown MODEL_TYPE '{MODEL_TYPE}'. Use 'yolo' or 'frcnn'.")

    print(f"  Total detections      : {len(predictions)}")
    print(f"  Prediction categories : {sorted({p['category_id'] for p in predictions})}")


    print(f"\n── COCO Evaluation  |  model={MODEL_TYPE}  split={RUN_SPLIT} ──")
    run_coco_eval(coco_gt, predictions)


if __name__ == "__main__":
    main()