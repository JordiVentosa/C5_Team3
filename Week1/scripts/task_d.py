"""
evaluate_coco.py
----------------
Evaluates either YOLOv8 or Faster R-CNN on the KITTI-MOTS dataset using
standard COCO metrics (AP, AP50, AP75, APs, APm, APl, AR@1, AR@10, AR@100).
Now uses KittyDataset for GT loading.

Usage
-----
Set MODEL_TYPE = "yolo" | "frcnn" and RUN_SPLIT = "train" | "val" in main().
"""

import io
import contextlib
import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from Week1.src.models.ultralytics_yolo import YOLOInference
from Week1.src.models.torchvision_faster_rcnn import FasterRCNNInference
from Week1.src.utils.dataset import KittyDataset  # adjust import path as needed


# ── Dataset constants ─────────────────────────────────────────────────────────

# KittyDataset already maps to COCO IDs: car→3, pedestrian→1
EVAL_CAT_INFO = [
    {"id": 1, "name": "Pedestrian", "supercategory": "person"},
    {"id": 3, "name": "Car",        "supercategory": "vehicle"},
]

# YOLO (0-indexed COCO): person=0 → eval 1,  car=2 → eval 3
YOLO_ALLOWED  = {0: 1, 2: 3}
# Torchvision Faster R-CNN (1-indexed COCO): person=1 → eval 1, car=3 → eval 3
FRCNN_ALLOWED = {1: 1, 3: 3}


# ── Build COCO GT from KittyDataset ──────────────────────────────────────────

def build_coco_gt_from_dataset(dataset: KittyDataset):
    """
    Converts a KittyDataset instance into an in-memory COCO GT object.

    KittyDataset already resolves image paths and annotations (in Pascal VOC
    [x1,y1,x2,y2] format) with COCO category IDs, so we just reformat them.

    Returns
    -------
    coco_gt    : COCO object with createIndex() already called
    image_meta : list of {"id": int, "path": str} to drive prediction
    """
    images_list = []
    anns_list   = []
    image_meta  = []
    ann_id      = 0

    for image_id, (img_path, annotation) in enumerate(
            zip(dataset.image_paths, dataset.annotations)):

        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        images_list.append({
            "id":        image_id,
            "file_name": img_path,
            "height":    h,
            "width":     w,
        })
        image_meta.append({"id": image_id, "path": img_path})

        # annotation["boxes"] is a list of [x1, y1, x2, y2] (Pascal VOC)
        # annotation["labels"] is a list of COCO category IDs
        for (x1, y1, x2, y2), cat_id in zip(
                annotation["boxes"], annotation["labels"]):
            bw = max(x2 - x1, 1)
            bh = max(y2 - y1, 1)
            anns_list.append({
                "id":          ann_id,
                "image_id":    image_id,
                "category_id": int(cat_id),
                "bbox":        [x1, y1, bw, bh],   # COCO: [x, y, w, h]
                "area":        float(bw * bh),
                "iscrowd":     0,
            })
            ann_id += 1

    coco_gt = COCO()
    coco_gt.dataset = {
        "info":        {"description": "KITTI-MOTS via KittyDataset"},
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
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ev.summarize()
        row = f"{cat['name']:<15}" + "".join(f"{v:>12.4f}" for v in ev.stats)
        print(row)

    print("────────────────────────────────────────────────────────────\n")
    return coco_eval


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    BASE_DIR   = "/home/msiau/workspace/jventosa/PostTFG/Master/C5_Team3/Week1/datasets/KITTI-MOTS"
    MODEL_TYPE = "yolo"   # "yolo" | "frcnn"
    RUN_SPLIT  = "all"  # "train" | "val"  ← matches KittyDataset mode arg
    CONF_THR   = 0.5

    print(f"\nBuilding COCO GT via KittyDataset  [split={RUN_SPLIT}] ...")
    dataset = KittyDataset(root_dir=BASE_DIR, mode=RUN_SPLIT)
    coco_gt, image_meta = build_coco_gt_from_dataset(dataset)

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