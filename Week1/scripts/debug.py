"""
debug_eval.py
-------------
Runs a small sanity-check on a single sequence to diagnose why COCO metrics
come out as zero.  Prints GT and prediction samples side-by-side and checks
for category-ID mismatches, empty outputs, IoU overlap, etc.

Set MODEL_TYPE and SEQ_ID below, then run:
    python debug_eval.py
"""

import os
import glob
import json
import cv2
import numpy as np
import torch
from torchvision import models, transforms
import torch.nn as nn
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from Week1.src.models.ultralytics_yolo import YOLOInference


# ── Config ────────────────────────────────────────────────────────────────────
MODEL_TYPE = "frcnn"      # "yolo" | "frcnn"
SEQ_ID     = 0            # sequence to debug (single sequence, fast)
CONF_THR   = 0.01         # very low threshold so we definitely get predictions
BASE_DIR   = "/home/msiau/data/tmp/jventosa/KITTI-MOTS"
MAX_FRAMES = 5             # only look at the first N frames to keep output short
# ─────────────────────────────────────────────────────────────────────────────

KITTI_TO_EVAL_CAT = {1: 1, 2: 2}
EVAL_CAT_INFO = [{"id": 1, "name": "Car"}, {"id": 2, "name": "Pedestrian"}]
YOLO_ALLOWED  = {0: 2, 2: 1}   # yolo cls -> eval cat
FRCNN_ALLOWED = {1: 2, 3: 1}   # torchvision cls -> eval cat


# ── Faster R-CNN ──────────────────────────────────────────────────────────────
class FasterRCNNInference(nn.Module):
    def __init__(self, conf_threshold=0.5, device=None):
        super().__init__()
        self.conf_threshold = conf_threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.to(self.device)
        self.model.eval()
        self.transform = transforms.Compose([transforms.ToTensor()])

    def forward(self, x):
        return self.model(x)

    def predict_raw(self, img_path):
        """Return raw torchvision output (labels, scores, boxes) — no filtering."""
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        tensor  = self.transform(img_rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model(tensor)[0]
        return out


# ── GT loading ────────────────────────────────────────────────────────────────
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


def load_gt_for_seq(seq_str):
    ann_file = os.path.join(BASE_DIR, "instances_txt", f"{seq_str}.txt")
    annotations = {}
    if not os.path.exists(ann_file):
        print(f"[ERROR] GT file not found: {ann_file}")
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


# ── IoU helper ────────────────────────────────────────────────────────────────
def iou(boxA, boxB):
    """Both boxes are (x1, y1, x2, y2)."""
    xA = max(boxA[0], boxB[0]);  yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]);  yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0


# ── Main diagnostic ───────────────────────────────────────────────────────────
def main():
    seq_str   = f"{SEQ_ID:04d}"
    seq_path  = os.path.join(BASE_DIR, "training", "image_02", seq_str)
    img_paths = sorted(glob.glob(os.path.join(seq_path, "*.png")))[:MAX_FRAMES]

    if not img_paths:
        print(f"[ERROR] No images found at {seq_path}")
        return

    print(f"\n{'='*60}")
    print(f" Debugging  model={MODEL_TYPE}  seq={seq_str}  conf_thr={CONF_THR}")
    print(f" Checking first {len(img_paths)} frames")
    print(f"{'='*60}\n")

    # Load GT
    gt_annotations = load_gt_for_seq(seq_str)
    total_gt = sum(len(v) for v in gt_annotations.values())
    print(f"[GT] Loaded {len(gt_annotations)} frames with annotations, "
          f"{total_gt} total objects")
    print(f"[GT] Category IDs in use: { {v[0] for vals in gt_annotations.values() for v in vals} }")

    # Sample GT boxes from first annotated frame
    sample_frame = next(iter(gt_annotations))
    print(f"\n[GT] Sample frame {sample_frame}:")
    for class_id, instance_id, bbox in gt_annotations[sample_frame][:5]:
        print(f"     class_id={class_id} -> eval_cat={KITTI_TO_EVAL_CAT[class_id]}  "
              f"bbox={bbox}  area={( bbox[2]-bbox[0])*(bbox[3]-bbox[1])}")

    # Load model
    print(f"\n[MODEL] Loading {MODEL_TYPE} ...")
    if MODEL_TYPE == "yolo":
        detector = YOLOInference(model_version="yolov8x.pt")
    else:
        detector = FasterRCNNInference(conf_threshold=CONF_THR)

    # Run predictions on sample frames and inspect
    print(f"\n[PRED] Running inference on {len(img_paths)} frames ...\n")

    all_gt_boxes   = []
    all_pred_boxes = []
    total_preds    = 0

    for img_path in img_paths:
        frame_id = int(os.path.splitext(os.path.basename(img_path))[0])
        frame_gt = gt_annotations.get(frame_id, [])
        img      = cv2.imread(img_path)
        h, w     = img.shape[:2]

        print(f"  Frame {frame_id:06d}  ({w}x{h})  GT objects: {len(frame_gt)}")

        # ── Ground truth boxes for this frame ──
        for class_id, instance_id, bbox in frame_gt:
            all_gt_boxes.append({
                "eval_cat": KITTI_TO_EVAL_CAT[class_id],
                "bbox_xyxy": bbox,
            })

        # ── Predictions ──
        if MODEL_TYPE == "yolo":
            results = detector.predict(img_path, conf_threshold=CONF_THR, save_results=False)
            frame_preds = []
            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    score  = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    # Show ALL classes so we can see if cars/persons are detected
                    # under a different class ID
                    mapped = YOLO_ALLOWED.get(cls_id, None)
                    frame_preds.append({
                        "raw_cls": cls_id,
                        "cls_name": result.names[cls_id],
                        "eval_cat": mapped,
                        "score": score,
                        "bbox_xyxy": (x1, y1, x2, y2),
                    })

        else:  # frcnn
            raw = detector.predict_raw(img_path)
            frame_preds = []
            for label, score, box in zip(raw["labels"], raw["scores"], raw["boxes"]):
                cls_id = int(label)
                score  = float(score)
                x1, y1, x2, y2 = map(int, box.tolist())
                mapped = FRCNN_ALLOWED.get(cls_id, None)
                frame_preds.append({
                    "raw_cls": cls_id,
                    "score": score,
                    "eval_cat": mapped,
                    "bbox_xyxy": (x1, y1, x2, y2),
                })

        # Print top-5 predictions (all classes, no filtering)
        frame_preds_sorted = sorted(frame_preds, key=lambda x: x["score"], reverse=True)
        print(f"    Top predictions (no class filter, no conf filter):")
        if not frame_preds_sorted:
            print("      [NONE] — model returned 0 detections for this frame!")
        for p in frame_preds_sorted[:5]:
            mapped_str = str(p["eval_cat"]) if p["eval_cat"] else "NOT MAPPED (skipped)"
            name = p.get("cls_name", f"cls{p['raw_cls']}")
            print(f"      raw_cls={p['raw_cls']} ({name})  "
                  f"score={p['score']:.3f}  eval_cat={mapped_str}  "
                  f"bbox={p['bbox_xyxy']}")

        # Count only allowed class predictions
        allowed_preds = [p for p in frame_preds if p["eval_cat"] is not None
                         and p["score"] >= CONF_THR]
        total_preds += len(allowed_preds)

        # Check IoU between GT and allowed preds for this frame
        if frame_gt and allowed_preds:
            print(f"    IoU check (GT vs allowed preds):")
            for gt_obj in frame_gt[:3]:
                gt_box = gt_obj[2]
                gt_cat = KITTI_TO_EVAL_CAT[gt_obj[0]]
                best_iou  = 0.0
                best_pred = None
                for p in allowed_preds:
                    if p["eval_cat"] != gt_cat:
                        continue
                    ov = iou(gt_box, p["bbox_xyxy"])
                    if ov > best_iou:
                        best_iou  = ov
                        best_pred = p
                print(f"      GT cat={gt_cat} box={gt_box}  "
                      f"best_IoU={best_iou:.3f}  "
                      f"matched_pred={best_pred['bbox_xyxy'] if best_pred else 'None'}")
        elif not frame_gt:
            print("    (no GT objects in this frame)")
        else:
            print("    [WARN] No allowed predictions for this frame after filtering!")

        print()

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"{'='*60}")
    print(f" SUMMARY")
    print(f"  Total GT boxes  : {len(all_gt_boxes)}")
    print(f"  Total predictions (allowed classes, conf>={CONF_THR}): {total_preds}")

    # Check for the most common root causes:
    print(f"\n[CHECK 1] Are there ANY predictions at all?")
    print(f"  -> {'YES' if total_preds > 0 else 'NO — model produces zero detections'}")

    print(f"\n[CHECK 2] Category ID mapping")
    print(f"  GT uses eval_cat IDs : { {KITTI_TO_EVAL_CAT[k] for k in KITTI_TO_EVAL_CAT} }")
    if MODEL_TYPE == "yolo":
        print(f"  YOLO maps to eval_cat: { set(YOLO_ALLOWED.values()) }")
    else:
        print(f"  FRCNN maps to eval_cat: { set(FRCNN_ALLOWED.values()) }")

    print(f"\n[CHECK 3] Confidence threshold")
    print(f"  Current CONF_THR={CONF_THR}. If predictions appear above "
          f"but metrics are 0, raise it slightly and re-run evaluate_coco.py.")

    print(f"\n[CHECK 4] Bounding-box coordinate system")
    if all_gt_boxes:
        b = all_gt_boxes[0]["bbox_xyxy"]
        print(f"  GT bbox sample (x1,y1,x2,y2): {b}  "
              f"w={b[2]-b[0]}  h={b[3]-b[1]}")
    print(f"  (COCO eval expects [x,y,w,h] — conversion is done in evaluate_coco.py)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()