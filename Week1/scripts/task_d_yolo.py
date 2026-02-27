import argparse
import contextlib
import io
import json
import os
from datetime import datetime

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import cv2

from src.models.ultralytics_yolo import YOLOInference
from src.utils.dataset import KittyDataset


# ── Constants ─────────────────────────────────────────────────────────────────

BASE_DIR = "/home/mcv/datasets/C5/KITTI-MOTS"

EVAL_CAT_INFO = [
    {"id": 1, "name": "Pedestrian", "supercategory": "person"},
    {"id": 3, "name": "Car",        "supercategory": "vehicle"},
]

# YOLO uses 0-indexed COCO IDs: person=0, car=2
YOLO_TO_COCO = {0: 1, 2: 3}


# ── Build COCO GT ─────────────────────────────────────────────────────────────

def build_coco_gt(dataset: KittyDataset):
    images_list, anns_list, image_meta = [], [], []
    ann_id = 0

    for image_id, (img_path, annotation) in enumerate(
            zip(dataset.image_paths, dataset.annotations)):
        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        images_list.append({"id": image_id, "file_name": img_path, "height": h, "width": w})
        image_meta.append({"id": image_id, "path": img_path})

        for (x1, y1, x2, y2), cat_id in zip(annotation["boxes"], annotation["labels"]):
            bw, bh = max(x2 - x1, 1), max(y2 - y1, 1)
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
    print(f"  GT: {len(images_list)} images | {len(anns_list)} annotations")
    return coco_gt, image_meta


# ── YOLO inference ────────────────────────────────────────────────────────────

def predict_yolo(detector, image_meta, conf_threshold):
    predictions = []
    for meta in image_meta:
        results = detector.predict(meta["path"], conf_threshold=conf_threshold,
                                   save_results=False)
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                if cls_id not in YOLO_TO_COCO:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                predictions.append({
                    "image_id":    meta["id"],
                    "category_id": YOLO_TO_COCO[cls_id],
                    "bbox":        [x1, y1, max(x2 - x1, 1), max(y2 - y1, 1)],
                    "score":       float(box.conf[0]),
                })
    return predictions


# ── COCO evaluation ───────────────────────────────────────────────────────────

METRIC_NAMES = [
    "AP[.5:.95]", "AP@.50", "AP@.75",
    "AP@small",   "AP@med", "AP@large",
    "AR@1",       "AR@10",  "AR@100",
    "AR@small",   "AR@med", "AR@large",
]

def run_coco_eval(coco_gt, predictions):
    if not predictions:
        print("[ERROR] No predictions to evaluate.")
        return None, []

    coco_pred = coco_gt.loadRes(predictions)

    # Overall
    print("\n── Overall COCO bbox metrics ──────────────────────────────")
    coco_eval = COCOeval(coco_gt, coco_pred, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Per category
    print("\n── Per-category COCO metrics ──────────────────────────────")
    header = f"{'Category':<15}" + "".join(f"{m:>12}" for m in METRIC_NAMES)
    print(header)
    print("─" * len(header))

    per_cat_stats = []
    for cat in EVAL_CAT_INFO:
        ev = COCOeval(coco_gt, coco_pred, iouType="bbox")
        ev.params.catIds = [cat["id"]]
        ev.evaluate()
        ev.accumulate()
        with contextlib.redirect_stdout(io.StringIO()):
            ev.summarize()
        print(f"{cat['name']:<15}" + "".join(f"{v:>12.4f}" for v in ev.stats))
        per_cat_stats.append(ev.stats)

    print("───────────────────────────────────────────────────────────\n")
    return coco_eval, per_cat_stats


# ── Save results ──────────────────────────────────────────────────────────────

def save_results(output_dir, split, model_version, predictions, coco_eval, per_cat_stats):
    os.makedirs(output_dir, exist_ok=True)

    # predictions.json
    pred_path = os.path.join(output_dir, "predictions.json")
    with open(pred_path, "w") as f:
        json.dump(predictions, f)

    # metrics.json
    metrics = {
        "model":  f"yolo ({model_version})",
        "split":  split,
        "date":   datetime.now().isoformat(),
        "overall": {k: float(v) for k, v in zip(METRIC_NAMES, coco_eval.stats)},
        "per_category": {
            cat["name"]: {k: float(v) for k, v in zip(METRIC_NAMES, stats)}
            for cat, stats in zip(EVAL_CAT_INFO, per_cat_stats)
        },
    }
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # summary.txt
    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Model : yolo ({model_version})\n")
        f.write(f"Split : {split}\n")
        f.write(f"Date  : {metrics['date']}\n\n")
        f.write("── Overall ──\n")
        for k, v in metrics["overall"].items():
            f.write(f"  {k:<15} {v:.4f}\n")
        f.write("\n── Per category ──\n")
        header = f"{'Category':<15}" + "".join(f"{m:>12}" for m in METRIC_NAMES)
        f.write(header + "\n" + "─" * len(header) + "\n")
        for cat_name, stats in metrics["per_category"].items():
            f.write(f"{cat_name:<15}" + "".join(f"{v:>12.4f}" for v in stats.values()) + "\n")

    print(f"\n  Results saved to: {output_dir}")
    print(f"    predictions.json")
    print(f"    metrics.json")
    print(f"    summary.txt")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate YOLO on KITTI-MOTS (COCO metrics)")
    parser.add_argument("--split",   choices=["train", "val", "all"], default="val")
    parser.add_argument("--conf",    type=float, default=0.5)
    parser.add_argument("--model",   default="yolo26x.pt", help="YOLO weights")
    parser.add_argument("--output-dir", default="outputs/task_d_yolo")
    return parser.parse_args()


def main():
    args = parse_args()

    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"{args.split}_{timestamp}")

    print(f"\n── YOLO Evaluation on KITTI-MOTS ──────────────────────────")
    print(f"  Model : {args.model}")
    print(f"  Split : {args.split}")
    print(f"  Conf  : {args.conf}")

    print(f"\nLoading dataset [split={args.split}] ...")
    dataset = KittyDataset(root_dir=BASE_DIR, mode=args.split)
    coco_gt, image_meta = build_coco_gt(dataset)

    print(f"\nLoading YOLO ({args.model}) ...")
    detector = YOLOInference(model_version=args.model)

    print(f"Running inference [conf>={args.conf}] ...")
    predictions = predict_yolo(detector, image_meta, conf_threshold=args.conf)
    print(f"  Total detections: {len(predictions)}")

    coco_eval, per_cat_stats = run_coco_eval(coco_gt, predictions)

    if coco_eval is not None:
        save_results(output_dir, args.split, args.model,
                     predictions, coco_eval, per_cat_stats)


if __name__ == "__main__":
    main()
