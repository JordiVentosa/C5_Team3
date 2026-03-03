import argparse
import contextlib
import io
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import csv

import cv2
import matplotlib.pyplot as plt
import yaml
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics import YOLO

from src.utils.mots import load_seqmap, load_txt, filename_to_frame_nr


# Constants
KITTI_MOTS_DIR = "/data/uabmcv2526/mcvstudent20/data/KITTI-MOTS"
IMAGE_DIR      = os.path.join(KITTI_MOTS_DIR, "training", "image_02")
INSTANCES_DIR  = os.path.join(KITTI_MOTS_DIR, "instances_txt")

_UTILS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "src", "utils")
TRAIN_SEQMAP = os.path.join(_UTILS_DIR, "train.seqmap")
VAL_SEQMAP   = os.path.join(_UTILS_DIR, "val.seqmap")

# KITTI: 1=Car, 2=Pedestrian  |  YOLO: 0=Car, 1=Pedestrian
KITTI_TO_YOLO = {1: 0, 2: 1}
CLASS_NAMES   = ["Car", "Pedestrian"]

# YOLO class 0 (Car) -> COCO 3,  YOLO class 1 (Ped) -> COCO 1
YOLO_TO_COCO = {0: 3, 1: 1}

EVAL_CAT_INFO = [
    {"id": 1, "name": "Pedestrian", "supercategory": "person"},
    {"id": 3, "name": "Car",        "supercategory": "vehicle"},
]

METRIC_NAMES = [
    "AP[.5:.95]", "AP@.50", "AP@.75",
    "AP@small",   "AP@med", "AP@large",
    "AR@1",       "AR@10",  "AR@100",
    "AR@small",   "AR@med", "AR@large",
]

AUG_CONFIGS = {
    "none":  "configs/yolo_aug_none.yaml",
    "light": "configs/yolo_aug_light.yaml",
    "heavy": "configs/yolo_aug_heavy.yaml",
}


def kitti_bbox_to_yolo(x, y, w, h, img_w, img_h):
    """Convert absolute [x,y,w,h] to normalised YOLO [cx,cy,w,h]."""
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    return cx, cy, nw, nh


def prepare_yolo_dataset(yolo_data_dir: str, force: bool = False):
    """
    Converts KITTI-MOTS to YOLO format under yolo_data_dir/:
        images/train/  images/val/
        labels/train/  labels/val/
    and writes a dataset.yaml.

    Filename pattern: <seq>_<frame>.{png,txt}
    """
    dataset_yaml = os.path.join(yolo_data_dir, "dataset.yaml")
    if os.path.exists(dataset_yaml) and not force:
        print(f"  YOLO dataset already prepared at {yolo_data_dir}  (use --force-prepare to redo)")
        return dataset_yaml

    print(f"  Preparing YOLO dataset -> {yolo_data_dir}")
    for split, seqmap_path in [("train", TRAIN_SEQMAP), ("val", VAL_SEQMAP)]:
        img_out = os.path.join(yolo_data_dir, "images", split)
        lbl_out = os.path.join(yolo_data_dir, "labels", split)
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)

        seqs, _ = load_seqmap(seqmap_path)
        n_imgs, n_labels = 0, 0

        for seq in seqs:
            ann_data = load_txt(os.path.join(INSTANCES_DIR, f"{seq}.txt"))
            img_dir  = os.path.join(IMAGE_DIR, seq)

            for img_file in sorted(Path(img_dir).glob("*.png")):
                frame = filename_to_frame_nr(img_file.name)
                stem  = f"{seq}_{img_file.stem}"   # e.g. 0000_000042

                dst_img = os.path.join(img_out, f"{stem}.png")
                if not os.path.exists(dst_img):
                    os.symlink(str(img_file.resolve()), dst_img)
                n_imgs += 1

                if frame not in ann_data:
                    continue

                img = cv2.imread(str(img_file))
                if img is None:
                    continue
                img_h, img_w = img.shape[:2]

                lines = []
                for obj in ann_data[frame]:
                    if obj.class_id not in KITTI_TO_YOLO:
                        continue
                    x, y, w, h = mask_utils.toBbox(obj.mask)
                    if w < 2 or h < 2:
                        continue
                    cx, cy, nw, nh = kitti_bbox_to_yolo(x, y, w, h, img_w, img_h)
                    # Clamp to [0,1]
                    cx  = max(0.0, min(1.0, cx))
                    cy  = max(0.0, min(1.0, cy))
                    nw  = max(0.001, min(1.0, nw))
                    nh  = max(0.001, min(1.0, nh))
                    yolo_cls = KITTI_TO_YOLO[obj.class_id]
                    lines.append(f"{yolo_cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

                if lines:
                    lbl_path = os.path.join(lbl_out, f"{stem}.txt")
                    with open(lbl_path, "w") as f:
                        f.write("\n".join(lines))
                    n_labels += 1

        print(f"  [{split}] {n_imgs} images, {n_labels} label files")

    data_yaml = {
        "path":  str(Path(yolo_data_dir).resolve()),
        "train": "images/train",
        "val":   "images/val",
        "nc":    len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }
    with open(dataset_yaml, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    print(f"  dataset.yaml written -> {dataset_yaml}")
    return dataset_yaml


def train_yolo(dataset_yaml, aug_mode, base_model, epochs, batch, imgsz,
               lr0, freeze, output_dir, run_name):
    """
    Fine-tunes YOLO and returns (best_weights_path, run_dir).

    freeze=0  -> full fine-tuning (all layers trainable)
    freeze=10 -> freeze backbone (~first 10 layers), train neck+head only
    freeze=20 -> freeze backbone+neck, train head only
    """
    aug_cfg_path = AUG_CONFIGS[aug_mode]
    print(f"\n  Loading aug config : {aug_cfg_path}")
    print(f"  Freeze layers      : {freeze}  ({'head only' if freeze >= 20 else 'neck+head' if freeze >= 10 else 'full fine-tune'})")

    with open(aug_cfg_path) as f:
        aug_params = yaml.safe_load(f) or {}

    model = YOLO(base_model)
    model.train(
        data     = dataset_yaml,
        epochs   = epochs,
        imgsz    = imgsz,
        batch    = batch,
        lr0      = lr0,
        freeze   = freeze if freeze > 0 else None,
        project  = output_dir,
        name     = run_name,
        exist_ok = True,
        verbose  = True,
        **aug_params,
    )

    run_dir      = os.path.join(output_dir, run_name)
    best_weights = os.path.join(run_dir, "weights", "best.pt")
    print(f"\n  Training done. Best weights: {best_weights}")
    return best_weights, run_dir


def plot_losses(run_dir: str, save_dir: str):
    """
    Reads results.csv written by YOLO during training and saves a loss plot
    with train/val box loss, cls loss and dfl loss.
    """
    csv_path = os.path.join(run_dir, "results.csv")
    if not os.path.exists(csv_path):
        print(f"  [WARN] results.csv not found at {csv_path} — skipping loss plot")
        return

    # Read CSV — YOLO uses columns with leading spaces in the header
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows   = list(reader)

    # Strip whitespace from keys
    rows = [{k.strip(): v.strip() for k, v in row.items()} for row in rows]

    epochs = [int(r["epoch"]) + 1 for r in rows]

    # YOLO column names vary slightly by version; try both forms
    def _get(row, *candidates):
        for c in candidates:
            if c in row:
                return float(row[c])
        return None

    train_box = [_get(r, "train/box_loss") for r in rows]
    train_cls = [_get(r, "train/cls_loss") for r in rows]
    train_dfl = [_get(r, "train/dfl_loss") for r in rows]
    val_box   = [_get(r, "val/box_loss")   for r in rows]
    val_cls   = [_get(r, "val/cls_loss")   for r in rows]
    val_dfl   = [_get(r, "val/dfl_loss")   for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, tr, vl, title in zip(
        axes,
        [train_box, train_cls, train_dfl],
        [val_box,   val_cls,   val_dfl],
        ["Box loss", "Cls loss", "DFL loss"],
    ):
        if tr[0] is not None:
            ax.plot(epochs, tr, label="train", color="steelblue")
        if vl[0] is not None:
            ax.plot(epochs, vl, label="val",   color="tomato", linestyle="--")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.suptitle("YOLO training losses", fontsize=13)
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "loss_curves.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Loss plot saved -> {out_path}")


def build_coco_gt_from_seqmap(seqmap_path):
    """Build in-memory COCO GT object from a seqmap (val split)."""
    seqs, _ = load_seqmap(seqmap_path)
    images_list, anns_list, image_meta = [], [], []
    ann_id = 0

    for seq in seqs:
        ann_data = load_txt(os.path.join(INSTANCES_DIR, f"{seq}.txt"))
        img_dir  = os.path.join(IMAGE_DIR, seq)

        for img_file in sorted(Path(img_dir).glob("*.png")):
            frame    = filename_to_frame_nr(img_file.name)
            image_id = len(images_list)

            img = cv2.imread(str(img_file))
            if img is None:
                continue
            h, w = img.shape[:2]

            images_list.append({"id": image_id, "file_name": str(img_file),
                                 "height": h, "width": w})
            image_meta.append({"id": image_id, "path": str(img_file)})

            if frame not in ann_data:
                continue
            for obj in ann_data[frame]:
                if obj.class_id not in KITTI_TO_YOLO:
                    continue
                x, y, bw, bh = mask_utils.toBbox(obj.mask)
                bw, bh = max(bw, 1), max(bh, 1)
                coco_cat = YOLO_TO_COCO[KITTI_TO_YOLO[obj.class_id]]
                anns_list.append({
                    "id":          ann_id,
                    "image_id":    image_id,
                    "category_id": coco_cat,
                    "bbox":        [float(x), float(y), float(bw), float(bh)],
                    "area":        float(bw * bh),
                    "iscrowd":     0,
                })
                ann_id += 1

    coco_gt = COCO()
    coco_gt.dataset = {
        "info":        {"description": "KITTI-MOTS val"},
        "categories":  EVAL_CAT_INFO,
        "images":      images_list,
        "annotations": anns_list,
    }
    coco_gt.createIndex()
    print(f"  GT: {len(images_list)} images | {len(anns_list)} annotations")
    return coco_gt, image_meta


def predict_yolo(weights, image_meta, conf_threshold):
    """Run YOLO inference and return COCO-format predictions."""
    model = YOLO(weights)
    predictions = []

    for meta in image_meta:
        results = model.predict(meta["path"], conf=conf_threshold,
                                verbose=False, save=False)
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                if cls_id not in YOLO_TO_COCO:
                    continue
                x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
                predictions.append({
                    "image_id":    meta["id"],
                    "category_id": YOLO_TO_COCO[cls_id],
                    "bbox":        [x1, y1, max(x2 - x1, 1), max(y2 - y1, 1)],
                    "score":       float(box.conf[0]),
                })
    return predictions


def run_coco_eval(coco_gt, predictions):
    if not predictions:
        print("[ERROR] No predictions.")
        return None, []

    coco_pred = coco_gt.loadRes(predictions)

    print("\nOverall COCO bbox metrics:")
    coco_eval = COCOeval(coco_gt, coco_pred, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    print("\nPer-category COCO metrics:")
    header = f"{'Category':<15}" + "".join(f"{m:>12}" for m in METRIC_NAMES)
    print(header)
    print("-" * len(header))

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

    return coco_eval, per_cat_stats


def save_results(output_dir, aug_mode, base_model, conf,
                 predictions, coco_eval, per_cat_stats):
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "predictions.json"), "w") as f:
        json.dump(predictions, f)

    metrics = {
        "model":   base_model,
        "aug":     aug_mode,
        "conf":    conf,
        "date":    datetime.now().isoformat(),
        "overall": {k: float(v) for k, v in zip(METRIC_NAMES, coco_eval.stats)},
        "per_category": {
            cat["name"]: {k: float(v) for k, v in zip(METRIC_NAMES, stats)}
            for cat, stats in zip(EVAL_CAT_INFO, per_cat_stats)
        },
    }
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    with open(os.path.join(output_dir, "summary.txt"), "w") as f:
        f.write(f"Model  : {base_model}\n")
        f.write(f"Aug    : {aug_mode}\n")
        f.write(f"Conf   : {conf}\n")
        f.write(f"Date   : {metrics['date']}\n\n")
        f.write("── Overall ──\n")
        for k, v in metrics["overall"].items():
            f.write(f"  {k:<15} {v:.4f}\n")
        f.write("\n── Per category ──\n")
        header = f"{'Category':<15}" + "".join(f"{m:>12}" for m in METRIC_NAMES)
        f.write(header + "\n" + "─" * len(header) + "\n")
        for cat_name, stats in metrics["per_category"].items():
            f.write(f"{cat_name:<15}" + "".join(f"{v:>12.4f}" for v in stats.values()) + "\n")

    print(f"\n  Results -> {output_dir}")
    print(f"    predictions.json | metrics.json | summary.txt")



def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune YOLO on KITTI-MOTS")
    parser.add_argument("--aug",      choices=["none", "light", "heavy"], default="none",
                        help="Augmentation level (default: none)")
    parser.add_argument("--model",    default="yolo26x.pt",
                        help="Base YOLO weights (default: yolo26x.pt)")
    parser.add_argument("--epochs",   type=int,   default=20)
    parser.add_argument("--batch",    type=int,   default=8)
    parser.add_argument("--imgsz",    type=int,   default=640)
    parser.add_argument("--lr0",      type=float, default=0.001)
    parser.add_argument("--freeze",   type=int,   default=0,
                        help="Layers to freeze: 0=full fine-tune, 10=head+neck, 20=head only")
    parser.add_argument("--conf",     type=float, default=0.5,
                        help="Confidence threshold for eval")
    parser.add_argument("--data-dir", default="outputs/kitti_mots_yolo",
                        help="Where to store YOLO-format dataset")
    parser.add_argument("--output-dir", default="outputs/task_e_yolo")
    parser.add_argument("--force-prepare", action="store_true",
                        help="Re-generate YOLO dataset even if it already exists")
    return parser.parse_args()


def main():
    args = parse_args()
    freeze_label = {0: "full", 10: "neck_head", 20: "head_only"}.get(args.freeze, f"freeze{args.freeze}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name  = f"aug_{args.aug}_{freeze_label}_{timestamp}"
    eval_dir  = os.path.join(args.output_dir, run_name, "eval")

    print(f"\nTask E - YOLO fine-tune on KITTI-MOTS")
    print(f"  model: {args.model} | aug: {args.aug} | freeze: {args.freeze} ({freeze_label})")
    print(f"  epochs: {args.epochs} | batch: {args.batch} | imgsz: {args.imgsz}\n")

    print("Step 1: Prepare YOLO dataset")
    dataset_yaml = prepare_yolo_dataset(args.data_dir, force=args.force_prepare)

    print("\nStep 2: Fine-tune YOLO")
    best_weights, train_run_dir = train_yolo(
        dataset_yaml = dataset_yaml,
        aug_mode     = args.aug,
        base_model   = args.model,
        epochs       = args.epochs,
        batch        = args.batch,
        imgsz        = args.imgsz,
        lr0          = args.lr0,
        freeze       = args.freeze,
        output_dir   = os.path.join(args.output_dir, run_name, "train"),
        run_name     = "yolo",
    )

    print("\nStep 3: Loss curves")
    plot_losses(train_run_dir, eval_dir)

    print("\nStep 4: COCO evaluation on val split")
    coco_gt, image_meta = build_coco_gt_from_seqmap(VAL_SEQMAP)

    print(f"  Running inference (conf >= {args.conf})...")
    predictions = predict_yolo(best_weights, image_meta, conf_threshold=args.conf)
    print(f"  Total detections: {len(predictions)}")

    coco_eval, per_cat_stats = run_coco_eval(coco_gt, predictions)

    if coco_eval is not None:
        print("\nStep 5: Saving results")
        save_results(eval_dir, args.aug, args.model, args.conf,
                     predictions, coco_eval, per_cat_stats)


if __name__ == "__main__":
    main()
