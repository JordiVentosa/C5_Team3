import io
import contextlib
import cv2
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from Week1.src.models.torchvision_faster_rcnn import FasterRCNNInference
from Week1.src.utils.dataset import KittyDataset

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

os.environ["CUDA_VISIBLE_DEVICES"]="2"

EVAL_CAT_INFO = [
    {"id": 1, "name": "Pedestrian", "supercategory": "person"},
    {"id": 3, "name": "Car",        "supercategory": "vehicle"},
]

FRCNN_ALLOWED = {1: 1, 3: 3}

model_names = ["resnet50","resnet50_v2","mobilenet_v3","mobilenet_320"]


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


def run_coco_eval(coco_gt, predictions, output_file="coco_eval_results.txt"):
    """
    Runs COCOeval and:
      - Prints standard 12-metric summary (overall)
      - Prints per-category table with all 12 metrics
      - Saves the same content to a text file
    """
    if not predictions:
        print("[ERROR] No predictions to evaluate.")
        return None

    coco_pred = coco_gt.loadRes(predictions)

    with open(output_file, "w", encoding="utf-8") as f:

        print("\n── Overall COCO bbox metrics ───────────────────────────────")
        f.write("\n── Overall COCO bbox metrics ───────────────────────────────\n")

        coco_eval = COCOeval(coco_gt, coco_pred, iouType="bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()

        # Capture summarize() output
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            coco_eval.summarize()
        summary_text = buf.getvalue()

        print(summary_text)
        f.write(summary_text)

        metric_names = [
            "AP[.5:.95]", "AP@.50", "AP@.75",
            "AP@small",   "AP@med", "AP@large",
            "AR@1",       "AR@10",  "AR@100",
            "AR@small",   "AR@med", "AR@large",
        ]

        print("\n── Per-category COCO metrics ───────────────────────────────")
        f.write("\n── Per-category COCO metrics ───────────────────────────────\n")

        header = f"{'Category':<15}" + "".join(f"{m:>12}" for m in metric_names)
        print(header)
        print("─" * len(header))
        f.write(header + "\n")
        f.write("─" * len(header) + "\n")

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
            f.write(row + "\n")

        print("────────────────────────────────────────────────────────────\n")
        f.write("────────────────────────────────────────────────────────────\n\n")

    print(f"[INFO] Results saved to {output_file}")

    return coco_eval


def main():
    BASE_DIR   = "/home/msiau/data/tmp/agarciat/MCVC/C5/KITTI-MOTS"
    RUN_SPLIT  = "all"
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

    for model_name in model_names:
        print("\nLoading Faster R-CNN (ResNet-50 FPN) ...")
        detector = FasterRCNNInference(conf_threshold=CONF_THR, model_name=model_name)
        print(f"Running Faster R-CNN predictions  [conf>={CONF_THR}] ...")
        predictions = predict_frcnn(detector, image_meta, conf_threshold=CONF_THR)
        print(f"  Total detections      : {len(predictions)}")
        print(f"  Prediction categories : {sorted({p['category_id'] for p in predictions})}")

        print(f"\n── COCO Evaluation  |  model=frcnn  split={RUN_SPLIT} ──")
        run_coco_eval(coco_gt, predictions, output_file=f"{model_name}.txt")

    


if __name__ == "__main__":
    main()