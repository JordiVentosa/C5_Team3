import os
import glob
import cv2
import numpy as np
from pycocotools import mask as mask_utils
from Week1.src.models.torchvision_faster_rcnn import FasterRCNNInference
from Week1.src.models.ultralytics_yolo import YOLOInference

# KITTI-MOTS ground-truth class IDs
KITTI_MOTS_CLASSES = {1: "Car", 2: "Pedestrian"}

GT_COLORS  = {1: (0, 220, 0), 2: (0, 160, 0)}
PRED_COLOR = (0, 0, 255)

# COCO  0 = "person"  <->  KITTI-MOTS 2 = Pedestrian
# COCO  2 = "car"     <->  KITTI-MOTS 1 = Car
YOLO_ALLOWED_CLASSES = {0, 2}

# COCO class IDs for Faster R-CNN (1-indexed in torchvision)
# 1 = person, 3 = car
FRCNN_ALLOWED_CLASSES = {1, 3}
FRCNN_CLASS_NAMES = {1: "person", 3: "car"}



# ── Drawing helpers ───────────────────────────────────────────────────────────

def draw_frcnn_boxes(img, detections):
    """Draw Faster R-CNN predicted bounding boxes in red on img (in-place)."""
    for label_id, score, (x1, y1, x2, y2) in detections:
        class_name = FRCNN_CLASS_NAMES[label_id]
        label = f"{class_name} {score:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), PRED_COLOR, 2)
        cv2.putText(img, label, (x1, min(y2 + 15, img.shape[0] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, PRED_COLOR, 1)


# ── Unchanged helpers ─────────────────────────────────────────────────────────

def decode_rle_to_bbox(rle_str, height, width):
    rle = {"counts": rle_str.encode(), "size": [height, width]}
    binary_mask = mask_utils.decode(rle)
    occupied_rows = np.any(binary_mask, axis=1)
    occupied_cols = np.any(binary_mask, axis=0)
    if not occupied_rows.any():
        return None
    y1, y2 = np.where(occupied_rows)[0][[0, -1]]
    x1, x2 = np.where(occupied_cols)[0][[0, -1]]
    return (int(x1), int(y1), int(x2), int(y2))


def load_gt_annotations(instances_txt_dir, seq_str):
    ann_file = os.path.join(instances_txt_dir, f"{seq_str}.txt")
    annotations = {}
    if not os.path.exists(ann_file):
        print(f"  [WARN] No GT annotation file found: {ann_file}")
        return annotations
    with open(ann_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            frame_id  = int(parts[0])
            object_id = int(parts[1])
            height    = int(parts[3])
            width     = int(parts[4])
            rle_str   = parts[5]
            class_id    = object_id // 1000
            instance_id = object_id % 1000
            if class_id not in KITTI_MOTS_CLASSES:
                continue
            bbox = decode_rle_to_bbox(rle_str, height, width)
            if bbox is None:
                continue
            annotations.setdefault(frame_id, []).append((class_id, instance_id, bbox))
    return annotations


def draw_gt_boxes(img, frame_annotations):
    for class_id, instance_id, (x1, y1, x2, y2) in frame_annotations:
        color = GT_COLORS[class_id]
        label = f"GT {KITTI_MOTS_CLASSES[class_id]} #{instance_id}"
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, max(y1 - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)


def draw_yolo_boxes(img, yolo_results, model_names):
    for result in yolo_results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in YOLO_ALLOWED_CLASSES:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            confidence = float(box.conf[0])
            class_name = model_names[cls_id]
            label = f"{class_name} {confidence:.2f}"
            cv2.rectangle(img, (x1, y1), (x2, y2), PRED_COLOR, 2)
            cv2.putText(img, label, (x1, min(y2 + 15, img.shape[0] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, PRED_COLOR, 1)


# ── Main inference loop ───────────────────────────────────────────────────────

def run_inference_on_sequences(detector, base_dir, seq_range, split_name, model_type="yolo"):
    """
    Run inference and overlay GT boxes for a range of sequences.

    Parameters
    ----------
    detector   : YOLOInference | FasterRCNNInference
    model_type : "yolo" | "frcnn"
    """
    instances_txt_dir = os.path.join(base_dir, "instances_txt")
    image_base_dir    = os.path.join(base_dir, "training", "image_02")
    output_base       = os.path.join("runs/inference", split_name)

    l = 0
    for seq_id in seq_range:
        seq_str  = f"{seq_id:04d}"
        seq_path = os.path.join(image_base_dir, seq_str)

        if not os.path.isdir(seq_path):
            print(f"  [SKIP] Sequence folder not found: {seq_path}")
            continue

        image_paths = sorted(glob.glob(os.path.join(seq_path, "*.png")))
        if not image_paths:
            print(f"  [SKIP] No images in {seq_path}")
            continue

        gt_annotations = load_gt_annotations(instances_txt_dir, seq_str)
        output_seq_dir = os.path.join(output_base, seq_str)
        os.makedirs(output_seq_dir, exist_ok=True)

        print(f"  [{split_name}] sequence {seq_str}: {len(image_paths)} frames")

        for img_path in image_paths:
            l += 1
            frame_id = int(os.path.splitext(os.path.basename(img_path))[0])
            img = cv2.imread(img_path)

            # Ground-truth boxes (green)
            draw_gt_boxes(img, gt_annotations.get(frame_id, []))

            # Predicted boxes (red) — branch on model type
            if model_type == "yolo":
                results = detector.predict(img_path, conf_threshold=0.5, save_results=False)
                draw_yolo_boxes(img, results, detector.model.names)
            elif model_type == "frcnn":
                results = detector.predict(img_path, conf_threshold=0.5)
                draw_frcnn_boxes(img, results)
            else:
                raise ValueError(f"Unknown model_type '{model_type}'. Use 'yolo' or 'frcnn'.")

            out_path = os.path.join(output_seq_dir, os.path.basename(img_path))
            cv2.imwrite(out_path, img)
    print(l, "images")

def main():
    base_dir = "/home/msiau/workspace/jventosa/PostTFG/Master/C5_Team3/Week1/datasets/KITTI-MOTS"


    MODEL_TYPE = "yolo"   # "yolo" | "frcnn"

    if MODEL_TYPE == "yolo":
        print("Loading YOLO model...")
        detector = YOLOInference(model_version='yolov8x.pt')
    elif MODEL_TYPE == "frcnn":
        print("Loading Faster R-CNN model...")
        detector = FasterRCNNInference(conf_threshold=0.5)
    else:
        raise ValueError(f"Unknown MODEL_TYPE: {MODEL_TYPE}")



    print("\n=== Training sequences")
    run_inference_on_sequences(detector, base_dir, [0,1,3,4,5,9,11,12,15,17,19,20],  "training", MODEL_TYPE)

    print("\n=== Test sequences")
    run_inference_on_sequences(detector, base_dir, [2,6,7,8,10,13,14,16,18,], "test",     MODEL_TYPE)

    print("\nDone!")


if __name__ == "__main__":
    main()