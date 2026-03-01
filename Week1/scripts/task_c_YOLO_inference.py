import argparse
import os
import glob
import cv2
import numpy as np
from pycocotools import mask as mask_utils
from src.models.ultralytics_yolo import YOLOInference

# KITTI-MOTS ground-truth class IDs
KITTI_MOTS_CLASSES = {1: "Car", 2: "Pedestrian"}

GT_COLORS  = {1: (0, 220, 0) , 2: (0, 160, 0) }
PRED_COLOR = (0, 0, 255)

#COCO  0 = "person"  <->  KITTI-MOTS 2 = Pedestrian
#COCO  2 = "car"     <->  KITTI-MOTS 1 = Car
YOLO_ALLOWED_CLASSES = {0, 2}


def decode_rle_to_bbox(rle_str, height, width):
    """
    Decode a KITTI-MOTS RLE-encoded mask and return its tight bounding box
    as (x1, y1, x2, y2).  Returns None if the mask is empty.
    """
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
    """
    Load all ground-truth annotations for one sequence.

    File format (instances_txt/<seq>.txt), one object per line:
        frame_id  object_id  class_id  height  width  rle

    Returns a dict mapping each frame_id to a list of annotations:
        { frame_id: [(class_id, instance_id, (x1, y1, x2, y2)), ...] }
    """
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

            # Format: frame_id  object_id  class_id  height  width  rle
            frame_id  = int(parts[0])
            object_id = int(parts[1])
            height    = int(parts[3])
            width     = int(parts[4])
            rle_str   = parts[5]

            class_id    = object_id // 1000
            instance_id = object_id % 1000

            # Skip ignore regions (class 10) and any unknown class
            if class_id not in KITTI_MOTS_CLASSES:
                continue

            bbox = decode_rle_to_bbox(rle_str, height, width)
            if bbox is None:
                continue

            annotations.setdefault(frame_id, []).append((class_id, instance_id, bbox))

    return annotations


def draw_gt_boxes(img, frame_annotations):
    """Draw ground-truth bounding boxes in green on img (in-place)."""
    for class_id, instance_id, (x1, y1, x2, y2) in frame_annotations:
        color = GT_COLORS[class_id]
        label = f"GT {KITTI_MOTS_CLASSES[class_id]} #{instance_id}"
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, max(y1 - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)


def draw_yolo_boxes(img, yolo_results, model_names):
    """Draw YOLO predicted bounding boxes in red on img (in-place).
    Only draws boxes for classes that exist in KITTI-MOTS (car & person)."""
    for result in yolo_results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in YOLO_ALLOWED_CLASSES:
                continue  # skip trucks, buses, cyclists, etc.

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            confidence = float(box.conf[0])
            class_name = model_names[cls_id]
            label = f"{class_name} {confidence:.2f}"

            cv2.rectangle(img, (x1, y1), (x2, y2), PRED_COLOR, 2)
            cv2.putText(img, label, (x1, min(y2 + 15, img.shape[0] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, PRED_COLOR, 1)



def run_inference_on_sequences(yolo_detector, base_dir, seq_range, split_name, output_dir):
    """Run YOLO inference and overlay GT boxes for a range of sequences."""
    instances_txt_dir = os.path.join(base_dir, "instances_txt")
    image_base_dir    = os.path.join(base_dir, "training", "image_02")
    output_base       = os.path.join(output_dir, split_name)

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

        # Load all GT annotations for this sequence
        gt_annotations = load_gt_annotations(instances_txt_dir, seq_str)

        output_seq_dir = os.path.join(output_base, seq_str)
        os.makedirs(output_seq_dir, exist_ok=True)

        print(f"  [{split_name}] sequence {seq_str}: {len(image_paths)} frames")

        for img_path in image_paths:
            frame_id = int(os.path.splitext(os.path.basename(img_path))[0])
            img = cv2.imread(img_path)

            # Draw ground-truth boxes (green)
            draw_gt_boxes(img, gt_annotations.get(frame_id, []))

            # Run YOLO and draw predicted boxes (red)
            yolo_results = yolo_detector.predict(img_path, conf_threshold=0.5, save_results=False)
            draw_yolo_boxes(img, yolo_results, yolo_detector.model.names)

            # Save the annotated frame
            out_path = os.path.join(output_seq_dir, os.path.basename(img_path))
            cv2.imwrite(out_path, img)


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO inference + GT overlay on KITTI-MOTS")
    parser.add_argument("--model", default="yolo26x.pt", help="YOLO weights (default: yolo26x.pt)")
    parser.add_argument("--base-dir", default="/home/mcv/datasets/C5/KITTI-MOTS",
                        help="Root KITTI-MOTS directory")
    parser.add_argument("--output-dir", default="runs/inference",
                        help="Base output directory for annotated frames")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading YOLO model ({args.model}) ...")
    yolo_detector = YOLOInference(model_version=args.model)

    # Sequences 0000-0015 -> training split
    print("\n=== Training sequences (0000–0015) ===")
    run_inference_on_sequences(yolo_detector, args.base_dir, range(0, 16), "training", args.output_dir)

    # Sequences 0016-0020 -> test split
    print("\n=== Test sequences (0016–0020) ===")
    run_inference_on_sequences(yolo_detector, args.base_dir, range(16, 21), "test", args.output_dir)

    print("\nDone!")

if __name__ == "__main__":
    main()