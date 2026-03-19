"""
MOTS Dataset for Segmentation
Based on: "MOTS: Multi-Object Tracking and Segmentation" (Voigtlaender et al., CVPR 2019)

Directory layout:
    root/
      instances/
        <seq_id>/   000000.png ...
      training/
        <seq_id>/   000000.png ...
      testing/
        <seq_id>/   000000.png ...

Instance PNG encoding (uint16):
    pixel = class_id * 1000 + instance_id
    pixel = 10000  ->  ignore region

Official KITTI MOTS splits (paper, Section 3 footnote 3)
---------------------------------------------------------
train: sequences 0-20 excluding val, images from training/
val  : sequences 2, 6, 7, 8, 10, 13, 14, 16, 18, images from testing/
"""

import numpy as np
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from pycocotools.coco import COCO

IGNORE_VALUE    = 10000
MAX_SEQUENCE    = 20

VAL_SEQUENCES   = {2, 6, 7, 8, 10, 13, 14, 16, 18}
ALL_SEQUENCES   = set(range(MAX_SEQUENCE + 1))
TRAIN_SEQUENCES = ALL_SEQUENCES - VAL_SEQUENCES

def build_coco_gt(root, split):
    """
    Build a COCO-formatted ground truth dict from KITTI MOTS annotations.
    Reads from instances_txt/ which contains pre-encoded RLE masks, avoiding
    any uint16 PNG loading issues.

    instances_txt format (one line per instance per frame):
        frame_id  obj_id  class_id  height  width  rle_counts

    Parameters
    ----------
    root  : str | Path
    split : "train" | "val"

    Returns
    -------
    dict with keys "info", "categories", "images", "annotations"
    ready to be passed to pycocotools.coco.COCO().

    Usage
    -----
        from pycocotools.coco import COCO

        coco_dict = build_coco_gt("data/KITTI-MOTS", split="val")

        coco_gt = COCO()
        coco_gt.dataset = coco_dict
        coco_gt.createIndex()
    """
    from pycocotools import mask as mask_utils

    root      = Path(root)
    valid_ids = VAL_SEQUENCES if split == "val" else TRAIN_SEQUENCES
    img_base  = root / ("training/image_02" if split == "val" else "training/image_02")
    txt_base  = root / "instances_txt"
    
    print(img_base)

    categories = [
        {"id": 1, "name": "car"},
        {"id": 2, "name": "pedestrian"},
    ]

    images      = []
    annotations = []
    image_id    = 0
    ann_id      = 0

    for txt_path in sorted(txt_base.glob("*.txt")):
        seq_id = int("".join(c for c in txt_path.stem if c.isdigit()))
        if seq_id not in valid_ids:
            continue

        seq_dir = img_base / f"{seq_id:04d}"
        if not seq_dir.exists():
            continue

        # parse all lines grouped by frame_id
        frames = {}
        with open(txt_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 6:
                    continue
                frame_id = int(parts[0])
                obj_id   = int(parts[1])
                class_id = int(parts[2])
                h, w     = int(parts[3]), int(parts[4])
                rle_str  = parts[5]

                # skip ignore regions
                if obj_id == IGNORE_VALUE or class_id == 10:
                    continue
                if class_id not in (1, 2):
                    continue

                rle  = {"size": [h, w], "counts": rle_str.encode("utf-8")}
                area = float(mask_utils.area(rle))
                if area == 0:
                    continue

                frames.setdefault(frame_id, []).append({
                    "class_id": class_id,
                    "rle":      {"size": [h, w], "counts": rle_str},
                    "area":     area,
                    "bbox":     mask_utils.toBbox(rle).tolist(),
                })

        for img_path in sorted(seq_dir.glob("*.png")):
            frame_id = int(img_path.stem)
            pil_img  = Image.open(img_path)
            W, H     = pil_img.size

            images.append({
                "id":        image_id,
                "file_name": str(img_path),
                "height":    H,
                "width":     W,
                "seq_id":    seq_id,
                "frame_id":  frame_id,
            })

            for ann in frames.get(frame_id, []):
                annotations.append({
                    "id":           ann_id,
                    "image_id":     image_id,
                    "category_id":  ann["class_id"],
                    "segmentation": ann["rle"],
                    "area":         ann["area"],
                    "iscrowd":      0,
                })
                ann_id += 1

            image_id += 1
            
            
    coco_dict = {
        "info":        {"description": "KITTI MOTS", "split": split},
        "categories":  categories,
        "images":      images,
        "annotations": annotations,
    }

    return coco_dict
    
if __name__=="__main__":
    
    coco_dict = build_coco_gt("/ghome/group03/mcv/datasets/C5/KITTI-MOTS", "train")
    print(coco_dict["images"][0]["file_name"])