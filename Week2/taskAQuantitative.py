import os
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import SamModel, SamProcessor
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mask_utils

# Import your custom dataset
from dataset import KittiMotsSamDataset


# --- CONFIGURATION ---
DATASET_PATH = "/home/mcv/datasets/C5/KITTI-MOTS"
OUTPUT_JSON = "metrics_points_only_huge.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Negative prompt parameters
NEG_MARGIN = 20
NEG_MAX_TRIES = 500


# ==========================================
# 1. PROMPT GENERATORS
# ==========================================

def nearest_foreground_to_point(mask: np.ndarray, yx_point: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return np.array([yx_point[1], yx_point[0]], dtype=np.float32)
    coords = np.stack([ys, xs], axis=1)
    d2 = np.sum((coords - yx_point[None, :]) ** 2, axis=1)
    best = coords[np.argmin(d2)]
    return np.array([best[1], best[0]], dtype=np.float32)


def prompt_center_positive(mask: np.ndarray):
    ys, xs = np.where(mask)
    centroid_y = int(np.round(np.mean(ys)))
    centroid_x = int(np.round(np.mean(xs)))
    point_xy = nearest_foreground_to_point(mask, np.array([centroid_y, centroid_x]))
    return np.array([point_xy], dtype=np.float32), np.array([1], dtype=np.int64)


def prompt_random_positive(mask: np.ndarray):
    ys, xs = np.where(mask)
    idx = np.random.randint(0, len(xs))
    point_xy = np.array([xs[idx], ys[idx]], dtype=np.float32)
    return np.array([point_xy], dtype=np.float32), np.array([1], dtype=np.int64)


def prompt_three_positives(mask: np.ndarray):
    ys, xs = np.where(mask)
    coords = np.stack([xs, ys], axis=1)
    center_pt, _ = prompt_center_positive(mask)
    chosen = [tuple(center_pt[0].astype(int).tolist())]

    if len(coords) > 1:
        perm = np.random.permutation(len(coords))
        for idx in perm:
            candidate = tuple(coords[idx].astype(int).tolist())
            if candidate not in chosen:
                chosen.append(candidate)
            if len(chosen) == 3:
                break

    while len(chosen) < 3:
        chosen.append(chosen[-1])

    return np.array(chosen, dtype=np.float32), np.array([1, 1, 1], dtype=np.int64)


def prompt_pos_plus_neg(mask: np.ndarray):
    pos_points, _ = prompt_center_positive(mask)
    pos_xy = pos_points[0]

    ys, xs = np.where(mask)
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()
    h, w = mask.shape

    x0, x1 = max(0, x_min - NEG_MARGIN), min(w - 1, x_max + NEG_MARGIN)
    y0, y1 = max(0, y_min - NEG_MARGIN), min(h - 1, y_max + NEG_MARGIN)

    neg_xy = None
    for _ in range(NEG_MAX_TRIES):
        x = np.random.randint(x0, x1 + 1)
        y = np.random.randint(y0, y1 + 1)
        if not mask[y, x]:
            neg_xy = np.array([x, y], dtype=np.float32)
            break

    if neg_xy is None:
        bg_ys, bg_xs = np.where(~mask)
        if len(bg_xs) == 0:
            neg_xy = pos_xy.copy()
        else:
            idx = np.random.randint(0, len(bg_xs))
            neg_xy = np.array([bg_xs[idx], bg_ys[idx]], dtype=np.float32)

    return np.array([pos_xy, neg_xy], dtype=np.float32), np.array([1, 0], dtype=np.int64)


# ==========================================
# 2. COCO EVALUATOR
# ==========================================

class HF_COCOEvaluator:
    def __init__(self):
        self.coco_gt = COCO()
        self.dataset = {
            "images": [],
            "annotations": [],
            "categories": [
                {"id": 1, "name": "Car"},
                {"id": 2, "name": "Pedestrian"},
            ],
        }
        self.ann_id = 1
        self.preds = []

    def add_gt(self, image_id, height, width, boxes, masks, classes):
        self.dataset["images"].append(
            {
                "id": image_id,
                "width": width,
                "height": height,
                "file_name": str(image_id),
            }
        )

        for box, mask, cls in zip(boxes, masks, classes):
            mask_fortran = np.asfortranarray(mask.astype(np.uint8))
            rle = mask_utils.encode(mask_fortran)
            rle["counts"] = rle["counts"].decode("utf-8")
            x1, y1, x2, y2 = map(float, box)

            self.dataset["annotations"].append(
                {
                    "id": self.ann_id,
                    "image_id": image_id,
                    "category_id": int(cls),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "segmentation": rle,
                    "area": float(mask_utils.area(rle)),
                    "iscrowd": 0,
                }
            )
            self.ann_id += 1

    def add_preds(self, image_id, masks, classes):
        for mask, cls in zip(masks, classes):
            mask_fortran = np.asfortranarray((mask > 0.5).astype(np.uint8))
            rle = mask_utils.encode(mask_fortran)
            rle["counts"] = rle["counts"].decode("utf-8")
            self.preds.append(
                {
                    "image_id": image_id,
                    "category_id": int(cls),
                    "segmentation": rle,
                    "score": 1.0,
                }
            )

    def evaluate(self):
        if not self.preds:
            return 0.0

        self.coco_gt.dataset = self.dataset
        self.coco_gt.createIndex()

        coco_dt = self.coco_gt.loadRes(self.preds)
        coco_eval = COCOeval(self.coco_gt, coco_dt, iouType="segm")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        return float(coco_eval.stats[0])  # mAP 0.50:0.95


# ==========================================
# 3. INFERENCE LOOP
# ==========================================

def run_evaluation(model, processor, dataset, prompt_type):
    print(f"\n--- Evaluating Prompt Strategy: {prompt_type} ---")
    evaluator = HF_COCOEvaluator()
    model.eval()

    np.random.seed(42)

    for i in tqdm(range(len(dataset)), desc=prompt_type):
        data = dataset[i]
        image = Image.fromarray(data["image"])
        img_w, img_h = image.size

        evaluator.add_gt(
            data["image_id"],
            img_h,
            img_w,
            data["boxes"],
            data["masks"],
            data["classes"],
        )

        pred_masks_for_image = []

        for gt_mask in data["masks"]:
            if prompt_type == "point_center":
                pts, lbls = prompt_center_positive(gt_mask)
            elif prompt_type == "point_random":
                pts, lbls = prompt_random_positive(gt_mask)
            elif prompt_type == "point_3_random":
                pts, lbls = prompt_three_positives(gt_mask)
            elif prompt_type == "point_pos_neg":
                pts, lbls = prompt_pos_plus_neg(gt_mask)
            else:
                raise ValueError(f"Unknown prompt type: {prompt_type}")

            inputs = processor(
                images=image,
                input_points=[[pts.tolist()]],
                input_labels=[[lbls.tolist()]],
                return_tensors="pt",
            )
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs, multimask_output=False)
                masks = processor.post_process_masks(
                    outputs.pred_masks,
                    inputs["original_sizes"].tolist(),
                    inputs["reshaped_input_sizes"].tolist(),
                    binarize=False,
                )
                pred_masks_for_image.append(
                    torch.sigmoid(masks[0].squeeze(0).squeeze(0)).cpu().numpy()
                )

        evaluator.add_preds(data["image_id"], pred_masks_for_image, data["classes"])

    mAP = evaluator.evaluate()
    print(f"[{prompt_type}] mAP: {mAP:.4f}")
    return mAP


# ==========================================
# 4. MAIN
# ==========================================

def main():
    print("Configurations:")
    print(f"  Dataset Path: {DATASET_PATH}")
    print(f"  Output JSON: {OUTPUT_JSON}")
    print(f"  Device: {DEVICE}")
    print("\nStarting evaluation process...")

    print("Loading Validation Dataset using KittiMotsSamDataset...")
    val_dataset = KittiMotsSamDataset(root_dir=DATASET_PATH, split="val", transforms=None)

    prompt_strategies = [
        "point_center",
        "point_random",
        "point_3_random",
        "point_pos_neg",
    ]

    print("\n=== EVALUATING SAM huge ===")
    huge_processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
    huge_model = SamModel.from_pretrained("facebook/sam-vit-huge").to(DEVICE)

    final_metrics = {}

    for p_type in prompt_strategies:
        final_metrics[p_type] = run_evaluation(
            huge_model,
            huge_processor,
            val_dataset,
            p_type,
        )

    del huge_model
    torch.cuda.empty_cache()

    print(f"\nSaving results to {OUTPUT_JSON}...")
    with open(OUTPUT_JSON, "w") as f:
        json.dump(final_metrics, f, indent=4)

    print("Done! Evaluation complete.")


if __name__ == "__main__":
    main()