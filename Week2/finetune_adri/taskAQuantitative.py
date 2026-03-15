import os
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import SamModel, SamProcessor

# Import your custom dataset
from dataset import KittiMotsSamDataset
from evaluators import HF_COCOEvaluator
from prompts import (prompt_center_positive, prompt_random_positive,
                     prompt_three_positives, prompt_pos_plus_neg)


# --- CONFIGURATION ---
DATASET_PATH = "/home/mcv/datasets/C5/KITTI-MOTS"
OUTPUT_JSON = "metrics_points_only_huge.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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