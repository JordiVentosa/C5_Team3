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

from ultralytics import YOLO


# --- CONFIGURATION ---
DATASET_PATH = "/home/mcv/datasets/C5/KITTI-MOTS"
FT_MODEL_PATH = "models/da_config_1/best_model"
YOLO_MODEL_NAME = "yolo26x.pt"
YOLO_DETECTIONS_PATH = "detections/yolo26x/detections.txt"
EVAL_MODE = "base"  # Options: "base", "ft", "both"
OUTPUT_JSON = "metrics_all_prompts_" + EVAL_MODE + ".json"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_yolo_detections(val_dataset):
    """Generates or loads YOLO detections using the exact images from the dataset."""
    if os.path.exists(YOLO_DETECTIONS_PATH):
        print(f"Loading YOLO detections from {YOLO_DETECTIONS_PATH}...")
        detections = {}
        with open(YOLO_DETECTIONS_PATH, 'r') as f:
            for line in f:
                seq, frame, cls_id, x1, y1, x2, y2, conf = line.strip().split(',')
                key = f"{seq}_{frame}"
                if key not in detections: detections[key] = []
                mapped_cls = 2 if int(cls_id) == 0 else (1 if int(cls_id) == 2 else -1)
                if mapped_cls != -1:
                    detections[key].append([float(x1), float(y1), float(x2), float(y2)])
        return detections

    print(f"Generating YOLO detections using {YOLO_MODEL_NAME}...")
    os.makedirs(os.path.dirname(YOLO_DETECTIONS_PATH), exist_ok=True)
    yolo_model = YOLO(YOLO_MODEL_NAME)
    detections = {}
    
    with open(YOLO_DETECTIONS_PATH, 'w') as f:
        # val_dataset.samples contains (img_path, mask_path, seq, filename)
        for img_path, _, seq, filename in tqdm(val_dataset.samples, desc="YOLO Inference"):
            frame = filename.split('.')[0]
            results = yolo_model(img_path, verbose=False)[0]
            
            key = f"{seq}_{frame}"
            detections[key] = []
            for box in results.boxes:
                cls_id = int(box.cls[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                f.write(f"{seq},{frame},{cls_id},{x1},{y1},{x2},{y2},{conf}\n")
                
                mapped_cls = 2 if cls_id == 0 else (1 if cls_id == 2 else -1)
                if mapped_cls != -1:
                    detections[key].append([x1, y1, x2, y2])
                    
    return detections


def run_evaluation(model, processor, dataset, yolo_detections, prompt_type):
    print(f"Evaluating prompt strategy: {prompt_type}")
    evaluator = HF_COCOEvaluator()
    model.eval()
    np.random.seed(42)

    for i in tqdm(range(len(dataset)), desc=prompt_type):
        data = dataset[i]
        image = Image.fromarray(data["image"])
        img_w, img_h = image.size
        evaluator.add_gt(data["image_id"], img_h, img_w, data["boxes"], data["masks"], data["classes"])
        
        pred_masks_for_image = []
        
        # Retrieve seq and frame manually from the dataset samples for YOLO matching
        _, _, seq, filename = dataset.samples[i]
        frame = filename.split('.')[0]
        
        if prompt_type == "perfect_bbox":
            inputs = processor(images=image, input_boxes=[data["boxes"]], return_tensors="pt")
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs, multimask_output=False)
                masks = processor.post_process_masks(outputs.pred_masks, inputs["original_sizes"].tolist(), inputs["reshaped_input_sizes"].tolist(), binarize=False)
                pred_masks_for_image.extend(torch.sigmoid(masks[0].squeeze(1)).cpu().numpy())
                
        elif prompt_type == "yolo_bbox":
            key = f"{seq}_{frame}"
            yolo_boxes = yolo_detections.get(key, [])
            if len(yolo_boxes) > 0:
                inputs = processor(images=image, input_boxes=[yolo_boxes], return_tensors="pt")
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = model(**inputs, multimask_output=False)
                    masks = processor.post_process_masks(outputs.pred_masks, inputs["original_sizes"].tolist(), inputs["reshaped_input_sizes"].tolist(), binarize=False)
                    pred_masks_for_image.extend(torch.sigmoid(masks[0].squeeze(1)).cpu().numpy())
                    
        else:
            for gt_mask in data["masks"]:
                if prompt_type == "point_center": pts, lbls = prompt_center_positive(gt_mask)
                elif prompt_type == "point_random": pts, lbls = prompt_random_positive(gt_mask)
                elif prompt_type == "point_3_random": pts, lbls = prompt_three_positives(gt_mask)
                elif prompt_type == "point_pos_neg": pts, lbls = prompt_pos_plus_neg(gt_mask)
                
                inputs = processor(images=image, input_points=[[pts.tolist()]], input_labels=[[lbls.tolist()]], return_tensors="pt")
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = model(**inputs, multimask_output=False)
                    masks = processor.post_process_masks(outputs.pred_masks, inputs["original_sizes"].tolist(), inputs["reshaped_input_sizes"].tolist(), binarize=False)
                    pred_masks_for_image.append(torch.sigmoid(masks[0].squeeze(0).squeeze(0)).cpu().numpy())

        evaluator.add_preds(data["image_id"], pred_masks_for_image, data["classes"])

    mAP = evaluator.evaluate()
    print(f"  {prompt_type}: {mAP:.4f}")
    return mAP

def main():
    print("Configurations:")
    print(f"  Dataset Path: {DATASET_PATH}")
    print(f"  Fine-Tuned Model Path: {FT_MODEL_PATH}")
    print(f"  YOLO Model: {YOLO_MODEL_NAME}")
    print(f"  Evaluation Mode: {EVAL_MODE}")
    print(f"  Output JSON: {OUTPUT_JSON}")
    print(f"  Device: {DEVICE}")
    print("\nStarting evaluation process...")

    print("Loading validation dataset...")
    val_dataset = KittiMotsSamDataset(root_dir=DATASET_PATH, split='val', transforms=None)
    yolo_detections = get_yolo_detections(val_dataset)
    
    prompt_strategies = ["perfect_bbox", "yolo_bbox", "point_center", "point_random", "point_3_random", "point_pos_neg"]
    
    final_metrics = {"SAM_Base": {}, "SAM_FineTuned": {}}
    if os.path.exists(OUTPUT_JSON):
        with open(OUTPUT_JSON, 'r') as f:
            try:
                final_metrics = json.load(f)
            except json.JSONDecodeError:
                pass
    
    if EVAL_MODE in ["base", "both"]:
        print("\nEvaluating SAM base...")
        base_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
        base_model = SamModel.from_pretrained("facebook/sam-vit-base").to(DEVICE)
        
        if "SAM_Base" not in final_metrics:
            final_metrics["SAM_Base"] = {}
            
        for p_type in prompt_strategies:
            final_metrics["SAM_Base"][p_type] = run_evaluation(base_model, base_processor, val_dataset, yolo_detections, p_type)
        
        del base_model
        torch.cuda.empty_cache()
    
    if EVAL_MODE in ["ft", "both"]:
        print("\nEvaluating SAM fine-tuned...")
        ft_processor = SamProcessor.from_pretrained(FT_MODEL_PATH)
        ft_model = SamModel.from_pretrained(FT_MODEL_PATH).to(DEVICE)
        
        if "SAM_FineTuned" not in final_metrics:
            final_metrics["SAM_FineTuned"] = {}
        for p_type in prompt_strategies:
            final_metrics["SAM_FineTuned"][p_type] = run_evaluation(ft_model, ft_processor, val_dataset, yolo_detections, p_type)
        
        del ft_model
        torch.cuda.empty_cache()
    
    print(f"Saving results to {OUTPUT_JSON}...")
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(final_metrics, f, indent=4)
    print("Evaluation complete.")

if __name__ == "__main__":
    main()