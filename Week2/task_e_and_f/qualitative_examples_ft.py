import os
import torch
import numpy as np
from PIL import Image
from transformers import SamModel, SamProcessor

from visualization import save_visualization_with_ignore as save_visualization

# --- CONFIGURATION ---
DATASET_PATH = "/home/mcv/datasets/C5/KITTI-MOTS"
FT_MODEL_PATH = "models/da_config_1/best_model" # Change to your real FT model path
OUTPUT_DIR = "kitti_qualitative_results"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Target images to visualize (Format: "seq_frame")
TARGET_IMAGES = [
    "0000_000000", # Classmates' choice
    "0000_000150", # Classmates' choice
    "0006_000100", # Cars and pedestrians mixed
]

def load_kitti_image_and_gt(seq, frame_id):
    """Loads a specific image and extracts ground truth boxes, masks, and the ignore mask."""
    img_path = os.path.join(DATASET_PATH, 'training', 'image_02', seq, f"{frame_id}.png")
    mask_path = os.path.join(DATASET_PATH, 'instances', seq, f"{frame_id}.png")
    
    if not os.path.exists(img_path) or not os.path.exists(mask_path):
        print(f"[WARNING] Image or mask not found: {seq}_{frame_id}")
        return None, None, None, None
        
    image = Image.open(img_path).convert("RGB")
    mask_img = np.array(Image.open(mask_path))
    
    inst_ids = np.unique(mask_img)
    boxes, masks = [], []
    ignore_mask = np.zeros_like(mask_img, dtype=bool)
    
    for inst_id in inst_ids:
        # Extract DontCare (Ignore) regions
        if inst_id == 10000: 
            ignore_mask |= (mask_img == inst_id)
            continue 
            
        class_id = inst_id // 1000
        if class_id not in [1, 2]: continue # Only Cars(1) and Pedestrians(2)
            
        coords = np.argwhere(mask_img == inst_id)
        if len(coords) < 10: continue
            
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        if x_max - x_min > 2 and y_max - y_min > 2:
            boxes.append([x_min, y_min, x_max, y_max])
            masks.append((mask_img == inst_id).astype(np.float32))
            
    return image, boxes, masks, ignore_mask


def infer_and_visualize(model, processor, model_name):
    print(f"Running inference with: {model_name}")
    model.eval()
    for target in TARGET_IMAGES:
        seq, frame_id = target.split('_')
        image, boxes, true_masks, ignore_mask = load_kitti_image_and_gt(seq, frame_id)
        if image is None or len(boxes) == 0:
            print(f"Skipping {target}")
            continue
        print(f"Processing {target}")
        
        inputs = processor(images=image, input_boxes=[boxes], return_tensors="pt")
        with torch.no_grad():
            outputs = model(**{k: v.to(DEVICE) for k, v in inputs.items()}, multimask_output=False)
            masks = processor.post_process_masks(outputs.pred_masks, inputs["original_sizes"].tolist(), inputs["reshaped_input_sizes"].tolist(), binarize=False)
            save_visualization(image, boxes, torch.sigmoid(masks[0].squeeze(1)).cpu().numpy(), true_masks, ignore_mask, model_name, target, OUTPUT_DIR)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    base_model = SamModel.from_pretrained("facebook/sam-vit-base").to(DEVICE)
    infer_and_visualize(base_model, base_processor, "SAM_Base")
    del base_model
    torch.cuda.empty_cache()
    ft_processor = SamProcessor.from_pretrained(FT_MODEL_PATH)
    ft_model = SamModel.from_pretrained(FT_MODEL_PATH).to(DEVICE)
    infer_and_visualize(ft_model, ft_processor, "SAM_FT_Config1")

if __name__ == '__main__':
    main()