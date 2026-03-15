import os
import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from transformers import SamModel, SamProcessor

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

def save_visualization(image, boxes, pred_masks, true_masks, ignore_mask, model_name, img_name, output_dir):
    image = np.array(image)
    safe_name = model_name.replace(" ", "_")
    np.random.seed(42)
    
    num_objects = max(len(pred_masks), len(true_masks))
    colors = np.random.rand(num_objects, 3)
    
    pred_overlay = np.zeros((image.shape[0], image.shape[1], 4)) 
    for idx, mask in enumerate(pred_masks):
        pred_overlay[mask > 0.5] = np.append(colors[idx], 0.85)
    
    gt_overlay = np.zeros((image.shape[0], image.shape[1], 4)) 
    for idx, mask in enumerate(true_masks):
        gt_overlay[mask > 0.5] = np.append(colors[idx], 0.85)
    
    ignore_overlay = None
    ignore_boxes = []
    if ignore_mask is not None and ignore_mask.any():
        ignore_overlay = np.zeros((image.shape[0], image.shape[1], 4))
        ignore_overlay[ignore_mask] = [120/255, 120/255, 120/255, 0.4]
        
        ignore_u8 = ignore_mask.astype(np.uint8)
        num_labels, labels_img, stats, _ = cv2.connectedComponentsWithStats(ignore_u8, connectivity=8)
        for comp_id in range(1, num_labels):
            stats_comp = stats[comp_id]
            area = int(stats_comp[cv2.CC_STAT_AREA])
            if area > 10:
                x, y, w, h = int(stats_comp[cv2.CC_STAT_LEFT]), int(stats_comp[cv2.CC_STAT_TOP]), int(stats_comp[cv2.CC_STAT_WIDTH]), int(stats_comp[cv2.CC_STAT_HEIGHT])
                ignore_boxes.append([x, y, x+w, y+h])
    
    def save_fig(img, overlay=None, draw_boxes=False, draw_ignore=False, suffix=""):
        fig, ax = plt.subplots(1, figsize=(12, 4))
        ax.imshow(img)
        if draw_ignore and ignore_overlay is not None:
            ax.imshow(ignore_overlay)
            for ix_min, iy_min, ix_max, iy_max in ignore_boxes:
                ax.add_patch(patches.Rectangle((ix_min, iy_min), ix_max - ix_min, iy_max - iy_min, linewidth=1.5, edgecolor=(120/255, 120/255, 120/255), facecolor='none'))
                ax.text(ix_min, max(10, iy_min - 5), "ignore class", color='white', fontsize=8, fontweight='bold', bbox=dict(facecolor=(120/255, 120/255, 120/255), alpha=0.7, pad=0.5, edgecolor='none'))
        if overlay is not None:
            ax.imshow(overlay)
        if draw_boxes:
            for x_min, y_min, x_max, y_max in boxes:
                ax.add_patch(patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=1.5, edgecolor='r', facecolor='none'))
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, f"{img_name}_{safe_name}_{suffix}.png"), bbox_inches='tight', pad_inches=0, dpi=200)
        plt.close(fig)

    save_fig(image, draw_ignore=False, suffix="1_original")
    save_fig(image, draw_boxes=True, draw_ignore=True, suffix="2_bboxes")
    save_fig(image, overlay=pred_overlay, draw_ignore=True, suffix="3_pred_masks")
    save_fig(image, overlay=gt_overlay, draw_ignore=True, suffix="4_gt_masks")
    save_fig(image, overlay=pred_overlay, draw_boxes=True, draw_ignore=True, suffix="5_combined")

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