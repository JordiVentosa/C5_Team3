from PIL import Image
import cv2
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
 
CATEGORY_COLORS = {
    1: (0,   114, 189),
    2: (217,  83,  25),
}
 
def instance_color(category_id: int, idx: int) -> tuple:
    base   = np.array(CATEGORY_COLORS.get(category_id, (128, 128, 128)), dtype=np.float32)
    factor = 0.6 + 0.4 * ((idx * 37) % 100) / 100.0
    return tuple((base * factor).clip(0, 255).astype(np.uint8).tolist())
 
 
def overlay_masks(image_rgb: np.ndarray, masks: list, cats: list,
                  alpha: float = 0.45) -> np.ndarray:
    canvas = image_rgb.copy().astype(np.float32)
    for i, (mask, cat) in enumerate(zip(masks, cats)):
        color = np.array(instance_color(cat, i), dtype=np.float32)
        canvas[mask] = canvas[mask] * (1 - alpha) + color * alpha
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(canvas.astype(np.uint8), contours, -1,
                         instance_color(cat, i), 2)
    return canvas.clip(0, 255).astype(np.uint8)
 
 
def save_overlays(image_rgb, gt_masks, gt_cats, pred_masks, pred_cats,
                  output_dir: Path, image_id: int):
    Image.fromarray(overlay_masks(image_rgb, gt_masks,   gt_cats))\
         .save(output_dir / f"{image_id:06d}_gt.png")
    Image.fromarray(overlay_masks(image_rgb, pred_masks, pred_cats))\
         .save(output_dir / f"{image_id:06d}_pred.png")
         
         
def save_overlays_with_boxes(image_rgb, gt_masks, gt_cats,
                             pred_masks, pred_cats, boxes_xyxy,
                             output_dir: Path, image_id: int):
    """Like save_overlays but also draws YOLO bounding boxes on the pred image."""
    
    # GT image — masks only, no boxes
    Image.fromarray(overlay_masks(image_rgb, gt_masks, gt_cats)).save(output_dir / f"{image_id:06d}_gt.png")
 
    # Pred image — masks + bounding boxes
    canvas = overlay_masks(image_rgb, pred_masks, pred_cats)
    
    for i, (box, cat) in enumerate(zip(boxes_xyxy, pred_cats)):
        
        x1, y1, x2, y2 = map(int, box)
        
        color = instance_color(cat, i)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color,
                      2)
        label = {1: "car", 2: "pedestrian"}.get(cat, str(cat))
        cv2.putText(canvas, label, (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        
    Image.fromarray(canvas).save(output_dir / f"{image_id:06d}_pred.png")