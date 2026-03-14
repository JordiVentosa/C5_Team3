import numpy as np
from pathlib import Path
from PIL import Image
 
import torch
from transformers import SamModel, SamProcessor

class SAMPredictor:
    def __init__(self, model_name: str, device: str):
        self.device    = device
        self.model     = SamModel.from_pretrained(model_name).to(device).eval()
        self.processor = SamProcessor.from_pretrained(model_name)
        self._pil      = None
        self._embeddings = None
 
    @torch.no_grad()
    def set_image(self, image_rgb: np.ndarray):
        self._pil = Image.fromarray(image_rgb)
        inputs    = self.processor(images=self._pil, return_tensors="pt")
        self._embeddings = self.model.get_image_embeddings(
            inputs["pixel_values"].to(self.device)
        )
 
    @torch.no_grad()
    def predict_point(self, point_coords: np.ndarray, point_labels: np.ndarray) -> tuple:
        """
        Returns (best_mask, best_score) where best_mask is (H, W) bool.
        """
        inputs = self.processor(
            images=self._pil,
            input_points=[point_coords.tolist()],
            input_labels=[point_labels.tolist()],
            return_tensors="pt",
        )
        outputs = self.model(
            input_points=inputs["input_points"].to(self.device),
            input_labels=inputs["input_labels"].to(self.device),
            image_embeddings=self._embeddings,
            multimask_output=True,
        )
        masks = self.processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"],
            inputs["reshaped_input_sizes"],
        )
        masks_np  = np.array(masks[0].squeeze(0))          # (3, H, W)
        scores_np = outputs.iou_scores.cpu().numpy().flatten()
        best      = int(np.argmax(scores_np))
        return masks_np[best].astype(bool), float(scores_np[best])
    
    @torch.no_grad()
    def predict_box(self, bboxes: np.ndarray) -> tuple:
        """
        Returns (best_mask, best_score) where best_mask is (H, W) bool.
        """
        inputs = self.processor(
            images=self._pil,
            input_boxes=bboxes,
            return_tensors="pt",
        )
        outputs = self.model(
            input_boxes=inputs["input_boxes"].to(self.device),
            image_embeddings=self._embeddings,
            multimask_output=True,
        )
        masks = self.processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"],
            inputs["reshaped_input_sizes"],
        )
        masks_np  = np.array(masks[0].squeeze(0))          # (3, H, W)
        scores_np = outputs.iou_scores.cpu().numpy().flatten()
        best      = int(np.argmax(scores_np))
        return masks_np[best].astype(bool), float(scores_np[best])