import torch
import numpy as np
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from transformers import SamModel, SamProcessor

class GroundedSAMPredictor:
    """
    Two-stage predictor: Grounding DINO (text -> boxes) + SAM (boxes -> masks).
 
    Grounding DINO is queried once per image with all class prompts joined,
    then its boxes are matched back to category IDs by label and passed to SAM.
    """
 
    def __init__(self, grounding_model: str, sam_model: str, device: str):
 
        self.device = device
 
 
        print("Loading DINO...")
        # Grounding DINO
        self.gdino_processor = AutoProcessor.from_pretrained(grounding_model)
        self.gdino_model     = AutoModelForZeroShotObjectDetection\
                                   .from_pretrained(grounding_model).to(device).eval()
 
        print("Loading SAM...")
        # SAM
        self.sam_model     = SamModel.from_pretrained(sam_model).to(device).eval()
        self.sam_processor = SamProcessor.from_pretrained(sam_model)
 
        print("Loading complete")
 
        self._pil        = None
        self._embeddings = None
 
    @torch.no_grad()
    def set_image(self, image_rgb: np.ndarray):
        self._pil        = Image.fromarray(image_rgb)
        inputs           = self.sam_processor(images=self._pil, return_tensors="pt")
        self._embeddings = self.sam_model.get_image_embeddings(
            inputs["pixel_values"].to(self.device)
        )
 
    @torch.no_grad()
    def predict_text(self, prompts: dict,
                     box_threshold:  float = 0.3,
                     text_threshold: float = 0.25) -> list:
        """
        Run Grounding DINO with all text prompts, then SAM on each detected box.
 
        Parameters
        ----------
        prompts         : {category_id: text_label}  e.g. {1: "car", 2: "pedestrian"}
        box_threshold   : Grounding DINO box confidence threshold
        text_threshold  : Grounding DINO text confidence threshold
 
        Returns
        -------
        list of (mask [H,W] bool, score float, category_id int)
            one entry per detected instance
        """
        # build joint text query:  "car . pedestrian ."
        # Grounding DINO expects period-separated labels
        label_to_cat = {v.lower(): k for k, v in prompts.items()}
        text_query   = " . ".join(prompts.values()) + " ."
 
        # --- Grounding DINO ---
        gdino_inputs = self.gdino_processor(
            images=self._pil,
            text=text_query,
            return_tensors="pt",
        ).to(self.device)
 
        with torch.no_grad():
            gdino_outputs = self.gdino_model(**gdino_inputs)
 
        results = self.gdino_processor.post_process_grounded_object_detection(
            gdino_outputs,
            gdino_inputs["input_ids"],
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[self._pil.size[::-1]],  # (H, W)
        )[0]
 
        boxes  = results["boxes"].cpu().numpy()   # (K, 4) xyxy
        labels = results["labels"]                # list of str
        scores = results["scores"].cpu().numpy()  # (K,)
 
        if len(boxes) == 0:
            return []
 
        # filter to known categories before passing to SAM
        cat_ids = [label_to_cat.get(l.lower()) for l in labels]
        keep    = [i for i, c in enumerate(cat_ids) if c is not None]
        if not keep:
            return []
 
        boxes   = boxes[keep]
        scores  = scores[keep]
        cat_ids = [cat_ids[i] for i in keep]
 
        # --- SAM one box at a time (image embeddings computed once above) ---
        detections = []
        for box, cat_id in zip(boxes, cat_ids):
            sam_inputs  = self.sam_processor(
                images=self._pil,
                input_boxes=[[box.tolist()]],
                return_tensors="pt",
            )
            sam_outputs = self.sam_model(
                input_boxes=sam_inputs["input_boxes"].to(self.device),
                image_embeddings=self._embeddings,
                multimask_output=False,
            )
            masks = self.sam_processor.image_processor.post_process_masks(
                sam_outputs.pred_masks.cpu(),
                sam_inputs["original_sizes"],
                sam_inputs["reshaped_input_sizes"],
            )
            mask  = np.array(masks[0].squeeze()).astype(bool)
            score = float(sam_outputs.iou_scores.cpu().numpy().flat[0])
            detections.append((mask, score, cat_id))
 
        return detections