import argparse
import numpy as np
from pathlib import Path
from PIL import Image
 
from pycocotools.coco import COCO
import yaml

import torch
from models import SAMPredictor, GroundedSAMPredictor
from datasets import build_coco_gt
from runners import *
 
 
 
# predictor registry — maps model.type -> class
PREDICTOR_REGISTRY = {
    "sam":          SAMPredictor,
    "grounded_sam": GroundedSAMPredictor,
}
 
def build_predictor(model_cfg: dict):
    model_type = model_cfg.get("type", "sam")
    if model_type not in PREDICTOR_REGISTRY:
        raise ValueError(f"Unknown model type '{model_type}'. "
                         f"Available: {list(PREDICTOR_REGISTRY)}")
 
    cls = PREDICTOR_REGISTRY[model_type]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Loading model (type={model_type}) on {device} ...")
 
    if model_type == "sam":
        return cls(model_cfg["name"], device)
    elif model_type == "grounded_sam":
        return cls(model_cfg["grounding_model"], model_cfg["sam_model"], device)
 
EVAL_RUNNERS = {
    "point_prompt":        run_point_prompt,
    "bbox_grounded":       run_text_prompt,
    "bbox_detection" :     run_bbox_best
    # add new runners here, e.g.:
    # "text_prompt":      run_text_prompt,
    # "bbox_detection":   run_bbox_detection,
}

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
 
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
 
    # 1. GT
    coco_dict = build_coco_gt(cfg["dataset"]["root"], cfg["dataset"]["split"])
    print("GT built")
 
    coco_gt = COCO()
    coco_gt.dataset = coco_dict
    coco_gt.createIndex()
 
    # 2. Model — chosen entirely from config
    model_cfg = cfg["model"]
    model = build_predictor(model_cfg)
    print("Model built")
 
 
    # 3. Single evaluation
    eval_cfg   = cfg["evaluation"]
    eval_type  = eval_cfg["type"]
    output_dir = Path(eval_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nRunning evaluation: {eval_type}, output saved at {output_dir}")
 
    if eval_type not in EVAL_RUNNERS:
        raise ValueError(f"Unknown evaluation type '{eval_type}'. "
                         f"Available: {list(EVAL_RUNNERS)}")
 
    # pass any extra config keys as kwargs to the runner
    extra = {k: v for k, v in eval_cfg.items() if k not in ("type", "output_dir")}
    EVAL_RUNNERS[eval_type](coco_gt, coco_dict, model, output_dir, **extra)
    print("\nDone.")
 
 
if __name__ == "__main__":
    main()