from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mask_utils
from pathlib import Path
import numpy as np
import sys
import io

# ---------------------------------------------------------------------------
# Mask utilities
# ---------------------------------------------------------------------------
 
def encode_mask(mask: np.ndarray) -> dict:
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    return {"size": rle["size"], "counts": rle["counts"].decode("utf-8")}
 
 
def decode_ann_mask(ann: dict) -> np.ndarray:
    seg = ann["segmentation"]
    return mask_utils.decode(
        {"size": seg["size"], "counts": seg["counts"].encode("utf-8")}
    ).astype(bool)
    
    
# ---------------------------------------------------------------------------
# COCO evaluation + save
# ---------------------------------------------------------------------------
 
def run_coco_eval(coco_gt: COCO, predictions: list, output_dir: Path):
    if not predictions:
        print("  [warn] no predictions — skipping eval")
        return
 
    coco_dt   = coco_gt.loadRes(predictions)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="segm")
    coco_eval.evaluate()
    coco_eval.accumulate()
 
    buf = io.StringIO()
    sys.stdout, old = buf, sys.stdout
    coco_eval.summarize()
    sys.stdout = old
 
    summary = buf.getvalue()
    print(summary)
    (output_dir / "metrics.txt").write_text(summary)
    print(f"  Metrics saved to {output_dir / 'metrics.txt'}")