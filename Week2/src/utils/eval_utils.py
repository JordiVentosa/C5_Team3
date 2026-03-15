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
    
    
def evaluate_coco_by_class(coco_gt: COCO, coco_dt: COCO, iou_type: str = "segm", output_dir: Path = None):
    cat_id_to_name = {cat["id"]: cat["name"] for cat in coco_gt.dataset["categories"]}
    save_path = None
    if output_dir:
        save_path = output_dir / "metrics.txt"

    def _run_eval(img_ids=None, cat_ids=None):
        coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
        if img_ids is not None:
            coco_eval.params.imgIds = img_ids
        if cat_ids is not None:
            coco_eval.params.catIds = cat_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        return _extract_metrics(coco_eval)

    def _extract_metrics(coco_eval):
        s = coco_eval.stats
        return {
            "AP@50:95":  s[0],
            "AP@50":     s[1],
            "AP@75":     s[2],
            "AP_small":  s[3],
            "AP_medium": s[4],
            "AP_large":  s[5],
            "AR@1":      s[6],
            "AR@10":     s[7],
            "AR@100":    s[8],
            "AR_small":  s[9],
            "AR_medium": s[10],
            "AR_large":  s[11],
        }

    results = {}
    results["all"] = _run_eval()
    for cat_id, cat_name in cat_id_to_name.items():
        results[cat_name] = _run_eval(cat_ids=[cat_id])

    if output_dir is not None:
        metrics = ["AP@50:95", "AP@50", "AP@75", "AP_small", "AP_medium", "AP_large"]
        header  = f"{'Category':<20}" + "".join(f"{m:>12}" for m in metrics)
        sep     = "=" * len(header)

        lines   = [sep, header, sep]
        ordered = ["all"] + [k for k in results if k != "all"]
        for name in ordered:
            row = f"{name:<20}" + "".join(f"{results[name][m]:>12.3f}" for m in metrics)
            lines.append(row)
        lines.append(sep)

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            f.write("\n".join(lines) + "\n")

    return results