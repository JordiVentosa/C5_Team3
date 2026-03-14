import os
import re
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any

import cv2
import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    SamModel,
    SamProcessor,
)

# =========================
# CONFIG
# =========================
IMAGE_PATH = "/ghome/group03/mcv/datasets/C5/KITTI-MOTS/training/image_02/0000/000000.png"
GT_TXT_PATH = "/ghome/group03/mcv/datasets/C5/KITTI-MOTS/instances_txt/0000.txt"

GROUNDING_DINO_MODEL_ID = "IDEA-Research/grounding-dino-base"
SAM_MODEL_ID = "facebook/sam-vit-huge"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FRAME_ID = None

# Variantes de text prompt para comparar, parecido a task a
# Grounding DINO suele funcionar mejor con minúsculas y con punto final.
TEXT_PROMPT_VARIANTS = {
    "car_person": {
        1: "car.",
        2: "person.",
    },
    "car_pedestrian": {
        1: "car.",
        2: "pedestrian.",
    },
}

# Thresholds de Grounding DINO
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25

# Threshold de matching predicción-GT para evaluación
MATCH_IOU_THRESHOLD = 0.5

# SAM
MULTIMASK_OUTPUT = True

# Salida
OUTPUT_DIR = "./grounded_sam_task_b_0000_000000"

# Reproducibilidad
RANDOM_SEED = 42

# Guardar overlays por predicción
SAVE_PER_PREDICTION = True

# Ignore overlay
DRAW_IGNORE_OVERLAY = True
DRAW_IGNORE_LABEL = True
IGNORE_ALPHA = 0.35
IGNORE_COLOR_BGR = (120, 120, 120)
IGNORE_TEXT = "ignore class=10"

# Clases objetivo de KITTI-MOTS
TARGET_CLASS_IDS = {
    1: "car",
    2: "person",  # GT es pedestrian, aquí lo guardamos como person en salidas
}


# =========================
# Utils generales
# =========================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_image(path: str, image_bgr: np.ndarray) -> None:
    ok = cv2.imwrite(path, image_bgr)
    if not ok:
        raise IOError(f"No se pudo guardar la imagen: {path}")


def infer_frame_id_from_image_name(image_path: str) -> int:
    stem = Path(image_path).stem
    numbers = re.findall(r"\d+", stem)
    if not numbers:
        raise ValueError(
            f"No pude inferir el frame_id desde el nombre de la imagen: {image_path}. "
            "Pon FRAME_ID manualmente en CONFIG."
        )
    return int(numbers[-1])


def color_from_id(seed_id: int) -> Tuple[int, int, int]:
    rng = np.random.default_rng(seed_id)
    color = rng.integers(60, 255, size=3).tolist()
    return int(color[0]), int(color[1]), int(color[2])


def move_to_device(batch: Dict[str, Any], device: str) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if hasattr(v, "to"):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


# =========================
# Parse GT KITTI-MOTS TXT
# =========================
def decode_kitti_mots_rle(height: int, width: int, counts_str: str) -> np.ndarray:
    rle = {"size": [height, width], "counts": counts_str.encode("utf-8")}
    mask = mask_utils.decode(rle)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return mask.astype(bool)


def parse_gt_txt_for_frame(
    gt_txt_path: str,
    frame_id: int,
    image_shape: Tuple[int, int],
) -> Tuple[List[Dict], np.ndarray]:
    image_h, image_w = image_shape[:2]
    instances: List[Dict] = []
    ignore_mask = np.zeros((image_h, image_w), dtype=bool)

    with open(gt_txt_path, "r", encoding="utf-8") as f:
        for line_idx, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 6:
                raise ValueError(
                    f"Línea mal formada en GT TXT (línea {line_idx}): {raw_line}"
                )

            time_frame = int(parts[0])
            obj_id = int(parts[1])
            class_id = int(parts[2])
            h = int(parts[3])
            w = int(parts[4])
            counts_str = parts[5]

            if time_frame != frame_id:
                continue

            if (h, w) != (image_h, image_w):
                raise ValueError(
                    f"Dimensiones del GT y de la imagen no coinciden en la línea {line_idx}. "
                    f"GT=({h},{w}) vs imagen=({image_h},{image_w})"
                )

            mask = decode_kitti_mots_rle(h, w, counts_str)

            if obj_id == 10000 or class_id == 10:
                ignore_mask |= mask
                continue

            if class_id not in TARGET_CLASS_IDS:
                continue

            area = int(mask.sum())
            if area == 0:
                continue

            instances.append(
                {
                    "frame_id": time_frame,
                    "obj_id": obj_id,
                    "class_id": class_id,
                    "class_name": TARGET_CLASS_IDS[class_id],
                    "mask": mask,
                    "area": area,
                }
            )

    return instances, ignore_mask


# =========================
# Métricas binarias
# =========================
def compute_binary_metrics(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    ignore_mask: np.ndarray,
) -> Dict[str, float]:
    valid = ~ignore_mask
    pred = pred_mask & valid
    gt = gt_mask & valid

    intersection = int(np.logical_and(pred, gt).sum())
    union = int(np.logical_or(pred, gt).sum())
    pred_area = int(pred.sum())
    gt_area = int(gt.sum())

    iou = intersection / union if union > 0 else 0.0
    dice = (2.0 * intersection) / (pred_area + gt_area) if (pred_area + gt_area) > 0 else 0.0
    precision = intersection / pred_area if pred_area > 0 else 0.0
    recall = intersection / gt_area if gt_area > 0 else 0.0

    return {
        "intersection": intersection,
        "union": union,
        "pred_area": pred_area,
        "gt_area": gt_area,
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
    }


def safe_div(num: float, den: float) -> float:
    return num / den if den != 0 else 0.0


# =========================
# Matching predicciones vs GT
# =========================
def greedy_match_predictions_to_gt(
    predictions: List[Dict],
    gt_instances: List[Dict],
    ignore_mask: np.ndarray,
    class_id: int,
    match_iou_threshold: float,
) -> Dict:
    pred_idxs = [i for i, p in enumerate(predictions) if p["class_id"] == class_id]
    gt_idxs = [i for i, g in enumerate(gt_instances) if g["class_id"] == class_id]

    candidate_pairs = []

    for pi in pred_idxs:
        for gi in gt_idxs:
            metrics = compute_binary_metrics(
                predictions[pi]["mask"],
                gt_instances[gi]["mask"],
                ignore_mask,
            )
            candidate_pairs.append(
                {
                    "pred_idx": pi,
                    "gt_idx": gi,
                    "iou": metrics["iou"],
                    "dice": metrics["dice"],
                    "metrics": metrics,
                }
            )

    candidate_pairs.sort(key=lambda x: x["iou"], reverse=True)

    used_preds = set()
    used_gts = set()
    matches = []

    for pair in candidate_pairs:
        if pair["iou"] < match_iou_threshold:
            break
        if pair["pred_idx"] in used_preds or pair["gt_idx"] in used_gts:
            continue
        used_preds.add(pair["pred_idx"])
        used_gts.add(pair["gt_idx"])
        matches.append(pair)

    unmatched_pred_idxs = [pi for pi in pred_idxs if pi not in used_preds]
    unmatched_gt_idxs = [gi for gi in gt_idxs if gi not in used_gts]

    tp = len(matches)
    fp = len(unmatched_pred_idxs)
    fn = len(unmatched_gt_idxs)

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)

    mean_iou = float(np.mean([m["iou"] for m in matches])) if matches else 0.0
    mean_dice = float(np.mean([m["dice"] for m in matches])) if matches else 0.0

    return {
        "class_id": class_id,
        "class_name": TARGET_CLASS_IDS[class_id],
        "n_pred": len(pred_idxs),
        "n_gt": len(gt_idxs),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_matched_iou": mean_iou,
        "mean_matched_dice": mean_dice,
        "matches": matches,
        "unmatched_pred_idxs": unmatched_pred_idxs,
        "unmatched_gt_idxs": unmatched_gt_idxs,
    }


def evaluate_predictions_against_gt(
    predictions: List[Dict],
    gt_instances: List[Dict],
    ignore_mask: np.ndarray,
    match_iou_threshold: float,
) -> Dict:
    by_class = {}
    all_matches = []
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_pred = 0
    total_gt = 0

    for class_id in sorted(TARGET_CLASS_IDS.keys()):
        cls_eval = greedy_match_predictions_to_gt(
            predictions=predictions,
            gt_instances=gt_instances,
            ignore_mask=ignore_mask,
            class_id=class_id,
            match_iou_threshold=match_iou_threshold,
        )
        by_class[TARGET_CLASS_IDS[class_id]] = cls_eval
        all_matches.extend(cls_eval["matches"])
        total_tp += cls_eval["tp"]
        total_fp += cls_eval["fp"]
        total_fn += cls_eval["fn"]
        total_pred += cls_eval["n_pred"]
        total_gt += cls_eval["n_gt"]

    overall_precision = safe_div(total_tp, total_tp + total_fp)
    overall_recall = safe_div(total_tp, total_tp + total_fn)
    overall_f1 = safe_div(2 * overall_precision * overall_recall, overall_precision + overall_recall)
    overall_mean_iou = float(np.mean([m["iou"] for m in all_matches])) if all_matches else 0.0
    overall_mean_dice = float(np.mean([m["dice"] for m in all_matches])) if all_matches else 0.0

    return {
        "match_iou_threshold": match_iou_threshold,
        "overall": {
            "n_pred": total_pred,
            "n_gt": total_gt,
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "precision": overall_precision,
            "recall": overall_recall,
            "f1": overall_f1,
            "mean_matched_iou": overall_mean_iou,
            "mean_matched_dice": overall_mean_dice,
        },
        "by_class": by_class,
    }


# =========================
# Grounding DINO predictor
# =========================
class GroundingDINOPredictor:
    def __init__(self, model_id: str, device: str):
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, image_pil: Image.Image, text_prompt: str) -> List[Dict]:
        prompt = text_prompt.strip().lower()
        if not prompt.endswith("."):
            prompt = prompt + "."

        inputs = self.processor(images=image_pil, text=prompt, return_tensors="pt")
        inputs = move_to_device(inputs, self.device)

        outputs = self.model(**inputs)

        # En algunas versiones de transformers la firma usa box_threshold,
        # y en otras usa threshold.
        try:
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                box_threshold=BOX_THRESHOLD,
                text_threshold=TEXT_THRESHOLD,
                target_sizes=[image_pil.size[::-1]],
            )
        except TypeError:
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                threshold=BOX_THRESHOLD,
                text_threshold=TEXT_THRESHOLD,
                target_sizes=[image_pil.size[::-1]],
            )

        result = results[0]

        boxes = result.get("boxes", [])
        scores = result.get("scores", [])
        labels = result.get("labels", [])

        detections = []
        for idx, (box, score, label) in enumerate(zip(boxes, scores, labels)):
            if torch.is_tensor(box):
                box = box.detach().cpu().numpy()
            else:
                box = np.asarray(box)

            if torch.is_tensor(score):
                score = float(score.detach().cpu().item())
            else:
                score = float(score)

            if isinstance(label, (list, tuple)):
                raw_label = " ".join([str(x) for x in label])
            else:
                raw_label = str(label)

            x1, y1, x2, y2 = box.tolist()
            detections.append(
                {
                    "det_id": idx,
                    "box": np.array([x1, y1, x2, y2], dtype=np.float32),
                    "score": score,
                    "label": raw_label,
                    "prompt_text": prompt,
                }
            )

        return detections


# =========================
# SAM box predictor
# =========================
def normalize_postprocessed_masks(masks) -> np.ndarray:
    if isinstance(masks, list):
        masks = masks[0]
    if torch.is_tensor(masks):
        masks = masks.detach().cpu().numpy()
    masks = np.asarray(masks)

    while masks.ndim > 3:
        masks = masks[0]

    if masks.ndim == 2:
        masks = masks[None, ...]

    return masks


def normalize_scores(scores) -> np.ndarray:
    if torch.is_tensor(scores):
        scores = scores.detach().cpu().numpy()
    scores = np.asarray(scores)

    while scores.ndim > 1:
        scores = scores[0]

    return scores.astype(np.float32)


class HFSamBoxPredictor:
    def __init__(self, model_id: str, device: str):
        self.device = device
        self.model = SamModel.from_pretrained(model_id).to(device)
        self.model.eval()
        self.processor = SamProcessor.from_pretrained(model_id)
        self.image_pil = None
        self.image_embeddings = None

    @torch.no_grad()
    def set_image(self, image_rgb: np.ndarray) -> None:
        self.image_pil = Image.fromarray(image_rgb)
        image_inputs = self.processor(images=self.image_pil, return_tensors="pt")
        pixel_values = image_inputs["pixel_values"].to(self.device)
        self.image_embeddings = self.model.get_image_embeddings(pixel_values)

    @torch.no_grad()
    def predict_from_box(self, box_xyxy: np.ndarray, multimask_output: bool = True) -> Dict:
        if self.image_pil is None or self.image_embeddings is None:
            raise RuntimeError("Debes llamar antes a set_image(...).")

        input_boxes = [[[float(box_xyxy[0]), float(box_xyxy[1]), float(box_xyxy[2]), float(box_xyxy[3])]]]

        inputs = self.processor(
            images=self.image_pil,
            input_boxes=input_boxes,
            return_tensors="pt",
        )

        original_sizes = inputs["original_sizes"]
        reshaped_input_sizes = inputs["reshaped_input_sizes"]
        inputs = move_to_device(inputs, self.device)

        outputs = self.model(
            input_boxes=inputs["input_boxes"],
            image_embeddings=self.image_embeddings,
            multimask_output=multimask_output,
        )

        post_masks = self.processor.image_processor.post_process_masks(
            outputs.pred_masks.detach().cpu(),
            original_sizes,
            reshaped_input_sizes,
        )

        masks_np = normalize_postprocessed_masks(post_masks)
        scores_np = normalize_scores(outputs.iou_scores)

        if masks_np.shape[0] != scores_np.shape[0]:
            n = min(masks_np.shape[0], scores_np.shape[0])
            masks_np = masks_np[:n]
            scores_np = scores_np[:n]

        best_idx = int(np.argmax(scores_np))
        best_mask = masks_np[best_idx].astype(bool)

        return {
            "best_mask": best_mask,
            "all_masks": masks_np.astype(bool),
            "scores": scores_np,
            "best_idx": best_idx,
            "best_score": float(scores_np[best_idx]),
        }


# =========================
# Overlays
# =========================
def draw_ignore_regions(
    canvas: np.ndarray,
    ignore_mask: np.ndarray,
    alpha: float = 0.35,
    color_bgr: Tuple[int, int, int] = (120, 120, 120),
    draw_label: bool = True,
    label_text: str = "ignore class=10",
) -> np.ndarray:
    if ignore_mask is None or not ignore_mask.any():
        return canvas

    out = canvas.copy()

    colored = np.zeros_like(out, dtype=np.uint8)
    colored[:, :] = color_bgr
    mask3 = np.repeat(ignore_mask[:, :, None], 3, axis=2)
    blended = cv2.addWeighted(out, 1 - alpha, colored, alpha, 0)
    out = np.where(mask3, blended, out)

    ignore_u8 = ignore_mask.astype(np.uint8)
    contours, _ = cv2.findContours(ignore_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color_bgr, 2)

    if draw_label:
        num_labels, labels_img, stats, _ = cv2.connectedComponentsWithStats(ignore_u8, connectivity=8)
        for comp_id in range(1, num_labels):
            x = int(stats[comp_id, cv2.CC_STAT_LEFT])
            y = int(stats[comp_id, cv2.CC_STAT_TOP])
            area = int(stats[comp_id, cv2.CC_STAT_AREA])

            if area <= 0:
                continue

            text_x = x
            text_y = max(20, y - 6)

            cv2.putText(
                out,
                label_text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

    return out


def overlay_gt_instances(
    image_bgr: np.ndarray,
    gt_instances: List[Dict],
    title: str = "",
    ignore_mask: np.ndarray = None,
) -> np.ndarray:
    canvas = image_bgr.copy()

    if DRAW_IGNORE_OVERLAY and ignore_mask is not None and ignore_mask.any():
        canvas = draw_ignore_regions(
            canvas=canvas,
            ignore_mask=ignore_mask,
            alpha=IGNORE_ALPHA,
            color_bgr=IGNORE_COLOR_BGR,
            draw_label=DRAW_IGNORE_LABEL,
            label_text=IGNORE_TEXT,
        )

    for item in gt_instances:
        mask = item["mask"].astype(bool)
        color = color_from_id(item["obj_id"])

        colored = np.zeros_like(canvas, dtype=np.uint8)
        colored[:, :] = color
        mask3 = np.repeat(mask[:, :, None], 3, axis=2)
        canvas = np.where(mask3, cv2.addWeighted(canvas, 0.55, colored, 0.45, 0), canvas)

        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, color, 2)

        ys, xs = np.where(mask)
        if len(xs) > 0:
            x_text = int(xs.min())
            y_text = int(max(20, ys.min() - 5))
            label = f"{item['class_name']} id={item['obj_id']}"
            cv2.putText(
                canvas,
                label,
                (x_text, y_text),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

    if title:
        cv2.putText(
            canvas,
            title,
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return canvas


def overlay_grounding_boxes(
    image_bgr: np.ndarray,
    predictions: List[Dict],
    title: str = "",
    ignore_mask: np.ndarray = None,
) -> np.ndarray:
    canvas = image_bgr.copy()

    if DRAW_IGNORE_OVERLAY and ignore_mask is not None and ignore_mask.any():
        canvas = draw_ignore_regions(
            canvas=canvas,
            ignore_mask=ignore_mask,
            alpha=IGNORE_ALPHA,
            color_bgr=IGNORE_COLOR_BGR,
            draw_label=DRAW_IGNORE_LABEL,
            label_text=IGNORE_TEXT,
        )

    for pred in predictions:
        x1, y1, x2, y2 = [int(round(v)) for v in pred["box"]]
        color = (0, 255, 255) if pred["class_id"] == 1 else (255, 255, 0)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

        label = (
            f"{pred['class_name']} "
            f"box={pred['box_score']:.3f} "
            f"sam={pred['sam_score']:.3f}"
        )
        cv2.putText(
            canvas,
            label,
            (x1, max(20, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    if title:
        cv2.putText(
            canvas,
            title,
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return canvas


def overlay_pred_masks(
    image_bgr: np.ndarray,
    predictions: List[Dict],
    title: str = "",
    ignore_mask: np.ndarray = None,
) -> np.ndarray:
    canvas = image_bgr.copy()

    if DRAW_IGNORE_OVERLAY and ignore_mask is not None and ignore_mask.any():
        canvas = draw_ignore_regions(
            canvas=canvas,
            ignore_mask=ignore_mask,
            alpha=IGNORE_ALPHA,
            color_bgr=IGNORE_COLOR_BGR,
            draw_label=DRAW_IGNORE_LABEL,
            label_text=IGNORE_TEXT,
        )

    for idx, pred in enumerate(predictions):
        mask = pred["mask"].astype(bool)
        color = color_from_id(100000 + idx)

        colored = np.zeros_like(canvas, dtype=np.uint8)
        colored[:, :] = color
        mask3 = np.repeat(mask[:, :, None], 3, axis=2)
        canvas = np.where(mask3, cv2.addWeighted(canvas, 0.55, colored, 0.45, 0), canvas)

        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, color, 2)

        ys, xs = np.where(mask)
        if len(xs) > 0:
            x_text = int(xs.min())
            y_text = int(max(20, ys.min() - 5))
            label = f"{pred['class_name']} det={pred['det_id']} b={pred['box_score']:.2f}"
            cv2.putText(
                canvas,
                label,
                (x_text, y_text),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )

    if title:
        cv2.putText(
            canvas,
            title,
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return canvas


# =========================
# Escritura de métricas
# =========================
def write_metrics_txt(
    output_txt_path: str,
    image_path: str,
    gt_txt_path: str,
    frame_id: int,
    gt_instances: List[Dict],
    ignore_mask: np.ndarray,
    all_variant_predictions: Dict[str, List[Dict]],
    all_variant_evals: Dict[str, Dict],
) -> None:
    lines = []
    lines.append("============================================================")
    lines.append("TASK B | Grounded SAM on KITTI-MOTS | Metrics")
    lines.append("============================================================")
    lines.append(f"IMAGE_PATH: {image_path}")
    lines.append(f"GT_TXT_PATH: {gt_txt_path}")
    lines.append(f"FRAME_ID: {frame_id}")
    lines.append(f"GROUNDING_DINO_MODEL_ID: {GROUNDING_DINO_MODEL_ID}")
    lines.append(f"SAM_MODEL_ID: {SAM_MODEL_ID}")
    lines.append(f"DEVICE: {DEVICE}")
    lines.append(f"BOX_THRESHOLD: {BOX_THRESHOLD}")
    lines.append(f"TEXT_THRESHOLD: {TEXT_THRESHOLD}")
    lines.append(f"MATCH_IOU_THRESHOLD: {MATCH_IOU_THRESHOLD}")
    lines.append(f"MULTIMASK_OUTPUT: {MULTIMASK_OUTPUT}")
    lines.append(f"N_GT_INSTANCES: {len(gt_instances)}")
    lines.append(f"IGNORE_PIXELS: {int(ignore_mask.sum())}")
    lines.append("")

    for variant_name in all_variant_predictions.keys():
        preds = all_variant_predictions[variant_name]
        ev = all_variant_evals[variant_name]
        ov = ev["overall"]

        lines.append("------------------------------------------------------------")
        lines.append(f"TEXT_PROMPT_VARIANT: {variant_name}")
        lines.append("------------------------------------------------------------")
        lines.append("Prompt mapping:")
        for cid, txt in TEXT_PROMPT_VARIANTS[variant_name].items():
            lines.append(f"  class_id={cid} ({TARGET_CLASS_IDS[cid]}) -> '{txt}'")
        lines.append("")

        lines.append("OVERALL")
        lines.append(f"  n_pred            = {ov['n_pred']}")
        lines.append(f"  n_gt              = {ov['n_gt']}")
        lines.append(f"  tp                = {ov['tp']}")
        lines.append(f"  fp                = {ov['fp']}")
        lines.append(f"  fn                = {ov['fn']}")
        lines.append(f"  precision         = {ov['precision']:.6f}")
        lines.append(f"  recall            = {ov['recall']:.6f}")
        lines.append(f"  f1                = {ov['f1']:.6f}")
        lines.append(f"  mean_matched_iou  = {ov['mean_matched_iou']:.6f}")
        lines.append(f"  mean_matched_dice = {ov['mean_matched_dice']:.6f}")
        lines.append("")

        lines.append("BY CLASS")
        for class_name, cls in ev["by_class"].items():
            lines.append(f"  CLASS={class_name}")
            lines.append(f"    n_pred            = {cls['n_pred']}")
            lines.append(f"    n_gt              = {cls['n_gt']}")
            lines.append(f"    tp                = {cls['tp']}")
            lines.append(f"    fp                = {cls['fp']}")
            lines.append(f"    fn                = {cls['fn']}")
            lines.append(f"    precision         = {cls['precision']:.6f}")
            lines.append(f"    recall            = {cls['recall']:.6f}")
            lines.append(f"    f1                = {cls['f1']:.6f}")
            lines.append(f"    mean_matched_iou  = {cls['mean_matched_iou']:.6f}")
            lines.append(f"    mean_matched_dice = {cls['mean_matched_dice']:.6f}")
            lines.append("")

        lines.append("PREDICTIONS DETAIL")
        lines.append(
            "det_id | class_name | raw_label | prompt_text | box_score | sam_score | box_xyxy"
        )
        for pred in preds:
            box_str = [round(float(x), 2) for x in pred["box"].tolist()]
            lines.append(
                f"{pred['det_id']} | {pred['class_name']} | {pred['raw_label']} | "
                f"{pred['prompt_text']} | {pred['box_score']:.6f} | {pred['sam_score']:.6f} | {box_str}"
            )
        lines.append("")

        lines.append("MATCHES DETAIL")
        lines.append(
            "class_name | pred_det_id | gt_obj_id | iou | dice | pred_area | gt_area"
        )
        for class_name, cls in ev["by_class"].items():
            for m in cls["matches"]:
                pred = preds[m["pred_idx"]]
                gt_obj = gt_instances[m["gt_idx"]]
                mm = m["metrics"]
                lines.append(
                    f"{class_name} | {pred['det_id']} | {gt_obj['obj_id']} | "
                    f"{m['iou']:.6f} | {m['dice']:.6f} | {mm['pred_area']} | {mm['gt_area']}"
                )
        lines.append("")

    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# =========================
# Main
# =========================
def main():
    set_seed(RANDOM_SEED)

    ensure_dir(OUTPUT_DIR)
    overlays_dir = os.path.join(OUTPUT_DIR, "overlays")
    per_pred_dir = os.path.join(OUTPUT_DIR, "per_prediction")
    ensure_dir(overlays_dir)
    if SAVE_PER_PREDICTION:
        ensure_dir(per_pred_dir)

    image_bgr = cv2.imread(IMAGE_PATH, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {IMAGE_PATH}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_h, image_w = image_bgr.shape[:2]
    image_pil = Image.fromarray(image_rgb)

    frame_id = FRAME_ID if FRAME_ID is not None else infer_frame_id_from_image_name(IMAGE_PATH)

    gt_instances, ignore_mask = parse_gt_txt_for_frame(
        GT_TXT_PATH,
        frame_id=frame_id,
        image_shape=(image_h, image_w),
    )

    if len(gt_instances) == 0:
        raise RuntimeError(
            f"No encontré instancias car/person para frame_id={frame_id} en el GT TXT."
        )

    # Guardar GT overlay
    gt_overlay = overlay_gt_instances(
        image_bgr=image_bgr,
        gt_instances=gt_instances,
        title=f"GT | frame={frame_id}",
        ignore_mask=ignore_mask,
    )
    save_image(os.path.join(overlays_dir, "gt_overlay.png"), gt_overlay)

    grounding_predictor = GroundingDINOPredictor(GROUNDING_DINO_MODEL_ID, DEVICE)
    sam_predictor = HFSamBoxPredictor(SAM_MODEL_ID, DEVICE)
    sam_predictor.set_image(image_rgb)

    all_variant_predictions: Dict[str, List[Dict]] = {}
    all_variant_evals: Dict[str, Dict] = {}

    for variant_name, class_prompt_map in TEXT_PROMPT_VARIANTS.items():
        variant_predictions: List[Dict] = []
        det_counter = 0

        for class_id, prompt_text in class_prompt_map.items():
            detections = grounding_predictor.predict(image_pil=image_pil, text_prompt=prompt_text)

            for det in detections:
                sam_out = sam_predictor.predict_from_box(
                    box_xyxy=det["box"],
                    multimask_output=MULTIMASK_OUTPUT,
                )

                variant_predictions.append(
                    {
                        "det_id": det_counter,
                        "class_id": class_id,
                        "class_name": TARGET_CLASS_IDS[class_id],
                        "prompt_text": prompt_text,
                        "raw_label": det["label"],
                        "box": det["box"].copy(),
                        "box_score": float(det["score"]),
                        "mask": sam_out["best_mask"],
                        "sam_score": float(sam_out["best_score"]),
                        "sam_best_idx": int(sam_out["best_idx"]),
                    }
                )
                det_counter += 1

        variant_eval = evaluate_predictions_against_gt(
            predictions=variant_predictions,
            gt_instances=gt_instances,
            ignore_mask=ignore_mask,
            match_iou_threshold=MATCH_IOU_THRESHOLD,
        )

        all_variant_predictions[variant_name] = variant_predictions
        all_variant_evals[variant_name] = variant_eval

        # Overlays globales
        boxes_overlay = overlay_grounding_boxes(
            image_bgr=image_bgr,
            predictions=variant_predictions,
            title=f"{variant_name} | grounding boxes",
            ignore_mask=ignore_mask,
        )
        save_image(os.path.join(overlays_dir, f"{variant_name}_grounding_boxes.png"), boxes_overlay)

        pred_overlay = overlay_pred_masks(
            image_bgr=image_bgr,
            predictions=variant_predictions,
            title=f"{variant_name} | grounded sam masks",
            ignore_mask=ignore_mask,
        )
        save_image(os.path.join(overlays_dir, f"{variant_name}_grounded_sam_masks.png"), pred_overlay)

        # Guardar JSON resumen simple por variante
        summary_json_path = os.path.join(OUTPUT_DIR, f"{variant_name}_summary.json")
        with open(summary_json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "variant_name": variant_name,
                    "prompt_mapping": class_prompt_map,
                    "eval": {
                        "overall": all_variant_evals[variant_name]["overall"],
                        "by_class": {
                            k: {
                                kk: vv
                                for kk, vv in v.items()
                                if kk not in ["matches", "unmatched_pred_idxs", "unmatched_gt_idxs"]
                            }
                            for k, v in all_variant_evals[variant_name]["by_class"].items()
                        },
                    },
                },
                f,
                indent=2,
            )

        # Guardado por predicción
        if SAVE_PER_PREDICTION:
            variant_pred_dir = os.path.join(per_pred_dir, variant_name)
            ensure_dir(variant_pred_dir)

            for pred in variant_predictions:
                subdir = os.path.join(
                    variant_pred_dir,
                    f"det_{pred['det_id']:03d}_{pred['class_name']}"
                )
                ensure_dir(subdir)

                single_box_overlay = overlay_grounding_boxes(
                    image_bgr=image_bgr,
                    predictions=[pred],
                    title=f"{variant_name} | box | det={pred['det_id']}",
                    ignore_mask=ignore_mask,
                )
                save_image(os.path.join(subdir, "grounding_box.png"), single_box_overlay)

                single_mask_overlay = overlay_pred_masks(
                    image_bgr=image_bgr,
                    predictions=[pred],
                    title=f"{variant_name} | mask | det={pred['det_id']}",
                    ignore_mask=ignore_mask,
                )
                save_image(os.path.join(subdir, "grounded_sam_mask.png"), single_mask_overlay)

    # metrics.txt global
    metrics_txt_path = os.path.join(OUTPUT_DIR, "metrics.txt")
    write_metrics_txt(
        output_txt_path=metrics_txt_path,
        image_path=IMAGE_PATH,
        gt_txt_path=GT_TXT_PATH,
        frame_id=frame_id,
        gt_instances=gt_instances,
        ignore_mask=ignore_mask,
        all_variant_predictions=all_variant_predictions,
        all_variant_evals=all_variant_evals,
    )

    print("Proceso completado.")
    print(f"Resultados guardados en: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
