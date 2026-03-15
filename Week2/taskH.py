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
IMG_NAME = "000150"
IMAGE_PATH = f"/ghome/group03/mcv/datasets/C5/KITTI-MOTS/training/image_02/0000/{IMG_NAME}.png"
GT_TXT_PATH = "/ghome/group03/mcv/datasets/C5/KITTI-MOTS/instances_txt/0000.txt"

# Grounded SAM para text prompting
GROUNDING_DINO_MODEL_ID = "IDEA-Research/grounding-dino-base"
PRETRAINED_SAM_MODEL_NAME_OR_PATH = "facebook/sam-vit-huge"

# OPCIONAL:
# Si guardaste el fine-tuned con model.save_pretrained(...), pon aquí la carpeta.
FINETUNED_SAM_MODEL_NAME_OR_PATH = None
# Si guardaste solo un .pth/.pt con state_dict, pon aquí la ruta.
FINETUNED_SAM_STATE_DICT_PATH = None

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Si lo dejas a None, se infiere desde el nombre del archivo.
FRAME_ID = None

# Variantes de text prompt para semantic segmentation
TEXT_PROMPT_VARIANTS = {
    "car_pedestrian": {
        1: "car.",
        2: "pedestrian.",
    },
    "car_person": {
        1: "car.",
        2: "person.",
    },
}

# Grounding DINO thresholds
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25

# SAM
MULTIMASK_OUTPUT = True
MIN_MASK_AREA = 25  # filtra máscaras diminutas

# Directorio de salida
OUTPUT_DIR = f"./outputs_taskH/semantic_kitti_mots_0000_{IMG_NAME}"

# Reproducibilidad
RANDOM_SEED = 42

# Guardar también detalles por detección
SAVE_PER_DETECTION = True

# Config visual para ignore
DRAW_IGNORE_OVERLAY = True
DRAW_IGNORE_LABEL = True
IGNORE_ALPHA = 0.35
IGNORE_COLOR_BGR = (120, 120, 120)
IGNORE_TEXT = "ignore class=10"

# Clases objetivo
TARGET_CLASS_IDS = {
    1: "car",
    2: "pedestrian",
}

# Colores semánticos BGR
SEMANTIC_COLORS = {
    0: (0, 0, 0),          # background
    1: (0, 255, 255),      # car
    2: (255, 255, 0),      # pedestrian
    255: (120, 120, 120),  # ignore
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def infer_frame_id_from_image_name(image_path: str) -> int:
    stem = Path(image_path).stem
    numbers = re.findall(r"\d+", stem)
    if not numbers:
        raise ValueError(
            f"No pude inferir el frame_id desde el nombre de la imagen: {image_path}. "
            "Pon FRAME_ID manualmente en la CONFIG."
        )
    return int(numbers[-1])


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
                raise ValueError(f"Línea mal formada en GT TXT (línea {line_idx}): {raw_line}")

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


def build_semantic_gt(
    instances: List[Dict],
    image_shape: Tuple[int, int],
) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
    image_h, image_w = image_shape[:2]
    semantic_map = np.zeros((image_h, image_w), dtype=np.uint8)
    class_masks: Dict[int, np.ndarray] = {}

    for class_id in sorted(TARGET_CLASS_IDS.keys()):
        class_mask = np.zeros((image_h, image_w), dtype=bool)
        for inst in instances:
            if inst["class_id"] == class_id:
                class_mask |= inst["mask"]
        class_masks[class_id] = class_mask
        semantic_map[class_mask] = class_id

    return semantic_map, class_masks


def move_to_device(batch: Dict[str, Any], device: str) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if hasattr(v, "to"):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


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


def safe_div(num: float, den: float) -> float:
    return num / den if den != 0 else 0.0


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


def compute_semantic_metrics(
    pred_semantic_map: np.ndarray,
    gt_semantic_map: np.ndarray,
    ignore_mask: np.ndarray,
) -> Dict[str, Any]:
    valid = ~ignore_mask

    overall_pixel_acc = float((pred_semantic_map[valid] == gt_semantic_map[valid]).mean()) if valid.any() else 0.0

    fg_valid = valid & (gt_semantic_map > 0)
    fg_pixel_acc = float((pred_semantic_map[fg_valid] == gt_semantic_map[fg_valid]).mean()) if fg_valid.any() else 0.0

    by_class = {}
    class_ious = []
    class_dices = []

    for class_id, class_name in TARGET_CLASS_IDS.items():
        pred_mask = pred_semantic_map == class_id
        gt_mask = gt_semantic_map == class_id

        metrics = compute_binary_metrics(pred_mask, gt_mask, ignore_mask)
        by_class[class_name] = metrics
        class_ious.append(metrics["iou"])
        class_dices.append(metrics["dice"])

    mean_iou = float(np.mean(class_ious)) if class_ious else 0.0
    mean_dice = float(np.mean(class_dices)) if class_dices else 0.0

    return {
        "overall": {
            "overall_pixel_accuracy": overall_pixel_acc,
            "foreground_pixel_accuracy": fg_pixel_acc,
            "mean_iou": mean_iou,
            "mean_dice": mean_dice,
        },
        "by_class": by_class,
    }


def color_from_id(seed_id: int) -> Tuple[int, int, int]:
    rng = np.random.default_rng(seed_id)
    color = rng.integers(60, 255, size=3).tolist()
    return int(color[0]), int(color[1]), int(color[2])


def save_image(path: str, image_bgr: np.ndarray) -> None:
    ok = cv2.imwrite(path, image_bgr)
    if not ok:
        raise IOError(f"No se pudo guardar la imagen: {path}")


def save_semantic_label_png(
    path: str,
    semantic_map: np.ndarray,
    ignore_mask: np.ndarray = None,
) -> None:
    out = semantic_map.copy().astype(np.uint8)
    if ignore_mask is not None:
        out[ignore_mask] = 255
    ok = cv2.imwrite(path, out)
    if not ok:
        raise IOError(f"No se pudo guardar el mapa semántico: {path}")


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
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(ignore_u8, connectivity=8)
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


def overlay_instance_masks(
    image_bgr: np.ndarray,
    items: List[Dict],
    alpha: float = 0.45,
    draw_label: bool = True,
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

    for item in items:
        mask = item["mask"].astype(bool)
        color = color_from_id(item["obj_id"])

        colored = np.zeros_like(canvas, dtype=np.uint8)
        colored[:, :] = color
        mask3 = np.repeat(mask[:, :, None], 3, axis=2)
        canvas = np.where(mask3, cv2.addWeighted(canvas, 1 - alpha, colored, alpha, 0), canvas)

        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, color, 2)

        if draw_label:
            ys, xs = np.where(mask)
            if len(xs) > 0:
                x_text = int(xs.min())
                y_text = int(max(0, ys.min() - 5))
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


def overlay_semantic_map(
    image_bgr: np.ndarray,
    semantic_map: np.ndarray,
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

    for class_id, class_name in TARGET_CLASS_IDS.items():
        class_mask = semantic_map == class_id
        if not class_mask.any():
            continue

        color = SEMANTIC_COLORS[class_id]
        colored = np.zeros_like(canvas, dtype=np.uint8)
        colored[:, :] = color

        mask3 = np.repeat(class_mask[:, :, None], 3, axis=2)
        canvas = np.where(mask3, cv2.addWeighted(canvas, 0.55, colored, 0.45, 0), canvas)

        contours, _ = cv2.findContours(class_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, color, 2)

        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(class_mask.astype(np.uint8), connectivity=8)
        for comp_id in range(1, num_labels):
            x = int(stats[comp_id, cv2.CC_STAT_LEFT])
            y = int(stats[comp_id, cv2.CC_STAT_TOP])
            area = int(stats[comp_id, cv2.CC_STAT_AREA])
            if area <= 0:
                continue

            text_x = x
            text_y = max(20, y - 6)
            cv2.putText(
                canvas,
                class_name,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
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


def overlay_boxes(
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
        color = SEMANTIC_COLORS[pred["class_id"]]

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

        label = f"{pred['class_name']} box={pred['box_score']:.3f} sam={pred['sam_score']:.3f}"
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


def make_side_by_side(left_bgr: np.ndarray, right_bgr: np.ndarray) -> np.ndarray:
    if left_bgr.shape[0] != right_bgr.shape[0]:
        raise ValueError("Las dos imágenes deben tener la misma altura para concatenarlas.")
    return np.concatenate([left_bgr, right_bgr], axis=1)


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
        raw_labels = result.get("text_labels", result.get("labels", []))

        detections = []
        for idx, (box, score, raw_label) in enumerate(zip(boxes, scores, raw_labels)):
            if torch.is_tensor(box):
                box = box.detach().cpu().numpy()
            else:
                box = np.asarray(box)

            if torch.is_tensor(score):
                score = float(score.detach().cpu().item())
            else:
                score = float(score)

            if isinstance(raw_label, (list, tuple)):
                raw_label = " ".join([str(x) for x in raw_label])
            else:
                raw_label = str(raw_label)

            x1, y1, x2, y2 = box.tolist()
            detections.append(
                {
                    "det_local_id": idx,
                    "box": np.array([x1, y1, x2, y2], dtype=np.float32),
                    "score": score,
                    "raw_label": raw_label,
                    "prompt_text": prompt,
                }
            )

        return detections


class HFSamBoxPredictor:
    def __init__(
        self,
        base_model_name_or_path: str,
        device: str,
        state_dict_path: str = None,
    ):
        self.device = device
        self.model = SamModel.from_pretrained(base_model_name_or_path).to(device)

        if state_dict_path is not None:
            ckpt = torch.load(state_dict_path, map_location="cpu")
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                ckpt = ckpt["state_dict"]

            clean_state_dict = {}
            for k, v in ckpt.items():
                new_k = k
                if new_k.startswith("model."):
                    new_k = new_k[len("model."):]
                clean_state_dict[new_k] = v

            missing, unexpected = self.model.load_state_dict(clean_state_dict, strict=False)
            print(f"[INFO] Fine-tuned weights cargados desde: {state_dict_path}")
            print(f"[INFO] Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")

        self.model.eval()
        self.processor = SamProcessor.from_pretrained(base_model_name_or_path)
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


def build_model_variants() -> Dict[str, Dict[str, Any]]:
    model_variants = {
        "pretrained": {
            "base_model_name_or_path": PRETRAINED_SAM_MODEL_NAME_OR_PATH,
            "state_dict_path": None,
            "description": PRETRAINED_SAM_MODEL_NAME_OR_PATH,
        }
    }

    if FINETUNED_SAM_MODEL_NAME_OR_PATH is not None:
        model_variants["finetuned"] = {
            "base_model_name_or_path": FINETUNED_SAM_MODEL_NAME_OR_PATH,
            "state_dict_path": None,
            "description": FINETUNED_SAM_MODEL_NAME_OR_PATH,
        }
    elif FINETUNED_SAM_STATE_DICT_PATH is not None:
        model_variants["finetuned"] = {
            "base_model_name_or_path": PRETRAINED_SAM_MODEL_NAME_OR_PATH,
            "state_dict_path": FINETUNED_SAM_STATE_DICT_PATH,
            "description": FINETUNED_SAM_STATE_DICT_PATH,
        }

    return model_variants


def run_semantic_text_prompt_pipeline(
    image_shape: Tuple[int, int],
    image_pil: Image.Image,
    grounding_predictor: GroundingDINOPredictor,
    sam_predictor: HFSamBoxPredictor,
    prompt_map: Dict[int, str],
) -> Tuple[List[Dict], np.ndarray, Dict[int, np.ndarray]]:
    image_h, image_w = image_shape[:2]
    predictions: List[Dict] = []
    class_score_maps = {
        class_id: np.zeros((image_h, image_w), dtype=np.float32)
        for class_id in TARGET_CLASS_IDS.keys()
    }

    det_counter = 0

    for class_id, text_prompt in prompt_map.items():
        detections = grounding_predictor.predict(image_pil=image_pil, text_prompt=text_prompt)

        for det in detections:
            sam_out = sam_predictor.predict_from_box(
                box_xyxy=det["box"],
                multimask_output=MULTIMASK_OUTPUT,
            )

            pred_mask = sam_out["best_mask"]
            pred_area = int(pred_mask.sum())
            if pred_area < MIN_MASK_AREA:
                continue

            combined_score = float(det["score"]) * float(sam_out["best_score"])

            score_map = class_score_maps[class_id]
            score_map[pred_mask] = np.maximum(score_map[pred_mask], combined_score)

            predictions.append(
                {
                    "det_id": det_counter,
                    "class_id": class_id,
                    "class_name": TARGET_CLASS_IDS[class_id],
                    "prompt_text": text_prompt,
                    "raw_label": det["raw_label"],
                    "box": det["box"].copy(),
                    "box_score": float(det["score"]),
                    "mask": pred_mask,
                    "mask_area": pred_area,
                    "sam_score": float(sam_out["best_score"]),
                    "combined_score": combined_score,
                    "sam_best_idx": int(sam_out["best_idx"]),
                }
            )
            det_counter += 1

    pred_semantic_map = np.zeros((image_h, image_w), dtype=np.uint8)

    class_ids = sorted(TARGET_CLASS_IDS.keys())
    if len(class_ids) > 0:
        score_stack = np.stack([class_score_maps[cid] for cid in class_ids], axis=0)
        max_scores = score_stack.max(axis=0)
        best_class_idx = score_stack.argmax(axis=0)

        for stack_idx, class_id in enumerate(class_ids):
            pred_semantic_map[(max_scores > 0) & (best_class_idx == stack_idx)] = class_id

    return predictions, pred_semantic_map, class_score_maps


def write_metrics_txt(
    output_txt_path: str,
    image_path: str,
    gt_txt_path: str,
    frame_id: int,
    instances: List[Dict],
    ignore_mask: np.ndarray,
    all_results: Dict[str, Dict[str, Any]],
) -> None:
    lines = []
    lines.append("============================================================")
    lines.append("TASK H | Semantic Segmentation on KITTI-MOTS")
    lines.append("============================================================")
    lines.append(f"IMAGE_PATH: {image_path}")
    lines.append(f"GT_TXT_PATH: {gt_txt_path}")
    lines.append(f"FRAME_ID: {frame_id}")
    lines.append(f"GROUNDING_DINO_MODEL_ID: {GROUNDING_DINO_MODEL_ID}")
    lines.append(f"PRETRAINED_SAM_MODEL_NAME_OR_PATH: {PRETRAINED_SAM_MODEL_NAME_OR_PATH}")
    lines.append(f"FINETUNED_SAM_MODEL_NAME_OR_PATH: {FINETUNED_SAM_MODEL_NAME_OR_PATH}")
    lines.append(f"FINETUNED_SAM_STATE_DICT_PATH: {FINETUNED_SAM_STATE_DICT_PATH}")
    lines.append(f"DEVICE: {DEVICE}")
    lines.append(f"BOX_THRESHOLD: {BOX_THRESHOLD}")
    lines.append(f"TEXT_THRESHOLD: {TEXT_THRESHOLD}")
    lines.append(f"MULTIMASK_OUTPUT: {MULTIMASK_OUTPUT}")
    lines.append(f"MIN_MASK_AREA: {MIN_MASK_AREA}")
    lines.append(f"N_INSTANCE_GT: {len(instances)}")
    lines.append(f"IGNORE_PIXELS: {int(ignore_mask.sum())}")
    lines.append("")

    for method_name, method_data in all_results.items():
        metrics = method_data["metrics"]
        overall = metrics["overall"]
        by_class = metrics["by_class"]
        predictions = method_data["predictions"]
        prompt_map = method_data["prompt_map"]

        lines.append("------------------------------------------------------------")
        lines.append(f"METHOD: {method_name}")
        lines.append("------------------------------------------------------------")
        lines.append(f"model_variant: {method_data['model_variant_name']}")
        lines.append(f"sam_model_source: {method_data['sam_model_description']}")
        lines.append(f"prompt_variant: {method_data['prompt_variant_name']}")
        lines.append("prompt_map:")
        for class_id, text_prompt in prompt_map.items():
            lines.append(f"  class_id={class_id} ({TARGET_CLASS_IDS[class_id]}) -> '{text_prompt}'")
        lines.append("")

        lines.append("OVERALL")
        lines.append(f"  overall_pixel_accuracy    = {overall['overall_pixel_accuracy']:.6f}")
        lines.append(f"  foreground_pixel_accuracy = {overall['foreground_pixel_accuracy']:.6f}")
        lines.append(f"  mean_iou                  = {overall['mean_iou']:.6f}")
        lines.append(f"  mean_dice                 = {overall['mean_dice']:.6f}")
        lines.append(f"  num_predictions           = {len(predictions)}")
        lines.append("")

        lines.append("BY CLASS")
        for class_name, cls in by_class.items():
            lines.append(f"  CLASS={class_name}")
            lines.append(f"    iou         = {cls['iou']:.6f}")
            lines.append(f"    dice        = {cls['dice']:.6f}")
            lines.append(f"    precision   = {cls['precision']:.6f}")
            lines.append(f"    recall      = {cls['recall']:.6f}")
            lines.append(f"    pred_area   = {cls['pred_area']}")
            lines.append(f"    gt_area     = {cls['gt_area']}")
            lines.append(f"    intersection= {cls['intersection']}")
            lines.append(f"    union       = {cls['union']}")
            lines.append("")

        lines.append("PREDICTIONS DETAIL")
        lines.append("det_id | class_name | raw_label | prompt_text | box_score | sam_score | combined_score | mask_area | box_xyxy")
        for pred in predictions:
            box_str = [round(float(x), 2) for x in pred["box"].tolist()]
            lines.append(
                f"{pred['det_id']} | {pred['class_name']} | {pred['raw_label']} | {pred['prompt_text']} | "
                f"{pred['box_score']:.6f} | {pred['sam_score']:.6f} | {pred['combined_score']:.6f} | "
                f"{pred['mask_area']} | {box_str}"
            )
        lines.append("")

    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    set_seed(RANDOM_SEED)

    ensure_dir(OUTPUT_DIR)
    overlays_dir = os.path.join(OUTPUT_DIR, "overlays")
    labelmaps_dir = os.path.join(OUTPUT_DIR, "label_maps")
    per_detection_dir = os.path.join(OUTPUT_DIR, "per_detection")
    ensure_dir(overlays_dir)
    ensure_dir(labelmaps_dir)
    if SAVE_PER_DETECTION:
        ensure_dir(per_detection_dir)

    image_bgr = cv2.imread(IMAGE_PATH, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {IMAGE_PATH}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_h, image_w = image_bgr.shape[:2]
    image_shape = (image_h, image_w)
    image_pil = Image.fromarray(image_rgb)

    frame_id = FRAME_ID if FRAME_ID is not None else infer_frame_id_from_image_name(IMAGE_PATH)

    instances, ignore_mask = parse_gt_txt_for_frame(
        GT_TXT_PATH,
        frame_id=frame_id,
        image_shape=image_shape,
    )

    if len(instances) == 0:
        raise RuntimeError(
            f"No encontré instancias car/pedestrian para frame_id={frame_id} en el GT TXT."
        )

    gt_semantic_map, gt_class_masks = build_semantic_gt(
        instances=instances,
        image_shape=image_shape,
    )

    # Guardar GT de instancia y GT semántico
    gt_instance_overlay = overlay_instance_masks(
        image_bgr=image_bgr,
        items=instances,
        alpha=0.45,
        draw_label=True,
        title=f"GT instances | frame={frame_id}",
        ignore_mask=ignore_mask,
    )
    save_image(os.path.join(overlays_dir, "gt_instance_overlay.png"), gt_instance_overlay)

    gt_semantic_overlay = overlay_semantic_map(
        image_bgr=image_bgr,
        semantic_map=gt_semantic_map,
        title=f"GT semantic | frame={frame_id}",
        ignore_mask=ignore_mask,
    )
    save_image(os.path.join(overlays_dir, "gt_semantic_overlay.png"), gt_semantic_overlay)
    save_semantic_label_png(
        os.path.join(labelmaps_dir, "gt_semantic_map.png"),
        gt_semantic_map,
        ignore_mask=ignore_mask,
    )

    grounding_predictor = GroundingDINOPredictor(GROUNDING_DINO_MODEL_ID, DEVICE)
    model_variants = build_model_variants()

    all_results: Dict[str, Dict[str, Any]] = {}

    for model_variant_name, model_cfg in model_variants.items():
        sam_predictor = HFSamBoxPredictor(
            base_model_name_or_path=model_cfg["base_model_name_or_path"],
            device=DEVICE,
            state_dict_path=model_cfg["state_dict_path"],
        )
        sam_predictor.set_image(image_rgb)

        for prompt_variant_name, prompt_map in TEXT_PROMPT_VARIANTS.items():
            method_name = f"{model_variant_name}__{prompt_variant_name}"

            predictions, pred_semantic_map, class_score_maps = run_semantic_text_prompt_pipeline(
                image_shape=image_shape,
                image_pil=image_pil,
                grounding_predictor=grounding_predictor,
                sam_predictor=sam_predictor,
                prompt_map=prompt_map,
            )

            metrics = compute_semantic_metrics(
                pred_semantic_map=pred_semantic_map,
                gt_semantic_map=gt_semantic_map,
                ignore_mask=ignore_mask,
            )

            all_results[method_name] = {
                "model_variant_name": model_variant_name,
                "sam_model_description": model_cfg["description"],
                "prompt_variant_name": prompt_variant_name,
                "prompt_map": prompt_map,
                "predictions": predictions,
                "pred_semantic_map": pred_semantic_map,
                "metrics": metrics,
            }

            # Overlays y mapas
            boxes_overlay = overlay_boxes(
                image_bgr=image_bgr,
                predictions=predictions,
                title=f"{method_name} | grounding boxes",
                ignore_mask=ignore_mask,
            )
            save_image(os.path.join(overlays_dir, f"{method_name}_boxes_overlay.png"), boxes_overlay)

            pred_semantic_overlay = overlay_semantic_map(
                image_bgr=image_bgr,
                semantic_map=pred_semantic_map,
                title=f"{method_name} | semantic prediction",
                ignore_mask=ignore_mask,
            )
            save_image(os.path.join(overlays_dir, f"{method_name}_semantic_overlay.png"), pred_semantic_overlay)

            comparison_panel = make_side_by_side(gt_semantic_overlay, pred_semantic_overlay)
            save_image(os.path.join(overlays_dir, f"{method_name}_semantic_compare.png"), comparison_panel)

            save_semantic_label_png(
                os.path.join(labelmaps_dir, f"{method_name}_semantic_map.png"),
                pred_semantic_map,
                ignore_mask=ignore_mask,
            )

            # JSON resumen
            summary_json_path = os.path.join(OUTPUT_DIR, f"{method_name}_summary.json")
            with open(summary_json_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "method_name": method_name,
                        "model_variant_name": model_variant_name,
                        "sam_model_description": model_cfg["description"],
                        "prompt_variant_name": prompt_variant_name,
                        "prompt_map": prompt_map,
                        "metrics": metrics,
                        "num_predictions": len(predictions),
                    },
                    f,
                    indent=2,
                )

            if SAVE_PER_DETECTION:
                method_det_dir = os.path.join(per_detection_dir, method_name)
                ensure_dir(method_det_dir)

                for pred in predictions:
                    det_dir = os.path.join(
                        method_det_dir,
                        f"det_{pred['det_id']:03d}_{pred['class_name']}"
                    )
                    ensure_dir(det_dir)

                    single_box_overlay = overlay_boxes(
                        image_bgr=image_bgr,
                        predictions=[pred],
                        title=f"{method_name} | det={pred['det_id']} | box",
                        ignore_mask=ignore_mask,
                    )
                    save_image(os.path.join(det_dir, "box_overlay.png"), single_box_overlay)

                    single_mask_semantic_map = np.zeros_like(pred_semantic_map, dtype=np.uint8)
                    single_mask_semantic_map[pred["mask"]] = pred["class_id"]

                    single_mask_overlay = overlay_semantic_map(
                        image_bgr=image_bgr,
                        semantic_map=single_mask_semantic_map,
                        title=f"{method_name} | det={pred['det_id']} | semantic mask",
                        ignore_mask=ignore_mask,
                    )
                    save_image(os.path.join(det_dir, "semantic_mask_overlay.png"), single_mask_overlay)

                    save_semantic_label_png(
                        os.path.join(det_dir, "semantic_mask_map.png"),
                        single_mask_semantic_map,
                        ignore_mask=ignore_mask,
                    )

    metrics_txt_path = os.path.join(OUTPUT_DIR, "metrics.txt")
    write_metrics_txt(
        output_txt_path=metrics_txt_path,
        image_path=IMAGE_PATH,
        gt_txt_path=GT_TXT_PATH,
        frame_id=frame_id,
        instances=instances,
        ignore_mask=ignore_mask,
        all_results=all_results,
    )

    print("Proceso completado.")
    print(f"Resultados guardados en: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
