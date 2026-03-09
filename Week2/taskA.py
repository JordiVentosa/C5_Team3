import os
import re
import random
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from transformers import SamModel, SamProcessor

# =========================
# CONFIG
# =========================
IMAGE_PATH = "/ghome/group03/mcv/datasets/C5/KITTI-MOTS/training/image_02/0000/000000.png"
GT_TXT_PATH = "/ghome/group03/mcv/datasets/C5/KITTI-MOTS/instances_txt/0000.txt"

HF_MODEL_NAME = "facebook/sam-vit-huge"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Si lo dejas a None, se infiere desde el nombre del archivo de la imagen.
FRAME_ID = None

# Si True, SAM devuelve varias máscaras candidatas y nos quedamos con la de mejor iou_score.
# Si False, SAM devuelve una sola máscara.
MULTIMASK_OUTPUT = True

# Directorio de salida
OUTPUT_DIR = "./sam_hf_kitti_mots_0000_000000"

# Reproducibilidad
RANDOM_SEED = 42

# Prompt negativo
NEG_MARGIN = 20
NEG_MAX_TRIES = 500

# Guardar overlays también por instancia
SAVE_PER_INSTANCE = True

# Config visual para ignore
DRAW_IGNORE_OVERLAY = True
DRAW_IGNORE_LABEL = True
IGNORE_ALPHA = 0.35
IGNORE_COLOR_BGR = (120, 120, 120)
IGNORE_TEXT = "ignore class=10"

# Clases a evaluar
TARGET_CLASS_IDS = {
    1: "car",
    2: "person",  # KITTI-MOTS usa pedestrian; aquí lo etiquetamos como person en las salidas
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


def nearest_foreground_to_point(mask: np.ndarray, yx_point: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise ValueError("La máscara está vacía; no se puede obtener un punto interior.")
    coords = np.stack([ys, xs], axis=1)
    d2 = np.sum((coords - yx_point[None, :]) ** 2, axis=1)
    best = coords[np.argmin(d2)]
    return np.array([best[1], best[0]], dtype=np.float32)  # x, y


def prompt_center_positive(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ys, xs = np.where(mask)
    centroid_y = int(np.round(np.mean(ys)))
    centroid_x = int(np.round(np.mean(xs)))
    point_xy = nearest_foreground_to_point(mask, np.array([centroid_y, centroid_x]))
    points = np.array([point_xy], dtype=np.float32)
    labels = np.array([1], dtype=np.int64)
    return points, labels


def prompt_random_positive(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ys, xs = np.where(mask)
    idx = np.random.randint(0, len(xs))
    point_xy = np.array([xs[idx], ys[idx]], dtype=np.float32)
    points = np.array([point_xy], dtype=np.float32)
    labels = np.array([1], dtype=np.int64)
    return points, labels


def prompt_three_positives(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ys, xs = np.where(mask)
    coords = np.stack([xs, ys], axis=1)

    center_pt, _ = prompt_center_positive(mask)
    chosen = [tuple(center_pt[0].astype(int).tolist())]

    if len(coords) > 1:
        perm = np.random.permutation(len(coords))
        for idx in perm:
            candidate = tuple(coords[idx].astype(int).tolist())
            if candidate not in chosen:
                chosen.append(candidate)
            if len(chosen) == 3:
                break

    while len(chosen) < 3:
        chosen.append(chosen[-1])

    points = np.array(chosen, dtype=np.float32)
    labels = np.array([1, 1, 1], dtype=np.int64)
    return points, labels


def prompt_pos_plus_neg(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pos_points, _ = prompt_center_positive(mask)
    pos_xy = pos_points[0]

    ys, xs = np.where(mask)
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()

    h, w = mask.shape
    x0 = max(0, x_min - NEG_MARGIN)
    x1 = min(w - 1, x_max + NEG_MARGIN)
    y0 = max(0, y_min - NEG_MARGIN)
    y1 = min(h - 1, y_max + NEG_MARGIN)

    neg_xy = None
    for _ in range(NEG_MAX_TRIES):
        x = np.random.randint(x0, x1 + 1)
        y = np.random.randint(y0, y1 + 1)
        if not mask[y, x]:
            neg_xy = np.array([x, y], dtype=np.float32)
            break

    if neg_xy is None:
        bg_ys, bg_xs = np.where(~mask)
        if len(bg_xs) == 0:
            neg_xy = pos_xy.copy()
        else:
            idx = np.random.randint(0, len(bg_xs))
            neg_xy = np.array([bg_xs[idx], bg_ys[idx]], dtype=np.float32)

    points = np.array([pos_xy, neg_xy], dtype=np.float32)
    labels = np.array([1, 0], dtype=np.int64)
    return points, labels


PROMPT_BUILDERS = {
    "prompt_1pos_center": prompt_center_positive,
    "prompt_1pos_random": prompt_random_positive,
    "prompt_3pos": prompt_three_positives,
    "prompt_1pos_1neg": prompt_pos_plus_neg,
}


def compute_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray, ignore_mask: np.ndarray) -> Dict[str, float]:
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


def color_from_id(obj_id: int) -> Tuple[int, int, int]:
    rng = np.random.default_rng(obj_id)
    color = rng.integers(60, 255, size=3).tolist()
    return int(color[0]), int(color[1]), int(color[2])


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

    # Overlay gris translúcido
    colored = np.zeros_like(out, dtype=np.uint8)
    colored[:, :] = color_bgr
    mask3 = np.repeat(ignore_mask[:, :, None], 3, axis=2)
    blended = cv2.addWeighted(out, 1 - alpha, colored, alpha, 0)
    out = np.where(mask3, blended, out)

    ignore_u8 = ignore_mask.astype(np.uint8)

    # Contorno global de ignore
    contours, _ = cv2.findContours(ignore_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color_bgr, 2)

    if draw_label:
        num_labels, labels_img, stats, _ = cv2.connectedComponentsWithStats(ignore_u8, connectivity=8)

        for comp_id in range(1, num_labels):
            x = int(stats[comp_id, cv2.CC_STAT_LEFT])
            y = int(stats[comp_id, cv2.CC_STAT_TOP])
            w = int(stats[comp_id, cv2.CC_STAT_WIDTH])
            h = int(stats[comp_id, cv2.CC_STAT_HEIGHT])
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

            cv2.rectangle(out, (x, y), (x + w, y + h), color_bgr, 1)

    return out


def overlay_masks(
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


def overlay_points(
    image_bgr: np.ndarray,
    prompt_items: List[Dict],
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

    for item in prompt_items:
        points = item["points"]
        labels = item["labels"]
        obj_id = item["obj_id"]
        class_name = item["class_name"]

        for point_xy, label in zip(points, labels):
            x, y = int(point_xy[0]), int(point_xy[1])
            if int(label) == 1:
                color = (0, 255, 0)
                cv2.circle(canvas, (x, y), 6, color, -1)
                cv2.circle(canvas, (x, y), 9, (255, 255, 255), 2)
            else:
                color = (0, 0, 255)
                cv2.drawMarker(
                    canvas,
                    (x, y),
                    color,
                    markerType=cv2.MARKER_TILTED_CROSS,
                    markerSize=14,
                    thickness=2,
                )

        x0, y0 = int(points[0][0]), int(points[0][1])
        cv2.putText(
            canvas,
            f"{class_name} id={obj_id}",
            (x0 + 5, max(15, y0 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
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


def save_image(path: str, image_bgr: np.ndarray) -> None:
    ok = cv2.imwrite(path, image_bgr)
    if not ok:
        raise IOError(f"No se pudo guardar la imagen: {path}")


def summarize_metrics(results: List[Dict]) -> Dict:
    summary = {"overall": {}, "by_class": {}}

    if not results:
        return summary

    def _aggregate(rows: List[Dict]) -> Dict[str, float]:
        return {
            "n": len(rows),
            "mean_iou": float(np.mean([r["metrics"]["iou"] for r in rows])) if rows else 0.0,
            "mean_dice": float(np.mean([r["metrics"]["dice"] for r in rows])) if rows else 0.0,
            "mean_precision": float(np.mean([r["metrics"]["precision"] for r in rows])) if rows else 0.0,
            "mean_recall": float(np.mean([r["metrics"]["recall"] for r in rows])) if rows else 0.0,
            "mean_pred_area": float(np.mean([r["metrics"]["pred_area"] for r in rows])) if rows else 0.0,
            "mean_gt_area": float(np.mean([r["metrics"]["gt_area"] for r in rows])) if rows else 0.0,
        }

    summary["overall"] = _aggregate(results)

    class_names = sorted(set(r["class_name"] for r in results))
    for cls_name in class_names:
        cls_rows = [r for r in results if r["class_name"] == cls_name]
        summary["by_class"][cls_name] = _aggregate(cls_rows)

    return summary


def write_metrics_txt(
    output_txt_path: str,
    image_path: str,
    gt_txt_path: str,
    frame_id: int,
    instances: List[Dict],
    ignore_mask: np.ndarray,
    all_prompt_results: Dict[str, List[Dict]],
) -> None:
    lines = []
    lines.append("============================================================")
    lines.append("SAM (HuggingFace) + KITTI-MOTS | Métricas")
    lines.append("============================================================")
    lines.append(f"IMAGE_PATH: {image_path}")
    lines.append(f"GT_TXT_PATH: {gt_txt_path}")
    lines.append(f"FRAME_ID: {frame_id}")
    lines.append(f"HF_MODEL_NAME: {HF_MODEL_NAME}")
    lines.append(f"DEVICE: {DEVICE}")
    lines.append(f"MULTIMASK_OUTPUT: {MULTIMASK_OUTPUT}")
    lines.append(f"N_INSTANCIAS_EVALUADAS: {len(instances)}")
    lines.append(f"IGNORE_PIXELS: {int(ignore_mask.sum())}")
    lines.append("")

    for prompt_name, results in all_prompt_results.items():
        summary = summarize_metrics(results)
        lines.append("------------------------------------------------------------")
        lines.append(f"PROMPT: {prompt_name}")
        lines.append("------------------------------------------------------------")
        ov = summary["overall"]
        lines.append(f"OVERALL | n={ov.get('n', 0)}")
        lines.append(f"  mean_iou       = {ov.get('mean_iou', 0.0):.6f}")
        lines.append(f"  mean_dice      = {ov.get('mean_dice', 0.0):.6f}")
        lines.append(f"  mean_precision = {ov.get('mean_precision', 0.0):.6f}")
        lines.append(f"  mean_recall    = {ov.get('mean_recall', 0.0):.6f}")
        lines.append(f"  mean_pred_area = {ov.get('mean_pred_area', 0.0):.2f}")
        lines.append(f"  mean_gt_area   = {ov.get('mean_gt_area', 0.0):.2f}")
        lines.append("")

        for cls_name, cls_sum in summary["by_class"].items():
            lines.append(f"CLASS={cls_name} | n={cls_sum['n']}")
            lines.append(f"  mean_iou       = {cls_sum['mean_iou']:.6f}")
            lines.append(f"  mean_dice      = {cls_sum['mean_dice']:.6f}")
            lines.append(f"  mean_precision = {cls_sum['mean_precision']:.6f}")
            lines.append(f"  mean_recall    = {cls_sum['mean_recall']:.6f}")
            lines.append(f"  mean_pred_area = {cls_sum['mean_pred_area']:.2f}")
            lines.append(f"  mean_gt_area   = {cls_sum['mean_gt_area']:.2f}")
            lines.append("")

        lines.append("DETALLE POR INSTANCIA:")
        lines.append(
            "obj_id | class_name | gt_area | pred_area | intersection | union | iou | dice | precision | recall"
        )
        for r in results:
            m = r["metrics"]
            lines.append(
                f"{r['obj_id']} | {r['class_name']} | {m['gt_area']} | {m['pred_area']} | "
                f"{m['intersection']} | {m['union']} | {m['iou']:.6f} | {m['dice']:.6f} | "
                f"{m['precision']:.6f} | {m['recall']:.6f}"
            )
        lines.append("")

    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def move_to_device(batch: Dict, device: str) -> Dict:
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


class HFSamPointPredictor:
    def __init__(self, model_name: str, device: str):
        self.device = device
        self.model = SamModel.from_pretrained(model_name).to(device)
        self.model.eval()
        self.processor = SamProcessor.from_pretrained(model_name)
        self.image_pil = None
        self.image_embeddings = None

    @torch.no_grad()
    def set_image(self, image_rgb: np.ndarray) -> None:
        self.image_pil = Image.fromarray(image_rgb)
        image_inputs = self.processor(images=self.image_pil, return_tensors="pt")
        pixel_values = image_inputs["pixel_values"].to(self.device)
        self.image_embeddings = self.model.get_image_embeddings(pixel_values)

    @torch.no_grad()
    def predict(self, points_xy: np.ndarray, labels: np.ndarray, multimask_output: bool = True):
        if self.image_pil is None or self.image_embeddings is None:
            raise RuntimeError("Debes llamar antes a set_image(...).")

        input_points = [[points_xy.tolist()]]
        input_labels = [[labels.astype(int).tolist()]]

        inputs = self.processor(
            images=self.image_pil,
            input_points=input_points,
            input_labels=input_labels,
            return_tensors="pt",
        )
        original_sizes = inputs["original_sizes"]
        reshaped_input_sizes = inputs["reshaped_input_sizes"]

        inputs = move_to_device(inputs, self.device)

        outputs = self.model(
            input_points=inputs["input_points"],
            input_labels=inputs["input_labels"],
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
        }


def main():
    set_seed(RANDOM_SEED)

    ensure_dir(OUTPUT_DIR)
    overlays_dir = os.path.join(OUTPUT_DIR, "overlays")
    points_dir = os.path.join(OUTPUT_DIR, "points")
    per_instance_dir = os.path.join(OUTPUT_DIR, "per_instance")
    ensure_dir(overlays_dir)
    ensure_dir(points_dir)
    if SAVE_PER_INSTANCE:
        ensure_dir(per_instance_dir)

    image_bgr = cv2.imread(IMAGE_PATH, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {IMAGE_PATH}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_h, image_w = image_bgr.shape[:2]

    frame_id = FRAME_ID if FRAME_ID is not None else infer_frame_id_from_image_name(IMAGE_PATH)

    instances, ignore_mask = parse_gt_txt_for_frame(
        GT_TXT_PATH,
        frame_id=frame_id,
        image_shape=(image_h, image_w),
    )

    if len(instances) == 0:
        raise RuntimeError(
            f"No encontré instancias car/person para frame_id={frame_id} en el GT TXT."
        )

    predictor = HFSamPointPredictor(HF_MODEL_NAME, DEVICE)
    predictor.set_image(image_rgb)

    gt_overlay = overlay_masks(
        image_bgr=image_bgr,
        items=instances,
        alpha=0.45,
        draw_label=True,
        title=f"GT | frame={frame_id}",
        ignore_mask=ignore_mask,
    )
    save_image(os.path.join(overlays_dir, "gt_overlay.png"), gt_overlay)

    all_prompt_results: Dict[str, List[Dict]] = {}

    for prompt_name, prompt_builder in PROMPT_BUILDERS.items():
        prompt_results: List[Dict] = []
        pred_items_for_overlay: List[Dict] = []
        prompt_items_for_overlay: List[Dict] = []

        prompt_dir = os.path.join(per_instance_dir, prompt_name)
        if SAVE_PER_INSTANCE:
            ensure_dir(prompt_dir)

        for inst in instances:
            gt_mask = inst["mask"]
            points, labels = prompt_builder(gt_mask)

            pred = predictor.predict(
                points_xy=points,
                labels=labels,
                multimask_output=MULTIMASK_OUTPUT,
            )
            pred_mask = pred["best_mask"]
            metrics = compute_metrics(pred_mask, gt_mask, ignore_mask)

            result = {
                "prompt_name": prompt_name,
                "frame_id": inst["frame_id"],
                "obj_id": inst["obj_id"],
                "class_id": inst["class_id"],
                "class_name": inst["class_name"],
                "points": points.copy(),
                "labels": labels.copy(),
                "metrics": metrics,
                "scores": pred["scores"].tolist(),
                "best_idx": pred["best_idx"],
            }
            prompt_results.append(result)

            pred_items_for_overlay.append(
                {
                    "obj_id": inst["obj_id"],
                    "class_name": inst["class_name"],
                    "mask": pred_mask,
                }
            )

            prompt_items_for_overlay.append(
                {
                    "obj_id": inst["obj_id"],
                    "class_name": inst["class_name"],
                    "points": points,
                    "labels": labels,
                }
            )

            if SAVE_PER_INSTANCE:
                instance_subdir = os.path.join(prompt_dir, f"obj_{inst['obj_id']}_{inst['class_name']}")
                ensure_dir(instance_subdir)

                pred_overlay_inst = overlay_masks(
                    image_bgr=image_bgr,
                    items=[{"obj_id": inst["obj_id"], "class_name": inst["class_name"], "mask": pred_mask}],
                    alpha=0.5,
                    draw_label=True,
                    title=f"{prompt_name} | pred | obj={inst['obj_id']}",
                    ignore_mask=ignore_mask,
                )
                gt_overlay_inst = overlay_masks(
                    image_bgr=image_bgr,
                    items=[inst],
                    alpha=0.5,
                    draw_label=True,
                    title=f"GT | obj={inst['obj_id']}",
                    ignore_mask=ignore_mask,
                )
                pts_overlay_inst = overlay_points(
                    image_bgr=image_bgr,
                    prompt_items=[{
                        "obj_id": inst["obj_id"],
                        "class_name": inst["class_name"],
                        "points": points,
                        "labels": labels,
                    }],
                    title=f"{prompt_name} | points | obj={inst['obj_id']}",
                    ignore_mask=ignore_mask,
                )

                save_image(os.path.join(instance_subdir, "pred_overlay.png"), pred_overlay_inst)
                save_image(os.path.join(instance_subdir, "gt_overlay.png"), gt_overlay_inst)
                save_image(os.path.join(instance_subdir, "points_overlay.png"), pts_overlay_inst)

        all_prompt_results[prompt_name] = prompt_results

        pred_overlay = overlay_masks(
            image_bgr=image_bgr,
            items=pred_items_for_overlay,
            alpha=0.45,
            draw_label=True,
            title=f"{prompt_name} | predicciones",
            ignore_mask=ignore_mask,
        )
        save_image(os.path.join(overlays_dir, f"{prompt_name}_pred_overlay.png"), pred_overlay)

        pts_overlay = overlay_points(
            image_bgr=image_bgr,
            prompt_items=prompt_items_for_overlay,
            title=f"{prompt_name} | puntos",
            ignore_mask=ignore_mask,
        )
        save_image(os.path.join(points_dir, f"{prompt_name}_points_overlay.png"), pts_overlay)

    metrics_txt_path = os.path.join(OUTPUT_DIR, "metrics.txt")
    write_metrics_txt(
        output_txt_path=metrics_txt_path,
        image_path=IMAGE_PATH,
        gt_txt_path=GT_TXT_PATH,
        frame_id=frame_id,
        instances=instances,
        ignore_mask=ignore_mask,
        all_prompt_results=all_prompt_results,
    )

    print("Proceso completado.")
    print(f"Resultados guardados en: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()