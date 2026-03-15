import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    SamModel,
    SamProcessor,
)

# ============================================================
# CONFIG
# ============================================================
KITTI_ROOT = "/ghome/group03/mcv/datasets/C5/KITTI-MOTS"
IMAGES_ROOT = os.path.join(KITTI_ROOT, "training", "image_02")
GT_TXT_ROOT = os.path.join(KITTI_ROOT, "instances_txt")

SELECTED_SEQUENCES = [2, 6, 7, 8, 10, 13, 14, 16, 18]

GROUNDING_DINO_MODEL_ID = "IDEA-Research/grounding-dino-base"
PRETRAINED_SAM_MODEL_NAME_OR_PATH = "facebook/sam-vit-huge"

# OPCIONAL:
# 1) Si tu modelo fine-tuned está guardado con save_pretrained(...), pon la carpeta aquí.
FINETUNED_SAM_MODEL_NAME_OR_PATH = None

# 2) Si tienes un state_dict .pt/.pth, pon la ruta aquí.
FINETUNED_SAM_STATE_DICT_PATH = None

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Prompt textual para Grounded SAM
TEXT_PROMPT_MAP = {
    1: "car.",
    2: "pedestrian.",
}

# Thresholds
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
BOX_NMS_IOU_THRESHOLD = 0.70
MIN_MASK_AREA = 25
MAX_DETECTIONS_PER_IMAGE = 100

# Si quieres probar solo unas pocas imágenes por secuencia para debug:
MAX_IMAGES_PER_SEQUENCE = None  # por ejemplo 10, o None para todas

# Ignore region
USE_IGNORE_MASK = True

# Métricas por clase además del global
PRINT_PER_CLASS = True

# Clases objetivo
TARGET_CLASS_IDS = {
    1: "car",
    2: "pedestrian",
}

RANDOM_SEED = 42


# ============================================================
# UTILIDADES
# ============================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def infer_frame_id_from_image_path(image_path: str) -> int:
    return int(Path(image_path).stem)


def move_to_device(batch: Dict[str, Any], device: str) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if hasattr(v, "to"):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def safe_div(num: float, den: float) -> float:
    return num / den if den != 0 else 0.0


def encode_binary_mask_to_coco_rle(mask: np.ndarray) -> Dict[str, Any]:
    mask_u8 = np.asfortranarray(mask.astype(np.uint8))
    rle = mask_utils.encode(mask_u8)
    if isinstance(rle["counts"], bytes):
        rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def decode_kitti_mots_rle(height: int, width: int, counts_str: str) -> np.ndarray:
    rle = {"size": [height, width], "counts": counts_str.encode("utf-8")}
    mask = mask_utils.decode(rle)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return mask.astype(bool)


def bbox_xyxy_to_xywh(box_xyxy: np.ndarray) -> List[float]:
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def box_iou_xyxy(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if boxes.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)

    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    inter = inter_w * inter_h

    area1 = np.maximum(0.0, box[2] - box[0]) * np.maximum(0.0, box[3] - box[1])
    area2 = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])

    union = area1 + area2 - inter
    return inter / np.maximum(union, 1e-8)


def nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
    if boxes.shape[0] == 0:
        return []

    order = scores.argsort()[::-1]
    keep: List[int] = []

    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break

        rest = order[1:]
        ious = box_iou_xyxy(boxes[i], boxes[rest])
        rest = rest[ious <= iou_threshold]
        order = rest

    return keep


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


# ============================================================
# PARSE GT KITTI-MOTS
# ============================================================
def index_gt_txt_by_frame(gt_txt_path: str) -> Dict[int, List[Tuple[int, int, int, int, str]]]:
    frame_records: Dict[int, List[Tuple[int, int, int, int, str]]] = {}

    with open(gt_txt_path, "r", encoding="utf-8") as f:
        for line_idx, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 6:
                raise ValueError(f"Línea mal formada en {gt_txt_path} (línea {line_idx}): {raw_line}")

            frame_id = int(parts[0])
            obj_id = int(parts[1])
            class_id = int(parts[2])
            h = int(parts[3])
            w = int(parts[4])
            counts_str = parts[5]

            frame_records.setdefault(frame_id, []).append((obj_id, class_id, h, w, counts_str))

    return frame_records


def parse_gt_frame_records(
    frame_records: List[Tuple[int, int, int, int, str]],
    image_shape: Tuple[int, int],
) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    image_h, image_w = image_shape[:2]
    instances: List[Dict[str, Any]] = []
    ignore_mask = np.zeros((image_h, image_w), dtype=bool)

    for obj_id, class_id, h, w, counts_str in frame_records:
        if (h, w) != (image_h, image_w):
            raise ValueError(
                f"Dimensiones GT e imagen no coinciden. GT=({h},{w}) vs imagen=({image_h},{image_w})"
            )

        mask = decode_kitti_mots_rle(h, w, counts_str)

        if obj_id == 10000 or class_id == 10:
            ignore_mask |= mask
            continue

        if class_id not in TARGET_CLASS_IDS:
            continue

        instances.append(
            {
                "obj_id": obj_id,
                "class_id": class_id,
                "class_name": TARGET_CLASS_IDS[class_id],
                "mask": mask,
            }
        )

    return instances, ignore_mask


# ============================================================
# PREDICTORES
# ============================================================
class GroundingDINOPredictor:
    def __init__(self, model_id: str, device: str):
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_id, use_fast=False)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, image_pil: Image.Image, text_prompt: str) -> List[Dict[str, Any]]:
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

        detections: List[Dict[str, Any]] = []
        for idx, (box, score, raw_label) in enumerate(zip(boxes, scores, raw_labels)):
            if torch.is_tensor(box):
                box = box.detach().cpu().numpy()
            else:
                box = np.asarray(box)

            if torch.is_tensor(score):
                score = float(score.detach().cpu().item())
            else:
                score = float(score)

            raw_label = str(raw_label)

            detections.append(
                {
                    "det_local_id": idx,
                    "box": box.astype(np.float32),
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
        state_dict_path: Optional[str] = None,
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
        self.processor = SamProcessor.from_pretrained(base_model_name_or_path, use_fast=False)
        self.image_pil: Optional[Image.Image] = None
        self.image_embeddings = None

    @torch.no_grad()
    def set_image(self, image_rgb: np.ndarray) -> None:
        self.image_pil = Image.fromarray(image_rgb)
        image_inputs = self.processor(images=self.image_pil, return_tensors="pt")
        pixel_values = image_inputs["pixel_values"].to(self.device)
        self.image_embeddings = self.model.get_image_embeddings(pixel_values)

    @torch.no_grad()
    def predict_from_box(self, box_xyxy: np.ndarray, multimask_output: bool = True) -> Dict[str, Any]:
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
            "scores": scores_np,
            "best_idx": best_idx,
            "best_score": float(scores_np[best_idx]),
        }


def build_model_variants() -> Dict[str, Dict[str, Optional[str]]]:
    model_variants = {
        "pretrained": {
            "base_model_name_or_path": PRETRAINED_SAM_MODEL_NAME_OR_PATH,
            "state_dict_path": None,
        }
    }

    if FINETUNED_SAM_MODEL_NAME_OR_PATH is not None:
        model_variants["finetuned"] = {
            "base_model_name_or_path": FINETUNED_SAM_MODEL_NAME_OR_PATH,
            "state_dict_path": None,
        }
    elif FINETUNED_SAM_STATE_DICT_PATH is not None:
        model_variants["finetuned"] = {
            "base_model_name_or_path": PRETRAINED_SAM_MODEL_NAME_OR_PATH,
            "state_dict_path": FINETUNED_SAM_STATE_DICT_PATH,
        }

    return model_variants


# ============================================================
# COCO GT
# ============================================================
def build_coco_gt_dataset() -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    coco_images: List[Dict[str, Any]] = []
    coco_annotations: List[Dict[str, Any]] = []

    ann_id = 1
    image_id = 1

    image_records: List[Dict[str, Any]] = []

    for seq_id in SELECTED_SEQUENCES:
        seq_name = f"{seq_id:04d}"
        image_dir = os.path.join(IMAGES_ROOT, seq_name)
        gt_txt_path = os.path.join(GT_TXT_ROOT, f"{seq_name}.txt")

        if not os.path.isdir(image_dir):
            raise FileNotFoundError(f"No existe el directorio de imágenes: {image_dir}")
        if not os.path.isfile(gt_txt_path):
            raise FileNotFoundError(f"No existe el GT TXT: {gt_txt_path}")

        frame_records_by_frame = index_gt_txt_by_frame(gt_txt_path)
        image_paths = sorted(str(p) for p in Path(image_dir).glob("*.png"))

        if MAX_IMAGES_PER_SEQUENCE is not None:
            image_paths = image_paths[:MAX_IMAGES_PER_SEQUENCE]

        for image_path in image_paths:
            image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise FileNotFoundError(f"No se pudo leer la imagen: {image_path}")

            image_h, image_w = image_bgr.shape[:2]
            frame_id = infer_frame_id_from_image_path(image_path)
            frame_records = frame_records_by_frame.get(frame_id, [])
            instances, ignore_mask = parse_gt_frame_records(frame_records, (image_h, image_w))

            coco_images.append(
                {
                    "id": image_id,
                    "file_name": f"{seq_name}/{Path(image_path).name}",
                    "width": image_w,
                    "height": image_h,
                }
            )

            image_records.append(
                {
                    "image_id": image_id,
                    "sequence_id": seq_id,
                    "sequence_name": seq_name,
                    "frame_id": frame_id,
                    "image_path": image_path,
                    "image_shape": (image_h, image_w),
                    "instances": instances,
                    "ignore_mask": ignore_mask,
                }
            )

            for inst in instances:
                mask = inst["mask"].copy()
                if USE_IGNORE_MASK:
                    mask = mask & (~ignore_mask)

                area = int(mask.sum())
                if area <= 0:
                    continue

                rle = encode_binary_mask_to_coco_rle(mask)
                bbox = mask_utils.toBbox(
                    {
                        "size": rle["size"],
                        "counts": rle["counts"].encode("utf-8"),
                    }
                ).tolist()

                coco_annotations.append(
                    {
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": inst["class_id"],
                        "segmentation": rle,
                        "area": float(area),
                        "bbox": [float(x) for x in bbox],
                        "iscrowd": 0,
                    }
                )
                ann_id += 1

            image_id += 1

    coco_dataset = {
        "info": {"description": "KITTI-MOTS selected sequences as COCO-style segm GT"},
        "licenses": [],
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": [
            {"id": 1, "name": "car"},
            {"id": 2, "name": "pedestrian"},
        ],
    }

    return coco_dataset, image_records


# ============================================================
# INFERENCIA EN UNA IMAGEN
# ============================================================
def run_inference_on_image(
    image_rgb: np.ndarray,
    image_pil: Image.Image,
    ignore_mask: np.ndarray,
    grounding_predictor: GroundingDINOPredictor,
    sam_predictor: HFSamBoxPredictor,
) -> List[Dict[str, Any]]:
    predictions: List[Dict[str, Any]] = []

    det_global_id = 0

    for class_id, text_prompt in TEXT_PROMPT_MAP.items():
        detections = grounding_predictor.predict(image_pil=image_pil, text_prompt=text_prompt)

        if len(detections) == 0:
            continue

        boxes = np.stack([d["box"] for d in detections], axis=0)
        scores = np.array([d["score"] for d in detections], dtype=np.float32)
        keep = nms_xyxy(boxes, scores, BOX_NMS_IOU_THRESHOLD)
        detections = [detections[i] for i in keep]

        for det in detections:
            sam_out = sam_predictor.predict_from_box(det["box"], multimask_output=True)
            pred_mask = sam_out["best_mask"].copy()

            if USE_IGNORE_MASK:
                pred_mask = pred_mask & (~ignore_mask)

            pred_area = int(pred_mask.sum())
            if pred_area < MIN_MASK_AREA:
                continue

            combined_score = float(det["score"]) * float(sam_out["best_score"])

            predictions.append(
                {
                    "det_id": det_global_id,
                    "class_id": class_id,
                    "class_name": TARGET_CLASS_IDS[class_id],
                    "raw_label": det["raw_label"],
                    "prompt_text": text_prompt,
                    "box": det["box"].copy(),
                    "box_score": float(det["score"]),
                    "sam_score": float(sam_out["best_score"]),
                    "score": combined_score,
                    "mask": pred_mask,
                    "mask_area": pred_area,
                }
            )
            det_global_id += 1

    if len(predictions) > MAX_DETECTIONS_PER_IMAGE:
        predictions = sorted(predictions, key=lambda x: x["score"], reverse=True)[:MAX_DETECTIONS_PER_IMAGE]

    return predictions


# ============================================================
# EVALUACIÓN COCO
# ============================================================
def evaluate_with_coco(
    coco_gt: COCO,
    results: List[Dict[str, Any]],
    image_ids: List[int],
    category_ids: List[int],
    title: str,
) -> Optional[np.ndarray]:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    if len(results) == 0:
        print("No hay predicciones. No se puede calcular COCOeval.")
        return None

    # IMPORTANTE:
    # pycocotools.loadRes modifica internamente la lista de resultados,
    # añadiendo campos como bbox/area/id. Si reutilizamos la misma lista
    # en otra llamada, puede romper al comparar bbox con [].
    # Por eso construimos una copia limpia.
    clean_results: List[Dict[str, Any]] = []

    for ann in results:
        clean_ann = {
            "image_id": int(ann["image_id"]),
            "category_id": int(ann["category_id"]),
            "segmentation": ann["segmentation"],
            "score": float(ann["score"]),
        }

        # Si por cualquier motivo ya existiera bbox, lo convertimos a lista.
        if "bbox" in ann:
            bbox = ann["bbox"]
            if isinstance(bbox, np.ndarray):
                bbox = bbox.tolist()
            clean_ann["bbox"] = [float(x) for x in bbox]

        clean_results.append(clean_ann)

    coco_dt = coco_gt.loadRes(clean_results)

    evaluator = COCOeval(coco_gt, coco_dt, iouType="segm")
    evaluator.params.imgIds = image_ids
    evaluator.params.catIds = category_ids
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    stats = evaluator.stats.copy()

    print("\nResumen corto:")
    print(f"AP   : {stats[0]:.6f}")
    print(f"AP50 : {stats[1]:.6f}")
    print(f"AP75 : {stats[2]:.6f}")
    print(f"APs  : {stats[3]:.6f}")
    print(f"APm  : {stats[4]:.6f}")
    print(f"APl  : {stats[5]:.6f}")

    return stats

def main():
    set_seed(RANDOM_SEED)

    print(f"[INFO] DEVICE = {DEVICE}")
    print(f"[INFO] Secuencias = {SELECTED_SEQUENCES}")
    print(f"[INFO] TEXT_PROMPT_MAP = {TEXT_PROMPT_MAP}")
    print(f"[INFO] BOX_THRESHOLD = {BOX_THRESHOLD}")
    print(f"[INFO] TEXT_THRESHOLD = {TEXT_THRESHOLD}")
    print(f"[INFO] BOX_NMS_IOU_THRESHOLD = {BOX_NMS_IOU_THRESHOLD}")
    print(f"[INFO] MIN_MASK_AREA = {MIN_MASK_AREA}")
    print(f"[INFO] MAX_DETECTIONS_PER_IMAGE = {MAX_DETECTIONS_PER_IMAGE}")
    print()

    coco_dataset, image_records = build_coco_gt_dataset()

    coco_gt = COCO()
    coco_gt.dataset = coco_dataset
    coco_gt.createIndex()

    image_ids = [img["id"] for img in coco_dataset["images"]]
    category_ids = sorted(TARGET_CLASS_IDS.keys())

    print(f"[INFO] Nº imágenes evaluadas: {len(coco_dataset['images'])}")
    print(f"[INFO] Nº anotaciones GT: {len(coco_dataset['annotations'])}")

    grounding_predictor = GroundingDINOPredictor(GROUNDING_DINO_MODEL_ID, DEVICE)
    model_variants = build_model_variants()

    for model_variant_name, model_cfg in model_variants.items():
        print("\n" + "#" * 80)
        print(f"[INFO] Evaluando modelo: {model_variant_name}")
        print(f"[INFO] base_model_name_or_path = {model_cfg['base_model_name_or_path']}")
        print(f"[INFO] state_dict_path        = {model_cfg['state_dict_path']}")
        print("#" * 80)

        sam_predictor = HFSamBoxPredictor(
            base_model_name_or_path=model_cfg["base_model_name_or_path"],
            device=DEVICE,
            state_dict_path=model_cfg["state_dict_path"],
        )

        coco_results: List[Dict[str, Any]] = []

        for idx, rec in enumerate(image_records, start=1):
            image_path = rec["image_path"]
            image_id = rec["image_id"]
            ignore_mask = rec["ignore_mask"]

            image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise FileNotFoundError(f"No se pudo leer la imagen: {image_path}")

            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image_rgb)

            sam_predictor.set_image(image_rgb)

            predictions = run_inference_on_image(
                image_rgb=image_rgb,
                image_pil=image_pil,
                ignore_mask=ignore_mask,
                grounding_predictor=grounding_predictor,
                sam_predictor=sam_predictor,
            )

            for pred in predictions:
                rle = encode_binary_mask_to_coco_rle(pred["mask"])
                coco_results.append(
                    {
                        "image_id": image_id,
                        "category_id": pred["class_id"],
                        "segmentation": rle,
                        "score": float(pred["score"]),
                    }
                )

            if idx % 25 == 0 or idx == len(image_records):
                print(
                    f"[{model_variant_name}] "
                    f"{idx}/{len(image_records)} imágenes procesadas | "
                    f"pred acumuladas: {len(coco_results)}"
                )

        stats_global = evaluate_with_coco(
            coco_gt=coco_gt,
            results=coco_results,
            image_ids=image_ids,
            category_ids=category_ids,
            title=f"COCO segm metrics | {model_variant_name} | global",
        )

        if PRINT_PER_CLASS:
            for class_id, class_name in TARGET_CLASS_IDS.items():
                _ = evaluate_with_coco(
                    coco_gt=coco_gt,
                    results=coco_results,
                    image_ids=image_ids,
                    category_ids=[class_id],
                    title=f"COCO segm metrics | {model_variant_name} | class={class_name}",
                )


if __name__ == "__main__":
    main()