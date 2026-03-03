from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


KITTI_CLASS_TO_COCO = {1: 3, 2: 1}  # 1=car->COCO 3, 2=pedestrian->COCO 1
KITTI_CLASS_TO_TRAIN_ID = {1: 0, 2: 1}  # contiguous IDs for fine-tuning


def list_kitti_rgb_images(kitti_root: Path, split: str, seqs: Optional[List[str]] = None) -> List[Path]:
    img_root = kitti_root / split / "image_02"
    if not img_root.exists():
        raise FileNotFoundError(f"Missing: {img_root}")

    seq_dirs = sorted([p for p in img_root.iterdir() if p.is_dir()])
    if seqs is not None:
        seqs_set = set(seqs)
        seq_dirs = [p for p in seq_dirs if p.name in seqs_set]

    paths: List[Path] = []
    for sd in seq_dirs:
        paths.extend(sorted(sd.glob("*.png")))
    return paths


def rel_from_split(p: Path, kitti_root: Path, split: str) -> Path:
    base = kitti_root / split
    try:
        return p.relative_to(base)
    except Exception:
        return Path(p.name)


def list_training_pairs(kitti_root: Path, seqs: Optional[List[str]] = None) -> List[Tuple[str, Path, Path]]:
    inst_root = kitti_root / "instances"
    rgb_root = kitti_root / "training" / "image_02"

    if not inst_root.exists():
        raise FileNotFoundError(f"Missing instances folder: {inst_root}")
    if not rgb_root.exists():
        raise FileNotFoundError(f"Missing training RGB folder: {rgb_root}")

    seq_dirs = sorted([p for p in inst_root.iterdir() if p.is_dir()])
    if seqs is not None:
        seqs_set = set(seqs)
        seq_dirs = [p for p in seq_dirs if p.name in seqs_set]

    pairs = []
    for seq_dir in seq_dirs:
        seq = seq_dir.name
        rgb_seq = rgb_root / seq
        if not rgb_seq.exists():
            continue

        for mask_path in sorted(seq_dir.glob("*.png")):
            rgb_path = rgb_seq / mask_path.name
            if rgb_path.exists():
                pairs.append((seq, rgb_path, mask_path))

    return pairs


def extract_gt_from_instance_png(
    mask_path: Path,
    class_mapping: Dict[int, int],
) -> List[Tuple[int, List[float]]]:
    mask = np.array(Image.open(mask_path), dtype=np.int32)
    ids = np.unique(mask)
    ids = ids[ids != 0]

    gt = []
    for inst_id in ids:
        class_id = int(inst_id // 1000)
        if class_id not in class_mapping:
            continue

        mapped_cat = class_mapping[class_id]
        ys, xs = np.where(mask == inst_id)
        if len(xs) == 0:
            continue

        x1, y1 = float(xs.min()), float(ys.min())
        x2, y2 = float(xs.max()) + 1.0, float(ys.max()) + 1.0
        w, h = x2 - x1, y2 - y1
        if w <= 1.0 or h <= 1.0:
            continue

        gt.append((mapped_cat, [x1, y1, w, h]))
    return gt


def sanitize_coco_bboxes(
    bboxes: List[List[float]],
    cats: List[int],
    img_w: int,
    img_h: int,
    min_size_px: float = 2.0,
) -> Tuple[List[int], List[List[float]]]:
    clean_bboxes, clean_cats = [], []
    for bb, cat in zip(bboxes, cats):
        if bb is None or len(bb) != 4:
            continue
        x, y, w, h = map(float, bb)
        if w <= 0 or h <= 0:
            continue

        x1, y1, x2, y2 = x, y, x + w, y + h
        if x2 <= 0 or y2 <= 0 or x1 >= img_w or y1 >= img_h:
            continue

        x1 = max(0.0, min(x1, img_w - 1.0))
        y1 = max(0.0, min(y1, img_h - 1.0))
        x2 = max(0.0, min(x2, float(img_w)))
        y2 = max(0.0, min(y2, float(img_h)))

        nw, nh = x2 - x1, y2 - y1
        if nw < min_size_px or nh < min_size_px:
            continue

        clean_bboxes.append([x1, y1, nw, nh])
        clean_cats.append(int(cat))

    return clean_cats, clean_bboxes


def format_annotations_for_processor(
    image_id: int,
    cats: List[int],
    bboxes_xywh: List[List[float]],
) -> Dict[str, Any]:
    anns = []
    for cat, bb in zip(cats, bboxes_xywh):
        x, y, w, h = map(float, bb)
        anns.append({
            "image_id": int(image_id),
            "category_id": int(cat),
            "bbox": [x, y, w, h],
            "area": float(w * h),
            "iscrowd": 0,
        })
    return {"image_id": int(image_id), "annotations": anns}


def build_coco_gt_from_pairs(
    pairs: List[Tuple[str, Path, Path]],
    class_mapping: Dict[int, int],
    categories: List[Dict],
) -> Tuple[COCO, List[int]]:
    images, annotations = [], []
    ann_id = 1

    for img_id, (seq, rgb_path, mask_path) in enumerate(pairs, start=1):
        with Image.open(rgb_path) as im:
            w, h = im.size
        images.append({"id": img_id, "file_name": str(rgb_path), "width": w, "height": h})

        gt_objs = extract_gt_from_instance_png(mask_path, class_mapping)
        for cat, bb in zip([g[0] for g in gt_objs], [g[1] for g in gt_objs]):
            x, y, bw, bh = map(float, bb)
            if bw <= 1.0 or bh <= 1.0:
                continue
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": int(cat),
                "bbox": [x, y, bw, bh],
                "area": float(bw * bh),
                "iscrowd": 0,
            })
            ann_id += 1

    gt_dict = {"images": images, "annotations": annotations, "categories": categories}
    coco_gt = COCO()
    coco_gt.dataset = gt_dict
    coco_gt.createIndex()
    img_ids = [img["id"] for img in images]
    return coco_gt, img_ids


def coco_eval_bbox(
    coco_gt: COCO,
    preds: List[Dict],
    img_ids: List[int],
    cat_ids: List[int],
    title: str,
):
    if len(preds) == 0:
        print(f"\n[{title}] No predictions, cannot evaluate.")
        return None

    coco_dt = coco_gt.loadRes(preds)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.params.imgIds = img_ids
    coco_eval.params.catIds = cat_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    print(f"\n===== {title} =====")
    coco_eval.summarize()
    return coco_eval.stats


@torch.no_grad()
def run_model_preds_coco(
    model,
    processor,
    pairs: List[Tuple[str, Path, Path]],
    device: torch.device,
    threshold: float,
    keep_cat_ids: set,
) -> List[Dict[str, Any]]:
    model.eval()
    preds = []

    for img_id, (seq, rgb_path, mask_path) in tqdm(
        list(enumerate(pairs, start=1)), desc="Predict", total=len(pairs),
    ):
        img = Image.open(rgb_path).convert("RGB")
        inputs = processor(images=img, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)

        target_sizes = torch.tensor([img.size[::-1]], device=device)
        results = processor.post_process_object_detection(
            outputs, threshold=threshold, target_sizes=target_sizes,
        )[0]

        boxes = results["boxes"].detach().cpu().numpy()
        scores = results["scores"].detach().cpu().numpy()
        labels = results["labels"].detach().cpu().numpy()

        for (x1, y1, x2, y2), s, lab in zip(boxes, scores, labels):
            lab = int(lab)
            if lab not in keep_cat_ids:
                continue
            w, h = float(x2 - x1), float(y2 - y1)
            if w <= 1.0 or h <= 1.0:
                continue
            preds.append({
                "image_id": int(img_id),
                "category_id": lab,
                "bbox": [float(x1), float(y1), w, h],
                "score": float(s),
            })
    return preds
