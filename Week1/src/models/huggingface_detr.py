import torch
from pathlib import Path
from typing import Dict, Optional
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoImageProcessor, AutoModelForObjectDetection


def load_font(size: int = 14) -> ImageFont.ImageFont:
    for fp in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]:
        try:
            if Path(fp).exists():
                return ImageFont.truetype(fp, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


def draw_detections(
    image: Image.Image,
    detections: Dict[str, torch.Tensor],
    id2label: Dict[int, str],
    keep_labels: Optional[set] = None,
) -> Image.Image:
    img = image.copy()
    draw = ImageDraw.Draw(img)
    font = load_font(14)

    boxes = detections["boxes"].tolist()
    scores = detections["scores"].tolist()
    labels = detections["labels"].tolist()

    for (x1, y1, x2, y2), s, lab in zip(boxes, scores, labels):
        name = id2label.get(int(lab), str(lab))
        if keep_labels and name not in keep_labels:
            continue

        draw.rectangle([x1, y1, x2, y2], width=3)
        txt = f"{name} {s:.2f}"

        try:
            tw, th = draw.textsize(txt, font=font)
        except Exception:
            bbox = draw.textbbox((0, 0), txt, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

        ty1 = max(0, y1 - th - 4)
        draw.rectangle([x1, ty1, x1 + tw + 6, ty1 + th + 4], fill="black")
        draw.text((x1 + 3, ty1 + 2), txt, fill="white", font=font)

    return img


def load_detr_model(checkpoint: str, device: torch.device, id2label=None, label2id=None, use_fast=False):
    processor = AutoImageProcessor.from_pretrained(checkpoint, use_fast=use_fast)

    model_kwargs = {}
    if id2label is not None:
        model_kwargs["id2label"] = id2label
        model_kwargs["label2id"] = label2id
        model_kwargs["ignore_mismatched_sizes"] = True

    model = AutoModelForObjectDetection.from_pretrained(checkpoint, **model_kwargs).to(device)
    model.eval()
    return processor, model


@torch.no_grad()
def run_inference(model, processor, image: Image.Image, device: torch.device, threshold: float = 0.5):
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]], device=device)
    results = processor.post_process_object_detection(
        outputs, threshold=threshold, target_sizes=target_sizes
    )[0]
    return results
