import copy

import torchvision
import torch
import wandb
from torch.utils.data import DataLoader
from Week1.src.utils.dataset import KittyDataset
import pycocotools.mask as mask_util
from torchvision.ops import box_convert
import numpy as np
import cv2
import random

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision.ops import box_convert
import json, io
from torchvision.datasets import CocoDetection

import albumentations as A
from albumentations.pytorch import ToTensorV2

from tqdm import tqdm


CLASS_NAMES = {1: "person", 3: "car"}
COLORS = {1: (0, 255, 0), 3: (255, 0, 0)}  # person=green, car=red
def draw_boxes(img, boxes, labels, color, scores=None, threshold=0.5):
    """Draw bounding boxes on image in-place."""
    for i, (box, label) in enumerate(zip(boxes, labels)):
        if scores is not None and scores[i] < threshold:
            continue
        x1, y1, x2, y2 = map(int, box)
        class_name = CLASS_NAMES.get(label.item(), "unknown")
        text = f"{class_name}" if scores is None else f"{class_name}: {scores[i]:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return img


def log_predictions_to_wandb(model, dataset, device, num_images=10, threshold=0.5,random=False):
    model.eval()
    if random:
        
        indices = random.sample(range(len(dataset)), min(num_images, len(dataset)))
    else:
        indices = range(0,len(dataset),len(dataset)//num_images)
        
    wandb_images = []
    wandb_images_overlap = []
    with torch.no_grad():
        for idx in indices:
            img_tensor, target = dataset[idx]
            pred = model([img_tensor.to(device)])[0]

            # Base image for side-by-side panels
            base_img = np.clip(img_tensor.permute(1, 2, 0).cpu().numpy(),0,1)
            base_img = (base_img * 255).astype(np.uint8)

            # GT panel (green)
            gt_img = draw_boxes(
                base_img.copy(), target["boxes"], target["labels"],
                color=(0, 255, 0)
            )

            # Prediction panel (red)
            pred_img = draw_boxes(
                base_img.copy(), pred["boxes"].cpu(), pred["labels"].cpu(),
                color=(0, 0, 255), scores=pred["scores"].cpu(), threshold=threshold
            )

            # Overlay panel: GT green + predictions red on the same image
            overlay_img = draw_boxes(
                base_img.copy(), target["boxes"], target["labels"],
                color=(0, 255, 0)
            )
            overlay_img = draw_boxes(
                overlay_img, pred["boxes"].cpu(), pred["labels"].cpu(),
                color=(0, 0, 255), scores=pred["scores"].cpu(), threshold=threshold
            )

            # Stack all three side by side
            combined = np.concatenate([gt_img, pred_img], axis=1)
            wandb_images.append(wandb.Image(
                combined,
                caption=f"GT (green) | Pred (red)  — idx={idx}"
            ))
            wandb_images_overlap.append(wandb.Image(
                overlay_img,
                caption=f"Overlay — idx={idx}"
            ))

    wandb.log({"predictions": wandb_images})
    wandb.log({"overlappeded_predictions": wandb_images_overlap})
    model.train()

def evaluate_coco(model, data_loader, device):
    model.eval()
    dataset = data_loader.dataset

    # Build COCO ground truth from KittyDataset
    coco_gt_dict = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "person"}, {"id": 3, "name": "car"}]
    }
    ann_id = 1
    for idx in range(len(dataset)):
        img, target = dataset[idx]
        image_id = target["image_id"].item()
        h, w = img.shape[-2], img.shape[-1]
        coco_gt_dict["images"].append({"id": image_id, "height": h, "width": w})
        boxes_xyxy = target["boxes"]
        if len(boxes_xyxy) > 0:
            boxes_xywh = box_convert(boxes_xyxy, in_fmt="xyxy", out_fmt="xywh")
            for box, label in zip(boxes_xywh, target["labels"]):
                coco_gt_dict["annotations"].append({
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": label.item(),
                    "bbox": box.tolist(),
                    "area": (box[2] * box[3]).item(),
                    "iscrowd": 0
                })
                ann_id += 1


    # Load into COCO object
    coco_gt = COCO()
    coco_gt.dataset = coco_gt_dict
    coco_gt.createIndex()

    results = []
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(tqdm(data_loader, desc="Evaluating")):
            images = [img.to(device) for img in images]
            outputs = model(images)

            # Recover image_ids based on batch position
            batch_size = len(images)
            start_idx = batch_idx * data_loader.batch_size

            for i, (output, target) in enumerate(zip(outputs, targets)):
                image_id = target["image_id"].item()  # use the real ID
                boxes = output["boxes"].cpu()
                scores = output["scores"].cpu()
                labels = output["labels"].cpu()
                boxes_xywh = box_convert(boxes, in_fmt="xyxy", out_fmt="xywh")

                for box, score, label in zip(boxes_xywh, scores, labels):
                    results.append({
                        "image_id": image_id,
                        "category_id": label.item(),
                        "bbox": box.tolist(),
                        "score": score.item(),
                    })

    if len(results) == 0:
        model.train()
        return None
    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    metric_names = [
        "AP@[IoU=0.50:0.95]", "AP@[IoU=0.50]", "AP@[IoU=0.75]",
        "AP@[small]", "AP@[medium]", "AP@[large]",
        "AR@[max=1]", "AR@[max=10]", "AR@[max=100]",
        "AR@[small]", "AR@[medium]", "AR@[large]",
    ]
    coco_metrics = {f"coco_{name}": coco_eval.stats[i] for i, name in enumerate(metric_names)}

    model.train()
    return coco_metrics
