import torchvision
import torch
import wandb
from torch.utils.data import DataLoader
from Week1.src.utils.dataset import KittyDataset

import albumentations as A
from albumentations.pytorch import ToTensorV2

train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.GaussianBlur(p=0.1),
    # Normalize and convert to Tensor (ToTensorV2 handles the HWC -> CHW swap)
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))


def collate_fn(batch):
    return tuple(zip(*batch))


dataset = KittyDataset(root_dir="/home/msiau/workspace/jventosa/PostTFG/Master/C5_Team3/Week1/datasets/KITTI-MOTS")#,transform=train_transform)

loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    collate_fn=collate_fn
)

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
    weights="DEFAULT"
)

params = [p for p in model.parameters() if p.requires_grad]

optimizer = torch.optim.AdamW(
    params,
    lr=1e-4
)

for param in model.backbone.parameters():
    param.requires_grad = False
    
    from tqdm import tqdm

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

num_epochs = 10

lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=3,
    gamma=0.1
)

wandb.init(
    project="kitty-fasterrcnn",
    config={
        "learning_rate": 0.005,
        "batch_size": 4,
        "epochs": 10,
        "optimizer": "SGD"
    }
)
import wandb
import numpy as np
import cv2
import torch
import random

# Define class names for visualization
CLASS_NAMES = {1: "person", 3: "car"}
COLORS = {1: (0, 255, 0), 3: (255, 0, 0)}  # person=green, car=red

def draw_boxes(image_tensor, boxes, labels, scores=None, threshold=0.5):
    """Draw bounding boxes on image and return annotated image."""
    img = image_tensor.permute(1, 2, 0).cpu().numpy()
    img = (img * 255).astype(np.uint8).copy()
    
    for i, (box, label) in enumerate(zip(boxes, labels)):
        if scores is not None and scores[i] < threshold:
            continue
        x1, y1, x2, y2 = map(int, box)
        color = COLORS.get(label.item(), (255, 255, 255))
        class_name = CLASS_NAMES.get(label.item(), "unknown")
        text = f"{class_name}" if scores is None else f"{class_name}: {scores[i]:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return img

def log_predictions_to_wandb(model, dataset, device, num_images=10, threshold=0.5):
    """Sample images from dataset, run inference, and log GT vs predictions to WandB."""
    model.eval()
    indices = random.sample(range(len(dataset)), min(num_images, len(dataset)))
    wandb_images = []

    with torch.no_grad():
        for idx in indices:
            img_tensor, target = dataset[idx]
            
            # Run inference
            pred = model([img_tensor.to(device)])[0]
            
            # Draw ground truth
            gt_img = draw_boxes(
                img_tensor,
                target["boxes"],
                target["labels"],
                scores=None
            )
            
            # Draw predictions
            pred_img = draw_boxes(
                img_tensor,
                pred["boxes"].cpu(),
                pred["labels"].cpu(),
                scores=pred["scores"].cpu(),
                threshold=threshold
            )
            
            # Stack GT and prediction side by side
            combined = np.concatenate([gt_img, pred_img], axis=1)
            wandb_images.append(wandb.Image(combined, caption=f"Left: GT | Right: Pred (idx={idx})"))

    wandb.log({"predictions": wandb_images})
    model.train()


for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for images, targets in tqdm(loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        epoch_loss += losses.item()
        wandb.log({
            "batch_loss": losses.item(),
            **{f"loss_{k}": v.item() for k, v in loss_dict.items()}
        })

    lr_scheduler.step()
    avg_loss = epoch_loss / len(loader)
    wandb.log({
        "epoch": epoch,
        "epoch_loss": avg_loss,
        "learning_rate": optimizer.param_groups[0]["lr"]
    })
    print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")

    # Log 10 images with GT and predictions to WandB
    log_predictions_to_wandb(model, dataset, device, num_images=10, threshold=0.5)