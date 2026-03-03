import argparse
import copy
import torch
import torchvision
import wandb
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from torch.utils.data import DataLoader
from Week1.src.utils.dataset import KittyDataset
from Week1.src.utils.fine_tune_utils import log_predictions_to_wandb, evaluate_coco

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

os.environ["CUDA_VISIBLE_DEVICES"]="0"

def get_args():
    parser = argparse.ArgumentParser(description="Fine-tune Faster R-CNN on KITTI")
    
    parser.add_argument('--unfreeze_mode', type=str, default='none', 
                        choices=['none', 'partial', 'all'],
                        help="none: backbone frozen, partial: last 2 blocks unfrozen, all: full model")
    
    parser.add_argument('--optimizer', type=str, default='sgd', 
                        choices=['sgd', 'adamw', 'adam_step'],
                        help="Choose optimizer configuration")
    
    parser.add_argument('--aug_level', type=str, default='light', 
                        choices=['light', 'medium', 'heavy'],
                        help="light: flip only, medium: basic spatial, heavy: noise/blur/dropout")

    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=16)
    return parser.parse_args()

def get_transforms(level):
    if level == 'light':
        return A.Compose([A.HorizontalFlip(p=0.5), A.ToFloat(max_value=255.0), ToTensorV2()],
                         bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
    
    elif level == 'medium':
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.ToFloat(max_value=255.0),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
    
    elif level == 'heavy':
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(p=0.5),
            A.OneOf([A.GaussNoise(), A.ISONoise()], p=0.3),
            A.OneOf([A.MotionBlur(), A.GaussianBlur()], p=0.2),
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.5),
            A.ToFloat(max_value=255.0),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize Model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    
    # Unfreeeeeeeeeeeeeze
    if args.unfreeze_mode == 'none':
        for param in model.backbone.parameters():
            param.requires_grad = False
    elif args.unfreeze_mode == 'partial':
        for name, param in model.backbone.named_parameters():
            if "body.layer3" in name or "body.layer4" in name or "fpn" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    elif args.unfreeze_mode == 'all':
        for param in model.parameters():
            param.requires_grad = True

    model.to(device)

    # Optimizers tested
    params = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=0.0005)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(params, lr=args.lr)
    elif args.optimizer == 'adam_step':

        backbone_params = [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad]
        head_params = [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]
        optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': args.lr * 0.1},
            {'params': head_params, 'lr': args.lr}
        ])

    train_transform = get_transforms(args.aug_level)
    dataset = KittyDataset(root_dir="/home/msiau/data/tmp/agarciat/MCVC/C5/KITTI-MOTS" , mode="train", transform=train_transform)
    val_dataset = KittyDataset(root_dir="/home/msiau/data/tmp/agarciat/MCVC/C5/KITTI-MOTS" , mode="val")
    
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    wandb.init(project="kitty-fasterrcnn", config=vars(args))

    coco_metrics = evaluate_coco(model, val_loader, device)
    log_predictions_to_wandb(model, dataset, device, num_images=10, threshold=0.5)

    best_map = 0
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        
        if coco_metrics and coco_metrics["coco_AP@[IoU=0.50:0.95]"] > best_map:
            best_map = coco_metrics["coco_AP@[IoU=0.50:0.95]"]
            torch.save(model.state_dict(), f"best_model_{args.unfreeze_mode}.pth")

        wandb.log({"train_loss": train_loss/len(loader), **coco_metrics})

        for images, targets in tqdm(loader, desc=f"Epoch {epoch}"):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            train_loss += losses.item()

        coco_metrics = evaluate_coco(model, val_loader, device)
        log_predictions_to_wandb(model, dataset, device, num_images=10, threshold=0.5)
        
        if coco_metrics and coco_metrics["coco_AP@[IoU=0.50:0.95]"] > best_map:
            best_map = coco_metrics["coco_AP@[IoU=0.50:0.95]"]
            torch.save(model.state_dict(), f"best_model_{args.unfreeze_mode}.pth")
        
        

        wandb.log({"train_loss": train_loss/len(loader), **coco_metrics})

if __name__ == "__main__":
    main()