import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import SamModel, SamProcessor
import numpy as np
from tqdm import tqdm
import argparse

from dataset import KittiMotsSamDataset
from augmentations import get_augmentations
from evaluators import SAMCOCOEvaluator
from collate import collate_fn_with_masks as collate_fn

from peft import LoraConfig, get_peft_model


parser = argparse.ArgumentParser()
parser.add_argument('--da_config', type=int, default=0, help='Data Augmentation Profile (0-3)')
args = parser.parse_args()


# Configuration
DA_CONFIG = args.da_config
USE_LORA = True
dataset_path = "/home/mcv/datasets/C5/KITTI-MOTS"
BATCH_SIZE = 8
EPOCHS = 50
LR = 1e-4 if USE_LORA else 1e-5
DEVICE = torch.device('cuda')
CPU_WORKERS= 8

class SAMLoss(nn.Module):
    def __init__(self, focal_weight=20.0, dice_weight=1.0, alpha=0.25, gamma=2.0, smooth=1e-5):
        super(SAMLoss, self).__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs_sigmoid = torch.sigmoid(inputs)
        
        # Dice loss
        inputs_flat = inputs_sigmoid.view(-1)
        targets_flat = targets.view(-1)
        intersection = (inputs_flat * targets_flat).sum()                            
        dice_loss = 1 - (2. * intersection + self.smooth) / (inputs_flat.sum() + targets_flat.sum() + self.smooth)  
        
        # Focal loss
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p_t = inputs_sigmoid * targets + (1 - inputs_sigmoid) * (1 - targets)
        focal_loss = bce_loss * ((1 - p_t) ** self.gamma)
        
        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_loss = alpha_t * focal_loss
            
        return (self.focal_weight * focal_loss.mean()) + (self.dice_weight * dice_loss)


def train():
    print(f"Loading processor and model (DA Config = {DA_CONFIG}, LoRA = {USE_LORA})...")
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    model = SamModel.from_pretrained("facebook/sam-vit-base")
    
    if USE_LORA:
        print("Applying LoRA to mask decoder...")
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"], 
            lora_dropout=0.05,
            bias="none",
            task_type=None
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        print("Freezing vision and prompt encoders...")
        for name, param in model.named_parameters():
            if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
                param.requires_grad_(False)
                
    model.to(DEVICE)
    
    # El optimizador solo cogerá las partes del modelo que tengan requires_grad=True
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    criterion = SAMLoss()
    
    print("Loading datasets...")
    train_transforms = get_augmentations(DA_CONFIG)
    train_dataset = KittiMotsSamDataset(root_dir=dataset_path, split='train', transforms=train_transforms)
    val_dataset = KittiMotsSamDataset(root_dir=dataset_path, split='val', transforms=None)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        drop_last=True, 
        collate_fn=lambda x: collate_fn(x, processor),
        num_workers=CPU_WORKERS,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        drop_last=False, 
        collate_fn=lambda x: collate_fn(x, processor),
        num_workers=CPU_WORKERS,
        pin_memory=True
    )

    best_map = -1.0
    lora_prefix = "lora_" if USE_LORA else ""
    models_dir = f"models/{lora_prefix}da_config_{DA_CONFIG}" 
    os.makedirs(models_dir, exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for batch in pbar:
            if batch is None: continue
            
            inputs, true_masks, orig_boxes, classes, image_ids, orig_images, valid_lengths = batch
            inputs_gpu = {k: v.to(DEVICE) for k, v in inputs.items()}
            
            optimizer.zero_grad()
            outputs = model(**inputs_gpu, multimask_output=False)
            
            post_processed_masks = processor.post_process_masks(
                outputs.pred_masks,
                original_sizes=inputs["original_sizes"].tolist(),
                reshaped_input_sizes=inputs["reshaped_input_sizes"].tolist(),
                binarize=False
            )

            loss = 0.0
            num_masks_total = 0

            for i in range(len(post_processed_masks)):
                valid_len = valid_lengths[i]
                b_true_masks = true_masks[i][:valid_len]
                batch_trues = torch.stack([torch.tensor(m, device=DEVICE, dtype=torch.float32) for m in b_true_masks])
                batch_preds = post_processed_masks[i][:valid_len].squeeze(1)
                
                num_obj = valid_len
                loss += criterion(batch_preds, batch_trues) * num_obj
                num_masks_total += num_obj

            loss = loss / num_masks_total
            loss.backward()
            optimizer.step()
                        
            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        evaluator = SAMCOCOEvaluator()
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
                if batch is None: continue
                
                inputs, true_masks, orig_boxes, classes, image_ids, orig_images, valid_lengths = batch
                inputs_gpu = {k: v.to(DEVICE) for k, v in inputs.items()}
                
                outputs = model(**inputs_gpu, multimask_output=False)
                predicted_masks_logits = outputs.pred_masks.squeeze(2)
                
                loss = 0.0
                num_masks_total = 0
                for i in range(len(predicted_masks_logits)):
                    valid_len = valid_lengths[i]
                    b_true_masks = true_masks[i][:valid_len]
                    batch_trues = torch.stack([torch.tensor(m, device=DEVICE, dtype=torch.float32) for m in b_true_masks])
                    batch_preds = predicted_masks_logits[i][:valid_len]
                    
                    if batch_preds.shape[-2:] != batch_trues.shape[-2:]:
                        batch_trues = F.interpolate(batch_trues.unsqueeze(0), size=batch_preds.shape[-2:], mode='nearest').squeeze(0)
                    
                    num_obj = valid_len
                    loss += criterion(batch_preds, batch_trues) * num_obj
                    num_masks_total += num_obj
                    
                val_loss += (loss / num_masks_total).item()
                
                post_processed_masks = processor.post_process_masks(
                    outputs.pred_masks,
                    original_sizes=inputs["original_sizes"].tolist(),
                    reshaped_input_sizes=inputs["reshaped_input_sizes"].tolist()
                )
                
                for i in range(len(image_ids)):
                    valid_len = valid_lengths[i] 
                    b_img_id = image_ids[i]
                    b_classes = classes[i][:valid_len] 
                    b_true_masks = true_masks[i][:valid_len]
                    img_shape = orig_images[i].shape[:2]
                    b_boxes = orig_boxes[i][:valid_len]
                    
                    evaluator.add_gt_batch(
                        image_id=b_img_id, 
                        image_size=img_shape, 
                        valid_boxes=b_boxes, 
                        valid_masks=b_true_masks, 
                        valid_classes=b_classes
                    )
                    
                    pred_m = torch.sigmoid(post_processed_masks[i].squeeze(1)).cpu().numpy()
                    pred_m = pred_m[:valid_len]
                    
                    evaluator.add_pred_batch(
                        image_id=b_img_id,
                        category_ids=b_classes,
                        pred_masks=pred_m
                    )
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"\nEpoch [{epoch+1}/{EPOCHS}] - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        evaluator.init_gt()
        stats = evaluator.evaluate()

        # GUARDADO DEL MODELO
        if stats is not None:
            current_map = stats[0] # AP @ IoU=0.50:0.95
            
            latest_dir = os.path.join(models_dir, "latest_model")
            model.save_pretrained(latest_dir)
            processor.save_pretrained(latest_dir)
            print(f"Latest model saved: {latest_dir}")
            
            if current_map > best_map:
                best_map = current_map
                best_dir = os.path.join(models_dir, "best_model")
                model.save_pretrained(best_dir)
                processor.save_pretrained(best_dir)
                print(f"Best model improved: {best_map:.4f}")
        
        torch.cuda.empty_cache()

if __name__ == '__main__':
    train()