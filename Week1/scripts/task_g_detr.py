import argparse
import torch
import wandb
import albumentations as A

from Week1.src.utils.dataset_hf import KittyDataset
from transformers import AutoModelForObjectDetection,AutoImageProcessor

from transformers import TrainingArguments

from Week1.src.utils.evaluate import MAPEvaluator

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

os.environ["CUDA_VISIBLE_DEVICES"]="2"

def get_args():
    parser = argparse.ArgumentParser(description="Fine-tune Faster R-CNN on DeArt")
    
    parser.add_argument('--unfreeze_mode', type=str, default='none', 
                        choices=['none', 'partial', 'all'],
                        help="none: backbone frozen, partial: last 2 blocks unfrozen, all: full model")
    
    parser.add_argument('--optimizer', type=str, default='sgd', 
                        choices=['sgd', 'adamw', 'adam_step'],
                        help="Choose optimizer configuration")
    
    parser.add_argument('--aug_level', type=str, default='light', 
                        choices=['light', 'medium', 'heavy'],
                        help="light: flip only, medium: basic spatial, heavy: noise/blur/dropout")
    parser.add_argument("--config_id",type=int, default=0)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=8)
    return parser.parse_args()

def get_transforms(level):
    # Notice we DO NOT resize or ToTensor here. HF ImageProcessor handles that.
    # We just do spatial/color augmentations.
    base_augs = []
    
    if level == 'light':
        base_augs.extend([A.HorizontalFlip(p=0.5)])
    elif level == 'medium':
        base_augs.extend([
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.3)
        ])
    elif level == 'heavy':
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(p=0.5),
            A.OneOf([A.GaussNoise(), A.ISONoise()], p=0.3),
            A.OneOf([A.MotionBlur(), A.GaussianBlur()], p=0.2),
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.5),

        ])

    
    return A.Compose(base_augs, bbox_params=A.BboxParams(format='coco', label_fields=['class_labels'], clip=True, min_area=1, min_width=1, min_height=1))

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = "PekingU/rtdetr_v2_r50vd"
    image_size = 480
    
    processor = AutoImageProcessor.from_pretrained(checkpoint,do_resize=True,
    size={"width": image_size, "height": image_size},
    use_fast=True,
    )
    
    id2label = {0: "car", 1: "person"}
    label2id = {"car": 0, "person": 1}
    model = AutoModelForObjectDetection.from_pretrained(
        checkpoint,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    model.to(device)
    def collate_fn(batch):
        data = {}
        data["pixel_values"] = torch.stack([x["pixel_values"] for x in batch])
        data["labels"] = [x["labels"] for x in batch]
        return data


    # 3. Datasets
    train_transform = get_transforms(args.aug_level)
    val_transform = A.Compose(
    [A.NoOp()],
    bbox_params=A.BboxParams(format="coco", label_fields=['class_labels'], clip=True, min_area=1, min_width=1, min_height=1),
    )   
    
    dataset = KittyDataset("/home/msiau/data/tmp/agarciat/MCVC/C5/KITTI-MOTS",processor, mode="train", transform=train_transform)
    val_dataset = KittyDataset("/home/msiau/data/tmp/agarciat/MCVC/C5/KITTI-MOTS",processor, mode="val", transform=val_transform)
    
    


    wandb.init(project="kitty-rtdetr-hf", config=vars(args))



    training_args = TrainingArguments(
        output_dir=f"rtdetr-v2-r50-cppe5-finetune-2_{args.config_id}",
        num_train_epochs=40,
        max_grad_norm=0.1,
        learning_rate=5e-5,
        warmup_steps=300,
        per_device_train_batch_size=3,
        dataloader_num_workers=2,
        metric_for_best_model="eval_map",
        greater_is_better=True,
        load_best_model_at_end=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        remove_unused_columns=False,
        eval_do_concat_batches=False,
        report_to="wandb",  # or "wandb"
    )
    
    from transformers import Trainer
    eval_compute_metrics_fn = MAPEvaluator(image_processor=processor, threshold=0.01, id2label=id2label)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=val_dataset,
        processing_class=processor,
        data_collator=collate_fn,
        compute_metrics=eval_compute_metrics_fn,
    )

    trainer.train()
    
    from pprint import pprint

    metrics = trainer.evaluate(eval_dataset=val_dataset, metric_key_prefix="eval")
    pprint(metrics)
    
    
if __name__ == "__main__":
    main()