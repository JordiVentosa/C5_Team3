import torch
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
from tqdm import tqdm

from models.baseline import Baseline
from models.train_wrapper import CaptioningModule
from datasets.vizwiz_dataset import VizWizDataset


def train_one_epoch(module, dataloader, epoch):
    total_loss = 0.0
    num_batches = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch in progress_bar:
        images, captions, _ = batch
        metrics = module.training_step((images, captions))
        loss = metrics['loss']
        total_loss += loss
        num_batches += 1
        progress_bar.set_postfix({'loss': f'{loss:.4f}'})
    
    return total_loss / num_batches


def validate(module, dataloader):
    total_loss = 0.0
    num_batches = 0
    all_predictions = []
    all_references = []
    progress_bar = tqdm(dataloader, desc="Validating")
    
    for batch in progress_bar:
        images, captions, caption_texts = batch
        metrics = module.validation_step((images, captions))
        loss = metrics['loss']
        total_loss += loss
        num_batches += 1
        
        predictions = module.predict(images)
        all_predictions.extend(predictions)
        all_references.extend(caption_texts)
        progress_bar.set_postfix({'loss': f'{loss:.4f}'})
    
    avg_loss = total_loss / num_batches
    metrics = module.compute_metrics(all_predictions, all_references)
    
    return avg_loss, metrics


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    print("Loading datasets...")
    train_dataset = VizWizDataset(data_root=Path(args.data_root), split='train')
    val_dataset = VizWizDataset(data_root=Path(args.data_root), split='test')
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                              num_workers=args.num_workers, pin_memory=(device == 'cuda'))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=(device == 'cuda'))
    
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    
    print("Creating model...")
    model = Baseline(device=device, resnet_model=args.resnet_model)
    module = CaptioningModule(
        model=model,
        learning_rate=args.learning_rate,
        device=device,
        teacher_forcing_ratio=args.teacher_forcing_ratio
    )
    
    print(f"\nStarting training for {args.epochs} epochs...")
    best_val_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*60}\nEpoch {epoch}/{args.epochs}\n{'='*60}")
        
        train_loss = train_one_epoch(module, train_loader, epoch)
        print(f"\nTrain Loss: {train_loss:.4f}")
        
        if epoch % args.val_every == 0:
            val_loss, val_metrics = validate(module, val_loader)
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val Metrics: {module.metric.format_metrics(val_metrics)}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = Path(args.output_dir) / "best_model.pth"
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                module.save_checkpoint(str(checkpoint_path))
                print(f"✓ Saved best model to {checkpoint_path}")
        
        if epoch % args.save_every == 0:
            checkpoint_path = Path(args.output_dir) / f"checkpoint_epoch_{epoch}.pth"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            module.save_checkpoint(str(checkpoint_path))
            print(f"✓ Saved checkpoint to {checkpoint_path}")
    
    print(f"\n{'='*60}\nTraining completed!\nBest validation loss: {best_val_loss:.4f}\n{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Image Captioning Model")

    parser.add_argument("--data_root", type=str, default="../data", help="Root directory of VizWiz dataset")
    parser.add_argument("--resnet_model", type=str, default="microsoft/resnet-18", help="ResNet model for encoder")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--teacher_forcing_ratio", type=float, default=0.0, help="Teacher forcing ratio (0.0 = no teacher forcing)")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Checkpoint directory")
    parser.add_argument("--val_every", type=int, default=1, help="Validate every N epochs")
    parser.add_argument("--save_every", type=int, default=5, help="Save checkpoint every N epochs")

    args = parser.parse_args()
    main(args)