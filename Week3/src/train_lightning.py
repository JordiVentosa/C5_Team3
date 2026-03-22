import torch
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import argparse
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from models.baseline import Baseline
from models.train_wrapper import *
from custom_datasets.vizwiz_dataset import VizWizDataset
import wandb

# Global seed for reproducibility
SEED = 42


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Set seed for reproducibility
    torch.manual_seed(SEED)

    print("Loading datasets...")
    dataset = VizWizDataset(data_root=Path(args.data_root), split='train')
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size],
                                               generator=torch.Generator().manual_seed(SEED))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=(device == 'cuda'), persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=(device == 'cuda'), persistent_workers=True)

    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    print("Creating model...")
    model = Baseline(device=device, resnet_model=args.resnet_model, rnn_type=args.rnn_type)
    module = TrainWrapper(
        model=model,
        learning_rate=args.learning_rate,
        teacher_forcing_ratio=args.teacher_forcing_ratio,
    )

    early_stopping = EarlyStopping(
        monitor='val/loss',
        patience=5,
        mode='min',
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output_dir,
        filename='best_model_test',
        monitor='val/loss',
        mode='min',
        save_top_k=1,
    )

    run = wandb.init(
        entity="C5-Team3",
        project="Captioning-Week3",
        name=args.run_name, 
        config=vars(args)
    )

    trainer = L.Trainer(
        max_epochs=args.epochs,
        check_val_every_n_epoch=args.val_every,
        callbacks=[checkpoint_callback, early_stopping],
        logger=WandbLogger(experiment=run),
    )

    trainer.fit(module, train_loader, val_loader)
    print(f"\nBest model saved to: {checkpoint_callback.best_model_path}")

    del train_loader
    del val_loader

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Image Captioning Model")

    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--resnet_model", type=str, default="microsoft/resnet-18")
    parser.add_argument("--rnn_type", type=str, default="GRU", choices=["GRU", "LSTM"])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--teacher_forcing_ratio", type=float, default=0)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--val_every", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=3)
    parser.add_argument("--run_name", type=str, default="Baseline", help="Name of the W&B run")

    args = parser.parse_args()
    main(args)