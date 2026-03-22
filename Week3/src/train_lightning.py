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
from text_tokenizers import get_tokenizer
import wandb

# Global seed for reproducibility
SEED = 42


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Set seed for reproducibility
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    # Initialize tokenizer
    print(f"Initializing {args.tokenizer_type} tokenizer...")
    tokenizer = get_tokenizer(tokenizer_type='word')

    # Load training dataset to build vocabulary
    print("Loading training dataset...")
    full_dataset = VizWizDataset(data_root=Path(args.data_root), split='train', tokenizer=None)

    # Build vocabulary from training captions
    print("Building vocabulary from training data...")
    train_captions = full_dataset.get_all_captions()
    tokenizer.build_vocab(train_captions)
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Max sequence length: {tokenizer.max_len}")

    # Now create datasets with the tokenizer
    print("Creating datasets with tokenizer...")
    dataset = VizWizDataset(data_root=Path(args.data_root), split='train', tokenizer=tokenizer)
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
    model = Baseline(
        tokenizer=tokenizer,
        device=device,
        resnet_model=args.resnet_model,
        rnn_type=args.rnn_type,
        freeze_encoder=(args.freeze_encoder == "yes"),
        attention=(args.attention == "yes")
    )
    module = TrainWrapper(
        model=model,
        tokenizer=tokenizer,
        learning_rate=0.0003070136460705484,
        teacher_forcing_ratio=0.25336340759892617,
        batch_size=args.batch_size,
        optimizer_type='adamw',
        scheduler_type="cosine"
    )

    early_stopping = EarlyStopping(
        monitor='val/loss',
        patience=5,
        mode='min',
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output_dir,
        filename=f'best_{args.run_name}',
        monitor='val/loss',
        mode='min',
        save_top_k=1,
    )

    run = wandb.init(
        entity="C5-Team3",
        project="Captioning-Week3",
        name="Hyperparameter3",
        config=vars(args)
    )

    trainer = L.Trainer(
        max_epochs=args.epochs,
        check_val_every_n_epoch=args.val_every,
        callbacks=[checkpoint_callback, early_stopping],
        logger=WandbLogger(experiment=run),
        log_every_n_steps=1
    )

    trainer.fit(module, train_loader, val_loader)
    print(f"\nBest model saved to: {checkpoint_callback.best_model_path}")

    del train_loader
    del val_loader

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Image Captioning Model")

    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--tokenizer_type", type=str, default="word",
                        choices=["character", "word", "subword"],
                        help="Tokenizer type: character, word, or subword (BERT)")
    parser.add_argument("--resnet_model", type=str, default="microsoft/resnet-34")
    parser.add_argument("--rnn_type", type=str, default="GRU", choices=["GRU", "LSTM"])
    parser.add_argument("--freeze_encoder", type=str, required=True, choices=["yes", "no"], help="Freeze encoder (ResNet) weights during training")
    parser.add_argument("--attention", type=str, required=True, choices=["yes", "no"], help="Use Bahdanau attention mechanism")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--teacher_forcing_ratio", type=float, default=0)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--val_every", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=3)
    parser.add_argument("--run_name", type=str, default="Baseline", help="Name of the W&B run")
    parser.add_argument("--optimizer", type=str, default="Adam", choices=["Adam", "SGD", "AdamW"], help="Optimizer type")

    args = parser.parse_args()
    main(args)
