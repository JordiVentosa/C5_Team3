import torch
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from models.baseline import Baseline
from models.train_wrapper import TrainWrapper
from custom_datasets.vizwiz_dataset import VizWizDataset
from text_tokenizers import get_tokenizer
import wandb

# ── Fixed settings ────────────────────────────────────────────────────────────
SEED        = 42
DATA_ROOT   = "./data"
OUTPUT_DIR  = "./checkpoints"
NUM_WORKERS = 8
EPOCHS      = 50
VAL_EVERY   = 1
BATCH_SIZE = 128
ENCODER = "microsoft/resnet-34"
DECODER = "GRU"
TOKEN_LEVEL = "word"
# ─────────────────────────────────────────────────────────────────────────────


def train():
    run = wandb.init()
    cfg = wandb.config

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    print(f"Initializing {TOKEN_LEVEL} tokenizer...")
    tokenizer = get_tokenizer(tokenizer_type=TOKEN_LEVEL)

    print("Loading training dataset...")
    full_dataset = VizWizDataset(data_root=Path(DATA_ROOT), split="train", tokenizer=None)

    print("Building vocabulary from training data...")
    tokenizer.build_vocab(full_dataset.get_all_captions())
    print(f"Vocabulary size : {tokenizer.vocab_size}")
    print(f"Max sequence len: {tokenizer.max_len}")

    # ── Datasets & loaders ────────────────────────────────────────────────────
    dataset = VizWizDataset(data_root=Path(DATA_ROOT), split='train', tokenizer=tokenizer)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size],
                                               generator=torch.Generator().manual_seed(SEED))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=(device == 'cuda'), persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=(device == 'cuda'), persistent_workers=True)
    
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    print("Creating model...")
    model = Baseline(
        tokenizer=tokenizer,
        device=device,
        resnet_model=ENCODER,
        rnn_type=DECODER,
        freeze_encoder=False,
    )
    module = TrainWrapper(
        model=model,
        tokenizer=tokenizer,
        learning_rate=cfg.learning_rate,
        teacher_forcing_ratio=cfg.teacher_forcing_ratio,
        optimizer_type=cfg.optimizer,
        scheduler_type=cfg.scheduler,
        batch_size=BATCH_SIZE,
    )

    # ── Callbacks ─────────────────────────────────────────────────────────────

    early_stopping = EarlyStopping(monitor="val/loss", patience=5, mode="min")

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = L.Trainer(
        max_epochs=EPOCHS,
        check_val_every_n_epoch=VAL_EVERY,
        callbacks=[early_stopping],
        logger=WandbLogger(experiment=run),
        log_every_n_steps=1,
        enable_checkpointing=False,
    )

    trainer.fit(module, train_loader, val_loader)


if __name__ == "__main__":
    train()