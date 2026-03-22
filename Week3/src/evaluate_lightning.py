import torch
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
import lightning as L

from models.baseline import Baseline
from models.train_wrapper import TrainWrapper
from custom_datasets.vizwiz_dataset import VizWizDataset
from text_tokenizers import get_tokenizer
import wandb
from lightning.pytorch.loggers import WandbLogger

SEED = 42


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(SEED)

    # ── 1. Rebuild tokenizer from training data (same as train.py) ─────────────
    print("Rebuilding tokenizer from training data...")
    tokenizer = get_tokenizer(tokenizer_type="character")
    train_dataset_raw = VizWizDataset(
        data_root=Path(args.data_root), split="train", tokenizer=None
    )
    tokenizer.build_vocab(train_dataset_raw.get_all_captions())
    print(f"Vocab size: {tokenizer.vocab_size}")

    # ── 2. Load the eval dataset ───────────────────────────────────────────────
    print(f"Loading '{args.split}' split...")
    eval_dataset = VizWizDataset(
        data_root=Path(args.data_root), split=args.split, tokenizer=tokenizer
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=True,
    )
    print(f"  {len(eval_dataset)} samples")

    # ── 3. Load model from checkpoint ─────────────────────────────────────────
    print(f"Loading checkpoint: {args.checkpoint}")
    model = Baseline(
        tokenizer=tokenizer,
        device=device,
        resnet_model="microsoft/resnet-18",
        rnn_type="GRU",
        freeze_encoder=False,
    )

    checkpoint = torch.load(args.checkpoint)

    state_dict = checkpoint["state_dict"]

    state_dict = {k.removeprefix("model."): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)

    module = TrainWrapper(
        model=model,
        tokenizer=tokenizer
    )

    run = wandb.init(
        entity="C5-Team3",
        project="Captioning-Week3",
        name="Hyperparameter",
        config=vars(args)
    )

    # ── 4. Run evaluation via Lightning trainer ────────────────────────────────
    trainer = L.Trainer(
        logger=WandbLogger(experiment=run),
    )

    results = trainer.test(module, dataloaders=eval_loader)

    print("\nResults:", results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Image Captioning Model")

    parser.add_argument("--checkpoint",  type=str, required=True, help="Path to .ckpt file saved by ModelCheckpoint")
    parser.add_argument("--data_root",   type=str, default="./data")
    parser.add_argument("--split",       type=str, default="val")
    parser.add_argument("--batch_size",  type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()
    main(args)