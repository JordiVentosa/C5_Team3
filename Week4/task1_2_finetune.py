"""
Task 1.2 - Fine-tune ViT-GPT2 on VizWiz

Fine-tune modes:
  backbone   - freeze the GPT-2 decoder, train only the ViT encoder
  captioner  - freeze the ViT encoder, train only the GPT-2 decoder
  all        - train the full model end-to-end

Use:
    python task1_2_finetune.py \
        --train_img_dir  data/vizwiz/train \
        --train_ann_file data/vizwiz/annotations/train.json \
        --val_img_dir    data/vizwiz/val \
        --val_ann_file   data/vizwiz/annotations/val.json \
        --output_dir     outputs/task1_2 \
        --finetune_mode  captioner \
        --epochs         5 \
        --lr             5e-5
"""

import argparse
import json
import sys
from pathlib import Path
from functools import partial
import random

import wandb
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torchvision import transforms as T
from transformers import (
    VisionEncoderDecoderModel,
    ViTImageProcessor,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent))
from src.dataset import VizWizDataset, collate_fn
from src.metrics import compute_metrics, print_metrics


# ---------------------------------------------------------------------------
# Collate for training  (adds tokenised labels)
# ---------------------------------------------------------------------------

def train_collate_fn(batch, tokenizer, max_target_length=64):
    """
    Like collate_fn but also tokenises one reference caption per image
    so the model can compute the cross-entropy loss internally.
    """
    pixel_values = torch.stack([item["pixel_values"] for item in batch])

    # Pick the first reference caption as the training target
    texts = [random.choice(item["captions"]) for item in batch]

    encoding = tokenizer(
        texts,
        padding="max_length",
        max_length=max_target_length,
        truncation=True,
        return_tensors="pt",
    )
    labels = encoding.input_ids
    # Replace padding token id with -100 so it is ignored in the loss
    labels[labels == tokenizer.pad_token_id] = -100

    return {
        "pixel_values": pixel_values,
        "labels": labels,
        "captions": [item["captions"] for item in batch],
        "image_paths": [item["image_path"] for item in batch],
    }


# ---------------------------------------------------------------------------
# Freeze helpers
# ---------------------------------------------------------------------------

def freeze_backbone(model: VisionEncoderDecoderModel):
    """Freeze the ViT encoder; only the GPT-2 decoder will be trained."""
    for param in model.encoder.parameters():
        param.requires_grad = False
    for param in model.decoder.parameters():
        param.requires_grad = True
    print("[INFO] Frozen: encoder (ViT backbone). Training: decoder (captioner).")


def freeze_captioner(model: VisionEncoderDecoderModel):
    """Freeze the GPT-2 decoder; only the ViT encoder will be trained."""
    for param in model.encoder.parameters():
        param.requires_grad = True
    for param in model.decoder.parameters():
        param.requires_grad = False
    print("[INFO] Frozen: decoder (captioner). Training: encoder (ViT backbone).")


def train_all(model: VisionEncoderDecoderModel):
    """Unfreeze everything."""
    for param in model.parameters():
        param.requires_grad = True
    print("[INFO] Training full model (encoder + decoder).")


FREEZE_FN = {
    "backbone":  freeze_backbone,   # train encoder only
    "captioner": freeze_captioner,  # train decoder only
    "all":       train_all,         # train everything
}


def train_one_epoch(model, dataloader, optimizer, scheduler, device, epoch,
                    use_wandb=False, step_offset=0):
    model.train()
    total_loss = 0.0

    for i, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch} [train]")):
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        if use_wandb:
            wandb.log({"train/loss_step": loss.item(), "step": step_offset + i})

    avg_loss = total_loss / len(dataloader)
    print(f"[INFO] Epoch {epoch} — avg train loss: {avg_loss:.4f}")
    if use_wandb:
        wandb.log({"train/loss_epoch": avg_loss, "epoch": epoch})
    return avg_loss



def generate_captions(model, tokenizer, dataloader, device, gen_kwargs):
    predictions, references, image_paths = [], [], []
    model.eval()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating captions"):
            pixel_values = batch["pixel_values"].to(device)

            output_ids = model.generate(pixel_values=pixel_values, **gen_kwargs)
            captions = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            captions = [cap.strip().lower() for cap in captions]

            predictions.extend(captions)
            references.extend(batch["captions"])
            image_paths.extend(batch["image_paths"])

    return predictions, references, image_paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune ViT-GPT2 on VizWiz")

    # Data
    parser.add_argument("--train_img_dir",  type=str, required=True,
                        help="Path to training images folder")
    parser.add_argument("--train_ann_file", type=str, required=True,
                        help="Path to training annotations JSON")
    parser.add_argument("--val_img_dir",    type=str, default=None,
                        help="Path to validation images folder (optional)")
    parser.add_argument("--val_ann_file",   type=str, default=None,
                        help="Path to validation annotations JSON (optional)")

    # Model
    parser.add_argument("--model_name", type=str,
                        default="nlpconnect/vit-gpt2-image-captioning",
                        help="HuggingFace model name or local path")
    parser.add_argument("--finetune_mode", type=str,
                        choices=["backbone", "captioner", "all"], default="captioner",
                        help=(
                            "backbone  → train ViT encoder only; "
                            "captioner → train GPT-2 decoder only; "
                            "all       → train full model"
                        ))

    # Training hyper-parameters
    parser.add_argument("--epochs",          type=int,   default=3)
    parser.add_argument("--lr",              type=float, default=None) #USELESS: DO NOT USE, is here for legacy reasons.
    parser.add_argument("--batch_size",      type=int,   default=16)
    parser.add_argument("--num_workers",     type=int,   default=4)
    parser.add_argument("--warmup_steps",    type=int,   default=1000)
    parser.add_argument("--max_target_len",  type=int,   default=64,
                        help="Max token length for target captions during training")
    parser.add_argument("--max_samples",     type=int,   default=None,
                        help="Limit samples (quick testing)")

    parser.add_argument("--output_dir", type=str, default="outputs/task1_2",
                        help="Directory to save model checkpoints and results")

    parser.add_argument("--wandb_project", type=str, default=None,
                        help="WandB project name. If not set, WandB logging is disabled.")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="WandB run name (optional).")

    # Early stopping
    parser.add_argument("--patience", type=int, default=None,
                        help="Early stopping patience (epochs without val improvement). "
                             "Requires --val_img_dir and --val_ann_file. Disabled if not set.")

    # Augmentations
    parser.add_argument("--augment", action="store_true",
                        help="Apply random augmentations to training images.")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    print("[INFO] Device check...", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}", flush=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- WandB setup -------------------------------------------------------
    use_wandb = args.wandb_project is not None
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
        )
        print(f"[INFO] WandB logging enabled: project={args.wandb_project}", flush=True)
    else:
        print("[INFO] WandB logging disabled (no --wandb_project set).", flush=True)

    # ---- Load model components ----------------------------------------
    print(f"[INFO] Loading feature extractor: {args.model_name}", flush=True)
    feature_extractor = ViTImageProcessor.from_pretrained(args.model_name)
    print("[INFO] Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    print("[INFO] Loading model...", flush=True)
    model = VisionEncoderDecoderModel.from_pretrained(args.model_name).to(device)

    # Ensure the decoder knows which token to start generation from
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    print("[INFO] Model loaded.", flush=True)

    # ---- Apply freeze strategy ----------------------------------------
    FREEZE_FN[args.finetune_mode](model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Trainable parameters: {trainable:,} / {total:,}")

    # ---- Augmentations (training only) ------------------------------------
    train_transform = None
    if args.augment:
        train_transform = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            # Removed ColorJitter and heavy rotation. 
            # VizWiz images are already noisy; applying slight affine translations mimics hand jitter.
            T.RandomAffine(degrees=5, translate=(0.05, 0.05)),
        ])
        print("[INFO] Training augmentations enabled (Mild Affine + Flip).", flush=True)

    # ---- Training dataset --------------------------------------------
    print("[INFO] Building training dataset...", flush=True)
    train_dataset = VizWizDataset(
        img_dir=args.train_img_dir,
        ann_file=args.train_ann_file,
        feature_extractor=feature_extractor,
        transform=train_transform,
    )
    if args.max_samples is not None:
        train_dataset.samples = train_dataset.samples[:args.max_samples]
        print(f"[INFO] Limiting to {args.max_samples} training samples.", flush=True)

    train_collate = partial(
        train_collate_fn,
        tokenizer=tokenizer,
        max_target_length=args.max_target_len,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=train_collate,
        pin_memory=(device.type == "cuda"),
    )

    # ---- Dynamic Learning Rate Assignment -----------------------------
    if args.lr is None:
        if args.finetune_mode == "all":
            args.lr = 1e-5
        elif args.finetune_mode == "backbone":
            args.lr = 2e-5
        else: # captioner
            args.lr = 5e-5
        print(f"[INFO] Auto-selected learning rate: {args.lr} for mode: {args.finetune_mode}", flush=True)

    # ---- Optimiser & scheduler ----------------------------------------
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
    )
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )

    # ---- Validation dataset (optional) --------------------------------
    val_loader = None
    if args.val_img_dir and args.val_ann_file:
        print("[INFO] Building validation dataset...", flush=True)
        val_dataset = VizWizDataset(
            img_dir=args.val_img_dir,
            ann_file=args.val_ann_file,
            feature_extractor=feature_extractor,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=(device.type == "cuda"),
        )

    gen_kwargs = {
        "max_new_tokens": 64,
        "num_beams": 4,
        "no_repeat_ngram_size": 3,
        "early_stopping": True,
    }

    # ---- Training loop ------------------------------------------------
    history = []
    best_val_score = -float("inf")
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        step_offset = (epoch - 1) * len(train_loader)
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, epoch,
            use_wandb=use_wandb, step_offset=step_offset,
        )
        epoch_info = {"epoch": epoch, "train_loss": train_loss}

        if val_loader is not None:
            preds, refs, _ = generate_captions(
                model, tokenizer, val_loader, device, gen_kwargs
            )
            metrics = compute_metrics(preds, refs)
            print_metrics(metrics, title=f"Epoch {epoch} Validation Metrics")
            epoch_info["val_metrics"] = metrics
            if use_wandb:
                wandb.log({"val/" + k: v for k, v in metrics.items()} | {"epoch": epoch})

            # Early stopping: monitor average of all val metrics
            val_score = sum(metrics.values()) / len(metrics)
            if val_score > best_val_score:
                best_val_score = val_score
                epochs_no_improve = 0
                best_ckpt_path = output_dir / "best_model"
                model.save_pretrained(best_ckpt_path)
                tokenizer.save_pretrained(best_ckpt_path)
                print(f"[INFO] New best val score: {val_score:.4f} — saved to {best_ckpt_path}")
            else:
                epochs_no_improve += 1
                print(f"[INFO] No improvement for {epochs_no_improve} epoch(s) "
                      f"(best: {best_val_score:.4f})")

        history.append(epoch_info)

        # Save checkpoint after each epoch
        ckpt_path = output_dir / f"checkpoint_epoch{epoch}"
        model.save_pretrained(ckpt_path)
        tokenizer.save_pretrained(ckpt_path)
        print(f"[INFO] Checkpoint saved to {ckpt_path}")

        if args.patience is not None and val_loader is not None and epochs_no_improve >= args.patience:
            print(f"[INFO] Early stopping triggered after {epoch} epochs "
                  f"({args.patience} epochs without improvement).")
            break

    # ---- Save final model --------------------------------------------
    final_path = output_dir / "final_model"
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"[INFO] Final model saved to {final_path}")

    # ---- Save training history ----------------------------------------
    history_path = output_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(
            {
                "model_name": args.model_name,
                "finetune_mode": args.finetune_mode,
                "epochs": args.epochs,
                "lr": args.lr,
                "history": history,
            },
            f,
            indent=4,
        )
    print(f"[INFO] Training history saved to {history_path}")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
