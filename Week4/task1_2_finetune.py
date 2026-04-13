"""
Task 1.2 - Fine-tune pre-built captioning models on VizWiz

Supported models (single --model_name checkpoint, no manual stitching):
  nlpconnect/vit-gpt2-image-captioning   → VisionEncoderDecoderModel
  Salesforce/blip-image-captioning-base  → BlipForConditionalGeneration
  microsoft/git-base-coco                → GitForCausalLM

Fine-tune modes:
  backbone   - freeze the decoder, train only the encoder
  captioner  - freeze the encoder, train only the decoder
  all        - train the full model end-to-end

Use:
    python task1_2_finetune.py \
        --train_img_dir  data/train \
        --train_ann_file data/annotations/train.json \
        --val_img_dir    data/val \
        --val_ann_file   data/annotations/val.json \
        --model_name     Salesforce/blip-image-captioning-base \
        --output_dir     outputs/blip \
        --finetune_mode  captioner \
        --epochs         20 \
        --augment
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
    AutoConfig,
    AutoProcessor,
    AutoTokenizer,
    VisionEncoderDecoderModel,
    BlipForConditionalGeneration,
    GitForCausalLM,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent))
from src.dataset import VizWizDataset, collate_fn
from src.metrics import compute_metrics, print_metrics


# ---------------------------------------------------------------------------
# Model + processor factory
# ---------------------------------------------------------------------------

# Map config model_type → model class.
# AutoModelForVision2Seq is not available in all transformers versions,
# so we resolve the class explicitly.
_MODEL_CLASS = {
    "vision-encoder-decoder": VisionEncoderDecoderModel,
    "blip":                   BlipForConditionalGeneration,
    "git":                    GitForCausalLM,
}


def build_model_and_processor(model_name: str, device):
    """
    Load a pre-trained captioning model and its processor.

    Returns:
        model        - on `device`
        processor    - AutoProcessor (used as feature_extractor in VizWizDataset)
        tokenizer    - text tokenizer extracted from the processor (pad_token guaranteed)
    """
    print(f"[INFO] Loading processor: {model_name}", flush=True)
    processor = AutoProcessor.from_pretrained(model_name)

    # Extract the text tokenizer — most processors expose it as .tokenizer
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Some decoders (GPT-2 inside ViT-GPT2, GIT) have no pad_token by default
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        if hasattr(processor, "tokenizer"):
            processor.tokenizer.pad_token = tokenizer.eos_token
        print(f"[INFO] No pad_token — set to eos_token ('{tokenizer.eos_token}')",
              flush=True)

    # Resolve model class from config model_type
    cfg = AutoConfig.from_pretrained(model_name)
    model_cls = _MODEL_CLASS.get(cfg.model_type)
    if model_cls is None:
        raise ValueError(
            f"Unsupported model_type='{cfg.model_type}' for '{model_name}'.\n"
            f"Supported: {list(_MODEL_CLASS.keys())}"
        )

    print(f"[INFO] Loading model ({model_cls.__name__}): {model_name}", flush=True)
    model = model_cls.from_pretrained(model_name).to(device)

    # Ensure the model config knows about pad / start tokens for generation
    if tokenizer.bos_token_id is not None:
        start_id = tokenizer.bos_token_id
    elif tokenizer.pad_token_id is not None:
        start_id = tokenizer.pad_token_id
    else:
        start_id = tokenizer.eos_token_id

    model.config.pad_token_id = tokenizer.pad_token_id
    if hasattr(model.config, "decoder_start_token_id"):
        if model.config.decoder_start_token_id is None:
            model.config.decoder_start_token_id = start_id

    print(f"[INFO] Model class : {model.__class__.__name__}", flush=True)
    print(f"[INFO] pad_token_id: {tokenizer.pad_token_id}  "
          f"decoder_start_token_id: {getattr(model.config, 'decoder_start_token_id', 'n/a')}",
          flush=True)

    return model, processor, tokenizer


# ---------------------------------------------------------------------------
# Unified forward pass
# ---------------------------------------------------------------------------

def model_forward(model, batch, device):
    """
    Route a training/validation batch through the model.

    VisionEncoderDecoderModel only needs pixel_values + labels.
    BLIP and GIT additionally need input_ids + attention_mask so the text
    decoder can compute the cross-entropy loss over the target tokens.
    """
    pixel_values = batch["pixel_values"].to(device)
    labels       = batch["labels"].to(device)

    if isinstance(model, VisionEncoderDecoderModel):
        return model(pixel_values=pixel_values, labels=labels)

    return model(
        pixel_values   = pixel_values,
        input_ids      = batch["input_ids"].to(device),
        attention_mask = batch["attention_mask"].to(device),
        labels         = labels,
    )


# ---------------------------------------------------------------------------
# Collate for training
# ---------------------------------------------------------------------------

def train_collate_fn(batch, tokenizer, max_target_length=64):
    """
    Tokenises one randomly chosen reference caption per image.

    Always returns input_ids + attention_mask alongside labels so the same
    collate function works for VisionEncoderDecoder, BLIP, and GIT.
    (VisionEncoderDecoder ignores input_ids / attention_mask.)
    """
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    texts = [random.choice(item["captions"]) for item in batch]

    encoding = tokenizer(
        texts,
        padding="max_length",
        max_length=max_target_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids      = encoding.input_ids
    attention_mask = encoding.attention_mask
    labels = input_ids.clone()
    labels[labels == tokenizer.pad_token_id] = -100

    return {
        "pixel_values":  pixel_values,
        "input_ids":     input_ids,
        "attention_mask": attention_mask,
        "labels":        labels,
        "captions":      [item["captions"]    for item in batch],
        "image_paths":   [item["image_path"]  for item in batch],
    }


# ---------------------------------------------------------------------------
# Freeze helpers
# ---------------------------------------------------------------------------

def _vision_language_parts(model):
    """
    Return ([vision modules], [language modules]) for each supported class.

    VisionEncoderDecoderModel : .encoder / .decoder
    BlipForConditionalGeneration: .vision_model / .text_decoder
    GitForCausalLM              : .git.image_encoder / (.git.embeddings,
                                   .git.encoder, .output)
    """
    if isinstance(model, VisionEncoderDecoderModel):
        return [model.encoder], [model.decoder]
    if isinstance(model, BlipForConditionalGeneration):
        return [model.vision_model], [model.text_decoder]
    if isinstance(model, GitForCausalLM):
        return (
            [model.git.image_encoder],
            [model.git.embeddings, model.git.encoder, model.output],
        )
    raise ValueError(f"Don't know how to split {model.__class__.__name__}")


def _set_grad(modules, value: bool):
    for m in modules:
        for p in m.parameters():
            p.requires_grad = value


def freeze_backbone(model):
    """Freeze the vision encoder; train only the language decoder."""
    vision, language = _vision_language_parts(model)
    _set_grad(vision,   False)
    _set_grad(language, True)
    print("[INFO] Frozen: vision encoder. Training: language decoder.")


def freeze_captioner(model):
    """Freeze the language decoder; train only the vision encoder."""
    vision, language = _vision_language_parts(model)
    _set_grad(vision,   True)
    _set_grad(language, False)
    print("[INFO] Frozen: language decoder. Training: vision encoder.")


def train_all(model):
    for param in model.parameters():
        param.requires_grad = True
    print("[INFO] Training full model.")


FREEZE_FN = {
    "backbone":  freeze_backbone,
    "captioner": freeze_captioner,
    "all":       train_all,
}


# ---------------------------------------------------------------------------
# Training / validation helpers
# ---------------------------------------------------------------------------

def train_one_epoch(model, dataloader, optimizer, scheduler, device, epoch,
                    use_wandb=False, step_offset=0):
    model.train()
    total_loss = 0.0

    for i, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch} [train]")):
        outputs = model_forward(model, batch, device)
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


def compute_val_loss(model, dataloader, device, epoch, use_wandb=False):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Epoch {epoch} [val loss]"):
            total_loss += model_forward(model, batch, device).loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"[INFO] Epoch {epoch} — avg val loss: {avg_loss:.4f}")
    if use_wandb:
        wandb.log({"val/loss_epoch": avg_loss, "epoch": epoch})
    return avg_loss


def generate_captions(model, tokenizer, dataloader, device, gen_kwargs):
    predictions, references, image_paths = [], [], []
    model.eval()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating captions"):
            pixel_values = batch["pixel_values"].to(device)
            output_ids = model.generate(pixel_values=pixel_values, **gen_kwargs)
            captions = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            captions = [c.strip().lower() for c in captions]

            predictions.extend(captions)
            references.extend(batch["captions"])
            image_paths.extend(batch["image_paths"])

    return predictions, references, image_paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune a pre-built captioning model on VizWiz"
    )

    # Data
    parser.add_argument("--train_img_dir",  type=str, required=True)
    parser.add_argument("--train_ann_file", type=str, required=True)
    parser.add_argument("--val_img_dir",    type=str, default=None)
    parser.add_argument("--val_ann_file",   type=str, default=None)

    # Model
    parser.add_argument("--model_name", type=str,
                        default="nlpconnect/vit-gpt2-image-captioning",
                        help=(
                            "Pre-trained captioning model. Supported:\n"
                            "  nlpconnect/vit-gpt2-image-captioning\n"
                            "  Salesforce/blip-image-captioning-base\n"
                            "  microsoft/git-base-coco"
                        ))
    parser.add_argument("--finetune_mode", type=str,
                        choices=["backbone", "captioner", "all"], default="captioner")

    # Training
    parser.add_argument("--epochs",         type=int,   default=10)
    parser.add_argument("--lr",             type=float, default=None)
    parser.add_argument("--batch_size",     type=int,   default=16)
    parser.add_argument("--num_workers",    type=int,   default=4)
    parser.add_argument("--warmup_steps",   type=int,   default=1000)
    parser.add_argument("--max_target_len", type=int,   default=64)
    parser.add_argument("--max_samples",    type=int,   default=None)
    parser.add_argument("--output_dir",     type=str,   default="outputs/task1_2")
    parser.add_argument("--wandb_project",  type=str,   default=None)
    parser.add_argument("--wandb_run_name", type=str,   default=None)
    parser.add_argument("--patience",       type=int,   default=None)
    parser.add_argument("--augment",        action="store_true")

    # Generation (validation)
    parser.add_argument("--gen_num_beams",      type=int, default=1,
                        help="Beam width for val generation. 1=greedy (fast).")
    parser.add_argument("--gen_max_new_tokens", type=int, default=32)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}", flush=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- WandB ----------------------------------------------------------------
    use_wandb = args.wandb_project is not None
    if use_wandb:
        wandb.init(project=args.wandb_project, name=args.wandb_run_name,
                   config=vars(args))
        print(f"[INFO] WandB: project={args.wandb_project}", flush=True)
    else:
        print("[INFO] WandB disabled.", flush=True)

    # ---- Model + processor ----------------------------------------------------
    model, processor, tokenizer = build_model_and_processor(args.model_name, device)

    # ---- Freeze strategy ------------------------------------------------------
    FREEZE_FN[args.finetune_mode](model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Trainable: {trainable:,} / {total:,}")

    # ---- Augmentations --------------------------------------------------------
    train_transform = None
    if args.augment:
        train_transform = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomAffine(degrees=5, translate=(0.05, 0.05)),
        ])
        print("[INFO] Augmentations enabled.", flush=True)

    # ---- Datasets + loaders ---------------------------------------------------
    print("[INFO] Building training dataset...", flush=True)
    train_dataset = VizWizDataset(
        img_dir=args.train_img_dir, ann_file=args.train_ann_file,
        feature_extractor=processor, transform=train_transform,
    )
    if args.max_samples:
        train_dataset.samples = train_dataset.samples[:args.max_samples]
        print(f"[INFO] Capped at {args.max_samples} samples.", flush=True)

    train_collate = partial(train_collate_fn, tokenizer=tokenizer,
                            max_target_length=args.max_target_len)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=train_collate,
        pin_memory=(device.type == "cuda"),
    )

    # ---- Learning rate --------------------------------------------------------
    if args.lr is None:
        args.lr = {"all": 1e-5, "backbone": 2e-5, "captioner": 5e-5}[args.finetune_mode]
        print(f"[INFO] Auto lr={args.lr} (mode={args.finetune_mode})", flush=True)

    # ---- Optimiser + scheduler ------------------------------------------------
    optimizer   = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler   = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps
    )

    # ---- Validation loaders ---------------------------------------------------
    val_loader = val_loss_loader = None
    if args.val_img_dir and args.val_ann_file:
        print("[INFO] Building validation dataset...", flush=True)
        val_dataset = VizWizDataset(
            img_dir=args.val_img_dir, ann_file=args.val_ann_file,
            feature_extractor=processor,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, collate_fn=collate_fn,
            pin_memory=(device.type == "cuda"),
        )
        val_loss_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers,
            collate_fn=partial(train_collate_fn, tokenizer=tokenizer,
                               max_target_length=args.max_target_len),
            pin_memory=(device.type == "cuda"),
        )

    gen_kwargs = {
        "max_new_tokens":     args.gen_max_new_tokens,
        "num_beams":          args.gen_num_beams,
        "no_repeat_ngram_size": 3,
        "early_stopping":     args.gen_num_beams > 1,
        "pad_token_id":       tokenizer.pad_token_id,
    }
    print(f"[INFO] Generation: num_beams={args.gen_num_beams}, "
          f"max_new_tokens={args.gen_max_new_tokens}", flush=True)

    # ---- Training loop --------------------------------------------------------
    history = []
    best_val_score  = -float("inf")
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        step_offset = (epoch - 1) * len(train_loader)
        train_loss  = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, epoch,
            use_wandb=use_wandb, step_offset=step_offset,
        )
        epoch_info = {"epoch": epoch, "train_loss": train_loss}

        if val_loader is not None:
            val_loss = compute_val_loss(model, val_loss_loader, device, epoch,
                                        use_wandb=use_wandb)
            epoch_info["val_loss"] = val_loss

            preds, refs, _ = generate_captions(model, tokenizer, val_loader,
                                               device, gen_kwargs)
            metrics = compute_metrics(preds, refs)
            print_metrics(metrics, title=f"Epoch {epoch} Validation Metrics")
            epoch_info["val_metrics"] = metrics
            if use_wandb:
                wandb.log({"val/" + k: v for k, v in metrics.items()} | {"epoch": epoch})

            val_score = sum(metrics.values()) / len(metrics)
            if val_score > best_val_score:
                best_val_score    = val_score
                epochs_no_improve = 0
                best_path = output_dir / "best_model"
                model.save_pretrained(best_path)
                tokenizer.save_pretrained(best_path)
                print(f"[INFO] New best {val_score:.4f} → {best_path}")
            else:
                epochs_no_improve += 1
                print(f"[INFO] No improvement {epochs_no_improve} epoch(s) "
                      f"(best {best_val_score:.4f})")

        history.append(epoch_info)

        ckpt = output_dir / f"checkpoint_epoch{epoch}"
        model.save_pretrained(ckpt)
        tokenizer.save_pretrained(ckpt)
        print(f"[INFO] Checkpoint → {ckpt}")

        if (args.patience and val_loader and epochs_no_improve >= args.patience):
            print(f"[INFO] Early stopping after {epoch} epochs.")
            break

    # ---- Final save -----------------------------------------------------------
    final = output_dir / "final_model"
    model.save_pretrained(final)
    tokenizer.save_pretrained(final)
    print(f"[INFO] Final model → {final}")

    with open(output_dir / "training_history.json", "w") as f:
        json.dump({"model_name": args.model_name, "finetune_mode": args.finetune_mode,
                   "epochs": args.epochs, "lr": args.lr, "history": history},
                  f, indent=4)

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
