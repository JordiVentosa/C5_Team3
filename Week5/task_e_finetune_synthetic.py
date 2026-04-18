# task2_2_synthetic.py
import argparse
import json
import random
import sys
from pathlib import Path
from functools import partial

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import AdamW
from torchvision import transforms as T
from transformers import (
    ViTImageProcessor,
    VisionEncoderDecoderModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent))
from src.dataset import VizWizDataset, collate_fn
from src.metrics import compute_metrics, print_metrics
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# ─────────────────────────────────────────────────────────────────────────────
# Caption cleaning
# ─────────────────────────────────────────────────────────────────────────────

def clean_caption(text):
    """Removes Qwen3.5 output artifacts and keeps the first sentence."""
    text = text.replace("<|im_start|>", "").replace("<|im_end|>", "")
    text = text.replace("assistant", "").replace("user", "")
    if "<think>" in text:
        text = text.split("</think>")[-1]
    text = text.strip()
    if "." in text:
        text = text.split(".")[0].strip()
    return text.lower()


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class ViTQwenCaptioner(nn.Module):
    def __init__(self, vit_encoder, qwen_model, vit_hidden_size, qwen_hidden_size):
        super().__init__()
        self.vit = vit_encoder
        self.projection = nn.Sequential(
            nn.Linear(vit_hidden_size, qwen_hidden_size),
            nn.GELU(),
            nn.Linear(qwen_hidden_size, qwen_hidden_size),
        )
        self.qwen = qwen_model

    def _get_visual_embeds(self, pixel_values):
        qwen_dtype = next(self.qwen.parameters()).dtype
        with torch.no_grad():
            vit_out = self.vit(pixel_values.to(torch.float32)).last_hidden_state
        proj = self.projection(vit_out)
        return proj.to(qwen_dtype)

    def forward(self, pixel_values, input_ids, attention_mask, labels):
        visual_embeds = self._get_visual_embeds(pixel_values)
        text_embeds   = self.qwen.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)

        B, N_vis = visual_embeds.shape[:2]
        vis_attn  = torch.ones(B, N_vis, device=pixel_values.device, dtype=attention_mask.dtype)
        full_attn = torch.cat([vis_attn, attention_mask], dim=1)

        vis_labels  = torch.full((B, N_vis), -100, device=labels.device, dtype=labels.dtype)
        full_labels = torch.cat([vis_labels, labels], dim=1)

        return self.qwen(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attn,
            labels=full_labels,
        )

    @torch.no_grad()
    def generate(self, pixel_values, tokenizer, max_new_tokens=64, **gen_kwargs):
        visual_embeds = self._get_visual_embeds(pixel_values)
        B, N_vis = visual_embeds.shape[:2]

        bos_id       = tokenizer.bos_token_id or tokenizer.eos_token_id
        start_ids    = torch.full((B, 1), bos_id, device=pixel_values.device, dtype=torch.long)
        start_embeds = self.qwen.get_input_embeddings()(start_ids)

        inputs_embeds = torch.cat([visual_embeds, start_embeds], dim=1)
        attn_mask     = torch.ones(B, N_vis + 1, device=pixel_values.device, dtype=torch.long)

        return self.qwen.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            max_new_tokens=max_new_tokens,
            **gen_kwargs,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Collate
# ─────────────────────────────────────────────────────────────────────────────

def train_collate_fn(batch, tokenizer, max_target_length=64):
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
    labels         = input_ids.clone()
    labels[labels == tokenizer.pad_token_id] = -100

    return {
        "pixel_values":   pixel_values,
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "labels":         labels,
        "captions":       [item["captions"]   for item in batch],
        "image_paths":    [item["image_path"] for item in batch],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Training helpers
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, dataloader, optimizer, scheduler, device, epoch, log_every=50):
    model.train()
    total_loss = 0.0

    for step, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch} [train]"), start=1):
        pixel_values   = batch["pixel_values"].to(device)
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        outputs = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=1.0
        )
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        if step % log_every == 0:
            avg_so_far = total_loss / step
            print(f"[INFO] Epoch {epoch} step {step}/{len(dataloader)} — loss: {loss.item():.4f} | avg: {avg_so_far:.4f}", flush=True)

    avg_loss = total_loss / len(dataloader)
    print(f"[INFO] Epoch {epoch} — avg train loss: {avg_loss:.4f}")
    return avg_loss


def evaluate(model, tokenizer, dataloader, device, gen_kwargs):
    predictions, references, image_paths = [], [], []
    model.eval()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            pixel_values = batch["pixel_values"].to(device)

            output_ids = model.generate(
                pixel_values=pixel_values,
                tokenizer=tokenizer,
                **gen_kwargs,
            )
            captions = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            captions = [clean_caption(cap) for cap in captions]

            predictions.extend(captions)
            references.extend(batch["captions"])
            image_paths.extend(batch["image_paths"])

    return predictions, references, image_paths


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Task 2.2 — Frozen ViT + Qwen3.5 decoder with LoRA + synthetic data"
    )
    # VizWiz real data
    parser.add_argument("--train_img_dir",  required=True)
    parser.add_argument("--train_ann_file", required=True)
    parser.add_argument("--val_img_dir",    default=None)
    parser.add_argument("--val_ann_file",   default=None)

    # Synthetic data
    parser.add_argument("--synth_img_dir",  default=None,
                        help="Path to synthetic images dir (optional)")
    parser.add_argument("--synth_ann_file", default=None,
                        help="Path to synthetic annotations JSON (optional)")

    # Models
    parser.add_argument("--vit_model",  required=True)
    parser.add_argument("--qwen_model", required=True)

    # LoRA
    parser.add_argument("--lora_r",       type=int,   default=8)
    parser.add_argument("--lora_alpha",   type=int,   default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_targets", type=str,
                        default="q_proj,v_proj,k_proj,o_proj")

    # Training
    parser.add_argument("--epochs",         type=int,   default=5)
    parser.add_argument("--lr",             type=float, default=1e-4)
    parser.add_argument("--batch_size",     type=int,   default=8)
    parser.add_argument("--num_workers",    type=int,   default=4)
    parser.add_argument("--warmup_steps",   type=int,   default=100)
    parser.add_argument("--max_target_len", type=int,   default=64)
    parser.add_argument("--max_samples",    type=int,   default=None)
    parser.add_argument("--patience",       type=int,   default=None)
    parser.add_argument("--augment",        action="store_true")

    parser.add_argument("--output_dir", type=str, default="outputs/task2_2")

    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}", flush=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load frozen ViT encoder ───────────────────────────────────────────────
    print(f"[INFO] Loading ViT encoder from: {args.vit_model}", flush=True)
    _full = VisionEncoderDecoderModel.from_pretrained(args.vit_model)
    vit_encoder       = _full.encoder
    feature_extractor = ViTImageProcessor.from_pretrained(args.vit_model)
    del _full

    vit_encoder.eval()
    for p in vit_encoder.parameters():
        p.requires_grad = False
    vit_hidden_size = vit_encoder.config.hidden_size
    print(f"[INFO] ViT encoder frozen. Hidden size: {vit_hidden_size}")

    # ── Load Qwen3.5 decoder ──────────────────────────────────────────────────
    print(f"[INFO] Loading Qwen3.5 from: {args.qwen_model}", flush=True)
    qwen_tokenizer = AutoTokenizer.from_pretrained(args.qwen_model)
    qwen_model     = AutoModelForCausalLM.from_pretrained(
        args.qwen_model,
        torch_dtype=torch.bfloat16,
    )
    qwen_hidden_size = qwen_model.config.hidden_size

    if qwen_tokenizer.pad_token is None:
        qwen_tokenizer.pad_token       = qwen_tokenizer.eos_token
        qwen_model.config.pad_token_id = qwen_tokenizer.pad_token_id

    # ── Apply LoRA ────────────────────────────────────────────────────────────
    lora_targets = [t.strip() for t in args.lora_targets.split(",")]
    lora_config  = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=lora_targets,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    qwen_model = get_peft_model(qwen_model, lora_config)
    qwen_model.print_trainable_parameters()

    # ── Build combined model ──────────────────────────────────────────────────
    model = ViTQwenCaptioner(
        vit_encoder=vit_encoder,
        qwen_model=qwen_model,
        vit_hidden_size=vit_hidden_size,
        qwen_hidden_size=qwen_hidden_size,
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Trainable params: {trainable:,} / {total:,}")

    # ── Augmentations ─────────────────────────────────────────────────────────
    train_transform = None
    if args.augment:
        train_transform = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomAffine(degrees=5, translate=(0.05, 0.05)),
        ])
        print("[INFO] Augmentations enabled.")

    # ── Datasets ──────────────────────────────────────────────────────────────
    print("[INFO] Building VizWiz training dataset...", flush=True)
    vizwiz_dataset = VizWizDataset(
        img_dir=args.train_img_dir,
        ann_file=args.train_ann_file,
        feature_extractor=feature_extractor,
        transform=train_transform,
    )
    if args.max_samples:
        vizwiz_dataset.samples = vizwiz_dataset.samples[:args.max_samples]
        print(f"[INFO] Limiting VizWiz to {args.max_samples} samples.")

    train_dataset = vizwiz_dataset

    # Combina con sintético si se proporciona
    if args.synth_img_dir and args.synth_ann_file:
        print("[INFO] Building synthetic training dataset...", flush=True)
        synth_dataset = VizWizDataset(
            img_dir=args.synth_img_dir,
            ann_file=args.synth_ann_file,
            feature_extractor=feature_extractor,
            transform=train_transform,
        )
        train_dataset = ConcatDataset([vizwiz_dataset, synth_dataset])
        print(f"[INFO] Combined dataset: {len(vizwiz_dataset)} VizWiz + "
              f"{len(synth_dataset)} synthetic = {len(train_dataset)} total")
    else:
        print(f"[INFO] Training on VizWiz only: {len(train_dataset)} samples")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=partial(train_collate_fn, tokenizer=qwen_tokenizer,
                           max_target_length=args.max_target_len),
        pin_memory=(device.type == "cuda"),
    )

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

    # ── Optimizer & scheduler ─────────────────────────────────────────────────
    optimizer   = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )

    gen_kwargs = {
        "max_new_tokens":    64,
        "do_sample":         False,
        "repetition_penalty": 1.3,
        "eos_token_id":      qwen_tokenizer.eos_token_id,
        "pad_token_id":      qwen_tokenizer.pad_token_id,
    }

    # ── Training loop ─────────────────────────────────────────────────────────
    history           = []
    best_val_score    = -float("inf")
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, epoch
        )
        epoch_info = {"epoch": epoch, "train_loss": train_loss}

        if val_loader is not None:
            preds, refs, img_paths = evaluate(
                model, qwen_tokenizer, val_loader, device, gen_kwargs
            )
            metrics = compute_metrics(preds, refs)
            print_metrics(metrics, title=f"Epoch {epoch} Validation")
            epoch_info["val_metrics"] = metrics

            captions_path = output_dir / f"captions_epoch{epoch}.json"
            with open(captions_path, "w") as f:
                json.dump([
                    {"image_path": p, "predicted": pred, "references": ref}
                    for p, pred, ref in zip(img_paths, preds, refs)
                ], f, indent=2)
            print(f"[INFO] Captions saved → {captions_path}")

            val_score = sum(metrics.values()) / len(metrics)
            if val_score > best_val_score:
                best_val_score    = val_score
                epochs_no_improve = 0
                best_path = output_dir / "best_model"
                best_path.mkdir(exist_ok=True)
                model.qwen.save_pretrained(best_path / "qwen_lora")
                torch.save(model.projection.state_dict(), best_path / "projection.pt")
                print(f"[INFO] New best: {val_score:.4f} — saved to {best_path}")
            else:
                epochs_no_improve += 1
                print(f"[INFO] No improvement "
                      f"({epochs_no_improve} epoch(s), best: {best_val_score:.4f})")

        history.append(epoch_info)

        ckpt = output_dir / f"checkpoint_epoch{epoch}"
        ckpt.mkdir(exist_ok=True)
        model.qwen.save_pretrained(ckpt / "qwen_lora")
        torch.save(model.projection.state_dict(), ckpt / "projection.pt")
        print(f"[INFO] Checkpoint → {ckpt}")

        if args.patience and val_loader and epochs_no_improve >= args.patience:
            print(f"[INFO] Early stopping after {epoch} epochs.")
            break

    # ── Save final model ──────────────────────────────────────────────────────
    final = output_dir / "final_model"
    final.mkdir(exist_ok=True)
    model.qwen.save_pretrained(final / "qwen_lora")
    torch.save(model.projection.state_dict(), final / "projection.pt")
    qwen_tokenizer.save_pretrained(final / "tokenizer")
    print(f"[INFO] Final model → {final}")

    with open(output_dir / "training_history.json", "w") as f:
        json.dump({
            "vit_model":    args.vit_model,
            "qwen_model":   args.qwen_model,
            "lora_r":       args.lora_r,
            "lora_alpha":   args.lora_alpha,
            "lora_targets": lora_targets,
            "epochs":       args.epochs,
            "lr":           args.lr,
            "history":      history,
        }, f, indent=4)
    print("[INFO] Training history saved.")


if __name__ == "__main__":
    main()