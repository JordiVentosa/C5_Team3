"""
infer_qualitative.py

Run caption inference with the ViT+Qwen (task_e) architecture on a small set
of images — intended for qualitative analysis only, not full-val evaluation.

The script accepts an image list from either:
  - a plain .txt with one image filename per line
  - the qualitative_qi_examples.txt produced by qualitative_quality_issues.py
    (it auto-detects the "=== Example N: <filename> ===" format)

Usage (on cluster):
    python Week5/infer_qualitative.py \
        --checkpoint_dir  outputs/task_e/checkpoint_epoch3 \
        --vit_model       checkpoints/vit-gpt2 \
        --qwen_model      checkpoints/qwen3.5-4b \
        --img_dir         /path/to/vizwiz/val \
        --image_list      Week5/outputs/qualitative_qi/qualitative_qi_examples.txt \
        --output_file     Week5/predictions/task_e_qualitative.json \
        --batch_size      4
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from transformers import (
    ViTImageProcessor,
    VisionEncoderDecoderModel,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import PeftModel
from tqdm import tqdm

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(f"Using GPU: {os.environ['CUDA_VISIBLE_DEVICES']}")


# ─────────────────────────────────────────────────────────────────────────────
# Model (mirrors task_e.py exactly)
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
        return self.projection(vit_out).to(qwen_dtype)

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


def clean_caption(text):
    text = text.replace("<|im_start|>", "").replace("<|im_end|>", "")
    text = text.replace("assistant", "").replace("user", "")
    if "<think>" in text:
        text = text.split("</think>")[-1]
    text = text.strip()
    if "." in text:
        text = text.split(".")[0].strip()
    return text.lower()


# ─────────────────────────────────────────────────────────────────────────────
# Image list parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_image_list(path: str) -> list[str]:
    """Return a list of filenames from either format."""
    lines = Path(path).read_text().splitlines()
    filenames = []
    # Try qualitative_qi_examples.txt format first
    for line in lines:
        m = re.match(r"=== Example \d+: (.+?) ===", line.strip())
        if m:
            filenames.append(m.group(1))
    if filenames:
        return filenames
    # Plain list fallback (one filename/path per line, skip blanks)
    return [l.strip() for l in lines if l.strip()]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--vit_model",    required=True,
                   help="Path to vit-gpt2 model (for the ViT encoder)")
    p.add_argument("--qwen_model",   required=True,
                   help="Path to base Qwen model (must have tokenizer files)")
    p.add_argument("--lora_dir",     required=True,
                   help="Path to the qwen_lora/ adapter folder")
    p.add_argument("--projection",   required=True,
                   help="Path to projection.pt")
    p.add_argument("--img_dir",      required=True,
                   help="Directory containing VizWiz val images")
    p.add_argument("--image_list",   required=True,
                   help="qualitative_qi_examples.txt or plain list of filenames")
    p.add_argument("--output_file",  default="Week5/predictions/task_e_qualitative.json")
    p.add_argument("--batch_size",   type=int, default=4)
    p.add_argument("--gpu",          type=str, default="0")
    return p.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # ── ViT encoder ──────────────────────────────────────────────────────────
    print(f"[INFO] Loading ViT encoder from {args.vit_model}")
    _full             = VisionEncoderDecoderModel.from_pretrained(args.vit_model)
    vit_encoder       = _full.encoder
    feature_extractor = ViTImageProcessor.from_pretrained(args.vit_model)
    del _full
    vit_encoder.eval()
    for p in vit_encoder.parameters():
        p.requires_grad = False
    vit_hidden_size = vit_encoder.config.hidden_size

    # ── Qwen base + LoRA ─────────────────────────────────────────────────────
    print(f"[INFO] Loading Qwen base from {args.qwen_model}")
    tokenizer  = AutoTokenizer.from_pretrained(args.qwen_model)
    qwen_model = AutoModelForCausalLM.from_pretrained(
        args.qwen_model, torch_dtype=torch.bfloat16
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token            = tokenizer.eos_token
        qwen_model.config.pad_token_id = tokenizer.pad_token_id

    print(f"[INFO] Loading LoRA adapter from {args.lora_dir}")
    qwen_model = PeftModel.from_pretrained(qwen_model, args.lora_dir)
    qwen_model = qwen_model.merge_and_unload()
    qwen_hidden_size = qwen_model.config.hidden_size

    # ── Assemble captioner ────────────────────────────────────────────────────
    model = ViTQwenCaptioner(
        vit_encoder=vit_encoder,
        qwen_model=qwen_model,
        vit_hidden_size=vit_hidden_size,
        qwen_hidden_size=qwen_hidden_size,
    ).to(device)

    print(f"[INFO] Loading projection from {args.projection}")
    model.projection.load_state_dict(
        torch.load(args.projection, map_location=device)
    )
    model.eval()

    gen_kwargs = {
        "max_new_tokens":     64,
        "do_sample":          False,
        "repetition_penalty": 1.3,
        "eos_token_id":       tokenizer.eos_token_id,
        "pad_token_id":       tokenizer.pad_token_id,
    }

    # ── Image list ────────────────────────────────────────────────────────────
    filenames = parse_image_list(args.image_list)
    print(f"[INFO] Running inference on {len(filenames)} images")

    results = []
    for i in range(0, len(filenames), args.batch_size):
        batch_files = filenames[i : i + args.batch_size]
        images, valid_files = [], []
        for fn in batch_files:
            img_path = os.path.join(args.img_dir, fn)
            try:
                images.append(Image.open(img_path).convert("RGB"))
                valid_files.append(fn)
            except Exception as e:
                print(f"[WARN] Could not open {img_path}: {e}")

        if not images:
            continue

        inputs       = feature_extractor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)

        output_ids = model.generate(pixel_values, tokenizer, **gen_kwargs)
        captions   = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        captions   = [clean_caption(c) for c in captions]

        for fn, cap in zip(valid_files, captions):
            results.append({"filename": fn, "predicted_caption": cap})
            print(f"  {fn}: {cap}")

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Saved] {out_path}")


if __name__ == "__main__":
    main()
