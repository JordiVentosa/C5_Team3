"""
task2_2_inference.py
Generates predictions from the best task 2.2 model (ViT + Qwen3.5 LoRA)
on the validation set and saves a JSON compatible with visualize_predictions.py

Usage:
    python task2_2_inference.py \
        --vit_model     /home/mcvstudent20/C5_Team3/Week4/modelo_vit_gpt2 \
        --qwen_model    /home/mcvstudent20/C5_Team3/Week4/modelo_qwen35_4b \
        --lora_dir      outputs/task2_2_4b_r8a16/best_model/qwen_lora \
        --proj_path     outputs/task2_2_4b_r8a16/best_model/projection.pt \
        --val_img_dir   /data/uabmcv2526/mcvstudent20/data/vizwiz/val \
        --val_ann_file  /data/uabmcv2526/mcvstudent20/data/vizwiz/annotations/val.json \
        --output_file   outputs/results/task2_2_best_predictions.json \
        --max_samples   10 \
        --seed          42
"""

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import (
    ViTImageProcessor,
    VisionEncoderDecoderModel,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import PeftModel
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent))
from src.dataset import VizWizDataset, collate_fn
from task2_2 import ViTQwenCaptioner


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vit_model",    required=True)
    parser.add_argument("--qwen_model",   required=True)
    parser.add_argument("--lora_dir",     required=True)
    parser.add_argument("--proj_path",    required=True)
    parser.add_argument("--val_img_dir",  required=True)
    parser.add_argument("--val_ann_file", required=True)
    parser.add_argument("--output_file",  default="outputs/results/task2_2_best_predictions.json")
    parser.add_argument("--batch_size",   type=int, default=4)
    parser.add_argument("--num_workers",  type=int, default=0)
    parser.add_argument("--max_samples",  type=int, default=None)
    parser.add_argument("--seed",         type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}", flush=True)

    # ── Load frozen ViT ───────────────────────────────────────────────────────
    print(f"[INFO] Loading ViT from {args.vit_model}", flush=True)
    _full = VisionEncoderDecoderModel.from_pretrained(args.vit_model)
    vit_encoder       = _full.encoder
    feature_extractor = ViTImageProcessor.from_pretrained(args.vit_model)
    del _full
    vit_encoder.eval()
    for p in vit_encoder.parameters():
        p.requires_grad = False
    vit_hidden_size = vit_encoder.config.hidden_size

    # ── Load Qwen3.5 base + LoRA ──────────────────────────────────────────────
    print(f"[INFO] Loading Qwen3.5 base from {args.qwen_model}", flush=True)
    qwen_tokenizer = AutoTokenizer.from_pretrained(args.qwen_model)
    if qwen_tokenizer.pad_token is None:
        qwen_tokenizer.pad_token = qwen_tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.qwen_model,
        torch_dtype=torch.bfloat16,
    )
    base_model.config.pad_token_id = qwen_tokenizer.pad_token_id

    print(f"[INFO] Loading LoRA adapters from {args.lora_dir}", flush=True)
    qwen_model = PeftModel.from_pretrained(base_model, args.lora_dir)
    qwen_hidden_size = qwen_model.config.hidden_size

    # ── Build combined model ──────────────────────────────────────────────────
    model = ViTQwenCaptioner(
        vit_encoder=vit_encoder,
        qwen_model=qwen_model,
        vit_hidden_size=vit_hidden_size,
        qwen_hidden_size=qwen_hidden_size,
    ).to(device)

    print(f"[INFO] Loading projection from {args.proj_path}", flush=True)
    proj_state = torch.load(args.proj_path, map_location=device)
    model.projection.load_state_dict(proj_state)
    model.eval()

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset = VizWizDataset(
        img_dir=args.val_img_dir,
        ann_file=args.val_ann_file,
        feature_extractor=feature_extractor,
    )

    if args.max_samples is not None:
        indices = random.sample(range(len(dataset)), min(args.max_samples, len(dataset)))
        dataset.samples = [dataset.samples[i] for i in sorted(indices)]
        print(f"[INFO] Using {len(dataset.samples)} samples (seed={args.seed})")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    gen_kwargs = {
        "max_new_tokens":     64,
        "do_sample":          False,
        "eos_token_id":       qwen_tokenizer.eos_token_id,
        "pad_token_id":       qwen_tokenizer.pad_token_id,
        "repetition_penalty": 1.3,   # Avoid loops and repetitions
    }

    # ── Generate ──────────────────────────────────────────────────────────────
    predictions, references, image_paths = [], [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating"):
            pixel_values = batch["pixel_values"].to(device)

            output_ids = model.generate(
                pixel_values=pixel_values,
                tokenizer=qwen_tokenizer,
                **gen_kwargs,
            )
            captions = qwen_tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            # Clean artifacts from the Qwen3.5 chat template
            clean = []
            for c in captions:
                c = c.replace("<|im_start|>", "").replace("<|im_end|>", "")
                c = c.replace("assistant", "").replace("user", "")
                # Remove thinking tags
                if "<think>" in c:
                    c = c.split("</think>")[-1]
                c = c.strip().lower()
                clean.append(c)

            predictions.extend(clean)
            references.extend(batch["captions"])
            image_paths.extend(batch["image_paths"])

    # ── Save JSON ─────────────────────────────────────────────────────────────
    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = [
        {
            "image":              img,
            "predicted_caption":  pred,
            "reference_captions": refs,
        }
        for img, pred, refs in zip(image_paths, predictions, references)
    ]

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[INFO] Saved {len(results)} predictions to {out_path}")


if __name__ == "__main__":
    main()