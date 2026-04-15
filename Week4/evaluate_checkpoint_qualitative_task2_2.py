"""
Evaluate checkpoint and generate qualitative results for Task 2.2

Loads a Qwen3.5 + ViT-GPT2 setup with LoRA and projection, generating qualitative visualizations grids.

Usage:
    python evaluate_checkpoint_qualitative_task2_2.py \
        --vit_model     nlpconnect/vit-gpt2-image-captioning \
        --qwen_model    Qwen/Qwen1.5-4B-Chat \
        --lora_dir      models/checkpoint_best2_2 \
        --proj_path     projection.pt \
        --img_dir       ../Week3/data/vizwiz/val \
        --ann_file      ../Week3/data/vizwiz/annotations/val.json \
        --output_dir    outputs/qualitative_task2_2 \
        --n             20
"""

import argparse
import sys
import random
import json
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
from src.metrics import compute_metrics, print_metrics
from task2_2 import ViTQwenCaptioner


def make_grid(samples, output_path, title="Qualitative results"):
    """Generate visualization grid."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    n = len(samples)
    fig, axes = plt.subplots(n, 1, figsize=(12, 5 * n))
    if n == 1:
        axes = [axes]

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.001)

    for ax, sample in zip(axes, samples):
        img = Image.open(sample["image"]).convert("RGB")
        ax.axis("off")

        divider_pos = 0.35

        ax_img = ax.inset_axes([0, 0, divider_pos, 1])
        ax_text = ax.inset_axes([divider_pos + 0.02, 0, 1 - divider_pos - 0.02, 1])

        ax_img.imshow(img)
        ax_img.axis("off")
        ax_text.axis("off")

        pred = sample["predicted_caption"]
        refs = sample["reference_captions"]

        text = f"PREDICTION:\n{pred}\n\n"
        text += "REFERENCES:\n"
        for i, r in enumerate(refs[:3], 1):
            text += f"  {i}. {r}\n"

        ax_text.text(
            0, 0.95, text,
            transform=ax_text.transAxes,
            fontsize=10,
            verticalalignment="top",
            wrap=True,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0f4ff", alpha=0.8),
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate checkpoint and generate qualitative results for Task 2.2")
    parser.add_argument("--vit_model", default="nlpconnect/vit-gpt2-image-captioning", help="Path or HF ID for ViT model")
    parser.add_argument("--qwen_model", default="Qwen/Qwen1.5-4B-Chat", help="Path or HF ID for Qwen model")
    parser.add_argument("--lora_dir", default="models/checkpoint_best2_2", help="Path to LoRA adapters")
    parser.add_argument("--proj_path", default="projection.pt", help="Path to projection.pt")
    parser.add_argument("--img_dir", required=True, help="Path to validation images folder")
    parser.add_argument("--ann_file", required=True, help="Path to validation annotations JSON")
    parser.add_argument("--output_dir", type=str, default="outputs/qualitative_task2_2", help="Directory to save results")
    parser.add_argument("--n", type=int, default=20, help="Number of qualitative examples to generate")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for generation")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for DataLoader")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("[INFO] Device check...", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}", flush=True)

    # ---- Load frozen ViT ----
    print(f"[INFO] Loading ViT from {args.vit_model}", flush=True)
    _full = VisionEncoderDecoderModel.from_pretrained(args.vit_model)
    vit_encoder = _full.encoder
    feature_extractor = ViTImageProcessor.from_pretrained(args.vit_model)
    del _full
    vit_encoder.eval()
    for p in vit_encoder.parameters():
        p.requires_grad = False
    vit_hidden_size = vit_encoder.config.hidden_size

    # ---- Load Qwen3.5 base + LoRA ----
    print(f"[INFO] Loading Qwen base from {args.qwen_model}", flush=True)
    qwen_tokenizer = AutoTokenizer.from_pretrained(args.qwen_model)
    if qwen_tokenizer.pad_token is None:
        qwen_tokenizer.pad_token = qwen_tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.qwen_model,
        dtype=torch.bfloat16,
    )
    base_model.config.pad_token_id = qwen_tokenizer.pad_token_id

    print(f"[INFO] Loading LoRA adapters from {args.lora_dir}", flush=True)
    qwen_model = PeftModel.from_pretrained(base_model, args.lora_dir)
    qwen_hidden_size = qwen_model.config.hidden_size

    # ---- Build combined model ----
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

    # ---- Build validation dataset ----
    print("[INFO] Building validation dataset...", flush=True)
    val_dataset = VizWizDataset(
        img_dir=args.img_dir,
        ann_file=args.ann_file,
        feature_extractor=feature_extractor,
    )
    print(f"[INFO] Validation dataset size: {len(val_dataset)}", flush=True)

    dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    gen_kwargs = {
        "max_new_tokens": 64,
        "do_sample": False,
        "eos_token_id": qwen_tokenizer.eos_token_id,
        "pad_token_id": qwen_tokenizer.pad_token_id,
        "repetition_penalty": 1.3,
    }

    # ---- Generate predictions ----
    predictions, references, image_paths = [], [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating captions"):
            pixel_values = batch["pixel_values"].to(device)
            output_ids = model.generate(
                pixel_values=pixel_values,
                tokenizer=qwen_tokenizer,
                **gen_kwargs,
            )
            captions = qwen_tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            
            clean = []
            for c in captions:
                c = c.replace("<|im_start|>", "").replace("<|im_end|>", "")
                c = c.replace("assistant", "").replace("user", "")
                if "<think>" in c:
                    c = c.split("</think>")[-1]
                c = c.strip().lower()
                clean.append(c)

            predictions.extend(clean)
            references.extend(batch["captions"])
            image_paths.extend(batch["image_paths"])

    # ---- Make visualization grids ----
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Compute Metrics ----
    print("[INFO] Computing metrics...", flush=True)
    metrics = compute_metrics(predictions, references)
    print_metrics(metrics, title="Task 2.2 Qualitative Evaluation")

    # ---- Save all predictions as JSON ----
    preds_path = out_dir / "all_predictions.json"
    preds_data = {
        "metrics": metrics,
        "predictions": [
            {
                "image": img,
                "predicted_caption": pred,
                "reference_captions": refs,
            }
            for img, pred, refs in zip(image_paths, predictions, references)
        ]
    }
    with open(preds_path, "w") as f:
        json.dump(preds_data, f, indent=2)
    print(f"[INFO] Saved all predictions to {preds_path}")

    # ---- Sample and visualize ----
    # Extract prediction list for sampling
    preds_list = preds_data["predictions"]
    samples_viz = random.sample(preds_list, min(args.n, len(preds_list)))
    print(f"[INFO] Generating qualitative visualizations ({len(samples_viz)} samples)...", flush=True)

    # Individual examples
    for i, sample in enumerate(samples_viz):
        make_grid(
            [sample],
            str(out_dir / f"example_{i+1:02d}.png"),
            title=f"Example {i+1}",
        )

    # Text summary
    txt_path = out_dir / "qualitative_examples.txt"
    with open(txt_path, "w") as f:
        for i, sample in enumerate(samples_viz, 1):
            f.write(f"=== Example {i} ===\n")
            f.write(f"Image: {sample['image']}\n")
            f.write(f"Prediction: {sample['predicted_caption']}\n")
            f.write("References:\n")
            for j, ref in enumerate(sample["reference_captions"], 1):
                f.write(f"  {j}. {ref}\n")
            f.write("\n")
    print(f"[INFO] Saved text summary to {txt_path}")

    print(f"\n[INFO] ✓ Qualitative results saved to {out_dir}/")
    print(f"  - example_01.png, example_02.png, ... (individual)")
    print(f"  - qualitative_examples.txt (text summary)")
    print(f"  - all_predictions.json (all {len(preds_data)} predictions)")

if __name__ == "__main__":
    main()