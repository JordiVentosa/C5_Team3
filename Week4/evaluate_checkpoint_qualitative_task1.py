"""
Evaluate checkpoint and generate qualitative results

Loads a fine-tuned ViT-GPT2 checkpoint and generates qualitative visualizations.

Usage:
    python evaluate_checkpoint_qualitative.py \
        --checkpoint_dir checkpoint_epoch22 \
        --img_dir ../Week3/data/vizwiz/val \
        --ann_file ../Week3/data/vizwiz/annotations/val.json \
        --output_dir outputs/qualitative_epoch22 \
        --n 20
"""

import argparse
import json
import sys
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent))
from src.dataset import VizWizDataset, collate_fn


def generate_captions(model, tokenizer, dataloader, device, gen_kwargs):
    """Generate captions using the model."""
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


def make_grid(samples, output_path, title="Qualitative results"):
    """Generate visualization grid (code from visualize_predictions.py)."""
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
    parser = argparse.ArgumentParser(
        description="Evaluate checkpoint and generate qualitative results"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to checkpoint directory (e.g., checkpoint_epoch22)",
    )
    parser.add_argument(
        "--feature_extractor_name",
        type=str,
        default="nlpconnect/vit-gpt2-image-captioning",
        help=(
            "Fallback image processor source if checkpoint has no "
            "preprocessor_config.json"
        ),
    )
    parser.add_argument(
        "--img_dir", type=str, required=True, help="Path to validation images folder"
    )
    parser.add_argument(
        "--ann_file",
        type=str,
        required=True,
        help="Path to validation annotations JSON",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/qualitative",
        help="Directory to save results",
    )
    parser.add_argument(
        "--n", type=int, default=10, help="Number of qualitative examples to generate"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for generation"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers for DataLoader"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("[INFO] Device check...", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}", flush=True)

    # ---- Load model components from checkpoint ----
    print("[INFO] Loading feature extractor...", flush=True)
    try:
        feature_extractor = ViTImageProcessor.from_pretrained(args.checkpoint_dir)
        print("[INFO] Loaded feature extractor from checkpoint.", flush=True)
    except OSError:
        print(
            "[WARN] Checkpoint has no image processor config. "
            f"Falling back to: {args.feature_extractor_name}",
            flush=True,
        )
        feature_extractor = ViTImageProcessor.from_pretrained(args.feature_extractor_name)
    print("[INFO] Loading tokenizer from checkpoint...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_dir)
    print(f"[INFO] Loading model from checkpoint: {args.checkpoint_dir}", flush=True)
    model = VisionEncoderDecoderModel.from_pretrained(args.checkpoint_dir).to(device)
    print("[INFO] Model loaded.", flush=True)

    # ---- Build validation dataset ----
    print("[INFO] Building validation dataset...", flush=True)
    val_dataset = VizWizDataset(
        img_dir=args.img_dir,
        ann_file=args.ann_file,
        feature_extractor=feature_extractor,
    )
    print(f"[INFO] Validation dataset size: {len(val_dataset)}", flush=True)

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    # ---- Generate captions ----
    gen_kwargs = {
        "max_new_tokens": 64,
        "num_beams": 4,
        "no_repeat_ngram_size": 3,
        "early_stopping": True,
    }

    print("[INFO] Starting caption generation...", flush=True)
    predictions, references, image_paths = generate_captions(
        model, tokenizer, val_loader, device, gen_kwargs
    )
    print(f"[INFO] Generation done. {len(predictions)} captions.", flush=True)

    # ---- Prepare output directory ----
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Save all predictions as JSON ----
    preds_path = out_dir / "all_predictions.json"
    preds_data = [
        {
            "image": img,
            "predicted_caption": pred,
            "reference_captions": refs,
        }
        for img, pred, refs in zip(image_paths, predictions, references)
    ]
    with open(preds_path, "w") as f:
        json.dump(preds_data, f, indent=2)
    print(f"[INFO] Saved all predictions to {preds_path}")

    # ---- Sample and visualize ----
    samples = random.sample(preds_data, min(args.n, len(preds_data)))
    print(f"[INFO] Generating qualitative visualizations ({len(samples)} samples)...", flush=True)

    # Single grid with all examples
    """
    make_grid(
        samples,
        out_dir / "qualitative_examples.png",
        title=f"Qualitative results — {len(samples)} random samples",
    )
    """

    # Individual examples
    for i, sample in enumerate(samples):
        make_grid(
            [sample],
            out_dir / f"example_{i+1:02d}.png",
            title=f"Example {i+1}",
        )

    # Text summary
    txt_path = out_dir / "qualitative_examples.txt"
    with open(txt_path, "w") as f:
        for i, sample in enumerate(samples, 1):
            f.write(f"=== Example {i} ===\n")
            f.write(f"Image: {sample['image']}\n")
            f.write(f"Prediction: {sample['predicted_caption']}\n")
            f.write("References:\n")
            for j, ref in enumerate(sample["reference_captions"], 1):
                f.write(f"  {j}. {ref}\n")
            f.write("\n")
    print(f"[INFO] Saved text summary to {txt_path}")

    print(f"\n[INFO] ✓ Qualitative results saved to {out_dir}/")
    print(f"  - qualitative_examples.png (grid)")
    print(f"  - example_01.png, example_02.png, ... (individual)")
    print(f"  - qualitative_examples.txt (text summary)")
    print(f"  - all_predictions.json (all {len(preds_data)} predictions)")


if __name__ == "__main__":
    main()
