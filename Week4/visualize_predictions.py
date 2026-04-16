"""
visualize_predictions.py
Reads the predictions JSON and generates an image with qualitative examples.
Usage:
    python visualize_predictions.py \
        --preds_file outputs/results/task1_1_val_predictions.json \
        --output_dir outputs/qualitative \
        --n 10
"""

import argparse
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # No display, works on cluster
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image


def make_grid(samples, output_path, title="Qualitative results"):
    n = len(samples)
    fig, axes = plt.subplots(n, 1, figsize=(12, 5 * n))
    if n == 1:
        axes = [axes]

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.001)

    for ax, sample in zip(axes, samples):
        # Image on the left
        img = Image.open(sample["image"]).convert("RGB")
        
        # Split the axis into image + text
        ax.axis("off")
        
        # Create sub-axes
        divider_pos = 0.35   # 35% for image, 65% for text
        
        ax_img  = ax.inset_axes([0, 0, divider_pos, 1])
        ax_text = ax.inset_axes([divider_pos + 0.02, 0, 1 - divider_pos - 0.02, 1])
        
        ax_img.imshow(img)
        ax_img.axis("off")
        
        ax_text.axis("off")
        
        pred = sample["predicted_caption"]
        refs = sample.get("reference_captions", [])

        # Formatted text
        text  = f"PREDICTION:\n{pred}\n\n"
        if refs:
            text += "REFERENCES:\n"
            for i, r in enumerate(refs[:3], 1):   # maximum 3 references
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds_file",  required=True,
                        help="JSON with predictions (output of task1_1)")
    parser.add_argument("--output_dir",  default="outputs/qualitative")
    parser.add_argument("--n",           type=int, default=10,
                        help="Number of examples to visualize")
    parser.add_argument("--seed",        type=int, default=42)
    return parser.parse_args()


def main():
    print("Starting visualization...")
    args = parse_args()
    random.seed(args.seed)

    print(f"Reading predictions from {args.preds_file}...")
    with open(args.preds_file) as f:
        data = json.load(f)

    # Adaptarlo si el JSON es un diccionario con keys "metrics" y "predictions"
    if isinstance(data, dict) and "predictions" in data:
        all_preds = data["predictions"]
    else:
        all_preds = data

    # Random sample
    samples = random.sample(all_preds, min(args.n, len(all_preds)))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    """
    # A single image with all examples
    make_grid(samples, out_dir / "qualitative_examples.png",
              title=f"Qualitative results — {args.n} random samples")
    """


        # Also save to .txt
    txt_path = out_dir / "qualitative_examples.txt"
    with open(txt_path, "w") as f:
        for i, sample in enumerate(samples, 1):
            f.write(f"=== Example {i} ===\n")
            f.write(f"Image: {sample['image']}\n")
            f.write(f"Prediction: {sample['predicted_caption']}\n")
            refs = sample.get("reference_captions", [])
            if refs:
                f.write("References:\n")
                for j, ref in enumerate(refs, 1):
                    f.write(f"  {j}. {ref}\n")
            f.write("\n")
    print(f"[Saved] {txt_path}")


    print(f"Selected {len(samples)} random examples for visualization. Saving each example individually...")
    # Also save each example individually
    for i, sample in enumerate(samples):
        make_grid([sample], out_dir / f"example_{i+1:02d}.png",
                  title=f"Example {i+1}")



    print(f"\n[Info] {len(samples)} examples saved in {out_dir}/")


if __name__ == "__main__":
    main()