"""
qualitative_quality_issues.py

Qualitative comparison of three models on VizWiz "quality issues" images.
Filters images where at least one reference caption is the quality-issues string,
then shows side-by-side predictions from all three models.

Usage:
    python Week5/qualitative_quality_issues.py \
        --output_dir Week5/outputs/qualitative_qi \
        --n 20 \
        --seed 42
"""

import argparse
import json
import os
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
PRED_FILES = {
    "captions_epoch3": "Week5/predictions/captions_epoch3.json",
    "task_e_eval_3ds": "Week5/predictions/task_e_eval_3ds.json",
    "task_e_eval":     "Week5/predictions/task_e_eval.json",
}

LOCAL_IMG_ROOT = "Week4/data/val"
QI_KEYWORD     = "quality issues"

COLORS = {
    "captions_epoch3": "#fff4cc",
    "task_e_eval_3ds": "#ccf0ff",
    "task_e_eval":     "#ccffcc",
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def local_path(raw_path: str) -> str:
    """Convert any image path (relative or cluster-absolute) to local path."""
    filename = os.path.basename(raw_path)
    return os.path.join(LOCAL_IMG_ROOT, filename)


def load_predictions(path: str) -> list[dict]:
    """Load a predictions file and normalise to list of dicts with keys:
       image_path, predicted, references
    """
    with open(path) as f:
        data = json.load(f)

    if isinstance(data, list):
        # captions_epoch3 format
        return [
            {
                "image_path": local_path(item["image_path"]),
                "predicted":  item["predicted"],
                "references": item["references"],
            }
            for item in data
        ]
    else:
        # task_e_eval* format  {"metrics": ..., "predictions": [...]}
        return [
            {
                "image_path": local_path(item["image"]),
                "predicted":  item["predicted_caption"],
                "references": item["reference_captions"],
            }
            for item in data["predictions"]
        ]


def is_quality_issue(references: list[str]) -> bool:
    return any(QI_KEYWORD in r.lower() for r in references)


def build_index(preds: list[dict]) -> dict[str, dict]:
    """Index predictions by normalised filename."""
    return {os.path.basename(p["image_path"]): p for p in preds}


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def make_comparison_figure(filename: str, model_preds: dict, output_path: Path):
    """One figure per image: left = photo, right = predictions from each model."""
    n_models = len(model_preds)
    fig, axes = plt.subplots(1, n_models + 1, figsize=(6 * (n_models + 1), 6),
                             gridspec_kw={"width_ratios": [1.2] + [2] * n_models})

    # --- image panel ---
    img_path = next(iter(model_preds.values()))["image_path"]
    try:
        img = Image.open(img_path).convert("RGB")
        axes[0].imshow(img)
    except Exception:
        axes[0].text(0.5, 0.5, "Image\nnot found", ha="center", va="center",
                     transform=axes[0].transAxes, fontsize=12, color="red")
    axes[0].axis("off")
    axes[0].set_title(filename, fontsize=8, wrap=True)

    # --- one text panel per model ---
    for ax, (model_name, pred) in zip(axes[1:], model_preds.items()):
        ax.axis("off")
        refs = pred["references"]
        text = f"PREDICTION:\n{pred['predicted']}\n\n"
        text += "REFERENCES:\n"
        for j, r in enumerate(refs[:3], 1):
            text += f"  {j}. {r}\n"
        ax.text(
            0.02, 0.97, text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            wrap=True,
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor=COLORS.get(model_name, "#f0f0f0"),
                alpha=0.85,
            ),
        )
        ax.set_title(model_name, fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default="Week5/outputs/qualitative_qi")
    p.add_argument("--n",    type=int, default=20, help="Number of images to visualise")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    # Load all models
    print("Loading predictions...")
    all_indices: dict[str, dict[str, dict]] = {}
    for model_name, path in PRED_FILES.items():
        preds = load_predictions(path)
        all_indices[model_name] = build_index(preds)
        print(f"  {model_name}: {len(preds)} predictions loaded")

    # Find common filenames present in all models
    common_filenames = set.intersection(*[set(idx.keys()) for idx in all_indices.values()])
    print(f"Common images across all models: {len(common_filenames)}")

    # Filter to quality-issues images (using first model's references as ground truth)
    first_idx = next(iter(all_indices.values()))
    qi_filenames = [
        fn for fn in common_filenames
        if is_quality_issue(first_idx[fn]["references"])
    ]
    print(f"Quality-issues images: {len(qi_filenames)}")

    # Random sample
    sample = random.sample(qi_filenames, min(args.n, len(qi_filenames)))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save txt summary
    txt_path = out_dir / "qualitative_qi_examples.txt"
    with open(txt_path, "w") as f:
        for i, fn in enumerate(sample, 1):
            f.write(f"=== Example {i}: {fn} ===\n")
            for model_name, idx in all_indices.items():
                pred = idx[fn]
                f.write(f"  [{model_name}] {pred['predicted']}\n")
            refs = first_idx[fn]["references"]
            f.write(f"  [references] {refs[0]}\n\n")
    print(f"[Saved] {txt_path}")

    # Save individual comparison figures
    for i, fn in enumerate(sample, 1):
        model_preds = {model_name: idx[fn] for model_name, idx in all_indices.items()}
        out_path = out_dir / f"qi_example_{i:02d}_{fn.replace('.jpg', '')}.png"
        make_comparison_figure(fn, model_preds, out_path)
        print(f"[{i}/{len(sample)}] Saved {out_path.name}")

    print(f"\nDone. {len(sample)} examples saved in {out_dir}/")


if __name__ == "__main__":
    main()
