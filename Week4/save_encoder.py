"""
Save the fine-tuned ViT encoder from a Task 1.2 checkpoint.

Usage:
    python save_encoder.py \
        --checkpoint_dir outputs/task1_2/best_model \
        --output_dir outputs/vit_encoder
"""

import argparse
from pathlib import Path
from transformers import VisionEncoderDecoderModel, ViTImageProcessor


def main():
    parser = argparse.ArgumentParser(
        description="Extract and save the ViT encoder from a Task 1.2 checkpoint"
    )
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Path to task 1.2 checkpoint (e.g. outputs/task1_2/best_model)")
    parser.add_argument("--output_dir", type=str, default="outputs/vit_encoder",
                        help="Where to save the extracted ViT encoder")
    parser.add_argument("--original_model", type=str,
                        default="nlpconnect/vit-gpt2-image-captioning",
                        help="Original model name (for loading the feature extractor)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading VisionEncoderDecoderModel from {args.checkpoint_dir}")
    model = VisionEncoderDecoderModel.from_pretrained(args.checkpoint_dir)

    # Extract and save the ViT encoder
    encoder = model.encoder
    encoder.save_pretrained(output_dir)
    print(f"[INFO] ViT encoder saved to {output_dir}")

    # Save the feature extractor alongside the encoder
    try:
        fe = ViTImageProcessor.from_pretrained(args.checkpoint_dir)
    except Exception:
        print("[INFO] Feature extractor not found in checkpoint, loading from original model...")
        fe = ViTImageProcessor.from_pretrained(args.original_model)
    fe.save_pretrained(output_dir)
    print(f"[INFO] Feature extractor saved to {output_dir}")

    # Print encoder info
    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"[INFO] Encoder parameters: {total_params:,}")
    print(f"[INFO] Hidden size: {encoder.config.hidden_size}")


if __name__ == "__main__":
    main()
