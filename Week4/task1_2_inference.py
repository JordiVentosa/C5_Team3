"""
Task 1.2 - Inference using a fine-tuned VisionEncoderDecoder checkpoint

The checkpoint saved by task1_2_finetune.py contains the model weights and
tokenizer, but NOT the image processor (feature_extractor).  You must supply
the original encoder name so the correct processor can be loaded from HuggingFace.

Use:
    python task1_2_inference.py \
        --checkpoint_dir  outputs/vit_gpt2/best_model \
        --encoder_name_or_path google/vit-base-patch16-224-in21k \
        --img_dir   data/vizwiz/val \
        --ann_file  data/vizwiz/annotations/val.json \
        --output_file outputs/task1_2_inference
"""
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import VisionEncoderDecoderModel, AutoImageProcessor, AutoTokenizer
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent))
from src.dataset import VizWizDataset, collate_fn


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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inference with a task1_2 fine-tuned checkpoint"
    )
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Path to a checkpoint saved by task1_2_finetune.py "
                             "(e.g. outputs/vit_gpt2/best_model)")
    parser.add_argument("--encoder_name_or_path", type=str, required=True,
                        help="HuggingFace encoder used during fine-tuning — needed to "
                             "reload the image processor (not saved in the checkpoint)")
    parser.add_argument("--img_dir",    type=str, required=True,
                        help="Path to folder with .jpg images")
    parser.add_argument("--ann_file",   type=str, required=True,
                        help="Path to val.json / test.json")
    parser.add_argument("--output_file", type=str, default="outputs/task1_2_inference",
                        help="Prefix for output files (metrics + predictions)")
    parser.add_argument("--batch_size",  type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit number of samples (quick testing)")
    parser.add_argument("--num_beams",      type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    return parser.parse_args()


def main():
    args = parse_args()

    print("[INFO] Device check...", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}", flush=True)

    # --- Load image processor from the original encoder (not in checkpoint) ---
    print(f"[INFO] Loading image processor from: {args.encoder_name_or_path}", flush=True)
    feature_extractor = AutoImageProcessor.from_pretrained(args.encoder_name_or_path)

    # --- Load tokenizer and model from checkpoint ---
    print(f"[INFO] Loading tokenizer from checkpoint: {args.checkpoint_dir}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[INFO] Loading model from checkpoint: {args.checkpoint_dir}", flush=True)
    model = VisionEncoderDecoderModel.from_pretrained(args.checkpoint_dir).to(device)
    print("[INFO] Model loaded.", flush=True)

    gen_kwargs = {
        "max_new_tokens":       args.max_new_tokens,
        "num_beams":            args.num_beams,
        "no_repeat_ngram_size": 3,
        "early_stopping":       args.num_beams > 1,
        "pad_token_id":         tokenizer.pad_token_id,
    }

    # --- Dataset + DataLoader ---
    print("[INFO] Building dataset...", flush=True)
    dataset = VizWizDataset(
        img_dir=args.img_dir,
        ann_file=args.ann_file,
        feature_extractor=feature_extractor,
    )
    if args.max_samples is not None:
        dataset.samples = dataset.samples[:args.max_samples]
        print(f"[INFO] Limiting to {args.max_samples} samples.", flush=True)

    print(f"[INFO] Building dataloader ({len(dataset)} samples, batch={args.batch_size})...",
          flush=True)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    # --- Generate ---
    print("[INFO] Starting caption generation...", flush=True)
    predictions, references, image_paths = generate_captions(
        model, tokenizer, dataloader, device, gen_kwargs
    )
    print(f"[INFO] Generation done. {len(predictions)} captions.", flush=True)

    # --- Save ---
    out_dir = Path(args.output_file).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    preds_path = out_dir / (Path(args.output_file).stem + "_predictions.json")
    with open(preds_path, "w") as f:
        json.dump([
            {"image": p, "predicted_caption": pred, "reference_captions": refs}
            for p, pred, refs in zip(image_paths, predictions, references)
        ], f)
    print(f"[INFO] Saved predictions to {preds_path}")


if __name__ == "__main__":
    main()
