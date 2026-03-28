"""
Task 1.1 - Evaluate direct using vit-gpt2-image-captioning


Use:
    python task1_1_pretrained.py
        --img_dir data/vizwiz/val \
        --ann_file data/vizwiz/annotations/val.json
        --output_file outputs/task1_1
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent))
from src.dataset import VizWizDataset, collate_fn
from src.metrics import compute_metrics, print_metrics

def generate_captions(model, tokenizer, dataloader, device, gen_kwarges):
    predicitions, refences, image_paths = [], [], []


    model.eval()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating captions"):
            pixel_values = batch['pixel_values'].to(device)

            outputs_ids = model.generate(pixel_values=pixel_values, **gen_kwarges)

            captions = tokenizer.batch_decode(outputs_ids, skip_special_tokens=True)
            captions = [cap.strip().lower() for cap in captions]

            predicitions.extend(captions)
            refences.extend(batch['captions'])
            image_paths.extend(batch['image_paths'])
        
    
    return predicitions, refences, image_paths


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate pretrained ViT-GPT2 on VizWiz")
    parser.add_argument('--img_dir', type=str, required=True, 
                        help="Path to folder with .jpg images")
    parser.add_argument('--ann_file', type=str, required=True,
                         help="Path to val.json / test.json")
    parser.add_argument('--output_file', type=str,
                         default="outputs/task1_1", help="Path to save generated captions and metrics")
    parser.add_argument('--batch_size', type=int,
                         default=16, help="Batch size for generation")
    parser.add_argument('--num_workers', type=int,
                         default=4, help="Number of workers for DataLoader")
    parser.add_argument('--model_name', type=str,
                        default="nlpconnect/vit-gpt2-image-captioning", help="HuggingFace model name")
    parser.add_argument('--max_samples', type=int, default=None,
                        help="Limit number of samples to process (for quick testing)")
    return parser.parse_args()

def main():
    args = parse_args()

    print("[INFO] device check...", flush=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}", flush=True)

    # LOAD MODEL
    print(f"[INFO] Loading feature extractor: {args.model_name}", flush=True)
    feature_extractor = ViTImageProcessor.from_pretrained(args.model_name)
    print("[INFO] Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    print("[INFO] Loading model...", flush=True)
    model = VisionEncoderDecoderModel.from_pretrained(args.model_name).to(device)
    print("[INFO] Model loaded.", flush=True)

    gen_kwarges = {
        "max_new_tokens": 64,
        "num_beams": 4,
        "no_repeat_ngram_size": 3,
        "early_stopping": True
    }

    print("[INFO] Building dataset...", flush=True)
    dataset = VizWizDataset(
        img_dir=args.img_dir,
        ann_file=args.ann_file,
        feature_extractor=feature_extractor
    )

    if args.max_samples is not None:
        dataset.samples = dataset.samples[:args.max_samples]
        print(f"[INFO] Limiting to {args.max_samples} samples for quick testing", flush=True)

    print(f"[INFO] Building dataloader ({len(dataset)} samples, batch={args.batch_size})...", flush=True)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device == torch.device('cuda'))
    )

    # GENERATE CAPTIONS
    print("[INFO] Starting caption generation...", flush=True)
    predictions, references, image_paths = generate_captions(
        model, tokenizer, dataloader, device, gen_kwarges
    )
    print(f"[INFO] Generation done. {len(predictions)} captions.", flush=True)

    # COMPUTE METRICS
    print("[INFO] Computing metrics...", flush=True)
    metrics = compute_metrics(predictions, references)
    print_metrics(metrics, title="Pretrained ViT-GPT2 Metrics")

    # SAVE RESULTS
    out_dir = Path(args.output_file).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # METRICS
    metrics_path = Path(args.output_file + "_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump ({"model_name": args.model_name, "metrics": metrics}, f, indent=4)
    print(f"[INFO] Saved metrics to {metrics_path}")

    preds_path = out_dir / (Path(args.output_file).stem + "_predictions.json")
    with open(preds_path, 'w') as f:
        json.dump([
            {"image": p, "predicted_caption": pred, "reference_captions": refs}
            for p, pred, refs in zip(image_paths, predictions, references)
        ], f)
    print(f"[INFO] Saved predictions to {preds_path}")

if __name__ == "__main__":
    main()