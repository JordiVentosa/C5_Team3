"""
Task 2.1 - Evaluate using Qwen2.5-VL-7B-Instruct on VizWiz

Use:
    python task2_1.py \
        --img_dir data/vizwiz/val \
        --ann_file data/vizwiz/annotations/val.json \
        --output_file outputs/task2_1
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent))
from src.dataset import VizWizDataset
from src.metrics import compute_metrics, print_metrics


CAPTION_PROMPT = "Describe this image briefly."


def make_messages(pil_image):
    return [
        {"role": "user", "content": [
            {"type": "image", "image": pil_image},
            {"type": "text", "text": CAPTION_PROMPT},
        ]}
    ]


def pil_collate_fn(batch):
    return {
        "images":      [item["image"]      for item in batch],
        "captions":    [item["captions"]   for item in batch],
        "image_paths": [item["image_path"] for item in batch],
    }


def generate_captions(model, processor, dataloader, device, gen_kwargs):
    predictions, references, image_paths = [], [], []
    model.eval()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating captions"):
            batch_messages = [make_messages(img) for img in batch["images"]]

            texts = [
                processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                for msgs in batch_messages
            ]

            # Extract PIL images from the messages structure
            image_inputs = [
                content["image"]
                for msgs in batch_messages
                for msg in msgs
                for content in msg["content"]
                if content["type"] == "image"
            ]

            inputs = processor(
                text=texts,
                images=image_inputs,
                return_tensors="pt",
                padding=True,
            ).to(device)

            input_len = inputs["input_ids"].shape[-1]
            output_ids = model.generate(**inputs, **gen_kwargs)

            # Decode only the newly generated tokens (skip the input prompt)
            new_tokens = output_ids[:, input_len:]
            captions = processor.batch_decode(new_tokens, skip_special_tokens=True)
            captions = [cap.strip().lower() for cap in captions]

            predictions.extend(captions)
            references.extend(batch["captions"])
            image_paths.extend(batch["image_paths"])

    return predictions, references, image_paths


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Qwen2.5-VL on VizWiz")
    parser.add_argument('--img_dir', type=str, required=True,
                        help="Path to folder with .jpg images")
    parser.add_argument('--ann_file', type=str, required=True,
                        help="Path to val.json / test.json")
    parser.add_argument('--output_file', type=str,
                        default="outputs/task2_1", help="Path to save generated captions and metrics")
    parser.add_argument('--batch_size', type=int,
                        default=4, help="Batch size for generation")
    parser.add_argument('--num_workers', type=int,
                        default=4, help="Number of workers for DataLoader")
    parser.add_argument('--model_name', type=str,
                        default="Qwen/Qwen2.5-VL-7B-Instruct",
                        help="HuggingFace model name")
    parser.add_argument('--max_samples', type=int, default=None,
                        help="Limit number of samples to process (for quick testing)")
    return parser.parse_args()


def main():
    args = parse_args()

    print("[INFO] device check...", flush=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}", flush=True)

    # LOAD MODEL
    print(f"[INFO] Loading processor: {args.model_name}", flush=True)
    processor = AutoProcessor.from_pretrained(args.model_name)
    processor.tokenizer.padding_side = "left"
    print("[INFO] Loading model...", flush=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
    ).to(device)
    print("[INFO] Model loaded.", flush=True)

    gen_kwargs = {
        "max_new_tokens": 64,
        "do_sample": False,
    }

    print("[INFO] Building dataset...", flush=True)
    dataset = VizWizDataset(
        img_dir=args.img_dir,
        ann_file=args.ann_file,
        feature_extractor=None,  # return raw PIL images for Qwen's AutoProcessor
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
        collate_fn=pil_collate_fn,
        pin_memory=False,
    )

    # GENERATE CAPTIONS
    print("[INFO] Starting caption generation...", flush=True)
    predictions, references, image_paths = generate_captions(
        model, processor, dataloader, device, gen_kwargs
    )
    print(f"[INFO] Generation done. {len(predictions)} captions.", flush=True)

    # COMPUTE METRICS
    print("[INFO] Computing metrics...", flush=True)
    metrics = compute_metrics(predictions, references)
    print_metrics(metrics, title="Qwen2.5-VL-7B Metrics")

    # SAVE RESULTS
    out_dir = Path(args.output_file).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = Path(args.output_file + "_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump({"model_name": args.model_name, "metrics": metrics}, f, indent=4)
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
