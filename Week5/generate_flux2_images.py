# generate_flux2_images.py
import torch
import os
import json
import argparse
from pathlib import Path
from diffusers import Flux2Pipeline, AutoModel
from transformers import Mistral3ForConditionalGeneration

STYLE_SUFFIX = "IMAGE STYLE: Close taken smartphone photo with severe photographic problems: Severe motion blur or out of focus, subject is partially cut off, the photo is an accidental tilt, it appears a shaky camera, the lighting has hutter lag blur and is poorly framed."

NUM_INFERENCE_STEPS = 20
GUIDANCE_SCALE      = 5.0
SEED                = 42
DEVICE              = "cuda:0"
IMAGE_ID_OFFSET     = 90000


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic_json",  type=str, required=True)
    parser.add_argument("--output_img_dir",  type=str, required=True)
    parser.add_argument("--output_ann_file", type=str, required=True)
    parser.add_argument("--repo_id",         type=str, required=True)
    parser.add_argument("--start_idx",       type=int, default=0)
    parser.add_argument("--end_idx",         type=int, default=None)
    parser.add_argument("--batch_size",      type=int, default=2)
    return parser.parse_args()


def load_pipeline(repo_id):
    print("[INFO] Loading Flux2 pipeline...")
    text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
        repo_id, subfolder="text_encoder", torch_dtype=torch.bfloat16, device_map="cpu"
    )
    dit = AutoModel.from_pretrained(
        repo_id, subfolder="transformer", torch_dtype=torch.bfloat16, device_map="cpu"
    )
    pipe = Flux2Pipeline.from_pretrained(
        repo_id, text_encoder=text_encoder, transformer=dit, torch_dtype=torch.bfloat16
    )
    pipe.enable_model_cpu_offload()
    print("[INFO] Flux2 pipeline ready.")
    return pipe


def main():
    args = parse_args()

    with open(args.synthetic_json, "r") as f:
        synthetic_data = json.load(f)

    synthetic_data = synthetic_data[args.start_idx:args.end_idx]
    print(f"[INFO] Processing {len(synthetic_data)} entries "
          f"(idx {args.start_idx} to {args.end_idx or 'end'})")

    Path(args.output_img_dir).mkdir(parents=True, exist_ok=True)
    Path(args.output_ann_file).parent.mkdir(parents=True, exist_ok=True)

    pipe = load_pipeline(args.repo_id)

    ann_images      = []
    ann_annotations = []
    caption_id      = args.start_idx * 5 + 1
    image_id_start  = IMAGE_ID_OFFSET + args.start_idx

    total = len(synthetic_data)
    for i in range(0, total, args.batch_size):
        batch_entries = synthetic_data[i:i + args.batch_size]
        prompts = [e["global_caption"] + " " + STYLE_SUFFIX for e in batch_entries]

        print(f"[{i+1}-{min(i+args.batch_size, total)}/{total}] Generating batch...")
        for p in prompts:
            print(f"  prompt: {p[:100]}...")

        images = pipe(
            prompt=prompts,
            generator=torch.Generator(device=DEVICE).manual_seed(SEED),
            num_inference_steps=NUM_INFERENCE_STEPS,
            guidance_scale=GUIDANCE_SCALE,
        ).images

        for j, (entry, image) in enumerate(zip(batch_entries, images)):
            global_idx = args.start_idx + i + j
            image_id   = IMAGE_ID_OFFSET + global_idx
            file_name  = f"synthetic_{global_idx:05d}.jpg"
            img_path   = os.path.join(args.output_img_dir, file_name)
            image.save(img_path)

            ann_images.append({
                "file_name":     file_name,
                "id":            image_id,
                "text_detected": False,
            })

            for cap in entry["captions"]:
                ann_annotations.append({
                    "image_id": image_id,
                    "id":       caption_id,
                    "caption":  cap,
                })
                caption_id += 1

        # Guardado incremental por si peta
        vizwiz_ann = {
            "info": {
                "description": "Synthetic captions generated with Qwen3.5 + Flux2.",
                "version": "1.0",
                "year": 2026,
            },
            "images":      ann_images,
            "annotations": ann_annotations,
        }
        with open(args.output_ann_file, "w") as f:
            json.dump(vizwiz_ann, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Done. {len(ann_images)} images saved to {args.output_img_dir}")
    print(f"[INFO] Annotations saved to {args.output_ann_file}")


if __name__ == "__main__":
    main()