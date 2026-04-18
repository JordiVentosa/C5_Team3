import json
import os
import argparse
import subprocess
from datetime import datetime


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


os.environ["CUDA_VISIBLE_DEVICES"] = "2"
print(f"Using GPU: {os.environ['CUDA_VISIBLE_DEVICES']}")

import torch
from diffusers import Flux2Pipeline, AutoModel, StableDiffusion3Pipeline
from transformers import Mistral3ForConditionalGeneration

STYLE_PREFIX = "extremely blurry, severe motion blur, heavily out of focus, dark, grainy, accidental snapshot,"
NEGATIVE_PROMPT = "sharp focus, professional photography, high quality, well-lit, perfect exposure, well-composed, studio photo"

FINETUNED_BASE = "stabilityai/stable-diffusion-3.5-medium"
FINETUNED_LORA = os.path.join(os.path.dirname(__file__), "checkpoints/finetuned/vizwiz_lora.safetensors")

MAX_CAPTIONS = 2000


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate augmented images from VizWiz train captions."
    )
    parser.add_argument("--ann_file", type=str, default="../Week4/data/annotations/train.json",
                        help="Path to VizWiz train.json annotations file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Root output directory.")
    parser.add_argument("--images_per_caption", type=int, default=1, help="Number of images to generate per caption.")
    parser.add_argument("--model", type=str, default="flux2-4bit",
                        choices=["flux2-4bit", "sd3.5-large", "sd3-finetuned"],
                        help="Model to use for generation.")
    return parser.parse_args()


def load_clean_captions(ann_file: str) -> list[str]:
    BAD = "quality issues are too severe to recognize visual content."

    with open(ann_file) as f:
        data = json.load(f)

    captions_by_id: dict[int, list[str]] = {}
    for ann in data["annotations"]:
        iid = ann["image_id"]
        captions_by_id.setdefault(iid, []).append(ann["caption"].strip().lower())

    all_clean_captions = []
    for captions in captions_by_id.values():
        clean = [c for c in captions if BAD not in c]
        if len(clean) >= 2:
            all_clean_captions.extend(clean)

    return all_clean_captions


def load_pipeline(model: str):
    if model == "flux2-4bit":
        repo_id = "diffusers/FLUX.2-dev-bnb-4bit"
        text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
            repo_id, subfolder="text_encoder", torch_dtype=torch.bfloat16, device_map="cuda"
        )
        transformer = AutoModel.from_pretrained(
            repo_id, subfolder="transformer", torch_dtype=torch.bfloat16, device_map="cuda"
        )
        pipe = Flux2Pipeline.from_pretrained(
            repo_id, text_encoder=text_encoder, transformer=transformer, torch_dtype=torch.bfloat16
        )
        pipe.vae = pipe.vae.to("cuda")
    elif model == "sd3.5-large":
        pipe = StableDiffusion3Pipeline.from_pretrained(
            "stabilityai/stable-diffusion-3.5-large", torch_dtype=torch.bfloat16
        ).to("cuda")
    elif model == "sd3-finetuned":
        pipe = StableDiffusion3Pipeline.from_pretrained(
            FINETUNED_BASE, torch_dtype=torch.bfloat16
        )
        pipe.load_lora_weights(
            os.path.dirname(FINETUNED_LORA),
            weight_name=os.path.basename(FINETUNED_LORA),
        )
        pipe = pipe.to("cuda")
    return pipe


def generate_image(pipe, prompt: str, model: str):
    if model == "flux2-4bit":
        return pipe(prompt=prompt, num_inference_steps=50, guidance_scale=4.0).images[0]
    elif model in ("sd3.5-large", "sd3-finetuned"):
        return pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            num_inference_steps=40,
            guidance_scale=7.5,
        ).images[0]


def main():
    args = parse_args()

    images_dir = os.path.join(args.output_dir, "augmented")
    annotations_dir = os.path.join(args.output_dir, "annotations")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(annotations_dir, exist_ok=True)

    all_captions = load_clean_captions(args.ann_file)
    captions = all_captions[:MAX_CAPTIONS]
    print(f"Collected {len(all_captions)} clean captions total, using first {len(captions)}.")

    pipe = load_pipeline(args.model)
    print(f"Using model: {args.model}")

    json_path = os.path.join(annotations_dir, "augmented_captions.json")
    images_list = []
    annotations_list = []
    total = len(captions) * args.images_per_caption
    global_idx = 0

    for caption_idx, caption in enumerate(captions):
        prompt = (STYLE_PREFIX + " " + caption).strip()
        for img_idx in range(args.images_per_caption):
            prefix = "SD3_lora_aug" if args.model == "sd3-finetuned" else "FLUX2_aug"
            file_name = f"{prefix}_{global_idx:08d}.jpg"
            image_path = os.path.join(images_dir, file_name)

            print(f"[{global_idx+1}/{total}] Caption {caption_idx+1}, image {img_idx+1}: {prompt[:80]}")
            image = generate_image(pipe, prompt, args.model)
            image.save(image_path)

            images_list.append({
                "file_name": file_name,
                "vizwiz_url": "",
                "id": global_idx,
                "text_detected": False,
            })
            annotations_list.append({
                "caption": caption,
                "image_id": global_idx,
                "is_precanned": False,
                "is_rejected": False,
                "id": global_idx,
                "text_detected": False,
            })

            global_idx += 1

            output = {
                "info": {
                    "description": "Augmented dataset generated with FLUX.2-dev 4-bit from VizWiz train captions.",
                    "license": {"url": "", "name": ""},
                    "url": "",
                    "version": "1.0",
                    "year": datetime.now().year,
                    "contributor": "FLUX.2-dev 4-bit",
                    "date_created": datetime.now().strftime("%Y-%m-%d"),
                },
                "images": images_list,
                "annotations": annotations_list,
            }
            with open(json_path, "w") as f:
                json.dump(output, f, indent=2)

    print(f"\nDone. {len(images_list)} images saved to {images_dir}")
    print(f"JSON saved to {json_path}")


if __name__ == "__main__":
    main()
