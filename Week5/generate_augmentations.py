import json
import os
import argparse
from datetime import datetime

import torch
from diffusers import StableDiffusion3Pipeline

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate augmented images from captions using SD 3.5 Medium and produce a VizWiz-format JSON."
    )
    parser.add_argument("--captions", type=str, required=True, help="Path to txt file with one caption per line.")
    parser.add_argument("--output_dir", type=str, required=True, help="Root output directory.")
    parser.add_argument("--images_per_caption", type=int, default=1, help="Number of images to generate per caption.")
    return parser.parse_args()


def load_pipeline():
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-medium",
        torch_dtype=torch.float16,
    ).to("cuda")
    return pipe


def generate_image(pipe, caption):
    return pipe(
        caption,
        num_inference_steps=40,
        guidance_scale=4.5,
    ).images[0]


def main():
    args = parse_args()

    images_dir = os.path.join(args.output_dir, "augmented")
    annotations_dir = os.path.join(args.output_dir, "annotations")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(annotations_dir, exist_ok=True)

    with open(args.captions) as f:
        captions = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(captions)} captions from {args.captions}")

    pipe = load_pipeline()

    images_list = []
    annotations_list = []

    global_idx = 0
    total = len(captions) * args.images_per_caption

    for caption_idx, caption in enumerate(captions):
        for img_idx in range(args.images_per_caption):
            file_name = f"SD35_aug_{global_idx:08d}.jpg"
            image_path = os.path.join(images_dir, file_name)

            print(f"[{global_idx+1}/{total}] Caption {caption_idx+1}, image {img_idx+1}: {caption[:80]}")
            image = generate_image(pipe, caption)
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
            "description": "Augmented dataset generated with Stable Diffusion 3.5 Medium from text captions.",
            "license": {"url": "", "name": ""},
            "url": "",
            "version": "1.0",
            "year": datetime.now().year,
            "contributor": "SD 3.5 Medium",
            "date_created": datetime.now().strftime("%Y-%m-%d"),
        },
        "images": images_list,
        "annotations": annotations_list,
    }

    json_path = os.path.join(annotations_dir, "augmented_captions.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nDone. {len(images_list)} images saved to {images_dir}")
    print(f"JSON saved to {json_path}")


if __name__ == "__main__":
    main()
