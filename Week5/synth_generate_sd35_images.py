import json
import os
import argparse
from datetime import datetime

import torch
from diffusers import StableDiffusion3Pipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate augmented images from captions using SD 3.5 Medium and produce a VizWiz-format JSON."
    )
    parser.add_argument("--captions", type=str, required=True, help="Path to txt file with one caption per line.")
    parser.add_argument("--output_dir", type=str, required=True, help="Root output directory.")
    parser.add_argument("--images_per_caption", type=int, default=1, help="Number of images to generate per caption.")
    parser.add_argument("--model_path", type=str, default="./models/sd35-medium", help="Path to SD 3.5 model dir.")
    parser.add_argument("--lora_weights", type=str, default=None,
                        help="Path to LoRA weights dir (output of train_lora_sd.py). "
                             "If not set, uses the base model.")
    parser.add_argument("--lora_scale", type=float, default=1.0,
                        help="LoRA fusion scale (0 = base model, 1 = full LoRA, default 1.0).")
    return parser.parse_args()


def load_pipeline(model_path: str, lora_weights: str | None = None, lora_scale: float = 1.0):
    pipe = StableDiffusion3Pipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
    ).to("cuda")

    if lora_weights is not None:
        print(f"Loading LoRA weights from {lora_weights} (scale={lora_scale})")
        pipe.load_lora_weights(lora_weights)
        pipe.fuse_lora(lora_scale=lora_scale)

    return pipe


STYLE_SUFFIX = (
    "IMAGE STYLE: Close taken smartphone photo with potential photographic problems: "
    "Possible severe motion blur or out of focus, subject might be partially cut off, "
    "the photo can be accidental tilt, it can appear a shaky camera, the lighting might be "
    "imperfect or have shutter lag blur or poorly framed."
)


def generate_image(pipe, caption):
    prompt = f"{caption}. {STYLE_SUFFIX}"
    return pipe(
        prompt,
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

    pipe = load_pipeline(args.model_path, args.lora_weights, args.lora_scale)

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
