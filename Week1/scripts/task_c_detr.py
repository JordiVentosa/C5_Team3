import argparse
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from Week1.src.models.huggingface_detr import load_detr_model, run_inference, draw_detections
from Week1.src.utils.kitti_helpers import list_kitti_rgb_images, rel_from_split

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def get_args():
    parser = argparse.ArgumentParser(description="Task C: Qualitative DeTR inference on KITTI-MOTS")
    parser.add_argument("--kitti_root", type=str, default="/home/mcv/datasets/C5/KITTI-MOTS")
    parser.add_argument("--split", type=str, default="testing", choices=["training", "testing"])
    parser.add_argument("--checkpoint", type=str, default="facebook/detr-resnet-50")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max_images", type=int, default=200)
    parser.add_argument("--output_dir", type=str, default="./out_detr_kitti_mots_task_c")
    return parser.parse_args()


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    kitti_root = Path(args.kitti_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    keep_classes = {"car", "person"}

    processor, model = load_detr_model(args.checkpoint, device)
    id2label = dict(model.config.id2label)

    img_paths = list_kitti_rgb_images(kitti_root, args.split)
    print(f"Found {len(img_paths)} images in {args.split}/image_02")

    if args.max_images > 0:
        img_paths = img_paths[:args.max_images]
        print(f"Using first {len(img_paths)} images")

    with torch.no_grad():
        for p in tqdm(img_paths, desc="DeTR inference"):
            img = Image.open(p).convert("RGB")
            results = run_inference(model, processor, img, device, args.threshold)
            vis = draw_detections(img, results, id2label=id2label, keep_labels=keep_classes)

            rel = rel_from_split(p, kitti_root, args.split)
            save_path = output_dir / args.split / rel
            save_path.parent.mkdir(parents=True, exist_ok=True)
            vis.save(save_path.with_suffix(".png"))

    print("Saved to:", output_dir.resolve())


if __name__ == "__main__":
    main()
