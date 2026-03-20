import torch
from pathlib import Path
import argparse
from PIL import Image

from models.model import Model
from models.train_wrapper import CaptioningModule


def load_and_preprocess_image(image_path: str, device: str):
    from torchvision import transforms
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    return img_tensor


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    print(f"Loading model from {args.checkpoint}...")
    model = Model(device=device, resnet_model=args.resnet_model)
    module = CaptioningModule(model=model, device=device)
    
    if Path(args.checkpoint).exists():
        module.load_checkpoint(args.checkpoint)
        print("✓ Model loaded successfully")
    else:
        print(f"⚠ Checkpoint not found: {args.checkpoint}")
        print("Using untrained model (will generate random captions)")
    
    if Path(args.image_path).is_dir():
        image_paths = list(Path(args.image_path).glob("*.jpg")) + list(Path(args.image_path).glob("*.png"))
        print(f"\nFound {len(image_paths)} images in directory")
    else:
        image_paths = [Path(args.image_path)]
    
    print("\nGenerating captions...")
    print("="*80)
    
    for img_path in image_paths[:args.max_images]:
        try:
            img_tensor = load_and_preprocess_image(str(img_path), device)
            captions = module.predict(img_tensor)
            
            print(f"\nImage: {img_path.name}")
            print(f"Caption: {captions[0]}")
            print("-"*80)
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")
    
    print("\n" + "="*80)
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate captions for images")
    parser.add_argument("image_path", type=str, help="Path to image file or directory")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/best_model.pth", help="Model checkpoint path")
    parser.add_argument("--resnet_model", type=str, default="microsoft/resnet-18", help="ResNet variant")
    parser.add_argument("--max_images", type=int, default=10, help="Max images to process")
    
    args = parser.parse_args()
    main(args)
