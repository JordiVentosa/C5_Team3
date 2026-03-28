"""
src/dataset.py
VizWiz dataset loader for image captioning.

VizWiz annotation format (same as COCO):
{
  "images":      [{"id": int, "file_name": str}, ...],
  "annotations": [{"image_id": int, "caption": str}, ...]
}
"""

import json
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

class VizWizDataset(Dataset):
    def __init__(self, img_dir: str, ann_file:str, feature_extractor=None):
        """
        img_dir         : path to folder with .jpg images
        ann_file        : path to val.json / test.json
        feature_extractor: HuggingFace ViTImageProcessor (optional,
                          if None return raw PIL images)
        """
        self.img_dir = Path(img_dir)
        self.feature_extractor = feature_extractor

        with open(ann_file) as f:
            data = json.load(f)

        self.id2file = {img['id']: img['file_name'] for img in data['images']}

        captions_by_id = {}
        for ann in data['annotations']:
            iid = ann['image_id']
            captions_by_id.setdefault(iid, []).append(ann['caption'].strip().lower())

        self.samples = []
        for iid, captions in captions_by_id.items():
            fname = self.id2file.get(iid)
            if fname is None:
                continue
            img_path = self.img_dir / fname
            if img_path.exists():
                self.samples.append({
                    "image_id": iid,
                    "image_path": str(img_path),
                    "captions": captions
                })
        print (f"[VizWizDataset] Loaded {len(self.samples)} samples from {ann_file}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = Image.open(sample['image_path']).convert('RGB')

        if self.feature_extractor is not None:
            pixel_values = self.feature_extractor(
                images=img, return_tensors="pt"
            ).pixel_values.squeeze(0) # [3, H, W]
            return {
                "pixel_values": pixel_values,
                "captions": sample['captions'],
                "image_path": sample['image_path'],
            }

        return {
            "image": img,
            "captions": sample['captions'],
            "image_path": sample['image_path'],
        }

def collate_fn(batch):
    """
    Collate a batch for DataLoaded when use feature_extractor.
    pixel_values are tensors, captions and image_paths are lists.
    """
    import torch
    return {
        "pixel_values": torch.stack([item['pixel_values'] for item in batch]),
        "captions": [item['captions'] for item in batch],
        "image_paths": [item['image_path'] for item in batch],
    }