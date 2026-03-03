import os
import glob
import cv2
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from pycocotools import mask as mask_utils
from src.utils.mots import load_seqmap, load_txt, filename_to_frame_nr


CLASS_MAPPING = {
    1: 3,  # car → COCO car
    2: 1,  # pedestrian → COCO person
}


class KittyDataset(Dataset):
    def __init__(self, root_dir, mode="train", transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.annotations = []
        _utils_dir = os.path.dirname(os.path.abspath(__file__))
        if mode == "train":
            seq_files = [os.path.join(_utils_dir, "train.seqmap")]
        elif mode == "val":
            seq_files = [os.path.join(_utils_dir, "val.seqmap")]
        else:
            seq_files = [
                os.path.join(_utils_dir, "train.seqmap"),
                os.path.join(_utils_dir, "val.seqmap"),
            ]
        self.load_metadata(seq_files)

    def load_metadata(self, sequence_maps):
        for seqmap in sequence_maps:
            seqmaps, _ = load_seqmap(seqmap)
            for seq in seqmaps:
                seq_path_txt = os.path.join(self.root_dir, "instances_txt", f"{seq}.txt")
                seq_path_folder = os.path.join(self.root_dir, "training", "image_02", seq)
                text = load_txt(seq_path_txt)
                
                for image_path in sorted(glob.glob(os.path.join(seq_path_folder, "*.png"))):
                    frame = filename_to_frame_nr(os.path.basename(image_path))
                    if frame not in text: continue

                    boxes, labels = [], []
                    for obj in text[frame]:
                        if obj.class_id not in CLASS_MAPPING: continue
                        x, y, w, h = mask_utils.toBbox(obj.mask)
                        boxes.append([x, y, x + w, y + h]) # Pascal_VOC format
                        labels.append(CLASS_MAPPING[obj.class_id])

                    if len(boxes) > 0:
                        self.image_paths.append(image_path)
                        self.annotations.append({"boxes": boxes, "labels": labels})

    def __getitem__(self, idx):
        # Read image
        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        target = self.annotations[idx]
        
        if self.transform:
            transformed = self.transform(
                image=img,
                bboxes=target["boxes"],
                labels=target["labels"]
            )
            img = transformed["image"]
            boxes = torch.as_tensor(transformed["bboxes"], dtype=torch.float32)
            labels = torch.as_tensor(transformed["labels"], dtype=torch.int64)
        else:
            # Default conversion if no transform is provided
            img = torch.as_tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
            boxes = torch.as_tensor(target["boxes"], dtype=torch.float32)
            labels = torch.as_tensor(target["labels"], dtype=torch.int64)

        return img, {"boxes": boxes, "labels": labels}

    def __len__(self):
        return len(self.image_paths)