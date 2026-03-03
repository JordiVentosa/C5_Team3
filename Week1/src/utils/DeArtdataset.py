import torch
import numpy as np
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
import json


CLASS_MAPPING = {
    "person": 1,
    "nude": 1,
    "angel": 1,
    "knight": 1,
    "monk": 1,
    "crucifixion":1,
    "god the father":1,
    "shepherd":1,
    "saturno":1,
    "judith" :1
}




class EuropeanArtDataset(Dataset):
    def __init__(self,  transform=None, dataset = None):
        """
        Args:
            mode (str): "train" or "validation" (HuggingFace split names)
            transform: albumentations transform pipeline (optional)
        """
        self.transform = transform
        self.image_paths = []
        self.annotations = []



        
        print("done")
        self.load_metadata(dataset)

    def load_metadata(self, dataset):
        
        for sample in tqdm(dataset):
            image = sample["image"]  # PIL Image
            
            # Skip samples without annotations
            if "annotations" not in sample or sample["annotations"] is None:
                continue

            annotations_dict = json.loads(sample["annotations"])
            anottations = annotations_dict["annotations"]
            boxes, labels = [], []

            for  i,annotation in enumerate(anottations):

                categ_id = annotation["category_id"]
                
                category = annotations_dict["categories"][categ_id-1]["name"]
                
                label = CLASS_MAPPING.get(category, -1)
                
                if label == -1:
                    continue
                bbox = annotation["bbox"]
                
                bbox[2] = bbox[2] + bbox[0]
                bbox[3] = bbox[3] + bbox[1]
                boxes.append(bbox)  
                labels.append(label)

            if len(boxes) > 0:
                self.image_paths.append(image)
                self.annotations.append({"boxes": boxes, "labels": labels})

    def __getitem__(self, idx):

        img = np.array(self.image_paths[idx].convert("RGB"))

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
            img = torch.as_tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
            boxes = torch.as_tensor(target["boxes"], dtype=torch.float32)
            labels = torch.as_tensor(target["labels"], dtype=torch.int64)
        if len(boxes) > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        return img, {"boxes": boxes, "labels": labels,"image_id": torch.tensor(idx)}

    def __len__(self):
        return len(self.image_paths)