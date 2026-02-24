import torch
import os
from pycocotools import mask as mask_utils
import numpy as np
import glob
import cv2
import torch
from torch.utils.data.dataset import Dataset
import os
from pycocotools import mask as mask_utils
import numpy as np
import glob
import cv2
from Week1.src.utils.mots import load_seqmap, load_txt, filename_to_frame_nr


CLASS_MAPPING = {
    1: 3,  # car → COCO car
    2: 1,  # pedestrian → COCO person
}

class KittyDataset(Dataset):
    def __init__(self, root_dir,mode="train"):
        self.root_dir = root_dir
        if mode == "train":
            self.load_data("/home/msiau/workspace/jventosa/PostTFG/Master/C5_Team3/Week1/src/utils/train.seqmap")
        elif mode == "test":
            self.load_data("/home/msiau/workspace/jventosa/PostTFG/Master/C5_Team3/Week1/src/utils/val.seqmap")

    def load_data(self, seqmap):

        image_data = []
        label_data = []
        seqmaps, _ = load_seqmap(seqmap)

        for seq in seqmaps:
            seq_path_txt = os.path.join(self.root_dir, "instances_txt", seq + ".txt")
            seq_path_folder = os.path.join(self.root_dir, "training", "image_02", seq)

            text = load_txt(seq_path_txt)
            images = sorted(glob.glob(os.path.join(seq_path_folder, "*.png")))

            for image_path in images:
                frame = filename_to_frame_nr(os.path.basename(image_path))

                if frame not in text:
                    continue

                # ---- IMAGE ----
                img = cv2.imread(image_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = torch.as_tensor(img, dtype=torch.float32)
                img = img.permute(2, 0, 1)  # HWC → CHW
                img /= 255.0

                boxes = []
                labels = []

                for obj in text[frame]:
                    bbox = mask_utils.toBbox(obj.mask)  # [x,y,w,h]

                    x, y, w, h = bbox
                    if obj.class_id not in CLASS_MAPPING:
                        continue

                    boxes.append([x, y, x + w, y + h])  # convert to xyxy
                    
                    labels.append(CLASS_MAPPING[obj.class_id])
                if len(boxes) == 0:
                    continue
                boxes = torch.as_tensor(boxes, dtype=torch.float32)
                labels = torch.as_tensor(labels, dtype=torch.int64)

                target = {
                    "boxes": boxes,
                    "labels": labels,
                }

                image_data.append(img)
                label_data.append(target)

        self.images = image_data
        self.labels = label_data

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]