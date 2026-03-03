import torch
import os
from pycocotools import mask as mask_utils
import glob
import cv2
from torch.utils.data.dataset import Dataset
from Week1.src.utils.mots import load_seqmap, load_txt, filename_to_frame_nr


CLASS_MAPPING = {
    1: 2,  # car → COCO car
    2: 0,  # pedestrian → COCO person
}


class KittyDataset(Dataset):
    def __init__(self, root_dir,image_processor, mode="train", transform=None,):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.annotations = []
        self.image_processor = image_processor
        if mode == "train" :
            seq_files = ["/home/msiau/workspace/Master/C5_Team3/Week1/src/utils/train.seqmap"]
        elif mode == "val":
            seq_files =  ["/home/msiau/workspace/Master/C5_Team3/Week1/src/utils/val.seqmap"]
        else:
            seq_files = ["/home/msiau/workspace/Master/C5_Team3/Week1/src/utils/train.seqmap","/home/msiau/workspace/Master/C5_Team3/Week1/src/utils/val.seqmap"]
        self.load_metadata(seq_files)

    @staticmethod
    def format_image_annotations_as_coco(image_id, old_annotations):

        annotations = []
        for box,label in zip(old_annotations[0],old_annotations[1]):
            
            formatted_annotation = {
                "image_id": image_id,
                "category_id": label,
                "bbox": list(box),
                "iscrowd": 0,
                "area": box[2] * box[3],
            }
            annotations.append(formatted_annotation)

        return {
            "image_id": image_id,
            "annotations": annotations,
        }

    def load_metadata(self, sequence_maps):
        for seqmap in sequence_maps:
            seqmaps, _ = load_seqmap(seqmap)
            for seq in seqmaps:
                seq_path_txt = os.path.join(self.root_dir, "instances_txt", f"{seq}.txt")
                seq_path_folder = os.path.join(self.root_dir, "training", "image_02", seq)
                text = load_txt(seq_path_txt)
                
                for image_path in sorted(glob.glob(os.path.join(seq_path_folder, "*.png"))):
                    frame = filename_to_frame_nr(os.path.basename(image_path))
                    if frame not in text: 
                        continue

                    boxes, labels = [], []
                    for obj in text[frame]:
                        if obj.class_id not in CLASS_MAPPING: 
                            continue
                        x, y, w, h = mask_utils.toBbox(obj.mask)
                        boxes.append([x, y, w, h]) # Coco format
                        labels.append(CLASS_MAPPING[obj.class_id])

                    if len(boxes) > 0:
                        self.image_paths.append(image_path)
                        self.annotations.append(( boxes,  labels))

    def __getitem__(self, idx):
        # Read image
        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        target = self.annotations[idx]
        
        transformed = self.transform(
            image=img,
            bboxes=target[0],
            class_labels=target[1]
        )
        img = transformed["image"]
        boxes = torch.as_tensor(transformed["bboxes"], dtype=torch.float32)
        categories = torch.as_tensor(transformed["class_labels"], dtype=torch.int64)
        
        formatted_annotations = self.format_image_annotations_as_coco(idx, (boxes,categories))
        
        result = self.image_processor(
            images=img, annotations=formatted_annotations, return_tensors="pt"
        )
        
        result = {k: v[0] for k, v in result.items()}

        return result

    def __len__(self):
        return len(self.image_paths)