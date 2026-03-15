import os
import glob
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

class KittiMotsSamDataset(Dataset):
    def __init__(self, root_dir, split, transforms=None):
        self.root_dir = root_dir
        self.split = split
        self.transforms = transforms
        
        self.image_dir = os.path.join(root_dir, 'training', 'image_02')
        self.mask_dir = os.path.join(root_dir, 'instances')
        
        if split == 'train':
            self.seqs = ['0000', '0001', '0003', '0004', '0005', '0009', '0011', '0012', '0015', '0017', '0019', '0020']
        else:
            self.seqs = ['0002', '0006', '0007', '0008', '0010', '0013', '0014', '0016', '0018']
            
        self.samples = []
        for seq in self.seqs:
            seq_img_dir = os.path.join(self.image_dir, seq)
            if not os.path.exists(seq_img_dir): 
                continue
            
            for img_path in sorted(glob.glob(os.path.join(seq_img_dir, '*.png'))):
                filename = os.path.basename(img_path)
                mask_path = os.path.join(self.mask_dir, seq, filename)
                if os.path.exists(mask_path):
                    self.samples.append((img_path, mask_path, seq, filename))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, seq, filename = self.samples[idx]
        image = np.array(Image.open(img_path).convert("RGB"))
        mask_img = np.array(Image.open(mask_path), dtype=np.int32)
        
        frame_id = int(os.path.splitext(filename)[0])
        image_id = int(seq) * 1000000 + frame_id
        
        boxes, mask_list, class_list = [], [], []
        for inst_id in np.unique(mask_img):
            if inst_id == 0: continue  # Background
            class_id = int(inst_id // 1000)
            if class_id not in [1, 2]: continue  # Only Car and Pedestrian
            
            y_indices, x_indices = np.where(mask_img == inst_id)
            if len(x_indices) == 0: continue
            
            x_min, x_max = float(x_indices.min()), float(x_indices.max())
            y_min, y_max = float(y_indices.min()), float(y_indices.max())
            
            if x_max - x_min < 2 or y_max - y_min < 2: continue  # Avoid degenerate boxes
                
            boxes.append([x_min, y_min, x_max, y_max])
            class_list.append(class_id)
            mask_list.append((mask_img == inst_id).astype(np.float32))
        
        # Apply augmentations
        if self.transforms and len(boxes) > 0:
            aug = self.transforms(image=image, bboxes=boxes, masks=mask_list, labels=class_list)
            valid_boxes, valid_masks, valid_classes = [], [], []
            for bbox, m, cls in zip(aug['bboxes'], aug['masks'], aug['labels']):
                if bbox[2] - bbox[0] > 1 and bbox[3] - bbox[1] > 1:
                    valid_boxes.append(bbox)
                    valid_masks.append(m)
                    valid_classes.append(cls)
            
            if len(valid_boxes) > 0:
                image = aug['image']
                boxes, mask_list, class_list = valid_boxes, valid_masks, valid_classes
            else:
                boxes = []
        
        # Skip if no valid objects after augmentation
        if len(boxes) == 0:
            return self.__getitem__((idx + 1) % len(self))
            
        return {
            "image": image,
            "boxes": boxes,
            "masks": mask_list,
            "classes": class_list,
            "image_id": image_id
        }
