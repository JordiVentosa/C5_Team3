import torch
from torch.utils.data import DataLoader
from transformers import SamModel, SamProcessor
import numpy as np
from tqdm import tqdm

from dataset import KittiMotsSamDataset
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mask_utils

# Configuration
dataset_path = "/home/mcv/datasets/C5/KITTI-MOTS"
BATCH_SIZE = 12
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CPU_WORKERS = 12

def collate_fn(batch, processor):
    batch = [b for b in batch if b is not None and len(b['boxes']) > 0]
    if len(batch) == 0: return None
    
    images = [b["image"] for b in batch]
    orig_boxes = [b["boxes"] for b in batch]
    orig_true_masks = [b["masks"] for b in batch]
    orig_classes = [b["classes"] for b in batch]
    image_ids = [b["image_id"] for b in batch]
    
    valid_lengths = [len(boxes) for boxes in orig_boxes]
    max_objects = max(valid_lengths)
    
    padded_boxes, padded_true_masks = [], []
    
    for i in range(len(batch)):
        pad_len = max_objects - valid_lengths[i]
        b_boxes = orig_boxes[i] + [[0.0, 0.0, 0.0, 0.0]] * pad_len
        padded_boxes.append(b_boxes)
        
        if pad_len > 0:
            mask_shape = orig_true_masks[i][0].shape
            dummy_masks = [np.zeros(mask_shape, dtype=np.float32) for _ in range(pad_len)]
            b_masks = orig_true_masks[i] + dummy_masks
        else:
            b_masks = orig_true_masks[i]
        padded_true_masks.append(b_masks)
    
    inputs = processor(images=images, input_boxes=padded_boxes, return_tensors="pt")
    return inputs, padded_true_masks, orig_boxes, orig_classes, image_ids, images, valid_lengths

class SAMCOCOEvaluator:
    def __init__(self):
        self.coco_gt = COCO()
        self.dataset = {"images": [], "annotations": [], "categories": [{"id": 1, "name": "Car"}, {"id": 2, "name": "Pedestrian"}]}
        self.ann_id = 1
        self.preds = []

    def add_gt_batch(self, image_id, image_size, valid_boxes, valid_masks, valid_classes):
        self.dataset["images"].append({"id": int(image_id), "width": image_size[1], "height": image_size[0], "file_name": str(image_id)})
        for box, mask, cls in zip(valid_boxes, valid_masks, valid_classes):
            mask = np.asfortranarray(mask.astype(np.uint8))
            rle = mask_utils.encode(mask)
            rle['counts'] = rle['counts'].decode('utf-8')
            area = float(mask_utils.area(rle))
            x1, y1, x2, y2 = map(float, box)
            w, h = x2 - x1, y2 - y1
            self.dataset["annotations"].append({"id": self.ann_id, "image_id": int(image_id), "category_id": int(cls),
                                                 "bbox": [x1, y1, w, h], "segmentation": rle, "area": area, "iscrowd": 0})
            self.ann_id += 1

    def init_gt(self):
        self.coco_gt.dataset = self.dataset
        self.coco_gt.createIndex()

    def add_pred_batch(self, image_id, category_ids, pred_masks):
        for i, mask in enumerate(pred_masks):
            mask = np.asfortranarray((mask > 0.5).astype(np.uint8))
            rle = mask_utils.encode(mask)
            rle['counts'] = rle['counts'].decode('utf-8')
            self.preds.append({"image_id": int(image_id), "category_id": int(category_ids[i]), "segmentation": rle, "score": 1.0})

    def evaluate(self):
        if len(self.preds) == 0:
            return None
        coco_dt = self.coco_gt.loadRes(self.preds)
        print("\nCOCO Segmentation Metrics")
        coco_eval = COCOeval(self.coco_gt, coco_dt, iouType="segm")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        return coco_eval.stats

def evaluate_baseline():
    print("Loading processor and baseline model...")
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    model = SamModel.from_pretrained("facebook/sam-vit-base").to(DEVICE)
    model.eval()
    
    print("Loading validation dataset...")
    val_dataset = KittiMotsSamDataset(root_dir=dataset_path, split='val', transforms=None)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False,
                            collate_fn=lambda x: collate_fn(x, processor), num_workers=CPU_WORKERS, pin_memory=True)

    evaluator = SAMCOCOEvaluator()
    print("Starting inference...")
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            if batch is None: continue
            
            inputs, true_masks, orig_boxes, orig_classes, image_ids, orig_images, valid_lengths = batch
            inputs_gpu = {k: v.to(DEVICE) for k, v in inputs.items()}
            outputs = model(**inputs_gpu, multimask_output=False)
            
            post_processed_masks = processor.post_process_masks(
                outputs.pred_masks,
                original_sizes=inputs["original_sizes"].tolist(),
                reshaped_input_sizes=inputs["reshaped_input_sizes"].tolist()
            )
            
            for i in range(len(image_ids)):
                valid_len = valid_lengths[i]
                b_img_id = image_ids[i]
                img_shape = orig_images[i].shape[:2]
                b_classes = orig_classes[i]
                b_boxes = orig_boxes[i]
                b_true_masks = true_masks[i][:valid_len]
                
                evaluator.add_gt_batch(image_id=b_img_id, image_size=img_shape,
                                      valid_boxes=b_boxes, valid_masks=b_true_masks, valid_classes=b_classes)
                
                pred_m = torch.sigmoid(post_processed_masks[i].squeeze(1)).cpu().numpy()[:valid_len]
                evaluator.add_pred_batch(image_id=b_img_id, category_ids=b_classes, pred_masks=pred_m)

    print("\nCalculating metrics...")
    evaluator.init_gt()
    evaluator.evaluate()

if __name__ == '__main__':
    evaluate_baseline()