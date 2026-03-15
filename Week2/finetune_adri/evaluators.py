import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mask_utils


class SAMCOCOEvaluator:
    def __init__(self):
        self.coco_gt = COCO()
        self.dataset = {
            "images": [],
            "annotations": [],
            "categories": [{"id": 1, "name": "Car"}, {"id": 2, "name": "Pedestrian"}]
        }
        self.ann_id = 1
        self.preds = []

    def add_gt_batch(self, image_id, image_size, valid_boxes, valid_masks, valid_classes):
        self.dataset["images"].append({
            "id": int(image_id),
            "width": image_size[1],
            "height": image_size[0],
            "file_name": str(image_id)
        })
        for box, mask, cls in zip(valid_boxes, valid_masks, valid_classes):
            mask = np.asfortranarray(mask.astype(np.uint8))
            rle = mask_utils.encode(mask)
            rle['counts'] = rle['counts'].decode('utf-8')
            area = float(mask_utils.area(rle))
            x1, y1, x2, y2 = map(float, box)
            w, h = x2 - x1, y2 - y1
            self.dataset["annotations"].append({
                "id": self.ann_id,
                "image_id": int(image_id),
                "category_id": int(cls),
                "bbox": [x1, y1, w, h],
                "segmentation": rle,
                "area": area,
                "iscrowd": 0
            })
            self.ann_id += 1

    def init_gt(self):
        self.coco_gt.dataset = self.dataset
        self.coco_gt.createIndex()

    def add_pred_batch(self, image_id, category_ids, pred_masks):
        for i, mask in enumerate(pred_masks):
            mask = np.asfortranarray((mask > 0.5).astype(np.uint8))
            rle = mask_utils.encode(mask)
            rle['counts'] = rle['counts'].decode('utf-8')
            self.preds.append({
                "image_id": int(image_id),
                "category_id": int(category_ids[i]),
                "segmentation": rle,
                "score": 1.0
            })

    def evaluate(self, print_header="\n===== COCO Segmentation Metrics ====="):
        if len(self.preds) == 0:
            print("No predictions to evaluate.")
            return None
        coco_dt = self.coco_gt.loadRes(self.preds)
        print(print_header)
        coco_eval = COCOeval(self.coco_gt, coco_dt, iouType="segm")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        return coco_eval.stats


class HF_COCOEvaluator:
    def __init__(self):
        self.coco_gt = COCO()
        self.dataset = {"images": [], "annotations": [], "categories": [{"id": 1, "name": "Car"}, {"id": 2, "name": "Pedestrian"}]}
        self.ann_id = 1
        self.preds = []

    def add_gt(self, image_id, height, width, boxes, masks, classes):
        self.dataset["images"].append({"id": image_id, "width": width, "height": height, "file_name": str(image_id)})
        for box, mask, cls in zip(boxes, masks, classes):
            mask_fortran = np.asfortranarray(mask.astype(np.uint8))
            rle = mask_utils.encode(mask_fortran)
            rle['counts'] = rle['counts'].decode('utf-8')
            x1, y1, x2, y2 = map(float, box)
            self.dataset["annotations"].append({
                "id": self.ann_id, "image_id": image_id, "category_id": int(cls),
                "bbox": [x1, y1, x2 - x1, y2 - y1], "segmentation": rle,
                "area": float(mask_utils.area(rle)), "iscrowd": 0
            })
            self.ann_id += 1

    def add_preds(self, image_id, masks, classes):
        for mask, cls in zip(masks, classes):
            mask_fortran = np.asfortranarray((mask > 0.5).astype(np.uint8))
            rle = mask_utils.encode(mask_fortran)
            rle['counts'] = rle['counts'].decode('utf-8')
            self.preds.append({"image_id": image_id, "category_id": int(cls), "segmentation": rle, "score": 1.0})

    def evaluate(self):
        if not self.preds: return 0.0
        self.coco_gt.dataset = self.dataset
        self.coco_gt.createIndex()
        coco_dt = self.coco_gt.loadRes(self.preds)
        coco_eval = COCOeval(self.coco_gt, coco_dt, iouType="segm")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        return float(coco_eval.stats[0])


class HF_COCOEvaluatorSingleClass:
    def __init__(self):
        self.coco_gt = COCO()
        self.dataset = {"images": [], "annotations": [], "categories": [{"id": 1, "name": "Object"}]}
        self.ann_id = 1
        self.preds = []

    def add_gt_batch(self, image_id, image_size, valid_boxes, valid_masks):
        self.dataset["images"].append({"id": int(image_id), "width": image_size[1], "height": image_size[0], "file_name": str(image_id)})
        for box, mask in zip(valid_boxes, valid_masks):
            mask = np.asfortranarray(mask.astype(np.uint8))
            rle = mask_utils.encode(mask)
            rle['counts'] = rle['counts'].decode('utf-8')
            x1, y1, x2, y2 = map(float, box)
            self.dataset["annotations"].append({
                "id": self.ann_id, "image_id": int(image_id), "category_id": 1,
                "bbox": [x1, y1, x2 - x1, y2 - y1], "segmentation": rle,
                "area": float(mask_utils.area(rle)), "iscrowd": 0
            })
            self.ann_id += 1

    def init_gt(self):
        self.coco_gt.dataset = self.dataset
        self.coco_gt.createIndex()

    def add_pred_batch(self, image_id, pred_masks):
        for mask in pred_masks:
            mask = np.asfortranarray((mask > 0.5).astype(np.uint8))
            rle = mask_utils.encode(mask)
            rle['counts'] = rle['counts'].decode('utf-8')
            self.preds.append({"image_id": int(image_id), "category_id": 1, "segmentation": rle, "score": 1.0})

    def evaluate(self):
        if len(self.preds) == 0:
            return None
        coco_dt = self.coco_gt.loadRes(self.preds)
        coco_eval = COCOeval(self.coco_gt, coco_dt, iouType="segm")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        return coco_eval.stats
