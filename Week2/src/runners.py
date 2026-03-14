import numpy as np
from utils import *
from PIL import Image
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Evaluation runners — register new ones here
# ---------------------------------------------------------------------------
 
 
def centroid_point(mask: np.ndarray):
    
    ys, xs = np.where(mask)
    cy, cx = int(np.round(ys.mean())), int(np.round(xs.mean()))
    
    coords = np.stack([ys, xs], axis=1)
    
    best   = coords[np.argmin(np.sum((coords - [cy, cx]) ** 2, axis=1))]
    
    return np.array([[best[1], best[0]]], dtype=np.float32), np.array([1], dtype=np.int64)
 
 
def run_point_prompt(coco_gt, coco_dict, model, output_dir, **kwargs):
    
    predictions = []
    
    for img_info in coco_dict["images"]:
        
        image_id = img_info["id"]
        image    = np.array(Image.open(img_info["file_name"]).convert("RGB"))
        
        model.set_image(image)
        anns = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=image_id))
        
        for ann in anns:
            
            gt_mask      = decode_ann_mask(ann)
            coords, lbls = centroid_point(gt_mask)
            
            pred, score  = model.predict_point(
                coords,
                lbls,
            )
            
            predictions.append({"image_id": image_id, "category_id": ann["category_id"],
                                 "segmentation": encode_mask(pred), "score": score})
            
        print(f"  {image_id}/{len(coco_dict['images'])-1}", end="\r")
            
    run_coco_eval(coco_gt, predictions, output_dir)
 
 
def run_text_prompt(coco_gt, coco_dict, model, output_dir,
                    prompts: dict,
                    box_threshold:  float = 0.3,
                    text_threshold: float = 0.25,
                    **kwargs):
    """
    Uses GroundedSAMPredictor. prompts = {category_id: text_label}.
    Grounding DINO detects instances from text; SAM segments each box.
    Category ID is assigned by matching the detected label back to prompts.
    """
    # ensure keys are ints (YAML may parse them as strings)
    prompts = {int(k): v for k, v in prompts.items()}
 
    predictions = []
    for img_info in coco_dict["images"]:
        image_id = img_info["id"]
        image    = np.array(Image.open(img_info["file_name"]).convert("RGB"))
        model.set_image(image)
 
        # predict
        detections = model.predict_text(prompts, box_threshold, text_threshold)
 
        for mask, score, cat_id in detections:
            predictions.append({
                "image_id":     image_id,
                "category_id":  cat_id,
                "segmentation": encode_mask(mask),
                "score":        score,
            })
 

        print(f"  {image_id}/{len(coco_dict['images'])-1}", end="\r")
        
    run_coco_eval(coco_gt, predictions, output_dir)


def run_bbox_best(coco_gt, coco_dict, model, output_dir,
                       yolo_weights: str = None,
                       confidence: float = 0.5,
                       **kwargs):
 
    # YOLO class index -> COCO category_id (adjust if your YOLO classes differ)
    yolo_to_cat = {
        0: 2,   # person      -> pedestrian
        2: 1,   # car         -> car
        3: 1,   # motorcycle  -> car
        5: 1,   # bus         -> car
        7: 1,   # truck       -> car
    }   
 
    yolo = YOLO(yolo_weights) if yolo_weights else YOLO("yolo26x.pt")
    predictions = []
 
    for img_info in coco_dict["images"]:
        image_id = img_info["id"]
        image    = np.array(Image.open(img_info["file_name"]).convert("RGB"))
 
        # YOLO inference
        yolo_out    = yolo(image, conf=confidence, verbose=False)[0]
        boxes_xyxy  = yolo_out.boxes.xyxy.cpu().numpy()
        class_ids   = yolo_out.boxes.cls.cpu().numpy().astype(int)
        yolo_scores = yolo_out.boxes.conf.cpu().numpy()
 
        if len(boxes_xyxy) == 0:
            print(f"  {image_id}/{len(coco_dict['images'])-1}", end="\r")
            continue
 
        # encode image once, run SAM per box
        model.set_image(image)
 
        for box, cls_id, yolo_score in zip(boxes_xyxy, class_ids, yolo_scores):
            cat_id = yolo_to_cat.get(int(cls_id))
            if cat_id is None:
                continue
            pred, sam_score = model.predict_box([[box.tolist()]])
            
            predictions.append({
                "image_id":     image_id,
                "category_id":  cat_id,
                "segmentation": encode_mask(pred),
                "score":        sam_score,
            })
            
        print(f"  {image_id}/{len(coco_dict['images'])-1}", end="\r")
 
    run_coco_eval(coco_gt, predictions, output_dir)
    
# ---------------------------------------------------------------------------
# Inference runners — register new ones here (exact equal to evaluation but outputing saved image over given images)
# ---------------------------------------------------------------------------    
    
def run_point_prompt_inf(coco_gt, coco_dict, model, output_dir, **kwargs):
    
    predictions = []
    
    index_list = kwargs.get("index_list", [])
    
    for img_info in coco_dict["images"]:
        
        image_id = img_info["id"]
        
        if image_id not in index_list:
            continue
        
        image    = np.array(Image.open(img_info["file_name"]).convert("RGB"))
        
        anns     = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=image_id))
        gt_masks = [decode_ann_mask(a) for a in anns]
        gt_cats  = [a["category_id"] for a in anns]
        
        model.set_image(image)
        pred_masks, pred_cats, pred_boxes = [], [], []
        
        for ann in anns:
            
            gt_mask      = decode_ann_mask(ann)
            coords, lbls = centroid_point(gt_mask)
            
            pred, score  = model.predict_point(
                coords,
                lbls,
            )
            
            predictions.append({"image_id": image_id, "category_id": ann["category_id"],
                                 "segmentation": encode_mask(pred), "score": score})
            
            pred_masks.append(pred)
            pred_cats.append(ann["category_id"])
            
        save_overlays(image, gt_masks, gt_cats, pred_masks, pred_cats, output_dir, image_id)
        print(f"  {image_id}/{len(coco_dict['images'])-1}", end="\r")
 
 
def run_text_prompt_inf(coco_gt, coco_dict, model, output_dir,
                    prompts: dict,
                    box_threshold:  float = 0.3,
                    text_threshold: float = 0.25,
                    **kwargs):
    """
    Uses GroundedSAMPredictor. prompts = {category_id: text_label}.
    Grounding DINO detects instances from text; SAM segments each box.
    Category ID is assigned by matching the detected label back to prompts.
    """
    # ensure keys are ints (YAML may parse them as strings)
    prompts = {int(k): v for k, v in prompts.items()}
 
    predictions = []
    
    index_list = kwargs.get("index_list", [])
    
    
    for img_info in coco_dict["images"]:
        image_id = img_info["id"]
        
        if image_id not in index_list:
            continue
        
        image    = np.array(Image.open(img_info["file_name"]).convert("RGB"))
        model.set_image(image)
        
        # GT for visualisation
        anns     = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=image_id))
        gt_masks = [decode_ann_mask(a) for a in anns]
        gt_cats  = [a["category_id"] for a in anns]
 
        # predict
        detections = model.predict_text(prompts, box_threshold, text_threshold)
        pred_masks = [d[0] for d in detections]
        pred_cats  = [d[2] for d in detections]
 
        for mask, score, cat_id in detections:
            predictions.append({
                "image_id":     image_id,
                "category_id":  cat_id,
                "segmentation": encode_mask(mask),
                "score":        score,
            })
 
        save_overlays(image, gt_masks, gt_cats, pred_masks, pred_cats, output_dir, image_id)
        print(f"  {image_id}/{len(coco_dict['images'])-1}", end="\r")


def run_bbox_best_inf(coco_gt, coco_dict, model, output_dir,
                       yolo_weights: str = None,
                       confidence: float = 0.5,
                       **kwargs):
 
    # YOLO class index -> COCO category_id (adjust if your YOLO classes differ)
    yolo_to_cat = {
        0: 2,   # person      -> pedestrian
        2: 1,   # car         -> car
        3: 1,   # motorcycle  -> car
        5: 1,   # bus         -> car
        7: 1,   # truck       -> car
    }   
 
    yolo = YOLO(yolo_weights) if yolo_weights else YOLO("yolo26x.pt")
    predictions = []
    
    index_list = kwargs.get("index_list", [])
 
    for img_info in coco_dict["images"]:
        image_id = img_info["id"]
        
        if image_id not in index_list:
            continue
        
        image    = np.array(Image.open(img_info["file_name"]).convert("RGB"))
        
        # GT masks for visualisation only
        anns     = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=image_id))
        gt_masks = [decode_ann_mask(a) for a in anns]
        gt_cats  = [a["category_id"] for a in anns]
 
        # YOLO inference
        yolo_out    = yolo(image, conf=confidence, verbose=False)[0]
        boxes_xyxy  = yolo_out.boxes.xyxy.cpu().numpy()
        class_ids   = yolo_out.boxes.cls.cpu().numpy().astype(int)
        yolo_scores = yolo_out.boxes.conf.cpu().numpy()
 
        if len(boxes_xyxy) == 0:
            save_overlays(image, gt_masks, gt_cats, [], [], output_dir, image_id)
            print(f"  {image_id}/{len(coco_dict['images'])-1}", end="\r")
            continue
 
        # encode image once, run SAM per box
        model.set_image(image)
        pred_masks, pred_cats, pred_boxes = [], [], []
 
        for box, cls_id, yolo_score in zip(boxes_xyxy, class_ids, yolo_scores):
            cat_id = yolo_to_cat.get(int(cls_id))
            if cat_id is None:
                continue
            pred, sam_score = model.predict_box([[box.tolist()]])
            
            pred_masks.append(pred)
            pred_cats.append(cat_id)
            pred_boxes.append(box)
            
            predictions.append({
                "image_id":     image_id,
                "category_id":  cat_id,
                "segmentation": encode_mask(pred),
                "score":        sam_score,
            })
            
        save_overlays_with_boxes(image, gt_masks, gt_cats,
                                  pred_masks, pred_cats, pred_boxes,
                                  output_dir, image_id)
        
        print(f"  {image_id}/{len(coco_dict['images'])-1}", end="\r")