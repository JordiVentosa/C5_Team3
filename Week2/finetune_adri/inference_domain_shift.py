import os
import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset, load_from_disk
from torch.utils.data import Dataset, DataLoader
from transformers import SamModel, SamProcessor
from peft import PeftModel
import scipy.ndimage as ndimage
import gc

from evaluators import HF_COCOEvaluatorSingleClass as HF_COCOEvaluator
from collate import collate_fn_boxes_only as collate_fn
from visualization import save_visualization_basic as save_visualization

FT_MODEL_PATH = "models/da_config_1/best_model"
LORA_MODEL_PATH = "models/lora_da_config_1/best_model"
LOCAL_DATASET_PATH = "dataset/iSAID_local"
QUALITATIVE_DIR = "domain_shift_qualitative_100"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_EVAL_IMAGES = 90
VISUALIZE_INDICES = [5,10,20,25,30,35,40,45,50,55,60,65,70,75,80,85]
BATCH_SIZE = 1
CPU_WORKERS = 4
MAX_OBJECTS_PER_IMAGE = 100


def extract_boxes_and_masks(ins_image, max_objects=None):
    """Extract bounding boxes and masks using scipy's C-optimized functions."""
    ins_arr = np.array(ins_image)
    
    if len(ins_arr.shape) == 3:
        ins_arr = np.dot(ins_arr.astype(np.uint32), [65536, 256, 1])
    
    unique_ids, contiguous_labels, counts = np.unique(ins_arr, return_inverse=True, return_counts=True)
    contiguous_labels = contiguous_labels.reshape(ins_arr.shape)
    
    if unique_ids[0] != 0:
        contiguous_labels += 1
        counts = np.insert(counts, 0, 0)
    
    slices = ndimage.find_objects(contiguous_labels)
    boxes, masks = [], []
    
    for i, slc in enumerate(slices):
        if slc is None: continue
        
        label_id = i + 1
        if counts[label_id] < 10: continue
            
        y_slice, x_slice = slc
        y_min, y_max, x_min, x_max = y_slice.start, y_slice.stop, x_slice.start, x_slice.stop
        
        if x_max - x_min > 2 and y_max - y_min > 2:
            mask = (contiguous_labels == label_id).astype(np.uint8)
            boxes.append([x_min, y_min, x_max, y_max])
            masks.append(mask)
            
            if max_objects is not None and len(boxes) >= max_objects:
                break
                
    return boxes, masks


class ISAIDDatasetWrapper(Dataset):
    """PyTorch wrapper for Hugging Face iSAID dataset."""
    def __init__(self, hf_dataset, num_images):
        self.hf_dataset = hf_dataset
        self.num_images = min(num_images, len(hf_dataset))

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        sample = self.hf_dataset[idx]
        image = sample['image'].convert("RGB")
        boxes, true_masks = extract_boxes_and_masks(sample['ins'], max_objects=MAX_OBJECTS_PER_IMAGE)
        return {"image": image, "boxes": boxes, "masks": true_masks, "image_id": idx + 1}


def evaluate_model(model, processor, dataloader, name="Model"):
    print(f"Evaluating {name}")
    evaluator = HF_COCOEvaluator()
    model.eval()
    PROMPT_CHUNK_SIZE = 25
    processed_images = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=name):
            if batch is None: continue
            
            inputs, orig_boxes, orig_masks, image_ids, valid_lengths, orig_images = batch
            inputs_gpu = {k: v.to(DEVICE) for k, v in inputs.items()}
            pixel_values = inputs_gpu.pop("pixel_values")
            image_embeddings = model.get_image_embeddings(pixel_values)
            
            B, N, _ = inputs_gpu["input_boxes"].shape
            batch_pred_masks = [[] for _ in range(B)]
            
            for start_idx in range(0, N, PROMPT_CHUNK_SIZE):
                end_idx = min(start_idx + PROMPT_CHUNK_SIZE, N)
                chunk_boxes = inputs_gpu["input_boxes"][:, start_idx:end_idx, :]
                
                outputs = model(image_embeddings=image_embeddings, input_boxes=chunk_boxes, multimask_output=False)
                chunk_post_masks = processor.post_process_masks(outputs.pred_masks,
                                                                original_sizes=inputs["original_sizes"].tolist(),
                                                                reshaped_input_sizes=inputs["reshaped_input_sizes"].tolist(),
                                                                binarize=False)
                
                for b_idx in range(B):
                    chunk_m = torch.sigmoid(chunk_post_masks[b_idx].squeeze(1)).cpu().numpy()
                    batch_pred_masks[b_idx].append(chunk_m)
                
                del outputs, chunk_post_masks, chunk_boxes
            
            for i in range(len(image_ids)):
                valid_len = valid_lengths[i]
                img_id = image_ids[i]
                b_boxes = orig_boxes[i][:valid_len]
                b_true_masks = orig_masks[i][:valid_len]
                img_shape = np.array(orig_images[i]).shape[:2]
                
                evaluator.add_gt_batch(img_id, img_shape, b_boxes, b_true_masks)
                pred_m = np.concatenate(batch_pred_masks[i], axis=0)[:valid_len]
                evaluator.add_pred_batch(img_id, pred_m)
                processed_images += 1
                
                if img_id in VISUALIZE_INDICES:
                    save_visualization(orig_images[i], b_boxes, pred_m, b_true_masks, name, img_id, QUALITATIVE_DIR)
            
            del image_embeddings, pixel_values, inputs_gpu, batch_pred_masks
            torch.cuda.empty_cache()
            gc.collect()
    
    print(f"Processed {processed_images} images for {name}.")
    evaluator.init_gt()
    stats = evaluator.evaluate()
    if stats is not None:
        print(f"  mAP (0.50:0.95): {stats[0]:.4f}")

def main():
    os.makedirs(QUALITATIVE_DIR, exist_ok=True)
    print("Inference on iSAID with configurations:")
    print(f"  FT Model: {FT_MODEL_PATH}")
    print(f"  LoRA Model: {LORA_MODEL_PATH}")
    print(f"  Local Dataset: {LOCAL_DATASET_PATH}")
    print(f"  Num Images: {NUM_EVAL_IMAGES}, Batch: {BATCH_SIZE}, Workers: {CPU_WORKERS}\n")
    
    if os.path.exists(LOCAL_DATASET_PATH):
        hf_dataset = load_from_disk(LOCAL_DATASET_PATH)
    else:
        hf_dataset = load_dataset("ariG23498/iSAID", split="train")
        hf_dataset.save_to_disk(LOCAL_DATASET_PATH)
    
    pytorch_dataset = ISAIDDatasetWrapper(hf_dataset, num_images=NUM_EVAL_IMAGES)
    base_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    dataloader = DataLoader(pytorch_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False,
                            collate_fn=lambda x: collate_fn(x, base_processor),
                            num_workers=CPU_WORKERS, pin_memory=True)
    
    print("Evaluating SAM Base...")
    base_model = SamModel.from_pretrained("facebook/sam-vit-base").to(DEVICE)
    evaluate_model(base_model, base_processor, dataloader, "SAM Base")
    del base_model
    torch.cuda.empty_cache()
    
    print(f"\nEvaluating SAM LoRA from {LORA_MODEL_PATH}...")
    try:
        ft_lora_processor = SamProcessor.from_pretrained(LORA_MODEL_PATH)
        base_model_for_lora = SamModel.from_pretrained("facebook/sam-vit-base")
        ft_lora_model = PeftModel.from_pretrained(base_model_for_lora, LORA_MODEL_PATH).to(DEVICE)
        evaluate_model(ft_lora_model, ft_lora_processor, dataloader, "SAM LoRA")
        del ft_lora_model
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"Error loading LoRA model: {e}")

    print(f"\nEvaluating SAM FT Normal from {FT_MODEL_PATH}...")
    try:
        ft_norm_processor = SamProcessor.from_pretrained(FT_MODEL_PATH)
        ft_norm_model = SamModel.from_pretrained(FT_MODEL_PATH).to(DEVICE)
        evaluate_model(ft_norm_model, ft_norm_processor, dataloader, "SAM FT Normal")
        del ft_norm_model
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"Error loading FT Normal model: {e}")

if __name__ == '__main__':
    main()