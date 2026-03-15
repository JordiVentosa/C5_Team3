import torch
from torch.utils.data import DataLoader
from transformers import SamModel, SamProcessor
from tqdm import tqdm

from dataset import KittiMotsSamDataset
from evaluators import SAMCOCOEvaluator
from collate import collate_fn_with_masks as collate_fn

# Configuration
dataset_path = "/home/mcv/datasets/C5/KITTI-MOTS"
BATCH_SIZE = 12
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CPU_WORKERS = 12


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
    evaluator.evaluate(print_header="\nCOCO Segmentation Metrics")

if __name__ == '__main__':
    evaluate_baseline()