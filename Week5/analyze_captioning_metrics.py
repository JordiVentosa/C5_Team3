import json
import os
from src.metrics import compute_metrics

# Standard phrase for unrecognizable images
UNANSWERABLE_PHRASE = "quality issues are too severe to recognize visual content"

def is_unanswerable(text: str) -> bool:
    """Checks if a string is exactly the quality issues phrase."""
    # Remove trailing periods and extra spaces for safe comparison
    return text.strip().lower().replace('.', '') == UNANSWERABLE_PHRASE.replace('.', '')

def safe_compute(preds, refs):
    """Safely computes metrics avoiding division by zero."""
    if not preds:
        return {'bleu-1': 0.0, 'bleu-2': 0.0, 'rouge-l': 0.0, 'meteor': 0.0}
    return compute_metrics(preds, refs)

def main(json_path: str, output_txt: str):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    predictions = data.get('predictions', [])
    total_images = len(predictions)
    
    # Dictionaries to store preds and refs for each subgroup
    groups = {
        'all': {'p': [], 'r': []},
        'pred_answerable': {'p': [], 'r': []},
        'pred_unanswerable': {'p': [], 'r': []},
        'tp': {'p': [], 'r': []}, # Pred=U, Refs>=3 U
        'fp': {'p': [], 'r': []}, # Pred=U, Refs<3 U
        'fn': {'p': [], 'r': []}, # Pred=A, Refs>=3 U
        'tn': {'p': [], 'r': []}, # Pred=A, Refs<3 U
        
        # Special subgroup: Annotator disagreement (1 to 4 references with quality issues)
        'mixed_quality_all': {'p': [], 'r': []},
        'mixed_quality_pred_u': {'p': [], 'r': []},
        'mixed_quality_pred_a': {'p': [], 'r': []}
    }
    
    for item in predictions:
        p_text = item['predicted_caption']
        r_texts = item['reference_captions']
        
        p_is_u = is_unanswerable(p_text)
        u_refs_count = sum(1 for r in r_texts if is_unanswerable(r))
        r_is_u = u_refs_count >= 3 # Original report threshold
        
        # Add to global group
        groups['all']['p'].append(p_text)
        groups['all']['r'].append(r_texts)
        
        # Main classification
        if p_is_u:
            groups['pred_unanswerable']['p'].append(p_text)
            groups['pred_unanswerable']['r'].append(r_texts)
            if r_is_u:
                groups['tp']['p'].append(p_text)
                groups['tp']['r'].append(r_texts)
            else:
                groups['fp']['p'].append(p_text)
                groups['fp']['r'].append(r_texts)
        else:
            groups['pred_answerable']['p'].append(p_text)
            groups['pred_answerable']['r'].append(r_texts)
            if r_is_u:
                groups['fn']['p'].append(p_text)
                groups['fn']['r'].append(r_texts)
            else:
                groups['tn']['p'].append(p_text)
                groups['tn']['r'].append(r_texts)
                
        # Classification for "Mixed Quality" (1 to 4 unanswerable refs)
        if 1 <= u_refs_count <= 4:
            groups['mixed_quality_all']['p'].append(p_text)
            groups['mixed_quality_all']['r'].append(r_texts)
            
            if p_is_u:
                groups['mixed_quality_pred_u']['p'].append(p_text)
                groups['mixed_quality_pred_u']['r'].append(r_texts)
            else:
                groups['mixed_quality_pred_a']['p'].append(p_text)
                groups['mixed_quality_pred_a']['r'].append(r_texts)

    # Metric calculation
    results = {}
    for group_name, lists in groups.items():
        results[group_name] = safe_compute(lists['p'], lists['r'])
        results[group_name]['count'] = len(lists['p'])

    # Confusion matrix metrics
    tp_cnt = results['tp']['count']
    fp_cnt = results['fp']['count']
    fn_cnt = results['fn']['count']
    tn_cnt = results['tn']['count']
    
    precision = tp_cnt / (tp_cnt + fp_cnt) if (tp_cnt + fp_cnt) > 0 else 0
    recall = tp_cnt / (tp_cnt + fn_cnt) if (tp_cnt + fn_cnt) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Generation of the TXT Report
    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write("=================================================================\n")
        f.write("EXTENSIVE MULTI-METRIC EVALUATION REPORT (IMAGE CAPTIONING)\n")
        f.write("=================================================================\n\n")
        
        f.write(f"Total analyzed images: {total_images}\n\n")
        
        f.write("1. 'QUALITY ISSUES' CONFUSION MATRIX AND CALIBRATION\n")
        f.write("-" * 65 + "\n")
        f.write(f"True Positives (TP): {tp_cnt}\n")
        f.write(f"False Positives (FP):     {fp_cnt}\n")
        f.write(f"False Negatives (FN):     {fn_cnt}\n")
        f.write(f"True Negatives (TN): {tn_cnt}\n\n")
        
        f.write(f"Precision (Unanswerable): {precision:.2%} (The model is correct {precision:.2%} of the times it outputs 'Quality issues')\n")
        f.write(f"Recall (Unanswerable):    {recall:.2%} (The model detects {recall:.2%} of truly unacceptable images)\n")
        f.write(f"F1-Score:                 {f1:.4f}\n\n")
        
        f.write("2. STRATIFIED METRICS\n")
        f.write("-" * 65 + "\n")
        f.write(f"{'Stratum':<25} | {'N':<5} | {'BLEU-1':<7} | {'BLEU-2':<7} | {'ROUGE-L':<7} | {'METEOR':<7}\n")
        f.write("-" * 65 + "\n")
        
        def write_row(name, data):
            f.write(f"{name:<25} | {data['count']:<5} | {data['bleu-1']:.4f}  | {data['bleu-2']:.4f}  | {data['rouge-l']:.4f}   | {data['meteor']:.4f}\n")
            
        write_row("All", results['all'])
        write_row("Pred=ANSWERABLE", results['pred_answerable'])
        write_row("Pred=UNANSWERABLE", results['pred_unanswerable'])
        write_row("TP (Unanswerable OK)", results['tp'])
        write_row("FP (False Positive)", results['fp'])
        write_row("FN (False Negative)", results['fn'])
        write_row("TN (Answerable OK)", results['tn'])
        f.write("\n")
        
        f.write("3. DISAGREEMENT IMAGE ANALYSIS (MIXED QUALITY)\n")
        f.write("Images where at least 1 but less than 5 annotators reported 'Quality Issues'.\n")
        f.write("-" * 65 + "\n")
        mq_cnt = results['mixed_quality_all']['count']
        if mq_cnt > 0:
            mq_pred_u_pct = results['mixed_quality_pred_u']['count'] / mq_cnt
            mq_pred_a_pct = results['mixed_quality_pred_a']['count'] / mq_cnt
            f.write(f"Total cases with disagreement (1 to 4 refs): {mq_cnt} ({(mq_cnt/total_images):.2%} of the dataset)\n\n")
            
            f.write(f"Model behavior under ambiguity:\n")
            f.write(f" - Decides to classify as UNANSWERABLE: {results['mixed_quality_pred_u']['count']} times ({mq_pred_u_pct:.2%})\n")
            f.write(f" - Decides to try to DESCRIBE it: {results['mixed_quality_pred_a']['count']} times ({mq_pred_a_pct:.2%})\n\n")
            
            f.write("Metrics in this subgroup:\n")
            f.write(f"{'Stratum':<25} | {'N':<5} | {'BLEU-1':<7} | {'BLEU-2':<7} | {'ROUGE-L':<7} | {'METEOR':<7}\n")
            f.write("-" * 65 + "\n")
            write_row("Mixed (All)", results['mixed_quality_all'])
            write_row("Mixed -> Pred=UNANSW", results['mixed_quality_pred_u'])
            write_row("Mixed -> Pred=ANSW", results['mixed_quality_pred_a'])
        else:
            f.write("No disagreement cases found in this dataset.\n")
            
    print(f"Report successfully generated at: {output_txt}")

if __name__ == "__main__":
    # Adjust these paths according to your environment
    INPUT_JSON = "../Week4/outputs/results/task2_2_eval_clean.json"
    OUTPUT_TXT = "./outputs/qwen_extensive_report.txt"
    
    if os.path.exists(INPUT_JSON):
        main(INPUT_JSON, OUTPUT_TXT)
    else:
        print(f"File {INPUT_JSON} not found.")
