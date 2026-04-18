# generate_synthetic_captions.py
import argparse
import json
import random
from pathlib import Path
from typing import List
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from dataset import VizWizDataset

model_id = "Qwen/Qwen3.5-35B-A3B"
QUALITY_ISSUE_CAPTION = "quality issues are too severe to recognize visual content."
SYSTEM_PROMPT = "You are a captioning assistant. Respond with a single caption sentence only. No explanations, no lists, no JSON."

def load_vizwiz_examples(img_dir: str, ann_file: str) -> List[List[str]]:
    dataset = VizWizDataset(img_dir=img_dir, ann_file=ann_file, feature_extractor=None)
    quality_sets = []
    for i in range(len(dataset)):
        item = dataset[i]
        captions = item["captions"]
        has_quality_issue = any(
            cap.lower().strip() == QUALITY_ISSUE_CAPTION.lower().strip()
            for cap in captions
        )
        if has_quality_issue:
            other_captions = [
                cap for cap in captions
                if cap.lower().strip() != QUALITY_ISSUE_CAPTION.lower().strip()
            ]
            if len(other_captions) >= 2:
                quality_sets.append(other_captions)
    print(f"[INFO] Caption sets con quality issues: {len(quality_sets)}")
    return quality_sets

def call_model(user_prompt: str, system_prompt: str, model, tokenizer, max_new_tokens=100) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    new_tokens = output[:, inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens[0], skip_special_tokens=True).strip()

def generate_global_caption(examples: List[List[str]], model, processor) -> str:
    sampled = random.sample(examples, min(5, len(examples)))
    examples_str = ""
    for caps in sampled:
        examples_str += "\n".join(f"  - {c}" for c in caps) + "\n\n"
    prompt = (
        f"Here are real captions written by people describing images with severe quality issues:\n\n"
        f"{examples_str}"
        f"Write a single global caption describing a NEW different image in the same style. Rules:\n"
        f"- Always describe what IS visible (objects, colors, setting), even if the image has quality issues\n"
        f"- Never say the image is 'too blurry to identify' or 'impossible to see' — always name the subject\n"
        f"- Occasionally mention a brand name, label text, or readable text visible on objects\n"
        f"- Short, natural, human-like. One sentence only."
    )
    return call_model(prompt, SYSTEM_PROMPT, model, processor)

def generate_specific_caption(global_caption: str, existing: List[str], model, processor) -> str:
    existing_str = "\n".join(f"- {c}" for c in existing) if existing else "None yet."
    prompt = (
        f"Global image description: {global_caption}\n\n"
        f"Already written captions for this image:\n{existing_str}\n\n"
        f"Write ONE new caption describing the same scene. Rules:\n"
        f"- Shorter and less detailed than the global caption, but not too vague\n"
        f"- Must be clearly different from the already written captions above\n"
        f"- Each caption should emphasize a different element (object, color, material, position)\n"
        f"- Vary the level of detail: some captions can be very specific, others more general\n"
        f"- If the global mentions a brand or text, only some captions should reference it\n"
        f"- Do not invent details not present in the global description\n"
        f"- Natural and human-like, one sentence only"
    )
    return call_model(prompt, SYSTEM_PROMPT, model, processor)

def generate_set(examples: List[List[str]], model, tokenizer) -> dict:
    global_cap = generate_global_caption(examples, model, tokenizer)
    specifics = []
    for _ in range(5):
        cap = generate_specific_caption(global_cap, specifics, model, tokenizer)
        specifics.append(cap)
    return {"global_caption": global_cap, "captions": specifics}

def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic VizWiz-style caption sets using Qwen LLM.")
    parser.add_argument("--img_dir", type=str, required=True)
    parser.add_argument("--ann_file", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--num_sets", type=int, default=100)
    parser.add_argument("--model", type=str, default=model_id)
    return parser.parse_args()

def load_model(mid):
    print(f"Loading model {mid}...")
    tokenizer = AutoTokenizer.from_pretrained(mid, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        mid,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
    )
    model.eval()
    return model, tokenizer

def main():
    args = parse_args()
    examples = load_vizwiz_examples(args.img_dir, args.ann_file)
    if not examples:
        print("[ERROR] No se encontraron caption sets con quality issues. Saliendo.")
        return
    print(f"Loaded {len(examples)} example caption sets with quality issues.")
    model, tokenizer = load_model(args.model)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for i in range(args.num_sets):
        print(f"[{i+1}/{args.num_sets}] Generating set...")
        set_data = generate_set(examples, model, tokenizer)
        results.append(set_data)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Done. {len(results)} sets saved to {output_path}")

if __name__ == "__main__":
    main()