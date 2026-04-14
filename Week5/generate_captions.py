import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate captions using a local Qwen LLM and append them to a txt file."
    )
    parser.add_argument("--instructions", type=str, required=True, help="Path to txt file describing the type of captions to generate.")
    parser.add_argument("--num_captions", type=int, required=True, help="Number of captions to generate.")
    parser.add_argument("--output", type=str, required=True, help="Path to output txt file (captions are appended).")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="HuggingFace model ID.")
    return parser.parse_args()


def load_model(model_id):
    print(f"Loading model {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    return model, tokenizer


def generate_caption(model, tokenizer, instructions):
    messages = [
        {"role": "system", "content": instructions},
        {"role": "user", "content": "Generate a single image caption. Output only the caption, nothing else."},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.9,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def main():
    args = parse_args()

    with open(args.instructions) as f:
        instructions = f.read().strip()

    model, tokenizer = load_model(args.model)

    print(f"Generating {args.num_captions} captions, appending to {args.output}")
    with open(args.output, "a") as out_f:
        for i in range(args.num_captions):
            caption = generate_caption(model, tokenizer, instructions)
            out_f.write(caption + "\n")
            out_f.flush()
            print(f"[{i+1}/{args.num_captions}] {caption}")

    print("Done.")


if __name__ == "__main__":
    main()
