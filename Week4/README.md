# C5 Project — Week 4: Image Captioning with Vision-Language Models

Team 3 repository for the [C5 — Visual Recognition](https://mcv.uab.cat/c5-visual-recognition/) course of the Master in Computer Vision at UAB.

**Team members:** Aleix Armero Rofes, Marc Artero Pons, Shinto Machado Furuichi, Adrià Ruiz Puig, Jordi Ventosa Altimira.

---

## Overview

This project explores **image captioning** on the [VizWiz-Captions](https://vizwiz.org/) dataset using both classical encoder-decoder architectures and modern Vision-Language Models (VLMs). We progressively scale from a small ViT-GPT2 baseline to large multimodal models like Qwen2.5-VL, Qwen3-VL and Qwen3.5, including LoRA fine-tuning of a hybrid ViT + Qwen captioner.

### Implemented Tasks

| Task | Description |
|---|---|
| **1.1** | Zero-shot evaluation of pre-trained ViT-GPT2 on VizWiz |
| **1.2** | Fine-tuning ViT-GPT2 on VizWiz (encoder-only / decoder-only / full) |
| **2.1** | Zero-shot evaluation of large VLMs (Qwen2.5-VL-7B, Qwen3-VL-8B, Qwen3.5-9B) with basic and advanced prompts |
| **2.2** | Hybrid ViT encoder + Qwen3.5 decoder fine-tuned with LoRA adapters |

---

## Repository Structure

```
Week4/
├── src/
│   ├── dataset.py                  # VizWiz dataset loader
│   ├── metrics.py                  # BLEU / METEOR / ROUGE / CIDEr metrics
│   └── __init__.py
├── task1_1_pretrained.py           # Task 1.1 — ViT-GPT2 zero-shot eval
├── task1_2_finetune.py             # Task 1.2 — ViT-GPT2 fine-tuning
├── task2_1.py                      # Task 2.1 — Qwen2.5-VL-7B eval
├── task2_1_qwen3vl.py              # Task 2.1 — Qwen3-VL-8B eval
├── task2_1_qwen35.py               # Task 2.1 — Qwen3.5-9B eval
├── task2_2.py                      # Task 2.2 — ViT + Qwen3.5 LoRA fine-tuning
├── save_encoder.py                 # Extract & save the frozen ViT encoder
├── visualize_predictions.py        # Generate qualitative figures from predictions
├── data/                           # Dataset directory (not in repo)
├── outputs/                        # Generated metrics, predictions and checkpoints
├── logs/                           # SLURM job logs
├── requirements.txt                # Dependencies
└── README.md                       # This file
```

---

## Environment Setup

```bash
conda create -n c5w4 python=3.10 -y
conda activate c5w4
pip install -r Week4/requirements.txt
```

> **Note:** Requires a CUDA-capable GPU. The large VLMs (Qwen2.5-VL, Qwen3-VL, Qwen3.5) need at least ~20 GB of VRAM in bfloat16. The LoRA fine-tuning of Task 2.2 was tested on H100 (80 GB).

For LoRA fine-tuning, the [PEFT](https://github.com/huggingface/peft) library is used.

---

## Dataset

The [VizWiz-Captions](https://vizwiz.org/) dataset consists of images taken by people who are blind, each annotated with 5 human-written captions. The dataset is expected at `Week4/data/` (or the absolute path passed via `--img_dir` / `--ann_file`):

```
data/
└── vizwiz/
    ├── train/                    # Training images (.jpg)
    ├── val/                      # Validation images (.jpg)
    └── annotations/
        ├── train.json
        └── val.json
```

Dataset statistics:
- **Train split:** ~23,000 images with ~115,000 captions (5 per image)
- **Val split:** ~7,750 images with ~38,750 captions (5 per image)

---

## Usage

All scripts are meant to be run from the **Week4/** directory. Most scripts accept `--help` for a full list of arguments.

### Task 1.1 — Pre-trained ViT-GPT2 Zero-Shot Evaluation

```bash
python task1_1_pretrained.py \
    --img_dir   data/vizwiz/val \
    --ann_file  data/vizwiz/annotations/val.json \
    --output_file outputs/task1_1
```

### Task 1.2 — Fine-tuning ViT-GPT2

Three fine-tuning modes are available: `backbone` (train only the ViT encoder), `captioner` (train only the GPT-2 decoder), or `all` (full end-to-end).

```bash
python task1_2_finetune.py \
    --train_img_dir  data/vizwiz/train \
    --train_ann_file data/vizwiz/annotations/train.json \
    --val_img_dir    data/vizwiz/val \
    --val_ann_file   data/vizwiz/annotations/val.json \
    --output_dir     outputs/task1_2 \
    --finetune_mode  captioner \
    --epochs         5 \
    --lr             5e-5
```

### Task 2.1 — Zero-shot VLM Evaluation

Each script supports two prompt modes: `basic` (a simple instruction) and `advanced` (a detailed system prompt tailored to visually impaired users).

```bash
# Qwen2.5-VL-7B
python task2_1.py \
    --img_dir   data/vizwiz/val \
    --ann_file  data/vizwiz/annotations/val.json \
    --model_name Qwen/Qwen2.5-VL-7B-Instruct \
    --prompt_mode basic \
    --output_file outputs/task2_1_val_basic

# Qwen3-VL-8B
python task2_1_qwen3vl.py \
    --img_dir   data/vizwiz/val \
    --ann_file  data/vizwiz/annotations/val.json \
    --model_name Qwen/Qwen3-VL-8B-Instruct \
    --prompt_mode advanced \
    --output_file outputs/task2_1_qwen3vl_val_advanced

# Qwen3.5-9B (vision integrated, requires trust_remote_code)
python task2_1_qwen35.py \
    --img_dir   data/vizwiz/val \
    --ann_file  data/vizwiz/annotations/val.json \
    --model_name Qwen/Qwen3.5-9B \
    --prompt_mode basic \
    --output_file outputs/task2_1_qwen35_val_basic
```

### Task 2.2 — ViT + Qwen3.5 LoRA Fine-tuning

A frozen ViT encoder (extracted from `nlpconnect/vit-gpt2-image-captioning`) is connected via a learned projection MLP to a Qwen3.5 decoder, which is fine-tuned with LoRA adapters on the attention layers.

```bash
python task2_2.py \
    --train_img_dir  data/vizwiz/train \
    --train_ann_file data/vizwiz/annotations/train.json \
    --val_img_dir    data/vizwiz/val \
    --val_ann_file   data/vizwiz/annotations/val.json \
    --vit_model      /path/to/modelo_vit_gpt2 \
    --qwen_model     /path/to/modelo_qwen35_4b \
    --lora_r         8 \
    --lora_alpha     16 \
    --batch_size     16 \
    --epochs         5 \
    --output_dir     outputs/task2_2/qwen35_4b/lora_r8_a16
```

LoRA configurations explored: `(r=8, α=16)`, `(r=16, α=32)`, `(r=8, α=32)`, `(r=16, α=16)`, on both Qwen3.5-0.8B and Qwen3.5-4B.

### Visualization

```bash
# Generate qualitative figures + text summary from any predictions JSON
python visualize_predictions.py \
    --preds_file outputs/results/task2_2_best_predictions.json \
    --output_dir outputs/qualitative \
    --n          10
```

---

## Evaluation Metrics

All quantitative evaluations use standard caption generation metrics:

- **BLEU-1, BLEU-2** — N-gram precision-based metrics
- **METEOR** — Alignment-based metric with synonyms and stemming
- **ROUGE-L** — Longest common subsequence-based metric

Predictions are compared against the 5 reference captions per image. Metrics are computed on lowercased strings.

---

## Models

| Model | Source | Approx. size | Used in |
|---|---|---|---|
| ViT-GPT2 | `nlpconnect/vit-gpt2-image-captioning` | ~240 M | Tasks 1.1, 1.2, 2.2 (encoder) |
| Qwen2.5-VL-7B | `Qwen/Qwen2.5-VL-7B-Instruct` | 7 B | Task 2.1 |
| Qwen3-VL-8B | `Qwen/Qwen3-VL-8B-Instruct` | 8 B | Task 2.1 |
| Qwen3.5-9B | `Qwen/Qwen3.5-9B` | 9 B | Task 2.1 |
| Qwen3.5-0.8B | `Qwen/Qwen3.5-0.8B` | 0.8 B | Task 2.2 (LoRA decoder) |
| Qwen3.5-4B | `Qwen/Qwen3.5-4B` | 4 B | Task 2.2 (LoRA decoder) |