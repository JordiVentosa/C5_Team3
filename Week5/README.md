# C5 Project — Week 5: Synthetic Data for Blind-Photography Captioning

Team 3 repository for the [C5 — Visual Recognition](https://mcv.uab.cat/c5-visual-recognition/) course of the Master in Computer Vision at UAB.

**Team members:** Aleix Armero Rofes, Marc Artero Pons, Shinto Machado Furuichi, Adrià Ruiz Puig, Jordi Ventosa Altimira.

---

## Overview

This week extends the Week 4 captioning work in three directions:

1. **Synthetic data generation** — A pipeline that produces VizWiz-style images (blind-photography artefacts) and their captions using Stable Diffusion 3.5, FLUX.2 and large language models (Qwen3.5), with the goal of augmenting the training set for images flagged as "quality issues".

2. **Text-to-image model comparison** — Side-by-side comparison of several diffusion models (SD 1.4 through SDXL, FLUX.2) and controlled ablations on the SDXL pipeline (sampler choice, CFG scale, number of steps, prompting style).

3. **Captioning with synthetic data (Task E)** — The best Week 4 captioning architecture (frozen ViT + Qwen3.5 decoder with LoRA) is re-trained including the synthetic images, and a qualitative analysis is carried out specifically on "quality issues" images.

### Implemented Tasks

| Task | Description |
|---|---|
| **1** | Text-to-image generation comparison across SD 1.4/1.5/2.1/XL and FLUX.2 |
| **2** | SDXL ablation experiments: sampler, CFG, steps, prompting, denoising visualisation |
| **E** | ViT + Qwen3.5 LoRA captioning re-trained with synthetic data; qualitative analysis |

---

## Repository Structure

```
Week5/
├── src/
│   ├── dataset.py                      # VizWiz dataset loader (shared with Week 4)
│   ├── metrics.py                      # BLEU / METEOR / ROUGE-L metrics
│   └── __init__.py
│
├── outputs/                            # Generated metrics, captions and qualitative figures
│
│ ── Text-to-Image Tasks ────────────────────────────────────────────────────
├── task1_t2i_generation.py             # Task 1 — multi-model T2I generation
├── task2_sdxl_experiments.py           # Task 2 — SDXL ablation experiments
│
│ ── Captioning (Task E) ─────────────────────────────────────────────────────
├── task_e_finetune.py                  # Task E — ViT+Qwen LoRA fine-tuning (VizWiz only)
├── task_e_finetune_synthetic.py        # Task E — same + optional synthetic data
├── task_e_evaluate.py                  # Task E — full validation-set evaluation
├── task_e_infer_qualitative.py         # Task E — inference on a small image subset
│
│ ── Synthetic Data Pipeline ────────────────────────────────────────────────
├── synth_generate_prompts.py           # Step 1 — vLLM/Qwen generates image prompts
├── synth_generate_sd35_images.py       # Step 2a — SD3.5 generates images from prompts
├── synth_generate_vizwiz_images.py     # Step 2b — FLUX2/SD3.5+LoRA VizWiz-style images
├── synth_generate_flux2_images.py      # Step 2c — FLUX.2 generates images from JSON
├── synth_generate_captions.py          # Step 3 — Qwen3.5-35B generates caption sets
├── synth_generate_text_captions.py     # Alt — Qwen LLM generates captions from instructions
│
│ ── Analysis & Visualisation ───────────────────────────────────────────────
├── analyze_captioning_metrics.py       # Stratified metrics: quality-issues confusion matrix
├── visualize_qualitative_qi.py         # Side-by-side qualitative comparison figures
│
│ ── Utilities ──────────────────────────────────────────────────────────────
├── utils_flux2_quantized.py            # FLUX.2 4-bit quantised inference (quick test)
└── utils_flux2_simple.py               # Simple FLUX.2 generation sanity check
```

---

## Environment Setup

```bash
conda activate C5   # same environment as Week 4
pip install -r Week5/requirements.txt
```

> **Note:** FLUX.2 and SD3.5 require ~24 GB VRAM. Task E fine-tuning was run on an H100 (80 GB). Image generation experiments used an A100 (40 GB).

### Stable Diffusion LoRA Training

The LoRA adapters used to steer Stable Diffusion towards the VizWiz "blind-photography" visual style were trained with [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts), using its standard training scripts and configuration files. The resulting weights are then loaded at inference time by the scripts in this repository.

---

## Dataset

The [VizWiz-Captions](https://vizwiz.org/) dataset is expected at `Week4/data/` (same as Week 4). Synthetic images are stored separately and passed via `--synth_img_dir` / `--synth_ann_file`.

---

## Usage

### Task 1 — Text-to-Image Model Comparison

Runs one or more diffusion models on a given set of prompts and saves the generated images.

```bash
python Week5/task1_t2i_generation.py \
    --models sd1.5 sdxl flux2-dev-4bit \
    --prompts "a blurry photo of a can of soup" \
    --save_dir Week5/outputs/task1
```

Available models: `sd1.4`, `sd1.5`, `sd1.5-lcm`, `sd2.1`, `sd2.1-turbo`, `sdxl-turbo`, `sdxl`, `flux2-dev-4bit`.

---

### Task 2 — SDXL Ablation Experiments

Runs controlled experiments on Stable Diffusion XL. Each experiment sweeps one variable while keeping the rest fixed.

```bash
python Week5/task2_sdxl_experiments.py \
    --prompt "a blurry hand-held photo of a medicine bottle" \
    --save_dir Week5/outputs/task2 \
    --experiments sampler steps cfg prompting denoising_viz
```

Available experiments: `sampler`, `steps`, `cfg`, `prompting`, `denoising_viz`.

---

### Task E — ViT + Qwen3.5 LoRA Fine-tuning

#### Fine-tuning (VizWiz only)

```bash
python Week5/task_e_finetune.py \
    --train_img_dir  Week4/data/train \
    --train_ann_file Week4/data/annotations/train.json \
    --val_img_dir    Week4/data/val \
    --val_ann_file   Week4/data/annotations/val.json \
    --vit_model      /path/to/vit-gpt2 \
    --qwen_model     /path/to/qwen3.5-4b \
    --lora_r 8 --lora_alpha 16 \
    --epochs 5 --lr 1e-4 \
    --output_dir     Week5/outputs/task_e
```

#### Fine-tuning with Synthetic Data

Same arguments as above, plus:

```bash
    --synth_img_dir  /path/to/synthetic/images \
    --synth_ann_file /path/to/synthetic/annotations.json
```

Use `task_e_finetune_synthetic.py` instead of `task_e_finetune.py`.

#### Full Validation Evaluation

```bash
python Week5/task_e_evaluate.py \
    --vit_model    /path/to/vit-gpt2 \
    --qwen_model   /path/to/qwen3.5-4b \
    --lora_dir     /path/to/best_model/qwen_lora \
    --proj_path    /path/to/best_model/projection.pt \
    --val_img_dir  /path/to/vizwiz/val \
    --val_ann_file /path/to/vizwiz/annotations/val.json \
    --output_file  Week5/outputs/task_e_eval.json
```

#### Inference on a Small Subset (for qualitative analysis)

Useful when running on the cluster and only predictions for a handful of images are needed.

```bash
python Week5/task_e_infer_qualitative.py \
    --vit_model    /path/to/vit-gpt2 \
    --qwen_model   /path/to/qwen3.5-4b \
    --lora_dir     /path/to/checkpoint_epochN/qwen_lora \
    --projection   /path/to/checkpoint_epochN/projection.pt \
    --img_dir      /path/to/vizwiz/val \
    --image_list   Week5/outputs/qualitative_qi/qualitative_qi_examples.txt \
    --output_file  Week5/outputs/task_e_qualitative.json
```

`--image_list` accepts either the `qualitative_qi_examples.txt` format (auto-detected) or a plain list of filenames (one per line).

---

### Synthetic Data Pipeline

The pipeline runs in three steps:

#### Step 1 — Generate Image Prompts

Uses vLLM with Qwen3.5-9B to produce diverse prompts describing VizWiz-style scenes.

```bash
python Week5/synth_generate_prompts.py
# Output: a JSON file with image descriptions / prompts
```

#### Step 2 — Generate Images

Three alternative backends are available:

**SD3.5 (with optional LoRA trained via kohya-ss/sd-scripts):**
```bash
python Week5/synth_generate_sd35_images.py \
    --captions     Week5/outputs/synth_prompts.txt \
    --output_dir   Week5/outputs/synth_images_sd35 \
    --model_path   /path/to/sd35-medium \
    --lora_weights /path/to/lora_finetuned   # optional
```

**FLUX.2 with VizWiz-style LoRA:**
```bash
python Week5/synth_generate_vizwiz_images.py \
    --output_dir  Week5/outputs/synth_images_vizwiz \
    --ann_file    /path/to/vizwiz/annotations/train.json
```

**FLUX.2 from annotation JSON:**
```bash
python Week5/synth_generate_flux2_images.py \
    --synthetic_json  Week5/outputs/synth_captions.json \
    --output_img_dir  Week5/outputs/synth_images_flux2 \
    --output_ann_file Week5/outputs/synth_annotations.json \
    --repo_id         diffusers/FLUX.2-dev-bnb-4bit
```

#### Step 3 — Generate Captions

Uses Qwen3.5-35B to generate 5 human-style captions per synthetic image, following the VizWiz annotation format.

```bash
python Week5/synth_generate_captions.py \
    --img_dir   /path/to/vizwiz/train \
    --ann_file  /path/to/vizwiz/annotations/train.json \
    --output    Week5/outputs/synth_captions.json \
    --num_sets  500
```

Alternatively, generate captions from a text instruction file:

```bash
python Week5/synth_generate_text_captions.py \
    --instructions Week5/outputs/caption_instructions.txt \
    --num_captions 200 \
    --output       Week5/outputs/captions.txt
```

---

### Analysis & Visualisation

#### Stratified Metrics (Quality-Issues Confusion Matrix)

Computes BLEU/METEOR/ROUGE-L broken down by whether the model and annotators agreed on "quality issues". Produces a detailed text report.

```bash
python Week5/analyze_captioning_metrics.py
# Edit INPUT_JSON / OUTPUT_TXT paths inside the script
```

#### Qualitative Comparison Figures

Generates side-by-side comparison PNGs for images flagged as "quality issues", showing each model's prediction alongside the reference captions.

```bash
python Week5/visualize_qualitative_qi.py \
    --output_dir Week5/outputs/qualitative_qi \
    --n   20 \
    --seed 42
```

Prediction files to compare are configured via `PRED_FILES` at the top of the script.

---

## Models

| Model | Source / ID | Size | Used in |
|---|---|---|---|
| ViT-GPT2 | `nlpconnect/vit-gpt2-image-captioning` | ~240 M | Task E (ViT encoder) |
| Qwen3.5-4B | `Qwen/Qwen3.5-4B` | 4 B | Task E (LoRA decoder) |
| Qwen3.5-35B-A3B | `Qwen/Qwen3.5-35B-A3B` | 35 B (MoE) | Synthetic caption generation |
| Qwen3.5-9B | `Qwen/Qwen3.5-9B` | 9 B | Synthetic prompt generation |
| SD 1.4 / 1.5 / 2.1 | CompVis / Runway / community | ~1 B | Task 1 |
| Stable Diffusion XL | `stabilityai/stable-diffusion-xl-base-1.0` | ~6.6 B | Tasks 1 & 2 |
| SD 3.5 Medium | `stabilityai/stable-diffusion-3.5-medium` | ~8 B | Synthetic image generation |
| FLUX.2-dev | `diffusers/FLUX.2-dev-bnb-4bit` | ~12 B (4-bit) | Synthetic image generation |

---

## Evaluation Metrics

All captioning evaluations use the same metrics as Week 4:

- **BLEU-1, BLEU-2** — N-gram precision
- **METEOR** — Synonym/stem-aware alignment
- **ROUGE-L** — Longest common subsequence

`analyze_captioning_metrics.py` additionally reports a **confusion matrix** for the "quality issues" class, computing precision, recall and F1 for the model's ability to detect unanswerable images.
