# C5 Project — Week 3: Image Captioning

Team 3 repository for the [C5 — Visual Recognition](https://mcv.uab.cat/c5-visual-recognition/) course of the Master in Computer Vision at UAB.

**Team members:** Aleix Armero Rofes, Marc Artero Pons, Shinto Machado Furuichi, Adrià Ruiz Puig, Jordi Ventosa Altimira.

---

## Overview

This project implements **image captioning** on the [VizWiz-Captions](https://vizwiz.org/) dataset using encoder-decoder architectures. The model generates natural language descriptions for images using a ResNet encoder and an RNN decoder with optional attention mechanisms.

### Model Architecture

| Component | Options |
|---|---|
| **Encoder** | ResNet-18, ResNet-34 (via HuggingFace) |
| **Decoder** | GRU, LSTM |
| **Attention** | Optional attention mechanism |
| **Tokenizers** | Character-level, Word-level, Subword (BPE) |

---

## Repository Structure

```
Week3/
├── src/                             # Source code
│   ├── models/
│   │   ├── baseline.py              # Encoder-decoder model
│   │   ├── train_wrapper.py         # PyTorch Lightning wrapper
│   │   └── metrics.py               # Evaluation metrics
│   ├── custom_datasets/
│   │   ├── vizwiz.py                # VizWiz dataset utilities
│   │   └── vizwiz_dataset.py        # PyTorch dataset wrapper
│   ├── text_tokenizers/
│   │   ├── character.py             # Character-level tokenizer
│   │   ├── word.py                  # Word-level tokenizer
│   │   ├── subword.py               # Subword (BPE) tokenizer
│   │   └── base.py                  # Base tokenizer class
│   ├── train_lightning.py           # Training script
│   ├── evaluate_lightning.py        # Evaluation script
│   ├── train_sweep.py               # WandB hyperparameter sweep
│   └── inference.py                 # Inference script
├── config/
│   └── sweep_best.yaml              # WandB sweep configuration
├── data/                            # Dataset directory (not in repo)
│   └── annotations/
│       ├── train.json
│       └── val.json
├── checkpoints/                     # Model checkpoints
├── logs/                            # Training logs
├── Baseline Model and Metrics.ipynb # EDA and baseline exploration
├── job.sbatch                       # Generic SLURM job
├── requirements.txt                 # Dependencies
└── README.md                        # This file
```

---

## Environment Setup

```bash
conda create -n c5_week3 python=3.10.19 -y
conda activate c5_week3
pip install -r Week3/requirements.txt
```

> **Note:** Requires a CUDA-capable GPU. PyTorch is installed with CUDA support.

---

## Dataset

The [VizWiz-Captions](https://vizwiz.org/) dataset consists of images taken by people who are blind, each with 5 human-written captions. The dataset is expected at `Week3/data/` with the following structure:

```
data/
├── images/
│   ├── train/
│   └── val/
└── annotations/
    ├── train.json
    └── val.json
```

Dataset statistics:
- **Train split:** ~23,000 images with ~115,000 captions (5 per image)
- **Val split:** ~7,750 images with ~38,750 captions (5 per image)

---

## Usage

All scripts are meant to be run from the **Week3/** directory.

### Training

```bash
# Basic training
python src/train_lightning.py \
    --run_name baseline_experiment \
    --data_root ./data \
    --tokenizer_type word \
    --resnet_model microsoft/resnet-34 \
    --rnn_type GRU \
    --freeze_encoder no \
    --batch_size 128 \
    --epochs 50 \
    --learning_rate 0.0003 \
    --teacher_forcing_ratio 0.25 \
    --num_workers 8 \
    --attention yes \
    --optimizer AdamW
```

### Hyperparameter Tuning with WandB

```bash
# Initialize sweep
wandb sweep config/sweep_best.yaml

# Run sweep agent
wandb agent <sweep_id>
```

Or use the provided script:

```bash
bash run-wandb-sweep.sh
```

### Evaluation

```bash
python src/evaluate_lightning.py \
    --checkpoint_path checkpoints/best_model.ckpt \
    --data_root ./data \
    --batch_size 128
```

### Inference

```bash
python src/inference.py \
    --checkpoint_path checkpoints/best_model.ckpt \
    --image_path path/to/image.jpg
```

### SLURM Cluster

For running on a SLURM cluster:

```bash
sbatch run-train.sh
```

---

## Evaluation Metrics

All quantitative evaluations use standard caption generation metrics via HuggingFace `evaluate`:

- **BLEU-1, BLEU-2:** N-gram precision-based metrics
- **ROUGE-L:** Longest common subsequence-based metric
- **METEOR:** Alignment-based metric with synonyms and stemming

These metrics are computed by comparing generated captions against reference (ground-truth) captions.

---

## Model Configuration

Key hyperparameters:

| Parameter | Description | Default |
|---|---|---|
| `tokenizer_type` | Type of tokenizer (character, word, subword) | word |
| `resnet_model` | ResNet variant for encoder | microsoft/resnet-34 |
| `rnn_type` | RNN decoder type (GRU, LSTM) | GRU |
| `freeze_encoder` | Freeze encoder weights (yes, no) | no |
| `attention` | Use attention mechanism (yes, no) | yes |
| `learning_rate` | Learning rate | 0.0003 |
| `teacher_forcing_ratio` | Teacher forcing ratio | 0.25 |
| `batch_size` | Training batch size | 128 |
| `epochs` | Number of training epochs | 50 |
| `optimizer` | Optimizer type (adam, adamw, sgd) | AdamW |
