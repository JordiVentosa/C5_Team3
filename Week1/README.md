# C5 Project — Week 1: Object Detection

Team 3 repository for the [C5 — Visual Recognition](https://mcv.uab.cat/c5-visual-recognition/) course of the Master in Computer Vision at UAB.

**Team members:** Aleix Armero Rofes, Marc Artero Pons, Shinto Machado Furuichi, Adrià Ruiz Puig, Jordi Ventosa Altimira.

---

## Overview

This project covers **object detection** on the [KITTI-MOTS](https://www.cvlibs.net/datasets/kitti/) dataset using four models from three different frameworks:

| Model | Framework | Env / Requirements |
|---|---|---|
| **Faster R-CNN** | PyTorch (torchvision) | `requirements_pytorch.txt` |
| **DeTR** | HuggingFace Transformers | `requirements_detr.txt` |
| **RT-DETR v2** | HuggingFace Transformers | `requirements_pytorch.txt` |
| **YOLO** | Ultralytics | `requirements_yolo.txt` |

The following tasks are implemented:

| Task | Description |
|---|---|
| **C** | Qualitative inference with pre-trained models on KITTI-MOTS |
| **D** | Quantitative evaluation (COCO metrics) on KITTI-MOTS |
| **E** | Fine-tuning on KITTI-MOTS (train seqs 0000–0015, val seqs 0016–0020) |
| **F** | Domain-shift fine-tuning (DEArt dataset for DeTR/Faster R-CNN) |
| **H** | Fine-tuning RT-DETR v2 on KITTI-MOTS |

---

## Repository Structure

```
Week1/
├── configs/                        # YOLO augmentation configs
│   ├── yolo_aug_none.yaml
│   ├── yolo_aug_light.yaml
│   └── yolo_aug_heavy.yaml
├── scripts/                        # Runnable task scripts
│   ├── task_c_faster.py            # Task C — Faster R-CNN inference
│   ├── task_c_yolo.py              # Task C — YOLO inference
│   ├── task_c_detr.py              # Task C — DeTR inference
│   ├── task_d_faster.py            # Task D — Faster R-CNN evaluation
│   ├── task_d_yolo.py              # Task D — YOLO evaluation
│   ├── task_d_detr.py              # Task D — DeTR evaluation
│   ├── task_e_faster.py            # Task E — Fine-tune Faster R-CNN
│   ├── task_e_detr.py              # Task E — Fine-tune DeTR
│   ├── task_e_yolo.py              # Task E — Fine-tune YOLO
│   ├── task_f_faster.py            # Task F — Fine-tune Faster R-CNN on DEArt
│   ├── task_f_detr.py              # Task F — Fine-tune DeTR on DEArt
│   ├── task_h_detr.py              # Task H — Fine-tune RT-DETR v2
│   └── debug.py
├── src/
│   ├── models/
│   │   ├── huggingface_detr.py     # DeTR model loading, inference & visualization
│   │   ├── torchvision_faster_rcnn.py  # Faster R-CNN wrapper
│   │   └── ultralytics_yolo.py     # YOLO wrapper
│   ├── utils/
│   │   ├── kitti_helpers.py        # KITTI-MOTS dataset helpers & COCO evaluation
│   │   ├── dataset.py              # KITTI dataset (Faster R-CNN)
│   │   ├── dataset_rcnn.py         # KITTI dataset with augmentations (Faster R-CNN)
│   │   ├── dataset_hf.py           # KITTI dataset for HuggingFace (RT-DETR)
│   │   ├── DeArtdataset.py         # DEArt dataset (Faster R-CNN domain shift)
│   │   ├── evaluate.py             # MAP evaluator for HuggingFace Trainer
│   │   ├── fine_tune_utils.py      # Training helpers (Faster R-CNN)
│   │   ├── fine_tune_utils_rt_detr.py  # Training helpers (RT-DETR)
│   │   ├── augmentations.py        # Shared augmentation pipelines
│   │   ├── metrics.py              # Additional metric utilities
│   │   ├── mots.py                 # KITTI-MOTS annotation parser
│   │   ├── train.seqmap            # Training split sequence list (seqs 0000–0015)
│   │   └── val.seqmap              # Validation split sequence list (seqs 0016–0020)
│   └── data/
│       └── coco_mapper.py          # COCO ↔ KITTI class mapping
├── notebooks/
│   └── 01_EDA_datasets.ipynb       # Exploratory data analysis
├── requirements_detr.txt           # DeTR environment
├── requirements_pytorch.txt        # Faster R-CNN & RT-DETR environment
└── requirements_yolo.txt           # YOLO environment
```

---

## Environment Setup

Each model group uses a **separate set of dependencies** to avoid version conflicts. We recommend creating one conda environment per group:

### 1. DeTR

```bash
conda create -n c5_detr python=3.11 -y
conda activate c5_detr
pip install -r Week1/requirements_detr.txt
```

### 2. Faster R-CNN & RT-DETR

```bash
conda create -n c5_pytorch python=3.11 -y
conda activate c5_pytorch
pip install -r Week1/requirements_pytorch.txt
```

### 3. YOLO

```bash
conda create -n c5_yolo python=3.11 -y
conda activate c5_yolo
pip install -r Week1/requirements_yolo.txt
```

> **Note:** All environments require a CUDA-capable GPU. The requirements include PyTorch with CUDA 12.1 support.

---

## Usage

All scripts are meant to be run from the **repository root** (the parent of `Week1/`). Most scripts accept `--help` for a full list of arguments.

### Task C — Qualitative Inference

```bash
# DeTR
python -m Week1.scripts.task_c_detr --kitti_root /path/to/KITTI-MOTS --split testing

# Faster R-CNN
python -m Week1.scripts.task_c_faster

# YOLO
python -m Week1.scripts.task_c_yolo
```

### Task D — Quantitative Evaluation

```bash
# DeTR
python -m Week1.scripts.task_d_detr --kitti_root /path/to/KITTI-MOTS

# Faster R-CNN
python -m Week1.scripts.task_d_faster

# YOLO
python -m Week1.scripts.task_d_yolo
```

### Task E — Fine-tuning on KITTI-MOTS

```bash
# DeTR
python -m Week1.scripts.task_e_detr --kitti_root /path/to/KITTI-MOTS --epochs 5 --lr 5e-5

# Faster R-CNN
python -m Week1.scripts.task_e_faster --unfreeze_mode partial --optimizer adamw --aug_level light --epochs 40

# YOLO
python -m Week1.scripts.task_e_yolo
```

### Task F — Domain-shift Fine-tuning (DEArt)

```bash
# DeTR
python -m Week1.scripts.task_f_detr --epochs 3 --lr 5e-5

# Faster R-CNN
python -m Week1.scripts.task_f_faster --unfreeze_mode partial --optimizer adamw --aug_level light --epochs 40
```

### Task H — RT-DETR v2

```bash
python -m Week1.scripts.task_h_detr --epochs 40 --lr 1e-4 --aug_level light
```

---

## Dataset

The [KITTI-MOTS](https://www.cvlibs.net/datasets/kitti/) dataset is expected at a path like `/path/to/KITTI-MOTS` with the following structure:

```
KITTI-MOTS/
├── training/
│   └── image_02/
│       ├── 0000/
│       ├── 0001/
│       └── ...
├── testing/
│   └── image_02/
├── instances/          # Instance segmentation masks — training only
└── instances_txt/      # Instance annotations (TXT) — training only
```

> Ground-truth annotations are **only available for the training split** (21 sequences). We divide it into train (seqs 0000–0015) and validation (seqs 0016–0020) using `src/utils/train.seqmap` and `src/utils/val.seqmap`. The testing split is used only for qualitative evaluation.

For domain-shift experiments (Task F), the [DEArt (European Art)](https://huggingface.co/datasets/biglam/european_art) dataset is loaded automatically from HuggingFace Hub.

---

## Evaluation

All quantitative evaluations use the official **COCO metrics** via `pycocotools`:

- AP @ IoU=0.50:0.95, AP @ 0.50, AP @ 0.75
- AP for small / medium / large objects
- AR @ 1, 10, 100 detections
- AR for small / medium / large objects