# C5 Project — Week 2: Object Segmentation

Team 3 repository for the [C5 — Visual Recognition](https://mcv.uab.cat/c5-visual-recognition/) course of the Master in Computer Vision at UAB.

**Team members:** Aleix Armero Rofes, Marc Artero Pons, Shinto Machado Furuichi, Adrià Ruiz Puig, Jordi Ventosa Altimira.

---

## Overview

This project covers **object segmentation** on the [KITTI-MOTS](https://www.cvlibs.net/datasets/kitti/) dataset using SAM (https://huggingface.co/docs/transformers/tasks/object_detection) with different types of prompts (points, text, bbox...)

The following tasks are implemented:

| Task | Description |
|---|---|
| **A** | SAM inference with point prompts |
| **B** | Grounded SAM inference with text prompts |
| **C** | SAM inference with YOLO bboxes |
| **E** | Fine-tune the Prompt-Decoder of SAM |
| **F** | Pre-trained SAM and the finetuned version domain shift |
| **H** | Semantic Segmentation |

---

## Repository Structure

```
Week2/
├── config/                         # Experiment configuration files
├── src/                            # Reusable project source code
├── .gitignore                      # Git ignored files rules
├── evaluate_job.sbatch             # SLURM script to launch evaluation on a cluster
├── README.md                       # Project documentation, usage, and instructions
├── taskA.py                        # Task A — Inference of pretrained SAM with point prompts
├── taskAQuantitative.py            # Task A — Quantitative evaluation of results
├── taskB.py                        # Task B — Inference with text prompts (Grounded SAM)
├── taskH.py                        # Task H — Semantic segmentation (optional)
└── taskHQuantitative.py            # Task H — Quantitative evaluation of semantic segmentation
```

---

## Usage

All scripts are meant to be run from the **repository root** (the parent of `Week1/`). Most scripts accept `--help` for a full list of arguments.

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