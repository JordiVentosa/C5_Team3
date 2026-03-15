# C5 Project — Week 2: Object Segmentation

Team 3 repository for the [C5 — Visual Recognition](https://mcv.uab.cat/c5-visual-recognition/) course of the Master in Computer Vision at UAB.

**Team members:** Aleix Armero Rofes, Marc Artero Pons, Shinto Machado Furuichi, Adrià Ruiz Puig, Jordi Ventosa Altimira.

---

## Overview

This project implements **instance segmentation** on the [KITTI-MOTS](https://www.cvlibs.net/datasets/kitti/) dataset using SAM ([Segment Anything Model](https://huggingface.co/docs/transformers/main/en/model_doc/sam)) with different types of prompts: point-based, text-based (Grounded SAM), and automatic fine-tuning.

### Implemented Tasks

| Task | Description | Status |
|---|---|---|
| **A** | SAM inference with point prompts
| **B** | Grounded SAM + Grounding DINO with text prompts
| **C** | SAM inference with YOLO bounding box prompts
| **E** | Fine-tune SAM's Prompt Decoder using LoRA
| **F** | Domain shift evaluation: pre-trained vs fine-tuned on iSAID
| **H** | Semantic segmentation using Grounded SAM

---

## Repository Structure

```
Week2/
├── config/                         # Experiment configuration files
├── src/                            # Reusable project source code
│   ├── models/                     # Model definitions and utilities
│   ├── utils/                      # Common utilities
│   ├── evaluate.py                 # Common evaluation functions
│   ├── inference.py                # Common inference functions
│   └── runners.py                  # Common runners for pipelines
├── task_e_and_f/                   # Tasks E & F — Fine-tuning and domain shift
│   ├── train_sam.py               # Fine-tune SAM prompt decoder with LoRA
│   ├── inference_domain_shift.py  # Evaluate pre-trained vs fine-tuned on domain shift
│   ├── dataset.py                 # KITTI-MOTS dataset loader
│   ├── augmentations.py           # Data augmentation strategies
│   ├── evaluators.py              # COCO evaluation metrics
│   ├── collate.py                 # DataLoader collate functions
│   ├── prompts.py                 # Prompt generation utilities
│   ├── visualization.py           # Visualization helpers
│   ├── SAM_Analysis.ipynb         # Analysis notebook for SAM behavior
│   ├── evaluate_pretrained.py     # Evaluation pipeline for pre-trained SAM
│   ├── evaluate_all_prompts.py    # Evaluation with different prompt types
│   ├── qualitative_examples_ft.py # Qualitative examples from fine-tuned model
│   └── [shell scripts]            # Helper scripts for running experiments (dshift*.sh, evallprompts*.sh, etc.)
├── .gitignore                      # Git ignored files rules
├── evaluate_job.sbatch             # SLURM script to launch evaluation on a cluster
├── README.md                       # This file
├── taskA.py                        # Task A — SAM inference with point prompts
├── taskAQuantitative.py            # Task A — Quantitative evaluation with COCO metrics
├── taskB.py                        # Task B — Grounded SAM with text prompts
├── taskH.py                        # Task H — Semantic segmentation with Grounded SAM
└── taskHQuantitative.py            # Task H — Quantitative evaluation with COCO metrics
```

---

## Usage

All scripts are meant to be run from the **repository root** (the parent of `Week2/`), except scripts from the `task_e_and_f` folder. Most scripts accept `--help` for a full list of arguments.

### Task b and c evaluation and inference

To evaluate and obtain metrics for these task you must do:
```
python src/evaluate.py --config /path/to/config/yaml
```

The configuration files for task b and c evaluations are found inside config/evaluations. These contain some parameters that can be tweaked, such as the model used or some thresholds or values that are specific for the used models.

To do inference a similar thing must be done:
```
python src/inference.py --config /path/to/config/yaml
```

The configuration files for task b and c inference are found inside config/inference.
In this case, to change the image or images to which you want to do inference, you just need to change the index_values list inside the configurations.

Both inference and evaluation configurations allows to specify the root folder for KITTI-MOTS, the split to use, and the output folder where the results will be dumped. If the output folder does not exists, it is created.

## Dataset

The [KITTI-MOTS](https://www.cvlibs.net/datasets/kitti/) dataset is expected at a path like `/path/to/KITTI-MOTS` with the following structure:

```
KITTI-MOTS/
├── training/
│   └── image_02/
│       ├── 0000/
│       ├── 0001/
│       └── ... (sequences up to 0020)
├── testing/
│   └── image_02/
├── instances/          # Instance segmentation masks — training only
└── instances_txt/      # Instance annotations (TXT) — training only
```

### Data Split

Ground-truth annotations are **only available for the training split** (21 sequences):
- **Training:** Sequences 0000–0015 (16 sequences)
- **Validation:** Sequences 0016–0020 (5 sequences)

See `src/utils/train.seqmap` and `src/utils/val.seqmap` for the split definition.

The testing split (sequences 0021+) is used only for qualitative evaluation as no ground-truth is available.

### Domain Shift Dataset

For Task F (domain shift evaluation), the [iSAID](https://captain-whu.github.io/DiRS/) dataset is automatically downloaded from HuggingFace Hub during the first run of `inference_domain_shift.py`.

---

## Evaluation Metrics

All quantitative evaluations use the official **COCO metrics** via `pycocotools`:

- **AP (Average Precision)** @ IoU thresholds: 0.50:0.95, 0.50, 0.75
- **AP for object sizes:** small, medium, large objects
- **AR (Average Recall)** @ different detection counts: 1, 10, 100
- **AR for object sizes:** small, medium, large objects

These metrics are computed for both instance and semantic segmentation tasks.

## Dependencies

Install them via:
```bash
pip install -r requirements.txt
```