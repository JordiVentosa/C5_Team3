#!/bin/bash
#SBATCH -p mhigh
#SBATCH --mem 24G
#SBATCH --gres gpu:1
#SBATCH -o logs/%x_%u_%j.out # File to which STDOUT will be written
#SBATCH -e logs/%x_%u_%j.err # File to which STDERR will be written

# Run your program
python src/evaluate.py --config config/evaluations/bbox_segmentation.yaml