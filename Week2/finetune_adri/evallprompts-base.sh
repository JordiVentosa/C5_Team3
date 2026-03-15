#!/bin/bash
#SBATCH -p mlow # Partition to submit to
#SBATCH --mem 24G # 24GB memory
#SBATCH --gres gpu:1 # Request of 1 gpu
#SBATCH -o logs/%x_%u_%j.out # File to which STDOUT will be written
#SBATCH -e logs/%x_%u_%j.err # File to which STDERR will be written

python evaluate_all_prompts.py