#!/bin/bash
#SBATCH --job-name=SAM_FT_LoRA_Array
#SBATCH -p mlow                 # Partition to submit to
#SBATCH --mem 24G               # 24GB memory
#SBATCH --gres gpu:1            # Request of 1 gpu
#SBATCH --array=0,1             #
#SBATCH -o logs/%x_%u_%A_%a.out # %A es el ID del array, %a es el ID de la tarea
#SBATCH -e logs/%x_%u_%A_%a.err 

python train_sam.py --da_config ${SLURM_ARRAY_TASK_ID}
