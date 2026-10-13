#!/bin/bash
#SBATCH --account=comp3710
#SBATCH --job-name=hipmri-train
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=train_%j.out
#SBATCH --error=train_%j.err

# Submit from the project folder, passing train.py arguments after the script:
#   sbatch slurm_train.sh --model dcgan --epochs 100

source $HOME/miniconda3/bin/activate torch

python3 train.py "$@"
