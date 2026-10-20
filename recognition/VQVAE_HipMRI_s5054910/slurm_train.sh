#!/bin/bash
# --------------------------------------------------------------------
# File:    slurm_train.sh
# Author:  Gurkirat Singh Randhawa (s5054910)
# Course:  COMP3710 Pattern Recognition, The University of Queensland
# Date:    October 2026
# Purpose: Slurm job script that runs train.py on one Rangpur A100.
# --------------------------------------------------------------------
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
