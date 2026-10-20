#!/bin/bash
# --------------------------------------------------------------------
# File:    slurm_predict.sh
# Author:  Gurkirat Singh Randhawa (s5054910)
# Course:  COMP3710 Pattern Recognition, The University of Queensland
# Date:    October 2026
# Purpose: Slurm job script that runs predict.py on one Rangpur A100.
# --------------------------------------------------------------------
#SBATCH --account=comp3710
#SBATCH --job-name=hipmri-predict
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=predict_%j.out
#SBATCH --error=predict_%j.err

# Submit from the project folder, passing predict.py arguments after the script:
#   sbatch slurm_predict.sh --model dcgan

source $HOME/miniconda3/bin/activate torch

python3 predict.py "$@"
