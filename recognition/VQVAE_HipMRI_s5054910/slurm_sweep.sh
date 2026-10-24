#!/bin/bash
# --------------------------------------------------------------------
# File:    slurm_sweep.sh
# Author:  Gurkirat Singh Randhawa (s5054910)
# Course:  COMP3710 Pattern Recognition, The University of Queensland
# Date:    October 2026
# Purpose: Submits the VQ-VAE-2 + aug sampling sweep (top-k, top-p, top-level
#          temperature) on the validation patients. Run on the login node:
#            bash slurm_sweep.sh
#          Each setting gets an FID job, then an audit job that reuses its samples.
# --------------------------------------------------------------------
BASE="--model pixelcnn2 --run-name pixelcnn2_aug --top-weights $HOME/hipmri_runs/pixelcnn2_aug_top/pixelcnn.pt --repeats 3 --reference validate"

SETTINGS=(
    "--temperature 1.0 --top-k 32"
    "--temperature 1.0 --top-p 0.9"
    "--temperature 1.0 --top-temperature 0.8"
    "--temperature 0.9 --top-p 0.95"
)

for setting in "${SETTINGS[@]}"; do
    fid=$(sbatch --parsable --time=01:00:00 slurm_predict.sh $BASE $setting --fid)
    sbatch --dependency=afterok:$fid --time=00:30:00 slurm_predict.sh $BASE $setting
    echo "submitted: $setting (fid job $fid)"
done
