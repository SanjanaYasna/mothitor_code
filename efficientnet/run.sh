#!/bin/bash -l
#SBATCH --job-name=eff_total
#SBATCH --output=total.txt
#SBATCH -p gpu-preempt 
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --mem-per-gpu=30G
#SBATCH --time=14:00:00 

module load conda/latest
conda activate mothitor

cd /work/pi_mrobson_smith_edu/mothitor/main/efficientnet 
python train.py 