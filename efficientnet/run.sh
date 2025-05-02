#!/bin/bash -l
#SBATCH --job-name=eff_total
#SBATCH --output=total.txt
#SBATCH -p gpu-preempt 
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --mem-per-gpu=20G
#SBATCH --time=10:00:00 

module load conda/latest
conda activate mothitor

cd /work/pi_mrobson_smith_edu/mothitor/code_main/efficientnet/
python predict_from_bounding_box.py