#!/bin/bash
#SBATCH --job-name=mae_pre
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00


module purge
module load cuda
source venv/bin/activate
cd ~/galaxy_mae_project

python train_mae.py --epochs 300 --batch_size 64 --exp_name debug_run