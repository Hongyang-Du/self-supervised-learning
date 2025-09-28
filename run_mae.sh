#!/bin/bash
#SBATCH --job-name=mae_pre
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=runs/mae_pretrain/slurm_%j.out

module purge
module load cuda
source ~/venvs/mae_env/bin/activate
cd ~/galaxy_mae_project

python train_mae.py \
  --exp_name mae-b16-mr75-e100 \
  --epochs 100 --batch_size 64 --lr 1.5e-4 --wd 0.05 --mask_ratio 0.75 --amp