#!/bin/bash
#SBATCH --job-name=mae_debug
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=runs/debug/slurm_%j.out

module purge
module load cuda
source venv/bin/activate
cd /oscar/data/csun45/galaxy_mae   # 改成你项目所在目录

# 只跑很少的 epoch，方便调试
python train_mae.py \
  --epochs 1 \
  --batch_size 16 \
  --exp_name debug \
  --mask_ratio 0.75