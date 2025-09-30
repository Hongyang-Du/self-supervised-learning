#!/bin/bash
#SBATCH --job-name=linear_probe_sweep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=linear_probe_logs/slurm_%j.out

module purge
module load cuda
source venv/bin/activate
cd /users/hdu15/data/galaxy_mae  


ratios=("mask_0.5" "mask_0.75")

for ratio in "${ratios[@]}"; do
  ckpt="runs/mae_pretrain/${ratio}/ckpts/best_encoder.pth"
  echo "🔍 Running linear probing with $ckpt"

  python train_linear_probe.py \
    --ckpt "$ckpt" \
    --epochs 50 \
    --batch_size 64 \
    --lr 1e-3 \
    --wd 0.05

  # 保存结果，不覆盖
  mv linear_probe_logs/train_log.csv linear_probe_logs/${ratio}_train_log.csv
  mv linear_probe_best.pth linear_probe_logs/${ratio}_best.pth
done