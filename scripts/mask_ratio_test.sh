#!/bin/bash
#SBATCH --job-name=mae_mask_sweep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-3
#SBATCH --output=runs/mask_sweep/slurm_%A_%a.out

module purge
module load cuda
source venv/bin/activate
cd /oscar/data/csun45/galaxy_mae   # 改成你项目的真实路径

# sweep 的 mask_ratio 列表
ratios=(0.0 0.25 0.5 0.75)
ratio=${ratios[$SLURM_ARRAY_TASK_ID]}

# 每个实验的 exp_name 会自动区分 log 文件夹
exp_name="mask_${ratio}"

echo "==== Running with mask_ratio=$ratio ===="
python train_mae.py \
  --epochs 100 \
  --batch_size 64 \
  --mask_ratio $ratio \
  --exp_name $exp_name