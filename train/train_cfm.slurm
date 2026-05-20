#!/bin/bash
#SBATCH --job-name=cfm-base
#SBATCH --output=/cluster/home/fusg/VT2/logs/cfm_base.out
#SBATCH --error=/cluster/home/fusg/VT2/logs/cfm_base.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --account=zav

set -euo pipefail

echo "=== Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $(hostname)"
echo "GPUs:   ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "Date:   $(date)"
echo "================"

# 1. Load CUDA + Python modules
module purge
module load cuda/13.0.2
module load python/3.11.9
module load uv/0.10.10

# 2. Use pre-created environment (no installs in job)
cd /cluster/home/fusg/VT2
if [[ ! -f .venv/bin/activate ]]; then
  echo "ERROR: existing env not found at /cluster/home/fusg/VT2/.venv" >&2
  echo "Refusing to create/install anything inside job." >&2
  exit 2
fi
source .venv/bin/activate

echo "Activated existing environment (.venv)"
echo "Using Python: $(which python)"

# 3. Quick GPU check
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('CUDA Device Count:', torch.cuda.device_count() if torch.cuda.is_available() else 0)"

mkdir -p logs models
# Keep allocator config unset here: `expandable_segments` is unsupported on this Torch build
unset PYTORCH_CUDA_ALLOC_CONF

# 4. Run training
python train/train_cfm.py \
    --epochs 200 \
    --batch-size 1024 \
    --lr 3e-4 \
    --weight-decay 1e-4 \
    --grad-clip 1.0 \
    --warmup-steps 2000 \
    --ema-decay 0.9995 \
    --patience 50 \
    --ckpt-path "${CKPT_PATH:-models/cfm_base.pt}" \
    --dataset-key "${DATASET_KEY:-87991419c1be4279}" \
    --aux-w 0.05 \
    --accum-steps 2 \
    --compile-mode none \
    --wandb-project BaseCFM \
    --wandb-name "${WANDB_NAME:-base_cfm_${SLURM_JOB_ID}}" \
    --seed 42

echo "=== Done: $(date) ==="
