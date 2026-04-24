#!/bin/bash
# ============================================================================
# TC1 — Stage-1 segmentation evaluation on test set
#
# Uses FPS sampling (quality mode) and npoints=10000 for best accuracy.
# Estimated run time: ~15–25 min on V100 for 360 test scans.
#
# Usage:
#   sbatch slurm/eval_seg.sh
# ============================================================================

#SBATCH --job-name=eval_seg
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --time=60
#SBATCH --output=logs/eval_seg_%j.out
#SBATCH --error=logs/eval_seg_%j.err

# ---- User configuration -------------------------------------------------
CONDA_ENV="teeth_env"
DATA_ROOT="/tc1home/FYP/lotz0001/data/teeth3ds"
# -------------------------------------------------------------------------

set -e
mkdir -p logs results

echo "Job ID: $SLURM_JOB_ID  Node: $(hostname)  Start: $(date)"

module load cuda/12.9
module load anaconda
source activate "$CONDA_ENV"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

python evaluate_seg.py \
    --data-root   "$DATA_ROOT"             \
    --split-file  splits/test.txt          \
    --ckpt        checkpoints/seg/best.pt  \
    --batch-size  8                        \
    --npoints     10000                    \
    --num-workers 4                        \
    --out         results/seg_eval.json

echo "Done: $(date)"
