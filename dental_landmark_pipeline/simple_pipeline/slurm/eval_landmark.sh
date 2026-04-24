#!/bin/bash
# ============================================================================
# TC1 — Stage-2 landmark evaluation on test set
#
# Estimated run time: ~30–45 min on V100 for 360 test scans.
#
# Usage:
#   sbatch slurm/eval_landmark.sh
# ============================================================================

#SBATCH --job-name=eval_lm
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --time=120
#SBATCH --output=logs/eval_landmark_%j.out
#SBATCH --error=logs/eval_landmark_%j.err

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

python evaluate_landmark.py \
    --data-root   "$DATA_ROOT"                    \
    --split-file  splits/test.txt                 \
    --ckpt        checkpoints/landmark/best.pt    \
    --npoints     4000                            \
    --out         results/landmark_eval.json

echo "Done: $(date)"
