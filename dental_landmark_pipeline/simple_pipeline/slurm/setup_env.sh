#!/bin/bash
# ============================================================================
# TC1 — Conda environment setup for the lean dental landmark pipeline
#
# Run this ONCE on the TC1 head node (not as a SLURM job — it only installs
# packages, no GPU needed).
#
# Usage:
#   bash slurm/setup_env.sh
#
# After this script completes:
#   1. Generate splits:  python splits/make_splits.py --data-root <DATA_ROOT>
#   2. Submit training:
#        sbatch slurm/train_seg.sh
#        sbatch slurm/train_landmark.sh   ← submit both simultaneously
# ============================================================================

set -e

ENV_NAME="teeth_env"
PYTHON_VERSION="3.10"

echo "====================================================="
echo "Setting up conda environment: $ENV_NAME"
echo "====================================================="

module load anaconda

# Create environment
conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y

# Activate
source activate "$ENV_NAME"

# PyTorch 2.0 with CUDA 12.x  (matches TC1 cuda/12.9 module)
conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 \
    -c pytorch -c nvidia

# Scientific stack
conda install -y numpy scipy scikit-learn -c conda-forge

# Verify
python - <<'EOF'
import torch
print(f"PyTorch : {torch.__version__}")
print(f"CUDA    : {torch.version.cuda}")
print(f"cuDNN   : {torch.backends.cudnn.version()}")
# torch.compile requires PyTorch >= 2.0 — check
major = int(torch.__version__.split(".")[0])
print(f"torch.compile available: {major >= 2}")
EOF

echo "====================================================="
echo "Environment '$ENV_NAME' ready."
echo "====================================================="
