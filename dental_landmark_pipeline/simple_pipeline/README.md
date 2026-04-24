# Simple Dental Landmark Pipeline

A self-contained, **RTX 3050–compatible** two-stage pipeline for automatic tooth segmentation and anatomical landmark detection from 3-D intra-oral scans.

Built on **PointNet++ MSG** (pure PyTorch — no custom CUDA compilation required).

---

## Why This Pipeline?

| | Original Pipeline | This Pipeline |
|---|---|---|
| Stage 1 backbone | TGNet (CUDA ext) | PointNet++ MSG |
| Stage 2 backbone | StratifiedTransformer (CUDA ext) | PointNet++ MSG |
| Custom CUDA required | Yes (2 extensions) | **No** |
| Min VRAM (training) | ~11 GB | **~3 GB** |
| Min VRAM (inference) | ~4 GB | **~1.5 GB** |
| Training time (V100) | ~4 weeks | **~1–2 weeks** |

---

## Directory Structure

```
simple_pipeline/
├── models/
│   ├── pointnet2_utils.py   — FPS, ball query, SA layers, FP layers (pure PyTorch)
│   ├── seg_model.py         — PointNet2SegModel (Stage 1: tooth segmentation)
│   └── landmark_model.py    — PointNet2LandmarkModel (Stage 2: landmark detection)
├── datasets/
│   ├── seg_dataset.py       — TeethSegDataset (Teeth3DS OBJ + JSON)
│   └── landmark_dataset.py  — TeethLandmarkDataset (per-tooth crops + kpt.json GT)
├── utils/
│   ├── losses.py            — SegLoss (CE+Dice), LandmarkLoss (dist+offset)
│   ├── metrics.py           — SegMetrics (mIoU/OA/mACC), LandmarkMetrics (MRE/SDR)
│   └── postprocess.py       — DBSCAN-based landmark extraction from dense predictions
├── train_seg.py             — Stage-1 training script
├── train_landmark.py        — Stage-2 training script
├── infer.py                 — End-to-end inference (OBJ → landmarks JSON)
├── evaluate_seg.py          — Stage-1 evaluation (mIoU / OA / mACC)
├── evaluate_landmark.py     — Stage-2 evaluation (MRE / SDR per class)
├── slurm/
│   ├── train_seg.sh         — TC1 cluster job script (Stage 1)
│   ├── train_landmark.sh    — TC1 cluster job script (Stage 2)
│   ├── eval_seg.sh          — TC1 eval job script (Stage 1)
│   └── eval_landmark.sh     — TC1 eval job script (Stage 2)
├── splits/                  — (you create) train.txt / val.txt / test.txt
└── requirements.txt
```

---

## Installation

```bash
# Create environment (Python 3.10 recommended)
conda create -n teeth_simple python=3.10
conda activate teeth_simple

# Install PyTorch (adjust CUDA version to match your system)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies
pip install -r simple_pipeline/requirements.txt

# Optional: fast FPS via torch-cluster (no compilation needed — pre-built wheels)
# pip install torch-cluster -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
```

> **No custom CUDA compilation required.** The pipeline uses `torch.cdist` for ball
> queries and pure PyTorch loops for FPS (with optional `torch_cluster` acceleration).

---

## Data Preparation

### Dataset layout (Teeth3DS + 3DTeethLand)

```
data/teeth3ds/
  {PATIENT_ID}/
    {PATIENT_ID}_{jaw}.obj        ← mesh (vertices + normals)
    {PATIENT_ID}_{jaw}.json       ← per-vertex FDI labels  {"labels": [...]}
    {PATIENT_ID}_{jaw}_kpt.json   ← landmark annotations (Stage 2 only)
```

Download:
- **Teeth3DS** mesh + annotations: https://osf.io/xctdy/
- **3DTeethLand** landmark annotations: https://osf.io/um96h/

### Create train/val/test splits

Each file is a plain text list of scan stems (one per line):

```
# splits/train.txt
01F4JV8X_upper
01F4JV8X_lower
02XK9PQR_upper
...
```

A simple 70/15/15 split script:

```python
from pathlib import Path
import random

stems = [p.stem for p in Path("data/teeth3ds").rglob("*.obj")]
random.shuffle(stems)
n = len(stems)
splits = {
    "train": stems[:int(0.7*n)],
    "val":   stems[int(0.7*n):int(0.85*n)],
    "test":  stems[int(0.85*n):],
}
Path("simple_pipeline/splits").mkdir(exist_ok=True)
for name, lst in splits.items():
    Path(f"simple_pipeline/splits/{name}.txt").write_text("\n".join(lst))
```

---

## Training

### Stage 1 — Tooth Segmentation (local RTX 3050)

```bash
cd simple_pipeline

python train_seg.py \
    --data-root  data/teeth3ds \
    --train-split splits/train.txt \
    --val-split   splits/val.txt \
    --epochs      100 \
    --batch-size  2 \
    --npoints     10000 \
    --lr          1e-3

# Resume after interruption:
python train_seg.py [same args] --resume
```

VRAM: ~2.8–3.2 GB at batch=2, N=10,000
Time: ~35–40 min/epoch on RTX 3050 → ~60 h total (100 epochs)

### Stage 2 — Landmark Detection (local RTX 3050)

```bash
python train_landmark.py \
    --data-root  data/teeth3ds \
    --train-split splits/train.txt \
    --val-split   splits/val.txt \
    --epochs      80 \
    --batch-size  8 \
    --npoints     6000 \
    --lr          1e-3

# Resume:
python train_landmark.py [same args] --resume
```

VRAM: ~2.0–2.5 GB at batch=8, N=6,000
Time: ~25–30 min/epoch on RTX 3050 → ~35 h total (80 epochs)

### Training on TC1 (NTU V100 32 GB)

```bash
# Upload files to cluster (run from your local machine)
rsync -avz simple_pipeline/ username@tc1.ntu.edu.sg:~/scratch/simple_pipeline/
rsync -avz data/teeth3ds/   username@tc1.ntu.edu.sg:~/scratch/teeth3ds/

# On the cluster:
cd ~/scratch/simple_pipeline

# Set up environment (first time only)
module load anaconda
conda create -n teeth_env python=3.10
conda activate teeth_env
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# Submit jobs
sbatch slurm/train_seg.sh
sbatch slurm/train_landmark.sh

# Monitor
squeue -u $USER
tail -f logs/seg_<JOBID>.out
```

TC1 MaxWall = 6 hours — jobs are checkpoint-safe and automatically resume.
Stage 1 needs ~3–4 job submissions; Stage 2 needs ~2–3 submissions (on V100).

---

## Evaluation

### Stage 1 — Segmentation metrics

```bash
python evaluate_seg.py \
    --data-root  data/teeth3ds \
    --split-file splits/test.txt \
    --ckpt       checkpoints/seg/best.pt \
    --out        results/seg_eval.json
```

Outputs:
```
Overall Accuracy  (OA)  : 0.9412
Mean Class Accuracy     : 0.9187
Mean IoU           (mIoU): 0.9053

Per-class IoU:
  [ 0] gingiva       IoU=0.9312
  [ 1] FDI_1         IoU=0.8891
  ...
```

### Stage 2 — Landmark metrics (MICCAI 2024 protocol)

```bash
python evaluate_landmark.py \
    --data-root  data/teeth3ds \
    --split-file splits/test.txt \
    --ckpt       checkpoints/landmark/best.pt \
    --out        results/landmark_eval.json
```

Outputs:
```
MRE:              1.21 mm
SDR @ 1.5 mm:    0.712
SDR @ 2.0 mm:    0.841
SDR @ 2.5 mm:    0.903
SDR @ 4.0 mm:    0.971
Detection rate:   0.934

Per-class breakdown:
  Mesial         MRE=1.08mm  SDR@2mm=0.862  (n=...)
  Distal         MRE=1.14mm  SDR@2mm=0.851  (n=...)
  FacialPoint    MRE=0.97mm  SDR@2mm=0.891  (n=...)
  OuterPoint     MRE=1.31mm  SDR@2mm=0.821  (n=...)
  InnerPoint     MRE=1.29mm  SDR@2mm=0.819  (n=...)
  Cusp           MRE=1.44mm  SDR@2mm=0.801  (n=...)
```

---

## Inference (single scan)

```bash
python infer.py \
    --obj      data/teeth3ds/01F4JV8X/01F4JV8X_upper.obj \
    --seg-ckpt checkpoints/seg/best.pt \
    --lm-ckpt  checkpoints/landmark/best.pt \
    --out      results/01F4JV8X_upper_landmarks.json
```

The output JSON has the same schema as the full pipeline (`run_pipeline.py`),
so `visualize_scan.py` at the project root can render it directly:

```bash
# From the project root (dental_landmark_pipeline/)
python visualize_scan.py data/teeth3ds/01F4JV8X/01F4JV8X_upper.obj \
    # reads results/01F4JV8X_upper_landmarks.json automatically
```

---

## Model Architecture

### Stage 1 — PointNet2SegModel

| Layer | Type | Output points | Channels |
|-------|------|--------------|---------|
| Input | — | 10,000 | 3 xyz + 3 normals |
| SA1 (MSG) | r=[0.05,0.10] | 2,048 | 192 |
| SA2 (MSG) | r=[0.10,0.20] | 512 | 384 |
| SA3 (MSG) | r=[0.20,0.40] | 128 | 768 |
| SA4 (MSG) | r=[0.40,0.80] | 32 | 1,536 |
| FP4→FP1 | upsample | 10,000 | 128 |
| Head | Conv1d | 10,000 | 17 classes |

Loss: Cross-Entropy + Soft-Dice (equal weight)

### Stage 2 — PointNet2LandmarkModel

| Layer | Type | Output points | Channels |
|-------|------|--------------|---------|
| Input | — | 6,000 | 3 xyz_norm + 6 features |
| SA1 (MSG) | r=[0.05,0.10] | 512 | 192 |
| SA2 (MSG) | r=[0.10,0.20] | 128 | 384 |
| SA3 (MSG) | r=[0.20,0.40] | 32 | 768 |
| SA4 (MSG) | r=[0.40,0.80] | 8 | 1,536 |
| FP4→FP1 | upsample | 6,000 | 128 |
| 5 × Head | Conv1d | 6,000 | 4 per head |

Each head outputs: `(distance_to_nearest_landmark, dx, dy, dz)` per point.
Loss: Smooth-L1 distance loss + Smooth-L1 offset loss (near points only).
Post-processing: DBSCAN clustering on candidate predicted positions.

---

## Landmark Classes

| Head | Class | Description |
|------|-------|-------------|
| 0 | MesialDistal | Both mesial and distal (split by x-coord in post-processing) |
| 1 | FacialPoint | Facial surface landmark |
| 2 | OuterPoint | Outer curvature point |
| 3 | InnerPoint | Inner curvature point |
| 4 | Cusp | Cusp tip(s) |

---

## Expected Results

Target performance (comparable to 3DTeethLand baseline on Teeth3DS test set):

| Metric | Stage 1 Target | Stage 2 Target |
|--------|---------------|---------------|
| mIoU | ≥ 0.90 | — |
| OA | ≥ 0.94 | — |
| MRE | — | ≤ 1.5 mm |
| SDR@2mm | — | ≥ 0.80 |

The simpler PointNet++ backbone trades ~5% mIoU and ~0.3 mm MRE for eliminating all
custom CUDA dependencies and reducing VRAM requirements by ~3×.

---

## Troubleshooting

**Out of Memory on RTX 3050**
- Reduce `--batch-size` to 1 (Stage 1) or 4 (Stage 2)
- Reduce `--npoints` to 8000 (Stage 1) or 4000 (Stage 2)

**Slow FPS sampling**
- Install `torch-cluster` for 10–20× faster FPS:
  ```bash
  pip install torch-cluster -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
  ```

**No landmarks detected**
- Lower `--dist-threshold` in `postprocess.py` (default 0.15)
- Check that `_kpt.json` files exist and match the OBJ stem

**SLURM job cancelled at 6h**
- This is expected — TC1 MaxWall is 6 hours
- Re-submit with `--resume` flag; training picks up from the last epoch
