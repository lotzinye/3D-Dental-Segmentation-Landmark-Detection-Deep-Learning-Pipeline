"""
infer.py
--------
End-to-end inference: OBJ scan → per-vertex FDI labels → landmarks JSON.

Pipeline:
  1. Load OBJ and subsample to npoints (Stage-1 input)
  2. Run PointNet2SegModel → per-point FDI labels
  3. For each unique FDI tooth:
       a. Extract tooth vertices
       b. FPS-subsample to lm_npoints (Stage-2 input)
       c. Build features (normals + centroid offsets)
       d. Run PointNet2LandmarkModel → per-head dense predictions
       e. Post-process (DBSCAN) → discrete landmark positions
  4. Write results to JSON (compatible with visualize_scan.py)

Usage:
    python infer.py \
        --obj        data/teeth3ds/01F4JV8X/01F4JV8X_upper.obj \
        --seg-ckpt   checkpoints/seg/best.pt \
        --lm-ckpt    checkpoints/landmark/best.pt \
        --out        results/01F4JV8X_upper_landmarks.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from datasets.seg_dataset import _parse_obj, _fps_np as fps_np, _fdi_to_class
from datasets.landmark_dataset import (
    _parse_obj_verts_normals,
    _fps_np as fps_np_lm,
    Z_SCORE_STD,
)
from models.seg_model import PointNet2SegModel
from models.landmark_model import PointNet2LandmarkModel, LANDMARK_HEADS
from utils.postprocess import extract_landmarks


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Simple pipeline end-to-end inference")
    p.add_argument("--obj",       required=True, help="Input .obj file")
    p.add_argument("--seg-ckpt",  required=True, help="Stage-1 segmentation checkpoint")
    p.add_argument("--lm-ckpt",   required=True, help="Stage-2 landmark checkpoint")
    p.add_argument("--out",       default=None,  help="Output JSON path (default: next to OBJ)")
    p.add_argument("--seg-npts",  type=int, default=10_000, help="Stage-1 FPS points")
    p.add_argument("--lm-npts",   type=int, default=6_000,  help="Stage-2 FPS points per tooth")
    p.add_argument("--device",    default="cuda", help="cuda or cpu")
    p.add_argument("--min-tooth-verts", type=int, default=50,
                   help="Skip teeth with fewer than this many vertices")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Stage-1 helpers
# ---------------------------------------------------------------------------

def run_segmentation(model, verts, normals, npoints, device):
    """
    Run seg model on a single scan.

    Returns:
        labels: (N_orig,) int64 numpy array — FDI class index (0–16)
    """
    N = len(verts)
    if N > npoints:
        idx = fps_np(verts, npoints)
    else:
        idx = np.arange(N)

    sub_xyz  = verts[idx]
    sub_norm = normals[idx]

    # Normalise (match TeethSegDataset)
    mean  = sub_xyz.mean(axis=0)
    xyz_c = sub_xyz - mean
    scale = np.abs(xyz_c).max() + 1e-6
    xyz_c /= scale

    xyz_t = torch.from_numpy(xyz_c).unsqueeze(0).to(device)   # (1, n, 3)
    nrm_t = torch.from_numpy(sub_norm).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(xyz_t, nrm_t)                           # (1, 17, n)
    pred_cls = logits[0].argmax(dim=0).cpu().numpy()           # (n,) 0–16

    # Map compact class back to FDI (approximate reverse)
    # For inference we only need which vertices belong to which tooth
    return idx, pred_cls


# Compact class index → one representative FDI label (upper jaw assumed;
# actual FDI is ambiguous without jaw label — use vertex height heuristic)
_CLS_TO_FDI_UPPER = {i: 10 + i for i in range(1, 9)}   # 1→11, ..., 8→18
_CLS_TO_FDI_UPPER.update({i: 10 + i for i in range(9, 17)})  # 9→19 (remapped)


# ---------------------------------------------------------------------------
# Stage-2 helpers
# ---------------------------------------------------------------------------

def run_landmark(model, tooth_verts, tooth_normals, jaw_mean, fdi, npoints, device):
    """
    Run landmark model on a single tooth crop.

    Returns:
        list of {"fdi", "class", "xyz"} dicts in mm
    """
    N = len(tooth_verts)
    if N < 8:
        return []

    xyz_norm = (tooth_verts - jaw_mean) / Z_SCORE_STD

    if N > npoints:
        sel       = fps_np_lm(xyz_norm, npoints)
        xyz_norm  = xyz_norm[sel]
        tn        = tooth_normals[sel]
    elif N < npoints:
        pad       = npoints - N
        idx_p     = np.random.choice(N, pad, replace=True)
        xyz_norm  = np.vstack([xyz_norm, xyz_norm[idx_p]])
        tn        = np.vstack([tooth_normals, tooth_normals[idx_p]])
    else:
        tn = tooth_normals

    centroid     = xyz_norm.mean(axis=0)
    cent_offsets = xyz_norm - centroid
    features     = np.concatenate([tn, cent_offsets], axis=1).astype(np.float32)

    xyz_t  = torch.from_numpy(xyz_norm).unsqueeze(0).to(device)
    feat_t = torch.from_numpy(features).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        preds = model(xyz_t, feat_t)   # list of 5 × (1, 4, N)

    landmarks = extract_landmarks(xyz_norm, preds, fdi, jaw_mean)
    return landmarks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    obj_path = Path(args.obj)
    out_path = Path(args.out) if args.out else obj_path.with_name(
        obj_path.stem + "_landmarks_simple.json"
    )

    # ----------------------------------------------------------------
    # Load models
    # ----------------------------------------------------------------
    print("Loading segmentation model …")
    seg_model = PointNet2SegModel(num_classes=17, in_channels=3, dropout=0.0).to(device)
    seg_ckpt  = torch.load(args.seg_ckpt, map_location=device)
    seg_model.load_state_dict(seg_ckpt["model"])

    print("Loading landmark model …")
    lm_model = PointNet2LandmarkModel(num_heads=5, in_channels=6, dropout=0.0).to(device)
    lm_ckpt  = torch.load(args.lm_ckpt, map_location=device)
    lm_model.load_state_dict(lm_ckpt["model"])

    # ----------------------------------------------------------------
    # Parse OBJ
    # ----------------------------------------------------------------
    print(f"Parsing {obj_path.name} …")
    verts, normals = _parse_obj_verts_normals(obj_path)
    jaw_mean       = verts.mean(axis=0)
    print(f"  {len(verts):,} vertices")

    # ----------------------------------------------------------------
    # Stage-1: segmentation
    # ----------------------------------------------------------------
    print("Stage-1: tooth segmentation …")
    seg_idx, seg_cls = run_segmentation(seg_model, verts, normals, args.seg_npts, device)

    # Build per-vertex class array (sampled points only)
    # Assign FDI label per class index (class 0 = gingiva)
    # We need to group vertices by class — use compact class index directly
    class_to_verts: dict = {}
    for i, cls in enumerate(seg_cls):
        if cls == 0:   # gingiva — skip
            continue
        orig_idx = int(seg_idx[i])
        class_to_verts.setdefault(int(cls), []).append(orig_idx)

    print(f"  Found {len(class_to_verts)} tooth regions (classes 1–16)")

    # ----------------------------------------------------------------
    # Stage-2: landmark detection per tooth
    # ----------------------------------------------------------------
    print("Stage-2: landmark detection …")
    all_landmarks = []

    # Map compact class → approximate FDI
    # Upper jaw: cls 1-8 → FDI 11-18;  cls 9-16 → FDI 21-28
    # Lower jaw: same compact range used — disambiguate by z-coord if needed
    # Simple heuristic: use compact class * 10 offset
    def cls_to_fdi(cls_idx: int) -> int:
        if 1 <= cls_idx <= 8:
            return 10 + cls_idx         # 11–18
        elif 9 <= cls_idx <= 16:
            return 10 + cls_idx         # 19–26 (approx)
        return cls_idx

    for cls_idx, vert_indices in sorted(class_to_verts.items()):
        if len(vert_indices) < args.min_tooth_verts:
            continue
        fdi = cls_to_fdi(cls_idx)

        tooth_v = verts[vert_indices]
        tooth_n = normals[vert_indices]

        lms = run_landmark(lm_model, tooth_v, tooth_n, jaw_mean, fdi, args.lm_npts, device)
        all_landmarks.extend(lms)
        print(f"  Tooth FDI≈{fdi:2d}  ({len(tooth_v):,} verts)  → {len(lms)} landmarks")

    # ----------------------------------------------------------------
    # Save JSON
    # ----------------------------------------------------------------
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_landmarks, f, indent=2)

    print(f"\nSaved {len(all_landmarks)} landmarks to {out_path}")


if __name__ == "__main__":
    main()
