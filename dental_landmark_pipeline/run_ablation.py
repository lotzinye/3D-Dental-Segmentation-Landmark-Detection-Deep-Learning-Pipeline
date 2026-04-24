"""
run_ablation.py
---------------
Inference-time parameter ablation for the TGNet + LandmarkNet pipeline.

This script re-runs the pipeline on a scan with varying post-processing
parameters and measures their effect on MRE and detection rate.  Since
all parameters are inference-time (not training-time), this is a valid
ablation that does not require re-training.

Parameters swept:
    dist_thresh      — landmark candidate threshold (mm)
    cluster_min_pts  — DBSCAN minimum cluster size
    cluster_max_dist — DBSCAN neighbourhood radius

Usage:
    python run_ablation.py \
        --obj  data/teeth3ds_sample/01F4JV8X/01F4JV8X_upper.obj \
        --gt   data/teeth3ds_sample/01F4JV8X/01F4JV8X_upper__kpt.json

    python run_ablation.py \
        --obj  data/teeth3ds_sample/01F4JV8X/QPYE7NOP_lower.obj \
        --gt   data/3DTeethLand_landmarks_train/lower/QPYE7NOP/QPYE7NOP_lower__kpt.json

Output: prints a LaTeX-ready table and saves CSV to results/ablation_results.csv

NOTE: Requires TGNet + 3DTeethLand CUDA extensions to be compiled.
See README.md and checkpoints/README.md for setup instructions.
"""

import argparse
import csv
import json
import sys
import tempfile
from pathlib import Path
from typing import List

import numpy as np

# ---- patch postprocess constants before importing pipeline ----
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))


def parse_args():
    p = argparse.ArgumentParser(description="Inference-time parameter ablation")
    p.add_argument("--obj", required=True, help="Input OBJ file")
    p.add_argument("--gt",  required=True, help="GT __kpt.json file")
    p.add_argument("--out", default="results/ablation_results.csv")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Evaluation helpers (same as evaluate_pipeline.py)
# ---------------------------------------------------------------------------

_GT_CLASSES = ["Mesial", "Distal", "Cusp", "InnerPoint", "OuterPoint", "FacialPoint"]
_PRED_TO_GT = {
    "MesialDistal": ["Mesial", "Distal"],
    **{c: [c] for c in _GT_CLASSES},
}
_SDR_THRESHOLDS = [1.5, 2.0, 2.5, 4.0]


def load_gt(path: Path) -> List[dict]:
    with open(path) as f:
        data = json.load(f)
    return [
        {"class": o["class"], "coord": np.array(o["coord"], dtype=np.float64)}
        for o in data.get("objects", [])
    ]


def load_pred(path: Path) -> List[dict]:
    with open(path) as f:
        data = json.load(f)
    lms = data.get("landmarks", data) if isinstance(data, dict) else data
    return [
        {"class": lm.get("class", ""), "coord": np.array(lm.get("coord", [0,0,0]), dtype=np.float64)}
        for lm in lms
    ]


def evaluate(pred_lms, gt_lms):
    from collections import defaultdict
    pred_by_cls = defaultdict(list)
    for lm in pred_lms:
        for c in _PRED_TO_GT.get(lm["class"], []):
            pred_by_cls[c].append(lm["coord"])

    errors, n_missed = [], 0
    for gt in gt_lms:
        cands = pred_by_cls.get(gt["class"], [])
        if not cands:
            n_missed += 1
            continue
        errors.append(min(np.linalg.norm(gt["coord"] - p) for p in cands))

    if not errors:
        return {"MRE": float("nan"), "detection_rate": 0.0,
                **{f"SDR@{t}": 0.0 for t in _SDR_THRESHOLDS}}
    e = np.array(errors)
    result = {
        "MRE": float(e.mean()),
        "detection_rate": len(errors) / len(gt_lms),
    }
    for t in _SDR_THRESHOLDS:
        result[f"SDR@{t}"] = float((e <= t).mean())
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    obj_path = Path(args.obj)
    gt_path  = Path(args.gt)
    out_csv  = Path(args.out)

    if not obj_path.exists():
        print(f"ERROR: OBJ not found: {obj_path}")
        sys.exit(1)
    if not gt_path.exists():
        print(f"ERROR: GT not found: {gt_path}")
        sys.exit(1)

    gt_lms = load_gt(gt_path)
    print(f"GT landmarks: {len(gt_lms)}")

    # Import pipeline (requires CUDA extensions)
    try:
        import pipeline.landmark_postprocess as lpp
        from pipeline.combined_pipeline import CombinedPipeline
    except ImportError as e:
        print(f"Pipeline import failed: {e}")
        print("Ensure CUDA extensions are compiled — see README.md")
        sys.exit(1)

    # ----------------------------------------------------------------
    # Parameter grid
    # ----------------------------------------------------------------
    dist_thresh_vals   = [0.08, 0.10, 0.12, 0.15, 0.20]
    cluster_min_pts_vals = [10, 15, 20, 30]
    cluster_max_dist_vals = [0.02, 0.03, 0.05]

    # Default = (0.12, 20, 0.03)
    results = []

    # Sweep dist_thresh (fix others at default)
    print("\n--- dist_thresh sweep ---")
    for dt in dist_thresh_vals:
        lpp.DIST_THRESH = dt
        lpp.CLUSTER_MIN_PTS = 20
        lpp.CLUSTER_MAX_DIST = 0.03

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        pipeline = CombinedPipeline()
        pipeline.run_and_save(str(obj_path), str(tmp_path))
        pred_lms = load_pred(tmp_path)
        tmp_path.unlink(missing_ok=True)

        m = evaluate(pred_lms, gt_lms)
        row = {"param": "dist_thresh", "value": dt, **m}
        results.append(row)
        print(f"  dist_thresh={dt:.2f}  MRE={m['MRE']:.3f}  SDR@2mm={m['SDR@2.0']:.3f}  det={m['detection_rate']:.3f}")

    # Sweep cluster_min_pts (fix dist_thresh at default)
    print("\n--- cluster_min_pts sweep ---")
    lpp.DIST_THRESH = 0.12
    lpp.CLUSTER_MAX_DIST = 0.03
    for cmp in cluster_min_pts_vals:
        lpp.CLUSTER_MIN_PTS = cmp

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        pipeline = CombinedPipeline()
        pipeline.run_and_save(str(obj_path), str(tmp_path))
        pred_lms = load_pred(tmp_path)
        tmp_path.unlink(missing_ok=True)

        m = evaluate(pred_lms, gt_lms)
        row = {"param": "cluster_min_pts", "value": cmp, **m}
        results.append(row)
        print(f"  cluster_min_pts={cmp}  MRE={m['MRE']:.3f}  SDR@2mm={m['SDR@2.0']:.3f}  det={m['detection_rate']:.3f}")

    # Sweep cluster_max_dist
    print("\n--- cluster_max_dist sweep ---")
    lpp.DIST_THRESH = 0.12
    lpp.CLUSTER_MIN_PTS = 20
    for cmd in cluster_max_dist_vals:
        lpp.CLUSTER_MAX_DIST = cmd

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        pipeline = CombinedPipeline()
        pipeline.run_and_save(str(obj_path), str(tmp_path))
        pred_lms = load_pred(tmp_path)
        tmp_path.unlink(missing_ok=True)

        m = evaluate(pred_lms, gt_lms)
        row = {"param": "cluster_max_dist", "value": cmd, **m}
        results.append(row)
        print(f"  cluster_max_dist={cmd:.2f}  MRE={m['MRE']:.3f}  SDR@2mm={m['SDR@2.0']:.3f}  det={m['detection_rate']:.3f}")

    # ----------------------------------------------------------------
    # Save CSV
    # ----------------------------------------------------------------
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["param", "value", "MRE", "detection_rate"] + [f"SDR@{t}" for t in _SDR_THRESHOLDS]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(results)
    print(f"\nSaved to {out_csv}")

    # ----------------------------------------------------------------
    # Print LaTeX table snippet
    # ----------------------------------------------------------------
    print("\n% LaTeX table — dist_thresh sweep:")
    print(r"\begin{tabular}{ccccc}")
    print(r"  dist\_thresh & MRE (mm) & SDR@1.5 & SDR@2.0 & Det. Rate \\")
    print(r"  \hline")
    for row in [r for r in results if r["param"] == "dist_thresh"]:
        default = " *" if abs(row["value"] - 0.12) < 1e-9 else ""
        print(f"  {row['value']:.2f}{default} & {row['MRE']:.3f} & "
              f"{row['SDR@1.5']:.3f} & {row['SDR@2.0']:.3f} & "
              f"{row['detection_rate']:.3f} \\\\")
    print(r"\end{tabular}")


if __name__ == "__main__":
    main()
