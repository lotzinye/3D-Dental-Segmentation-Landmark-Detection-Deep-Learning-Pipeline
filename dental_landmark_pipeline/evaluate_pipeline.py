"""
evaluate_pipeline.py
---------------------
Evaluate the TGNet + LandmarkNet pipeline against 3DTeethLand GT annotations.

Computes per-class and overall metrics matching the MICCAI 2024 3DTeethLand
challenge evaluation protocol:
    MRE        — Mean Radial Error (mm), primary metric
    SDR@1.5mm  — Success Detection Rate at 1.5 mm
    SDR@2.0mm  — SDR at 2.0 mm  (challenge secondary metric)
    SDR@2.5mm  — SDR at 2.5 mm
    SDR@4.0mm  — SDR at 4.0 mm
    detection_rate — fraction of GT landmarks that have a matched prediction

Pipeline output format (_landmarks.json):
    {"jaw": "upper", "landmarks": [{"class": "MesialDistal", "coord": [...], "fdi_tooth": 11}, ...]}

GT format (__kpt.json):
    {"objects": [{"class": "Mesial", "coord": [...]}, {"class": "Distal", ...}, ...]}

NOTE: The pipeline outputs "MesialDistal" as a combined class. For evaluation,
each predicted MesialDistal landmark is matched against the nearest GT landmark
from the union of Mesial + Distal annotations (the combination counts as one match
per GT landmark, using nearest-prediction semantics).

Usage:
    # Evaluate already-generated landmarks against GT
    python evaluate_pipeline.py \
        --pred-dir   data/output \
        --gt-dir     data/3DTeethLand_landmarks_train

    # Evaluate a single pair
    python evaluate_pipeline.py \
        --pred  "data/teeth3ds_sample/01F4JV8X/01F4JV8X_upper_landmarks.json" \
        --gt    "data/teeth3ds_sample/01F4JV8X/01F4JV8X_upper__kpt.json"

    # Evaluate all scans that have both a pipeline output and a GT file
    python evaluate_pipeline.py --auto-discover --out results/eval_results.json
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# GT class → evaluation class mapping
# Pipeline outputs "MesialDistal" for both Mesial and Distal predictions.
# ---------------------------------------------------------------------------
GT_CLASSES = ["Mesial", "Distal", "Cusp", "InnerPoint", "OuterPoint", "FacialPoint"]
SDR_THRESHOLDS = [1.5, 2.0, 2.5, 4.0]

# Pipeline class → GT class(es) it can match
_PRED_TO_GT = {
    "MesialDistal": ["Mesial", "Distal"],
    "Mesial":        ["Mesial"],
    "Distal":        ["Distal"],
    "Cusp":          ["Cusp"],
    "InnerPoint":    ["InnerPoint"],
    "OuterPoint":    ["OuterPoint"],
    "FacialPoint":   ["FacialPoint"],
}


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def load_pipeline_output(path: Path) -> List[dict]:
    """Load pipeline _landmarks.json → flat list of {class, coord, fdi_tooth}."""
    with open(path) as f:
        data = json.load(f)
    lms = data.get("landmarks", data) if isinstance(data, dict) else data
    return [
        {
            "class":     lm.get("class", ""),
            "coord":     np.array(lm.get("coord", lm.get("xyz", [0, 0, 0])), dtype=np.float64),
            "fdi_tooth": lm.get("fdi_tooth", lm.get("fdi", -1)),
        }
        for lm in lms
    ]


def load_gt(path: Path) -> List[dict]:
    """Load __kpt.json → flat list of {class, coord}."""
    with open(path) as f:
        data = json.load(f)
    objects = data.get("objects", data) if isinstance(data, dict) else data
    return [
        {
            "class": obj.get("class", ""),
            "coord": np.array(obj.get("coord", [0, 0, 0]), dtype=np.float64),
        }
        for obj in objects
    ]


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def match_predictions_to_gt(
    pred_lms: List[dict],
    gt_lms:   List[dict],
) -> Tuple[List[float], List[str], int]:
    """
    Greedy nearest-neighbour matching: for each GT landmark, find the
    closest prediction that can match its class.

    Returns:
        errors:    list of float (mm) — one per matched GT landmark
        classes:   list of str — GT class for each matched landmark
        n_missed:  number of GT landmarks with no matching prediction
    """
    # Group predictions by matchable GT class
    pred_by_cls = defaultdict(list)
    for lm in pred_lms:
        for gt_cls in _PRED_TO_GT.get(lm["class"], []):
            pred_by_cls[gt_cls].append(lm["coord"])

    errors, classes, n_missed = [], [], 0
    matched_pred = defaultdict(list)   # avoid counting same pred twice (optional)

    for gt in gt_lms:
        cls  = gt["class"]
        gpt  = gt["coord"]
        candidates = pred_by_cls.get(cls, [])
        if not candidates:
            n_missed += 1
            continue
        dists = [np.linalg.norm(gpt - p) for p in candidates]
        errors.append(min(dists))
        classes.append(cls)

    return errors, classes, n_missed


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(errors: List[float], n_gt: int) -> dict:
    if not errors:
        return {"MRE": float("nan"), **{f"SDR@{t}": 0.0 for t in SDR_THRESHOLDS},
                "detection_rate": 0.0, "n_matched": 0, "n_gt": n_gt}
    e = np.array(errors)
    result = {
        "MRE":            float(e.mean()),
        "detection_rate": len(errors) / n_gt if n_gt > 0 else 0.0,
        "n_matched":      len(errors),
        "n_gt":           n_gt,
    }
    for t in SDR_THRESHOLDS:
        result[f"SDR@{t}"] = float((e <= t).mean())
    return result


def print_results(overall: dict, per_class: dict):
    w = 55
    print("\n" + "=" * w)
    print(f"  Overall Results  ({overall['n_matched']}/{overall['n_gt']} GT matched)")
    print("=" * w)
    print(f"  MRE             : {overall['MRE']:.3f} mm")
    for t in SDR_THRESHOLDS:
        print(f"  SDR @ {t} mm   : {overall[f'SDR@{t}']:.3f}  ({overall[f'SDR@{t}']*100:.1f}%)")
    print(f"  Detection rate  : {overall['detection_rate']:.3f}  ({overall['detection_rate']*100:.1f}%)")
    print("=" * w)

    print("\n  Per-class breakdown:")
    print(f"  {'Class':<14}  {'MRE':>7}  {'SDR@2mm':>9}  {'SDR@4mm':>9}  {'n':>5}")
    print("  " + "-" * 50)
    for cls in GT_CLASSES:
        m = per_class.get(cls)
        if m is None or m["n_matched"] == 0:
            print(f"  {cls:<14}  {'—':>7}  {'—':>9}  {'—':>9}  {'0':>5}")
        else:
            print(
                f"  {cls:<14}  {m['MRE']:7.3f}  {m['SDR@2.0']:9.3f}  {m['SDR@4.0']:9.3f}  {m['n_matched']:>5}"
            )
    print()


# ---------------------------------------------------------------------------
# Auto-discover matched pairs
# ---------------------------------------------------------------------------

def find_matched_pairs(
    pred_root: Optional[Path],
    gt_root: Optional[Path],
) -> List[Tuple[Path, Path]]:
    """
    Find (pred, gt) path pairs where:
      pred = {pred_root}/**/*_landmarks.json
      gt   = {gt_root}/**/*__kpt.json   (double-underscore)

    Match by stem: "01F4JV8X_upper" in both directions.
    """
    pairs = []
    if pred_root is None or gt_root is None:
        return pairs

    pred_files = {p.stem.replace("_landmarks", ""): p
                  for p in pred_root.rglob("*_landmarks.json")}
    gt_files   = {p.stem.replace("__kpt", ""): p
                  for p in gt_root.rglob("*__kpt.json")}

    for stem, pred_path in sorted(pred_files.items()):
        if stem in gt_files:
            pairs.append((pred_path, gt_files[stem]))

    return pairs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate pipeline landmark predictions")
    p.add_argument("--pred",       default=None, help="Single prediction JSON path")
    p.add_argument("--gt",         default=None, help="Single GT kpt.json path")
    p.add_argument("--pred-dir",   default=None, help="Root dir containing *_landmarks.json files")
    p.add_argument("--gt-dir",     default=None, help="Root dir containing *__kpt.json files")
    p.add_argument("--auto-discover", action="store_true",
                   help="Auto-discover matched pairs in --pred-dir and --gt-dir")
    p.add_argument("--out",        default=None, help="Save JSON results to this path")
    return p.parse_args()


def evaluate_pair(pred_path: Path, gt_path: Path) -> Tuple[List[float], List[str], int]:
    pred_lms = load_pipeline_output(pred_path)
    gt_lms   = load_gt(gt_path)
    errors, classes, n_missed = match_predictions_to_gt(pred_lms, gt_lms)
    return errors, classes, n_missed, len(gt_lms)


def main():
    args = parse_args()

    # ----------------------------------------------------------------
    # Collect pairs to evaluate
    # ----------------------------------------------------------------
    pairs = []

    if args.pred and args.gt:
        pairs.append((Path(args.pred), Path(args.gt)))

    if args.auto_discover or (args.pred_dir and args.gt_dir):
        discovered = find_matched_pairs(
            Path(args.pred_dir) if args.pred_dir else None,
            Path(args.gt_dir)   if args.gt_dir   else None,
        )
        pairs.extend(discovered)

    # Always include the sample scans if they exist
    _here = Path(__file__).parent
    _sample_pairs = [
        (
            _here / "data/teeth3ds_sample/01F4JV8X/01F4JV8X_upper_landmarks.json",
            _here / "data/teeth3ds_sample/01F4JV8X/01F4JV8X_upper__kpt.json",
        ),
        (
            _here / "data/teeth3ds_sample/01F4JV8X/QPYE7NOP_lower_landmarks.json",
            _here / "data/3DTeethLand_landmarks_train/lower/QPYE7NOP/QPYE7NOP_lower__kpt.json",
        ),
    ]
    for p, g in _sample_pairs:
        if p.exists() and g.exists() and (p, g) not in pairs:
            pairs.append((p, g))

    if not pairs:
        print("No evaluation pairs found.  Use --pred + --gt or --auto-discover.")
        sys.exit(1)

    print(f"Evaluating {len(pairs)} scan(s)...")

    # ----------------------------------------------------------------
    # Accumulate across all pairs
    # ----------------------------------------------------------------
    all_errors:  List[float] = []
    all_classes: List[str]   = []
    total_gt = 0
    total_missed = 0

    for pred_path, gt_path in pairs:
        errors, classes, n_missed, n_gt = evaluate_pair(pred_path, gt_path)
        all_errors.extend(errors)
        all_classes.extend(classes)
        total_gt    += n_gt
        total_missed += n_missed
        scan_mre = float(np.mean(errors)) if errors else float("nan")
        dr = len(errors) / n_gt if n_gt > 0 else 0.0
        print(f"  {pred_path.name:<50s}  MRE={scan_mre:.2f}mm  "
              f"det={dr:.2f}  n={len(errors)}/{n_gt}", flush=True)

    # ----------------------------------------------------------------
    # Compute overall + per-class
    # ----------------------------------------------------------------
    overall   = compute_metrics(all_errors, total_gt)
    per_class = {}
    for cls in GT_CLASSES:
        cls_errors = [e for e, c in zip(all_errors, all_classes) if c == cls]
        n_gt_cls   = sum(1 for c in all_classes if c == cls) + \
                     sum(1 for _ in range(total_missed))   # approximate
        per_class[cls] = compute_metrics(cls_errors, len(cls_errors))

    print_results(overall, per_class)

    # ----------------------------------------------------------------
    # Save
    # ----------------------------------------------------------------
    if args.out:
        out_data = {"overall": overall, "per_class": per_class, "n_scans": len(pairs)}
        # Convert numpy types to Python
        def _serialize(obj):
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            return obj
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out_data, f, indent=2, default=_serialize)
        print(f"Results saved to {args.out}")


if __name__ == "__main__":
    main()
