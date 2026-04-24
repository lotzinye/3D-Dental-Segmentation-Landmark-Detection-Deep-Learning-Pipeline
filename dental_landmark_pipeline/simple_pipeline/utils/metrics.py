"""
metrics.py
----------
Evaluation metrics for Stage-1 (segmentation) and Stage-2 (landmark).

Stage-1 metrics (per-epoch accumulation):
    SegMetrics.update(pred_labels, gt_labels)
    SegMetrics.compute() -> {"mIoU", "OA", "mACC"}

Stage-2 metrics (per-scan accumulation after post-processing):
    LandmarkMetrics.update(pred_lms, gt_lms)
    LandmarkMetrics.compute() -> {"MRE", "SDR@1.5", "SDR@2.0", "SDR@2.5", "SDR@4.0"}

Both classes have a .reset() method and can be used across epochs.
"""

from __future__ import annotations
from typing import Dict, List

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Stage-1 Segmentation Metrics
# ---------------------------------------------------------------------------

class SegMetrics:
    """
    Accumulates per-batch predictions and computes epoch-level segmentation
    metrics: Overall Accuracy (OA), mean class Accuracy (mACC), mean IoU (mIoU).

    Args:
        num_classes: number of classes (default 17; 0 = gingiva, 1-16 = teeth)
    """

    def __init__(self, num_classes: int = 17):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self._conf = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, pred: torch.Tensor, gt: torch.Tensor):
        """
        Args:
            pred: (B, N) long — predicted class indices
            gt:   (B, N) long — ground-truth class indices
        """
        pred = pred.cpu().numpy().ravel().astype(np.int64)
        gt   = gt.cpu().numpy().ravel().astype(np.int64)
        mask = (gt >= 0) & (gt < self.num_classes)
        pred, gt = pred[mask], gt[mask]
        np.add.at(self._conf, (gt, pred), 1)

    def compute(self) -> Dict[str, float]:
        conf = self._conf
        tp   = np.diag(conf)
        fn   = conf.sum(axis=1) - tp
        fp   = conf.sum(axis=0) - tp

        # Per-class IoU and accuracy (skip classes with no GT samples)
        denom_iou = tp + fp + fn
        denom_acc = tp + fn
        iou  = np.where(denom_iou > 0, tp / denom_iou, np.nan)
        acc  = np.where(denom_acc > 0, tp / denom_acc, np.nan)

        mIoU = float(np.nanmean(iou))
        mACC = float(np.nanmean(acc))
        OA   = float(tp.sum() / conf.sum()) if conf.sum() > 0 else 0.0

        return {"mIoU": mIoU, "OA": OA, "mACC": mACC}

    def __repr__(self) -> str:
        m = self.compute()
        return (
            f"SegMetrics  mIoU={m['mIoU']:.4f}  OA={m['OA']:.4f}  mACC={m['mACC']:.4f}"
        )


# ---------------------------------------------------------------------------
# Stage-2 Landmark Metrics
# ---------------------------------------------------------------------------

# SDR thresholds in millimetres (same as 3DTeethLand / MICCAI 2024 challenge)
_SDR_THRESHOLDS_MM = [1.5, 2.0, 2.5, 4.0]


class LandmarkMetrics:
    """
    Accumulates per-scan landmark errors and computes:
      • MRE  — Mean Radial Error (mm)
      • SDR@t — Success Detection Rate at threshold t mm
                (fraction of predicted landmarks within t mm of GT)

    Usage::

        metrics = LandmarkMetrics()
        for scan in test_set:
            pred_lms = ...   # list of {"class": str, "xyz": [x,y,z]} dicts
            gt_lms   = ...   # same structure, from _kpt.json
            metrics.update(pred_lms, gt_lms)
        results = metrics.compute()

    Args:
        thresholds_mm: SDR thresholds (default [1.5, 2.0, 2.5, 4.0])
    """

    def __init__(self, thresholds_mm: List[float] = None):
        self.thresholds = thresholds_mm or _SDR_THRESHOLDS_MM
        self.reset()

    def reset(self):
        self._errors: List[float] = []   # per matched landmark, mm
        self._total  = 0
        self._missed = 0

    def update(
        self,
        pred_lms: List[dict],
        gt_lms:   List[dict],
    ):
        """
        Match predicted landmarks to GT by class name (nearest match within class).

        Args:
            pred_lms: list of {"class": str, "xyz": [x, y, z], "fdi": int}
            gt_lms:   list of {"class": str, "xyz": [x, y, z], "fdi": int}
        """
        from collections import defaultdict

        # Group by (class, fdi)
        def _group(lms):
            d = defaultdict(list)
            for lm in lms:
                key = (lm["class"], lm.get("fdi", -1))
                d[key].append(np.array(lm["xyz"], dtype=np.float64))
            return d

        pred_g = _group(pred_lms)
        gt_g   = _group(gt_lms)

        for key, gt_pts in gt_g.items():
            self._total += len(gt_pts)
            if key not in pred_g:
                self._missed += len(gt_pts)
                continue
            pred_pts = pred_g[key]
            # Greedy nearest-match (O(N²), small N)
            for gpt in gt_pts:
                dists = [np.linalg.norm(gpt - p) for p in pred_pts]
                self._errors.append(min(dists))

    def compute(self) -> Dict[str, float]:
        if not self._errors:
            return {"MRE": float("nan"), **{f"SDR@{t}": 0.0 for t in self.thresholds}}

        errors = np.array(self._errors, dtype=np.float64)
        result: Dict[str, float] = {"MRE": float(errors.mean())}
        for t in self.thresholds:
            result[f"SDR@{t}"] = float((errors <= t).mean())

        det_rate = len(self._errors) / self._total if self._total > 0 else 0.0
        result["detection_rate"] = det_rate
        return result

    def __repr__(self) -> str:
        m = self.compute()
        sdr = "  ".join(f"SDR@{t}={m[f'SDR@{t}']:.3f}" for t in self.thresholds)
        return f"LandmarkMetrics  MRE={m['MRE']:.3f}mm  {sdr}"
