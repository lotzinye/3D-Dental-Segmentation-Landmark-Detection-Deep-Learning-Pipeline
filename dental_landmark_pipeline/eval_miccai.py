#!/usr/bin/env python3
"""
eval_miccai.py
--------------
Evaluate the combined pipeline against both challenge metrics:

  MICCAI 2022  Teeth Segmentation Challenge:
      TLA  — Teeth Localisation Accuracy    (exp-normalised; higher = better)
      TSA  — Teeth Segmentation Accuracy    (micro-F1; higher = better)
      TIR  — Teeth Identification Rate      (%; higher = better)
      Score = (mean_TLA + mean_TSA + mean_TIR) / 3

  MICCAI 2024  3DTeethLand Landmark Challenge:
      mAP  — mean Average Precision across 6 classes
      mAR  — mean Average Recall  across 6 classes
      AP_cusp, AR_cusp — Cusp-only scores
      AP / AR per 4 clinical categories

Data requirements (240 scans satisfy all three):
    data/{upper,lower}/{PID}/{PID}_{jaw}.obj        — raw mesh
    data/{upper,lower}/{PID}/{PID}_{jaw}.json       — GT seg (labels + instances)
    data/3DTeethLand_landmarks_train/{jaw}/{PID}/   — GT landmark __kpt.json

Usage:
    python eval_miccai.py                          # 10 scans, quick smoke-test
    python eval_miccai.py --max-scans 50           # 50 scans
    python eval_miccai.py --max-scans 0            # all 240 scans
    python eval_miccai.py --reuse                  # skip pipeline, load saved JSON
    python eval_miccai.py --out results/eval.json  # save summary
    python eval_miccai.py --crop-k 8000            # VRAM-safe mode for RTX 3050
"""

import argparse
import json
import math
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.spatial.distance as compute_dist_matrix
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import f1_score

# ── make sub-packages importable ────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent
for _p in [
    _ROOT,
    _ROOT / "stage1_segmentation",
    _ROOT / "stage2_landmarks",
    _ROOT / "extensions" / "tgnet_ops",
    _ROOT / "extensions" / "teethland_ops",
]:
    sys.path.insert(0, str(_p))

# ── landmark classes (MICCAI 2024 order) ────────────────────────────────────
LM_CLASSES = ["Mesial", "Distal", "Cusp", "InnerPoint", "OuterPoint", "FacialPoint"]

# ── z-score std (must match training; also used in combined_pipeline.py) ────
Z_SCORE_STD: float = 17.3281  # mm

# ── default checkpoint paths ─────────────────────────────────────────────────
_DEFAULT_FPS = "checkpoints/CGIP_TGN_checkpoints/ckpts(new)/tgnet_fps.h5"
_DEFAULT_BDL = "checkpoints/CGIP_TGN_checkpoints/ckpts(new)/tgnet_bdl.h5"
_DEFAULT_LM  = "checkpoints/Teethland-checkpoints/landmarks_full.ckpt"


# ============================================================
# OBJ vertex reader (handles Teeth3DS  v x y z r g b  format)
# ============================================================

def read_obj_vertices(path: Path) -> np.ndarray:
    verts = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("v "):
                p = line.split()
                verts.append([float(p[1]), float(p[2]), float(p[3])])
    return np.array(verts, dtype=np.float32)


# ============================================================
# MICCAI 2022 metric helpers
# (adapted from "evaluation MICCAI2022.py" provided by the challenge)
# ============================================================

def _tooth_size(points: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum((centroid - points) ** 2, axis=0))


def _build_gt_inst_dict(mesh_verts, gt_instances, gt_labels):
    """Build per-instance centroid + size + FDI-label dict from GT arrays."""
    inst_dict = {}
    u_inst = np.unique(gt_instances)
    u_inst = u_inst[u_inst != 0]
    for l in u_inst:
        mask = gt_instances == l
        lbl = np.unique(gt_labels[mask])
        assert len(lbl) == 1, f"GT instance {l} has multiple FDI labels: {lbl}"
        verts = mesh_verts[mask]
        center = verts.mean(axis=0)
        size = _tooth_size(verts, center)
        inst_dict[str(l)] = {"label": int(lbl[0]), "centroid": center, "tooth_size": size}
    return inst_dict


def _build_pred_inst_dict(mesh_verts, pred_instances, pred_labels):
    """Build per-instance centroid + FDI-label dict from prediction arrays.
    Instances whose points carry more than one FDI label are discarded (same
    rule as the official MICCAI 2022 evaluation script)."""
    pred_instances = pred_instances.copy()
    pred_labels    = pred_labels.copy()
    inst_dict = {}
    u_inst = np.unique(pred_instances)
    u_inst = u_inst[u_inst != 0]
    for pi in u_inst:
        mask = pred_instances == pi
        lbl  = np.unique(pred_labels[mask])
        if len(lbl) == 1:
            verts  = mesh_verts[mask]
            center = verts.mean(axis=0)
            inst_dict[str(pi)] = {"label": int(lbl[0]), "centroid": center}
        else:
            pred_instances[mask] = 0
            pred_labels[mask]    = 0
    return inst_dict, pred_instances


def _match_centroids(gt_dict, pred_dict):
    """Hungarian matching: GT instance → nearest predicted instance centroid."""
    gt_cents   = [v["centroid"] for v in gt_dict.values()]
    pred_cents = [v["centroid"] for v in pred_dict.values()]
    M          = compute_dist_matrix.cdist(gt_cents, pred_cents)
    row_ind, col_ind = linear_sum_assignment(M)
    return {
        list(gt_dict.keys())[i]: list(pred_dict.keys())[j]
        for i, j in zip(row_ind, col_ind)
    }


def calc_TLA(gt_dict, pred_dict, matching):
    """Teeth Localisation Accuracy: mean normalised centroid distance."""
    TLA = 0.0
    for inst, info in gt_dict.items():
        if inst in matching:
            diff = gt_dict[inst]["centroid"] - pred_dict[matching[inst]]["centroid"]
            TLA += np.linalg.norm(diff / gt_dict[inst]["tooth_size"])
        else:
            TLA += 5.0 * np.linalg.norm(gt_dict[inst]["tooth_size"])
    return TLA / len(gt_dict)


def calc_TSA(gt_instances, pred_instances):
    """Teeth Segmentation Accuracy: micro-F1 (tooth vs gingiva)."""
    gt_bin   = (gt_instances   != 0).astype(int)
    pred_bin = (pred_instances  != 0).astype(int)
    return float(f1_score(gt_bin, pred_bin, average="micro"))


def calc_TIR(gt_dict, pred_dict, matching, threshold=0.5):
    """Teeth Identification Rate: % of correctly located + labelled teeth."""
    if not matching:
        return 0.0
    tir = 0
    for gt_inst, pred_inst in matching.items():
        dist = np.linalg.norm(
            (gt_dict[gt_inst]["centroid"] - pred_dict[pred_inst]["centroid"])
            / gt_dict[gt_inst]["tooth_size"]
        )
        if dist < threshold and gt_dict[gt_inst]["label"] == pred_dict[pred_inst]["label"]:
            tir += 1
    return tir / len(matching)


def compute_miccai2022(mesh_verts, gt_seg, pred_seg):
    """
    Compute TLA, TSA, TIR for one jaw scan.

    Args:
        mesh_verts : (N, 3) float32 original vertex positions in mm
        gt_seg     : {"labels": [...], "instances": [...]}  from .json
        pred_seg   : {"labels": ndarray(N,), "instances": ndarray(N,)}
                      from pipeline Stage 1 output

    Returns:
        (jaw_TLA, jaw_TSA, jaw_TIR)  — raw per-jaw scores
    """
    gt_instances  = np.array(gt_seg["instances"])
    gt_labels     = np.array(gt_seg["labels"])
    pred_instances = pred_seg["instances"].astype(int)
    pred_labels    = pred_seg["labels"].astype(int)

    gt_dict   = _build_gt_inst_dict(mesh_verts, gt_instances, gt_labels)
    pred_dict, pred_instances_clean = _build_pred_inst_dict(
        mesh_verts, pred_instances, pred_labels
    )

    if not pred_dict:
        tsa = calc_TSA(gt_instances, pred_instances_clean)
        return 0.0, tsa, 0.0

    matching = _match_centroids(gt_dict, pred_dict)
    tla = calc_TLA(gt_dict, pred_dict, matching)
    tsa = calc_TSA(gt_instances, pred_instances_clean)
    tir = calc_TIR(gt_dict, pred_dict, matching)
    return tla, tsa, tir


# ============================================================
# MICCAI 2024 metric helpers
# (adapted from metrics.py provided by the challenge)
# ============================================================

def _voc_ap(rec, prec):
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))


def _voc_ar(dist_thresh_list, recall_by_thresh, cat):
    mrec = np.array(dist_thresh_list[::-1])
    mpre = np.array(recall_by_thresh[::-1])
    mrec = np.concatenate(([0.], mrec, [1.]))
    mpre = np.concatenate(([0.], mpre, [0.]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))


def _eval_det_cls(pred_cls, gt_cls, dist_thresh):
    """
    Compute precision/recall/AP for a single class at one distance threshold.

    pred_cls : {scan_key: [[coord, score], ...]}
    gt_cls   : {scan_key: [coord, ...]}
    """
    # Build GT records
    class_recs = {}
    npos = 0
    for key, coords in gt_cls.items():
        kp_arr = np.array(coords, dtype=np.float64)
        det    = [False] * len(kp_arr)
        npos  += len(kp_arr)
        class_recs[key] = {"kp": kp_arr, "det": det}
    for key in pred_cls:
        if key not in class_recs:
            class_recs[key] = {"kp": np.array([]), "det": []}

    # Flatten predictions, sort by descending confidence
    scan_keys, confidence, KP = [], [], []
    for key, entries in pred_cls.items():
        for coord, score in entries:
            scan_keys.append(key)
            confidence.append(score)
            KP.append(coord)

    if not KP:
        return np.array([0.0]), np.array([0.0]), 0.0

    confidence = np.array(confidence)
    KP         = np.array(KP)
    order      = np.argsort(-confidence)
    KP         = KP[order]
    scan_keys  = [scan_keys[x] for x in order]

    nd = len(scan_keys)
    tp = np.zeros(nd)
    fp = np.zeros(nd)

    for d in range(nd):
        R    = class_recs[scan_keys[d]]
        kp   = KP[d]
        KPGT = R["kp"]
        dmin = np.inf
        jmin = -1
        if KPGT.size > 0:
            dists = np.linalg.norm(kp.reshape(1, 3) - KPGT, axis=1)
            dmin  = dists.min()
            jmin  = dists.argmin()

        if dmin < dist_thresh:
            if not R["det"][jmin]:
                tp[d] = 1.0
                R["det"][jmin] = 1
            else:
                fp[d] = 1.0
        else:
            fp[d] = 1.0

    fp  = np.cumsum(fp)
    tp  = np.cumsum(tp)
    rec = tp / float(npos) if npos > 0 else np.zeros_like(tp)
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap  = _voc_ap(rec, prec)
    return rec, prec, ap


def compute_miccai2024(pred_all: Dict, gt_all: Dict) -> Dict:
    """
    Compute mAP, mAR and per-class AP/AR at 30 distance thresholds [0, 0.1, ..., 2.9] mm.

    pred_all : {classname: {scan_key: [[coord, score], ...]}}
    gt_all   : {classname: {scan_key: [coord, ...]}}

    Returns dict with mAP, mAR, AP_cusp, AR_cusp, AP/AR per category.
    """
    dist_thresholds = [0.1 * i for i in range(30)]   # 0.0 … 2.9 mm

    # Accumulate AP and recall-at-threshold per class
    ap_by_thresh  = {c: [] for c in LM_CLASSES}   # list of 30 AP values
    rec_by_thresh = {c: [] for c in LM_CLASSES}   # list of 30 recall values

    for dist in dist_thresholds:
        for cls in LM_CLASSES:
            rec, prec, ap = _eval_det_cls(
                pred_all.get(cls, {}),
                gt_all.get(cls, {}),
                dist,
            )
            ap_by_thresh[cls].append(ap)
            rec_by_thresh[cls].append(float(rec[-1]) if len(rec) > 0 else 0.0)

    # Mean AP per class (average over 30 thresholds)
    mean_ap = {cls: float(np.mean(ap_by_thresh[cls])) for cls in LM_CLASSES}

    # AR per class: area under Recall vs exp(-threshold) curve
    exp_thresholds = np.exp(-np.array(dist_thresholds))
    mean_ar = {
        cls: _voc_ar(exp_thresholds.tolist(), rec_by_thresh[cls], cls)
        for cls in LM_CLASSES
    }

    overall_mAP = float(np.mean(list(mean_ap.values())))
    overall_mAR = float(np.mean(list(mean_ar.values())))

    return {
        "mAP":  overall_mAP,
        "mAR":  overall_mAR,
        # Per-class AP
        "AP_per_class":     mean_ap,
        "AP_cusp":          mean_ap["Cusp"],
        "AP_mesial_distal": (mean_ap["Mesial"] + mean_ap["Distal"]) / 2,
        "AP_inner_outer":   (mean_ap["InnerPoint"] + mean_ap["OuterPoint"]) / 2,
        "AP_facial":        mean_ap["FacialPoint"],
        # Per-class AR
        "AR_per_class":     mean_ar,
        "AR_cusp":          mean_ar["Cusp"],
        "AR_mesial_distal": (mean_ar["Mesial"] + mean_ar["Distal"]) / 2,
        "AR_inner_outer":   (mean_ar["InnerPoint"] + mean_ar["OuterPoint"]) / 2,
        "AR_facial":        mean_ar["FacialPoint"],
    }


# ============================================================
# Mesial / Distal splitter
# (same heuristic as visualize_scan.split_mesial_distal)
# ============================================================

def split_mesial_distal(landmarks: List[Dict]) -> List[Dict]:
    """
    Split "MesialDistal" pipeline predictions into "Mesial" / "Distal"
    using the x-coordinate and the FDI quadrant:
        Quadrants 1 & 4 (right side, x > 0):  smaller  x = Mesial
        Quadrants 2 & 3 (left  side, x < 0):  larger   x = Mesial
    """
    md_by_tooth: Dict = defaultdict(list)
    other: List = []
    for lm in landmarks:
        if lm["class"] == "MesialDistal":
            md_by_tooth[lm["fdi_tooth"]].append(lm)
        else:
            other.append(lm)

    resolved = list(other)
    for fdi, group in md_by_tooth.items():
        reverse = (fdi // 10) in (2, 3)   # left-side quadrants → reverse sort
        sorted_group = sorted(group, key=lambda l: l["coord"][0], reverse=reverse)
        for k, lm in enumerate(sorted_group):
            resolved.append({**lm, "class": "Mesial" if k == 0 else "Distal"})
    return resolved


# ============================================================
# Pipeline runner (Stage 1 + Stage 2 in one pass)
# Replicates combined_pipeline.run() but also returns seg output
# for MICCAI 2022 evaluation — avoids running Stage 1 twice.
# ============================================================

LANDMARK_HEADS = ["MesialDistal", "FacialPoint", "OuterPoint", "InnerPoint", "Cusp"]


class EvalPipeline:
    """
    Wraps TGNet (Stage 1) + LandmarkNet (Stage 2) for evaluation.
    Exposes both per-vertex segmentation output and per-tooth landmarks
    in a single forward pass per scan.
    """

    def __init__(self, fps_ckpt: str, bdl_ckpt: str, lm_ckpt: str,
                 device: str = "cuda", crop_k: int = 12_000):
        import torch
        from inference_pipeline import InferencePipeline
        from teethland.models.landmarknet import LandmarkNet

        self.device = device
        self.crop_k = crop_k

        # Stage 1
        self.seg_pipeline = InferencePipeline(
            fps_ckpt=fps_ckpt, bdl_ckpt=bdl_ckpt, device=device
        )

        # Stage 2 — manual instantiation (checkpoint has no stored hparams)
        _ckpt = torch.load(lm_ckpt, weights_only=False, map_location=device)
        self.lm_model = LandmarkNet(
            lr=0.0006, weight_decay=0.0001, epochs=500, warmup_epochs=5,
            dbscan_cfg={"max_neighbor_dist": 0.03, "min_points": 20,  # must match CLUSTER_MIN_PTS in landmark_postprocess.py
                        "weighted_cluster": True, "weighted_average": True},
            in_channels=9,
            channels_list=[48, 96, 192, 256],
            out_channels=[1, 4, 4, 4, 4, 4],
            depths=[3, 9, 3],
            heads_list=[6, 12, 24],
            window_sizes=[0.1, 0.2, 0.4],
            point_embedding={"use": True, "kpconv_point_influence": 0.02,
                             "kpconv_ball_radius": 0.05},
            stratified_union=False,
            downsample_ratio=0.26,
            max_drop_path_prob=0.3,
            stratified_downsample_ratio=0.26,
            crpe_bins=80,
            transformer_lr_ratio=0.1,
        )
        self.lm_model.load_state_dict(_ckpt["state_dict"])
        self.lm_model.eval().to(device)

    @staticmethod
    def _get_scale_factor(xyz_mm: np.ndarray) -> float:
        y_extent = float(xyz_mm[:, 1].max() - xyz_mm[:, 1].min())
        if y_extent < 1e-6:
            raise ValueError("Zero Y extent in mesh.")
        return y_extent / 1.8   # TGNet maps jaw Y to [-0.8, 1.0] (span = 1.8)

    def run(self, obj_path: Path) -> Dict:
        """
        One-pass evaluation run.

        Returns:
            {
              "jaw"       : "upper" | "lower",
              "labels"    : ndarray(N,)  per-vertex FDI labels     (Stage 1)
              "instances" : ndarray(N,)  per-vertex instance IDs   (Stage 1)
              "landmarks" : list of {class, coord, score, fdi_tooth}
            }
        """
        import torch
        import gen_utils as gu
        from sklearn.neighbors import KDTree as _KDT
        from teethland import PointTensor as _PT
        from pipeline.landmark_postprocess import extract_landmarks

        # ── Stage 1 ─────────────────────────────────────────────────────────
        seg = self.seg_pipeline.run(str(obj_path))
        jaw          = seg["jaw"]
        labels       = seg["labels"]       # (V,) FDI per original vertex
        instances    = seg["instances"]    # (V,) instance ID per original vertex
        sampled_xyz  = seg["sampled_xyz"]  # (24000, 3) TGNet normalised
        sampled_sem  = np.array(seg["sampled_sem_labels"])  # (24000,) FDI

        # ── Coordinate prep for Stage 2 ─────────────────────────────────────
        orig_verts  = gu.read_txt_obj_ls(str(obj_path), ret_mesh=False,
                                          use_tri_mesh=False)[0]
        orig_mm     = orig_verts[:, :3]         # (V, 3) mm
        orig_normals = orig_verts[:, 3:]        # (V, 3) normals
        scale_factor = self._get_scale_factor(orig_mm)

        jaw_mean = orig_mm.mean(axis=0)                         # (3,)
        jaw_norm = (orig_mm - jaw_mean) / Z_SCORE_STD           # (V, 3)
        min_y_c  = float(orig_mm[:, 1].min()) - float(jaw_mean[1])

        norm_tree = _KDT(jaw_norm, leaf_size=16)

        # ── Stage 2: per-tooth landmark detection ───────────────────────────
        all_lms: List[Dict] = []
        unique_fdi = sorted(int(l) for l in np.unique(sampled_sem) if l != 0)

        with torch.no_grad():
            for fdi in unique_fdi:
                torch.cuda.empty_cache()
                mask = sampled_sem == fdi
                if mask.sum() < 10:
                    continue

                # Centroid: TGNet space → z-score normalised space
                centroid_tgnet = sampled_xyz[mask].mean(axis=0)
                centroid_norm  = (
                    (centroid_tgnet + 0.8) * scale_factor + min_y_c
                ) / Z_SCORE_STD

                # Crop from original mesh
                crop_idxs = norm_tree.query(
                    centroid_norm[None], k=self.crop_k, return_distance=False
                )[0]
                crop_xyz  = jaw_norm[crop_idxs]          # (K, 3)
                crop_nrm  = orig_normals[crop_idxs]      # (K, 3)
                # Normalise normals to unit length — training does this; inference must match.
                _nrm_len = np.linalg.norm(crop_nrm, axis=1, keepdims=True)
                crop_nrm = crop_nrm / np.where(_nrm_len > 1e-8, _nrm_len, 1.0)
                cent_off  = crop_xyz - centroid_norm      # (K, 3)

                # Build PointTensor
                cxt = torch.from_numpy(crop_xyz).float().to(self.device)
                cnt = torch.from_numpy(crop_nrm).float().to(self.device)
                cot = torch.from_numpy(cent_off).float().to(self.device)
                feats = torch.cat([cxt, cnt, cot], dim=1)   # (K, 9)
                pt = _PT(
                    coordinates=cxt,
                    features=feats,
                    batch_counts=torch.tensor([cxt.shape[0]], device=self.device),
                )

                # Forward pass
                _seg_head, *lm_heads = self.lm_model(pt)

                # Post-process each landmark head
                for head_name, head_out in zip(LANDMARK_HEADS, lm_heads):
                    detected = extract_landmarks(head_out)
                    if detected is None:
                        continue
                    lm_mm = (detected.C * Z_SCORE_STD
                             + torch.tensor(jaw_mean, dtype=torch.float32,
                                            device=self.device))
                    for i in range(lm_mm.shape[0]):
                        all_lms.append({
                            "class":     head_name,
                            "coord":     lm_mm[i].cpu().tolist(),
                            "score":     float(detected.F[i, 0].cpu()),
                            "fdi_tooth": fdi,
                        })

        return {
            "jaw":       jaw,
            "labels":    labels,
            "instances": instances,
            "landmarks": all_lms,
        }


# ============================================================
# GT-seg pipeline  (Stage 2 only — bypasses TGNet)
# ============================================================

class GtSegEvalPipeline:
    """
    LandmarkNet-only evaluation pipeline.

    Instead of using TGNet predictions to locate teeth, this class reads the GT
    per-vertex FDI labels from the annotation .json file and uses them to build
    exact tooth centroids.  This isolates Stage 2 (landmark detection) quality
    from Stage 1 (segmentation) errors.

    Usage:
        pipeline = GtSegEvalPipeline(lm_ckpt=..., device="cuda")
        result = pipeline.run(obj_path, seg_json)
    """

    def __init__(self, lm_ckpt: str, device: str = "cuda", crop_k: int = 12_000):
        import torch
        from teethland.models.landmarknet import LandmarkNet

        self.device = device
        self.crop_k = crop_k

        _ckpt = torch.load(lm_ckpt, weights_only=False, map_location=device)
        self.lm_model = LandmarkNet(
            lr=0.0006, weight_decay=0.0001, epochs=500, warmup_epochs=5,
            dbscan_cfg={"max_neighbor_dist": 0.03, "min_points": 20,
                        "weighted_cluster": True, "weighted_average": True},
            in_channels=9,
            channels_list=[48, 96, 192, 256],
            out_channels=[1, 4, 4, 4, 4, 4],
            depths=[3, 9, 3],
            heads_list=[6, 12, 24],
            window_sizes=[0.1, 0.2, 0.4],
            point_embedding={"use": True, "kpconv_point_influence": 0.02,
                             "kpconv_ball_radius": 0.05},
            stratified_union=False,
            downsample_ratio=0.26,
            max_drop_path_prob=0.3,
            stratified_downsample_ratio=0.26,
            crpe_bins=80,
            transformer_lr_ratio=0.1,
        )
        self.lm_model.load_state_dict(_ckpt["state_dict"])
        self.lm_model.eval().to(device)

    def run(self, obj_path: Path, seg_json: Path) -> Dict:
        """
        Run landmark detection using GT tooth masks.

        Returns:
            {
              "jaw"       : "upper" | "lower"  (inferred from filename)
              "landmarks" : list of {class, coord, score, fdi_tooth}
            }
        """
        import torch
        import gen_utils as gu
        from sklearn.neighbors import KDTree as _KDT
        from teethland import PointTensor as _PT
        from pipeline.landmark_postprocess import extract_landmarks

        # Determine jaw from filename
        jaw = "lower" if "lower" in obj_path.name else "upper"

        # Load mesh vertices + normals
        orig_verts   = gu.read_txt_obj_ls(str(obj_path), ret_mesh=False, use_tri_mesh=False)[0]
        orig_mm      = orig_verts[:, :3]    # (V, 3) mm
        orig_normals = orig_verts[:, 3:]    # (V, 3) — will be unit-normalised below
        _nlen = np.linalg.norm(orig_normals, axis=1, keepdims=True)
        orig_normals = orig_normals / np.where(_nlen > 1e-8, _nlen, 1.0)

        # Z-score normalise full jaw
        jaw_mean = orig_mm.mean(axis=0)
        jaw_norm = (orig_mm - jaw_mean) / Z_SCORE_STD   # (V, 3)

        # Load GT FDI labels from annotation JSON
        with open(seg_json) as f:
            ann = json.load(f)
        fdi_arr  = np.array(ann["labels"],    dtype=np.int32)   # (V,)

        # Build KDTree in z-score space for crop queries
        norm_tree = _KDT(jaw_norm, leaf_size=16)

        all_lms: List[Dict] = []
        unique_fdi = sorted(int(v) for v in np.unique(fdi_arr) if v > 0)

        with torch.no_grad():
            for fdi in unique_fdi:
                torch.cuda.empty_cache()
                mask = fdi_arr == fdi
                if mask.sum() < 10:
                    continue

                # Exact centroid in z-score space — no TGNet space conversion needed
                centroid_norm = jaw_norm[mask].mean(axis=0)   # (3,)

                # Crop K nearest vertices from original mesh
                crop_idxs = norm_tree.query(
                    centroid_norm[None], k=min(self.crop_k, len(jaw_norm)),
                    return_distance=False,
                )[0]
                crop_xyz = jaw_norm[crop_idxs]          # (K, 3)
                crop_nrm = orig_normals[crop_idxs]      # (K, 3) already unit-normalised
                cent_off = crop_xyz - centroid_norm      # (K, 3)

                # Build PointTensor
                cxt   = torch.from_numpy(crop_xyz).float().to(self.device)
                cnt   = torch.from_numpy(crop_nrm).float().to(self.device)
                cot   = torch.from_numpy(cent_off).float().to(self.device)
                feats = torch.cat([cxt, cnt, cot], dim=1)   # (K, 9)
                pt    = _PT(
                    coordinates=cxt,
                    features=feats,
                    batch_counts=torch.tensor([cxt.shape[0]], device=self.device),
                )

                _seg_head, *lm_heads = self.lm_model(pt)

                for head_name, head_out in zip(LANDMARK_HEADS, lm_heads):
                    detected = extract_landmarks(head_out)
                    if detected is None:
                        continue
                    lm_mm = (detected.C * Z_SCORE_STD
                             + torch.tensor(jaw_mean, dtype=torch.float32,
                                            device=self.device))
                    for i in range(lm_mm.shape[0]):
                        all_lms.append({
                            "class":     head_name,
                            "coord":     lm_mm[i].cpu().tolist(),
                            "score":     float(detected.F[i, 0].cpu()),
                            "fdi_tooth": fdi,
                        })

        return {"jaw": jaw, "landmarks": all_lms}


# ============================================================
# GT loading
# ============================================================

def load_gt_seg(seg_json: Path) -> Dict:
    with open(seg_json) as f:
        return json.load(f)   # {"labels": [...], "instances": [...], ...}


def load_gt_landmarks(kpt_json: Path) -> Tuple[str, Dict[str, List]]:
    """
    Returns (scan_key, {classname: [coord, ...]}).
    scan_key is taken from the "key" field in the JSON.
    """
    with open(kpt_json) as f:
        data = json.load(f)
    scan_key = data.get("key", kpt_json.stem.replace("__kpt", ""))
    by_class: Dict[str, List] = {c: [] for c in LM_CLASSES}
    for obj in data.get("objects", []):
        cls = obj.get("class", "")
        if cls in by_class:
            by_class[cls].append(obj["coord"])
    return scan_key, by_class


# ============================================================
# Discovery: find all scans with OBJ + seg.json + kpt.json
# ============================================================

def discover_triplets(root: Path) -> List[Tuple[Path, Path, Path]]:
    kpt_dirs = [
        root / "data" / "3DTeethLand_landmarks_train",
        root / "data" / "3DTeethLand_landmarks_test",
    ]
    all_kpt = []
    for d in kpt_dirs:
        if d.exists():
            all_kpt.extend(d.rglob("*__kpt.json"))

    triplets = []
    for kpt in sorted(all_kpt):
        pid = kpt.parent.name
        jaw = "lower" if "lower" in kpt.name else "upper"
        obj = root / "data" / jaw / pid / f"{pid}_{jaw}.obj"
        seg = root / "data" / jaw / pid / f"{pid}_{jaw}.json"
        if obj.exists() and seg.exists():
            triplets.append((obj, seg, kpt))
    return triplets


# ============================================================
# Output directory for intermediate pipeline JSONs
# ============================================================

def lm_json_path(out_root: Path, obj_path: Path) -> Path:
    return out_root / obj_path.stem / (obj_path.stem + "_landmarks.json")


# ============================================================
# Printing helpers
# ============================================================

def _bar(width=62):
    print("=" * width)


def print_miccai2022(results: Dict):
    n = results["n_scans"]
    TLAs = results["TLA_list"]
    TSAs = results["TSA_list"]
    TIRs = results["TIR_list"]

    mean_TLA = float(np.mean([math.exp(-t) for t in TLAs]))
    mean_TSA = float(np.mean(TSAs))
    mean_TIR = float(np.mean(TIRs))
    score    = (mean_TLA + mean_TSA + mean_TIR) / 3

    _bar()
    print(f"  MICCAI 2022 — Teeth Segmentation  ({n} scans)")
    _bar()
    print(f"  TLA   (exp-norm, ↑)  : {mean_TLA:.4f}  ±{np.std([math.exp(-t) for t in TLAs]):.4f}")
    print(f"  TSA   (micro-F1, ↑)  : {mean_TSA:.4f}  ±{np.std(TSAs):.4f}")
    print(f"  TIR   (%, ↑)         : {mean_TIR:.4f}  ±{np.std(TIRs):.4f}")
    print(f"  ── Challenge Score   : {score:.4f}")
    _bar()
    print()


def print_miccai2024(m: Dict, n: int):
    _bar()
    print(f"  MICCAI 2024 — Landmark Detection  ({n} scans)")
    _bar()
    print(f"  mAP  (↑) : {m['mAP']:.4f}")
    print(f"  mAR  (↑) : {m['mAR']:.4f}")
    print()
    print(f"  {'Category':<18}  {'AP':>8}  {'AR':>8}")
    print("  " + "-" * 38)
    cats = [
        ("Mesial/Distal", "AP_mesial_distal", "AR_mesial_distal"),
        ("Cusp",          "AP_cusp",          "AR_cusp"),
        ("Inner/Outer",   "AP_inner_outer",   "AR_inner_outer"),
        ("Facial",        "AP_facial",        "AR_facial"),
    ]
    for name, ap_key, ar_key in cats:
        print(f"  {name:<18}  {m[ap_key]:8.4f}  {m[ar_key]:8.4f}")
    print()
    print(f"  Per-class AP:")
    for cls in LM_CLASSES:
        print(f"    {cls:<14}  {m['AP_per_class'][cls]:.4f}")
    print(f"  Per-class AR:")
    for cls in LM_CLASSES:
        print(f"    {cls:<14}  {m['AR_per_class'][cls]:.4f}")
    _bar()
    print()


# ============================================================
# Main
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="MICCAI 2022 + 2024 pipeline evaluation")
    p.add_argument("--max-scans",  type=int, default=10,
                   help="Number of scans to evaluate (0 = all, default 10)")
    p.add_argument("--reuse",      action="store_true",
                   help="Skip pipeline inference; load existing _landmarks.json files")
    p.add_argument("--fps-ckpt",   default=_DEFAULT_FPS)
    p.add_argument("--bdl-ckpt",   default=_DEFAULT_BDL)
    p.add_argument("--lm-ckpt",    default=_DEFAULT_LM)
    p.add_argument("--crop-k",     type=int, default=12_000,
                   help="Points per tooth crop (lower to 8000 for RTX 3050 OOM)")
    p.add_argument("--device",     default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--out",        default=None,
                   help="Save JSON summary to this path")
    p.add_argument("--skip-2022",  action="store_true",
                   help="Skip MICCAI 2022 metrics (useful if only landmarks matter)")
    p.add_argument("--skip-2024",  action="store_true",
                   help="Skip MICCAI 2024 metrics")
    p.add_argument("--use-gt-seg", action="store_true",
                   help=(
                       "Bypass TGNet (Stage 1): use GT FDI labels from the annotation JSON "
                       "to build exact tooth centroids for LandmarkNet. "
                       "Isolates Stage 2 landmark quality from Stage 1 segmentation errors. "
                       "MICCAI 2022 segmentation metrics are skipped automatically."
                   ))
    return p.parse_args()


def main():
    args = parse_args()

    # ── Discover matched scan triplets ──────────────────────────────────────
    triplets = discover_triplets(_ROOT)
    if not triplets:
        print("[ERROR] No matched (OBJ + seg JSON + kpt JSON) triplets found.")
        print("        Expected layout:")
        print("          data/{upper,lower}/{PID}/{PID}_{jaw}.obj")
        print("          data/{upper,lower}/{PID}/{PID}_{jaw}.json")
        print("          data/3DTeethLand_landmarks_*/{jaw}/{PID}/{PID}_{jaw}__kpt.json")
        sys.exit(1)

    n_total = len(triplets)
    n_eval  = n_total if args.max_scans == 0 else min(args.max_scans, n_total)
    triplets = triplets[:n_eval]

    # --use-gt-seg skips Stage 1 entirely; MICCAI 2022 needs seg labels so auto-skip it.
    if args.use_gt_seg:
        args.skip_2022 = True

    print(f"\n[INFO] {n_total} matched triplets found; evaluating {n_eval}.")
    print(f"[INFO] MICCAI 2022: {'SKIP' if args.skip_2022 else 'YES'}")
    print(f"[INFO] MICCAI 2024: {'SKIP' if args.skip_2024 else 'YES'}")
    print(f"[INFO] GT-seg mode: {'YES (Stage 1 bypassed)' if args.use_gt_seg else 'NO (TGNet Stage 1 active)'}")
    print()

    # ── Load models (unless --reuse) ────────────────────────────────────────
    pipeline = None
    if not args.reuse:
        if not Path(args.lm_ckpt).exists():
            print(f"[ERROR] Missing checkpoint: LandmarkNet at {args.lm_ckpt}")
            sys.exit(1)

        if args.use_gt_seg:
            print("Loading LandmarkNet (GT-seg mode — TGNet not loaded) …")
            pipeline = GtSegEvalPipeline(
                lm_ckpt=args.lm_ckpt,
                device=args.device,
                crop_k=args.crop_k,
            )
        else:
            for ck_attr, ck_path, ck_name in [
                ("fps_ckpt", args.fps_ckpt, "TGNet FPS"),
                ("bdl_ckpt", args.bdl_ckpt, "TGNet BDL"),
            ]:
                if not Path(ck_path).exists():
                    print(f"[ERROR] Missing checkpoint: {ck_name} at {ck_path}")
                    sys.exit(1)

            print("Loading models …")
            pipeline = EvalPipeline(
                fps_ckpt=args.fps_ckpt,
                bdl_ckpt=args.bdl_ckpt,
                lm_ckpt=args.lm_ckpt,
                device=args.device,
                crop_k=args.crop_k,
            )
        print("Models loaded.\n")

    # ── Per-scan evaluation ──────────────────────────────────────────────────
    out_root = _ROOT / "data" / "output"

    # MICCAI 2022 accumulators
    TLA_list: List[float] = []
    TSA_list: List[float] = []
    TIR_list: List[float] = []

    # MICCAI 2024 accumulators
    pred_all: Dict[str, Dict] = {c: {} for c in LM_CLASSES}
    gt_all:   Dict[str, Dict] = {c: {} for c in LM_CLASSES}

    for scan_idx, (obj_path, seg_path, kpt_path) in enumerate(triplets):
        stem = obj_path.stem
        t0   = time.time()
        print(f"[{scan_idx+1:3d}/{n_eval}]  {stem}", end="  ", flush=True)

        # ── Load GT ──────────────────────────────────────────────────────
        gt_seg = load_gt_seg(seg_path)
        scan_key, gt_lm_by_class = load_gt_landmarks(kpt_path)

        # Accumulate GT for MICCAI 2024
        if not args.skip_2024:
            for cls in LM_CLASSES:
                if scan_key not in gt_all[cls]:
                    gt_all[cls][scan_key] = []
                gt_all[cls][scan_key].extend(gt_lm_by_class[cls])

        # ── Run pipeline (or load cached) ─────────────────────────────────
        lm_out_path = lm_json_path(out_root, obj_path)

        if args.reuse and lm_out_path.exists():
            with open(lm_out_path) as f:
                saved = json.load(f)
            landmarks_raw = saved.get("landmarks", [])
            seg_result    = None   # not available in reuse mode
            print("(reused)", end="  ")
        else:
            try:
                if args.use_gt_seg:
                    result        = pipeline.run(obj_path, seg_path)
                    landmarks_raw = result["landmarks"]
                    seg_result    = None   # GT-seg mode: no predicted seg labels
                else:
                    result        = pipeline.run(obj_path)
                    landmarks_raw = result["landmarks"]
                    seg_result    = {"labels": result["labels"], "instances": result["instances"]}

                # Save landmark JSON for optional reuse
                lm_out_path.parent.mkdir(parents=True, exist_ok=True)
                with open(lm_out_path, "w") as f:
                    json.dump({"jaw": result["jaw"], "landmarks": landmarks_raw}, f, indent=2)
            except Exception as e:
                print(f"FAILED: {e}")
                traceback.print_exc()
                continue

        elapsed = time.time() - t0
        print(f"{len(landmarks_raw)} landmarks  ({elapsed:.1f}s)")

        # ── MICCAI 2022: Segmentation metrics ─────────────────────────────
        if not args.skip_2022 and seg_result is not None:
            try:
                mesh_verts = read_obj_vertices(obj_path)
                tla, tsa, tir = compute_miccai2022(mesh_verts, gt_seg, seg_result)
                TLA_list.append(tla)
                TSA_list.append(tsa)
                TIR_list.append(tir)
                print(f"         2022: TLA={math.exp(-tla):.3f}  TSA={tsa:.3f}  TIR={tir:.3f}")
            except Exception as e:
                print(f"         2022 ERROR: {e}")
        elif not args.skip_2022 and args.reuse:
            print("         2022: skipped (seg labels not saved in reuse mode)")

        # ── MICCAI 2024: Landmark metrics (accumulate) ────────────────────
        if not args.skip_2024:
            # Split MesialDistal → Mesial + Distal
            landmarks = split_mesial_distal(landmarks_raw)
            for lm in landmarks:
                cls   = lm["class"]
                coord = lm["coord"]
                score = lm.get("score", 0.5)
                if cls not in pred_all:
                    continue
                if scan_key not in pred_all[cls]:
                    pred_all[cls][scan_key] = []
                pred_all[cls][scan_key].append([coord, score])

    # ── Aggregate results ──────────────────────────────────────────────────
    print()
    summary = {}

    if not args.skip_2022 and TLA_list:
        mean_TLA_exp = float(np.mean([math.exp(-t) for t in TLA_list]))
        mean_TSA     = float(np.mean(TSA_list))
        mean_TIR     = float(np.mean(TIR_list))
        challenge_score_2022 = (mean_TLA_exp + mean_TSA + mean_TIR) / 3

        r2022 = {
            "n_scans":      len(TLA_list),
            "TLA_list":     TLA_list,
            "TSA_list":     TSA_list,
            "TIR_list":     TIR_list,
            "mean_TLA_exp": mean_TLA_exp,
            "mean_TSA":     mean_TSA,
            "mean_TIR":     mean_TIR,
            "score":        challenge_score_2022,
        }
        print_miccai2022(r2022)
        summary["miccai2022"] = {k: v for k, v in r2022.items()
                                  if k not in ("TLA_list", "TSA_list", "TIR_list")}

    if not args.skip_2024 and any(pred_all[c] for c in LM_CLASSES):
        m2024 = compute_miccai2024(pred_all, gt_all)
        print_miccai2024(m2024, n_eval)
        summary["miccai2024"] = m2024
    elif not args.skip_2024:
        print("[WARNING] No landmark predictions accumulated — MICCAI 2024 skipped.")

    # ── Save JSON summary ──────────────────────────────────────────────────
    if args.out and summary:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        def _serial(obj):
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2, default=_serial)
        print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
