"""
postprocess.py
--------------
Convert dense per-point landmark predictions into discrete 3-D positions.

Algorithm (per landmark head, per tooth):
  1. Select candidate points: predicted distance < DIST_THRESHOLD
  2. Compute candidate positions: xyz_point + pred_offset_vector
  3. Cluster candidates with DBSCAN (eps, min_samples configurable)
  4. Cluster centre = mean of member predicted positions
  5. Convert back from z-score norm → millimetres

For the MesialDistal head (head 0) we expect 2 clusters (one Mesial,
one Distal).  Splitting is done by x-coordinate relative to the jaw
quadrant after converting to mm.

Output format matches the pipeline JSON schema:
    [
      {"fdi": 11, "class": "Mesial",      "xyz": [x, y, z]},
      {"fdi": 11, "class": "Distal",      "xyz": [x, y, z]},
      {"fdi": 11, "class": "FacialPoint", "xyz": [x, y, z]},
      ...
    ]
"""

from __future__ import annotations
from typing import List, Optional

import numpy as np

try:
    from sklearn.cluster import DBSCAN
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

# Landmark head order — must match landmark_model.py and landmark_dataset.py
LANDMARK_HEADS = ["MesialDistal", "FacialPoint", "OuterPoint", "InnerPoint", "Cusp"]
Z_SCORE_STD    = 17.3281    # mm

# Candidate selection: points with predicted distance below this threshold
DIST_THRESHOLD = 0.15       # normalised space  (~2.6 mm)

# DBSCAN defaults (normalised space)
DBSCAN_EPS     = 0.05       # ~0.87 mm
DBSCAN_MINSAMP = 3


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _cluster_positions(
    positions: np.ndarray,
    eps: float,
    min_samples: int,
    max_clusters: Optional[int] = None,
) -> List[np.ndarray]:
    """
    Run DBSCAN on candidate positions and return cluster centres.

    Returns:
        List of (3,) float32 arrays — one per cluster, sorted by cluster size.
    """
    if len(positions) == 0:
        return []

    if not _HAS_SKLEARN:
        # Fallback: treat all candidates as a single cluster
        return [positions.mean(axis=0).astype(np.float32)]

    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(positions)
    unique = set(labels) - {-1}
    if not unique:
        return []

    centres = []
    for lbl in unique:
        mask   = labels == lbl
        centre = positions[mask].mean(axis=0).astype(np.float32)
        centres.append((mask.sum(), centre))

    # Sort by cluster size (largest first)
    centres.sort(key=lambda x: x[0], reverse=True)
    if max_clusters:
        centres = centres[:max_clusters]
    return [c for _, c in centres]


def _norm_to_mm(xyz_norm: np.ndarray, jaw_mean: np.ndarray) -> np.ndarray:
    return xyz_norm * Z_SCORE_STD + jaw_mean


def _split_mesial_distal(
    clusters_norm: List[np.ndarray],
    fdi: int,
    jaw_mean: np.ndarray,
) -> List[dict]:
    """
    Given 1 or 2 cluster centres from the MesialDistal head,
    assign 'Mesial' / 'Distal' based on x-coordinate convention.

    FDI quadrant 1 & 4 (right, x > 0): ascending x → first = Mesial
    FDI quadrant 2 & 3 (left,  x < 0): descending x → first = Mesial
    """
    if not clusters_norm:
        return []

    # Convert to mm for final output
    clusters_mm = [_norm_to_mm(c, jaw_mean) for c in clusters_norm]

    quadrant = fdi // 10
    if len(clusters_mm) == 1:
        return [{"class": "Mesial", "xyz": clusters_mm[0].tolist()}]

    # Sort by x in mm space
    clusters_mm_sorted = sorted(clusters_mm, key=lambda c: c[0])
    if quadrant in (1, 4):   # right side — smaller x = more mesial
        mesial, distal = clusters_mm_sorted[0], clusters_mm_sorted[-1]
    else:                     # left side  — larger x = more mesial
        mesial, distal = clusters_mm_sorted[-1], clusters_mm_sorted[0]

    return [
        {"class": "Mesial", "xyz": mesial.tolist()},
        {"class": "Distal", "xyz": distal.tolist()},
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_landmarks(
    xyz_norm: np.ndarray,
    preds: list,
    fdi: int,
    jaw_mean: np.ndarray,
    dist_threshold: float = DIST_THRESHOLD,
    dbscan_eps: float = DBSCAN_EPS,
    dbscan_min_samples: int = DBSCAN_MINSAMP,
) -> List[dict]:
    """
    Convert dense PointNet++ predictions to discrete landmark positions.

    Args:
        xyz_norm:  (N, 3) float32 — z-score normalised point positions
        preds:     list of 5 tensors, each (1, 4, N) or (4, N) float32
                   — output of PointNet2LandmarkModel.forward()
        fdi:       FDI tooth label (used for Mesial/Distal assignment)
        jaw_mean:  (3,) float32 — per-scan mean in mm (for denorm)
        dist_threshold:    candidate selection cutoff (normalised)
        dbscan_eps:        DBSCAN neighbourhood radius (normalised)
        dbscan_min_samples: DBSCAN minimum cluster size

    Returns:
        list of {"fdi": int, "class": str, "xyz": [x_mm, y_mm, z_mm]}
    """
    results = []

    for head_idx, pred_tensor in enumerate(preds):
        head_name = LANDMARK_HEADS[head_idx]

        # Allow (1, 4, N) or (4, N)
        if pred_tensor.ndim == 3:
            pred_tensor = pred_tensor[0]            # (4, N)

        pred_dist = pred_tensor[0].cpu().numpy()    # (N,)
        pred_off  = pred_tensor[1:].T.cpu().numpy() # (N, 3)

        # Select candidates
        mask       = pred_dist < dist_threshold
        if mask.sum() == 0:
            continue
        cand_pos   = (xyz_norm[mask] + pred_off[mask]).astype(np.float32)

        # Cluster
        max_k = 2 if head_name == "MesialDistal" else 1
        centres = _cluster_positions(cand_pos, dbscan_eps, dbscan_min_samples, max_clusters=max_k)

        if head_name == "MesialDistal":
            entries = _split_mesial_distal(centres, fdi, jaw_mean)
        else:
            if centres:
                xyz_mm = _norm_to_mm(centres[0], jaw_mean)
                entries = [{"class": head_name, "xyz": xyz_mm.tolist()}]
            else:
                entries = []

        for entry in entries:
            entry["fdi"] = fdi
            results.append(entry)

    return results
