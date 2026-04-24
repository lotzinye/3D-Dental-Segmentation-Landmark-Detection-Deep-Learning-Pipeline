"""
landmark_postprocess.py
-----------------------
Extracts final landmark (x, y, z) coordinates from a 3DTeethLand
landmark head output (distance score + 3D offset vector per point).

Logic mirrors fullnet.py lines 334-346 in the original 3DTeethLand repo.
"""

from typing import Optional

import torch

from teethland import PointTensor
from teethland.cluster import cluster

# Threshold in z-score units (multiply by 17.3281 to get mm).
# 0.12 z-score ≈ 2.08 mm  — points predicted within this distance of a landmark
# are included as landmark candidates (same default as 3DTeethLand training).
DIST_THRESH: float = 0.12   # z-score units (~2.08 mm)

# DBSCAN params for merging nearby candidate points into one landmark.
# 0.03 z-score ≈ 0.52 mm epsilon; min_points must stay in sync with dbscan_cfg
# in combined_pipeline.py and eval_miccai.py.
CLUSTER_MAX_DIST: float = 0.03   # z-score units (~0.52 mm)
CLUSTER_MIN_PTS:  int   = 20


def extract_landmarks(
    head_output: PointTensor,
    dist_thresh:      float = DIST_THRESH,
    max_neighbor_dist: float = CLUSTER_MAX_DIST,
    min_points:        int   = CLUSTER_MIN_PTS,
) -> Optional[PointTensor]:
    """
    Post-process one landmark head's output into final landmark coordinates.

    Args:
        head_output: PointTensor where
            head_output.C  = (N, 3) point coordinates in z-score space
            head_output.F  = (N, 4):
                               col 0: predicted distance to landmark (z-score units)
                               cols 1-3: predicted (dx, dy, dz) offset to landmark
        dist_thresh:       candidate inclusion threshold (z-score units, ~2.08 mm).
        max_neighbor_dist: DBSCAN epsilon (z-score units, ~0.52 mm).
        min_points:        DBSCAN minimum cluster size.  Use a smaller value (e.g. 5)
                           for sparse heads such as MesialDistal where predictions
                           cluster at tooth margins with few supporting points.

    Returns:
        PointTensor of detected landmarks:
            .C = (K, 3) landmark positions in z-score space
            .F = (K, 1) confidence scores in [0, 1]
        Returns None if fewer than min_points candidate points survive the threshold.
    """
    distances = head_output.F[:, 0]   # (N,)
    offsets   = head_output.F[:, 1:]  # (N, 3)

    mask = distances < dist_thresh
    if mask.sum() < min_points:
        return None

    # Landmark position = mesh point + predicted offset
    coords  = head_output.C[mask] + offsets[mask]           # (M, 3)
    dists   = distances[mask].clamp(0.0, dist_thresh)        # (M,)
    weights = (dist_thresh - dists) / dist_thresh            # (M,) higher = closer = better

    candidates = PointTensor(
        coordinates=coords,
        features=weights.unsqueeze(1),
        batch_counts=torch.tensor([coords.shape[0]], device=coords.device),
    )

    return cluster(
        candidates,
        max_neighbor_dist=max_neighbor_dist,
        min_points=min_points,
        weighted_cluster=True,
        weighted_average=True,
    )
