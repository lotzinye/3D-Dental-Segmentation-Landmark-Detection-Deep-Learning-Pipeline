"""
landmark_model.py  (Lean Edition)
----------------------------------
3-block PointNet++ MSG landmark detection model.

Lean changes vs the original 4-block model
-------------------------------------------
  • 3 SA layers instead of 4 (SA3 = global)
  • Reduced SA1 centroids: 512 → 512  (unchanged — already compact)
  • Reduced SA2 centroids: 128 → 128  (unchanged)
  • SA3 = global instead of 32-centroid SA3 + 8-centroid SA4
  • Input points per tooth: 6 000 → 3 000
  • Regression heads: 128 → 64 channels
  • Larger batch size (8 → 24 on V100; 8 → 10 on RTX 3050 w/ AMP)

Architecture
------------
Encoder:
  SA1 (MSG): 512 centroids, radii [0.05, 0.10]  → 192 ch
  SA2 (MSG): 128 centroids, radii [0.10, 0.20]  → 384 ch
  SA3 (global):  1 virtual centroid              → 256 ch

Decoder:
  FP3 (640 → 256)  →  FP2 (448 → 128)  →  FP1 (137 → 64)

Heads (5 × independent):
  Dropout + Conv1d(64, 4)
    ch 0   : predicted distance to nearest landmark  (≥ 0 after clamp)
    ch 1–3 : predicted 3-D offset vector

Input:  (B, N, 3) xyz_norm  +  (B, N, 6) features  [N ≈ 3 000]
Output: list of 5 tensors, each (B, 4, N)

VRAM  ≈ 1.5–1.8 GB at B=8,  N=3 000  (RTX 3050 with AMP)
      ≈ 3.5–4.5 GB at B=24, N=3 000  (V100 with AMP)
"""

import torch
import torch.nn as nn

from .pointnet2_utils import (
    PointNetSetAbstractionMsg,
    PointNetSetAbstraction,
    PointNetFeaturePropagation,
)

LANDMARK_HEADS = [
    "MesialDistal",   # head 0 — detects mesial & distal together
    "FacialPoint",
    "OuterPoint",
    "InnerPoint",
    "Cusp",
]
NUM_HEADS = len(LANDMARK_HEADS)   # 5


class PointNet2LandmarkModel(nn.Module):
    """
    Lean PointNet++ landmark detection model.

    Args:
        num_heads:   number of landmark heads (default 5)
        in_channels: input feature channels excluding xyz (default 6)
        dropout:     dropout before each regression head (default 0.4)
    """

    def __init__(
        self,
        num_heads: int = NUM_HEADS,
        in_channels: int = 6,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.num_heads  = num_heads
        self.head_names = LANDMARK_HEADS

        # ------------------------------------------------------------------
        # Encoder
        # ------------------------------------------------------------------
        self.sa1 = PointNetSetAbstractionMsg(
            npoint=512,
            radius_list=[0.05, 0.10],
            nsample_list=[16, 32],
            in_channel=in_channels,
            mlp_list=[[32, 32, 64], [64, 64, 128]],
        )   # → 192 ch

        self.sa2 = PointNetSetAbstractionMsg(
            npoint=128,
            radius_list=[0.10, 0.20],
            nsample_list=[16, 32],
            in_channel=192,
            mlp_list=[[64, 64, 128], [128, 128, 256]],
        )   # → 384 ch

        # SA3 global — replaces old SA3 (32 centroids) + SA4 (8 centroids)
        self.sa3 = PointNetSetAbstraction(
            npoint=None, radius=None, nsample=None,
            in_channel=384,
            mlp=[256, 256],
            group_all=True,
        )   # → 256 ch, output shape (B, 256, 1)

        # ------------------------------------------------------------------
        # Decoder
        # ------------------------------------------------------------------
        # FP3: global (1 pt, 256 ch) → SA2 (128 pts, 384 ch skip)
        self.fp3 = PointNetFeaturePropagation(
            in_channel=384 + 256,   # 640
            mlp=[256, 256],
        )

        # FP2: → SA1 (512 pts, 192 ch skip)
        self.fp2 = PointNetFeaturePropagation(
            in_channel=192 + 256,   # 448
            mlp=[128, 128],
        )

        # FP1: → input (N pts); skip = raw features + xyz (9 ch total)
        self.fp1 = PointNetFeaturePropagation(
            in_channel=in_channels + 3 + 128,   # 6 + 3 + 128 = 137
            mlp=[64, 64],
        )

        # ------------------------------------------------------------------
        # Regression heads
        # ------------------------------------------------------------------
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(dropout),
                nn.Conv1d(64, 4, kernel_size=1),
            )
            for _ in range(num_heads)
        ])

    def forward(
        self,
        xyz: torch.Tensor,
        features: torch.Tensor,
    ) -> list:
        """
        Args:
            xyz:      (B, N, 3)            — z-score normalised coordinates
            features: (B, N, in_channels)  — normals + centroid offsets

        Returns:
            preds: list of num_heads tensors, each (B, 4, N)
                   ch 0 = distance prediction
                   ch 1–3 = offset vector prediction
        """
        pts = features.permute(0, 2, 1)    # (B, C, N)

        # Encode
        l1_xyz, l1_pts = self.sa1(xyz, pts)          # (B,512,3), (B,192,512)
        l2_xyz, l2_pts = self.sa2(l1_xyz, l1_pts)    # (B,128,3), (B,384,128)
        l3_xyz, l3_pts = self.sa3(l2_xyz, l2_pts)    # (B,  1,3), (B,256,  1)

        # Decode
        l2_pts = self.fp3(l2_xyz, l3_xyz, l2_pts, l3_pts)   # (B,256,128)
        l1_pts = self.fp2(l1_xyz, l2_xyz, l1_pts, l2_pts)   # (B,128,512)

        # FP1 skip: raw xyz + features concatenated
        l0_skip = torch.cat([xyz, features], dim=-1).permute(0, 2, 1)  # (B,9,N)
        l0_pts  = self.fp1(xyz, l1_xyz, l0_skip, l1_pts)               # (B,64,N)

        return [head(l0_pts) for head in self.heads]   # list of (B, 4, N)
