"""
seg_model.py  (Lean Edition)
----------------------------
3-block PointNet++ MSG segmentation model for per-point FDI tooth labelling.

Lean changes vs the original 4-block model
-------------------------------------------
  • 3 SA layers instead of 4 (SA3 = global, no FPS/ball-query overhead)
  • Reduced SA1 centroids: 2048 → 1024
  • Reduced SA2 centroids: 512  → 256
  • Input points:  10,000 → 6,000  (set in train_seg.py / TeethSegDataset)
  • Final MLP head: 128 → 64 channels

Architecture
------------
Encoder:
  SA1 (MSG): 1 024 centroids, radii [0.05, 0.10]  → 192 ch
  SA2 (MSG): 256  centroids, radii [0.10, 0.20]  → 384 ch
  SA3 (global SA): all points, no radius limit    → 256 ch  (1 virtual centroid)

Decoder:
  FP3 (640 → 256)  →  FP2 (448 → 128)  →  FP1 (134 → 64)

Head:
  Dropout + Conv1d(64, 17)

Input:  (B, N, 3) xyz  +  (B, N, 3) normals   [N ≈ 6 000 after FPS]
Output: (B, 17, N)  raw logits

VRAM  ≈ 1.8–2.2 GB at B=2, N=6 000  (fits RTX 3050 with headroom for AMP)
      ≈ 4.5–5.5 GB at B=12, N=6 000 on V100 with AMP FP16
"""

import torch
import torch.nn as nn

from .pointnet2_utils import (
    PointNetSetAbstractionMsg,
    PointNetSetAbstraction,
    PointNetFeaturePropagation,
)

NUM_CLASSES = 17   # 0 = gingiva, 1–16 = teeth


class PointNet2SegModel(nn.Module):
    """
    Lean PointNet++ MSG segmentation model.

    Args:
        num_classes: output classes (default 17)
        in_channels: input feature channels excluding xyz (default 3 — normals)
        dropout:     dropout before final head (default 0.5)
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        in_channels: int = 3,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.num_classes = num_classes

        # ------------------------------------------------------------------
        # Encoder
        # ------------------------------------------------------------------
        # SA1 — MSG, 1 024 centroids
        self.sa1 = PointNetSetAbstractionMsg(
            npoint=1024,
            radius_list=[0.05, 0.10],
            nsample_list=[16, 32],
            in_channel=in_channels,
            mlp_list=[[32, 32, 64], [64, 64, 128]],
        )   # → 64 + 128 = 192 ch

        # SA2 — MSG, 256 centroids
        self.sa2 = PointNetSetAbstractionMsg(
            npoint=256,
            radius_list=[0.10, 0.20],
            nsample_list=[16, 32],
            in_channel=192,
            mlp_list=[[64, 64, 128], [128, 128, 256]],
        )   # → 128 + 256 = 384 ch

        # SA3 — global (groups ALL remaining points into one virtual centroid)
        # Replaces the old SA3+SA4 pair; captures full scan context in one layer.
        self.sa3 = PointNetSetAbstraction(
            npoint=None, radius=None, nsample=None,
            in_channel=384,
            mlp=[256, 256],
            group_all=True,
        )   # → 256 ch,  output shape (B, 256, 1)

        # ------------------------------------------------------------------
        # Decoder
        # ------------------------------------------------------------------
        # FP3: upsample global (1 pt, 256 ch) → SA2 (256 pts, 384 ch skip)
        self.fp3 = PointNetFeaturePropagation(
            in_channel=384 + 256,   # SA2 skip + SA3 up
            mlp=[256, 256],
        )

        # FP2: upsample → SA1 (1 024 pts, 192 ch skip)
        self.fp2 = PointNetFeaturePropagation(
            in_channel=192 + 256,
            mlp=[128, 128],
        )

        # FP1: upsample → input (N pts); skip = raw xyz+normals (6 ch)
        self.fp1 = PointNetFeaturePropagation(
            in_channel=in_channels + 3 + 128,   # 6 + 128 = 134
            mlp=[64, 64],
        )

        # ------------------------------------------------------------------
        # Segmentation head
        # ------------------------------------------------------------------
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv1d(64, num_classes, kernel_size=1),
        )

    def forward(self, xyz: torch.Tensor, normals: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xyz:     (B, N, 3)  — normalised point positions
            normals: (B, N, 3)  — unit surface normals

        Returns:
            logits: (B, num_classes, N)
        """
        l0_xyz = xyz                                      # (B, N, 3)
        l0_pts = normals.permute(0, 2, 1)                 # (B, 3, N)

        # Encode
        l1_xyz, l1_pts = self.sa1(l0_xyz, l0_pts)        # (B,1024,3), (B,192,1024)
        l2_xyz, l2_pts = self.sa2(l1_xyz, l1_pts)        # (B, 256,3), (B,384, 256)
        l3_xyz, l3_pts = self.sa3(l2_xyz, l2_pts)        # (B,   1,3), (B,256,   1)

        # Decode
        l2_pts = self.fp3(l2_xyz, l3_xyz, l2_pts, l3_pts)   # (B,256,256)
        l1_pts = self.fp2(l1_xyz, l2_xyz, l1_pts, l2_pts)   # (B,128,1024)

        # FP1 skip = raw xyz + normals concatenated (channels-first)
        l0_skip = torch.cat([l0_xyz, normals], dim=-1).permute(0, 2, 1)  # (B,6,N)
        l0_pts  = self.fp1(l0_xyz, l1_xyz, l0_skip, l1_pts)              # (B,64,N)

        return self.head(l0_pts)   # (B, 17, N)

    @torch.no_grad()
    def predict(self, xyz: torch.Tensor, normals: torch.Tensor) -> torch.Tensor:
        """Returns hard class predictions (B, N) long."""
        return self.forward(xyz, normals).argmax(dim=1)
