"""
losses.py
---------
Loss functions for Stage-1 segmentation and Stage-2 landmark detection.

Stage 1 — SegLoss
    Combined Cross-Entropy + Soft-Dice loss.
    CE handles class imbalance via optional class weights; Dice enforces
    region overlap, which is critical when gingiva (class 0) dominates.

Stage 2 — LandmarkLoss
    Per-head regression:
      • Distance loss  : smooth L1 on predicted vs GT distance
      • Offset loss    : smooth L1 on predicted vs GT 3-D offset,
                         masked to points where GT distance < MAX_DIST
                         (only 'close' points carry reliable offset signal)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets.landmark_dataset import MAX_DIST


# ---------------------------------------------------------------------------
# Stage-1 Segmentation loss
# ---------------------------------------------------------------------------

class SegLoss(nn.Module):
    """
    Cross-Entropy + Soft-Dice loss for point-cloud segmentation.

    Args:
        num_classes:   number of output classes (default 17)
        ce_weight:     scalar weight on CE term (default 1.0)
        dice_weight:   scalar weight on Dice term (default 1.0)
        class_weights: (num_classes,) tensor of per-class CE weights, or None
        smooth:        Dice smoothing epsilon (default 1.0)
        ignore_index:  class index to ignore (-1 to disable; default -1)
    """

    def __init__(
        self,
        num_classes: int = 17,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        class_weights: torch.Tensor = None,
        smooth: float = 1.0,
        ignore_index: int = -1,
    ):
        super().__init__()
        self.num_classes  = num_classes
        self.ce_weight    = ce_weight
        self.dice_weight  = dice_weight
        self.smooth       = smooth
        self.ignore_index = ignore_index
        self.register_buffer("class_weights", class_weights)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (B, C, N) — raw class scores
            targets: (B, N)    — ground-truth class indices

        Returns:
            scalar loss tensor
        """
        # Cross-Entropy
        ce = F.cross_entropy(
            logits, targets,
            weight=self.class_weights,
            ignore_index=self.ignore_index if self.ignore_index >= 0 else -100,
        )

        # Soft Dice per class
        probs   = F.softmax(logits, dim=1)                       # (B, C, N)
        one_hot = F.one_hot(
            targets.clamp(0), self.num_classes
        ).permute(0, 2, 1).float()                               # (B, C, N)

        intersection = (probs * one_hot).sum(dim=(0, 2))         # (C,)
        cardinality  = (probs + one_hot).sum(dim=(0, 2))         # (C,)
        dice_per_cls = (2 * intersection + self.smooth) / (cardinality + self.smooth)
        dice = 1.0 - dice_per_cls.mean()

        return self.ce_weight * ce + self.dice_weight * dice


# ---------------------------------------------------------------------------
# Stage-2 Landmark loss
# ---------------------------------------------------------------------------

class LandmarkLoss(nn.Module):
    """
    Per-head distance + offset regression loss for landmark detection.

    For each of the num_heads landmark heads:
      loss = dist_loss + offset_weight * offset_loss

    Distance loss:   Smooth-L1 over all points.
    Offset loss:     Smooth-L1 over points where GT distance < MAX_DIST
                     (only points close to a landmark have a meaningful GT
                     offset direction).

    Args:
        num_heads:      number of landmark heads (default 5)
        offset_weight:  relative weight of offset term (default 1.0)
        beta:           Smooth-L1 beta (transition point; default 0.1)
    """

    def __init__(
        self,
        num_heads: int = 5,
        offset_weight: float = 1.0,
        beta: float = 0.1,
    ):
        super().__init__()
        self.num_heads     = num_heads
        self.offset_weight = offset_weight
        self.beta          = beta

    def forward(
        self,
        preds: list,
        dist_targets: torch.Tensor,
        offset_targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            preds:          list of num_heads tensors, each (B, 4, N)
                              ch 0   = predicted distance
                              ch 1-3 = predicted offset vector
            dist_targets:   (B, num_heads, N)   float32
            offset_targets: (B, num_heads, N, 3) float32

        Returns:
            scalar total loss
        """
        total = torch.tensor(0.0, device=dist_targets.device)

        for h, pred in enumerate(preds):
            pred_dist   = pred[:, 0, :]          # (B, N)
            pred_off    = pred[:, 1:, :].permute(0, 2, 1)   # (B, N, 3)

            gt_dist = dist_targets[:, h, :]      # (B, N)
            gt_off  = offset_targets[:, h, :, :] # (B, N, 3)

            # Distance loss — all points
            dist_loss = F.smooth_l1_loss(
                pred_dist, gt_dist, beta=self.beta, reduction="mean"
            )

            # Offset loss — only near-landmark points
            near_mask = (gt_dist < MAX_DIST)     # (B, N) bool
            if near_mask.any():
                off_loss = F.smooth_l1_loss(
                    pred_off[near_mask],
                    gt_off[near_mask],
                    beta=self.beta,
                    reduction="mean",
                )
            else:
                off_loss = torch.tensor(0.0, device=pred.device)

            total = total + dist_loss + self.offset_weight * off_loss

        return total / self.num_heads
