"""
combined_pipeline.py
--------------------
CombinedDentalPipeline: chains Stage 1 (TGNet segmentation) and
Stage 2 (3DTeethLand landmark detection) into a single inference call.

Stage 1 — TGNet (inference_pipeline.py):
    Input : full jaw .obj mesh
    Output: per-vertex FDI labels (11-48) + jaw type (upper/lower)

Stage 2 — 3DTeethLand LandmarkNet:
    Input : single-tooth point cloud (PointTensor, in mm)
    Output: 6 × landmark head predictions (distance + 3D offset per point)

Stage 3 — Post-processing (landmark_postprocess.py):
    Converts each head to final (x, y, z) landmark coordinates via DBSCAN

Stage 4 — FDI label inheritance:
    Each detected landmark inherits the FDI number from Stage 1
"""

import sys
import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional

# --- Make stage modules importable ---
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "stage1_segmentation"))   # bare imports like gen_utils
sys.path.insert(0, str(_ROOT / "stage2_landmarks"))       # teethland package

import gen_utils as gu
import ops_utils as tu
from inference_pipeline import InferencePipeline

from teethland.models.landmarknet import LandmarkNet
from teethland import PointTensor

from pipeline.data_bridge import get_scale_factor, tgnet_crop_to_pointtensor
from pipeline.landmark_postprocess import extract_landmarks

# 3DTeethLand global z-score std used during training
Z_SCORE_STD: float = 17.3281  # mm


# The model has 5 landmark heads (after the segmentation head).
# Head 0 is trained jointly on Mesial (class 0) + Distal (class 1) landmarks;
# the remaining heads each cover one class.
LANDMARK_HEADS = [
    "MesialDistal",  # head 0: detects both mesial & distal margins
    "FacialPoint",
    "OuterPoint",
    "InnerPoint",
    "Cusp",
]


class CombinedDentalPipeline:
    """
    End-to-end pipeline: jaw .obj → FDI-labelled anatomical landmarks.

    Args:
        tgnet_fps_ckpt : path to TGNet Stage-1 (FPS) checkpoint (.pth / .pt)
        tgnet_bdl_ckpt : path to TGNet Stage-2 (BDL boundary) checkpoint
        landmark_ckpt  : path to 3DTeethLand LandmarkNet checkpoint (.ckpt)
        device         : 'cuda' or 'cpu'
        crop_k         : nearest-neighbour count for tooth crop extraction
                         (default 12 000, matching 3DTeethLand training size;
                          lower to ~8 000 if RTX 3050 runs out of VRAM)
    """

    def __init__(
        self,
        tgnet_fps_ckpt: str,
        tgnet_bdl_ckpt: str,
        landmark_ckpt:  str,
        device: str = "cuda",
        crop_k: int = 12_000,
    ):
        self.device = device
        self.crop_k = crop_k

        # --- Stage 1: TGNet segmentation ---
        self.seg_pipeline = InferencePipeline(
            fps_ckpt=tgnet_fps_ckpt,
            bdl_ckpt=tgnet_bdl_ckpt,
            device=device,
        )

        # --- Stage 2: 3DTeethLand landmark network ---
        # The checkpoint has no 'hyper_parameters' stored, so PL's load_from_checkpoint
        # cannot reconstruct the model automatically.  We instantiate it manually using
        # the known architecture config and load the state dict directly.
        _ckpt = torch.load(landmark_ckpt, weights_only=False, map_location=device)
        self.landmark_model = LandmarkNet(
            # training hparams (not used at inference, but required by __init__)
            lr=0.0006,
            weight_decay=0.0001,
            epochs=500,
            warmup_epochs=5,
            dbscan_cfg={
                'max_neighbor_dist': 0.03,
                'min_points': 20,   # must match CLUSTER_MIN_PTS in landmark_postprocess.py
                'weighted_cluster': True,
                'weighted_average': True,
            },
            # StratifiedTransformer architecture (must match checkpoint)
            in_channels=9,
            channels_list=[48, 96, 192, 256],
            out_channels=[1, 4, 4, 4, 4, 4],
            # [seg, MesialDistal, FacialPoint, OuterPoint, InnerPoint, Cusp]
            
            depths=[3, 9, 3],
            heads_list=[6, 12, 24],
            window_sizes=[0.1, 0.2, 0.4],
            point_embedding={
                'use': True,
                'kpconv_point_influence': 0.02,
                'kpconv_ball_radius': 0.05,
            },
            stratified_union=False,
            downsample_ratio=0.26,
            max_drop_path_prob=0.3,
            stratified_downsample_ratio=0.26,
            crpe_bins=80,
            transformer_lr_ratio=0.1,
        )
        self.landmark_model.load_state_dict(_ckpt['state_dict'])
        self.landmark_model.eval().to(device)

    @torch.no_grad()
    def run(self, obj_path: str) -> Dict[str, Any]:
        """
        Run the full pipeline on a single jaw scan.

        Args:
            obj_path: path to input .obj file

        Returns:
            dict with keys:
              'jaw'       : 'upper' or 'lower'
              'landmarks' : list of dicts, each with
                              'class'     : landmark type name
                              'coord'     : [x, y, z] in mm
                              'score'     : confidence in [0, 1]
                              'fdi_tooth' : FDI tooth number (int)
        """
        # ---- STAGE 1: global jaw segmentation --------------------------------
        # Load original vertices (in mm) to build a dense tooth crop for Stage 2
        # read_txt_obj_ls returns [vertices_with_normals (N,6)] when ret_mesh=False
        original_verts = gu.read_txt_obj_ls(obj_path, ret_mesh=False, use_tri_mesh=False)[0]
        orig_xyz_mm    = original_verts[:, :3]    # (V, 3) raw mm coords
        orig_normals   = original_verts[:, 3:]    # (V, 3) per-vertex normals
        scale_factor   = get_scale_factor(orig_xyz_mm)

        # --- Z-score normalise full jaw to match 3DTeethLand training ---
        # Training: ZScoreNormalize(mean=None, std=17.3281) → per-scan mean subtracted,
        # then divided by 17.3281.  PointTensor C and F[0:3] are in this normalised space.
        jaw_mean  = orig_xyz_mm.mean(axis=0)              # (3,) per-axis mean
        jaw_norm  = (orig_xyz_mm - jaw_mean) / Z_SCORE_STD  # (V, 3) z-score normalised

        # TGNet centres the mesh first, so min_y in TGNet space is w.r.t. centred coords
        min_y_c = float(orig_xyz_mm[:, 1].min()) - float(jaw_mean[1])   # scalar

        # Run TGNet — returns dict: {'jaw', 'labels', 'instances', 'sampled_feats', 'sampled_xyz'}
        seg_result = self.seg_pipeline.run(obj_path)
        jaw          = seg_result["jaw"]
        sampled_xyz  = seg_result["sampled_xyz"]            # (24000, 3) in TGNet norm space
        labels       = np.array(seg_result["sampled_sem_labels"])  # (24000,) FDI per sampled pt

        # Build KDTree in z-score normalised space for tooth-neighbourhood lookup
        from sklearn.neighbors import KDTree as _KDTree
        norm_tree = _KDTree(jaw_norm, leaf_size=16)

        # ---- STAGES 2-4: per-tooth landmark detection -----------------------
        all_landmarks: List[Dict] = []
        unique_fdi = sorted(set(int(l) for l in labels if l != 0))

        from teethland import PointTensor as _PT

        for fdi_label in unique_fdi:
            torch.cuda.empty_cache()

            # --- Stage 2a: ROI extraction — crop from ORIGINAL mesh in z-score space ---
            tooth_mask = labels == fdi_label
            if tooth_mask.sum() < 10:
                continue   # skip degenerate tiny clusters

            # Convert TGNet-norm centroid → 3DTeethLand norm space.
            # TGNet: norm = (centered - min_y_c) / y_range * 1.8 - 0.8
            #   => centered = (norm + 0.8) * scale_factor + min_y_c
            #   => mm = centered + jaw_mean   (jaw_mean cancels in the next step)
            # 3DTeethLand: norm3d = (mm - jaw_mean) / Z_SCORE_STD
            #   => norm3d = ((tgnet_norm + 0.8) * scale_factor + min_y_c) / Z_SCORE_STD
            tooth_norm_xyz  = sampled_xyz[tooth_mask]                   # TGNet space
            centroid_tgnet  = tooth_norm_xyz.mean(axis=0)               # (3,)
            centroid_norm   = ((centroid_tgnet + 0.8) * scale_factor + min_y_c) / Z_SCORE_STD

            # Select crop_k nearest original-mesh vertices in z-score space
            crop_idxs = norm_tree.query(
                centroid_norm[None], k=self.crop_k, return_distance=False
            )[0]                                              # (crop_k,)

            crop_xyz_norm  = jaw_norm[crop_idxs]             # (crop_k, 3) z-score normalised
            crop_normals   = orig_normals[crop_idxs]         # (crop_k, 3)
            # Normalise normals to unit length — training does this; inference must match.
            nrm_norms = np.linalg.norm(crop_normals, axis=1, keepdims=True)
            crop_normals = crop_normals / np.where(nrm_norms > 1e-8, nrm_norms, 1.0)

            # Centroid offsets in z-score space (matching CentroidOffsetsAsFeatures in training)
            centroid_offsets = crop_xyz_norm - centroid_norm  # (crop_k, 3)

            # Build PointTensor: features = [xyz_norm, normals, centroid_offsets]
            crop_xyz_t   = torch.from_numpy(crop_xyz_norm).float().to(self.device)
            crop_norm_t  = torch.from_numpy(crop_normals).float().to(self.device)
            cent_off_t   = torch.from_numpy(centroid_offsets).float().to(self.device)
            features     = torch.cat([crop_xyz_t, crop_norm_t, cent_off_t], dim=1)  # (N, 9)

            pt = _PT(
                coordinates=crop_xyz_t,
                features=features,
                batch_counts=torch.tensor([crop_xyz_t.shape[0]], device=self.device),
            )

            # --- Stage 3: dual-head landmark detection ---
            seg_head, *lm_heads = self.landmark_model(pt)

            # --- Stage 4: post-process each landmark head ---
            # Landmark coords are in z-score normalised space; convert back to mm.
            for head_name, head_out in zip(LANDMARK_HEADS, lm_heads):
                detected = extract_landmarks(head_out)
                if detected is None:
                    continue
                # Convert from z-score normalised space to mm
                landmark_mm = detected.C * Z_SCORE_STD + torch.from_numpy(jaw_mean).float().to(self.device)
                n_lm = landmark_mm.shape[0]
                for i in range(n_lm):
                    all_landmarks.append(
                        {
                            "class":     head_name,
                            "coord":     landmark_mm[i].cpu().tolist(),
                            "score":     float(detected.F[i, 0].cpu()),
                            "fdi_tooth": fdi_label,
                        }
                    )

        return {"jaw": jaw, "landmarks": all_landmarks}

    def run_and_save(self, obj_path: str, out_path: str) -> None:
        """Convenience wrapper: run pipeline and save JSON output."""
        result = self.run(obj_path)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved {len(result['landmarks'])} landmarks to {out_path}")
