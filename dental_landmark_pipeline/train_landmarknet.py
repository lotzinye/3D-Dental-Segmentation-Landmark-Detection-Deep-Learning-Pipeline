"""
train_landmarknet.py
---------------------
Training script for Stage 2: LandmarkNet anatomical landmark detection.

Architecture
------------
  LandmarkNet (PyTorch Lightning)
    └─ StratifiedTransformer (multi-scale windowed attention)
         Windows: [0.1, 0.2, 0.4] z-score units ≈ [1.7mm, 3.5mm, 6.9mm]
         Heads: seg (tooth/gingiva) + 5 landmark heads
                (MesialDistal, FacialPoint, OuterPoint, InnerPoint, Cusp)

Data layout
-----------
  Per tooth crop:
    data/teeth3ds/{PID}/{PID}_{jaw}.obj    — full jaw mesh
    data/teeth3ds/{PID}/{PID}_{jaw}.json   — per-vertex FDI labels + instances
    data/teeth3ds/{PID}/{PID}_{jaw}_kpt.json — GT landmark coordinates

  The dataset crops each tooth individually using a KDTree radius query,
  then builds a PointTensor with 9-channel features:
    [xyz_norm (3), normals (3), centroid_offsets_norm (3)]
  Targets:
    landmarks — PointTensor(C=GT_landmark_coords, F=landmark_class_ids)
    labels    — PointTensor(F=signed-dist to nearest GT landmark, −1=gingiva)

Usage (RTX 3050)
----------------
  python train_landmarknet.py \\
      --data-root   data/teeth3ds \\
      --train-split splits/train.txt \\
      --val-split   splits/val.txt \\
      --ckpt-dir    checkpoints/landmarknet \\
      --epochs 200  --batch-size 4  --npoints 12000

Resume:
  python train_landmarknet.py [same args] --resume

Notes
-----
  - LandmarkNet takes one tooth crop at a time as a PointTensor.
  - The DataLoader produces lists; a custom collate stacks them into a
    batched PointTensor using teethland.stack().
  - Batch size refers to number of tooth crops per GPU step.
  - Z_SCORE_STD = 17.3281 mm (from the 3DTeethLand dataset statistics).
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d
import pytorch_lightning as pl
import torch
from scipy.spatial import cKDTree
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE / "stage2_landmarks"))
sys.path.insert(0, str(_HERE / "extensions"))

import teethland
from teethland import PointTensor
from teethland.models.landmarknet import LandmarkNet


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
Z_SCORE_STD: float = 17.3281      # mm; global normalisation std from dataset stats
CROP_K: int = 12_000              # nearest neighbours per tooth crop
DIST_THRESH: float = 0.12         # landmark distance threshold in z-score units

# Landmark class names in order (match training targets)
LANDMARK_CLASSES = {
    "Mesial":      0,
    "Distal":      1,
    "FacialPoint": 2,
    "OuterPoint":  3,
    "InnerPoint":  4,
    "Cusp":        5,
}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TeethLandmarkDataset(Dataset):
    """
    Produces one (PointTensor, (landmarks_PT, labels_PT)) per tooth per scan.

    A scan with 14 visible teeth produces 14 samples.

    Each PointTensor x has:
        C = (N, 3)  z-score normalised coordinates
        F = (N, 9)  [xyz_norm | normals | centroid_offsets_norm]

    landmarks_PT:
        C = (L, 3)  GT landmark coordinates in z-score space
        F = (L, 1)  landmark class id (0-5)

    labels_PT:
        F = (N, 1)  signed dist to nearest GT landmark (negative = gingiva)
    """

    def __init__(
        self,
        root: str,
        split_file: Optional[str] = None,
        npoints: int = CROP_K,
        augment: bool = True,
    ):
        self.root    = Path(root)
        self.npoints = npoints
        self.augment = augment

        # Each element: (obj_path, json_path, kpt_path, fdi_tooth_id)
        self.samples: List[Tuple[Path, Path, Path, int]] = []
        self._discover(split_file)
        print(f"  {len(self.samples)} tooth samples from {root}")

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def _discover(self, split_file: Optional[str]):
        if split_file is not None:
            ids = [l.strip() for l in Path(split_file).read_text().splitlines() if l.strip()]
        else:
            # Auto-discover: all patient dirs that contain _kpt.json files
            ids = None

        for obj in sorted(self.root.glob("**/*.obj")):
            pid_jaw = obj.stem                 # e.g. 01F4JV8X_upper
            jsn = obj.with_suffix(".json")
            kpt = obj.parent / f"{pid_jaw}_kpt.json"

            if not jsn.exists() or not kpt.exists():
                continue

            if ids is not None:
                # match on patient-id prefix (before underscore)
                pid = pid_jaw.split("_")[0]
                if pid not in ids and pid_jaw not in ids:
                    continue

            # Count unique tooth FDIs in this scan
            with open(jsn) as f:
                ann = json.load(f)
            fdi_arr = np.array(ann["labels"], dtype=np.int32)
            tooth_fdis = [int(v) for v in np.unique(fdi_arr) if v > 0]

            for fdi in tooth_fdis:
                self.samples.append((obj, jsn, kpt, fdi))

    # ------------------------------------------------------------------
    # Load helpers
    # ------------------------------------------------------------------

    def _load_scan(self, obj_path: Path):
        """Load OBJ, compute normals, return (vertices, normals) float32."""
        mesh = o3d.io.read_triangle_mesh(str(obj_path))
        mesh.remove_duplicated_vertices()
        mesh.remove_degenerate_triangles()
        mesh.remove_unreferenced_vertices()
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()
        xyz     = np.asarray(mesh.vertices,       dtype=np.float32)
        normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
        return xyz, normals

    def _zscore_norm(self, xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Z-score normalise the full jaw.
        Returns (xyz_norm, jaw_mean, jaw_std).
        jaw_std is fixed to Z_SCORE_STD (global dataset std, not per-scan).
        """
        jaw_mean = xyz.mean(axis=0)
        xyz_norm = (xyz - jaw_mean) / Z_SCORE_STD
        return xyz_norm, jaw_mean

    def _load_landmarks(self, kpt_path: Path, fdi: int) -> np.ndarray:
        """
        Load GT landmarks for a specific tooth FDI.
        Returns (L, 4) array: [x, y, z, class_id] in mm.
        Returns empty (0, 4) if no landmarks for this tooth.
        """
        with open(kpt_path) as f:
            kpt_data = json.load(f)

        landmarks = []
        # 3DTeethLand kpt format: list of {coord, class, label}
        # OR dict keyed by FDI
        if isinstance(kpt_data, list):
            for entry in kpt_data:
                entry_fdi = entry.get("label", entry.get("fdi", -1))
                if int(entry_fdi) != fdi:
                    continue
                cls_name = entry.get("class", entry.get("type", ""))
                cls_id   = LANDMARK_CLASSES.get(cls_name, -1)
                if cls_id < 0:
                    continue
                coord = entry.get("coord", entry.get("coordinates", [0, 0, 0]))
                landmarks.append([coord[0], coord[1], coord[2], cls_id])
        elif isinstance(kpt_data, dict):
            for cls_name, points in kpt_data.items():
                cls_id = LANDMARK_CLASSES.get(cls_name, -1)
                if cls_id < 0:
                    continue
                for entry in points:
                    entry_fdi = entry.get("label", entry.get("fdi", -1))
                    if int(entry_fdi) != fdi:
                        continue
                    coord = entry.get("coord", entry.get("coordinates", [0, 0, 0]))
                    landmarks.append([coord[0], coord[1], coord[2], cls_id])

        if not landmarks:
            return np.zeros((0, 4), dtype=np.float32)
        return np.array(landmarks, dtype=np.float32)

    def _crop_tooth(
        self,
        xyz_norm: np.ndarray,
        normals: np.ndarray,
        fdi_arr: np.ndarray,
        fdi: int,
        npoints: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Crop a tooth ROI using KDTree: take the npoints nearest neighbours
        of the tooth centroid in z-score space.

        Returns (xyz_crop, normals_crop, is_tooth_mask)
            is_tooth_mask: bool (N,) True if vertex belongs to fdi
        """
        tooth_mask = fdi_arr == fdi
        if tooth_mask.sum() == 0:
            # Fallback: return whole scan (should not happen after discovery)
            idxs = np.random.choice(len(xyz_norm), min(npoints, len(xyz_norm)), replace=False)
            return xyz_norm[idxs], normals[idxs], tooth_mask[idxs]

        centroid = xyz_norm[tooth_mask].mean(axis=0)
        tree     = cKDTree(xyz_norm)
        k        = min(npoints, len(xyz_norm))
        _, idxs  = tree.query(centroid, k=k)

        return xyz_norm[idxs], normals[idxs], tooth_mask[idxs]

    def _augment(
        self,
        xyz: np.ndarray,
        normals: np.ndarray,
        landmarks: np.ndarray,
    ):
        """Random Y-axis rotation + small scale jitter."""
        theta = np.random.uniform(0, 2 * np.pi)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
        xyz     = xyz @ R.T
        normals = normals @ R.T
        if landmarks.shape[0] > 0:
            landmarks[:, :3] = landmarks[:, :3] @ R.T

        scale = np.random.uniform(0.95, 1.05)
        xyz   = xyz * scale
        if landmarks.shape[0] > 0:
            landmarks[:, :3] = landmarks[:, :3] * scale

        return xyz, normals, landmarks

    def _build_label_field(
        self,
        xyz_crop: np.ndarray,
        tooth_mask: np.ndarray,
        landmarks_mm: np.ndarray,
    ) -> np.ndarray:
        """
        Per-point label: signed distance to nearest GT landmark.
          > 0 → foreground (close to a landmark)
          < 0 → background (gingiva or far from landmark)

        Convention matches LandmarkNet training:
          labels.F >= 0 → foreground (tooth surface)
          Otherwise    → gingiva / background
        """
        if landmarks_mm.shape[0] == 0 or not tooth_mask.any():
            # No GT landmarks or no tooth points → all gingiva
            return np.full(len(xyz_crop), -1.0, dtype=np.float32)

        # Compute unsigned distance from each point to nearest GT landmark
        tree  = cKDTree(landmarks_mm[:, :3])
        dists, _ = tree.query(xyz_crop)

        labels = np.where(tooth_mask, dists, -dists).astype(np.float32)
        return labels

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------

    def __getitem__(self, index: int):
        obj_path, jsn_path, kpt_path, fdi = self.samples[index]

        # --- Load scan ---
        xyz_raw, normals = self._load_scan(obj_path)

        # --- Z-score normalise jaw ---
        xyz_norm, jaw_mean = self._zscore_norm(xyz_raw)

        # --- FDI labels ---
        with open(jsn_path) as f:
            ann = json.load(f)
        fdi_arr = np.array(ann["labels"], dtype=np.int32)

        # --- GT landmarks (in mm → convert to z-score space) ---
        landmarks_raw = self._load_landmarks(kpt_path, fdi)   # (L, 4) mm + class_id
        if landmarks_raw.shape[0] > 0:
            landmarks_znorm = landmarks_raw.copy()
            landmarks_znorm[:, :3] = (landmarks_raw[:, :3] - jaw_mean) / Z_SCORE_STD
        else:
            landmarks_znorm = landmarks_raw  # empty (0, 4)

        # --- Crop tooth ROI ---
        xyz_crop, nrm_crop, tooth_mask = self._crop_tooth(
            xyz_norm, normals, fdi_arr, fdi, self.npoints
        )

        # --- Augmentation ---
        if self.augment:
            xyz_crop, nrm_crop, landmarks_znorm = self._augment(
                xyz_crop, nrm_crop, landmarks_znorm.copy()
            )

        # --- Build 9-channel features [xyz | normals | centroid_offsets] ---
        centroid = xyz_crop[tooth_mask].mean(axis=0) if tooth_mask.any() else xyz_crop.mean(axis=0)
        offsets  = xyz_crop - centroid
        nrm_crop = nrm_crop / (np.linalg.norm(nrm_crop, axis=1, keepdims=True) + 1e-8)
        features = np.concatenate([xyz_crop, nrm_crop, offsets], axis=1)   # (N, 9)

        # --- Per-point label field ---
        labels_field = self._build_label_field(
            xyz_crop, tooth_mask, landmarks_znorm
        )

        # --- Build PointTensors ---
        N = xyz_crop.shape[0]
        x = PointTensor(
            coordinates=torch.from_numpy(xyz_crop).float(),
            features=torch.from_numpy(features).float(),
            batch_counts=torch.tensor([N], dtype=torch.int32),
        )

        if landmarks_znorm.shape[0] > 0:
            lm_coords  = torch.from_numpy(landmarks_znorm[:, :3]).float()
            lm_classes = torch.from_numpy(landmarks_znorm[:, 3:4]).float()
        else:
            lm_coords  = torch.zeros((0, 3), dtype=torch.float32)
            lm_classes = torch.zeros((0, 1), dtype=torch.float32)

        L = lm_coords.shape[0]
        landmarks_pt = PointTensor(
            coordinates=lm_coords,
            features=lm_classes,
            batch_counts=torch.tensor([L], dtype=torch.int32),
        )

        labels_pt = PointTensor(
            coordinates=torch.from_numpy(xyz_crop).float(),
            features=torch.from_numpy(labels_field[:, None]).float(),
            batch_counts=torch.tensor([N], dtype=torch.int32),
        )

        return x, (landmarks_pt, labels_pt)

    def __len__(self) -> int:
        return len(self.samples)


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def collate_fn(
    batch: List[Tuple[PointTensor, Tuple[PointTensor, PointTensor]]],
) -> Tuple[PointTensor, Tuple[PointTensor, PointTensor]]:
    """
    Stack a list of (x, (landmarks, labels)) tuples into batched PointTensors.
    Uses teethland.stack() which concatenates along the point dimension and
    builds correct batch_counts.
    """
    xs, targets = zip(*batch)
    landmark_pts, label_pts = zip(*targets)

    x_batch        = teethland.stack(list(xs))
    landmark_batch = teethland.stack(list(landmark_pts))
    labels_batch   = teethland.stack(list(label_pts))

    return x_batch, (landmark_batch, labels_batch)


# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------

def build_model_args(args) -> dict:
    """
    StratifiedTransformer hyperparameters matching the 3DTeethLand paper.
    These are passed as **model_args to LandmarkNet.__init__().
    """
    return {
        # --- network topology ---
        # NOTE: Must match the pretrained checkpoint (landmarks_full.ckpt) and the
        # architecture used in combined_pipeline.py / eval_miccai.py exactly.
        # Mismatched weights will not load and produce garbage output.
        "in_channels":                 9,           # xyz + normals + centroid_offsets
        "channels_list":               [48, 96, 192, 256],
        "out_channels":                [1, 4, 4, 4, 4, 4],  # 6 heads: seg + 5 landmark
        "depths":                      [3, 9, 3],
        "heads_list":                  [6, 12, 24],
        "window_sizes":                [0.1, 0.2, 0.4],

        # --- point embedding ---
        "point_embedding": {
            "use":                   True,
            "kpconv_point_influence": 0.02,
            "kpconv_ball_radius":     0.05,
        },

        # --- stratified transformer settings ---
        "stratified_union":            False,
        "downsample_ratio":            0.26,
        "max_drop_path_prob":          0.3,
        "stratified_downsample_ratio": 0.26,
        "crpe_bins":                   80,
        "transformer_lr_ratio":        0.1,

        # --- training settings (passed to LandmarkNet) ---
        "lr":                          args.lr,
        "weight_decay":                args.weight_decay,
        "epochs":                      args.epochs,
        "warmup_epochs":               args.warmup_epochs,

        # --- post-processing (for val inference, not loss) ---
        "dbscan_cfg": {
            "max_neighbor_dist": 0.03,
            "min_points":        20,
            "weighted_cluster":  True,
            "weighted_average":  True,
        },
    }


# ---------------------------------------------------------------------------
# Checkpoint callback
# ---------------------------------------------------------------------------

class BestCheckpointCallback(pl.Callback):
    """Saves best and per-epoch checkpoints in a custom directory."""

    def __init__(self, ckpt_dir: Path):
        self.ckpt_dir = ckpt_dir
        self.best_loss = float("inf")
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: LandmarkNet):
        metrics   = trainer.logged_metrics
        val_loss  = metrics.get("loss/val", metrics.get("loss/train", float("inf")))
        epoch     = trainer.current_epoch

        ckpt_path = self.ckpt_dir / f"epoch_{epoch:04d}.pt"
        trainer.save_checkpoint(str(ckpt_path))

        if val_loss < self.best_loss:
            self.best_loss = val_loss
            trainer.save_checkpoint(str(self.ckpt_dir / "best.pt"))
            print(f"  → New best  (loss={float(val_loss):.5f})")


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train LandmarkNet Stage 2")
    p.add_argument("--data-root",      required=True)
    p.add_argument("--train-split",    default=None)
    p.add_argument("--val-split",      default=None)
    p.add_argument("--ckpt-dir",       default="checkpoints/landmarknet")
    p.add_argument("--epochs",         type=int,   default=200)
    p.add_argument("--batch-size",     type=int,   default=4,
                   help="Tooth crops per batch (4 for RTX 3050; 16 for V100)")
    p.add_argument("--npoints",        type=int,   default=CROP_K)
    p.add_argument("--lr",             type=float, default=6e-4)
    p.add_argument("--weight-decay",   type=float, default=1e-4)
    p.add_argument("--warmup-epochs",  type=int,   default=10)
    p.add_argument("--num-workers",    type=int,   default=4)
    p.add_argument("--no-augment",     action="store_true")
    p.add_argument("--resume",         action="store_true",
                   help="Resume from latest checkpoint in --ckpt-dir")
    p.add_argument("--no-compile",     action="store_true",
                   help="Disable torch.compile (required for PyTorch < 2.0)")
    p.add_argument("--precision",      default="16-mixed",
                   choices=["32", "16-mixed", "bf16-mixed"],
                   help="Training precision (16-mixed = AMP FP16)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    print("="*60)
    print("Stage 2: LandmarkNet Training")
    print("="*60)

    # --- Datasets ---
    train_ds = TeethLandmarkDataset(
        root=args.data_root,
        split_file=args.train_split,
        npoints=args.npoints,
        augment=not args.no_augment,
    )
    val_ds = TeethLandmarkDataset(
        root=args.data_root,
        split_file=args.val_split,
        npoints=args.npoints,
        augment=False,
    ) if args.val_split else None

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=args.num_workers > 0,
    ) if val_ds else None

    print(f"Train: {len(train_ds)} tooth crops")
    print(f"Val:   {len(val_ds) if val_ds else 0} tooth crops")

    # --- Model ---
    model_args = build_model_args(args)
    model = LandmarkNet(**model_args)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    if not args.no_compile:
        try:
            model = torch.compile(model)
            print("torch.compile: enabled")
        except Exception as e:
            print(f"torch.compile: skipped ({e})")

    # --- Callbacks & logger ---
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_cb  = BestCheckpointCallback(ckpt_dir)
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="epoch")

    # Optional: TensorBoard logger
    tb_logger = pl.loggers.TensorBoardLogger(
        save_dir=str(ckpt_dir / "tb_logs"),
        name="landmarknet",
    )

    # --- Resume ---
    ckpt_path = None
    if args.resume:
        latest_ckpts = sorted(ckpt_dir.glob("epoch_*.pt"))
        if latest_ckpts:
            ckpt_path = str(latest_ckpts[-1])
            print(f"Resuming from {ckpt_path}")
        else:
            print("No checkpoint found — starting from scratch.")

    # --- Trainer ---
    # Detect MPS (Apple) vs CUDA vs CPU
    if torch.cuda.is_available():
        accelerator, devices = "gpu", 1
    else:
        accelerator, devices = "cpu", 1

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=accelerator,
        devices=devices,
        precision=args.precision,
        callbacks=[ckpt_cb, lr_monitor],
        logger=tb_logger,
        log_every_n_steps=10,
        enable_progress_bar=True,
        gradient_clip_val=10.0,
    )

    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=ckpt_path,
    )

    print(f"\nTraining complete.")
    print(f"Best checkpoint: {ckpt_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
