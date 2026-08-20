"""
train_tgnet.py
--------------
Training script for Stage 1: TGNet tooth segmentation.

Trains the FPS (Farthest Point Sampling) grouping network first, then
trains the BDL (Boundary Detail Learning) network which depends on a
trained FPS checkpoint to identify boundary points.

Architecture
------------
  Stage 1a — FpsGroupingNetworkModel  (FPS + 7 loss terms)
  Stage 1b — BdlGroupingNetworkModel  (boundary-aware refinement)

Data layout (Teeth3DS)
----------------------
  data/teeth3ds/{PID}/{PID}_{jaw}.obj   — mesh
  data/teeth3ds/{PID}/{PID}_{jaw}.json  — per-vertex FDI labels
                                          {"labels": [...], "instances": [...]}

FDI → gt_seg_label mapping (as used by TGNet)
----------------------------------------------
  gingiva (label 0)  → -1
  upper:  11-18 → 0-7,  21-28 → 8-15
  lower:  41-48 → 0-7,  31-38 → 8-15

Usage
-----
  # Train FPS model only
  python train_tgnet.py fps \\
      --data-root data/teeth3ds \\
      --ckpt-fps  checkpoints/tgnet_fps \\
      --epochs 100

  # Then train BDL (requires completed FPS checkpoint)
  python train_tgnet.py bdl \\
      --data-root   data/teeth3ds \\
      --ckpt-fps    checkpoints/tgnet_fps \\
      --ckpt-bdl    checkpoints/tgnet_bdl \\
      --cache-dir   checkpoints/bdl_cache \\
      --epochs 100

  # Train both sequentially
  python train_tgnet.py both \\
      --data-root   data/teeth3ds \\
      --ckpt-fps    checkpoints/tgnet_fps \\
      --ckpt-bdl    checkpoints/tgnet_bdl \\
      --cache-dir   checkpoints/bdl_cache

Resume:
  Add --resume to continue from the latest checkpoint in the given dir.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import open3d as o3d
import torch
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE / "stage1_segmentation"))
sys.path.insert(0, str(_HERE / "extensions"))

from stage1_segmentation.models.fps_grouping_network_model import FpsGroupingNetworkModel
from stage1_segmentation.models.bdl_grouping_network_model import BdlGroupingNetworkModel


# ---------------------------------------------------------------------------
# Constants  (match TGNet inference pipeline exactly)
# ---------------------------------------------------------------------------
N_POINTS = 24_000          # points sampled per scan
N_CLASSES = 17             # 16 teeth + 1 gingiva (class 0 in CrossEntropy)
# Global Y-axis normalisation used by BDL (from bdl_grouping_network_model.py)
Y_AXIS_MAX =  33.15232091532151
Y_AXIS_MIN = -36.9843781139949


# ---------------------------------------------------------------------------
# FDI label conversion
# ---------------------------------------------------------------------------

def fdi_to_gt_label(fdi: int) -> int:
    """Convert a single FDI tooth number to the TGNet gt_seg_label.

    Returns -1 for gingiva (fdi == 0), else 0-15 for the 16 teeth.

    Upper jaw: 11-18 → 0-7,  21-28 → 8-15
    Lower jaw: 41-48 → 0-7,  31-38 → 8-15
    """
    if fdi == 0:
        return -1
    quadrant = fdi // 10
    tooth_num = fdi % 10 - 1      # 0-7 within quadrant
    if quadrant in (1, 4):        # right side → class 0-7
        return tooth_num
    elif quadrant in (2, 3):      # left side  → class 8-15
        return tooth_num + 8
    return -1                     # unknown


def fdi_array_to_labels(fdi_arr: np.ndarray) -> np.ndarray:
    """Vectorised FDI → gt_seg_label conversion. Returns int32 array."""
    labels = np.full_like(fdi_arr, -1, dtype=np.int32)
    for fdi in np.unique(fdi_arr):
        labels[fdi_arr == fdi] = fdi_to_gt_label(int(fdi))
    return labels


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TeethSegDataset(Dataset):
    """
    Loads Teeth3DS OBJ + JSON pairs for TGNet Stage 1 training.

    Each item is a dict with keys expected by FpsGroupingNetworkModel.step():
        feat         — (6, N_POINTS) float32  [X, Y, Z, Nx, Ny, Nz]  normalised
        gt_seg_label — (1, N_POINTS) int32   values in {-1, 0..15}

    For BDL training, two extra keys are added:
        mesh_path    — str, path to OBJ file
        aug_obj      — open3d.geometry.TriangleMesh (the processed mesh)
    """

    def __init__(
        self,
        root: str,
        split_file: Optional[str] = None,
        npoints: int = N_POINTS,
        augment: bool = True,
        bdl_mode: bool = False,
    ):
        self.root = Path(root)
        self.npoints = npoints
        self.augment = augment
        self.bdl_mode = bdl_mode

        self.scan_paths = self._discover(split_file)
        print(f"  Found {len(self.scan_paths)} scans under {root}")

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def _discover(self, split_file: Optional[str]) -> List[Tuple[Path, Path]]:
        if split_file is not None:
            ids = Path(split_file).read_text().splitlines()
            ids = [i.strip() for i in ids if i.strip()]
            pairs = []
            for pid in ids:
                for jaw in ("upper", "lower"):
                    obj = self.root / pid / f"{pid}_{jaw}.obj"
                    jsn = self.root / pid / f"{pid}_{jaw}.json"
                    if obj.exists() and jsn.exists():
                        pairs.append((obj, jsn))
            return pairs

        # Auto-discover all OBJ+JSON pairs
        pairs = []
        for obj in sorted(self.root.glob("**/*.obj")):
            jsn = obj.with_suffix(".json")
            if jsn.exists():
                pairs.append((obj, jsn))
        return pairs

    # ------------------------------------------------------------------
    # Load helpers
    # ------------------------------------------------------------------

    def _load_mesh(self, obj_path: Path) -> o3d.geometry.TriangleMesh:
        mesh = o3d.io.read_triangle_mesh(str(obj_path))
        mesh.remove_duplicated_vertices()
        mesh.remove_degenerate_triangles()
        mesh.remove_unreferenced_vertices()
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()
        return mesh

    def _normalise(
        self,
        xyz: np.ndarray,
    ) -> Tuple[np.ndarray, float, float]:
        """
        TGNet normalisation:
          1. Centre by mean (all 3 axes)
          2. Scale so Y-span → 1.8 (range [-0.8, 1.0])
        Returns (xyz_norm, scale_factor, min_y_centred)
        """
        mean = xyz.mean(axis=0)
        centered = xyz - mean
        y_range = centered[:, 1].max() - centered[:, 1].min()
        if y_range < 1e-6:
            y_range = 1.0
        scale = 1.8 / y_range
        min_y_c = centered[:, 1].min()
        xyz_norm = centered * scale
        # shift Y to [-0.8, 1.0]
        xyz_norm[:, 1] = xyz_norm[:, 1] - min_y_c * scale - 0.8
        return xyz_norm, scale, min_y_c

    def _augment(self, xyz: np.ndarray, normals: np.ndarray):
        """Light augmentation: random rotation around Y-axis + small jitter."""
        theta = np.random.uniform(0, 2 * np.pi)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
        xyz = xyz @ R.T
        normals = normals @ R.T
        xyz += np.random.randn(*xyz.shape).astype(np.float32) * 0.002
        return xyz, normals

    def _sample(
        self,
        xyz: np.ndarray,
        normals: np.ndarray,
        labels: np.ndarray,
        n: int,
    ):
        """Random sampling (or padding by repeat) to exactly n points."""
        total = xyz.shape[0]
        if total >= n:
            idx = np.random.choice(total, n, replace=False)
        else:
            idx = np.concatenate([
                np.arange(total),
                np.random.choice(total, n - total, replace=True),
            ])
        return xyz[idx], normals[idx], labels[idx]

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------

    def __getitem__(self, index: int) -> dict:
        obj_path, jsn_path = self.scan_paths[index]

        # Load mesh
        mesh = self._load_mesh(obj_path)
        xyz_raw = np.asarray(mesh.vertices, dtype=np.float32)
        normals  = np.asarray(mesh.vertex_normals, dtype=np.float32)

        # Load FDI labels → gt_seg_label
        with open(jsn_path) as f:
            ann = json.load(f)
        fdi_arr = np.array(ann["labels"], dtype=np.int32)
        gt_labels = fdi_array_to_labels(fdi_arr)   # -1 or 0-15

        # Normalise coordinates
        xyz_norm, _scale, _min_y = self._normalise(xyz_raw)
        normals_norm = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8)

        # Augmentation
        if self.augment:
            xyz_norm, normals_norm = self._augment(xyz_norm, normals_norm)

        # Sample to fixed point count
        xyz_s, nrm_s, lbl_s = self._sample(xyz_norm, normals_norm, gt_labels, self.npoints)

        # Feature tensor: (6, N)
        feat = np.concatenate([xyz_s, nrm_s], axis=1).T.astype(np.float32)   # (6, N)
        gt_seg = lbl_s.reshape(1, -1).astype(np.int32)                        # (1, N)

        item = {
            "feat":         torch.from_numpy(feat),
            "gt_seg_label": torch.from_numpy(gt_seg),
        }

        if self.bdl_mode:
            # BDL needs the raw mesh object (for boundary cache) + its path
            item["mesh_path"] = str(obj_path)
            item["aug_obj"]   = mesh     # pass-through for boundary sampler

        return item

    def __len__(self) -> int:
        return len(self.scan_paths)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(state: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_latest_checkpoint(ckpt_dir: Path) -> Optional[Path]:
    ckpts = sorted(ckpt_dir.glob("epoch_*.pt"))
    return ckpts[-1] if ckpts else None


# ---------------------------------------------------------------------------
# FPS training
# ---------------------------------------------------------------------------

def build_fps_config(args) -> dict:
    """Minimal config dict expected by FpsGroupingNetworkModel."""
    return {
        # ---- optimiser ----
        "optimizer":         "adam",
        "learning_rate":     args.lr,
        "checkpoint_path":   str(Path(args.ckpt_fps) / "fps_best"),

        # ---- model / grouping ----
        "input_feat_dim":    6,      # xyz + normals
        "up_ratio":          4,
        "num_classes":       N_CLASSES,

        # ---- loss weights ----
        "class_weight":      1.0,
        "offset_weight":     1.0,
        "chamfer_weight":    1.0,

        # ---- scheduler ----
        "scheduler":         "cosine",
        "max_epoch":         args.fps_epochs,
    }


def train_fps(args):
    """Train the FPS model for Stage 1a."""
    print("\n" + "="*60)
    print("Stage 1a: FPS Grouping Network")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Dataset
    train_ds = TeethSegDataset(
        root=args.data_root,
        split_file=args.train_split,
        npoints=args.npoints,
        augment=not args.no_augment,
        bdl_mode=False,
    )
    val_ds = TeethSegDataset(
        root=args.data_root,
        split_file=args.val_split,
        npoints=args.npoints,
        augment=False,
        bdl_mode=False,
    ) if args.val_split else None

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    ) if val_ds else None

    print(f"Train: {len(train_ds)} scans  |  Val: {len(val_ds) if val_ds else 0} scans")

    # Model
    cfg = build_fps_config(args)
    model = FpsGroupingNetworkModel(cfg).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    ckpt_dir    = Path(args.ckpt_fps)
    start_epoch = 1
    best_loss   = float("inf")

    # Resume
    if args.resume:
        latest = load_latest_checkpoint(ckpt_dir)
        if latest:
            print(f"Resuming from {latest}")
            state = torch.load(latest, map_location=device)
            model.load_state_dict(state["model"])
            model.optimizer.load_state_dict(state["optimizer"])
            start_epoch = state["epoch"] + 1
            best_loss   = state.get("best_loss", float("inf"))
        else:
            print("No checkpoint found — starting from scratch.")

    # Training loop
    for epoch in range(start_epoch, args.fps_epochs + 1):
        t0 = time.time()

        # --- train epoch ---
        model.train()
        tr_loss = 0.0
        for batch in train_loader:
            # Move tensors to device
            batch_item = {
                "feat":         batch["feat"].to(device),
                "gt_seg_label": batch["gt_seg_label"].to(device),
            }
            loss_dict = model.step(batch_item, is_train=True)
            tr_loss  += loss_dict["loss_sum"]

        tr_loss /= max(len(train_loader), 1)

        # --- val epoch ---
        va_loss_str = ""
        monitor = tr_loss
        if val_loader:
            model.eval()
            va_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    batch_item = {
                        "feat":         batch["feat"].to(device),
                        "gt_seg_label": batch["gt_seg_label"].to(device),
                    }
                    loss_dict = model.step(batch_item, is_train=False)
                    va_loss  += loss_dict["loss_sum"]
            va_loss /= max(len(val_loader), 1)
            va_loss_str = f"  val={va_loss:.5f}"
            monitor = va_loss

        elapsed = time.time() - t0
        print(f"[{epoch:3d}/{args.fps_epochs}]  train={tr_loss:.5f}{va_loss_str}  ({elapsed:.1f}s)")

        # Checkpoint
        is_best   = monitor < best_loss
        best_loss = min(best_loss, monitor)
        state = {
            "epoch":     epoch,
            "model":     model.state_dict(),
            "optimizer": model.optimizer.state_dict(),
            "best_loss": best_loss,
        }
        save_checkpoint(state, ckpt_dir / f"epoch_{epoch:04d}.pt")
        if is_best:
            save_checkpoint(state, ckpt_dir / "best.pt")
            # Also save in .h5 format expected by BDL loader
            model.save()
            print(f"  → New best  (loss={best_loss:.5f})")

    print(f"\nFPS training complete.  Best checkpoint: {ckpt_dir / 'best.pt'}")


# ---------------------------------------------------------------------------
# BDL training
# ---------------------------------------------------------------------------

def build_bdl_config(args) -> dict:
    """Config dict for BdlGroupingNetworkModel."""
    fps_ckpt = str(Path(args.ckpt_fps) / "fps_best")
    return {
        # ---- FPS checkpoint (loaded in __init__) ----
        "fps_checkpoint_path":   fps_ckpt,

        # ---- optimiser ----
        "optimizer":             "adam",
        "learning_rate":         args.lr,
        "checkpoint_path":       str(Path(args.ckpt_bdl) / "bdl_best"),

        # ---- BDL-specific ----
        "input_feat_dim":        6,
        "num_classes":           N_CLASSES,
        "bdl_ratio":             0.5,       # boundary sampling ratio
        "cache_dir":             args.cache_dir,

        # ---- loss weights ----
        "class_weight":          1.0,
        "offset_weight":         1.0,
        "chamfer_weight":        1.0,

        # ---- scheduler ----
        "scheduler":             "cosine",
        "max_epoch":             args.bdl_epochs,
    }


def train_bdl(args):
    """Train the BDL refinement model for Stage 1b."""
    print("\n" + "="*60)
    print("Stage 1b: BDL Boundary Grouping Network")
    print("="*60)

    # Verify FPS checkpoint exists
    fps_h5 = Path(args.ckpt_fps) / "fps_best.h5"
    if not fps_h5.exists():
        raise FileNotFoundError(
            f"FPS checkpoint not found at {fps_h5}.\n"
            "Run 'python train_tgnet.py fps ...' first."
        )
    print(f"Using FPS checkpoint: {fps_h5}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # BDL dataset needs bdl_mode=True (adds mesh_path + aug_obj)
    train_ds = TeethSegDataset(
        root=args.data_root,
        split_file=args.train_split,
        npoints=args.npoints,
        augment=not args.no_augment,
        bdl_mode=True,
    )
    val_ds = TeethSegDataset(
        root=args.data_root,
        split_file=args.val_split,
        npoints=args.npoints,
        augment=False,
        bdl_mode=True,
    ) if args.val_split else None

    # BDL uses batch_size=1 (boundary cache built per scan)
    train_loader = DataLoader(
        train_ds,
        batch_size=1,
        shuffle=True,
        num_workers=0,        # must be 0; BDL builds cache during step()
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    ) if val_ds else None

    print(f"Train: {len(train_ds)} scans  |  Val: {len(val_ds) if val_ds else 0} scans")

    # Model — BDL loads FPS weights internally
    cfg = build_bdl_config(args)
    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    model = BdlGroupingNetworkModel(cfg).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    ckpt_dir    = Path(args.ckpt_bdl)
    start_epoch = 1
    best_loss   = float("inf")

    if args.resume:
        latest = load_latest_checkpoint(ckpt_dir)
        if latest:
            print(f"Resuming from {latest}")
            state = torch.load(latest, map_location=device)
            model.load_state_dict(state["model"])
            model.optimizer.load_state_dict(state["optimizer"])
            start_epoch = state["epoch"] + 1
            best_loss   = state.get("best_loss", float("inf"))
        else:
            print("No checkpoint found — starting from scratch.")

    for epoch in range(start_epoch, args.bdl_epochs + 1):
        t0 = time.time()

        # --- train epoch ---
        model.train()
        tr_loss = 0.0
        for batch in train_loader:
            batch_item = {
                "feat":         batch["feat"].to(device),
                "gt_seg_label": batch["gt_seg_label"].to(device),
                "mesh_path":    batch["mesh_path"][0],     # str (batch_size=1)
                "aug_obj":      batch["aug_obj"],           # mesh object
            }
            loss_dict = model.step(batch_item, is_train=True)
            tr_loss  += loss_dict["loss_sum"]

        tr_loss /= max(len(train_loader), 1)

        # --- val epoch ---
        va_loss_str = ""
        monitor = tr_loss
        if val_loader:
            model.eval()
            va_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    batch_item = {
                        "feat":         batch["feat"].to(device),
                        "gt_seg_label": batch["gt_seg_label"].to(device),
                        "mesh_path":    batch["mesh_path"][0],
                        "aug_obj":      batch["aug_obj"],
                    }
                    loss_dict = model.step(batch_item, is_train=False)
                    va_loss  += loss_dict["loss_sum"]
            va_loss /= max(len(val_loader), 1)
            va_loss_str = f"  val={va_loss:.5f}"
            monitor = va_loss

        elapsed = time.time() - t0
        print(f"[{epoch:3d}/{args.bdl_epochs}]  train={tr_loss:.5f}{va_loss_str}  ({elapsed:.1f}s)")

        is_best   = monitor < best_loss
        best_loss = min(best_loss, monitor)
        state = {
            "epoch":     epoch,
            "model":     model.state_dict(),
            "optimizer": model.optimizer.state_dict(),
            "best_loss": best_loss,
        }
        save_checkpoint(state, ckpt_dir / f"epoch_{epoch:04d}.pt")
        if is_best:
            save_checkpoint(state, ckpt_dir / "best.pt")
            model.save()
            print(f"  → New best  (loss={best_loss:.5f})")

    print(f"\nBDL training complete.  Best checkpoint: {ckpt_dir / 'best.pt'}")


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train TGNet Stage 1 (FPS + BDL)")
    p.add_argument(
        "stage",
        choices=["fps", "bdl", "both"],
        help="Which stage to train: 'fps', 'bdl', or 'both' (fps then bdl)",
    )
    p.add_argument("--data-root",    required=True,  help="Root of Teeth3DS dataset")
    p.add_argument("--train-split",  default=None,   help="Split file with training scan IDs")
    p.add_argument("--val-split",    default=None,   help="Split file with validation scan IDs")
    p.add_argument("--ckpt-fps",     default="checkpoints/tgnet_fps",
                   help="Directory for FPS checkpoints")
    p.add_argument("--ckpt-bdl",     default="checkpoints/tgnet_bdl",
                   help="Directory for BDL checkpoints")
    p.add_argument("--cache-dir",    default="checkpoints/bdl_cache",
                   help="Directory for BDL boundary-point cache (.npy files)")
    p.add_argument("--fps-epochs",   type=int, default=100)
    p.add_argument("--bdl-epochs",   type=int, default=100)
    p.add_argument("--batch-size",   type=int, default=2,
                   help="Batch size for FPS (BDL always uses 1)")
    p.add_argument("--npoints",      type=int, default=N_POINTS)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--num-workers",  type=int, default=4)
    p.add_argument("--no-augment",   action="store_true")
    p.add_argument("--resume",       action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.stage in ("fps", "both"):
        train_fps(args)

    if args.stage in ("bdl", "both"):
        train_bdl(args)


if __name__ == "__main__":
    main()
