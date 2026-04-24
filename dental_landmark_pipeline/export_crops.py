"""
export_crops.py
---------------
Runs Stage 1 segmentation + coordinate bridge and saves each per-tooth
crop as a PLY point cloud in <out_dir>/.

Usage:
    python export_crops.py <obj_path> <out_dir>

Example:
    python export_crops.py data/teeth3ds_sample/01F4JV8X/01F4JV8X_upper.obj exports/crops/

Each output file is named:  fdi_<FDI>_crop.ply
Open them in MeshLab to inspect individual tooth crops.
"""

import sys
import os
import numpy as np
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "stage1_segmentation"))
sys.path.insert(0, str(_ROOT / "stage2_landmarks"))

import gen_utils as gu
from inference_pipeline import InferencePipeline
from pipeline.data_bridge import get_scale_factor

Z_SCORE_STD = 17.3281


def save_ply(path: str, xyz_mm: np.ndarray, normals: np.ndarray = None) -> None:
    """Write a point cloud to a binary PLY file."""
    n = xyz_mm.shape[0]
    has_normals = normals is not None

    with open(path, "wb") as f:
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {n}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
        )
        if has_normals:
            header += (
                "property float nx\n"
                "property float ny\n"
                "property float nz\n"
            )
        header += "end_header\n"
        f.write(header.encode("ascii"))

        if has_normals:
            data = np.hstack([xyz_mm, normals]).astype(np.float32)
        else:
            data = xyz_mm.astype(np.float32)
        f.write(data.tobytes())


def main(obj_path: str, out_dir: str, fps_ckpt: str, bdl_ckpt: str,
         device: str = "cuda", crop_k: int = 12_000) -> None:

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # --- Stage 1 ---
    seg = InferencePipeline(fps_ckpt=fps_ckpt, bdl_ckpt=bdl_ckpt, device=device)

    original_verts = gu.read_txt_obj_ls(obj_path, ret_mesh=False, use_tri_mesh=False)[0]
    orig_xyz_mm  = original_verts[:, :3]
    orig_normals = original_verts[:, 3:]

    scale_factor = get_scale_factor(orig_xyz_mm)
    jaw_mean     = orig_xyz_mm.mean(axis=0)
    jaw_norm     = (orig_xyz_mm - jaw_mean) / Z_SCORE_STD
    min_y_c      = float(orig_xyz_mm[:, 1].min()) - float(jaw_mean[1])

    seg_result   = seg.run(obj_path)
    jaw          = seg_result["jaw"]
    sampled_xyz  = seg_result["sampled_xyz"]
    labels       = np.array(seg_result["sampled_sem_labels"])

    print(f"Jaw: {jaw}  |  unique FDI labels: {sorted(set(int(l) for l in labels if l != 0))}")

    from sklearn.neighbors import KDTree
    norm_tree = KDTree(jaw_norm, leaf_size=16)

    unique_fdi = sorted(set(int(l) for l in labels if l != 0))
    for fdi_label in unique_fdi:
        tooth_mask = labels == fdi_label
        if tooth_mask.sum() < 10:
            continue

        tooth_norm_xyz = sampled_xyz[tooth_mask]
        centroid_tgnet = tooth_norm_xyz.mean(axis=0)
        centroid_norm  = ((centroid_tgnet + 0.8) * scale_factor + min_y_c) / Z_SCORE_STD

        crop_idxs     = norm_tree.query(centroid_norm[None], k=crop_k, return_distance=False)[0]
        crop_xyz_mm   = orig_xyz_mm[crop_idxs]
        crop_normals  = orig_normals[crop_idxs]

        out_path = str(Path(out_dir) / f"fdi_{fdi_label:02d}_crop.ply")
        save_ply(out_path, crop_xyz_mm, crop_normals)
        print(f"  FDI {fdi_label:02d} → {crop_idxs.shape[0]} pts  saved: {out_path}")

    print("Done.")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Export per-tooth KDTree crops as PLY files")
    p.add_argument("scan",    type=Path, help="Input .obj file")
    p.add_argument("out_dir", type=Path, help="Output directory for crop PLYs")
    p.add_argument("--fps-ckpt", type=Path,
                   default=Path("checkpoints/CGIP_TGN_checkpoints/ckpts(new)/tgnet_fps.h5"))
    p.add_argument("--bdl-ckpt", type=Path,
                   default=Path("checkpoints/CGIP_TGN_checkpoints/ckpts(new)/tgnet_bdl.h5"))
    p.add_argument("--crop-k", type=int, default=12_000)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = p.parse_args()

    main(str(args.scan), str(args.out_dir),
         str(args.fps_ckpt), str(args.bdl_ckpt),
         device=args.device, crop_k=args.crop_k)
