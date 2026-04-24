"""
run_pipeline.py
---------------
Command-line entry point for the combined dental landmark pipeline.

Usage
-----
Single scan:
    python run_pipeline.py scan.obj

Specify checkpoints explicitly:
    python run_pipeline.py scan.obj \\
        --fps-ckpt  checkpoints/tgnet_fps.pth \\
        --bdl-ckpt  checkpoints/tgnet_bdl.pth \\
        --lm-ckpt   checkpoints/landmarks_full.ckpt

Batch mode (process an entire folder):
    python run_pipeline.py --batch data/teeth3ds/01A6HAN6/

Memory-saving mode (use fewer points; for RTX 3050 4 GB):
    python run_pipeline.py scan.obj --crop-k 8000

All outputs are written to  data/output/<scan_stem>/
"""

import argparse
import sys
from pathlib import Path

# ── make all sub-packages importable ───────────────────────────────────────
_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "stage1_segmentation"))
sys.path.insert(0, str(_ROOT / "stage2_landmarks"))
# ── built-in-place CUDA extensions ─────────────────────────────────────────
sys.path.insert(0, str(_ROOT / "extensions" / "tgnet_ops"))
sys.path.insert(0, str(_ROOT / "extensions" / "teethland_ops"))


def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Dental landmark detection pipeline (TGNet + 3DTeethLand)"
    )
    p.add_argument(
        "scan",
        nargs="?",
        type=Path,
        help="Path to input .obj scan file (or use --batch for a folder)",
    )
    p.add_argument(
        "--batch",
        type=Path,
        default=None,
        metavar="DIR",
        help="Process all .obj files in DIR recursively",
    )
    p.add_argument(
        "--fps-ckpt",
        type=Path,
        default=Path("checkpoints/CGIP_TGN_checkpoints/ckpts(new)/tgnet_fps.h5"),
        help="TGNet Stage-1 FPS checkpoint",
    )
    p.add_argument(
        "--bdl-ckpt",
        type=Path,
        default=Path("checkpoints/CGIP_TGN_checkpoints/ckpts(new)/tgnet_bdl.h5"),
        help="TGNet Stage-2 BDL checkpoint",
    )
    p.add_argument(
        "--lm-ckpt",
        type=Path,
        default=Path("checkpoints/Teethland-checkpoints/landmarks_full.ckpt"),
        help="3DTeethLand LandmarkNet checkpoint",
    )
    p.add_argument(
        "--crop-k",
        type=int,
        default=12_000,
        metavar="K",
        help=(
            "Points per tooth crop sent to LandmarkNet (default: 12000). "
            "Lower to 8000 if GPU runs out of memory."
        ),
    )
    p.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run inference on (default: cuda)",
    )
    p.add_argument(
        "--marker-radius",
        type=float,
        default=1.2,
        metavar="MM",
        help="Landmark octahedron radius in mm for the PLY output (default: 1.2)",
    )
    return p.parse_args()


def check_checkpoints(args: argparse.Namespace) -> None:
    missing = []
    for attr, name in [
        ("fps_ckpt", "TGNet FPS"),
        ("bdl_ckpt", "TGNet BDL"),
        ("lm_ckpt",  "LandmarkNet"),
    ]:
        path = getattr(args, attr)
        if not path.exists():
            missing.append(f"  {name}: {path}")
    if missing:
        print("[ERROR] Missing checkpoint files:")
        for m in missing:
            print(m)
        print()
        print("  Place checkpoint files under checkpoints/ and see checkpoints/README.md")
        sys.exit(1)


def process_single(pipeline, obj_path: Path, out_dir: Path,
                   marker_radius: float = 1.2) -> None:
    print(f"Processing: {obj_path.name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    stem     = obj_path.stem
    out_path = out_dir / (stem + "_landmarks.json")
    pipeline.run_and_save(str(obj_path), str(out_path))

    from visualize_scan import (
        read_obj_with_faces, load_fdi_labels, build_vertex_colours,
        load_landmarks, split_mesial_distal, build_landmark_geometry,
        write_ply, write_split_layers, _GINGIVA_COLOUR,
    )
    import numpy as np

    ann_path = obj_path.with_suffix(".json")
    verts, faces = read_obj_with_faces(obj_path)

    if ann_path.exists():
        labels = load_fdi_labels(ann_path)
        if len(labels) == len(verts):
            vert_colours = build_vertex_colours(labels)
        else:
            vert_colours = np.full((len(verts), 3), _GINGIVA_COLOUR, dtype=np.uint8)
    else:
        vert_colours = np.full((len(verts), 3), _GINGIVA_COLOUR, dtype=np.uint8)

    # mesh only — no landmarks (for manual overlay in MeshLab)
    empty_v = np.empty((0, 3), np.float32)
    empty_c = np.empty((0, 3), np.uint8)
    empty_f = np.empty((0, 3), np.int32)
    mesh_ply = out_dir / (stem + "_mesh.ply")
    write_ply(mesh_ply, verts, vert_colours, faces, empty_v, empty_c, empty_f)
    print(f"Saved mesh PLY to {mesh_ply}")

    # mesh + all landmark octahedra combined
    raw_lms   = load_landmarks(out_path)
    landmarks = split_mesial_distal(raw_lms)
    extra_v, extra_f, extra_c = build_landmark_geometry(landmarks, marker_radius)
    colored_ply = out_dir / (stem + "_colored.ply")
    write_ply(colored_ply, verts, vert_colours, faces, extra_v, extra_c, extra_f)
    print(f"Saved colored PLY to {colored_ply}")

    # one PLY per landmark class
    print("Saving per-class landmark layers:")
    write_split_layers(out_dir / stem, landmarks, marker_radius)


def main() -> None:
    args = build_args()

    if args.scan is None and args.batch is None:
        print("[ERROR] Provide a scan file or --batch directory.")
        sys.exit(1)

    check_checkpoints(args)

    # Lazy import to avoid slow startup when just checking --help
    from pipeline.combined_pipeline import CombinedDentalPipeline

    print("Loading models...")
    pipeline = CombinedDentalPipeline(
        tgnet_fps_ckpt=str(args.fps_ckpt),
        tgnet_bdl_ckpt=str(args.bdl_ckpt),
        landmark_ckpt=str(args.lm_ckpt),
        device=args.device,
        crop_k=args.crop_k,
    )
    print("Models loaded.\n")

    if args.batch:
        obj_files = sorted(args.batch.rglob("*.obj"))
        if not obj_files:
            print(f"[WARNING] No .obj files found under {args.batch}")
            sys.exit(0)
        print(f"Batch mode: {len(obj_files)} scans found.\n")
        for obj_path in obj_files:
            out_dir = _ROOT / "data" / "output" / obj_path.stem
            process_single(pipeline, obj_path, out_dir, args.marker_radius)
    else:
        obj_path = args.scan
        out_dir  = _ROOT / "data" / "output" / obj_path.stem
        process_single(pipeline, obj_path, out_dir, args.marker_radius)

    print("\nDone.")


if __name__ == "__main__":
    main()
