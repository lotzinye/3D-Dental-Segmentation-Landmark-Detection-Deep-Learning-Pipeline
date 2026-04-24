"""
visualize_scan.py
-----------------
Produces a coloured PLY file from a Teeth3DS jaw scan.

Mesh vertices are coloured by FDI tooth label (from the ground-truth JSON).
Landmark points (from the pipeline _landmarks.json) are overlaid as small
3D octahedra — solid geometry that looks the same from every angle:

    Mesial        red
    Distal        green
    Cusp          blue
    InnerPoint    yellow
    OuterPoint    cyan
    FacialPoint   magenta

Usage:
    python visualize_scan.py path/to/scan.obj
    python visualize_scan.py path/to/scan.obj --out my_output.ply
    python visualize_scan.py path/to/scan.obj --radius 1.5
    python visualize_scan.py path/to/scan.obj --no-landmarks

The annotation JSON ({stem}.json) must exist alongside the OBJ.
The landmark JSON ({stem}_landmarks.json) is loaded automatically if present.

Open the output .ply in MeshLab:
    File > Import Mesh  (no extra settings needed — markers are solid geometry)
"""

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Colour tables
# ---------------------------------------------------------------------------

# 16-colour qualitative palette for teeth (vivid & distinct)
_TOOTH_PALETTE = np.array([
    [ 31, 119, 180],   # blue
    [255, 127,  14],   # orange
    [ 44, 160,  44],   # green
    [214,  39,  40],   # red
    [148, 103, 189],   # purple
    [140,  86,  75],   # brown
    [227, 119, 194],   # pink
    [127, 127, 127],   # mid-grey
    [188, 189,  34],   # olive
    [ 23, 190, 207],   # teal
    [174, 199, 232],   # light blue
    [255, 187, 120],   # light orange
    [152, 223, 138],   # light green
    [255, 152, 150],   # light red
    [197, 176, 213],   # light purple
    [255, 221,  99],   # yellow
], dtype=np.uint8)

_GINGIVA_COLOUR = np.array([180, 180, 180], dtype=np.uint8)

# Landmark class → RGB colour
_LANDMARK_COLOURS = {
    "Mesial":       np.array([255,   0,   0], dtype=np.uint8),   # red
    "Distal":       np.array([  0, 210,   0], dtype=np.uint8),   # green
    "Cusp":         np.array([  0,   0, 255], dtype=np.uint8),   # blue
    "InnerPoint":   np.array([255, 220,   0], dtype=np.uint8),   # yellow
    "OuterPoint":   np.array([  0, 220, 255], dtype=np.uint8),   # cyan
    "FacialPoint":  np.array([255,   0, 255], dtype=np.uint8),   # magenta
    "MesialDistal": np.array([255, 100,   0], dtype=np.uint8),   # orange fallback
}


# ---------------------------------------------------------------------------
# OBJ reader
# ---------------------------------------------------------------------------

def read_obj_with_faces(obj_path: Path):
    """
    Parse OBJ and return vertices (N,3) float32 + triangles (F,3) int32.
    Handles the Teeth3DS format  v x y z r g b  (7 values per vertex line).
    """
    verts, faces = [], []
    with open(obj_path, "r") as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("v "):
                p = line.split()
                verts.append([float(p[1]), float(p[2]), float(p[3])])
            elif line.startswith("f "):
                p = line.split()[1:]
                idxs = [int(tok.split("/")[0]) - 1 for tok in p]
                for k in range(1, len(idxs) - 1):
                    faces.append([idxs[0], idxs[k], idxs[k + 1]])
    return np.array(verts, dtype=np.float32), np.array(faces, dtype=np.int32)


# ---------------------------------------------------------------------------
# Annotation loader
# ---------------------------------------------------------------------------

def load_fdi_labels(json_path: Path) -> np.ndarray:
    with open(json_path) as f:
        ann = json.load(f)
    return np.array(ann["labels"], dtype=np.int32)


def build_vertex_colours(labels: np.ndarray) -> np.ndarray:
    unique_fdis = sorted(set(int(l) for l in labels if l != 0))
    fdi_to_idx  = {fdi: i for i, fdi in enumerate(unique_fdis)}
    colours = np.empty((len(labels), 3), dtype=np.uint8)
    colours[:] = _GINGIVA_COLOUR
    for i, label in enumerate(labels):
        if label != 0:
            colours[i] = _TOOTH_PALETTE[fdi_to_idx[int(label)] % len(_TOOTH_PALETTE)]
    return colours


# ---------------------------------------------------------------------------
# Landmark loader + Mesial/Distal splitter
# ---------------------------------------------------------------------------

def load_landmarks(lm_path: Path):
    with open(lm_path) as f:
        return json.load(f)["landmarks"]


def split_mesial_distal(landmarks):
    """
    Split 'MesialDistal' entries into 'Mesial' / 'Distal' using x-coord.
    Quadrants 1 & 4 (right side, x>0): smaller x = Mesial.
    Quadrants 2 & 3 (left side,  x<0): larger  x = Mesial.
    """
    from collections import defaultdict
    md_by_tooth, other = defaultdict(list), []
    for lm in landmarks:
        (md_by_tooth[lm["fdi_tooth"]] if lm["class"] == "MesialDistal" else other).append(lm)

    resolved = list(other)
    for fdi, group in md_by_tooth.items():
        reverse = (fdi // 10) in (2, 3)
        for k, lm in enumerate(sorted(group, key=lambda l: l["coord"][0], reverse=reverse)):
            resolved.append({**lm, "class": "Mesial" if k == 0 else "Distal"})
    return resolved


# ---------------------------------------------------------------------------
# 3-D marker geometry
# ---------------------------------------------------------------------------

# Canonical octahedron: 6 vertices at ±1 on each axis, 8 triangular faces.
_OCT_VERTS = np.array([
    [ 1,  0,  0],
    [-1,  0,  0],
    [ 0,  1,  0],
    [ 0, -1,  0],
    [ 0,  0,  1],
    [ 0,  0, -1],
], dtype=np.float32)

_OCT_FACES = np.array([
    [0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4],
    [0, 3, 5], [3, 1, 5], [1, 2, 5], [2, 0, 5],
], dtype=np.int32)


def make_octahedron(center, radius, colour):
    """
    Return (verts (6,3), faces (8,3), colours (6,3)) for one octahedron
    centred at `center` with the given `radius` and uniform `colour`.
    """
    v = _OCT_VERTS * radius + np.array(center, dtype=np.float32)
    c = np.tile(colour, (6, 1))
    return v, _OCT_FACES.copy(), c


def build_landmark_geometry(landmarks, radius):
    """
    Convert a list of landmark dicts into concatenated octahedron geometry.
    Returns:
        extra_verts   (M*6, 3) float32
        extra_faces   (M*8, 3) int32   — local indices, caller must offset
        extra_colours (M*6, 3) uint8
    """
    all_v, all_f, all_c = [], [], []
    v_offset = 0
    for lm in landmarks:
        colour = _LANDMARK_COLOURS.get(lm["class"],
                                       np.array([255, 255, 255], dtype=np.uint8))
        v, f, c = make_octahedron(lm["coord"], radius, colour)
        all_v.append(v)
        all_f.append(f + v_offset)
        all_c.append(c)
        v_offset += 6

    if not all_v:
        return (np.empty((0, 3), np.float32),
                np.empty((0, 3), np.int32),
                np.empty((0, 3), np.uint8))
    return (np.concatenate(all_v),
            np.concatenate(all_f),
            np.concatenate(all_c))


# ---------------------------------------------------------------------------
# Per-class split layer writer
# ---------------------------------------------------------------------------

def write_split_layers(stem: Path, landmarks, radius: float) -> None:
    """
    Write one PLY per landmark class containing only that class's octahedra.
    Files are named  {stem}_{ClassName}.ply  (e.g. scan_Mesial.ply).
    Load them as separate layers in MeshLab and toggle the eye icon to show/hide each class.
    """
    from collections import defaultdict
    by_class = defaultdict(list)
    for lm in landmarks:
        by_class[lm["class"]].append(lm)

    empty_v = np.empty((0, 3), np.float32)
    empty_c = np.empty((0, 3), np.uint8)
    empty_f = np.empty((0, 3), np.int32)

    for cls_name, cls_lms in by_class.items():
        out_path = Path(str(stem) + f"_{cls_name}.ply")
        ev, ef, ec = build_landmark_geometry(cls_lms, radius)
        write_ply(out_path, empty_v, empty_c, empty_f, ev, ec, ef)
        print(f"  {cls_name}: {len(cls_lms)} markers -> {out_path.name}")


# ---------------------------------------------------------------------------
# PLY writer
# ---------------------------------------------------------------------------

def write_ply(out_path, mesh_v, mesh_c, mesh_f, extra_v, extra_c, extra_f):
    """
    Binary-little-endian PLY.
    mesh_v / mesh_c : (N,3) mesh vertices + colours
    mesh_f          : (F,3) mesh triangles (index into mesh_v)
    extra_v / extra_c / extra_f : landmark octahedra (indices are local,
                                  caller already offset by len(mesh_v))
    """
    n_v = len(mesh_v)
    total_v = n_v + len(extra_v)
    total_f = len(mesh_f) + len(extra_f)

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {total_v}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        f"element face {total_f}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    )

    with open(out_path, "wb") as fh:
        fh.write(header.encode("ascii"))

        # vertices: mesh first, then landmark octahedra
        for i in range(n_v):
            fh.write(struct.pack("<fff", *mesh_v[i]))
            fh.write(struct.pack("<BBB", *mesh_c[i]))
        for i in range(len(extra_v)):
            fh.write(struct.pack("<fff", *extra_v[i]))
            fh.write(struct.pack("<BBB", *extra_c[i]))

        # faces: mesh triangles
        for tri in mesh_f:
            fh.write(struct.pack("<Biii", 3, int(tri[0]), int(tri[1]), int(tri[2])))

        # faces: landmark octahedra (indices already offset by n_v)
        for tri in extra_f:
            fh.write(struct.pack("<Biii", 3,
                                 int(tri[0]) + n_v,
                                 int(tri[1]) + n_v,
                                 int(tri[2]) + n_v))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Create a coloured PLY from a Teeth3DS scan.")
    parser.add_argument("obj_path", help="Path to the .obj file")
    parser.add_argument("--out", default=None,
                        help="Output .ply path (default: {stem}_colored.ply)")
    parser.add_argument("--radius", type=float, default=1.2,
                        help="Landmark marker radius in mm (default: 1.2)")
    parser.add_argument("--no-landmarks", action="store_true",
                        help="Skip landmark overlay")
    args = parser.parse_args()

    obj_path = Path(args.obj_path)
    stem     = obj_path.with_suffix("")
    ann_path = stem.with_suffix(".json")
    lm_path  = Path(str(stem) + "_landmarks.json")
    out_path = Path(args.out) if args.out else Path(str(stem) + "_colored.ply")

    # --- mesh ---
    print(f"Reading mesh:   {obj_path}")
    verts, faces = read_obj_with_faces(obj_path)
    print(f"  {len(verts):,} vertices, {len(faces):,} triangles")

    # --- FDI colours ---
    if not ann_path.exists():
        print(f"WARNING: no annotation JSON at {ann_path} — mesh will be grey")
        vert_colours = np.full((len(verts), 3), _GINGIVA_COLOUR, dtype=np.uint8)
    else:
        print(f"Reading labels: {ann_path}")
        labels = load_fdi_labels(ann_path)
        if len(labels) != len(verts):
            sys.exit(f"ERROR: label count {len(labels)} != vertex count {len(verts)}")
        vert_colours = build_vertex_colours(labels)
        unique_fdis = sorted(set(int(l) for l in labels if l != 0))
        print(f"  FDI teeth found: {unique_fdis}")

    # --- landmark octahedra ---
    extra_v = np.empty((0, 3), np.float32)
    extra_c = np.empty((0, 3), np.uint8)
    extra_f = np.empty((0, 3), np.int32)

    if not args.no_landmarks and lm_path.exists():
        print(f"Reading landmarks: {lm_path}  (marker radius={args.radius} mm)")
        raw_lms   = load_landmarks(lm_path)
        landmarks = split_mesial_distal(raw_lms)

        from collections import Counter
        print(f"  {len(landmarks)} landmarks: {dict(Counter(l['class'] for l in landmarks))}")

        extra_v, extra_f, extra_c = build_landmark_geometry(landmarks, args.radius)
        print(f"  Added {len(extra_v)//6} octahedra "
              f"({len(extra_v)} verts, {len(extra_f)} tris)")
    elif not args.no_landmarks:
        print(f"No landmark file found at {lm_path} — skipping.")

    # --- write ---
    print(f"Writing PLY:    {out_path}")
    write_ply(out_path, verts, vert_colours, faces, extra_v, extra_c, extra_f)
    print("Done.")
    print()
    print("Open in MeshLab:  File > Import Mesh  (markers are solid 3D geometry)")


if __name__ == "__main__":
    main()
