# dental_landmark_pipeline — Developer Guide

Operational reference for the codebase. For project background, results, and the FYP
publication, see the [**project README**](../README.md); for the full evaluation write-up, see
[**docs/RESULTS.md**](../docs/RESULTS.md).

A self-contained three-stage pipeline for 3D dental segmentation and landmark detection.

| Stage | Model | Task |
|---|---|---|
| 1 | PyTorch Point Transformer (+ boundary refinement) | Global jaw segmentation → per-vertex FDI labels |
| 2 | Coordinate-space bridge (no learned weights) | Reconcile Stage-1 and Stage-3 normalisation; per-tooth crop extraction |
| 3 | PyTorch Stratified Transformer | Per-tooth anatomical landmark detection (6 classes) |

**Output**: a JSON file of per-landmark `(x, y, z)` coordinates in mm, confidence scores, and
clinical FDI tooth numbers — plus eight colour-coded PLY visualisation layers.

---

## Environment

| Item | Version |
|---|---|
| Python | 3.10.11 |
| PyTorch | 2.10.0+cu126 |
| CUDA Toolkit | 12.6 |
| Reference GPU | RTX 3050 Laptop (Ampere sm_86, 4 GB) |
| Build tools | Visual Studio Build Tools (C++ workload) |

The pipeline runs on 4 GB of VRAM at `--crop-k 8000`. At the default `--crop-k 12000` budget for
roughly 6 GB.

---

## Setup

### 1 — Python dependencies

```bat
pip install -r requirements.txt
```

> PyTorch itself is deliberately **not** listed in `requirements.txt` — install the CUDA build
> matching your toolkit first, or the extension compile in step 2 will link against the wrong ABI.

### 2 — Build the CUDA extensions

```bat
build_extensions.bat
```

Compiles `tgnet_ops` and `teethland_ops` in place and installs a version-matched `torch-scatter`.
`torch-scatter` must match the exact PyTorch + CUDA build string, which is why it is installed
here rather than pinned in `requirements.txt`.

### 3 — Checkpoints

Already present in this repository. See [checkpoints/README.md](checkpoints/README.md).

| Weight | Path |
|---|---|
| TGNet FPS (Stage 1) | `checkpoints/CGIP_TGN_checkpoints/ckpts(new)/tgnet_fps.h5` |
| TGNet BDL (boundary refinement) | `checkpoints/CGIP_TGN_checkpoints/ckpts(new)/tgnet_bdl.h5` |
| 3DTeethLand LandmarkNet | `checkpoints/Teethland-checkpoints/landmarks_full.ckpt` |

The `.h5` files are standard PyTorch ZIP checkpoints — `torch.load()` reads them directly despite
the extension.

### 4 — Dataset

Download and layout instructions: [data/README_DATA.md](data/README_DATA.md).

```bat
python data\prepare_dataset.py
python data\prepare_dataset.py --verbose
```

Prints scans found, how many carry landmark annotations, and any layout errors.

> Landmark files use a **double underscore**: `{ID}_upper__kpt.json`. This is how the dataset is
> distributed and `prepare_dataset.py` expects it.

---

## Inference

```bat
:: single scan
python run_pipeline.py data\upper\01F4JV8X\01F4JV8X_upper.obj

:: bundled sample (no dataset download needed)
python run_pipeline.py data\teeth3ds_sample\teeth3ds_sample\01F4JV8X\01F4JV8X_upper.obj

:: batch — every .obj under a folder, recursively
python run_pipeline.py --batch data\upper\01F4JV8X\

:: low-VRAM mode
python run_pipeline.py scan.obj --crop-k 8000

:: CPU fallback (slow)
python run_pipeline.py scan.obj --device cpu
```

Outputs land in `data/output/<scan_stem>/`.

### `run_pipeline.py` options

| Flag | Default | Purpose |
|---|---|---|
| `scan` (positional) | — | Input `.obj` path; omit when using `--batch` |
| `--batch DIR` | — | Process all `.obj` files under `DIR` recursively |
| `--fps-ckpt` | `checkpoints/CGIP_TGN_checkpoints/ckpts(new)/tgnet_fps.h5` | Stage-1 FPS weights |
| `--bdl-ckpt` | `checkpoints/CGIP_TGN_checkpoints/ckpts(new)/tgnet_bdl.h5` | Boundary-refinement weights |
| `--lm-ckpt` | `checkpoints/Teethland-checkpoints/landmarks_full.ckpt` | LandmarkNet weights |
| `--crop-k K` | `12000` | Points per tooth crop into Stage 3. Lower on OOM. |
| `--device` | `cuda` | `cuda` or `cpu` |
| `--marker-radius MM` | `1.2` | Landmark octahedron radius in the PLY output |

---

## Other entry points

| Script | Purpose |
|---|---|
| `eval_miccai.py` | Both challenge protocols (MICCAI 2022 segmentation + 2024 landmarks) |
| `evaluate_pipeline.py` | Prediction-vs-GT landmark comparison — MRE, detection rate, SDR@{1.5, 2.0, 2.5, 4.0} mm |
| `visualize_scan.py` | Render a scan with landmark markers |
| `export_stages.py` | Dump every pipeline stage as inspectable PLY files |
| `export_crops.py` | Export per-tooth crops only |
| `run_ablation.py` | Ablation driver |
| `train_tgnet.py` | Train the Stage-1 segmentation network |
| `train_landmarknet.py` | Train the Stage-3 landmark network |
| `debug_lm.py` | Landmark-head debugging scratchpad |

### Evaluation

```bat
python eval_miccai.py --max-scans 240
python eval_miccai.py --skip-2024                 :: segmentation metrics only
python eval_miccai.py --skip-2022                 :: landmark metrics only
python eval_miccai.py --use-gt-seg                :: isolate Stage 3 from Stage-1 error
python eval_miccai.py --reuse                     :: reuse cached predictions
python eval_miccai.py --out results\eval.json
```

`--use-gt-seg` is the key diagnostic flag: it feeds ground-truth segmentation into Stage 3,
separating landmark-network error from segmentation error propagating downstream.

### Localisation metrics

```bat
python evaluate_pipeline.py --pred data\output\01F4JV8X_upper\01F4JV8X_upper_landmarks.json ^
                            --gt   data\3DTeethLand_landmarks_test\upper\01F4JV8X\01F4JV8X_upper__kpt.json

python evaluate_pipeline.py --auto-discover ^
                            --pred-dir data\output ^
                            --gt-dir   data\3DTeethLand_landmarks_test ^
                            --out      results\eval_results.json
```

### Stage inspection

```bat
python export_stages.py scan.obj exports\stages\
python export_stages.py scan.obj exports\stages\ --focus-fdi 16
```

| Flag | Default | Purpose |
|---|---|---|
| `--focus-fdi NN` | all teeth | Export Stage-2/3 intermediates for one FDI tooth only |
| `--crop-k K` | `12000` | Must match the inference setting to reproduce a run |
| `--device` | `cuda` | `cuda` or `cpu` |

Produces `stage1a` → `stage3d` PLY files. The `stage3b_candidates` →
`stage3c_dbscan` → `stage3d_tooth_landmarks` sequence is the most useful for diagnosing a
duplicate or missing landmark. See [docs/RESULTS.md §5](../docs/RESULTS.md#5-stage-by-stage-inspection).

### Visualisation

```bat
python visualize_scan.py scan.obj
python visualize_scan.py scan.obj --out render\ --radius 1.5
python visualize_scan.py scan.obj --no-landmarks
```

---

## Output format

Nine files per scan in `data/output/<scan_stem>/`:

| File | Contents |
|---|---|
| `{stem}_landmarks.json` | Landmark coordinates, classes, scores, FDI labels |
| `{stem}_mesh.ply` | FDI-coloured mesh, no landmarks |
| `{stem}_colored.ply` | FDI-coloured mesh + all landmark octahedra |
| `{stem}_Mesial.ply` | Mesial octahedra only — red |
| `{stem}_Distal.ply` | Distal octahedra only — green |
| `{stem}_Cusp.ply` | Cusp octahedra only — blue |
| `{stem}_InnerPoint.ply` | InnerPoint octahedra only — yellow |
| `{stem}_OuterPoint.ply` | OuterPoint octahedra only — cyan |
| `{stem}_FacialPoint.ply` | FacialPoint octahedra only — purple |

Per-class separation lets a clinician toggle individual landmark classes in a 3D viewer
(MeshLab, CloudCompare, ArcGIS) and inspect detection quality class-by-class.

```json
{
  "jaw": "upper",
  "landmarks": [
    { "class": "Cusp",   "coord": [3.21, 12.05, -5.43], "score": 0.93, "fdi_tooth": 16 },
    { "class": "Mesial", "coord": [1.10, 11.80, -4.90], "score": 0.88, "fdi_tooth": 16 }
  ]
}
```

Coordinates are in **millimetres, in the original scanner coordinate frame** — the same frame as
the input `.obj` and the `_kpt.json` ground truth, so predictions overlay the raw scan directly
with no transform.

---

## Project structure

```
dental_landmark_pipeline/
├── run_pipeline.py            ← main CLI entry point
├── eval_miccai.py             ← MICCAI 2022 + 2024 evaluation harness
├── evaluate_pipeline.py       ← MRE / SDR localisation metrics
├── visualize_scan.py          ← mesh + landmark rendering
├── export_stages.py           ← per-stage PLY dumps
├── export_crops.py            ← per-tooth crop export
├── run_ablation.py            ← ablation driver
├── train_tgnet.py             ← Stage-1 training
├── train_landmarknet.py       ← Stage-3 training
├── requirements.txt
├── build_extensions.bat       ← compile CUDA libs (run once)
│
├── pipeline/                  ← ★ original contribution — see pipeline/README.md
│   ├── combined_pipeline.py   ←   orchestration; applies the bridge end-to-end
│   ├── data_bridge.py         ←   scale-factor derivation, coordinate conversion
│   └── landmark_postprocess.py←   candidate filtering, DBSCAN, weighted centroids
│
├── stage1_segmentation/       ← Point Transformer segmentation (TGNet-derived)
│   ├── inference_pipeline.py
│   ├── gen_utils.py
│   ├── ops_utils.py
│   └── models/
│
├── stage2_landmarks/          ← Stratified Transformer landmark network
│   └── teethland/
│       ├── config/config.yaml
│       ├── models/landmarknet.py
│       └── nn/modules/stratified_transformer.py
│
├── extensions/
│   ├── tgnet_ops/             ← Stage-1 CUDA kernels (renamed from pointops)
│   └── teethland_ops/         ← Stage-3 CUDA kernels
│
├── data/                      ← datasets + prepare_dataset.py + outputs
├── checkpoints/               ← model weights
├── exports/                   ← stage-inspection PLY dumps
└── results/                   ← evaluation JSON
```

> Naming note: the directory `stage2_landmarks/` holds the **Stage 3** landmark network. The name
> predates the introduction of the coordinate-space bridge as an explicit stage; the report and
> both READMEs use the three-stage numbering (segmentation → bridge → landmarks).

---

## Troubleshooting

| Problem | Fix |
|---|---|
| CUDA OOM during Stage 1 | `--crop-k 8000` |
| CUDA OOM during Stage 3 | Also set `proposal_points: 8000` in `stage2_landmarks/teethland/config/config.yaml` |
| `tgnet_ops` import error | Re-run `build_extensions.bat` |
| `teethland_ops` import error | Re-run `build_extensions.bat` |
| `torch_scatter` not found | `pip install torch-scatter -f https://data.pyg.org/whl/torch-2.10.0+cu126.html` |
| Extension builds but crashes at runtime | PyTorch was reinstalled after the build — recompile with `build_extensions.bat` |
| `prepare_dataset.py` finds 0 landmark files | Check the **double underscore**: `{ID}_upper__kpt.json`, not `_kpt.json` |
| Duplicate Distal / Mesial landmarks | Known post-processing artefact — see [`FIX_MESIAL_DISTAL_LANDMARKS.md`](FIX_MESIAL_DISTAL_LANDMARKS.md) |
| Missing landmark on a terminal molar | Fixed-`k` crop truncation — raise `--crop-k` for that scan |
| Scan has < 24,000 vertices | Handled automatically by midpoint subdivision in Stage 1 |
