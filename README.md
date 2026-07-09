# dental_landmark_pipeline

A self-contained two-stage pipeline for 3D dental landmark detection.

| Stage | Model | Task |
|-------|-------|------|
| 1 | Pytorch Point Transformer | Global jaw segmentation → per-vertex FDI labels |
| 2 | Pytorch Stratified Transformer | Per-tooth anatomical landmark detection |

**Output**: JSON file with per-landmark `(x, y, z)` coordinates, confidence
scores, and clinical FDI tooth numbers.

---

## Environment

| Item | Version |
|------|---------|
| Python | 3.10.11 |
| PyTorch | 2.10.0+cu126 |
| CUDA Toolkit | 12.6 |
| GPU | RTX 3050 Laptop (Ampere sm_86, 4 GB) |

---

## Setup

### 1 — Install Python dependencies

```bat
pip install -r requirements.txt
```

### 2 — Build CUDA extensions

```bat
build_extensions.bat
```

This compiles two custom CUDA libraries (`tgnet_ops`, `teethland_ops`) and
installs `torch-scatter`. Requires Visual Studio Build Tools (C++ workload).

### 3 — Place model checkpoints

See [checkpoints/README.md](checkpoints/README.md) for download instructions.

### 4 — Prepare dataset

See [data/README_DATA.md](data/README_DATA.md) for download and layout instructions.

```bat
python data\prepare_dataset.py
```

---

## Inference

**Single scan:**
```bat
python run_pipeline.py data\teeth3ds\01A6HAN6\01A6HAN6_upper.obj
```

**Batch (entire patient folder):**
```bat
python run_pipeline.py --batch data\teeth3ds\01A6HAN6\
```

**Memory-saving mode (RTX 3050 — fewer points per crop):**
```bat
python run_pipeline.py scan.obj --crop-k 8000
```

**All options:**
```bat
python run_pipeline.py --help
```

---

## Output format

```json
{
  "jaw": "upper",
  "landmarks": [
    {
      "class":     "Cusp",
      "coord":     [3.21, 12.05, -5.43],
      "score":     0.93,
      "fdi_tooth": 16
    },
    {
      "class":     "Mesial",
      "coord":     [1.10, 11.80, -4.90],
      "score":     0.88,
      "fdi_tooth": 16
    }
  ]
}
```

---

## Project structure

```
dental_landmark_pipeline/
├── run_pipeline.py            ← main CLI entry point
├── requirements.txt
├── build_extensions.bat       ← compile CUDA libs (run once)
│
├── extensions/
│   ├── tgnet_ops/             ← TGNet CUDA extension (renamed from pointops)
│   └── teethland_ops/         ← 3DTeethLand CUDA extension
│
├── stage1_segmentation/       ← TGNet components
│   ├── inference_pipeline.py
│   ├── gen_utils.py
│   ├── ops_utils.py
│   └── models/
│
├── stage2_landmarks/          ← 3DTeethLand components
│   └── teethland/
│       ├── models/landmarknet.py
│       └── nn/modules/stratified_transformer.py
│
├── pipeline/                  ← Bridge + orchestration (new code)
│   ├── combined_pipeline.py
│   ├── data_bridge.py
│   └── landmark_postprocess.py
│
├── data/
│   ├── README_DATA.md
│   ├── prepare_dataset.py
│   └── teeth3ds/              ← place dataset here
│
└── checkpoints/
    └── README.md              ← place .pth / .ckpt files here
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| CUDA OOM during Stage 1 | Reduce to `--crop-k 8000` |
| CUDA OOM during Stage 2 | Also reduce: edit `proposal_points` in `stage2_landmarks/teethland/config/config.yaml` to `8000` |
| `tgnet_ops` import error | Re-run `build_extensions.bat` |
| `teethland_ops` import error | Re-run `build_extensions.bat` |
| `torch_scatter` not found | Run: `pip install torch-scatter -f https://data.pyg.org/whl/torch-2.10.0+cu126.html` |
