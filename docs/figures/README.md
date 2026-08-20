# Figures

Two kinds of asset live here.

## 1. Generated diagrams (already present)

Hand-authored SVGs — no external dependencies, render natively on GitHub, and stay legible
because each carries an explicit white background.

| File | Used by | Contents |
|---|---|---|
| `architecture.svg` | root `README.md`, `docs/RESULTS.md` | Full three-stage pipeline flow with layer dimensions and hyperparameters |
| `results.svg` | root `README.md`, `docs/RESULTS.md` | Segmentation + landmark metric bars, and leaderboard context |
| `landmark-classes.svg` | root `README.md`, `docs/RESULTS.md` | Six landmark classes, marker colours, and reference-scan detection counts |

Edit them as plain text; there is no build step.

## 2. Report screenshots (drop-in slots — **currently empty**)

The renders in the FYP report live inside the PDF and are not in this repository. Export them
(or re-render them from the pipeline) to the filenames below and the READMEs will pick them up
automatically — the `<img>` tags are already wired to these exact paths.

| Filename to create | Report figure | What it shows |
|---|---|---|
| `scan-raw.png` | Fig. 13 | Original `01F4JV8X_upper.obj` mesh, unsegmented |
| `scan-segmented.png` | Fig. 14 | Same mesh with 14 individually coloured teeth |
| `scan-landmarks-all.png` | Fig. 15 | `_colored.ply` — all landmark octahedra on the mesh |
| `landmarks-cusp.png` | Fig. 16 | `_Cusp.ply` — blue octahedra |
| `landmarks-distal.png` | Fig. 17 | `_Distal.ply` — green octahedra |
| `landmarks-mesial.png` | Fig. 18 | `_Mesial.ply` — red octahedra |
| `landmarks-inner.png` | Fig. 19 | `_InnerPoint.ply` — yellow octahedra |
| `landmarks-outer.png` | Fig. 20 | `_OuterPoint.ply` — cyan octahedra |
| `landmarks-facial.png` | Fig. 21 | `_FacialPoint.ply` — purple octahedra |
| `eval-terminal.png` | Fig. 12 | `eval_miccai.py` terminal output |

### Regenerating them from the pipeline

Rather than cropping the PDF, you can re-render at full quality:

```bat
cd dental_landmark_pipeline
python run_pipeline.py data\teeth3ds_sample\teeth3ds_sample\01F4JV8X\01F4JV8X_upper.obj
```

Then open the PLY files written to `data/output/01F4JV8X_upper/` in MeshLab, CloudCompare, or
ArcGIS and screenshot each layer. Aim for **≤ 1600 px wide** and save as PNG so GitHub renders
them quickly.

### Guidelines

- Keep the filenames exactly as listed — the READMEs reference them literally.
- Prefer a light or neutral background; the surrounding diagrams are light-background, and a
  dark screenshot beside them reads as inconsistent.
- Two viewpoints per class (occlusal top-down + lateral) communicate far more than one; combine
  them side-by-side into a single PNG rather than adding extra files.
