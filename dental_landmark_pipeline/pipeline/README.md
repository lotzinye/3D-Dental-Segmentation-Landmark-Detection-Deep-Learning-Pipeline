# `pipeline/` — Coordinate-Space Bridge & Post-Processing

**This directory is the original contribution of the project.** Stages 1 and 3 adapt published
architectures (TGNet / CGIP and 3DTeethLand LandmarkNet); the code here is what makes them
function as one system rather than two disconnected models.

← [Developer guide](../README.md) · [Project README](../../README.md) · [Results](../../docs/RESULTS.md)

| Module | Responsibility |
|---|---|
| [`data_bridge.py`](data_bridge.py) | Derive the mm-per-unit scale factor; convert TGNet crops into 3DTeethLand `PointTensor`s |
| [`combined_pipeline.py`](combined_pipeline.py) | End-to-end orchestration: load both networks, run Stage 1, apply the bridge, run Stage 3, assemble outputs |
| [`landmark_postprocess.py`](landmark_postprocess.py) | Turn dense per-point distance/offset fields into discrete landmark coordinates |

---

## 1. The problem the bridge solves

The two networks were trained under **mutually incompatible normalisation conventions**:

| | Stage 1 (segmentation) | Stage 3 (landmarks) |
|---|---|---|
| Trained on | Teeth3DS / 3DTeethSeg'22 | 3DTeethLand / MICCAI 2024 |
| Normalisation | Fixed **range** — jaw Y mapped to `[-0.8, 1.0]`, span 1.8 units | Fixed **statistical dispersion** — z-score, σ = 17.3281 mm |
| Origin | 3D mean of the scan, then Y-range rescale | 3D mean of the scan |

Because one normalises to a *range* and the other to a *dispersion*, **no single scalar converts
between them.** The ratio depends on each individual patient's jaw dimensions. Feeding Stage-1
coordinates into Stage 3 directly produces landmarks that are silently, patient-dependently
wrong — the network runs without error and returns plausible-looking garbage.

The bridge therefore routes through the one space both conventions are defined against:

```
Stage 1 normalised space  →  physical millimetres  →  Stage 3 z-score space
```

---

## 2. `data_bridge.py`

### Constants

```python
TGNET_Y_RANGE: float = 1.8      # TGNet maps jaw Y to [-0.8, 1.0] → span 1.8 units
Z_SCORE_STD:   float = 17.3281  # 3DTeethLand global z-score std, in mm
```

`Z_SCORE_STD` is taken from the 3DTeethLand dataset statistics (`transforms.py` in the original
repo). It is a **global constant, not a per-scan statistic** — it must not be recomputed from the
input scan, or the crops will no longer match the distribution LandmarkNet was trained on.

### Scale-factor derivation

```python
def get_scale_factor(original_vertices_mm: np.ndarray) -> float:
    y_extent_mm = float(original_vertices_mm[:, 1].max() - original_vertices_mm[:, 1].min())
    if y_extent_mm < 1e-6:
        raise ValueError("Original mesh has zero Y extent — check input mesh.")
    return y_extent_mm / TGNET_Y_RANGE
```

Because Stage 1 normalised the jaw's Y-extent to exactly 1.8 units, dividing the **original**
millimetre Y-extent by 1.8 recovers how many millimetres one Stage-1 unit represents — per scan.
This is the quantity that varies patient-to-patient and that no fixed scalar could capture.

The zero-extent guard catches degenerate or mis-parsed meshes at the earliest possible point,
before they propagate into an unfalsifiable coordinate error downstream.

### Centroid conversion

Applied in `combined_pipeline.py`:

```python
tooth_norm_xyz = sampled_xyz[tooth_mask]                       # TGNet space
centroid_tgnet = tooth_norm_xyz.mean(axis=0)                   # (3,)
centroid_norm  = ((centroid_tgnet + 0.8) * scale_factor + min_y_c) / Z_SCORE_STD
```

| Operation | Purpose |
|---|---|
| `+ 0.8` | Un-shifts the TGNet range from `[-0.8, 1.0]` back to `[0, 1.8]` |
| `× scale_factor` | Converts to millimetres, relative to the centred origin |
| `+ min_y_c` | Restores the centring offset |
| `/ Z_SCORE_STD` | Applies 3DTeethLand's z-score scaling |

**On `min_y_c`** — the subtle term. Stage 1's normalisation subtracts the **3D mean first**, then
normalises by the Y-range. Subtracting the 3D mean displaces the Y-minimum from its raw value,
and `min_y_c` captures exactly that displacement for the Y-axis. Omitting it produces a constant
vertical offset in every landmark — small enough to look like model error rather than a
coordinate bug, which is precisely what makes it dangerous.

### Full-jaw z-score normalisation

Computed in parallel, providing the point pool that crops are drawn from:

```python
jaw_mean = orig_xyz_mm.mean(axis=0)
jaw_norm = (orig_xyz_mm - jaw_mean) / Z_SCORE_STD
```

### Per-tooth crop extraction

```python
from sklearn.neighbors import KDTree as _KDTree
norm_tree = _KDTree(jaw_norm, leaf_size=16)
crop_idxs = norm_tree.query(centroid_norm[None], k=self.crop_k, return_distance=False)[0]
```

**Why fixed-k KNN and not a ball query** (which Stage 1 uses internally): KNN guarantees an
identically-shaped tensor for every tooth in every scan regardless of local scan density. A ball
query would return too few points in sparse regions and too many in dense ones — and the
Stratified Transformer requires a consistent input shape.

The trade-off is documented in [docs/RESULTS.md §4.3](../../docs/RESULTS.md): for teeth at the
extremes of the arch, the k nearest neighbours skew toward the arch interior, which can truncate
the facial surface. This is the known cause of the occasional missed posterior facial point.

### `PointTensor` construction

```python
crop_xyz_norm    = jaw_norm[crop_idxs]                    # (crop_k, 3)
crop_normals     = orig_normals[crop_idxs]                # (crop_k, 3)
centroid_offsets = crop_xyz_norm - centroid_norm          # (crop_k, 3)
features         = torch.cat([crop_xyz_t, crop_norm_t, cent_off_t], dim=1)   # (N, 9)
```

A `PointTensor` keeps coordinates `C` (N, 3) **separate** from features `F` (N, 9), alongside
`batch_counts` / `batch_indices` for batched variable-size point clouds. The separation is
architecturally load-bearing: `C` drives spatial operations (KDTree queries, FPS downsampling,
CRPE position encoding) while `F` is what the transformer layers actually transform.

| Channels | Content | Rationale |
|---|---|---|
| 0–2 | Z-score normalised XYZ | Absolute position within the crop |
| 3–5 | Surface normals | Local surface orientation |
| 6–8 | Centroid offsets (Δx, Δy, Δz) | Signed displacement from the tooth centre — matches `CentroidOffsetsAsFeatures` used during LandmarkNet training |

Channels 6–8 must be constructed exactly as in training. They are what tells the network where
the tooth's centre is relative to each point, and every head's offset regression is calibrated
against that frame.

---

## 3. `landmark_postprocess.py`

Each landmark head emits, for **every one of the 12,000 crop points**, a 4-channel prediction:

| Channel | Content |
|---|---|
| `F[:, 0]` | Predicted distance from this point to the nearest landmark of this head's type (z-score units) |
| `F[:, 1:4]` | Predicted 3D offset `(Δx, Δy, Δz)` from this point toward that landmark |

`extract_landmarks()` collapses those 12,000 votes into discrete positions.

### Step 1 — candidate filtering

```python
mask = distances < dist_thresh
if mask.sum() < min_points:
    return None          # landmark declared absent on this tooth
```

Returning `None` is a real prediction, not a failure path — it is how the pipeline expresses
"this tooth has no cusp", which is correct for incisors.

### Step 2 — confidence-weighted voting

```python
coords  = head_output.C[mask] + offsets[mask]        # point + predicted offset
dists   = distances[mask].clamp(0.0, dist_thresh)
weights = (dist_thresh - dists) / dist_thresh        # closer ⇒ higher weight
```

Points predicting a shorter distance get more say, on the reasoning that proximity implies a more
reliable offset regression — a point 0.5 mm from a cusp tip knows where it is far better than one
2 mm away.

### Step 3 — DBSCAN clustering

```python
CLUSTER_MAX_DIST: float = 0.03   # z-score units (~0.52 mm) — DBSCAN epsilon
CLUSTER_MIN_PTS:  int   = 20
```

DBSCAN is the deliberate choice over k-means or a fixed-output head: it is **density-based and
never needs the cluster count specified in advance**. However many dense vote regions exist, that
is how many landmarks are emitted. This is what makes variable-cardinality cusp detection
tractable — and it is why cusps score as well as the fixed-cardinality classes
([docs/RESULTS.md §3.3](../../docs/RESULTS.md#33-cusps-the-hard-class-solved)).

The final position is the **confidence-weighted centroid** of each accepted cluster, converted
back to physical millimetres and tagged with class name, confidence, and the FDI number inherited
from Stage 1.

### Per-head parameter overrides

Landmark classes have genuinely different prediction densities, so uniform thresholds
underperform. The module therefore carries head-specific overrides:

```python
DIST_THRESH: float = 0.15                       # default, ~2.60 mm
DIST_THRESH_BY_HEAD = {
    "MesialDistal": 0.10,                       # ~1.73 mm — matches training LandmarkLoss
}

MIN_POINTS_BY_HEAD = {
    "MesialDistal": 3,    # sparse predictions at tooth contact margins; needs 2 clusters
    "FacialPoint": 15,    # semi-sparse band on the facial surface
    "Cusp":        20,    # dense predictions on the occlusal surface
    "InnerPoint":  20,    # dense predictions in the central fossa/pit
    "OuterPoint":  20,    # dense predictions along the buccal contour
}
```

**Why MesialDistal is special.** The head was trained with `LandmarkLoss(dist_thresh=0.10)`.
Using the default 0.15 at inference pulls in noisy points from *adjacent teeth* across the narrow
interproximal contact margin — degrading both DBSCAN quality and the two-means split below.
Matching the training threshold keeps the candidate pool clean. This is a general lesson worth
stating: **inference thresholds must match training thresholds per head**, not per network.

### Mesial/Distal two-means separation

Mesial and Distal share a single head (`out_channels = 4`, classes 0 and 1), so their votes
arrive as one cloud that must be split into exactly two landmarks on opposite proximal faces.
`_two_means_mesial_distal()` handles this, guarded by a minimum separation:

```python
_MD_MIN_SEP: float = 0.10   # ~1.73 mm
```

Mesial and distal contact points sit typically 2–6 mm apart (0.12–0.35 z-score units). Below
`_MD_MIN_SEP` the two candidate "clusters" are treated as one landmark rather than being split —
preventing a single contact point from being duplicated into a spurious pair.

> **Note:** this two-means logic and the per-head overrides post-date the FYP report's
> Table 16 numbers, which were produced with uniform `dist_thresh = 0.12` / `min_points = 20`.
> See [`../FIX_MESIAL_DISTAL_LANDMARKS.md`](../FIX_MESIAL_DISTAL_LANDMARKS.md) for the analysis
> behind the change, and re-run `eval_miccai.py` to obtain figures matching the current code.

---

## 4. Tuning guide

Symptom-driven, in order of what to reach for first:

| Symptom | Likely cause | Parameter to adjust |
|---|---|---|
| Duplicate landmarks on a single-instance class | Vote cloud split into two clusters | Raise `CLUSTER_MAX_DIST`, or add per-FDI deduplication |
| Landmark missing on a class you expect | Fewer than `min_points` candidates survived | Lower that head's `MIN_POINTS_BY_HEAD`, or raise its `dist_thresh` |
| Landmarks drifting toward neighbouring teeth | Candidate pool contaminated across the contact margin | Lower that head's `dist_thresh` |
| Systematic offset in *every* landmark | Coordinate-bridge bug — check `min_y_c` and `Z_SCORE_STD` | Not a tuning issue; verify the bridge |
| Missing landmark only on terminal molars | Fixed-`k` crop truncated the surface | Raise `--crop-k` |
| Noisy, scattered landmarks everywhere | `dist_thresh` too permissive | Lower `DIST_THRESH` |

Verify any change by exporting the vote cloud and inspecting it directly:

```bat
python export_stages.py scan.obj exports\stages\ --focus-fdi 16
```

The `stage3b_candidates` → `stage3c_dbscan` → `stage3d_tooth_landmarks` sequence shows precisely
where a bad landmark originates: a scattered candidate cloud implicates the network, an
over-split clustering implicates `max_neighbor_dist`, and a rejected cluster implicates
`min_points`.

> `min_points` is duplicated in `combined_pipeline.py` and `eval_miccai.py` (`dbscan_cfg`).
> **Keep all three in sync**, or evaluation will not reflect inference.
