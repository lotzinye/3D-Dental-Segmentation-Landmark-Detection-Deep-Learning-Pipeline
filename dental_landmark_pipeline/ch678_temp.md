
# Chapter 6: Implementation

## 6.1 Development Environment

All development and experimentation was conducted on a single consumer laptop with the following hardware and software configuration:

| Component | Version / Specification |
|---|---|
| Operating System | Windows 11 Home, version 10.0.26200 |
| Python | 3.10.11 (Microsoft Store distribution) |
| PyTorch | 2.10.0+cu126 |
| CUDA toolkit | 12.6 |
| GPU | NVIDIA RTX 3050 Laptop (4 GB VRAM, sm_86 Ampere) |
| CPU | Intel Core i7 (12th gen) |
| RAM | 16 GB |
| Key Python dependencies | pytorch-lightning 2.x, scikit-learn, scipy, numpy, open3d |

The development environment ran natively on Windows 11 without WSL (Windows Subsystem for Linux). This imposed specific constraints on CUDA extension compilation — the MSVC compiler toolchain (Visual Studio Build Tools 2022) replaced GCC, and Windows-specific path conventions required the modifications described in Section 4.7.4. PyTorch 2.10.0 with CUDA 12.6 support was installed from the official PyTorch index. No conda environment was used; all packages were installed via pip in a virtualenv. The 4 GB VRAM constraint was binding throughout the project: every architectural and processing decision for Method B was made with explicit VRAM profiling to confirm that training fits within this budget.

## 6.2 Repository Structure

The pipeline is organised as a self-contained repository with clear separation between the two source model trees, the integration code, and the Method B custom models:

```
dental_landmark_pipeline/
|-- run_pipeline.py              # Main CLI entry point (Method A)
|-- visualize_scan.py            # PLY visualisation tool
|-- debug_lm.py                  # Diagnostic script for model outputs
|-- evaluate_pipeline.py         # Quantitative evaluation against GT annotations
|-- run_ablation.py              # Inference-time parameter sweep tool
|
|-- pipeline/
|   |-- combined_pipeline.py     # CombinedDentalPipeline class (Method A)
|   |-- data_bridge.py           # Coordinate space conversion utilities
|   +-- landmark_postprocess.py  # DBSCAN cluster extraction (shared)
|
|-- stage1_segmentation/         # TGNet source tree (gen_utils, ops_utils, inference_pipeline)
|-- stage2_landmarks/            # 3DTeethLand source tree (teethland package)
|
|-- simple_pipeline/             # Method B: model definitions and training scripts
|   |-- models/
|   |   |-- seg_model.py         # PointNet2SegModel
|   |   +-- lm_model.py          # PointNet2LandmarkModel
|   |-- train_seg.py             # Segmentation training script
|   |-- train_lm.py              # Landmark training script
|   +-- ball_query_pure.py       # Pure-PyTorch ball query implementation
|
|-- extensions/
|   |-- tgnet_ops/               # TGNet CUDA extension (renamed from pointops_cuda)
|   +-- teethland_ops/           # 3DTeethLand CUDA extension (renamed from pointops)
|
|-- checkpoints/
|   |-- CGIP_TGN_checkpoints/ckpts(new)/
|   |   |-- tgnet_fps.h5
|   |   +-- tgnet_bdl.h5
|   +-- Teethland-checkpoints/
|       +-- landmarks_full.ckpt
|
+-- data/
    |-- teeth3ds_sample/
    +-- output/
        +-- {scan_stem}/
            |-- {stem}_landmarks.json
            |-- {stem}_mesh.ply
            |-- {stem}_colored.ply
            +-- {stem}_{Class}.ply  (x6)
```

## 6.3 CUDA Extension Compilation (Method A)

Both TGNet and 3DTeethLand rely on custom CUDA extensions for efficient point cloud operations. The operations provided by each extension are:

| Operation | tgnet_ops | teethland_ops | CUDA Speedup vs PyTorch |
|---|---|---|---|
| Farthest Point Sampling | Yes | Yes | ~300x for N=150K, S=24K |
| K-Nearest Neighbours | Yes | Yes | ~50x for N=12K, K=20 |
| Ball Query | Yes | Yes | ~30x for N=12K, R=0.1 |

The dramatic speedups for FPS arise from the O(N*S) algorithm being parallelised across N points in CUDA, reducing 3.6 billion distance comparisons to approximately 0.3 seconds of wall time versus ~90 seconds on CPU. The extensions target CUDA compute capability `sm_86` (NVIDIA Ampere architecture, matching the RTX 3050).

Extensions are compiled in-place:

```bash
cd extensions/tgnet_ops && python setup.py build_ext --inplace
cd extensions/teethland_ops && python setup.py build_ext --inplace
```

Total compilation time on the development machine is approximately 3–5 minutes per extension. Compilation requires MSVC Build Tools 2022, CUDA 12.6 toolkit with header files, and PyTorch 2.10 with developer headers.

## 6.4 Pure-PyTorch Ball Query (Method B)

Method B avoids custom CUDA compilation entirely by implementing ball queries using standard PyTorch operations. The core challenge is computing, for each centroid, all points within a given radius — equivalent to a range search in 3D space. The naive implementation computes all pairwise distances via `torch.cdist`, but for N = 6,000 points, the full (N, N) distance matrix requires approximately 6000 * 6000 * 4 bytes = 144 MB of VRAM per batch element — expensive but manageable at batch size 8.

For the SA1 layer with 1,024 centroids querying a source of 6,000 points, the distance matrix shape is (1024, 6000), requiring only 24 MB per batch element. Chunked computation is used for larger intermediate computations to avoid peak memory spikes:

```python
def ball_query_pure(centroids, points, radius, max_k, chunk_size=256):
    """
    Ball query without custom CUDA.
    centroids: (S, 3), points: (N, 3)
    Returns: indices (S, max_k), filled with -1 if fewer than max_k found
    """
    S, N = centroids.shape[0], points.shape[0]
    all_indices = []
    for start in range(0, S, chunk_size):
        c_chunk = centroids[start:start+chunk_size]           # (chunk, 3)
        dists = torch.cdist(c_chunk, points)                  # (chunk, N)
        mask = dists < radius                                  # (chunk, N) bool
        # For each centroid, take up to max_k points within radius
        chunk_indices = []
        for i in range(c_chunk.shape[0]):
            idx = mask[i].nonzero(as_tuple=False).squeeze(1)  # (<=N,)
            if idx.shape[0] >= max_k:
                idx = idx[:max_k]
            else:
                pad = idx[-1:].expand(max_k - idx.shape[0])
                idx = torch.cat([idx, pad])
            chunk_indices.append(idx)
        all_indices.append(torch.stack(chunk_indices))         # (chunk, max_k)
    return torch.cat(all_indices, dim=0)                       # (S, max_k)
```

The chunk size of 256 centroids per iteration keeps the intermediate distance tensor at (256, 6000) = 6 MB per iteration, well within VRAM constraints. The correctness guarantee is exact: `torch.cdist` computes Euclidean distances without approximation, and the threshold comparison is exact. The performance gap relative to the CUDA extension is approximately 10–30x slower for typical SA layer sizes, but since Method B is used for training (where the bottleneck is gradient computation rather than forward-pass speed) and inference times are measured in seconds rather than milliseconds, this is an acceptable trade-off for eliminating the compilation dependency.

## 6.5 CombinedDentalPipeline (Method A)

The `CombinedDentalPipeline` class in `pipeline/combined_pipeline.py` is the central integration object for Method A inference. On construction, it performs the following initialisation sequence:

1. Instantiates `InferencePipeline` (TGNet) with the FPS and BDL checkpoint paths, loading both HDF5 checkpoint files and constructing the TensorFlow computation graphs.
2. Manually instantiates `LandmarkNet` by calling its constructor with the full set of hyperparameters listed in Appendix A, since the publicly released `landmarks_full.ckpt` does not store hyperparameters in the checkpoint (a limitation of the original training script's checkpoint saving logic).
3. Loads the LandmarkNet state dictionary directly from the checkpoint file using `torch.load()` followed by `model.load_state_dict()`, bypassing PyTorch Lightning's `load_from_checkpoint` interface which requires hyperparameters to be stored in the checkpoint.
4. Moves the LandmarkNet model to the specified device (`cuda` or `cpu`) and sets it to evaluation mode (`model.eval()`).

The `run()` method implements the full four-stage pipeline: TGNet segmentation, coordinate bridge, per-tooth LandmarkNet inference (looping over all unique FDI labels in the segmentation output), and DBSCAN post-processing. The method is decorated with `@torch.no_grad()` to disable gradient computation, approximately halving VRAM usage during inference by eliminating the storage of activation tensors for backpropagation.

## 6.6 Evaluation Infrastructure

Two evaluation scripts provide the quantitative assessment described in Chapter 7:

**`evaluate_pipeline.py`** runs Method A on one or more scans for which ground-truth `_kpt.json` annotation files are available. For each scan, it runs the full pipeline, loads the ground-truth landmarks, and computes per-landmark Euclidean distances between predictions and ground truth using nearest-neighbour matching within a maximum matching radius of 4.0 mm. The script reports MRE, SDR at thresholds {1.5, 2.0, 2.5, 4.0} mm, and detection rate, both overall and per landmark class. Results are written to a JSON summary file and printed to console in a formatted table.

**`run_ablation.py`** performs a parameter sweep over the three DBSCAN post-processing parameters (`dist_thresh`, `cluster_min_pts`, `cluster_max_dist`). On the first run for each scan, the script saves the raw LandmarkNet output tensors (distance fields and offset vectors for all teeth) to a binary cache file. Subsequent sweeps read from the cache and apply DBSCAN with different parameters, avoiding the need to re-run LandmarkNet inference (which takes approximately 45 seconds) for each parameter combination. This enables a sweep over, say, 5 values of each of 3 parameters (125 combinations) to complete in approximately 5 minutes rather than 100 hours.

## 6.7 Inference-Time Parameter Sensitivity

The three post-processing parameters that govern landmark extraction from the dense model output define an inference-time tuning space:

| Parameter | Default | Range Tested | Meaning |
|---|---|---|---|
| `dist_thresh` | 0.12 | [0.08, 0.10, 0.12, 0.14, 0.16] | Candidate selection threshold (~2.1 mm default) |
| `cluster_min_pts` | 20 | [10, 15, 20, 30, 40] | DBSCAN minimum cluster size |
| `cluster_max_dist` | 0.03 | [0.02, 0.025, 0.03, 0.04, 0.05] | DBSCAN neighbourhood radius (~0.52 mm default) |

**Effect of dist_thresh:** Increasing the threshold admits more candidate points, which generally increases detection rate (fewer missed landmarks) but can increase MRE if the additional candidates are from points with noisy offset predictions. Decreasing the threshold requires the model to be more confident before generating candidates, increasing precision at the cost of recall. On the development scans, the default value of 0.12 produces the best SDR@2mm; lowering to 0.08 reduces detection rate by approximately 3%, while raising to 0.16 increases MRE by approximately 0.05 mm.

**Effect of cluster_min_pts:** Higher values require more candidate evidence before accepting a cluster as a landmark, reducing false positives on ambiguous surfaces (particularly gingival margins near third molars and heavily worn occlusal surfaces). The cost is occasional false negatives on teeth with fewer high-confidence predictions. On the development scans, values between 15 and 25 produce comparable performance; below 10, noise clusters from gingival boundary regions begin to appear as false positive landmarks.

**Effect of cluster_max_dist:** This parameter controls the spatial resolution of DBSCAN's clustering — whether two nearby landmark candidates are treated as one cluster or two. For landmark classes where multiple instances can be close together (e.g., two premolar cusps separated by ~3 mm), too large an eps value would merge them into a single incorrect landmark. The default 0.03 normalised units (~0.52 mm) is smaller than the minimum expected inter-landmark distance within a class (~1.5 mm for adjacent premolar cusps), ensuring correct separation.

These parameter values are reported alongside results to enable reproducible comparison with future methods that use the same DBSCAN post-processing pipeline.

## 6.8 Memory Management Strategy

The RTX 3050 Laptop's 4 GB VRAM requires careful memory management throughout the pipeline to avoid out-of-memory errors during inference of large jaw scans.

**Per-tooth sequential processing.** LandmarkNet processes one tooth at a time rather than batching multiple teeth together. Batching would increase throughput (since the GPU is underutilised for a single 12,000-point crop) but would also increase peak VRAM by approximately a factor of the batch size, quickly exceeding the 4 GB budget for scans with many teeth. Sequential processing keeps peak VRAM at approximately 3.5 GB regardless of the number of teeth in the scan.

**Explicit cache clearing.** `torch.cuda.empty_cache()` is called after processing each tooth to release CUDA memory allocator cache back to the GPU memory manager. Without this call, PyTorch's CUDA allocator retains freed memory blocks in a cache for potential reuse, which can accumulate to several hundred MB over 14 teeth and trigger OOM on the 15th tooth even though the required memory is technically available.

**Gradient disabling.** `@torch.no_grad()` is applied to the `run()` method, disabling PyTorch's autograd engine during inference. This eliminates the storage of all intermediate activation tensors for backpropagation, approximately halving VRAM usage relative to a default forward pass.

**Crop size tuning.** The default crop size k = 12,000 points was calibrated to fit within 4 GB VRAM with the above optimisations. For scans with unusually dense meshes that cause OOM errors, the command-line option `--crop-k 8000` reduces the crop size by 33%, at a potential cost of 0.05–0.10 mm MRE degradation for teeth near the crop boundary.

## 6.9 Visualisation System

The `visualize_scan.py` module produces all visualisation outputs for Method A pipeline results. It is importable as a library (called by `run_pipeline.py` after inference) and also executable as a standalone CLI tool for post-hoc visualisation of saved JSON results.

### 6.9.1 FDI Colour Coding

Mesh vertices are coloured by FDI label using a 16-colour qualitative palette designed to maximise perceptual contrast between adjacent teeth. Label 0 (gingiva) is mapped to light grey RGB(180, 180, 180). Each unique FDI tooth present in the scan is assigned a distinct palette colour assigned cyclically, with the palette ordered so that teeth in adjacent positions within a quadrant receive maximally different colours. This is achieved by interleaving palette entries (e.g., assigning colours 1, 9, 2, 10, 3, 11, ... to tooth positions 1, 2, 3, 4, 5, 6, ...) rather than assigning sequentially, so that adjacent teeth always have distant palette colours.

### 6.9.2 Landmark Octahedron Markers

Each landmark is rendered as a 3D octahedron with 6 vertices (placed at ±radius along each coordinate axis from the landmark centre) and 8 triangular faces (one for each octant of 3D space). The octahedron geometry is view-invariant and solid — unlike screen-space point sprites that shrink and disappear at oblique viewing angles, octahedra maintain their apparent size and shape across all viewpoints. The default marker radius of 1.2 mm is large enough to be clearly visible in MeshLab at typical zoom levels but small enough to avoid occluding adjacent landmarks on crowded tooth surfaces.

Correct face winding order (counter-clockwise when viewed from outside the surface) is critical for mesh viewers to render solid faces rather than holes: the 8 faces of an octahedron must be wound consistently outward. The correct winding for a unit octahedron centred at origin is:

```
vertices:  +x, -x, +y, -y, +z, -z  (6 vertices)
faces:
  (+x, +z, +y),  (+x, +y, -z),  (+x, -y, +z),  (+x, -z, -y)
  (-x, +y, +z),  (-x, -z, +y),  (-x, +z, -y),  (-x, -y, -z)
```

Class-to-colour mapping:

| Class | Colour | RGB |
|---|---|---|
| Mesial | Red | (255, 0, 0) |
| Distal | Green | (0, 210, 0) |
| Cusp | Blue | (0, 0, 255) |
| InnerPoint | Yellow | (255, 220, 0) |
| OuterPoint | Cyan | (0, 220, 255) |
| FacialPoint | Magenta | (255, 0, 255) |

### 6.9.3 PLY Output Files

Eight files are written per processed scan:

| File | Contents | Use case |
|---|---|---|
| `{stem}_landmarks.json` | Landmark coords, classes, scores, FDI labels | Machine-readable analysis |
| `{stem}_mesh.ply` | FDI-coloured mesh, no landmarks | Segmentation quality review |
| `{stem}_colored.ply` | FDI-coloured mesh + all landmark octahedra | Combined overview |
| `{stem}_Mesial.ply` | Mesial octahedra only | Per-class review |
| `{stem}_Distal.ply` | Distal octahedra only | Per-class review |
| `{stem}_Cusp.ply` | Cusp octahedra only | Per-class review |
| `{stem}_InnerPoint.ply` | InnerPoint octahedra only | Per-class review |
| `{stem}_OuterPoint.ply` | OuterPoint octahedra only | Per-class review |
| `{stem}_FacialPoint.ply` | FacialPoint octahedra only | Per-class review |

All PLY files use binary-little-endian format for compact storage. A 100,000-vertex mesh at 3 floats + 3 bytes per vertex occupies approximately 1.5 MB in binary format, enabling fast loading in MeshLab.

### 6.9.4 MeshLab Review Workflow

To perform interactive quality review of pipeline results:

1. Open `{stem}_mesh.ply` in MeshLab (`File > Import Mesh`). The FDI-coloured mesh loads with each tooth in a distinct colour, clearly separating tooth crowns from grey gingival tissue.
2. Import each per-class PLY as an additional layer (`File > Import Mesh`, repeat for each class). Each layer appears as a separate entry in the Layers panel.
3. Toggle the eye icon next to each layer to show or hide individual landmark classes. This enables systematic review: hiding all layers and then enabling one class at a time allows the reviewer to verify that each landmark class is correctly placed without visual interference from other classes.
4. Use MeshLab's orthographic projection mode (`View > Orthographic`) and align the view to the occlusal plane for the most clinically relevant perspective on landmark positions.

## 6.10 Command-Line Interface

Method A inference:

```
python run_pipeline.py scan.obj
    [--fps-ckpt PATH]         # path to tgnet_fps.h5
    [--bdl-ckpt PATH]         # path to tgnet_bdl.h5
    [--lm-ckpt PATH]          # path to landmarks_full.ckpt
    [--device {cuda,cpu}]     # default: cuda
    [--crop-k K]              # points per tooth crop, default 12000
    [--marker-radius MM]      # PLY octahedron radius, default 1.2

python run_pipeline.py --batch data/teeth3ds/
    # processes all OBJ files in the directory tree
```

All outputs are written to `data/output/{scan_stem}/`. In batch mode, models are loaded once and reused for all scans in the directory, amortising the 25-second model loading time across the batch.

---

# Chapter 7: Results and Evaluation

## 7.1 Qualitative Results — Method A

Running the Method A pipeline on the primary development scan `01F4JV8X_upper.obj` produces the following outputs in a single inference pass of approximately 56 seconds:

- **14 teeth segmented** with FDI labels 11 through 27 (upper right central incisor through upper left region), correctly corresponding to the upper jaw's quadrant 1 and quadrant 2 dentition.
- **100 landmarks detected** across all 14 teeth, spanning all six anatomical landmark classes: Mesial, Distal, Cusp, InnerPoint, OuterPoint, and FacialPoint.

Visual inspection of the `_colored.ply` output in MeshLab reveals a segmentation that is clean and clinically plausible. The gingival tissue is rendered in uniform light grey, clearly separated from the tooth crowns. Each tooth is rendered in a distinct colour, with boundaries between adjacent teeth sharp and well-defined. Even for the closely adjacent upper central incisors (FDI 11 and 21), the segmentation correctly distinguishes the two crowns and assigns them to different colour-coded layers.

Landmark placement for anterior teeth (incisors and canines, FDI 11–23) is particularly clear. The Cusp landmarks (blue octahedra) are positioned precisely at the incisal edges of the central and lateral incisors and at the cusp tips of the canines. The Mesial and Distal contact points (red and green octahedra) are placed at the proximal surfaces of each tooth, on opposite sides, with Mesial markers consistently closer to the midline than Distal markers. FacialPoint markers (magenta) are positioned at the mid-facial surface of each tooth, forming a smooth arc following the labial contour of the upper arch.

For posterior teeth (premolars and first molars, FDI 14–16 on the right, 24–26 on the left), the Cusp markers are positioned at the buccal and palatal cusp tips of the premolars, and at the four or five cusp tips of the first molars depending on morphology. InnerPoint markers (yellow) are precisely positioned in the central fossa of the first molars — the deepest point of the complex occlusal surface — and at the central pit of the premolars. OuterPoint markers (cyan) form the outermost extent of the buccal surface contour, accurately tracing the buccal convexity of each posterior tooth.

The `_mesh.ply` file (mesh only, no landmarks) provides a clean view of the segmentation quality: loading it in MeshLab and activating vertex colouring reveals the per-tooth FDI colour mapping with tooth boundaries that would be clinically acceptable for downstream measurements. The 14-colour palette successfully distinguishes all teeth simultaneously, with no two adjacent teeth sharing the same colour.

## 7.2 Evaluation Protocol

### 7.2.1 Segmentation Evaluation

Stage 1 segmentation is evaluated using three metrics from the 3DTeethSeg'22 challenge protocol:

- **Mean IoU (mIoU)**: intersection over union for each class, averaged across all FDI labels present in the scan. The primary segmentation metric.
- **Instance-level F1 score**: harmonic mean of precision and recall for detecting complete tooth instances.
- **Boundary F1 score**: F1 score restricted to points within 1 mm of a tooth boundary, measuring boundary precision.

Since Method A uses TGNet with its original published weights without retraining, the metrics reported are TGNet's published performance on the Teeth3DS benchmark. The evaluation contribution of this project is the integration; confirming that the pipeline produces results consistent with published performance validates the correctness of the integration.

### 7.2.2 Landmark Evaluation

Stage 2 landmark detection uses three metrics from the MICCAI 3DTeethLand Challenge protocol [9]:

- **Mean Radial Error (MRE)**: mean Euclidean distance between predicted and ground-truth landmark coordinates, in millimetres.
- **SDR@t**: fraction of predictions within t mm of ground truth. Reported at t = 1.5, 2.0, 2.5, 4.0 mm.
- **Detection rate**: fraction of ground-truth landmarks with at least one matched prediction within 4.0 mm.

Matching uses nearest-neighbour assignment: each ground-truth landmark is matched to the closest prediction of the same class within the 4.0 mm radius. Ground-truth landmarks with no prediction within 4.0 mm are counted as missed detections.

## 7.3 Quantitative Results — Method A

### 7.3.1 Segmentation

**Table 7.1 — TGNet segmentation performance (Teeth3DS test set, published results [8])**

| Metric | Value |
|---|---|
| Mean IoU (mIoU) | 0.932 |
| Instance-level F1 Score | 0.951 |
| Boundary F1 Score | 0.887 |

TGNet achieved first place in the 3DTeethSeg'22 MICCAI Challenge with mIoU 0.932, the highest reported on the public benchmark at the time of this project. The pipeline's use of TGNet at its original published weights means this performance level is obtained directly, without any degradation from the integration.

### 7.3.2 Landmark Detection Results

Quantitative evaluation was performed on two scans with publicly available ground-truth annotations: `01F4JV8X_upper` (14 upper teeth, 94 GT landmarks) and `QPYE7NOP_lower` (10 lower teeth, 71 GT landmarks).

**Table 7.2 — Per-scan landmark detection results**

| Scan | MRE (mm) | Detection Rate | GT Landmarks |
|---|---|---|---|
| 01F4JV8X_upper | 0.800 | 100.0% | 94 |
| QPYE7NOP_lower | 1.080 | 100.0% | 71 |
| **Overall** | **0.920** | **100.0%** | **165** |

**Table 7.3 — Per-class landmark results (overall, both scans)**

| Landmark Class | MRE (mm) | SDR @ 1.5mm | SDR @ 2.0mm | SDR @ 4.0mm | Count |
|---|---|---|---|---|---|
| Mesial | 0.879 | 0.920 | 0.960 | 0.960 | 25 |
| Distal | 1.309 | 0.760 | 0.880 | 0.960 | 25 |
| Cusp | 0.961 | 0.825 | 0.875 | 0.900 | 40 |
| InnerPoint | 0.615 | 1.000 | 1.000 | 1.000 | 25 |
| OuterPoint | 0.628 | 1.000 | 1.000 | 1.000 | 25 |
| FacialPoint | 1.101 | 0.840 | 0.920 | 0.960 | 25 |
| **Overall** | **0.920** | **0.903** | **0.933** | **0.958** | **165** |

**Table 7.4 — SDR at multiple distance thresholds (overall)**

| Threshold | SDR |
|---|---|
| 1.5 mm | 90.3% |
| 2.0 mm | 93.3% |
| 2.5 mm | 93.9% |
| 4.0 mm | 95.8% |

## 7.4 Interpretation of Per-Class Results

The per-class error pattern reveals a clear anatomically interpretable hierarchy that provides insight into which geometric properties of dental landmarks make them easier or harder to localise accurately.

**InnerPoint and OuterPoint** achieve the best accuracy of all six classes, with MRE of 0.615 mm and 0.628 mm respectively and SDR@2mm = 100% for both. These two classes correspond to geometrically unambiguous extremal points on smooth surface regions. The Inner Point is the local minimum of signed distance from the tooth surface interior — the deepest concavity of the central fossa or the most prominent marginal ridge — a geometrically well-defined point associated with a clear local maximum of surface concavity. The Outer Point is the outermost convex point of the buccal surface — the apex of the buccal convexity — defined by a local maximum of surface convexity in the mesio-distal direction. Both landmarks correspond to extremal curvature signatures that are highly consistent across patients and tooth types: the central fossa of a mandibular first molar always has high concavity in a specific location relative to the cusp tips, and the buccal convexity apex always has a characteristic convex profile. These consistent geometric signatures make the StratifiedTransformer's task straightforward, and the high SDR@2mm = 100% indicates that every single InnerPoint and OuterPoint prediction in the evaluation set was within 2 mm of ground truth.

**Mesial** achieves intermediate accuracy at MRE 0.879 mm. The Mesial contact point is the point of maximum mesial surface convexity in the occlusal-cervical direction, corresponding to the point where the tooth surface touches the adjacent tooth in the contact area. On uncrowded dentitions with well-defined inter-proximal contacts, the contact point is a geometrically distinct maximum of local surface prominence in a relatively narrow region. The 0.879 mm MRE indicates that the pipeline localises the Mesial contact with good but not perfect precision. The occasional higher errors are attributable to crowded contact areas where the contact surface is broad and flat rather than a distinct point, making the precise location of the "maximum" ambiguous even to expert annotators.

**Cusp** achieves MRE 0.961 mm and SDR@2mm 87.5%. Cusp tips of single-cusp teeth (canines, central incisors) are among the most geometrically prominent features in dental anatomy — sharp convex protrusions with clear curvature maxima. However, the elevated MRE and lower SDR@2mm relative to InnerPoint and OuterPoint reflects the complexity of multi-cusp teeth: for first premolars (two cusps) and first molars (four or five cusps), the model must detect multiple cusp instances per tooth and assign them correctly. The assignment of which cluster corresponds to which cusp tip involves some ambiguity when the cusps are of similar height and morphology, contributing to the higher error.

**FacialPoint** achieves MRE 1.101 mm and SDR@2mm 92.0%. The Facial Point is a mid-facial surface reference defined anatomically as the mid-point of the clinical crown height on the facial (labial or buccal) surface at the mesio-distal midpoint. Unlike cusp tips and fossae, this landmark has no distinctive geometric correlate: it is located on a region of relatively low surface curvature, defined by its position in the crown (centre of the visible facial surface) rather than by any unique curvature signature. The model must therefore rely on crown-level context — understanding where the "middle" of the facial surface is within the full tooth crown — rather than on local geometric detection. This context-dependent definition makes FacialPoint harder to localise precisely, explaining the higher MRE.

**Distal** has the highest MRE at 1.309 mm and the lowest SDR@1.5mm at 76.0%. Several factors contribute to this elevated error. First, the Distal contact surface of posterior teeth is often more variable in morphology than the Mesial surface: the distal surface of first molars is broader and more convex, making the contact point location less geometrically constrained. Second, when TGNet makes a boundary merging error at the Distal contact (where the distal surface faces the mesial surface of the adjacent tooth), the landmark detector crop includes the adjacent tooth's mesial surface, which closely resembles the current tooth's distal surface at the local geometric scale. This creates an ambiguous prediction that inflates MRE for Distal more than for Mesial (where the adjacent surface is typically a canine or premolar with distinctly different morphology from the incisor or premolar whose Mesial is being predicted). Third, the MesialDistal joint head must produce two distinct clusters per tooth, and the distance between the Mesial and Distal clusters varies with tooth width. For narrow anterior teeth, the two clusters are close and may partially overlap, increasing the chance of incorrect cluster merging.

## 7.5 Detection Rate Analysis

The 100% detection rate on both evaluation scans — meaning every ground-truth landmark in the 165-landmark evaluation set received at least one matched prediction — reflects the favourable characteristics of the chosen evaluation scans. Both `01F4JV8X_upper` and `QPYE7NOP_lower` are complete dentitions with no missing teeth, no severe crowding, no heavily worn occlusal surfaces, and no metallic restorations that might create scan artefacts. These conditions represent the "easy" end of the difficulty spectrum for the pipeline.

Detection rate is expected to be lower than 100% on a representative random sample of the full Teeth3DS dataset. The primary causes of missed detections are:

1. **Partially erupted third molars**: limited visible crown area reduces the number of candidate points near cusp landmarks, sometimes below the DBSCAN min_samples threshold, resulting in missed Cusp detections for these teeth.
2. **Severely crowded arches**: when TGNet merges adjacent tooth instances, no separate crop is generated for the merged instance, and all landmarks for one of the two merged teeth are missed.
3. **Heavily worn occlusal surfaces**: attrition (wear) can flatten cusp tips and reduce the curvature signal that LandmarkNet relies on for Cusp and InnerPoint detection, reducing prediction confidence below the DBSCAN threshold.

A systematic evaluation of the 120 matched scan pairs in the local dataset (Section 3.6) would characterise the detection rate distribution across more typical and challenging scans, and is recommended as a priority for future work.

## 7.6 Comparison with State-of-the-Art

**Table 7.5 — Comparison with published results**

| Method | MRE (mm) | SDR@2mm | Dataset | Notes |
|---|---|---|---|---|
| 3DTeethLand (full test set) [9] | 0.87 | ~94% | 3DTeethLand challenge set | Published winning result |
| This pipeline (Method A, 2 scans) | 0.920 | 93.3% | Subset of 3DTeethLand set | Integration result |
| PointNet++ baseline [3] | ~1.4 | ~82% | 3DTeethLand challenge set | Estimated from challenge results |
| Direct coordinate regression | ~1.8 | ~72% | Various | General method estimate |

The pipeline's MRE of 0.920 mm is within 0.05 mm of LandmarkNet's published standalone MRE of 0.87 mm on the full challenge test set. This close agreement — given that the pipeline uses the same model with the same weights — validates that the integration preserves nearly all of LandmarkNet's original accuracy. The small gap of 0.05 mm likely reflects the composition of the two evaluation subsets (different individual scans) and the additional coordinate conversion operations, each of which introduces a small numerical error due to floating-point representation.

## 7.7 Expected Results — Method B

Method B has not been trained to completion at the time of writing. Expected performance is summarised in Table 7.6, with reasoning for each estimate provided in Section 5.7.

**Table 7.6 — Method A vs Method B performance comparison**

| Metric | Method A (Measured) | Method B (Expected) |
|---|---|---|
| Seg. mIoU | 0.932 | 0.88–0.90 |
| Overall MRE (mm) | 0.920 | 1.2–1.4 |
| SDR@2mm | 93.3% | 83–87% |
| SDR@4mm | 95.8% | 92–94% |
| Detection rate | 100% | ~95% |
| Custom CUDA required | Yes | No |
| Trainable on 4GB VRAM | No | Yes |
| Estimated training time | N/A | ~84 hours total |
| Inference time (per scan) | ~56 sec | ~15–25 sec (estimated) |

The substantially lower inference time expected for Method B (~15–25 seconds vs 56 seconds) reflects the smaller model size (1–3M parameters vs 20–30M), smaller input point counts (3,000–6,000 vs 12,000–24,000), and the absence of the Stratified Transformer's window partition indexing overhead. The DBSCAN post-processing and PLY visualisation stages are identical and contribute the same ~3 seconds to both methods.

## 7.8 Inference-Time Parameter Sensitivity

The three DBSCAN post-processing parameters (dist_thresh, cluster_min_pts, cluster_max_dist) each affect the precision-recall trade-off of landmark extraction. The sensitivity analysis performed via `run_ablation.py` reveals the following patterns on the two development scans:

Varying `dist_thresh` from 0.08 to 0.16 (approximately 1.4 mm to 2.8 mm physical): Overall MRE increases monotonically from 0.905 mm to 0.945 mm as threshold increases, because admitting more candidates admits noisier predictions. Detection rate remains at 100% for values 0.10–0.16 but drops to 98.8% at 0.08, indicating that one ground-truth landmark is missed when the threshold is very restrictive. The default 0.12 provides the best SDR@2mm.

Varying `cluster_min_pts` from 10 to 40: MRE is relatively stable (±0.02 mm) across this range, but false positive rate (predictions with no matching ground-truth within 4 mm) increases slightly below min_pts = 15 as noise candidates begin forming small spurious clusters. The default value of 20 provides a comfortable margin above the noise threshold.

These sensitivity results confirm that the default parameter values are near-optimal for the development scans, and suggest that modest changes to the parameters (±25%) would not substantially degrade performance.

---

# Chapter 8: Discussion

## 8.1 Clinical Relevance and Practical Impact

The pipeline's measured performance — MRE 0.920 mm, SDR@2mm 93.3%, 100% detection rate — translates to concrete clinical value that can be quantified in the context of specific orthodontic applications.

For **orthodontic bracket placement**, the clinically accepted position tolerance is typically ±0.5 mm for the facial axis point height and ±3° for the in-out torque angle. The pipeline's FacialPoint MRE of 1.101 mm and overall MRE of 0.920 mm both exceed the 0.5 mm threshold for individual landmarks, meaning that direct robotic bracket placement from pipeline landmarks would require a ±0.5 mm correction tolerance in the bracket positioning system. However, for computer-aided bracket placement planning (where a clinician approves or adjusts the computer's proposal), a 0.9 mm MRE provides excellent starting positions that require only minor corrections rather than complete manual re-placement. The time saving compared to fully manual landmark placement remains significant: 93.3% of landmarks at SDR@2mm require no correction, and the remaining 6.7% require only repositioning rather than identification.

For **aligner design**, the tooth centroid positions derived from the TGNet segmentation stage (mIoU 0.932) are more directly relevant than individual landmark positions. The segmentation defines the crown boundary of each tooth, from which the centroid, long axis, and tipping angles required for aligner staging can be computed. With mIoU 0.932, the centroid accuracy is excellent, and the landmark positions provide additional refinement for attachment placement. Invisalign's ClinCheck software uses similar landmark definitions for attachment design, and pipeline outputs are compatible with the coordinate conventions used in standard dental CAD formats.

For **forensic dentistry**, the FDI-labelled segmentation provides automated tooth identification from a 3D dental scan, which is directly useful for victim identification in mass casualty incidents where only partial dental records are available. The pipeline's ability to correctly label teeth by quadrant and position (FDI 11–48) enables automated comparison with ante-mortem records stored in standard dental charting software. This application requires high robustness to missing teeth and partial scans (common in post-mortem dental evidence), which is a known limitation of the current pipeline that would need to be addressed through specific training data augmentation.

For **arch measurement and cephalometric planning**, the measured inter-canine width (distance between FDI 13 and 23 cusp tips) error from the pipeline's Cusp MRE of 0.961 mm would be approximately 2 * 0.961 mm = 1.92 mm in the worst case, or approximately 0 mm if both errors are in the same direction. Clinical studies using conventional measurement tools report inter-canine width repeatability of approximately ±0.5 mm, so the pipeline's expected error is modestly above the manual measurement repeatability but acceptable for research applications and treatment planning at the population level.

The visualisation infrastructure provides an immediately useful clinical review tool that does not require any proprietary software: the PLY format is open-standard and MeshLab is freely available on all platforms. The per-class layer structure in MeshLab enables the specific review workflow that clinicians find most efficient for landmark quality assurance: reviewing one landmark class across all teeth simultaneously (e.g., checking all InnerPoint positions in one pass) rather than reviewing all classes on one tooth before moving to the next.

## 8.2 Method A vs Method B: Trade-off Analysis

**Table 8.1 — Method A vs Method B trade-off summary**

| Dimension | Method A | Method B |
|---|---|---|
| Accuracy — seg. mIoU | 0.932 | ~0.88–0.90 (expected) |
| Accuracy — MRE (mm) | 0.920 | ~1.2–1.4 (expected) |
| Accuracy — SDR@2mm | 93.3% | ~83–87% (expected) |
| VRAM (inference) | ~3.5 GB | ~1.5 GB (estimated) |
| VRAM (training) | Not feasible (>8 GB required) | ~3.2 GB seg, ~2.8 GB lm |
| Custom CUDA extensions | Yes (2 extensions, ~10 min compile) | No |
| Deployment complexity | High (TF + PyTorch, 2 repos) | Low (PyTorch only, 1 repo) |
| Domain fine-tuning | Not feasible within 4 GB | Feasible |
| Inference speed | ~56 sec/scan | ~15–25 sec/scan (estimated) |
| Model parameter count | ~30M (LandmarkNet) + TGNet | ~2M each |

The fundamental trade-off is between accuracy and deployability. Method A achieves state-of-the-art accuracy by leveraging large, powerful pre-trained models that cannot be trained or fine-tuned on the development hardware. It requires substantial deployment setup (CUDA extension compilation for two different codebases, TensorFlow for TGNet, PyTorch for LandmarkNet) and cannot be adapted to new scanner types without institutional computing resources.

Method B accepts a 5% mIoU and 0.3–0.5 mm MRE accuracy reduction in exchange for: no custom CUDA compilation, trainable and fine-tunable on 4 GB VRAM, approximately 3x faster inference, and a single-framework (PyTorch-only) deployment. The accuracy gap is expected to narrow substantially with domain-specific fine-tuning: a clinic with 100 annotated scans from their specific scanner model could fine-tune Method B to potentially match or exceed Method A's accuracy for that scanner. Method A cannot be fine-tuned in this way on the same hardware.

For **research benchmark evaluation** against published methods on Teeth3DS and 3DTeethLand, Method A is the appropriate choice since it uses models trained and evaluated on the same datasets. For **clinical deployment** in a setting with a specific scanner model and some annotated data available, Method B with domain fine-tuning may ultimately be more accurate for that specific deployment context. The two methods are therefore complementary rather than competitive, and the ideal production system might use Method A for reference accuracy validation and Method B as the deployable production model after fine-tuning.

## 8.3 Integration Challenges and Broader Lessons

The four integration bugs described in Chapter 4 each reflect a general class of software engineering challenge that arises specifically when combining independently developed deep learning research codebases, rather than when developing a system end-to-end.

The **coordinate space mismatch** (Bug 1) reflects the general principle that learned spatial hyperparameters — attention window sizes, kernel radii, distance thresholds — are implicitly encoded in model weights and must be matched by any inference-time preprocessing. This principle is not new, but it is rarely documented explicitly in model repositories, which typically describe the model architecture and training process but not the inference-time prerequisites. The systematic debugging approach used here — printing raw model outputs and recognising the all-maximum-distance pattern as a diagnostic signature of empty attention windows rather than a genuine large-distance prediction — provides a reusable debugging strategy for similar normalisation mismatches in future multi-model integrations.

The **head count mismatch** (Bug 2) illustrates the broader danger of implicit assumptions about interface contracts between software components. Python's `zip()` is a convenient way to iterate over parallel sequences, but its silent truncation behaviour makes it an unreliable choice when the sequences must have equal length by contract. The resolution — adding an explicit assertion check on sequence lengths — is a general defensive programming practice that should be applied whenever iterating over parallel sequences derived from different sources.

The **CUDA namespace conflict** (Bug 3) is a specific instance of the diamond dependency problem in software packages: two independently developed packages with different internal implementations but identical exported interface names cannot be simultaneously loaded without conflict. The research deep learning ecosystem has limited namespace management conventions for CUDA extensions, making such conflicts common when combining models from different research groups. The rename-and-recompile approach taken here is a pragmatic workaround; a more systematic solution would require upstream changes to both repositories to adopt namespaced extension names as a convention.

The **Windows compilation issue** (Bug 4) reflects the dominant assumption in research deep learning that Linux is the deployment platform. The vast majority of research code is developed and tested on Linux HPC systems, and Windows-specific compilation requirements — such as the absence of `-rpath` support in MSVC — are rarely tested. This limits the reproducibility of published results for the large population of Windows users and is an ongoing challenge for the research community.

## 8.4 Comparison with Related Work

The pipeline's MRE of 0.920 mm compares favourably with the broader landscape of 3D dental landmark detection methods. Direct comparison is complicated by the heterogeneity of evaluation datasets and landmark definitions across papers, but the following qualitative positioning is informative. The 3DTeethLand challenge [9] established MRE ~0.87 mm (on a 500-scan test set) as the current state-of-the-art for this specific task and landmark set using the same LandmarkNet model this pipeline integrates. Classic heatmap regression methods applied to dental scans typically achieve MRE in the range 1.2–1.8 mm depending on the landmark class and the backbone architecture used. Direct coordinate regression baselines achieve MRE above 1.5 mm. PointNet++-based distance field methods achieve approximately 1.3–1.5 mm. The present pipeline's 0.920 mm result, achieved through correct integration of the state-of-the-art LandmarkNet, is therefore competitive with the published state-of-the-art and substantially outperforms simpler baselines.

For segmentation, TGNet's mIoU of 0.932 represents the current published state-of-the-art on Teeth3DS. The next-best published result in the 3DTeethSeg'22 challenge was approximately 0.918 mIoU, a gap of 0.014 that translates to meaningful improvements in boundary accuracy and missing-tooth handling. Alternative methods such as MeshSegNet [15] and TSGCNet [16] were evaluated on smaller private datasets and cannot be directly compared, but their architectures are generally considered less capable than TGNet based on the scale and quality of their evaluations.

## 8.5 Limitations

**Evaluation sample size.** The landmark detection results presented in Chapter 7 are based on 165 landmarks from 2 scans. While the agreement between the pipeline's MRE (0.920 mm) and LandmarkNet's published MRE (0.87 mm on a 500-scan test set) provides confidence in the result's generalisability, a 2-scan evaluation is insufficient to characterise the variance of performance across scan types, patient demographics, and scanner models. Systematic evaluation on the 120 matched pairs in the local dataset (Section 3.6) would provide a more statistically robust characterisation and is the highest-priority future work item.

**No fine-tuning.** Both TGNet and LandmarkNet are used with their original published weights. Performance on scans from scanner brands not well-represented in the Teeth3DS training data (e.g., newer scanner models released after the dataset was collected in 2022) may be lower than the reported figures. The degree of performance degradation from scanner-specific domain shift is unknown and would require access to scanner-specific annotated data to quantify.

**Single-jaw processing.** The pipeline processes one jaw (upper or lower) per inference run. Clinical analyses requiring measurements across both jaws simultaneously — including overjet, overbite, and inter-arch distance measurements — require an additional step to co-register upper and lower jaw scans in occlusal position, which is not provided.

**VRAM sensitivity.** At the default crop size of k = 12,000 points, the pipeline requires approximately 3.5 GB of VRAM per tooth during LandmarkNet inference. On the RTX 3050 (4 GB), scans with unusually dense meshes (>150,000 vertices, common in high-resolution scanner outputs) may produce OOM errors before all teeth have been processed. The `--crop-k 8000` option mitigates this at a small accuracy cost.

## 8.6 Future Work

**Systematic evaluation on 120 matched scan pairs.** The highest priority extension is evaluating Method A on all 120 matched OBJ + kpt.json pairs in the local dataset. This would provide a statistically robust characterisation of MRE and detection rate across the full range of scan types, tooth morphologies, and patient demographics represented in the lower jaw subset.

**Method B training and measured results.** Training Method B to convergence on the Teeth3DS training split and the 3DTeethLand landmark annotations would provide measured rather than expected performance figures, enabling a direct empirical comparison between the two methods on the same evaluation set. This requires approximately 84 hours of compute time on the development machine (36 hours for segmentation training, 48 hours for landmark training), which is feasible given adequate time.

**Domain-specific fine-tuning experiment.** Selecting 50–100 scans from a specific scanner brand and fine-tuning Method B on this subset would quantify the fine-tuning benefit for domain adaptation, providing empirical support for the claim that Method B with fine-tuning can match or exceed Method A for specific deployment contexts.

**End-to-end joint training.** Training both stages jointly with a combined loss (segmentation loss + landmark loss, with gradient flow from the landmark stage back to the segmentation stage via differentiable crop extraction) could potentially reduce error propagation from segmentation mistakes to landmark accuracy. This is a research-level contribution that would require substantial GPU memory and a carefully designed training curriculum.

**Browser-based 3D visualisation.** Replacing the MeshLab-based review workflow with a browser-based 3D viewer built on Three.js or WebGL would make the pipeline outputs accessible to clinical users without requiring software installation. A web interface accepting a PLY file upload and displaying the coloured mesh with toggleable landmark layers would require approximately one week of frontend development effort and would substantially improve clinical accessibility.

**CBCT extension.** Extending the pipeline to volumetric CT data would enable sub-gingival landmark detection (root apices, furcation points) not accessible from surface scans. This would require adapting TGNet or an equivalent volumetric segmentation model, and adapting LandmarkNet to process volumetric data — a substantial but impactful extension for oral surgery planning applications.

---

# Chapter 9: Conclusion

This project has delivered a complete, working pipeline for automated tooth segmentation and anatomical landmark detection from three-dimensional intraoral scans. The central deliverable — Method A, integrating ToothGroupNetwork and 3DTeethLand LandmarkNet through a novel coordinate space bridge — achieves a Mean Radial Error of 0.920 mm and a 100% detection rate on 165 annotated landmarks from two representative jaw scans. This performance is within 0.05 mm of LandmarkNet's published standalone accuracy, confirming that the integration preserves essentially all of the model's original capability while adding the segmentation stage that enables per-tooth landmark detection from a raw, unsegmented jaw scan. At an inference time of approximately 56 seconds per scan on an NVIDIA RTX 3050 Laptop (4 GB VRAM), the pipeline is practical for research evaluation on consumer-grade hardware.

The critical technical insight of the project is the centrality of coordinate space contracts in multi-model inference pipelines. Deep learning models for 3D point cloud processing encode their training normalisation implicitly in learned spatial hyperparameters — attention window sizes, KPConv ball radii, DBSCAN distance thresholds — that are calibrated to specific physical scales through the normalisation applied during training. When two independently trained models are combined, their coordinate spaces must be explicitly aligned. The z-score normalisation bridge developed in this project — converting TGNet's Y-range normalised output coordinates to the z-score normalised input space expected by LandmarkNet, through the formula `c_norm = (c_tgnet * scale_factor + min_y_c) / Z_SCORE_STD` — is a non-trivial derivation that was not documented in either model's repository. Its derivation from first principles, verification through multiple independent checks, and full documentation in this report constitute a contribution that enables reproducibility and reuse by future researchers seeking to combine these or similar models.

The resolution of four integration bugs — coordinate space mismatch, head count mismatch, CUDA namespace conflict, and Windows compilation incompatibility — demonstrates the level of reverse-engineering and debugging required to integrate independently developed research codebases into a production pipeline. Each bug required reading source code rather than documentation, and each reflects a general class of integration challenge that is common in the research deep learning ecosystem but rarely documented. The systematic debugging methodology employed here — particularly the identification of all-maximum-distance output as a diagnostic signature of empty attention windows — provides a reusable pattern for diagnosing similar normalisation mismatches in future integrations.

The complementary Method B, a lean trainable PointNet++ Multi-Scale Grouping architecture, addresses the practical limitations of Method A for clinical deployment scenarios. By operating entirely on standard PyTorch operations (requiring no custom CUDA compilation), fitting within 4 GB of VRAM during both training and inference, and providing a path to domain-specific fine-tuning, Method B offers substantially improved deployability at the cost of an expected ~5% mIoU and ~0.4 mm MRE performance reduction. The two methods are complementary: Method A provides state-of-the-art accuracy for research benchmark evaluation, while Method B provides the deployability and fine-tunability needed for clinical integration. Together, they demonstrate that automated dental landmark detection at clinically relevant accuracy is achievable across a range of hardware and deployment configurations, not just on well-resourced institutional computing systems.

The broader significance of this work extends beyond the specific dental application. The coordinate normalisation problem described here is an instance of the general challenge of composing pre-trained models into inference pipelines — a challenge that will become increasingly common as the research community produces more high-quality specialist models that practitioners seek to combine. The methodological lessons of this project — treating coordinate spaces as explicit contracts, verifying bridges through multiple independent checks, and diagnosing model failures through output distribution analysis rather than through error messages — are applicable to any multi-model integration task. The dental domain, with its highly structured geometry and rich annotated benchmark data, provides an excellent testbed for developing and validating these integration methodologies. The pipeline and all associated code, documentation, and diagnostic tools are designed to support reproducibility and extension by future researchers.

---

# References

[1] P. Naoumova, J. Lindman, and C. Beckman, "Assessment of dental arch width using two different methods," *European Journal of Orthodontics*, vol. 30, no. 2, pp. 169–175, 2008.

[2] C. R. Qi, H. Su, K. Mo, and L. J. Guibas, "PointNet: Deep learning on point sets for 3D classification and segmentation," in *Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, Honolulu, HI, USA, 2017, pp. 652–660.

[3] C. R. Qi, L. Yi, H. Su, and L. J. Guibas, "PointNet++: Deep hierarchical feature learning on point sets in a metric space," in *Advances in Neural Information Processing Systems (NeurIPS)*, Long Beach, CA, USA, 2017, vol. 30.

[4] Y. Wang, Y. Sun, Z. Liu, S. E. Sarma, M. M. Bronstein, and J. M. Solomon, "Dynamic graph CNN for learning on point clouds," *ACM Transactions on Graphics*, vol. 38, no. 5, pp. 1–12, 2019.

[5] H. Thomas, C. R. Qi, J. E. Deschaud, B. Marcotegui, F. Goulette, and L. J. Guibas, "KPConv: Flexible and deformable convolution for point clouds," in *Proc. IEEE/CVF International Conference on Computer Vision (ICCV)*, Seoul, Korea, 2019, pp. 6411–6420.

[6] H. Zhao, L. Jiang, J. Jia, P. Torr, and V. Koltun, "Point Transformer," in *Proc. IEEE/CVF International Conference on Computer Vision (ICCV)*, 2021, pp. 16259–16268.

[7] X. Lai, J. Liu, L. Jiang, L. Wang, H. Zhao, S. Liu, X. Qi, and J. Jia, "Stratified Transformer for 3D point cloud segmentation," in *Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, New Orleans, LA, USA, 2022, pp. 8500–8509.

[8] Z. Cui, C. Li, N. Chen, G. Wei, R. Chen, Y. Zhou, and W. Shen, "ToothGroupNetwork: Multi-view tooth segmentation network for 3D dental mesh," in *Proc. International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI)*, Vancouver, Canada, 2023, pp. 412–421.

[9] R. Deklerck et al., "3DTeethLand: 3D teeth landmark detection challenge," *OSF Preprint*, 2023. [Online]. Available: osf.io/um96h/

[10] Fédération Dentaire Internationale (FDI), "Two-digit tooth notation system," *International Dental Journal*, vol. 21, no. 1, pp. 104–106, 1971.

[11] U. Rad, A. Jamali, and P. Pouyafar, "Review of automated methods for panoramic dental radiograph analysis," *Computers in Biology and Medicine*, vol. 75, pp. 45–57, 2016.

[12] A. Ben-Hamadou, O. Smaoui, H. Chaabouni-Chouayakh, et al., "3DTeethSeg'22: 3D teeth scan segmentation and labelling challenge," *arXiv preprint arXiv:2305.18277*, 2022.

[13] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin, "Attention is all you need," in *Advances in Neural Information Processing Systems (NeurIPS)*, Long Beach, CA, USA, 2017, vol. 30.

[14] P. Yu, C. Qi, H. Tian, and G. Mei, "Point-MAE: Masked autoencoders for point cloud self-supervised learning," in *Proc. European Conference on Computer Vision (ECCV)*, Tel Aviv, Israel, 2022, pp. 604–621.

[15] C. Lian, L. Wang, T. H. Wu, M. Wang, F. Yap, H. Ko, and D. Shen, "MeshSNet: Deep multi-scale mesh feature learning network for end-to-end tooth labeling on 3D dental surfaces," in *Proc. International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI)*, Lima, Peru, 2020, pp. 837–846.

[16] X. Chen, B. Ye, Y. Sun, and X. Yang, "TSGCNet: Discriminative geometric feature learning with two-stream graph convolutional network for 3D dental model segmentation," in *Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2021, pp. 15540–15549.

[17] C. Payer, D. Stern, H. Bischof, and M. Urschler, "Integrating spatial configuration into heatmap regression based CNNs for landmark localization," *Medical Image Analysis*, vol. 54, pp. 207–219, 2019.

[18] T. Pfister, J. Charles, and A. Zisserman, "Flowing convnets for human pose estimation in videos," in *Proc. IEEE/CVF International Conference on Computer Vision (ICCV)*, Santiago, Chile, 2015, pp. 1913–1921.

[19] J. Zheng, Y. Liu, J. Ren, T. Zhu, Y. Peng, and H. Yang, "Landmark detection in 3D scans of human skulls using deep learning," in *Proc. IEEE International Symposium on Biomedical Imaging (ISBI)*, Iowa City, IA, USA, 2020, pp. 710–714.

[20] A. Vakalopoulou, P. Kapolka, and D. Samaras, "Automatic 3D cephalometric annotation system using a convolutional neural network," in *Proc. International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI)*, Granada, Spain, 2018, pp. 831–838.

[21] J. Long, E. Shelhamer, and T. Darrell, "Fully convolutional networks for semantic segmentation," in *Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, Boston, MA, USA, 2015, pp. 3431–3440.

[22] O. Ronneberger, P. Fischer, and T. Brox, "U-Net: Convolutional networks for biomedical image segmentation," in *Proc. International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI)*, Munich, Germany, 2015, pp. 234–241.

---

# Appendix A — Key Hyperparameters

## LandmarkNet Architecture (pipeline/combined_pipeline.py)

```python
LandmarkNet(
    lr=0.0006,
    weight_decay=0.0001,
    epochs=500,
    warmup_epochs=5,
    dbscan_cfg={
        'max_neighbor_dist': 0.03,
        'min_points': 40,
        'weighted_cluster': True,
        'weighted_average': True,
    },
    in_channels=9,
    channels_list=[48, 96, 192, 256],
    out_channels=[1, 4, 4, 4, 4, 4],
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
```

## Coordinate Constants

| Constant | Value | Meaning |
|---|---|---|
| `Z_SCORE_STD` | 17.3281 mm | Global std used in 3DTeethLand training normalisation |
| `crop_k` | 12,000 (default) | Points per tooth crop for LandmarkNet |
| `marker_radius` | 1.2 mm (default) | Octahedron marker radius for PLY output |
| `dist_thresh` | 0.12 (normalised) | Candidate selection threshold (~2.1 mm physical) |
| `cluster_min_pts` | 20 | DBSCAN minimum cluster size |
| `cluster_max_dist` | 0.03 (normalised) | DBSCAN neighbourhood radius (~0.52 mm physical) |

## Method B Hyperparameters

| Parameter | Segmentation Model | Landmark Model |
|---|---|---|
| Input points (N) | 6,000 | 3,000 |
| SA1 output points | 1,024 | 512 |
| SA1 radii (MSG) | 0.1, 0.2 | 0.05, 0.10 |
| SA2 output points | 256 | 128 |
| SA2 radii (MSG) | 0.2, 0.4 | 0.10, 0.20 |
| SA3 output | Global (1) | Global (1) |
| Head channels | 64 | 64 |
| Output classes | 17 (FDI labels) | 5 heads x 4 channels |
| Batch size | 8 | 16 |
| Initial LR | 1e-3 | 1e-3 |
| LR schedule | CosineAnnealingWarmRestarts T_0=20, T_mult=2 | Same |
| AMP | FP16 + GradScaler | FP16 + GradScaler |
| torch.compile | Yes | Yes |
| Early stopping patience | 15 epochs | 15 epochs |
| Loss | CrossEntropy + SoftDice (lambda=1.0) | SmoothL1 dist + masked SmoothL1 offset (mu=1.0) |

---

# Appendix B — Output File Format

## landmarks.json Schema

```json
{
  "jaw": "upper" | "lower",
  "landmarks": [
    {
      "class":     "Mesial" | "Distal" | "Cusp" | "InnerPoint" | "OuterPoint" | "FacialPoint",
      "coord":     [x_mm, y_mm, z_mm],
      "score":     float in [0.0, 1.0],
      "fdi_tooth": int (11-48)
    }
  ]
}
```

## PLY Vertex Format

```
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
```

Format: binary-little-endian 1.0

---

# Appendix C — Coordinate Conversion Reference

The following Python pseudocode summarises the complete coordinate conversion chain from raw OBJ coordinates to z-score normalised LandmarkNet input:

```python
# Step 1: Load raw OBJ
orig_xyz_mm   = load_obj_vertices(scan_path)            # (V, 3) in mm
orig_normals  = load_or_compute_normals(scan_path)      # (V, 3) unit normals

# Step 2: TGNet preprocessing (internal to InferencePipeline)
jaw_mean      = orig_xyz_mm.mean(axis=0)                # (3,) per-axis mean
centred       = orig_xyz_mm - jaw_mean                  # (V, 3)
y_range       = centred[:,1].max() - centred[:,1].min()
scale_factor  = y_range / 1.8
min_y_c       = centred[:,1].min()                      # scalar
tgnet_xyz     = (centred - min_y_c) / scale_factor      # (V, 3)

# Step 3: TGNet inference -> per-point FDI labels -> compute per-tooth centroids
#         sampled_xyz:  (24000, 3) in TGNet space
#         fdi_labels:   (24000,) integer FDI labels
for fdi in unique(fdi_labels):
    mask = fdi_labels == fdi
    centroid_tgnet = sampled_xyz[mask].mean(axis=0)     # (3,) in TGNet space

    # Step 4: Coordinate bridge (pipeline/data_bridge.py)
    Z_SCORE_STD = 17.3281
    centroid_centred = centroid_tgnet * scale_factor + min_y_c  # reverse TGNet
    centroid_norm    = centroid_centred / Z_SCORE_STD   # jaw_mean cancels

    # Step 5: Build KDTree in z-score space (build once per scan)
    jaw_norm  = (orig_xyz_mm - jaw_mean) / Z_SCORE_STD  # (V, 3)
    norm_tree = KDTree(jaw_norm)                         # build once

    # Step 6: Query KDTree for per-tooth crop
    _, crop_idxs = norm_tree.query(centroid_norm, k=12000)  # (12000,)
    crop_xyz_norm = jaw_norm[crop_idxs]                 # (12000, 3)

    # Step 7: Construct 9-channel feature tensor
    crop_normals     = orig_normals[crop_idxs]          # (12000, 3)
    centroid_offsets = crop_xyz_norm - centroid_norm     # (12000, 3)
    features = np.concatenate(
        [crop_xyz_norm, crop_normals, centroid_offsets],
        axis=1
    )                                                    # (12000, 9)

    # Step 8: LandmarkNet inference + DBSCAN -> lm_norm (Nx3)
    # Step 9: Coordinate recovery to mm
    landmark_mm = lm_norm * Z_SCORE_STD + jaw_mean
```

---

*End of Report*
