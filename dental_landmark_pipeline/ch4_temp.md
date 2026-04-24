
# Chapter 4: Methodology — Method A: TGNet + LandmarkNet Pipeline

## 4.1 System Architecture Overview

Method A is a cascade pipeline structured as four sequential processing stages that transform a raw jaw mesh file into a structured set of FDI-labelled, millimetre-precision 3D landmark coordinates:

```
Input: jaw_scan.obj
       |
       v
+----------------------------------+
|  Stage 1: TGNet Segmentation     |  --> per-vertex FDI labels
|  (FPS network + BDL network)     |      sampled_xyz in TGNet space
+----------------------------------+
       |
       v
+----------------------------------+
|  Stage 2: Coordinate Bridge      |  --> tooth centroids in z-score space
|  (TGNet -> 3DTeethLand space)    |      + per-tooth point cloud crops (K=12000)
+----------------------------------+
       |
       v
+----------------------------------+
|  Stage 3: LandmarkNet            |  --> per-point distance fields (K,1)
|  (StratifiedTransformer)         |      + offset vectors (K,3) per head
+----------------------------------+
       |
       v
+----------------------------------+
|  Stage 4: Post-processing        |  --> final (x,y,z) landmark coords
|  (DBSCAN cluster extraction)     |      with FDI label + class + score
+----------------------------------+
       |
       v
Output: {stem}_landmarks.json
        {stem}_mesh.ply
        {stem}_colored.ply
        {stem}_{Class}.ply  (x6)
```

The cascade design was chosen over a joint multi-task architecture for three concrete reasons. First, TGNet and LandmarkNet are both mature, publicly released models with published state-of-the-art performance on their respective benchmark tasks. Retraining either from scratch as part of a joint architecture would sacrifice this established accuracy and require computational resources far beyond the 4 GB VRAM development platform. Second, the cascade structure enables independent validation of each stage: Stage 1 quality can be assessed via segmentation mIoU, and Stage 2 quality can be assessed via MRE and SDR independently, allowing the source of any accuracy limitation to be precisely identified. If the pipeline's MRE is close to LandmarkNet's published standalone MRE (~0.87 mm), this confirms that the integration has not introduced significant accuracy degradation, and any remaining gap is attributable to the integration rather than to either model's inherent capability. Third, the cascade enables modular replacement: if a better tooth segmentation model becomes available, it can replace TGNet without requiring any changes to the landmark detection stage, provided the coordinate bridge is updated to account for the new model's normalisation scheme.

The principal engineering challenge in assembling this cascade was defining and correctly implementing the coordinate space contract between the two models — specifically, the conversion from TGNet's Y-range normalised output coordinates to the z-score normalised input space required by LandmarkNet. This mismatch, its symptoms, its diagnosis, and its resolution are the central technical narrative of this chapter, described in full in Section 4.3.

## 4.2 Stage 1 — Tooth Instance Segmentation (TGNet)

### 4.2.1 Model Overview

ToothGroupNetwork (TGNet) [8] is a two-stage point cloud segmentation network designed specifically for intraoral scan segmentation at the instance level. Its distinguishing feature relative to general-purpose point cloud segmentation methods is the explicit treatment of the tooth grouping problem as separate from semantic labelling: a first stage predicts coarse per-point FDI class labels, and a second stage uses boundary information to produce clean, clinically usable per-instance masks. TGNet was trained on the Teeth3DS training split (1,200 scans) using a combination of semantic cross-entropy loss, instance grouping loss, and explicit boundary detection loss. Its checkpoint files — `tgnet_fps.h5` (Stage 1 FPS network) and `tgnet_bdl.h5` (Stage 2 BDL network) — are stored in HDF5 format as produced by the original TensorFlow-based training implementation.

### 4.2.2 Input Preprocessing

The raw OBJ mesh is read by a custom parser that extracts the vertex list (x, y, z positions) and the face list (triangular face vertex indices). Per-vertex normals are read from `vn` entries if present in the OBJ file, or computed from the mesh faces using area-weighted face normal averaging if absent. TGNet's internal preprocessing applies two transformations to the raw vertex positions. First, the mesh is centred by subtracting the per-axis mean of all vertices. Second, a uniform scaling is applied based on the Y-axis range of the centred mesh:

```
jaw_mean      = mean(orig_xyz, axis=0)        # (3,) per-axis mean
centred       = orig_xyz - jaw_mean            # centre the mesh
y_range       = max(centred[:,1]) - min(centred[:,1])
min_y_c       = min(centred[:,1])              # minimum centred Y (scalar)
scale_factor  = y_range / 1.8
tgnet_xyz     = (centred - min_y_c) / scale_factor
```

This maps the centred Y-axis extent to the range [0, 1.8], and because the same `scale_factor` is applied to all three axes, the physical aspect ratio of the tooth shapes is preserved. The transformation maps typical jaw scans to a coordinate range of approximately [-0.8, 1.0] in all axes. The values `jaw_mean`, `min_y_c`, and `scale_factor` are preserved as auxiliary variables throughout the pipeline because they are required to invert the TGNet normalisation in the coordinate bridge (Stage 2).

### 4.2.3 Stage 1a — Farthest Point Sampling Network

The FPS sub-stage reduces the full-resolution mesh (80,000–200,000 vertices for typical jaw scans) to a fixed set of 24,000 representative points using iterative farthest point sampling. The FPS algorithm is a greedy spatial coverage procedure:

```
FPS pseudocode:
  selected = [random_starting_index]
  min_dists = distance(all_points, selected[0])
  for i in range(1, 24000):
      next_idx = argmax(min_dists)
      selected.append(next_idx)
      new_dists = distance(all_points, all_points[next_idx])
      min_dists = elementwise_min(min_dists, new_dists)
  return all_points[selected]
```

The key property of FPS is spatial uniformity: each successive sample maximises its minimum distance to all previously selected samples, guaranteeing that the resulting 24,000-point set covers the surface with approximately uniform spatial density regardless of the local vertex density of the input mesh. This is critical for downstream processing: dense mesh regions (such as high-resolution cusp tip captures) and sparse regions (such as flat gingival surfaces) both contribute proportionally to their surface area rather than their vertex count. The 24,000-point subsampled set is the input to TGNet's neural network stages, and this fixed size ensures consistent memory usage across scans with widely varying input sizes.

The computational cost of FPS for N = 150,000 input vertices and S = 24,000 output samples is O(N * S) = 3.6 billion distance comparisons. This is reduced to approximately 0.3 seconds of wall time on the RTX 3050 via the `tgnet_ops` CUDA extension, which parallelises the distance comparisons across CUDA thread blocks. A CPU-only implementation of the same computation would require approximately 90 seconds, making the CUDA extension non-negotiable for real-time inference. Each of the 24,000 sampled points is processed by a PointNet-style hierarchical encoder that computes features by applying shared MLPs to local k-nearest-neighbour groups at multiple spatial scales, producing a (24,000, C) feature tensor alongside per-point semantic label logits.

### 4.2.4 Stage 1b — Boundary-aware Deep Learning Network

The BDL network operates on the 24,000 sampled points and their features from Stage 1a, producing instance-level tooth labels through three parallel prediction heads:

The **semantic head** predicts the FDI tooth label for each of the 24,000 sampled points as a 17-class probability distribution (gingiva + 16 tooth classes, corresponding to FDI labels 0, 11–18, 21–28 for upper and 31–38, 41–48 for lower). This head is trained with standard cross-entropy weighted by inverse class frequency to compensate for the gingiva dominance in the training data.

The **boundary head** predicts the probability that each point lies on the boundary between two adjacent tooth instances. Points within a specified distance of a ground-truth instance boundary (approximately 0.5 mm in normalised units) are labelled as positive boundary examples. The boundary head uses binary cross-entropy with positive examples upweighted by a factor of 10 relative to background, since boundary points constitute only a small fraction of all points (approximately 2–5%) but carry disproportionate clinical importance. The boundary head's output is a heatmap of inter-tooth contact regions: high values indicate that the model believes two different tooth instances meet at this point.

The **instance head** combines semantic label predictions and boundary probability predictions to produce clean per-instance segmentation masks. Points with low boundary probability are assigned to the tooth instance predicted by the semantic head. Points with high boundary probability are assigned to the nearest instance centroid in a learned instance embedding space, where centroids are computed as the weighted mean of all non-boundary points assigned to each FDI label. This two-pass strategy prevents the semantic head's occasional boundary errors from being directly inherited by the final instance labels.

The boundary-aware design addresses the most clinically critical failure mode of dental segmentation. Standard cross-entropy training achieves high overall accuracy by correctly labelling the many gingiva and tooth-interior points, potentially at the cost of misclassifying the smaller number of boundary points. However, it is precisely the boundary region — the few hundred vertices at the inter-proximal contact between adjacent teeth — that determines whether the segmentation is clinically usable. Misclassified boundary vertices produce merged tooth instances whose landmark predictions are attributed to the wrong FDI label, directly reducing clinical usability. The boundary penalty forces the network to devote sufficient gradient signal to correctly classifying these critical few hundred points per tooth boundary. TGNet's boundary F1 score of 0.887 reflects this design emphasis.

### 4.2.5 Error Propagation from Segmentation to Landmark Detection

Understanding how TGNet segmentation errors affect downstream landmark detection quality is important for interpreting the per-class MRE results in Chapter 7. TGNet's mIoU of 0.932 means the vast majority of teeth in a scan are correctly segmented, but a small fraction will have errors that propagate to the landmark stage.

**Boundary merging errors** are the most consequential. When adjacent teeth (most commonly canine-premolar or premolar-molar pairs in crowded arches) are partially merged at their inter-proximal contact, the landmark detector crop for the merged instance includes surface geometry from both teeth. The Mesial and Distal landmark heads receive ambiguous evidence — the contact surface geometry of one tooth faces the buccal surface geometry of the adjacent tooth — and may produce landmarks at the interface rather than at the correct contact points. This directly explains the elevated MRE for Distal (1.309 mm) and Mesial (0.879 mm) landmarks relative to InnerPoint (0.615 mm) and OuterPoint (0.628 mm), which lie in the tooth interior and are unaffected by boundary merging.

**Third molar omission** occurs when a partially erupted third molar is classified as gingiva in its entirety due to limited visible crown area. This produces no crop and no landmark prediction for that tooth. Since the evaluation protocol counts unmatched ground-truth landmarks as missed detections, this failure mode reduces detection rate but does not affect MRE (which is computed only over matched pairs).

**Quadrant confusion** is rare (<0.5% of scans) but produces a specific error pattern: a midline tooth is assigned to the wrong quadrant (e.g., FDI 11 vs 21), causing the Mesial/Distal midline assignment to be inverted for that tooth, swapping the two contact point labels.

## 4.3 Stage 2 — Coordinate Space Bridge

### 4.3.1 Motivation and Problem Framing

LandmarkNet was trained with point cloud coordinates normalised using a global z-score operation:

```
norm_coord = (mm_coord - per_scan_mean) / sigma
```

where sigma = 17.3281 mm is a fixed constant estimated from the standard deviation of scan coordinates across the 3DTeethLand training set, and `per_scan_mean` is the per-scan centroid in millimetres. All spatial hyperparameters of LandmarkNet are defined in this normalised space: attention window sizes [0.1, 0.2, 0.4] (~1.73 mm, ~3.46 mm, ~6.93 mm), KPConv ball radius 0.05 (~0.87 mm), and DBSCAN distance threshold 0.03 (~0.52 mm). The critical point is that these hyperparameters were calibrated during training to produce correct behaviour at specific physical scales, and they only produce correct behaviour if the input coordinates are in the expected normalised space.

TGNet's output coordinate space spans approximately [-0.8, 1.0] in all axes, with total extent approximately 1.8 units. The z-score normalised space spans approximately [-3, +3] in all axes, with total extent approximately 6 units. The two spaces therefore differ not only in absolute scale but also in the location of the coordinate origin (TGNet places the origin at the minimum centred Y-coordinate offset, while z-score normalisation places it at the per-scan mean). A direct substitution of TGNet coordinates into LandmarkNet — even after a naive uniform rescaling — would leave the centroid offset features (channels 6–8 of the 9-channel feature vector) internally inconsistent with the coordinate channels (channels 0–2), because offsets computed in TGNet space have different magnitudes from offsets computed in z-score space. This inconsistency is the deep cause of the all-maximum-distance failure signature observed in the initial integration.

The coordinate bridge must accomplish three things: (1) convert TGNet normalised tooth centroids to z-score normalised coordinates; (2) extract the k nearest original-mesh vertices in z-score space; and (3) compute centroid offset features in z-score space. All three must be done consistently to produce a well-formed 9-channel feature tensor.

### 4.3.2 The Debugging Narrative

The normalisation mismatch was discovered through empirical debugging rather than documentation, because the issue is not described in either model's repository. The investigation proceeded through five identifiable steps.

**Step 1 — Observing the failure.** The initial pipeline ran without any Python exception. TGNet produced 14 tooth labels for the test scan. The coordinate bridge ran and produced 14 per-tooth feature tensors. LandmarkNet ran on each tooth and returned output tensors of shape (12000, 4) per head. The DBSCAN post-processing ran and completed. But the output JSON contained zero landmark entries. The DBSCAN detection threshold of `dist < 0.12` was never satisfied for any point in any tooth.

**Step 2 — Inspecting raw model outputs.** A single-line diagnostic was inserted after the LandmarkNet forward pass:

```python
print(f"distances: min={dist.min():.4f} max={dist.max():.4f} mean={dist.mean():.4f}")
```

The output for every tooth, every head was: `distances: min=0.1948 max=0.1999 mean=0.1987`. Every predicted distance value was within 0.005 units of the maximum clamped value 0.20. The model was uniformly asserting "no landmark nearby" for every point in every crop.

**Step 3 — Reading training preprocessing code.** Reading the 3DTeethLand `dataset.py` training file revealed the preprocessing chain. The key lines were:

```python
# 3DTeethLand dataset.py (training)
scan_mean = xyz_mm.mean(axis=0)
xyz_norm = (xyz_mm - scan_mean) / 17.3281
# ... build KDTree in xyz_norm space
# ... compute centroid_norm = centroid_mm / 17.3281 (after mean subtraction)
centroid_offsets = crop_xyz_norm - centroid_norm
features = np.concatenate([crop_xyz_norm, normals, centroid_offsets], axis=1)
```

This confirmed: z-score normalisation with sigma = 17.3281 mm, per-scan mean subtraction before normalisation, KDTree in normalised space, and centroid offsets in normalised space.

**Step 4 — Deriving the conversion formula.** Understanding the TGNet normalisation inversion and z-score application, the algebraic formula was derived as shown in Section 4.3.3. The key insight that jaw_mean cancels in the composition (making the formula depend only on `scale_factor`, `min_y_c`, and `Z_SCORE_STD`) was verified both symbolically and numerically.

**Step 5 — Verification.** After implementing the bridge, the diagnostic print produced: `distances: min=0.0121 max=0.1943 mean=0.0847`. The spatially structured distribution (many points near zero for points close to landmarks, many points near 0.20 for distant points) confirmed that the model was now operating correctly. DBSCAN extracted 100 landmark clusters from the 14-tooth scan.

### 4.3.3 TGNet Normalisation Inversion

The TGNet forward transform applied to millimetre coordinates `p_mm`:

```
p_centred = p_mm - jaw_mean
p_tgnet   = (p_centred - min_y_c) / scale_factor
```

Inverting to recover millimetres from TGNet-space:

```
p_centred = p_tgnet * scale_factor + min_y_c
p_mm      = p_centred + jaw_mean
```

Composing with z-score normalisation (subtracting jaw_mean, dividing by Z_SCORE_STD):

```
p_norm = (p_mm - jaw_mean) / Z_SCORE_STD
       = (p_tgnet * scale_factor + min_y_c + jaw_mean - jaw_mean) / Z_SCORE_STD
       = (p_tgnet * scale_factor + min_y_c) / Z_SCORE_STD
```

The jaw_mean terms cancel exactly. The final formula in Python:

```python
Z_SCORE_STD = 17.3281
centroid_centred = centroid_tgnet * scale_factor + min_y_c
centroid_norm    = centroid_centred / Z_SCORE_STD
```

### 4.3.4 KDTree Crop Extraction

The KDTree is built over the full original mesh in z-score normalised space:

```python
jaw_norm  = (orig_xyz_mm - jaw_mean) / Z_SCORE_STD   # (V, 3)
norm_tree = KDTree(jaw_norm)
```

For each tooth, the k = 12,000 nearest vertices to the tooth centroid in normalised space are queried. This k value exactly matches the 3DTeethLand training configuration, ensuring crop density and spatial extent at inference match the training distribution.

### 4.3.5 Feature Construction

The 9-channel LandmarkNet input feature vector:

| Channels | Content | Computation |
|---|---|---|
| 0–2 | xyz in z-score normalised space | jaw_norm[crop_idxs] |
| 3–5 | per-vertex surface normal (unit vector) | orig_normals[crop_idxs] |
| 6–8 | offset from tooth centroid (z-score space) | crop_xyz_norm - centroid_norm |

```python
crop_xyz_norm     = jaw_norm[crop_idxs]               # (K, 3)
crop_normals      = orig_normals[crop_idxs]           # (K, 3)
centroid_offsets  = crop_xyz_norm - centroid_norm      # (K, 3)
features = np.concatenate(
    [crop_xyz_norm, crop_normals, centroid_offsets], axis=1
)                                                      # (K, 9)
```

The centroid offset features are particularly important: they provide each point with explicit information about its position within the tooth crown (mesial side, occlusal surface, buccal face) without requiring the model to infer this from absolute coordinates alone.

### 4.3.6 Verification and Validation of the Bridge

**Algebraic identity check.** For p_mm = [10.0, -5.0, 8.0] with jaw_mean = [0, -10, 5], y_range = 40 mm, min_y_c = -20.0: the forward transform gives p_tgnet = [1.350, 1.125, 1.035]. The bridge conversion gives p_norm = [1.350*22.222 + (-20.0)]/17.3281 = [10.0, 5.0, 3.0]/17.3281 = [0.577, 0.289, 0.173]. Independent computation: (p_mm - jaw_mean) / 17.3281 = [10, 5, 3] / 17.3281 = [0.577, 0.289, 0.173]. Values match to numerical precision.

**Before-and-after inference comparison.** Without bridge: all distances 0.19–0.20, zero DBSCAN clusters. With bridge: distances 0.01–0.19 (mean 0.085), 100 clusters on 14-tooth scan.

**Bounding box plausibility check.** All 100 predicted landmarks from 01F4JV8X_upper lie within the physical bounding box of the original scan (padded by 5 mm), confirming no sign errors or large-scale conversion mistakes.

## 4.4 Stage 3 — Landmark Detection (LandmarkNet)

### 4.4.1 Stratified Transformer Architecture

The Stratified Transformer [7] computes multi-head self-attention within non-overlapping spatial windows partitioned from the input point cloud bounding box. At each resolution level, the bounding box is divided into a regular grid with cell size equal to the window size parameter. Points are assigned to grid cells, and self-attention is computed within each cell independently, at cost O(W^2) per cell where W is the mean points per cell. Cross-window context is provided through stratified anchor points selected from a coarser grid and included in every window's attention computation.

The Conditional Relative Position Encoding (CRPE) module modulates attention logits by the 3D spatial relationship between each query-key pair. The relative displacement (p_key - p_query) is quantised into 80 spatial bins, and a learned per-bin bias is added to the raw dot-product attention score. CRPE allows the model to learn that spatially proximate points should attend more strongly to each other, without hardcoding a specific decay function, and without requiring any canonical orientation of the coordinate system — critical for dental scans arriving in arbitrary patient-head orientations.

The full LandmarkNet architecture configuration:

| Parameter | Value | Physical Interpretation |
|---|---|---|
| Input channels | 9 | xyz norm + normals + centroid offsets |
| Channel list | [48, 96, 192, 256] | Feature dimensions at each resolution level |
| Transformer depths | [3, 9, 3] | Self-attention blocks per level |
| Attention heads | [6, 12, 24] | Multi-head attention heads per level |
| Window sizes | [0.1, 0.2, 0.4] | ~1.73 mm, ~3.46 mm, ~6.93 mm physical |
| Downsample ratio | 0.26 | 26% of points retained at each downsampling |
| CRPE bins | 80 | Spatial quantisation bins for position encoding |
| KPConv ball radius | 0.05 | ~0.87 mm for initial point embedding |

### 4.4.2 KPConv Point Embedding

Before the Transformer layers, each input point is processed through a KPConv embedding module (radius 0.05 normalised units, ~0.87 mm) that aggregates features from nearby points using learnable kernel points. The KPConv embedding produces a 48-channel local geometric descriptor that captures surface curvature patterns, cusp sharpness, and normal variation at sub-millimetre scale. This enriched per-point representation provides the Transformer layers with informative starting features: the local curvature at a cusp tip (high convexity), central fossa (high concavity), and flat facial surface (low curvature) are all distinguishable in the KPConv features before any attention computation has occurred.

### 4.4.3 Multi-Head Output Structure

Five landmark heads and one segmentation head are attached to the final decoder feature map:

| Head | Output per point | Class |
|---|---|---|
| Segmentation | (K, 1) binary | Tooth surface vs background |
| Head 1 | (K, 4) | MesialDistal distance + offset |
| Head 2 | (K, 4) | FacialPoint distance + offset |
| Head 3 | (K, 4) | OuterPoint distance + offset |
| Head 4 | (K, 4) | InnerPoint distance + offset |
| Head 5 | (K, 4) | Cusp distance + offset |

The single MesialDistal head detects both Mesial and Distal jointly, with Mesial/Distal assignment performed in post-processing based on midline-relative position. This design reduces head count from six to five, simplifying the model while leveraging the fact that Mesial and Distal always appear in pairs on opposite sides of the same tooth, with their relative position entirely determined by the FDI quadrant assignment.

### 4.4.4 Why Window Size Calibration Is Critical

The three window sizes [0.1, 0.2, 0.4] are calibrated to three qualitatively distinct spatial scales of dental anatomy:

The **small window** (~1.73 mm) captures sub-cusp surface detail: the local curvature gradient at a cusp tip (high convexity), inner fossa (high concavity), and flat facial surface (near-zero curvature) are all distinguishable at this scale, providing the fine-grained geometric discriminant needed for sub-millimetre landmark localisation.

The **medium window** (~3.46 mm) captures a full cusp region or inter-proximal contact area. At this scale, a cusp apex (surrounded by sloping cusp faces on all sides) is geometrically distinct from an Outer Point (surrounded by flat buccal surface), enabling the model to discriminate between landmark classes that have similar local geometry but differ in their broader anatomical context.

The **large window** (~6.93 mm) captures the full visible crown of most single-rooted teeth. Whole-tooth context enables the Mesial/Distal discrimination beyond what is available from midline position alone, and allows the model to use the presence or absence of adjacent cusp tips as context for determining the tooth type and expected landmark pattern.

These calibrations are invalidated whenever input coordinates are at the wrong scale. A window of 0.1 in a coordinate space where the full jaw spans 200 units (raw millimetres) would contain fewer than one point per window on average, making every attention computation vacuous. The z-score normalisation with sigma = 17.3281 mm ensures that the coordinate range is approximately [-3, +3] units, so window sizes [0.1, 0.2, 0.4] correspond to 1.7%, 3.3%, and 6.7% of the full scan range — the calibrated physical scales described above.

## 4.5 Stage 4 — Post-Processing and Coordinate Recovery

### 4.5.1 Candidate Landmark Extraction

For each landmark head, candidate positions are computed from points with predicted distance below the detection threshold `dist_thresh = 0.12` normalised units (~2.1 mm):

```
candidate_coord[i]  = point_xyz[i] + predicted_offset[i]
candidate_weight[i] = (dist_thresh - dist[i]) / dist_thresh
```

Candidates with higher weight are produced by points very close to a landmark (low predicted distance), which provides both the most accurate offset direction and the highest confidence. With a crop of 12,000 points and the default threshold, typical posterior teeth produce 2,000–4,000 candidates per landmark head.

### 4.5.2 DBSCAN Clustering

DBSCAN is applied to the candidate set with parameters eps = 0.03 normalised units (~0.52 mm) and min_samples = 20. Clusters correspond to individual landmark instances; each cluster's weighted centroid is the predicted landmark position.

A worked example with five candidates from the InnerPoint head of a hypothetical first molar:

```
Candidate A: coord (0.142, 0.033, 0.071), weight 0.82
Candidate B: coord (0.148, 0.029, 0.075), weight 0.90
Candidate C: coord (0.151, 0.037, 0.069), weight 0.88
Candidate D: coord (0.145, 0.031, 0.073), weight 0.91
Candidate E: coord (0.201, 0.121, 0.055), weight 0.30

Pairwise distances:
  d(A,B)=0.009, d(A,C)=0.012, d(A,D)=0.006  <  eps=0.030 (all connected)
  d(A,E)=0.099, d(B,E)=0.095                 >  eps (E disconnected)

DBSCAN result:
  Cluster 1: {A, B, C, D}  (all mutually within eps, min_samples met)
  Noise:     {E}            (not reachable from any core point)

Weighted centroid of Cluster 1:
  lm = (0.82*A + 0.90*B + 0.88*C + 0.91*D) / 3.51
     ~ (0.147, 0.032, 0.072)
```

Point E is a noise candidate from a distant surface point with an overshooting offset prediction. DBSCAN correctly identifies it as noise and excludes it from the landmark position estimate. In practice, realistic candidate sets contain hundreds to thousands of points, and DBSCAN robustly separates distinct landmark instances (e.g., two premolar cusps separated by ~3 mm) because their candidate clouds do not overlap at the eps = 0.52 mm scale.

Noise points in practical inference arise from: (1) surface points just below the distance threshold with imprecise offset predictions, producing scattered candidates; and (2) candidates near the crop boundary where incomplete tooth context reduces offset accuracy. The min_samples = 20 requirement filters both noise sources effectively.

### 4.5.3 Confidence Score Computation

Each detected cluster receives a confidence score summarising the quality of the supporting evidence:

```
score = mean(candidate_weights in cluster)
      = mean((dist_thresh - dist[i]) / dist_thresh for i in cluster)
```

Scores near 1.0 indicate clusters supported by points very close to the landmark (low predicted distances) — the expected signature of geometrically prominent landmarks such as InnerPoint and OuterPoint. Scores near 0.0 indicate weakly expressed landmarks (ambiguous geometry, or landmark near the crop boundary). The score is stored in the output JSON and can be used for human-in-the-loop review, filtering low-confidence predictions for manual correction.

### 4.5.4 Coordinate Recovery and Mesial/Distal Assignment

Predicted landmark positions are recovered from normalised to millimetre space:

```
landmark_mm = landmark_norm * Z_SCORE_STD + jaw_mean
```

The MesialDistal head produces up to two clusters per tooth. Mesial/Distal assignment uses the FDI quadrant:

- **Quadrants 1 and 4** (right side, FDI 1X and 4X): smaller x-coordinate cluster is Mesial (faces toward the midline at x = 0).
- **Quadrants 2 and 3** (left side, FDI 2X and 3X): larger x-coordinate cluster is Mesial (also faces toward the midline).

This rule implements the FDI anatomical definition: Mesial is the tooth surface directed toward the median sagittal plane of the face.

## 4.6 Output Format

The pipeline produces a JSON landmark file with the following structure:

```json
{
  "jaw": "upper",
  "landmarks": [
    {
      "class":     "Mesial",
      "coord":     [12.34, -5.67, 8.90],
      "score":     0.87,
      "fdi_tooth": 21
    },
    {
      "class":     "Cusp",
      "coord":     [13.12, -4.88, 11.23],
      "score":     0.94,
      "fdi_tooth": 21
    }
  ]
}
```

Each entry contains the class name, 3D coordinate in millimetres (original scan coordinate system), confidence score in [0, 1], and FDI tooth label inherited from Stage 1 segmentation. The coordinate system is the raw scanner coordinate system of the input OBJ file, requiring no additional transformation for comparison with ground-truth annotations stored in the same system.

## 4.7 Integration Challenges and Resolutions

**Table 4.3 — Integration bugs: diagnosis and resolution**

| # | Bug | Symptom | Root Cause | Fix |
|---|---|---|---|---|
| 1 | Coordinate space mismatch | Zero landmarks; all distances 0.19–0.20 | TGNet normalised coordinates fed to z-score-trained model | Derived full conversion formula; built KDTree in z-score space |
| 2 | LANDMARK_HEADS count mismatch | Zero Cusp predictions | 6-entry list zip()d with 5-head model; Cusp silently dropped | Corrected to ["MesialDistal","FacialPoint","OuterPoint","InnerPoint","Cusp"] |
| 3 | CUDA namespace conflict | RuntimeError or silent symbol overwrite | Both extensions export same C++ function names | Renamed to tgnet_ops and teethland_ops throughout |
| 4 | Windows runtime_library_dirs | Build fails with MSVCCompiler AttributeError | Linux-only -rpath flag in setup.py | Removed runtime_library_dirs line from setup.py |

### 4.7.1 Bug 1 — Coordinate Space Mismatch

**Discovery:** Zero landmarks in output JSON. Diagnostic print of raw distance values showed all-maximum pattern (0.1948–0.1999) for every point in every crop on every tooth.

**What incorrect behaviour looked like:** The pipeline completed all stages without exception. Output tensors were shaped correctly. The failure was entirely silent from the perspective of Python's exception handling. Only explicit inspection of the numerical values of the model's outputs revealed the problem.

**Diagnostic steps:** (1) Observed all-maximum distance values. (2) Identified the all-maximum pattern as the LandmarkLoss sentinel for "point far from all landmarks" — confirmed by reading the LandmarkLoss implementation. (3) Read 3DTeethLand training `dataset.py` to identify the z-score normalisation with sigma = 17.3281 mm and the per-scan mean subtraction. (4) Derived conversion formula algebraically. (5) Implemented bridge and verified with test scan.

**Fix:** `compute_jaw_norm()` in `pipeline/data_bridge.py` applies z-score normalisation to original mesh vertices; `convert_centroid_to_norm()` applies the composed TGNet-to-z-score conversion to each tooth centroid. Both functions are called in the pipeline's Stage 2 code path.

**Verification:** Post-fix distance distribution on test scan: min 0.0121, max 0.1943, mean 0.0847. All 100 landmarks within physical scan bounding box.

### 4.7.2 Bug 2 — LANDMARK_HEADS Count Mismatch

**Discovery:** After Bug 1 fix, per-class output count showed zero Cusp predictions despite all other five classes having predictions. Since cusp tips are geometrically prominent features, zero Cusp detections strongly indicated a code-level issue.

**What incorrect behaviour looked like:** Predictions for five classes appeared plausible. No exception, no warning. The only visible symptom was the absence of Cusp entries in the output JSON, which required explicitly inspecting per-class counts to notice.

**Diagnostic steps:** (1) Logged per-class output count. (2) Observed zero Cusp count. (3) Traced head assignment loop — found six-entry name list zip()d with five-head output. (4) Read LandmarkNet model source — identified correct five-head order: MesialDistal (combined), FacialPoint, OuterPoint, InnerPoint, Cusp. (5) Added assertion: `assert len(head_names) == len(model_outputs)`.

**Fix:** Head name list changed to `["MesialDistal", "FacialPoint", "OuterPoint", "InnerPoint", "Cusp"]`. Assertion guard added.

**Verification:** Post-fix per-class counts: MesialDistal ~28 (2 per tooth * 14 teeth), FacialPoint ~14, OuterPoint ~14, InnerPoint ~14, Cusp ~30 (averaging ~2 per tooth for mixed anterior-posterior dentition).

### 4.7.3 Bug 3 — CUDA Namespace Conflict

**Discovery:** Loading both models in the same Python process caused either a RuntimeError on the second extension import or, in environments with permissive dynamic linking, silent replacement of the first extension's functions with the second's.

**Root cause:** Both extensions exported C++ functions with identical names (`knn_query`, `ball_query`, `farthest_point_sampling`) registered into Python's global module namespace via PYBIND11. The second `torch.utils.cpp_extension.load()` call overwrote the first extension's registrations, causing subsequent calls to TGNet's Python wrappers to execute 3DTeethLand's CUDA kernels. Since both extensions implement the same geometric operations with compatible interfaces, this produced no immediate crash but caused subtle numerical differences in FPS and k-NN outputs that would be detectable only through careful ablation.

**Fix:** Renamed TGNet extension from `pointops_cuda` to `tgnet_ops` and 3DTeethLand extension from `pointops` to `teethland_ops`, updating both the PYBIND11_MODULE declarations in C++ and all Python import statements throughout both source trees.

**Verification:** Both extensions loaded in same process; per-extension unit tests with known inputs all pass after dual-load.

### 4.7.4 Bug 4 — Windows runtime_library_dirs Compilation Error

**Discovery:** `python setup.py build_ext --inplace` in `extensions/teethland_ops/` on Windows produced: `AttributeError: 'MSVCCompiler' object has no attribute 'runtime_library_dirs'`.

**Root cause:** The `CUDAExtension()` call included `runtime_library_dirs=['/usr/local/lib']`, a Linux GCC `-rpath` directive not supported by Windows MSVC. The Python `distutils.MSVCCompiler` class does not implement this attribute.

**Fix:** Removed `runtime_library_dirs=['/usr/local/lib']` from `extensions/teethland_ops/setup.py`. No functional impact on Windows: CUDA runtime DLL is found via PATH as set by the CUDA toolkit installer.

**Verification:** Build completed successfully; `import teethland_ops` and basic function calls confirmed correct compilation.

