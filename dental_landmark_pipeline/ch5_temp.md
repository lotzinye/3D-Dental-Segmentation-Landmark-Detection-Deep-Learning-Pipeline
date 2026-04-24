
# Chapter 5: Methodology — Method B: Lean PointNet++ Pipeline

## 5.1 Motivation

Method B is motivated by three practical limitations of Method A that constrain its usability in real-world clinical deployment scenarios beyond the benchmark evaluation described in Chapter 7.

**VRAM constraint for training.** TGNet is implemented in TensorFlow and was trained on institutional GPU hardware with substantially more VRAM than the RTX 3050 Laptop's 4 GB. A rough VRAM estimate for retraining TGNet from scratch with a realistic batch size illustrates the constraint: the model processes 24,000 input points per scan, and with batch size 2 and a three-level SA encoder at feature widths [128, 256, 512], the intermediate activation tensors for a single forward pass require approximately 5–8 GB depending on the specific layer configuration. This estimate exceeds the RTX 3050's 4 GB budget even before allocating memory for gradient storage, which roughly doubles the forward-pass memory requirement. LandmarkNet's Stratified Transformer processes 12,000 points per tooth crop. With batch size 2, the intermediate attention matrices and feature tensors at the three encoder levels [48, 96, 192, 256] require approximately 6–10 GB. Training either model from scratch on the development machine is therefore not feasible at their original configurations.

**No custom CUDA compilation requirement.** Both `tgnet_ops` and `teethland_ops` CUDA extensions must be compiled from C++ source against the local CUDA toolkit and target GPU architecture. While this was accomplished on the development machine (Windows 11, CUDA 12.6, RTX 3050), the compilation process requires: a compatible MSVC compiler with the Windows SDK, the CUDA toolkit with header files, and matching PyTorch installation with developer headers. Each of these must be at a compatible version combination, and the combination changes with every PyTorch major release. For a clinician or researcher without a configured development environment, this compilation step is a significant barrier to reproducing or deploying the pipeline. Method B is designed to operate entirely on standard PyTorch tensor operations with no custom CUDA required.

**Fine-tuning to scanner-specific data.** Method A uses TGNet and LandmarkNet with their original published weights, which were trained on the Teeth3DS dataset containing scans from multiple scanner manufacturers. If a deployment context uses a specific scanner brand (e.g., only iTero Element 5D scans), the performance might be improved by fine-tuning the models on a small annotated dataset from that specific scanner. However, fine-tuning either of Method A's models requires the same VRAM budget as training them from scratch (since full-model fine-tuning uses the same memory pattern as training). Method B's lean architecture, with VRAM consumption approximately 40% of the original models, enables fine-tuning within the 4 GB budget.

Against these practical advantages, Method B accepts a modest accuracy trade-off: approximately 5% lower segmentation mIoU (0.88–0.90 vs 0.932) and approximately 0.3–0.5 mm higher MRE (1.2–1.4 mm vs 0.920 mm). As discussed in Chapter 8, this accuracy gap is within the range where clinical utility is preserved for routine cases, and the gap can be narrowed by domain-specific fine-tuning.

## 5.2 Architecture Design Decisions

The lean PointNet++ MSG architecture was designed through a series of choices, each motivated by specific considerations of accuracy, VRAM, and deployment simplicity.

### 5.2.1 Three SA Blocks Instead of Four

Standard PointNet++ segmentation implementations use four set abstraction (SA) layers at point resolutions of 1024, 256, 64, and 16 respectively. The fourth SA layer (16 points) is intended to capture global context but contributes minimal incremental benefit for the relatively small point cloud sizes used in Method B (6,000 points for segmentation, 3,000 for landmarks). More importantly, the fourth SA layer's intermediate features must be kept in memory during backpropagation to compute gradients, contributing approximately 0.3–0.5 GB per batch element to peak VRAM usage.

Method B replaces the two finest SA layers (64 and 16 points) with a single global SA layer (SA3) that groups all remaining points into a single global context vector. This global pooling operation, analogous to the [CLS] token in Vision Transformer architectures, provides a scan-level summary feature that every point in the decoder can attend to via skip connections. The reduction from four SA layers to three reduces peak VRAM by approximately 30% while retaining the multi-scale hierarchy needed for accurate landmark localisation.

### 5.2.2 Multi-Scale Grouping (MSG)

Method B uses Multi-Scale Grouping (MSG) rather than Single-Scale Grouping (SSG) at each SA layer. With MSG, each SA layer applies two ball query radii simultaneously and concatenates the resulting feature vectors. This provides multi-scale representational richness at each resolution level without requiring additional SA layers.

For dental landmark detection, the two radii at each SA layer are calibrated to capture qualitatively different geometric contexts. At SA1 (finest resolution): radius 0.1 normalised units (~1.73 mm) captures sub-cusp detail (curvature at tip vs face of cusp), while radius 0.2 normalised units (~3.46 mm) captures the full cusp width in context. At SA2 (intermediate resolution): radius 0.2 units captures individual teeth in context, while radius 0.4 units captures inter-tooth relationships. This dual-radius strategy allows a single SA1 layer to capture both the fine curvature patterns needed for precise cusp localisation and the broader crown context needed for tooth type discrimination, at a VRAM cost only slightly higher than single-radius SSG.

### 5.2.3 Reduced Input Points

The full mesh crops used in Method A (12,000 points per tooth for LandmarkNet) are reduced to 6,000 points for the Method B segmentation model and 3,000 points for the Method B landmark model. These values were determined by VRAM profiling on the RTX 3050 with PyTorch 2.10 and AMP FP16:

- Segmentation model: 6,000 input points, batch size 8 → peak VRAM ~3.2 GB
- Landmark model: 3,000 input points, batch size 16 → peak VRAM ~2.8 GB

Both values fit within 4 GB with headroom for the gradient scaler and optimiser state. The reduction from 12,000 to 3,000 points for the landmark model (a factor of 4) is partly compensated by the MSG's multi-scale grouping, which extracts richer features per point than SSG at the same resolution.

### 5.2.4 Lean Prediction Head

Method B uses 64-channel fully connected layers in the prediction head rather than the 128-channel layers standard in PointNet++ implementations. This roughly halves the parameter count in the head (64*64 = 4,096 vs 128*128 = 16,384 weights for the first FC layer) while maintaining sufficient capacity for the 17-class segmentation task and the 5-head landmark detection task. The 64-channel head is implemented as two sequential Conv1d layers (Conv1d(128, 64) + BatchNorm + ReLU, then Conv1d(64, num_classes)) applied along the point dimension.

## 5.3 Segmentation Model (PointNet2SegModel)

The segmentation model takes a full jaw point cloud as input and produces per-point FDI label predictions. It follows the encoder-decoder structure of PointNet++ with MSG and three SA layers.

**Table 5.1 — Lean PointNet++ MSG segmentation architecture**

| Layer | Output Points | Output Channels | Radii (MSG) | Notes |
|---|---|---|---|---|
| Input | 6,000 | 6 (xyz + normals) | — | z-score normalised + unit normals |
| SA1 | 1,024 | 64+128=192 | 0.1, 0.2 | MSG concat of two radii |
| SA2 | 256 | 128+256=384 | 0.2, 0.4 | MSG concat of two radii |
| SA3 | 1 (global) | 256+512=768 | — | Global max-pooling over all SA2 points |
| FP3 | 256 | 512 | — | Upsample SA3 features to SA2 resolution |
| FP2 | 1,024 | 256 | — | Upsample FP3 features to SA1 resolution |
| FP1 | 6,000 | 128 | — | Upsample FP2 features to input resolution |
| Head | 6,000 | 64 -> 17 | — | Conv1d(128,64)+BN+ReLU + Conv1d(64,17) |

A batch of data flows through the model as follows, tracing tensor shapes for batch size B = 2:

- **Input:** (2, 6000, 6) — batch of point clouds with xyz and normal features.
- **SA1:** FPS selects 1024 centroids. Two ball queries (radii 0.1, 0.2) produce feature tensors of shape (2, 1024, 64) and (2, 1024, 128) respectively. Concatenated to (2, 1024, 192).
- **SA2:** FPS selects 256 centroids from SA1 output. Two ball queries (radii 0.2, 0.4) produce (2, 256, 128) and (2, 256, 256). Concatenated to (2, 256, 384).
- **SA3:** Global abstraction groups all 256 SA2 points into one global feature. Two ball queries with effectively infinite radii produce (2, 1, 256) and (2, 1, 512). Concatenated to (2, 1, 768). This single global vector encodes the full-scan context.
- **FP3:** Interpolates SA3 features to SA2 resolution (2, 256, 768). Concatenated with SA2 skip-connection features (2, 256, 384) to give (2, 256, 1152). Passed through shared MLP to produce (2, 256, 512).
- **FP2:** Interpolates FP3 output to SA1 resolution (2, 1024, 512). Concatenated with SA1 skip features (2, 1024, 192) to give (2, 1024, 704). Passed through shared MLP to produce (2, 1024, 256).
- **FP1:** Interpolates FP2 output to input resolution (2, 6000, 256). Concatenated with input features (2, 6000, 6) to give (2, 6000, 262). Passed through shared MLP to produce (2, 6000, 128).
- **Head:** Conv1d(128, 64) + BatchNorm + ReLU, then Conv1d(64, 17) → (2, 17, 6000). Transposed to (2, 6000, 17) for per-point class logits.

The Feature Propagation (FP) layers use inverse-distance-weighted trilinear interpolation to upsample from a coarser set of centroids to a finer set. For each point at the target resolution, its feature is computed as the distance-weighted average of its three nearest neighbours in the source (coarser) feature set. This interpolation is correct and exact for points that were centroids at the source resolution (they receive their own features with weight 1.0), and produces smooth feature gradients for other points. The interpolated features are then concatenated with the skip-connection features from the encoder layer at the same resolution, providing the decoder with both high-level semantic context (from the deeper layers) and fine-grained local geometry (from the encoder skip).

The predicted FDI label for each point is the argmax of the per-point class logit vector. During evaluation, this produces the final per-vertex segmentation for the full jaw scan. During training, softmax probabilities are used to compute the cross-entropy + Dice loss described in Section 5.6.1.

## 5.4 Landmark Model (PointNet2LandmarkModel)

The landmark model takes a per-tooth point cloud as input and produces per-point predictions for five landmark heads, matching the output format of Method A's LandmarkNet to enable direct post-processing reuse and performance comparison.

**Table 5.2 — Lean PointNet++ MSG landmark architecture**

| Layer | Output Points | Output Channels | Radii (MSG) | Notes |
|---|---|---|---|---|
| Input | 3,000 | 9 (xyz norm + normals + centroid offset) | — | Same 9-channel format as LandmarkNet |
| SA1 | 512 | 64+128=192 | 0.05, 0.10 | Fine radii for sub-cusp detail |
| SA2 | 128 | 128+256=384 | 0.10, 0.20 | Medium radii for cusp/contact context |
| SA3 | 1 (global) | 256+512=768 | — | Global abstraction of full tooth |
| FP3 | 128 | 512 | — | Upsample to SA2 resolution |
| FP2 | 512 | 256 | — | Upsample to SA1 resolution |
| FP1 | 3,000 | 128 | — | Upsample to input resolution |
| Heads (x5) | 3,000 | 64 -> 4 each | — | Conv1d(128,64)+BN+ReLU + Conv1d(64,4) |

The input uses the same 9-channel feature format as LandmarkNet (channels 0–2: z-score normalised xyz; channels 3–5: unit normals; channels 6–8: centroid offsets in z-score space), enabling the same coordinate bridge developed for Method A to be reused directly with Method B's landmark model. The output format per head — (K, 4) with channel 0 as predicted distance and channels 1–3 as predicted offset — is also identical, allowing the same DBSCAN post-processing to be applied without modification.

The five independent heads are implemented as separate Conv1d modules rather than a single shared head with multiple output channels. This design is motivated by the different learning dynamics of each landmark class: InnerPoint and OuterPoint have geometrically consistent targets (extremal surface curvature) with low variance, while FacialPoint and Distal have ambiguous geometric correlates with high variance. Separate heads allow each to converge at its own rate and with its own loss contribution, whereas a shared head with multiple outputs would force all classes to share the same feature extraction and might converge to a compromise that underperforms for the easier classes to achieve acceptable performance on the harder ones.

## 5.5 Training Optimisations

Four complementary training optimisations are applied to Method B to maximise training efficiency within the VRAM and wall-time constraints of the development machine.

### 5.5.1 Automatic Mixed Precision (AMP) FP16

Automatic Mixed Precision (AMP) uses 16-bit floating-point (FP16) arithmetic for the forward pass and most backward-pass computations, while maintaining 32-bit (FP32) master weights for parameter updates. FP16 has half the memory footprint of FP32 for activation tensors, reducing peak VRAM by approximately 30–40% during training. On Ampere-architecture GPUs (including the RTX 3050), matrix multiplications in FP16 execute on dedicated Tensor Core hardware, providing approximately 1.8x throughput improvement over FP32 for the same memory bandwidth.

The AMP workflow uses two components from `torch.cuda.amp`:

```python
scaler = torch.cuda.amp.GradScaler()

for batch in dataloader:
    optimiser.zero_grad()
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        outputs = model(inputs)
        loss = criterion(outputs, targets)
    scaler.scale(loss).backward()
    scaler.step(optimiser)
    scaler.update()
```

The `autocast` context manager automatically selects FP16 for operations that are numerically safe in half precision (matrix multiplications, convolutions, batch normalisation) and keeps FP32 for numerically sensitive operations (loss computation, softmax). The `GradScaler` addresses the FP16 gradient underflow problem: FP16 can represent values as small as approximately 6e-5, but gradient values during training can be much smaller. GradScaler multiplies the loss by a large scale factor (default 65536) before backpropagation, shifting the gradient magnitudes into the FP16 representable range. It then divides the gradients by the same factor before the optimiser step, recovering the correct gradient magnitudes. If a gradient overflow is detected (resulting in NaN or Inf gradients in FP16), GradScaler skips the optimiser step for that batch and reduces the scale factor, automatically adapting to the training dynamics.

### 5.5.2 torch.compile()

PyTorch 2.0 introduced `torch.compile()`, a one-line optimisation that uses the TorchDynamo graph capture subsystem to trace the model's computation graph and applies the TorchInductor compiler backend to fuse adjacent operations and eliminate Python-level interpreter overhead. For a PointNet++ model, the most impactful fusion opportunities are in the FP decoder MLP blocks: each FP layer consists of several sequential linear layers, batch normalisations, and ReLU activations that TorchInductor can fuse into a single GPU kernel, eliminating the memory bandwidth overhead of writing and re-reading intermediate tensors between kernel launches.

The optimisation is applied as a single line after model construction:

```python
model = torch.compile(model)
```

This adds a one-time compilation cost on the first training batch (typically 30–60 seconds for the Method B models), after which subsequent batches run at the compiled speed. The expected throughput improvement is 15–20% for repeated forward-backward passes on architectures with many small sequential operations, such as the FP decoder MLP blocks. The compiled model produces identical numerical outputs to the uncompiled model; the compilation only affects execution speed, not correctness.

### 5.5.3 Cosine Annealing with Warm Restarts

The learning rate schedule uses CosineAnnealingWarmRestarts from `torch.optim.lr_scheduler`, configured with initial period T_0 = 20 epochs and period-doubling factor T_mult = 2. The learning rate at epoch t within a cycle of length T is:

```
eta(t) = eta_min + (eta_max - eta_min) * (1 + cos(pi * t_curr / T_curr)) / 2
```

where `t_curr` is the position within the current cycle and `T_curr` is the current cycle length. At the start of each cycle, the learning rate is reset to `eta_max` (typically 1e-3) and decreases smoothly to `eta_min` (typically 1e-6) following a cosine curve. At the end of the cycle, the period is doubled: T_curr = T_0 * T_mult^k where k is the cycle index. This produces cycle lengths of 20, 40, 80, 160 epochs in sequence.

The rationale for warm restarts is that a decaying learning rate converges to a local minimum, but the found minimum may be shallow — a minimum that generalises poorly to unseen data. Restarting the learning rate to a high value forces the optimiser to escape this shallow minimum and explore other regions of the loss landscape, potentially converging to a deeper minimum with better generalisation. The increasing cycle length ensures that later cycles have sufficient epochs for genuine convergence rather than oscillating too frequently.

For the segmentation model trained with cosine annealing and early stopping (patience = 15), the typical training trajectory is: rapid initial convergence in cycle 1 (epochs 1–20), modest improvement after restart (epochs 21–60), and gradual improvement in cycle 3 (epochs 61–140) before early stopping triggers when validation mIoU plateaus.

### 5.5.4 Early Stopping

Early stopping monitors the primary validation metric (mIoU for segmentation, MRE for landmark detection) and terminates training when the metric fails to improve by at least delta = 0.001 for patience = 15 consecutive epochs. At termination, the model weights from the best validation epoch are restored from a saved checkpoint.

The patience value of 15 epochs was chosen to be large enough to tolerate the transient metric degradation that occurs at warm restarts: when the learning rate is reset to a high value at the start of each cycle, the model may exhibit slightly worse validation performance for 2–5 epochs as the optimiser navigates the loss landscape from a different starting velocity. A patience of fewer than 10 epochs would prematurely terminate training during warm restarts.

Early stopping is particularly important for the landmark model, where the number of annotated training examples (approximately 1,000–1,200 teeth from 120 matched scan-annotation pairs) is much smaller than the segmentation training set (potentially thousands of scans). Smaller training sets increase the risk of overfitting, where the model memorises training examples rather than learning generalisable features. Early stopping prevents this by monitoring generalisation directly via held-out validation performance.

The checkpoint and resume system saves the full training state at the end of each epoch: model weights, optimiser state, scheduler state, current epoch number, and best validation metric. This enables training to resume from any checkpoint without loss of progress — important for long training runs on the development machine where power interruptions or system restarts may occur.

## 5.6 Loss Functions

### 5.6.1 Segmentation Loss: Cross-Entropy + Soft-Dice

The segmentation model is trained with a combined loss function:

```
L_seg = L_CE + lambda * L_Dice
```

where L_CE is the standard multi-class cross-entropy loss and L_Dice is the Soft-Dice loss, with lambda = 1.0.

**Cross-entropy loss** computes the negative log-likelihood of the true class label under the predicted probability distribution:

```
L_CE = -(1/N) * sum_i log(p_i[y_i])
```

where p_i[y_i] is the predicted probability for the true class y_i at point i, and N is the total number of points in the batch. While cross-entropy is efficient and well-behaved, it assigns loss proportional to the number of points in each class. In the Teeth3DS class distribution, where gingiva (class 0) accounts for 60–70% of points, the cross-entropy gradient is dominated by the gingiva class even when that class is already accurately predicted.

**Soft-Dice loss** is derived from the Dice coefficient (equivalent to the F1 score), computed separately for each class and averaged:

```
L_Dice = 1 - (1/C) * sum_c (2 * sum_i p_ic * y_ic + eps) / (sum_i p_ic + sum_i y_ic + eps)
```

where p_ic is the predicted probability for class c at point i, y_ic is the one-hot ground-truth indicator, C is the number of classes, and eps = 1e-6 prevents division by zero. The Soft-Dice loss is class-balanced by construction: it computes the F1-score overlap ratio independently for each class and averages them, giving equal weight to the gingiva class (with many points) and to each individual tooth class (with few points). A class with zero true positive points and zero false positive predictions receives a Dice contribution of 1.0 (perfect, no penalty), while a class with any false positives or false negatives receives a Dice contribution below 1.0. This makes the Dice loss particularly effective at penalising predictions that completely miss small tooth classes.

A numerical illustration of the imbalance problem: consider a batch with 7,000 gingiva points and 3,000 tooth points from a single tooth class. A model that predicts gingiva for all points achieves cross-entropy of approximately -log(0.9) = 0.10 nats (low, seemingly good), but Dice = 0 for the tooth class (catastrophic). Adding the Dice term to the cross-entropy prevents the optimiser from converging to this gingiva-dominant solution.

### 5.6.2 Landmark Loss: Smooth-L1 Distance + Masked Smooth-L1 Offset

The landmark model is trained with a combined loss:

```
L_lm = L_dist + mu * L_offset
```

where L_dist is the Smooth-L1 (Huber) loss on predicted distances for all points, L_offset is the Smooth-L1 loss on predicted offset vectors for near-landmark points only, and mu = 1.0.

**Smooth-L1 (Huber) distance loss** applies to all K points in a crop for all five landmark heads:

```
L_dist = (1/K) * sum_i SmoothL1(pred_dist[i], gt_dist[i])

where SmoothL1(x, y) = {
    0.5 * (x-y)^2 / delta    if |x-y| < delta
    |x-y| - 0.5 * delta      otherwise
}  with delta = 1.0
```

The clamped ground-truth distance is min(true_euclidean_distance_normalised, 0.20), matching the LandmarkLoss training convention of Method A's LandmarkNet. Points far from any landmark are supervised to predict the maximum value 0.20, while points close to a landmark are supervised to predict their precise distance. Smooth-L1 is preferred over MSE because large prediction errors (common early in training) contribute linearly rather than quadratically to the loss, preventing occasional bad batches from producing extremely large gradient updates that destabilise training.

**Masked Smooth-L1 offset loss** applies only to points within the distance threshold `dist_thresh` of a ground-truth landmark:

```python
near_mask = (gt_distances < dist_thresh).float()   # (K,) binary mask
L_offset  = sum(SmoothL1(pred_offsets * near_mask, gt_offsets * near_mask))
          / (near_mask.sum() + eps)
```

The masking strategy is essential because offset vectors are only well-defined for points near a landmark. For a point far from any landmark, there is no single correct offset direction — the point might be offset from any of several distant landmarks — and supervising offset predictions for these points would inject contradictory gradient signals. The mask restricts offset supervision to the near-landmark region where each point has a unique, well-defined target offset (pointing toward the nearest landmark of the class). The normalisation by `near_mask.sum()` ensures the loss magnitude is independent of the proportion of near-landmark points, which varies substantially between tooth types (molars with large central fossae have many near-InnerPoint points; incisors have few).

## 5.7 Expected vs Published Performance

Method B has not been trained to completion at the time of writing, as doing so requires approximately 200 epochs of the segmentation model (~36 hours at estimated 10 min/epoch on RTX 3050 with AMP) and 200 epochs of the landmark model (~48 hours). Expected performance is estimated from published PointNet++ MSG results on comparable dental analysis tasks and from the known performance hierarchy between PointNet++ and Transformer architectures on the Teeth3DS and 3DTeethLand benchmarks.

**Table 5.3 — Expected performance comparison (Method A vs Method B)**

| Metric | Method A (Measured) | Method B (Expected) | Basis |
|---|---|---|---|
| Segmentation mIoU | 0.932 | 0.88–0.90 | Published PointNet++ dental results [3][16] |
| Overall MRE (mm) | 0.920 | 1.2–1.4 | PointNet++ landmark baselines [9] |
| SDR@2mm | 93.3% | 83–87% | Estimated from MRE via SDR-MRE correlation |
| SDR@4mm | 95.8% | 92–94% | Expected near-convergence at 4mm threshold |
| Detection rate | 100% | ~95% | Lower model capacity may miss some third molars |
| VRAM (training) | N/A (fixed weights) | ~3.2 GB seg, ~2.8 GB lm | VRAM profiling on RTX 3050 |
| Training time (estimated) | N/A | 36h seg + 48h lm | Estimated from per-epoch timing |
| Custom CUDA required | Yes (tgnet_ops, teethland_ops) | No | Architecture design |
| Trainable from scratch on 4GB | No | Yes | VRAM design target |
| Fine-tunable on local data | Not feasible | Feasible | VRAM design target |

The expected 5% mIoU gap reflects the known performance difference between PointNet++ MSG architectures and TGNet's boundary-aware Transformer on Teeth3DS. The expected 0.3–0.5 mm MRE gap reflects the general performance hierarchy observed in the 3DTeethLand challenge between PointNet++ and Transformer-based methods [9]. These gaps are acceptable trade-offs for the deployability and fine-tuning benefits provided by Method B.

The accuracy gap is expected to narrow significantly if Method B is fine-tuned on a domain-specific dataset from the target scanner model. Fine-tuning on 50–100 annotated scans from a specific scanner is estimated to recover 2–4 percentage points of mIoU for segmentation (based on analogous fine-tuning experiments in the medical imaging literature) and 0.1–0.2 mm of MRE for landmark detection. In a deployment context where such annotated data is available, Method B with fine-tuning may therefore match or exceed Method A's performance for that specific scanner model.

The SDR@2mm threshold is the primary clinical benchmark: Method A's 93.3% means approximately 11 landmarks per 165-landmark scan pair require correction. Method B's expected 83–87% means approximately 21–28 landmarks per scan pair would require correction — approximately twice as many, but still substantially fewer than fully manual placement of all 165 landmarks. For research applications where population-level statistics are the goal rather than individual patient accuracy, Method B's accuracy level may be sufficient without any fine-tuning.

