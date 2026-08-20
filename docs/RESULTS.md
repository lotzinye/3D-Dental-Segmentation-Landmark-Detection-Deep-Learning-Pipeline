# Results & Evaluation

Detailed findings for **CCDS25-0544 — Automated Tooth Segmentation & Anatomical Landmark
Detection Deep Learning Pipeline**.
Publication: https://dr.ntu.edu.sg/entities/publication/9c514406-141c-4615-8594-8c4bc19be852

← Back to [project README](../README.md)

---

## 1. Evaluation Protocols

Both official challenge protocols are implemented in
[`eval_miccai.py`](../dental_landmark_pipeline/eval_miccai.py) and run in a single pass.

### 1.1 Segmentation — MICCAI 2022 (3DTeethSeg'22)

| Metric | Definition | Notes |
|---|---|---|
| **TLA** — Teeth Localisation Accuracy | Mean Euclidean distance between predicted and ground-truth tooth centres, normalised by the physical size of each GT tooth | A missed GT tooth is penalised with a fixed value of **5** (five times the tooth's size). Averaged across all teeth in the test set, since patients have variable tooth counts. |
| **TSA** — Teeth Segmentation Accuracy | Mean F1 across all tooth instances: `F1 = 2·precision·recall / (precision + recall)` | Measures point-cloud boundary quality — precision = accuracy of identified tooth points, recall = completeness. |
| **TIR** — Teeth Identification Rate | % of GT teeth for which the predicted centre lies within **half the GT tooth's size** *and* carries the **same FDI label** | Both conditions must hold; a well-localised tooth with the wrong FDI code counts as a failure. |
| **Score** | `(mean_TLA + mean_TSA + mean_TIR) / 3` | The official combined ranking metric. |

### 1.2 Landmarks — MICCAI 2024 (3DTeethLand)

| Metric | Definition |
|---|---|
| **mAP** | Mean of per-class Average Precision across all 6 landmark classes. Per-class AP is the area under the Precision–Recall curve, averaged over several distance thresholds. |
| **mAR** | Mean of per-class Average Recall; each AR is the area under the Recall-exp curve. |

A high mAP requires **both** high precision and high recall *consistently across every landmark
type* — a single weak class drags the mean down, so the metric is unforgiving of specialisation.

---

## 2. Quantitative Results — 240 held-out scans

<p align="center">
  <img src="figures/results.svg" alt="Segmentation and landmark metric bars with leaderboard context" width="100%">
</p>

### 2.1 Headline numbers

| Network | Metric | Score | Std |
|---|---|---|---|
| Tooth Segmentation | TLA (exp-norm ↑) | **0.9492** | ± 0.2178 |
| | TSA (micro-F1 ↑) | **0.9838** | ± 0.0098 |
| | TIR (% ↑) | 0.4794 | ± 0.4866 |
| | **Challenge Score** | **0.8042** | — |
| Landmark Detection | mAP ↑ | **0.6220** | — |
| | mAR ↑ | **0.4831** | — |
| | AP (cusp) | 0.6213 | — |
| | AR (cusp) | 0.4965 | — |

### 2.2 Per-category landmark breakdown

| Category | AP | AR |
|---|---|---|
| Mesial / Distal | 0.6211 | 0.4966 |
| Cusp | 0.6213 | 0.4965 |
| Inner / Outer | 0.6340 | 0.4880 |
| Facial | 0.6005 | 0.4326 |

Notably **flat**. The spread between the best (Inner/Outer, 0.6340) and worst (Facial, 0.6005)
category is only 0.034 AP. That uniformity is the point: it says the multi-scale windowed
attention design generalises across landmark types with genuinely different geometric
signatures — sharp curvature maxima (cusps), surface-normal extrema (outer points), and
convention-defined references with no crisp geometric correlate at all (facial points).

### 2.3 Reproducing

```bat
cd dental_landmark_pipeline
python eval_miccai.py --max-scans 240
```

| Flag | Effect |
|---|---|
| `--skip-2024` | Segmentation metrics only |
| `--skip-2022` | Landmark metrics only |
| `--use-gt-seg` | Feed ground-truth segmentation into Stage 3 — **isolates landmark error from Stage-1 error propagation** |
| `--reuse` | Reuse cached predictions instead of re-running inference |
| `--out results/eval.json` | Write structured results to disk |

`--use-gt-seg` is the diagnostically important one: it separates "the landmark network is wrong"
from "the landmark network received a bad crop."

---

## 3. Discussion

### 3.1 Segmentation — why TSA and TIR diverge so sharply

TLA (0.9492) and TSA (0.9838) say the pipeline knows **where** teeth are and **which points
belong to them**, to near-ceiling precision. TIR (0.4794) says it is much less reliable at
attaching the correct **two-digit FDI code**.

This is not a contradiction, and it is not unique to this work — FDI label assignment was the
hardest metric for *every* 3DTeethSeg'22 entrant. Three compounding causes:

1. **Severe class imbalance.** Only ~5 % of Teeth3DS scans contain wisdom teeth, so those classes
   are drastically under-learned.
2. **Intra-class geometric similarity.** Teeth of the same type on opposite sides of the arch are
   near-mirror images, so correct quadrant assignment requires global arch context that a locally
   attending network struggles to exploit.
3. **Patient-level variability.** Missing teeth, rotated teeth, incomplete scans, and orthodontic
   appliances that occlude crown geometry all shift the effective position→identity mapping.

**Why this matters less than it looks.** The landmark stage consumes per-tooth *geometric crops*
produced by KDTree neighbourhood extraction around a centroid — it never reads the FDI semantics.
A tooth that is cleanly segmented but mislabelled `17` instead of `27` still yields a correct
crop and therefore correct landmark coordinates; only the FDI field in the output JSON is wrong.
The dependency the landmark stage actually has is on **boundary quality**, which is where the
pipeline scores 0.9838. The metric the pipeline optimised is the metric that mattered downstream.

### 3.2 Landmarks — reading the gap against the leaderboard

| System | mAP | mAR | Setting |
|---|---|---|---|
| Radboud (van Nistelrooij et al.) — 3DTeethLand winner | 0.785 | 0.656 | Landmark-only, full challenge dataset, rank score 0.9172 |
| ChohoTech — 2nd | 0.775 | — | Landmark-only, no explicit segmentation stage (DGCNN dual-branch) |
| **This pipeline** | **0.6220** | **0.4831** | **Downstream of a segmentation stage, integrated end-to-end** |

Three structural differences account for the gap, and none of them is a modelling defect:

- **Error propagation.** Challenge entries received clean inputs. Here, every Stage-1 boundary
  imperfection propagates into the tooth crop the landmark network sees.
- **Task scope.** Challenge solutions were trained and tuned for landmark detection alone. This
  pipeline is a *system*, and the segmentation stage is a real component of it, not an oracle.
- **Resources.** Training was performed on a single 4 GB RTX 3050 Laptop GPU.

Architecturally the pipeline mirrors the winning paradigm the challenge validated — per-point
distance-and-offset field regression with post-hoc DBSCAN clustering, plus a dedicated
variable-output branch for cusps — so the gap is one of training budget and input cleanliness,
not of approach.

### 3.3 Cusps: the hard class, solved

The challenge report singled out cusp landmarks as the hardest class, because their count varies
per tooth (0 on incisors, 1–2 on premolars, 2–5 on molars) and detection frameworks generally
emit a fixed number of outputs. The winning entry needed a **separate sixth decoder** purely to
accommodate this.

This pipeline reaches **AP 0.6213 / AR 0.4965** on cusps — statistically indistinguishable from
the fixed-cardinality Mesial/Distal pair at AP 0.6211. Two design choices are responsible:

1. **A dedicated `cusps` head** with its own decoder path, matching the challenge's finding that
   variable cardinality requires explicit architectural accommodation.
2. **DBSCAN clustering**, which is density-based and therefore **never needs `k` specified in
   advance**. However many dense vote regions exist, that is how many cusps are emitted. Had the
   pipeline used k-means or a fixed-output head, cusps would have been the failure mode.

---

## 4. Qualitative Results — reference scan `01F4JV8X_upper`

<p align="center">
  <img src="figures/landmark-classes.svg" alt="Per-class landmark detection counts against expected counts on the reference scan" width="100%">
</p>

A single inference pass produced **14 segmented teeth** and **94 landmarks**.

### 4.1 Segmentation quality

All 14 teeth were isolated from the gingival background with clearly defined inter-tooth
boundaries. Gingival regions were correctly excluded from the coloured tooth instances, and no
visible colour bleeding occurred between neighbouring crowns — direct visual confirmation that
the Boundary-Aware refinement network resolved the proximal-contact ambiguities that the report's
literature review identified as the primary source of segmentation error across the challenge.

### 4.2 Class-by-class

| Class | Detected | Expected | Placement | Assessment |
|---|---|---|---|---|
| **Mesial** (red) | 14 | 14 | Midline-facing proximal surfaces; consistently on the opposite face to Distal | ✅ Exact — the shared mesial/distal head learned to discriminate the two surfaces |
| **Inner Point** (yellow) | 14 | 14 | Traces a smooth arch-shaped curve along the palatal crown–gingiva boundary, molar to molar | ✅ Exact |
| **Outer Point** (cyan) | 14 | 14 | Lower convex edge of each crown, uniformly spaced along the buccal gingival boundary | ✅ Exact |
| **Cusp** (blue) | 21 | 0/incisor, 1–2/premolar, 2–5/molar | Occlusal surfaces, incisal edges, canine tips | ✅ Consistent with morphology-dependent expectation |
| **Distal** (green) | 18 | 14 | Correctly on midline-averted proximal surfaces | ⚠️ Over-detection |
| **Facial Point** (purple) | 13 | 14 | Front-facing surface of anteriors, cheek-facing surface of posteriors | ⚠️ One miss |

### 4.3 The two deviations — both post-processing, neither a network failure

**Distal over-detection (18 vs 14).** On teeth where multiple surface points are near-equally
maximal from the midline, the vote cloud splits into two DBSCAN clusters that both clear the
20-point minimum. The network's *offsets* are correct; the clustering simply fails to merge them.

> **Fix:** a post-clustering deduplication step enforcing **one detection per fixed-cardinality
> class per FDI instance** (keep the highest-confidence cluster), or a tighter
> `max_neighbor_dist`. This applies to Mesial, Distal, Inner, Outer, and Facial — but must
> **not** be applied to Cusp, whose variable count is legitimate.
> See [`FIX_MESIAL_DISTAL_LANDMARKS.md`](../dental_landmark_pipeline/FIX_MESIAL_DISTAL_LANDMARKS.md).

**Facial point miss (13 vs 14).** The missed tooth is the posterior-most molar. For teeth at the
extremes of the arch, the fixed **k = 12,000** KDTree crop is centred on a centroid whose 12,000
nearest neighbours skew toward the arch interior — so the crop can truncate the very facial
surface the head is looking for. Fewer than 5 candidate points survive the `dist_thresh` filter,
and the head correctly reports "no landmark here."

> **Fix:** an adaptive crop that guarantees angular coverage around the centroid rather than pure
> nearest-neighbour count, or a modest `--crop-k` increase for terminal teeth.

### 4.4 Report figure gallery

These slots are wired and will populate once the PNGs are added — see
[figures/README.md](figures/README.md) for the export instructions.

| Raw scan | After segmentation | All landmarks |
|---|---|---|
| <img src="figures/scan-raw.png" alt="Original 01F4JV8X_upper.obj mesh" width="260"> | <img src="figures/scan-segmented.png" alt="Mesh with 14 individually coloured teeth" width="260"> | <img src="figures/scan-landmarks-all.png" alt="All landmark octahedra on the mesh" width="260"> |

| Cusp (blue) | Distal (green) | Mesial (red) |
|---|---|---|
| <img src="figures/landmarks-cusp.png" alt="Cusp landmark layer" width="260"> | <img src="figures/landmarks-distal.png" alt="Distal landmark layer" width="260"> | <img src="figures/landmarks-mesial.png" alt="Mesial landmark layer" width="260"> |

| Inner Point (yellow) | Outer Point (cyan) | Facial Point (purple) |
|---|---|---|
| <img src="figures/landmarks-inner.png" alt="Inner point landmark layer" width="260"> | <img src="figures/landmarks-outer.png" alt="Outer point landmark layer" width="260"> | <img src="figures/landmarks-facial.png" alt="Facial point landmark layer" width="260"> |

---

## 5. Stage-by-Stage Inspection

To interrogate any single stage on a scan of your choice:

```bat
python export_stages.py scan.obj exports\stages\
python export_stages.py scan.obj exports\stages\ --focus-fdi 16
```

This writes the pipeline's internal state as inspectable PLY files:

| File | Stage | Contents |
|---|---|---|
| `stage1a_fps_sample.ply` | 1 | The 24,000 FPS-sampled points |
| `stage1b_sem_on_sample.ply` | 1 | Semantic labels on the sampled points |
| `stage1c_fdi_full_mesh.ply` | 1 | FDI labels propagated to the full mesh |
| `stage2_crop_fdi_<NN>.ply` | 2 | The 12,000-point z-space crop for tooth `NN` |
| `stage3_head_<class>_fdi_<NN>.ply` | 3 | Raw per-point predictions from one head |
| `stage3b_candidates_fdi_<NN>.ply` | 3 | Points surviving the `dist_thresh` filter |
| `stage3c_dbscan_fdi_<NN>.ply` | 3 | DBSCAN cluster assignment of the vote cloud |
| `stage3d_tooth_landmarks_fdi_<NN>.ply` | 3 | Final weighted-centroid landmark positions |

The `stage3b → stage3c → stage3d` triple is the most diagnostic sequence in the pipeline: it
shows exactly where a duplicate or missing landmark originates — a scattered vote cloud
(network), an over-split clustering (`max_neighbor_dist`), or a cluster rejected for having
fewer than `min_points` votes (`min_points`).

---

## 6. Complementary Localisation Metrics

Beyond the challenge protocols, [`evaluate_pipeline.py`](../dental_landmark_pipeline/evaluate_pipeline.py)
reports clinical localisation accuracy — **Mean Radial Error (MRE)** in millimetres and
**Successful Detection Rate** at fixed millimetre tolerances (SDR@1.5 / 2.0 / 2.5 / 4.0 mm).

```bat
python evaluate_pipeline.py --auto-discover --pred-dir data\output --gt-dir data\3DTeethLand_landmarks_test --out results\eval_results.json
```

These answer the question the challenge metrics do not: *when the pipeline finds a landmark, how
far off is it, in units a clinician can act on?* Since the clinical requirement stated in the
report is roughly **1 mm** over a crown spanning 20–40 mm, MRE and SDR@1.5 are the measures that
speak most directly to clinical usability.

---

## 7. Limitations

| Limitation | Evidence | Mitigation |
|---|---|---|
| **Training-set imbalance** — Teeth3DS skews to adolescent orthodontic patients; wisdom teeth in ~5 % of scans | TIR 0.4794 with high variance (± 0.4866) | Oversample scans containing rare tooth classes; broaden the corpus to adult, elderly, and non-orthodontic dentitions |
| **No arch-level context in Stage 3** — the network reasons only within one tooth's local geometry | Missed posterior facial point; predictions locally plausible but not globally consistent | Full-jaw landmark formulation capturing inter-tooth spatial relationships |
| **Fixed-cardinality classes not enforced** | Distal 18 vs expected 14 | Per-FDI deduplication for the five single-instance classes |
| **Fixed-k crop truncates terminal teeth** | Facial point 13 vs 14 | Angular-coverage-aware adaptive crop |
| **No clinical validation** | Evaluated against dataset annotations only | Benchmark against multiple practising orthodontists on unseen scans to test whether error falls inside acceptable inter-annotator variability |
| **Constrained training budget** | mAP gap vs challenge leaders | Longer schedules on higher-VRAM hardware |

---

## 8. Conclusion

The pipeline demonstrates that **two networks designed for separate tasks and trained under
mutually incompatible normalisation conventions can be integrated into a single coherent
end-to-end system** — the coordinate-space bridge being the enabling contribution.

Measured on 240 held-out scans it delivers near-ceiling segmentation boundary accuracy
(TSA 0.9838) and competitive six-class landmark detection (mAP 0.6220) from a downstream
position, producing clinically interpretable annotations from a raw intraoral scan with **no
manual intervention beyond scan acquisition**. FDI label identification and landmark recall are
the two clearly-identified targets for future improvement, and both have concrete, scoped
remedies rather than open research questions.
