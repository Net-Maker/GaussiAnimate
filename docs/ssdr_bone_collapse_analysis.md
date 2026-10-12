# SSDR Bone Collapse Analysis

**Date:** 2026-08-22  
**Case:** RigGS Horse, reconstruction to SSDR  
**Target:** explain why a request for 30 or 50 bones can terminate with only 2-18 bones

## Executive conclusion

The bone loss is an **initialization-default regression**, not a limitation of
the final SSDR alternating optimizer. An exact replay on the historical Horse
trajectory isolates two changed defaults:

1. `clusterThreshold`: fixed `3` became density-dependent `floor(nV / 500)`;
2. `labelAssignmentMode`: global minimum-error assignment became 8-NN graph
   propagation.

Restoring those two values reproduces the saved 2025 Horse weights and
transforms element-for-element when KNN weight smoothing is disabled. KNN
weight smoothing is not the regression: enabling it preserves the bone count
and lowers RMSE.

Single-point split seeds and immediate zero-growth termination make the new
defaults more destructive, but both were already present in the historical
code that produced the valid 30-bone result. They are contributing weaknesses,
not the changed lines responsible for this regression.

On the full Horse cloud, `nV = 38,844`, so `floor(nV / 500) = 77`. Every new bone begins with one point but must acquire at least 77 points before the first prune. The direct backend run therefore returned only **2 bones**.

This is not fixed by changing only the threshold. On a controlled Horse subset, threshold `3` still stops at 36 instead of 50 because single-point seeding remains ill-conditioned. A correct repair must address both the seed representation and the density-dependent pruning policy.

## Observed results

| Run | Points | Requested | Returned | Initialization | Geometry RMSE |
|---|---:|---:|---:|---|---:|
| Historical GaussianAnimator Horse | 24,615 | 30 | 30 | saved June 9 result | 0.014989 coordinate / 0.025962 vertex |
| Current backend, exact legacy controls | 24,615 | 30 | 30 | fixed threshold + global labels | 0.014989 coordinate / 0.025962 vertex |
| Fixed defaults with KNN weight smoothing | 24,615 | 30 | 30 | fixed threshold + global labels | 0.009238 coordinate / 0.016000 vertex |
| Current direct DemBones | 38,844 | 50 | 2 | modified LBG split | 0.259632 |
| Opacity-sampled workaround | 24,615 | 50 | 15 | modified LBG split | 0.021255 |
| Motion-initialized repair | 38,844 | 50 | 50 | motion + spatial clustering | 0.006351 |

The 50-bone result proves that the global weight/transform optimization can retain and optimize 50 bones when it receives valid initial labels. The collapse happens before that stage.

### Exact replay and RMSE normalization

The original 200-frame, 24,615-point trajectory was regenerated from
`tempexp/exp-lbs-horse` at Gaussian and deformation iteration 40,000. The saved
rest pose matches frame zero exactly.

Two RMSE conventions appeared in earlier logs:

```text
coordinate RMSE = sqrt(mean(error_xyz^2))
vertex RMSE     = sqrt(mean(sum_xyz(error_xyz^2)))
```

For three coordinates, `vertex RMSE = sqrt(3) * coordinate RMSE`. All tables in
this report now state which convention is used.

With 30 bones and 10 global iterations, the current backend under exact legacy
controls returns:

```text
weights_max_abs_difference    = 0
transforms_max_abs_difference = 0
coordinate RMSE               = 0.014989096589287815
vertex/backend RMSE           = 0.025961876852203875
```

This byte-exact replay rules out `computeWeights()`, transform estimation, and
the alternating SSDR solver as the source of the historical discrepancy.

## Where the number changes

The modified initialization is:

```text
start with one cluster
while bone count is growing:
    split clusters by inserting one new seed point
    repeat nInitIters times:
        estimate one rigid transform per label
        reassign labels through the KNN graph
        prune every cluster smaller than floor(nV / 500)
    stop if final count did not exceed the count before this split round
convert labels to one-hot weights
run global SSDR optimization
```

The relevant implementation is in `native/pyssdr/include/DemBones/DemBones.h`:

- Initialization and termination: lines 212-238
- KNN label propagation: lines 673-735
- Rigid transform estimation: lines 783-793
- Split seed insertion: lines 810-876
- Pruning: lines 881-905

No bone-removal operation runs in the later alternating weight/transform solver. `pruneBones` is part of LBG initialization.

## Root cause 1: single-point split seeds are rank deficient

The original EA implementation assigns the seed's mesh neighborhood to the new label:

```cpp
for (SparseMatrix::InnerIterator it(laplacian, seed); it; ++it)
    label(it.row()) = newLabel;
```

The June 9 modification commented this out and retained only:

```cpp
label(seed) = newLabel;
```

`computeTransFromLabel()` then estimates a rigid transform from the points assigned to each label. For a new label containing one correspondence `(p, q)`, its centered cross-covariance is exactly zero:

```text
q p^T - q p^T = 0
```

Consequences:

- translation can map the seed point;
- rotation is unobservable;
- Kabsch/SVD returns an arbitrary or identity-like rotation;
- the new transform fits its seed but usually does not explain a rotating limb region;
- the established parent bone wins most points during the first reassignment.

At least three non-collinear correspondences are required to constrain a 3D rigid transform. In practice the seed should be a small, motion-coherent neighborhood rather than the minimum mathematical case.

**Confidence: confirmed by source and linear algebra.**

## Root cause 2: the threshold became density dependent

The threshold history is:

| Date / commit | Split threshold | Prune threshold | Label update |
|---|---:|---:|---|
| EA reference | 3 | 3 | graph propagation; neighborhood seed |
| 2025-06-09 `1f7b0f2` | 3 | 3 | global best-error assignment; single seed |
| 2025-06-11 `9f23deb` | 3 | `nV/100` | global best-error assignment; single seed |
| 2025-06-15 `91bd635` | `nV/500` | `nV/500` | global best-error assignment; single seed |
| 2025-06-18 `42fbcdf` | `nV/500` | `nV/500` | KNN graph propagation; single seed |

For the datasets involved:

| Cloud | Points | `nV/500` | Fraction required per bone |
|---|---:|---:|---:|
| Historical Horse | 24,615 | 49 | 0.199% |
| Current Horse | 38,844 | 77 | 0.198% |

The historical 30-bone result contains a valid bone with only **23 points**. It would be deleted by either threshold 49 or 77. This is direct evidence that the old result was produced by the constant-3 regime, not the later density-scaled regime.

Gaussian densification makes point count a poor proxy for physical part importance. A thin but visually important part can contain fewer than 0.2% of the Gaussians, while an oversampled torso can dominate the budget.

**Confidence: confirmed by Git history and saved historical weights.**

## Root cause 3: KNN propagation removed the compensation for a single seed

The June 9 implementation used `computeLabel_simple()`. It tests every point against every bone globally. Although expensive and spatially unconstrained, it allowed a single seed to recruit distant points with a compatible motion immediately.

The June 18 version restored graph propagation and used a rest-space 8-NN graph for point clouds. This is more spatially coherent, but it is incompatible with the one-point seed under aggressive pruning:

- growth must pass through local graph competition;
- the seed transform has no valid rotation;
- pruning occurs immediately after every reassignment;
- the new label often has no time to form a stable rigid region.

The Horse KNN graph also does not perfectly represent semantic connectivity:

- 24.47% of directed 8-NN edges cross the 50-cluster motion initialization boundaries;
- 2.52% of edges connect points whose temporal displacement differs by more than 0.1 scene units;
- only 76.81% of directed neighbor relations are mutual before symmetrization.

These graph statistics are an amplifier, not the primary cause. A correctly initialized multi-point bone can survive the same geometry.

**Confidence: source-confirmed mechanism; measured amplification.**

## Root cause 4: one zero-growth round terminates initialization

The loop continues only when:

```cpp
cont = (nB < targetNB) && (nB > prev);
```

Here `prev` is recorded before split, but `nB` is evaluated after all pruning. If a round goes `5 -> 10 -> 5`, the loop stops even though a different seed, a delayed prune, or another refinement round could succeed.

This turns a local split failure into a permanent global bone-count failure.

**Confidence: confirmed by source and trace.**

## Root cause 5: split priority is biased by sample density

The split eligibility uses cumulative cluster error:

```text
ce_j = sum of per-point errors in cluster j
split if ce_j > average(ce) / 100
```

Because `ce_j` is a sum rather than a mean or an estimated reduction in objective, dense clusters are more likely to split. In a densified Gaussian cloud this allocates bones partly according to sampling density instead of motion complexity.

This does not directly delete an existing bone, but it selects poor parents for the limited set of split attempts and can leave small articulated regions underrepresented.

**Confidence: confirmed bias in the formulation; exact effect requires a dedicated priority ablation.**

## Controlled Horse split trace

To isolate threshold effects, the same deterministic subset was used for every run:

- 5,000 Horse points
- 40 uniformly sampled frames
- target 50 bones
- 8-NN propagation
- 10 initialization refinements
- zero global SSDR iterations
- only `clusterThreshold` changed

| Threshold | Final bones | Result |
|---:|---:|---|
| 3 | 36 | threshold alone is not sufficient |
| 10 (`5000/500`) | 8 | default density-scaled policy collapses |
| 77 (full-Horse threshold) | 5 | full-cloud threshold is extremely destructive on the same motion |

The threshold-77 trace is especially clear:

| Round | Before split | After seed insertion | After first prune | New-label sizes after first reassignment |
|---:|---:|---:|---:|---|
| 0 | 1 | 2 | 2 | 1072 |
| 1 | 2 | 4 | 3 | 492, 1 |
| 2 | 3 | 6 | 5 | 235, 920, 2 |
| 3 | 5 | 10 | 5 | 38, 1, 1, 1, 57 |

In the final round all five newborn bones are below 77, all five are pruned, net growth is zero, and initialization terminates at five bones.

Even with threshold 3, late rounds create many new labels whose first-reassignment sizes are only one or two points. This is why reverting only `nV/500` still stops at 36.

### One-control RMSE ablation

The final isolation used the same 5,000-point subset for every case, retained
all 200 frames, requested 50 bones, and ran 10 global iterations. Each row
changes only the named control from the historical initialization protocol.

| Case | Final bones | Coordinate RMSE | Vertex RMSE |
|---|---:|---:|---:|
| Historical initialization | 49 | 0.007528 | 0.013039 |
| Only KNN weight smoothing enabled | 49 | 0.005246 | 0.009086 |
| Only graph label propagation enabled | 40 | 0.019338 | 0.033494 |
| Only threshold changed from 3 to 10 | 30 | 0.010647 | 0.018442 |
| Both current defaults enabled | 22 | 0.015655 | 0.027116 |

The threshold change causes the larger direct bone-count loss. Graph label
propagation causes the larger isolated RMSE increase. Their combination is
nonlinear and produces the current collapse. KNN weight smoothing improves the
fit and is intentionally retained.

Machine-readable traces and the aggregate summary are stored under:

```text
outputs/ssdr_split_analysis/
├── horse_subsample.npz
├── horse_motion_init_labels_bones50.npy
├── trace_threshold_3/
├── trace_threshold_10/
├── trace_threshold_77/
└── summary.json
```

## Why motion initialization retains 50 bones

The current motion initializer clusters trajectory features before calling DemBones:

1. sample 40 frames;
2. form per-point displacement trajectories;
3. project them to 16 motion components;
4. add a low-weight rest-position feature;
5. run deterministic k-means++ for exactly 50 labels;
6. estimate transforms from full clusters rather than one-point seeds.

Before SSDR optimization, the 50 Horse clusters contain between **109 and 3,706 points**; none is below 77. After optimization, all 50 weight columns remain, although four bones have fewer than 77 argmax-assigned points. This is expected: SSDR uses blended weights, and a useful bone need not be the maximum influence for many points.

This also shows why pruning solely by hard-label population is not compatible with a later soft-weight SSDR objective.

## Historical result interpretation

The historical file:

```text
/home/ubuntu/research/GA/GaussianAnimator/tempexp/exp-lbs-horse/ssdr_result.npy
```

contains:

- 24,615 points;
- 30 bones;
- hard one-hot initialization weights;
- cluster populations from 23 to 4,807 points.

It was added with the GaussianAnimator experiment in commit `07229e3` on 2025-06-09. The corresponding experiment requested 30 bones and 10 global iterations. This aligns with the June 9 DemBones variant: constant threshold 3, single seed, and global best-error reassignment.

It does **not** align with the June 15/18 `nV/500 + KNN` variant. Therefore the newer code is not an exact implementation of the historical experiment protocol.

## Source and binary drift

There are currently multiple non-equivalent backends:

| Backend | State |
|---|---|
| `native/pyssdr/include/DemBones/DemBones.h` | vendored source; defaults repaired to threshold 3 and global assignment |
| `/home/ubuntu/research/code/DemBones-python` | dirty working tree with trace support and configurable threshold |
| DemBones-python `build/pyssdr...so` | trace-capable build, SHA-256 starts `2aa7f338...` |
| py38 `site-packages/pyssdr...so` | older build, SHA-256 starts `a0ae53d3...` |

Importing `pyssdr` normally in the py38 environment selects the older site-package binary, while explicitly setting `PYTHONPATH` selects the trace build. A run cannot be considered reproducible until the selected binary hash and source revision are recorded with the experiment.

## Implemented repair and verification

The repair is now implemented in the public package and the vendored native
backend. It includes:

- explicit `motion_pointcloud` and `paper_legacy` protocols;
- strict requested/final bone-count validation;
- native controls for threshold, label assignment, split neighborhood, prune
  delay, zero-growth patience, and point-cloud weight smoothing;
- the missing native label initialization API used by motion clustering;
- requested, initialized, final, effective, hard-population, soft-mass, backend
  path, and backend SHA-256 fields in saved artifacts;
- the missing `require_num_bones` argument in the monocular evaluator.

The minimal native default repair is restricted to two constructor values:

```cpp
clusterThreshold(3)
labelAssignmentMode(1)
```

`pointCloudSmoothMode(1)` remains enabled because the controlled RMSE ablation
shows that it improves the fit without reducing the bone count. The explicit
`paper_legacy` protocol sets it to zero when exact historical replay is needed.

The complete 80k Horse trajectory was refitted after rebuilding the vendored
backend:

| Field | Verified value |
|---|---:|
| Frames | 200 |
| Points per frame | 38,844 |
| Requested bones | 50 |
| Initialized bones | 50 |
| Final bones | 50 |
| Effective bones | 50 |
| Minimum hard population | 3 |
| Minimum soft mass | 4.3653 |
| Geometry RMSE | 0.0063510270 |

The artifact is
`outputs/horse_full80k_ssdr/protocol_repair_20260822/ssdr_results_bones50.npz`.
Its backend SHA-256 is
`6447a2b39e9aa6525cd0459344835ad6cd530f709ad95fcd92e4842b502c8176`.

The repaired artifact was then rendered through all 200 held-out Horse views.
`outputs/horse_full80k_ssdr/protocol_repair_20260822/metrics.json` records:

| Metric | Repaired motion protocol |
|---|---:|
| PSNR | 34.46043 |
| SSIM | 0.977767 |
| LPIPS | 0.046247 |

The corresponding full80k reconstruction PNG evaluation is 35.95706 PSNR, so
the measured SSDR drop is 1.49663 dB. This verifies the bone-count repair but
does **not** certify the paper's `35.87` PSNR. A separate historical preview
reports 35.867; it comes from a different SSDR artifact/protocol path and must
not be presented as the same reproduction.

## Recommended repair architecture

Do not keep one ambiguous initialization mode. Preserve two explicit protocols.

### 1. `paper_legacy`

Purpose: reproduce the historical table as closely as possible.

- exact historical point selection/checkpoint;
- 30 requested bones for the old Horse experiment;
- constant split and prune threshold 3;
- June 9 global best-error reassignment;
- 10 global iterations;
- record the exact backend source and binary hash.

This mode should be frozen after matching the historical artifact.

### 2. `motion_pointcloud`

Purpose: robust public SSDR for arbitrary Gaussian trajectories.

- initialize all requested bones from motion-coherent multi-point clusters;
- require nonempty clusters and a minimum rank/conditioning check before transform estimation;
- do not prune during the first transform/reassignment step;
- evaluate soft bone influence after optimization rather than hard-label count alone;
- make any reduction in bone count explicit in metrics and artifacts;
- use KNN only as a spatial regularizer, not as the sole mechanism that lets a new bone exist.

## Required validation gates

Every SSDR experiment should record:

- requested, initialized and final bone counts;
- cluster sizes at seed, first reassignment and each prune;
- transform covariance rank or condition number for each new bone;
- hard argmax population and soft weight mass per final bone;
- geometry RMSE before and after global optimization;
- render PSNR/SSIM/LPIPS when Gaussian attributes are available;
- trajectory checksum, config, Git revision and pyssdr binary hash.

The release pipeline should fail when a strict experiment requests 50 bones but returns fewer, rather than silently naming the output after the reduced count.

## Final diagnosis

The sharp decrease is caused by two changed initialization defaults interacting
with pre-existing fragile split behavior:

```text
single-point split
+ density-scaled immediate prune (regression)
+ rest-space KNN label growth (regression)
+ one-round zero-growth termination
= systematic bone collapse
```

The old 30-bone Horse artifact and the repaired 50-bone motion initialization show that the data and the SSDR optimizer can support many bones. The regression is specifically in how new bone clusters are born, allowed to grow, and removed during initialization.
