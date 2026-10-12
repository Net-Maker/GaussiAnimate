# Paper Experiments

We have fixed several implementation bugs since the original experiments, so reproduced results may differ slightly from the values reported in the paper.

Every paper table, ablation, and runtime result has one declarative config.
`experiments/catalog.yaml` is the machine-readable index and records whether
each target is verified, partial, in progress, or blocked. Run
`gaussianimate-paper-audit`; `--require-ready` gates full paper reproduction,
not publication of the source code. Table 3 numerical reproduction is deferred
for this source release; its implementation and configs remain available.

## Architecture

The experiment system has three explicit layers:

- `engine/`: config composition, typed jobs, stage registry, fingerprints,
  atomic manifests, resume logic, and coverage-aware reports.
- `methods/`: shared point split, SSDR cache, prediction artifact, metric, and
  diagnostic contracts.
- Domain folders: dataset adapters, method implementations, and paper configs.

Large configs use `method_files`, so every method has one readable YAML
component. The multiview experiment keeps D3DGS, LBS, BoB, and the two PartMM
variants in `multiview_gaussian/methods/` while preserving the existing CLI
and output paths. Running only `--stage fit` records a `partial` job; a job is
`complete` only after every declared stage passes its current fingerprint and
all outputs exist. Averages are written only for complete case coverage.

| Target ID | camera-ready v40 display | Config |
| --- | --- | --- |
| `synthetic_novel_view` | Table 2: synthetic novel views | `monocular_gaussian/novel_view.yaml` |
| `human_novel_pose` | Table 3: real-human novel poses | `multiview_gaussian/novel_pose.yaml` |
| `multiview_d3dgs_reconstruction` | Table 3: shared D3DGS reconstructions | `multiview_gaussian/reconstruction.yaml` |
| `vto_garment_animation` | Table 4: VTO garment mean vertex error | `mesh/vto_animation.yaml` |
| `d4d_animal_animation` | Table 4: D4D animal mean vertex error | `mesh/dt4d_animation.yaml` |
| `partmm_patch_length` | Table 5: PartMM patch length | `mesh/patch_length.yaml` |
| `partmm_database_scaling` | Table R.1: database coverage and cost | `mesh/database_scaling.yaml` |
| `arap_ssdr_ablation` | Fig. 6: ARAP / SSDR ablation | `monocular_gaussian/arap_ssdr_ablation.yaml` |
| `progressive_skeletonization` | Fig. 7: progressive skeletonization | `monocular_gaussian/progressive_skeleton.yaml` |
| `partmm_runtime` | Fig. R.2: runtime | `mesh/runtime.yaml` |

## SSDR Followed by PSNR

The direct path is **reconstructed Gaussian trajectories -> SSDR -> rendered
test views -> PSNR/SSIM/LPIPS**. It does not require RigGS stage 2. The
[evaluator](monocular_gaussian/ssdr_evaluate.py) loads a RigGS-compatible
reconstruction, replaces Gaussian center motion with SSDR/LBS motion, and
writes `metrics.json`, the fitted SSDR archive, and test renders.

Use the reconstruction's CUDA environment (including its `model_loader`,
Gaussian rasterizer, Torch/Torchvision and metric dependencies), install this
repository there, and build `pyssdr`. Then configure the saved checkpoint:

```bash
export GA_RIGGS_ROOT=/path/to/RigGS
export GA_HORSE_RECONSTRUCTION=/path/to/reconstruction
export GA_HORSE_SSDR_OUTPUT=/path/to/new/ssdr_evaluation
python -m experiments.monocular_gaussian.ssdr_evaluate \
  --config experiments/monocular_gaussian/horse_ssdr.yaml --stage all
```

Set `iteration` in the YAML to the checkpoint iteration you actually have.
The example requests 50 bones; it is a runnable diagnostic, not a claim of
matching a printed table entry. To render an existing fit without fitting again:

```bash
python -m experiments.monocular_gaussian.ssdr_evaluate \
  --config experiments/monocular_gaussian/horse_ssdr.yaml \
  --stage render --ssdr-result /path/to/ssdr_results_bones50.npz
```

For the camera-ready Table 3 human novel-pose protocol, the corresponding
pieces are [training-only SSDR](multiview_gaussian/ssdr_cache.py),
[held-out rendering and metrics](multiview_gaussian/partmm_eval.py), and the
[method configs](multiview_gaussian/methods/smpl_partmm.yaml). After setting
the resource paths in those configs, run:

```bash
gaussianimate-paper experiments/multiview_gaussian/novel_pose.yaml \
  --dataset dna_rendering --case dna_0012_11 --method smpl_partmm --gpus 0
```

That held-out protocol includes motion prediction; direct all-frame SSDR
fitting is not a substitute for novel-pose evaluation. Numerical reconciliation
is left for a later revision.

For the sigmoid-RGB Table 3 rerun, report the two PSNR values together:

```bash
python -m experiments.multiview_gaussian.table3_psnr \
  --root "$GA_MULTIVIEW_RELEASE_ROOT"
```

`TrainSet PSNR` renders the saved SSDR reconstruction on the training prefix;
`Test Set PSNR` reads the repaired PartMM evaluation on held-out poses. Both
use the same four cameras and target-frame Gaussian colors. The command prints
only those two metrics and writes per-case JSON plus `runs/table3_psnr/summary.json`.
Use `--method lbs` or `--method bob` for those baselines. Their training PSNR
uses each method's saved driven sequence; their test PSNR uses the matching
held-out evaluation. After all three methods finish, `--method all` prints the
six dataset means. BoB can be fitted on the same D3DGS checkpoints with
`scripts/reproduce_table3_bob.sh`.

## Local Reproductions

The paper matrix above is kept separate from experiments reproduced in other
article codebases on this workstation:

| Reproduction | Config | Checked-in evidence |
| --- | --- | --- |
| DressRecon baseline and motion matching | `reproductions/dressrecon_original.yaml` | 14 DNA sequences, 159 frames |
| DressRecon BoB / robust LBS / SSDR diagnostic | `reproductions/human_novel_pose.yaml` | 8 DNA + 10 ActorsHQ sequences |
| BoB historical 20k diagnostic | `reproductions/dressrecon_bob_recompute.yaml` | recovered filtered/subsampled path |
| Cross-method skeletonization | `reproductions/cross_method_skeletonization.yaml` | 65 attempted runs, including failures |
| Deform-2DGS observation split | `reproductions/deform2dgs_observation_split.yaml` | 20 runs, 14 complete |

`reproductions/catalog.yaml` records source revisions and completion status.
`gaussianimate-sync-reproductions` regenerates all portable JSON files and
their SHA-256 provenance from the local workspaces.

The snapshots contain metrics, case identities, status, and source hashes.
Datasets, model checkpoints, rasterizer binaries, and render trees are not
copied because they are large or independently licensed.

## Multiview Resources

`multiview_gaussian/resources.yaml` is the complete resource contract for the
8 DNA-Rendering and 10 ActorsHQ sequences in the recovered table aggregate.
The paper prose declares 8 ActorsHQ subjects without naming the subset, so the
10-sequence aggregate is retained and the mismatch is reported as unresolved.
The contract records raw captures, annotation files, exact recovered windows,
prepared RGB/mask directories, checkpoints, and derived method caches.

As of 2026-09-11, D3DGS, robust LBS, training-only SSDR, and paper-spec PartMM
artifacts pass the current validator for all 18 cases (8 DNA and 10 ActorsHQ).
Twelve cases also retain older all-Gaussian BoB models and held-out metrics,
but those models use the previous 66-dimensional pose contract and cannot be
relabeled as current results. The released sigmoid-RGB Table 3 rerun records
the 69-dimensional BoB-v2 recomputation complete for all 18 cases (8 DNA and
10 ActorsHQ).

Prepare stable links to the NAS-backed ActorsHQ source and run the complete
raw-data audit:

```bash
gaussianimate-multiview-resources --prepare-links --full --require-raw \
  --require-runtime \
  --output experiments/multiview_gaussian/resource_audit.local.json
```

Add `--require-novel-pose-base --require-derived` for the final release gate.
The derived gate covers strict LBS, BoB, training-only SSDR, and PartMM for
every declared sequence; historical aggregate-only evidence cannot satisfy it.

The full audit checks every RGB/mask filename referenced by metadata, all 300
SMPL-X poses per ActorsHQ paper window, DNA SMC and annotation files, and the
initial point-cloud schema. It also imports the reconstruction, LBS, and BoB
runtime modules and checks their model and reader assets. Use `--case CASE_ID`
for a single-sequence preflight. The local audit contains machine paths and is
intentionally excluded from version control.

Freeze the recovered 18-case evaluation summaries separately from models and
per-frame predictions:

```bash
gaussianimate-multiview-evidence \
  --bob-model-root ~/research/GA/Dynamic3DGaussians/output/bob_results_iso \
  --bob-model-root ~/research/code/dressrecon/bob_output \
  --adopt-dir "$RELEASE_ROOT/resources/multiview/historical_evaluation"
```

These recovered D3DGS, LBS, and BoB aggregates correspond to the printed rows.
The table-era BoB archive is aggregate-only evidence because only two of its
18 fitted models were recovered; it is separate from the complete fresh
BoB-v2 runs. The historical SSDR result is explicitly rejected as PartMM.

The config separates `smpl_partmm`, which follows the written 5-part/k=7/
5-level protocol, from `smpl_partmm_historical`, which executes and hashes the
recovered table-era source. Both run training-only SSDR, SMPL motion export,
held-out D3DGS rendering, and metric collection. They are kept separate because
the recovered source uses six fixed SMPL groups and a different refinement
path. Neither is table evidence until the discrepancy is resolved and every
case passes the final metric record.

## Monocular Resources

`monocular_gaussian/resources.yaml` locates the historical GaussiAnimate
reconstruction-stage and RigGS final-stage test renders. The audit verifies the
method flags in `cfg_args`, selects the latest complete iteration, and requires
exactly 20 D-NeRF or 200 DG-Mesh prediction/ground-truth pairs.

```bash
gaussianimate-monocular-resources --content-hash \
  --prepare-release \
  --output experiments/monocular_gaussian/resource_audit.local.json
```

Release preparation creates per-image links and provenance JSON instead of
copying the large model trees. RigGS auxiliary `_rig` renders are deliberately
excluded from metric inputs.

The historical training code records a different metric path from the later
PNG evaluator. Recover both without conflating them:

```bash
gaussianimate-monocular-resources --content-hash --prepare-release \
  --evaluate-release --extract-events --package-evaluation \
  --output experiments/monocular_gaussian/resource_audit.local.json
```

`summary.json` contains metrics recomputed from quantized PNG files.
`tensorboard_summary.json` selects each sequence's maximum test PSNR and reads
SSIM and LPIPS from the same training step. The latter reproduces the RigGS
D-NeRF paper row after rounding; the recovered GaussiAnimate stage-one tree
does not reproduce the paper's Ours row and is classified as diagnostic
evidence only.

### Direct SSDR evaluation

RigGS rendering and the recovered Dem Bones binding use different Python
environments. The Horse adapter therefore makes the trajectory boundary
explicit instead of relying on a machine-specific mixed environment:

```bash
# RigGS environment: query and cache the registered Gaussian trajectory.
python experiments/monocular_gaussian/ssdr_evaluate.py \
  --config experiments/monocular_gaussian/horse_ssdr.yaml --stage extract

# Dem Bones environment: fit density-invariant motion clusters and SSDR.
python experiments/monocular_gaussian/ssdr_evaluate.py \
  --config experiments/monocular_gaussian/horse_ssdr.yaml --stage fit \
  --trajectory-cache "$GA_HORSE_SSDR_OUTPUT/trajectory.npz"

# RigGS environment: render the LBS trajectory and evaluate held-out views.
python experiments/monocular_gaussian/ssdr_evaluate.py \
  --config experiments/monocular_gaussian/horse_ssdr.yaml --stage render \
  --ssdr-result "$GA_HORSE_SSDR_OUTPUT/ssdr_results_bones50.npz"
```

The config requires all 50 requested bones. A collapsed initializer now fails
immediately rather than producing a misleading metric from one or two bones.

There are two explicit SSDR protocols:

- `motion_pointcloud` is the public default. It creates exactly the requested
  number of nonempty motion clusters before Dem Bones optimization and is
  insensitive to Gaussian point density.
- `paper_legacy` freezes the recovered June 2025 behavior: threshold `3`,
  single-point split seeds, global best-error reassignment, and no point-cloud
  KNN pre-smoothing. Use
  `monocular_gaussian/horse_ssdr_paper_legacy.yaml` only with the matching
  24,615-point historical trajectory cache.

Every new SSDR artifact stores `requested_bones`, `initialized_bones`,
`num_bones`, `effective_bones`, hard populations, soft masses, and the loaded
backend path and SHA-256. These fields distinguish a successful decomposition
from an output whose filename merely contains the requested count.

The two qualitative ablations have separate historical-evidence audits:

```bash
gaussianimate-progressive-evidence --adopt-dir "$RELEASE_ROOT/resources/monocular/progressive_skeleton"
gaussianimate-arap-evidence --raw-node-manifest "$RELEASE_ROOT/resources/monocular/arap_ssdr/girlwalk/arap_no_ssdr/raw_node_skinning.json" \
  --adopt-dir "$RELEASE_ROOT/resources/monocular/arap_ssdr/girlwalk"
```

These commands retain the paper PDF, implementation, checkpoint, and result
hashes without treating an incomplete historical branch as a reproduction.

## Mesh Resources

The VTO evidence command validates every stored prediction against its ground
truth, recomputes the reported per-case distance, freezes content hashes, and
copies all eight result arrays into the release resource tree:

```bash
gaussianimate-vto-evidence \
  --robust-tshirt-root experiments/runs/vto_animation/vto/tshirt/robust_lbs/lbs_arrays \
  --robust-dress-root experiments/runs/vto_animation/vto/dress/robust_lbs/lbs_arrays \
  --adopt-dir "$RELEASE_ROOT/resources/mesh/vto"
gaussianimate-dt4d-evidence \
  --adopt-dir "$RELEASE_ROOT/resources/mesh/dt4d"
gaussianimate-runtime-evidence \
  --adopt-dir "$RELEASE_ROOT/resources/mesh/runtime"
```

The VTO package includes the recomputed Robust-LBS transferred weights and
per-frame predictions in addition to the four recovered comparison methods.
The resulting manifest records paper consistency per method. It does not
replace a differing recomputation with the number printed in the paper. The
D4D package is explicitly aggregate-only because the recovered artifact has
per-instance metrics but no per-frame prediction geometry.
For both VTO and D4D, the recovered evaluator computes mean per-vertex
Euclidean distance in millimeters; `RMSE` is the paper's label, not the
implemented reduction.
The runtime package keeps the exact 13.5 ms paper record and the later
remeasurement as separate files; it also records that the original measurement
environment is not recoverable from the current archive.

## Run

Validate or inspect a config without loading datasets:

```bash
gaussianimate-paper experiments/mesh/dt4d_animation.yaml --validate-only
gaussianimate-paper experiments/mesh/dt4d_animation.yaml --list
gaussianimate-paper experiments/mesh/dt4d_animation.yaml --dry-run
```

Run one method or the complete matrix:

```bash
gaussianimate-paper experiments/mesh/dt4d_animation.yaml --method partmm
gaussianimate-paper experiments/multiview_gaussian/novel_pose.yaml \
  --dataset dna_rendering --workers 4 --gpus 0,1,2,3
```

GPU IDs are assigned round-robin. At most one job runs on each declared GPU,
even when `--workers` is larger than the GPU count. Parallel stages that
declare or explicitly select CUDA require `--gpus`; CPU-only jobs may still
use multiple workers without that option.

Stages are resumable. Each job writes `job.json`, stage logs, standard metric
JSON, and the aggregated `results.csv`, `averages.csv`, and `summary.md`.
`--force` recomputes completed stages.

Legacy stage records created before fingerprints are never accepted
automatically. They can be migrated only when a current completed downstream
stage consumed the exact declared output:

```bash
gaussianimate-paper CONFIG --case CASE --method METHOD \
  --stage STAGE --adopt-legacy
```

Leaf stages and records whose current downstream fingerprint is stale must be
recomputed.

## Execution Modes

- `native`: implemented by the public `skelebone`, `partmm`, exporters, and
  metric code in this repository.
- `external`: an executable adapter to the historical RigGS, D3DGS, HOOD/VTO,
  or D4D implementation. Paths are environment-overridable and default to the
  local research workspace.
- `artifact`: evaluation of a published baseline output whose independently
  licensed training repository is not vendored here.
- `blocked`: a declared paper method whose current implementation would not
  follow the paper protocol. The runner refuses to execute it.

This distinction is recorded in every job manifest. A baseline artifact is
never reported as a locally trained method.

For the local reproductions, `artifact` means a checked-in snapshot derived
from an actual local run. The corresponding source result hash is stored in
that snapshot. It is distinct from a paper number entered by hand.

## Protocol

The configs record the intended paper settings: 40k monocular reconstruction
iterations; 8/2 real-human splits; ActorsHQ high-motion 300-frame selection and
4x downsampling; VTO five-motion low-data training; D4D high-motion frame
selection; 50 outer bones; joint threshold `0.3`; and PartMM defaults of five
parts, seven neighbors, patch size seven, five pyramid levels, and blend `0.7`.
Protocol correspondence and completed outputs are separate acceptance checks;
the catalog audit is authoritative.

The ActorsHQ table was computed from 10 sequences even though the prose says
"8 subjects"; the config preserves the 10-sequence table protocol and records
the discrepancy. The D4D source has 57 categories, but the recovered
26-instance artifact uses motion-prefix leave-one-out with fallbacks rather
than the paper's stated uniform split. Its LBS and PartMM-labelled fields are
historical protocol proxies, and FullMM re-aggregates to 53.4 rather than the
displayed 53.3.
