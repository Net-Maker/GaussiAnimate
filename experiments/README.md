# Paper Experiments

We have fixed several implementation bugs since the original experiments, so reproduced results may differ slightly from the values reported in the paper.

Choose the table you want to reproduce below.

## Setup

All commands run from the repository root. Install the repository following
[`../README.md#quick-start`](../README.md#quick-start), then export:

```bash
export GAUSSIANIMATE_ROOT="$PWD"
```

Config paths accept `GA_*` environment overrides; each table lists the
overrides it needs, so set only those before running. External method code,
model files, and datasets are not vendored in this repository.

## Table 2: Synthetic Novel-View Rendering

**Prepare:** a trained RigGS-compatible reconstruction, its CUDA rendering
environment, and `pyssdr`. The Horse example below fits 50 SSDR bones and
renders the test views; no RigGS stage-2 training is needed.

```bash
export GA_RIGGS_ROOT=/path/to/RigGS
export GA_HORSE_RECONSTRUCTION=/path/to/reconstruction
export GA_HORSE_SSDR_OUTPUT=/path/to/new/ssdr_evaluation
python -m experiments.monocular_gaussian.ssdr_evaluate \
  --config experiments/monocular_gaussian/horse_ssdr.yaml --stage all
```

**Config:** [horse_ssdr.yaml](monocular_gaussian/horse_ssdr.yaml). Set
`iteration` to your saved checkpoint; use a separate config and output
directory for each sequence.

**Output:** `metrics.json` (PSNR/SSIM/LPIPS), the SSDR fit, and `renders/`
with matching `gt/` images in `GA_HORSE_SSDR_OUTPUT`.

## Table 3: Human Novel-Pose Rendering

```bash
export GA_MULTIVIEW_RELEASE_ROOT=/path/to/release
export GA_EXPERIMENT_RUNS_ROOT="$GA_MULTIVIEW_RELEASE_ROOT/runs"
```

**Prepare:** DNA-Rendering or ActorsHQ captures and pose annotations. Set the
external runtime and model paths in
[resources.yaml](multiview_gaussian/resources.yaml) and the method YAMLs
under [multiview_gaussian/methods/](multiview_gaussian/methods/): D3DGS and
DressRecon research code, SMPL/SMPL-X model files, and the In2You /
Motion2Motion research code. None of these are vendored here.

```bash
gaussianimate-paper experiments/multiview_gaussian/reconstruction.yaml --gpus 0
gaussianimate-paper experiments/multiview_gaussian/novel_pose.yaml \
  --method d3dgs_gt --method smpl_lbs --method smpl_bob --method smpl_partmm \
  --gpus 0
```

**Configs:** [reconstruction.yaml](multiview_gaussian/reconstruction.yaml)
and [novel_pose.yaml](multiview_gaussian/novel_pose.yaml).

**Output:** per-case PSNR/SSIM/LPIPS under `GA_EXPERIMENT_RUNS_ROOT`, with
dataset means in `averages.csv`. Evaluation renders are used to compute
metrics but are not saved as images by this runner.

For completed repaired runs, print the TrainSet/TestSet PSNR comparison below.
This requires prior
TrainSet/TestSet metric caches for all three methods and
`runs/partmm_repaired_evaluation/*.json` under the release root, and is not
part of the fresh run chain.

```bash
python -m experiments.multiview_gaussian.table3_psnr \
  --root "$GA_MULTIVIEW_RELEASE_ROOT" --method all --summary-only
```

Some D3DGS reconstructions contain distant Gaussian-center artifacts; see the
[Actor01 analysis](multiview_gaussian/diagnostics/actor01_artifacts/README.md).

## Table 4: Garment and Animal Animation

**VTO** needs the HOOD/VTO runtime, SMPL model files, and the VTO dataset:

```bash
gaussianimate-paper experiments/mesh/vto_animation.yaml --gpus 0
```

**D4D** needs the external benchmark workspace and its recovered result summary:

```bash
gaussianimate-paper experiments/mesh/dt4d_animation.yaml --gpus 0
```

**Configs:** [vto_animation.yaml](mesh/vto_animation.yaml) and
[dt4d_animation.yaml](mesh/dt4d_animation.yaml).

**Output:** mean vertex error in millimeters, in each config's `output_dir`
under `experiments/runs/`.

## Table 5: Temporal Patch-Length Ablation

Needs the external VTO PartMM runner and a 5-sequence SSDR cache. Set
`GA_VTO_PARTMM_CODE` (runner code), `GA_VTO_ROOT` (VTO dataset),
`GA_SMPL_MODEL_DIR` (SMPL models), and `GA_VTO_SSDR_CACHE` (SSDR cache);
[patch_length.yaml](mesh/patch_length.yaml) documents the defaults:

```bash
gaussianimate-paper experiments/mesh/patch_length.yaml --gpus 0
```

Outputs land in `experiments/runs/partmm_patch_length/` (per-case
`raw/per_case_summary.json` and `geometry_metrics.json` plus the standard
aggregates described at the end).

## Table R.1: Database Size, Accuracy, and Runtime

Profile first, then aggregate. Profiling runs in the HOOD/VTO Python
environment: set `GA_HOOD_PYTHON` to its interpreter (the mesh configs
default to `~/anaconda3/envs/hood/bin/python`). The SSDR cache template must
contain `{sequences}`, with caches present for sizes 5, 20, and 50. The
profile JSON is written to `--profile-source`, which is also the file the
aggregation stage reads:

```bash
"$GA_HOOD_PYTHON" -m experiments.mesh.database_scaling --profile \
  --profile-source experiments/runs/partmm_database_scaling/raw/table_r1_profile.json \
  --raw-root experiments/runs/partmm_database_scaling/raw \
  --runner-code-root /path/to/partmm/runner/code \
  --vto-root /path/to/vto-dataset \
  --smpl-model-path /path/to/SMPL/models \
  --smpl-fk-model-dir /path/to/SMPL/models \
  --ssdr-cache-template /path/to/ssdr_cache/tshirt_n{sequences}_bones20.npy \
  --expected-cases tshirt_shape00_91_33,tshirt_shape00_91_36,tshirt_shape00_91_38,tshirt_shape00_91_61,tshirt_shape00_91_62 \
  --expected-sizes 5,20,50 --device cuda
```

Aggregation cross-checks the profile against an independent VTO accuracy
file (`GA_VTO_SCALING_ACCURACY`); see
[database_scaling.yaml](mesh/database_scaling.yaml) for the default path.

```bash
gaussianimate-paper experiments/mesh/database_scaling.yaml --gpus 0
```

**Output:** accuracy, pose coverage, runtime, and peak GPU memory in
`experiments/runs/partmm_database_scaling/`.

## Results

Each run writes per-case metrics plus `results.csv`, `averages.csv`, and
`summary.md` into the config's `output_dir`. `--case` and `--method` filter
the run to a subset; `--force` recomputes completed stages.
