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
overrides it needs, so set only those before running. Experiment-specific
scripts live under `experiments/backends`; upstream rendering engines, body
models, and raw datasets are obtained separately through [Resources](#resources).

## Resources

Official sources for the external data, model files, and research code used
by the tables below. Obtain each resource from its source under its own
license; nothing in this table is redistributed. Research backends bundled in
this repository instead carry their source citations at the top of each file
and are not listed here.

| Resource | Get It |
| --- | --- |
| AMA | [Project](https://people.csail.mit.edu/drdaniel/mesh_animation/) / [T_samba meshes](https://groups.csail.mit.edu/graphics/mesh_animation/T_samba/meshes.tgz) |
| VTO dataset | [Simulations and data format](https://github.com/isantesteban/vto-dataset) |
| HOOD | [Code and installation](https://github.com/Dolorousrtur/HOOD) / [Auxiliary data](https://drive.google.com/file/d/1RdA4L6Fy50VsKZ8k7ySp5ps5YtWoHSgs/view) |
| DeformingThings4D | [Data application and loader](https://github.com/rabbityl/DeformingThings4D#download-data-google-form) (application required) |
| DNA-Rendering | [Request data](https://dna-rendering.github.io/inner-download.html) (license plus email request; request the `data_used_in_4K4D` subset listed in the config) |
| ActorsHQ | [Dataset](https://actors-hq.com/) / [Access request and downloader](https://github.com/synthesiaresearch/humanrf#data-access-and-download) |
| SMPL / SMPL-X | [SMPL](https://smpl.is.tue.mpg.de/) / [SMPL-X](https://smpl-x.is.tue.mpg.de/) (register and accept the model license; obtain model files separately) |
| Other Table 2 baselines | [TiNeuVox](https://github.com/hustvl/TiNeuVox) / [4D-GS](https://github.com/hustvl/4DGaussians) / [AP-NeRF](https://github.com/lukasuz/Articulated-Point-NeRF) |
| Dynamic3DGaussians | [Code and installation](https://github.com/JonathonLuiten/Dynamic3DGaussians) |
| DressRecon / BoB | [Code and installation](https://github.com/jefftan969/dressrecon) |
| D-NeRF dataset | [Official instructions](https://github.com/albertpumarola/D-NeRF#download-dataset) / [Data archive](https://www.dropbox.com/scl/fi/cdcmkufncwcikk1dzbgb4/data.zip?rlkey=n5m21i84v2b2xk6h7qgiu8nkg&dl=0) |
| DG-Mesh dataset | [Official instructions](https://github.com/Isabella98Liu/DG-Mesh#training-on-the-dg-mesh-dataset) / [Data archive](https://drive.google.com/file/d/1yBga2DsIKG6zQK9V2WextewvhaV8Xho3/view) |

**Author-generated resources:** the PartMM example, SSDR databases, and
reference caches are organized under
[`resources/partmm/`](../resources/partmm/README.md) for a separate download.
No public download URL is available yet; official HOOD/VTO downloads
do not contain these fitted caches.

Trained Gaussian/BoB checkpoints and processed poses/evaluation data are not
being released at this time. The commands below operate on local inputs or
freshly trained results; they do not download our checkpoints. Table 2 baseline
render trees also need to be generated with the respective upstream methods.

Before redistributing the copied research backends, review their
[licensing notices](../THIRD_PARTY_NOTICES.md); copying a dependency into this
repository does not change its license.

**Remaining code dependency:** the VTO Table 4/5/R.1 runners still use
author-specific HOOD experiment scripts that have not yet been migrated.
Cloning official HOOD does not supply those scripts. The D4D benchmark's
full regeneration entry point also remains to be migrated; its independent
raw-data converter is already included below.

## Table 2: Synthetic Novel-View Rendering

**Prepare:** a trained monocular Gaussian reconstruction and `pyssdr`. The
monocular backend code is bundled under `experiments/backends/monocular/`;
its Python dependencies come from the `monocular` extra. The CUDA extensions
[rasterizer](https://github.com/ashawkey/diff-gaussian-rasterization) and
[simple-knn](https://gitlab.inria.fr/bkerbl/simple-knn), and
[PyTorch3D](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md),
are not included and must be installed
separately into the same environment. The Horse example below fits 50 SSDR
bones and renders the test views.

```bash
python -m pip install -e '.[monocular]'
export GA_HORSE_DATA=/path/to/dgmesh/horse
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

To train a reconstruction first, place the official sequences under
`GA_SYNTHETIC_DATA_ROOT/<sequence>` and run the bundled backend:

```bash
export GA_SYNTHETIC_DATA_ROOT=/path/to/synthetic_sequences
gaussianimate-paper experiments/monocular_gaussian/novel_view.yaml \
  --method ours --dataset dgmesh --case horse --gpus 0
```

Set `GA_HORSE_RECONSTRUCTION` to the resulting `model_node` directory for
the SSDR command above. This training command alone evaluates reconstruction,
not the SSDR result.

## Table 3: Human Novel-Pose Rendering

```bash
export GA_MULTIVIEW_RELEASE_ROOT=/path/to/release
export GA_EXPERIMENT_RUNS_ROOT="$GA_MULTIVIEW_RELEASE_ROOT/runs"
```

**Prepare:** [DNA-Rendering](#resources) or [ActorsHQ](#resources) captures
and pose annotations. Set the external runtime and model paths in
[resources.yaml](multiview_gaussian/resources.yaml) and the method YAMLs
under [multiview_gaussian/methods/](multiview_gaussian/methods/):
[Dynamic3DGaussians](#resources) (D3DGS) and [DressRecon](#resources)
research code and [SMPL](#resources)/[SMPL-X](#resources) model files.
The research PartMM adapter is included under
`experiments/backends/partmm_research/`; it is separate from the standalone
NumPy-only `partmm/` runtime.

**Capture selection:** DNA uses `0012_11`, `0013_01`, `0013_03`, `0013_09`,
`0013_11`, `0019_08`, `0021_03`, and `0023_06`, with 150 prepared frames per
capture: frames 0-119 for motion fitting and 120-149 for novel-pose evaluation.
The conversion scripts use raw frames 0-149; the older checkpoints do not
record an independently verifiable raw-to-prepared mapping.

ActorsHQ uses these inclusive raw frame ranges, downsampled by 4 in image
resolution. Each capture has 240 motion-fitting frames followed by 60 test
frames; frame order is not shuffled.

| Official Capture | Train Frames | Test Frames |
| --- | --- | --- |
| Actor01 / Sequence1 | 822-1061 | 1062-1121 |
| Actor01 / Sequence2 | 758-997 | 998-1057 |
| Actor02 / Sequence1 | 743-982 | 983-1042 |
| Actor02 / Sequence2 | 1013-1252 | 1253-1312 |
| Actor04 / Sequence2 | 717-956 | 957-1016 |
| Actor05 / Sequence1 | 870-1109 | 1110-1169 |
| Actor06 / Sequence1 | 916-1155 | 1156-1215 |
| Actor06 / Sequence2 | 206-445 | 446-505 |
| Actor07 / Sequence1 | 966-1205 | 1206-1265 |
| Actor08 / Sequence1 | 858-1097 | 1098-1157 |

Full capture names, raw-frame rules, cameras and fitting/test ranges are
recorded in [frame_splits.yaml](multiview_gaussian/frame_splits.yaml). The
split applies to SSDR and motion transfer; D3DGS reconstructs the observation
sequence, including frames later used as motion-transfer test poses.

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

Some D3DGS reconstructions contain distant Gaussian-center artifacts.

## Table 4: Garment and Animal Animation

**VTO** needs the [HOOD](#resources)/VTO runtime, [SMPL](#resources) model
files, and the [VTO](#resources) dataset:

```bash
gaussianimate-paper experiments/mesh/vto_animation.yaml --gpus 0
```

**D4D:** obtain the raw data from [DeformingThings4D](#resources), then convert
an `.anime` capture into the public point-sequence format. This step needs no
private experiment workspace:

```bash
python -m experiments.mesh.d4d_prepare \
  --input /path/to/animal_motion.anime --output outputs/d4d/sequence.npz
gaussianimate outputs/d4d/sequence.npz outputs/d4d/skelebones --num-bones 50
```

The historical Table 4 aggregation remains available below, but its complete
multi-sequence benchmark regeneration still depends on the legacy runner.
The preprocessing command above is not a substitute for that benchmark:

```bash
gaussianimate-paper experiments/mesh/dt4d_animation.yaml --gpus 0
```

**Configs:** [vto_animation.yaml](mesh/vto_animation.yaml) and
[dt4d_animation.yaml](mesh/dt4d_animation.yaml).

**Output:** mean vertex error in millimeters, in each config's `output_dir`
under `experiments/runs/`.

## Table 5: Temporal Patch-Length Ablation

Needs the external VTO PartMM runner and a 5-sequence SSDR cache. Set
`GA_VTO_PARTMM_CODE` (runner code), `GA_VTO_ROOT` ([VTO](#resources)
dataset), `GA_SMPL_MODEL_DIR` ([SMPL](#resources) models), and
`GA_VTO_SSDR_CACHE` (SSDR cache);
[patch_length.yaml](mesh/patch_length.yaml) documents the defaults:

```bash
gaussianimate-paper experiments/mesh/patch_length.yaml --gpus 0
```

Outputs land in `experiments/runs/partmm_patch_length/` (per-case
`raw/per_case_summary.json` and `geometry_metrics.json` plus the standard
aggregates described at the end).

## Table R.1: Database Size, Accuracy, and Runtime

Profile first, then aggregate. Profiling runs in the
[HOOD](#resources)/[VTO](#resources) Python environment: set
`GA_HOOD_PYTHON` to its interpreter (the mesh configs
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
