# Gaussian and Point-Cloud Animation

Input must be registered point trajectories: row `n` is the same point in
every frame.

Install the [core and geometry backends](../README.md#quick-start) first.

## Import Your Sequence

For ordered PLY or NPY frames:

```bash
python -m skelebone.pointcloud /path/to/frames outputs/my_asset/sequence.npz \
  --pattern 'pointcloud_t*.ply'
```

For Gaussian means, add `--gaussian-ply /path/to/point_cloud.ply` from the
checkpoint that produced the trajectories, in exactly the same row order.

## Extract and Animate

For Gaussians, [export a TSDF mesh](#start-from-a-monocular-checkpoint) and
append `--mesh /path/to/canonical_mesh.ply` to the extraction command.

```bash
gaussianimate outputs/my_asset/sequence.npz outputs/my_asset/result --num-bones 50
gaussianimate-view outputs/my_asset/result/skelebones.npz \
  --sequence outputs/my_asset/sequence.npz
```

Open the printed URL, select a joint, and rotate it with the gizmo or angle
controls. The reference toggle plays separately and never changes the edited
pose. `Auto` uses PartMM when every part has a source pose within
`--max-match-angle` degrees (default `15`) and falls back to LBS otherwise;
choose `LBS` to keep direct skeletal control for every pose. Export pose
saves the displayed points and pose as NPZ. The viewer renders colored
Gaussian **centers**, not a Gaussian splatting render.

## Start From a Monocular Checkpoint

Export a trained checkpoint using the
[monocular CUDA environment](../experiments/README.md#table-2-synthetic-novel-view-rendering):

```bash
python -m experiments.monocular_gaussian.export /path/to/model \
  /path/to/tracks/dnerf_trex_gaussiananimator \
  --source-path /path/to/dnerf/trex --iteration 40000
python -m experiments.monocular_gaussian.tsdf /path/to/model \
  /path/to/tracks/dnerf_trex_gaussiananimator/canonical_mesh.ply \
  --source-path /path/to/dnerf/trex --iteration 40000 --fid 0
```

Install `open3d>=0.19,<0.20` in the monocular environment for TSDF fusion.
For each configured D-NeRF sequence, export both the registered frames and
TSDF mesh into `dnerf_<scene>_gaussiananimator`, then run:

```bash
export GA_POINT_TRACK_ROOT=/path/to/tracks
export GA_OUTPUT_ROOT="$PWD/outputs/dnerf_skelebone"
python -m experiments.monocular_gaussian.skelebone \
  --config experiments/monocular_gaussian/skelebone.yaml
gaussianimate-view "$GA_OUTPUT_ROOT" --up-axis z
```

`--case trex` processes one sequence, `--stage prepare|extract|animate`
selects a stage, and `--force` recomputes completed outputs.

The viewer loads Three.js and Lucide modules from the jsDelivr CDN and needs
internet access; extraction and PartMM run locally.
