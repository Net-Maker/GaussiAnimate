# PartMM

PartMM transfers motion from an inner kinematic skeleton to free-form outer
bones. It retrieves motion locally for connected skeleton parts, blends the
retrieved rigid transforms, and applies them to canonical points with linear
blend skinning.

The directory is deliberately self-contained. Copy `partmm/` into another
project and the animation runtime will work with NumPy alone; it never imports
GaussiAnimate or `skelebone`.

![VTO T-shirt PartMM animation](examples/partmm_hood.gif)

```text
source skeleton motion + outer-bone motion   novel skeleton motion
                    \                         /
                     \---- PartMM retrieval -/
                                  |
                       animated points + transforms
```

## Run the VTO T-shirt example

The `tshirt_shape00_91_33` example uses an inner SMPL control rig and an outer
SSDR deformation rig fitted to the official VTO T-shirt simulations.

Follow [data preparation](../resources/partmm/README.md) to build the example
locally from the official VTO simulations and SMPL model.

Animate the prepared inputs:

```bash
python -m partmm resources/partmm/example/partmm_input.npz \
  resources/partmm/example/partmm_query.npz outputs/partmm_tshirt.npz \
  --partition-mode humanoid_five
```

Render and evaluate the same bundle:

```bash
python -m pip install "numpy<1.26" "open3d>=0.19,<0.20" Pillow

python -m partmm.examples.quickstart outputs/partmm_tshirt_demo \
  --bundle resources/partmm/example
```

Use `--preview-format none` for headless evaluation.

It creates five inspectable artifacts:

```text
outputs/partmm_tshirt_demo/
├── partmm_input.npz    source motion database
├── partmm_query.npz    novel inner-skeleton motion
├── partmm_output.npz   animated points and retrieved transforms
├── partmm_preview.gif  synchronized SMPL, SSDR, and output animation
└── metrics.json        case, config, dimensions, and RMSE against cloth GT
```

`metrics.json` reports coordinate RMSE and mean Euclidean vertex error against
the prepared garment ground truth.

[`examples/quickstart.py`](examples/quickstart.py) shows the complete animate,
render, and evaluate flow over the prepared files.

## Use a skelebone result

The recommended input is the `skelebones.npz` produced by GaussiAnimate. Its
field names already match the standalone PartMM contract.

```python
from partmm import (
    PartMMConfig,
    animate,
    load_input,
    load_query,
    save_output,
)

database = load_input("skelebones.npz")
query = load_query("novel_pose.npz")

result = animate(database, query, PartMMConfig())
save_output("animation.npz", result)
```

`PartMMConfig` uses the paper's patch, neighbor, and pyramid sizes, with explicit
alignment and blending conventions. The included T-shirt
example uses the semantic five-part SMPL partition and exposes the main matching
parameters as command-line options.

## Use NumPy arrays directly

No GaussiAnimate class is required. Construct the two public inputs directly:

```python
from partmm import PartMMConfig, PartMMInput, PartMMQuery, animate

database = PartMMInput(
    rest_joints=rest_joints,                    # [J, 3]
    parents=parents,                            # [J], root parent is -1
    source_local_rotations=source_quaternions,  # [T, J, 4], [w, x, y, z]
    source_root_translation=source_root_motion, # [T, 3]
    bone_transforms=outer_bone_transforms,      # [T, B, 4, 4]
    source_points=canonical_points,             # [N, 3]
    skinning_weights=weights,                   # [N, B]
    fps=30.0,
    joint_names=joint_names,                    # optional length-J sequence
)

query = PartMMQuery(
    local_rotations=query_quaternions,          # [Q, J, 4]
    root_translation=query_root_motion,         # [Q, 3]
    fps=30.0,
)

result = animate(database, query, PartMMConfig())
```

Weights are normalized on input. Quaternions are scalar-first and are also
normalized. Invalid shapes, non-finite values, cyclic parents, and zero-weight
points fail early with a `ValueError`.

## Data contract

### Source database

| Array | Shape | Meaning |
| --- | --- | --- |
| `rest_joints` | `[J, 3]` | Inner-skeleton joints in the bind pose |
| `parents` | `[J]` | Acyclic hierarchy; each root uses `-1` |
| `source_local_rotations` | `[T, J, 4]` | Database motion as scalar-first local quaternions |
| `source_root_translation` | `[T, 3]` | Root displacement from the bind pose |
| `bone_transforms` | `[T, B, 4, 4]` | Outer-bone transforms from canonical space to each source frame |
| `source_points` | `[N, 3]` | Canonical points animated by the outer bones |
| `skinning_weights` | `[N, B]` | Point-to-outer-bone blend weights |
| `fps` | scalar | Source frame rate |

Use `save_input()` and `load_input()` for a compressed, pickle-free NPZ. The
loader accepts both this schema and GaussiAnimate's `skelebones.npz`.

### Query motion

| Array | Shape | Meaning |
| --- | --- | --- |
| `local_rotations` | `[Q, J, 4]` | Novel local joint rotations; topology must match the database |
| `root_translation` | `[Q, 3]` | Novel root displacement |
| `fps` | scalar | Output frame rate |

Use `save_query()` and `load_query()` to exchange query motions.

### Output

| Array | Shape | Meaning |
| --- | --- | --- |
| `points` | `[Q, N, 3]` | Animated points |
| `bone_transforms` | `[Q, B, 4, 4]` | Retrieved and blended outer-bone transforms |
| `joint_positions` | `[Q, J, 3]` | Query skeleton in global coordinates |
| `matched_source_frames` | `[P, W, K]` | Retrieved source starts per part, query window, and neighbor |
| `matched_part_ids` | `[P]` | Part ID for each matching array |
| `fps` | scalar | Output frame rate |

`save_output()` preserves both the animation and retrieval provenance.

## Keep a skin-transfer baseline

A nearest neighbor always exists, even when every example is a poor match.
Normalized KNN weights do not measure that uncertainty. For an asset with
Robust Skin Transfer weights, keep its LBS prediction as a baseline and
calibrate the retrieval decision on a held-out segment of the **training**
motion.

`SkinningBaseline` consumes transferred point-to-inner-joint weights `[N, J]`
and points in one observed reference pose. It applies relative joint transforms
to those points. Weight transfer itself remains a preprocessing step; the
NumPy runtime does not replace it with nearest-bone assignments.

`PartMMFallback` compares position errors per body part on the training
validation segment. The default `distance_policy="validation_error"` enables
retrieval only for parts that beat the baseline there. Optional query-distance
gates are experimental: `distance_policy="validation"` uses a strict validation-distance
quantile. `distance_policy="error_ratio"` scales that limit by the measured
baseline-to-candidate squared-error ratio: a large demonstrated improvement
allows more pose variation. This second policy assumes local position MSE
grows with squared angular distance; it is an estimate, not an error bound.
Part boundaries blend using the transferred weights.

```python
from partmm.fallback import PartMMFallback

# Validation predictions must come from a database excluding this segment.
guard = PartMMFallback.calibrate(
    validation_partmm,          # [V, N, 3]
    validation_skin_transfer,   # [V, N, 3]
    validation_observations,    # [V, N, 3], training data only
    point_part_weights,        # [N, P], summed from transferred joint weights
    validation_distances,      # [V, P], mean squared SO(3) radians
    distance_policy="validation_error",
)
points, diagnostics = guard.apply(
    query_partmm, query_skin_transfer, query_distances,
)
```

Use `partmm.matching.frame_match_distances()` with the matcher's part masks to
compute these absolute distances. The diagnostics report which parts passed
validation, their errors, and how often each part falls back. This is a
validation-based decision, not a guarantee about unknown test geometry.
`animate()` alone remains the raw retrieval API; it cannot provide Robust Skin
Transfer without the transferred weights or a baseline prediction. Guarded
`points` combine two rigs, so the raw retrieval's `bone_transforms` must not be
presented as transforms reproducing that combined output.

The repository's `experiments/multiview_gaussian/partmm_repair.yaml` runs this
comparison using existing training-only SSDR and Robust LBS artifacts:

```bash
python -m experiments.multiview_gaussian.partmm_repair \
  --config experiments/multiview_gaussian/partmm_repair.yaml
```

Alignment is evaluated per frame inside each retrieved patch. Transform
blending preserves each bone's weighted influence center.

## Visualize a result

The Open3D GIF renderer is optional and separate from the core runtime:

```bash
python -m pip install "open3d>=0.19,<0.20" Pillow
```

```python
from partmm.open3d_renderer import save_partmm_gif

save_partmm_gif(
    "partmm_preview.gif",
    result,
    faces=faces,
    parents=database.parents,
    source_points=database.source_points,
    skinning_weights=database.skinning_weights,
)
```

The three panels use one fixed camera and timeline: SMPL query motion, SSDR
outer-bone frames, and the PartMM surface output. For a dependency-light static
diagnostic, install Matplotlib and run the example with `--preview-format png`.

## Command line

After installing the full GaussiAnimate package, the same file contract is
available through:

```bash
gaussianimate-animate skelebones.npz novel_pose.npz animation.npz
```

For a copied folder without package installation:

```bash
python -m partmm skelebones.npz novel_pose.npz animation.npz
```

Validate shapes and counts without running retrieval:

```bash
python -m partmm skelebones.npz novel_pose.npz --validate-only
```

## What to copy

Copy the complete `partmm/` directory, including `examples/`, alongside your
prepared input and query files. Optional preview dependencies are Open3D and
Pillow for GIF, Matplotlib for PNG, and Dear PyGui for `partmm.gui`.
