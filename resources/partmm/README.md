# Prepare the PartMM T-shirt Example

Obtain these two resources directly from their authors:

1. [VTO T-shirt simulations](https://github.com/isantesteban/vto-dataset/tree/main/tshirt/simulations).
2. [SMPL model](https://smpl.is.tue.mpg.de/) (register, accept the terms, and download the female model).

Use the `shape00` simulations below, preserving their filenames under
`<vto-root>/tshirt/simulations/`:

| Role | Files | Frames |
| --- | --- | --- |
| Source | `tshirt_shape00_01_01.pkl` | All 241 |
| Source | `tshirt_shape00_02_04.pkl` | All 121 |
| Source | `tshirt_shape00_02_05.pkl` | All 121 |
| Source | `tshirt_shape00_05_20.pkl` | All 274 |
| Source | `tshirt_shape00_07_02.pkl` | All 83 |
| Query and ground truth | `tshirt_shape00_91_33.pkl` | All 203 |

## Generate Locally

From the repository root:

```bash
python -m pip install --no-build-isolation -e '.[vto,visualization]'
bash scripts/install_geometry.sh --pyssdr-only --python "$(command -v python)"
python -m experiments.mesh.prepare_partmm resources/partmm/example \
  --vto-root /path/to/vto-dataset \
  --smpl-model-path /path/to/SMPL_FEMALE.pkl \
  --num-bones 20 --iterations 100 --seed 42
python -m partmm.examples.quickstart outputs/partmm_tshirt_demo \
  --bundle resources/partmm/example
```

Preparation fits SSDR locally and writes `partmm_input.npz`,
`partmm_query.npz`, `partmm_reference.npz` (garment faces, GT, and query
joints), and `preparation.json`. Continue with the
[PartMM guide](../../partmm/README.md#run-the-vto-t-shirt-example) for animation,
rendering, and evaluation.

The fitted SSDR arrays in `partmm_input.npz` are `source_points` (rest
vertices), `weights` (point-by-bone), and `bone_transforms` (frame-by-bone
homogeneous transforms). They can be read with `numpy.load(..., allow_pickle=False)`.

The source and query use the original vertex order, meters, and pelvis-centered
root-motion removal. The female SMPL model uses zero shape coefficients for
`shape00`. SSDR is fitted only to the source sequences. See
[manifest.json](manifest.json) for provenance.

Only processing code is distributed here. Load simulation/model pickle files
only from trusted sources. VTO data and SMPL assets retain their upstream
terms; the repository's MIT license does not license those assets or grant
commercial-use rights. Consult the [SMPL-Model](https://smpl.is.tue.mpg.de/modellicense.html)
and [SMPL-Body](https://smpl.is.tue.mpg.de/bodylicense.html) terms as applicable.
The VTO repository does not publish an explicit commercial-use license;
confirm permission with its authors. SMPL's standard model license is for
non-commercial use. Internal commercial evaluation is not automatically
authorized by keeping the results private; obtain the applicable permission
from each rights holder.
