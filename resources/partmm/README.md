# PartMM Resource Bundle

Real HOOD/VTO T-shirt inputs and author-generated reference caches. The
example uses SMPL as its inner rig and a 20-bone SSDR outer rig.

| Directory | Contents |
| --- | --- |
| `example/` | Pickle-free `partmm_input.npz` and `partmm_query.npz` for `tshirt_shape00_91_33` |
| `ssdr/` | Databases of 5, 20, and 50 source sequences, each with 20 SSDR bones; the combined T-shirt SSDR result |
| `reference/` | T-shirt ground truth and recorded scalability/runtime results |

See [manifest.json](manifest.json) for exact file sizes and SHA-256 checksums.
The bundle contains nine files, about 1.01 GiB, copied without changing their
contents. It contains no SMPL model files or trained Gaussian/BoB checkpoints.

**Download: pending author upload.** These local files are not committed to
Git or included in the Python package. A public sharing URL must be added
before readers can download the fitted caches. Their use and redistribution
remain subject to the [VTO dataset](https://github.com/isantesteban/vto-dataset)
and applicable body-model terms.

After obtaining this bundle, place its contents under `resources/partmm/` in
the repository. `example/` is the small standalone-demo download; the larger
`ssdr/` databases are needed only for the paper ablations.
