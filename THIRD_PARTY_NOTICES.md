# Third-Party Notices

The root MIT license covers the original public implementation distributed in
the `skelebone`, `partmm`, and `experiments` Python packages. It does not
relicense external tools, datasets, pretrained models, or the local legacy
research snapshot. Each of those remains subject to its own terms.

## Native runtime tools

- `native/pyssdr` contains the headers from
  [DemBones-python](https://github.com/Net-Maker/DemBones-python), revision
  `2258a11a9eabbd659d04120eef0194a185485a0f`, and a focused binding for the
  API used by Skelebone extraction. Copyright (c) 2021 Shubh Maheshwari; MIT
  License. The vendored header differs only in the documented mesh/point-cloud
  adjacency selection and bounded-memory KNN implementation. Its license is
  included in `native/pyssdr/LICENSE`.
- `native/pyssdr/vendor/pybind11` is pybind11 2.8.0, distributed under its
  three-clause BSD license. Its license is included beside the vendored files.
- `native/curve_skeleton` uses the CGAL Surface Mesh Skeletonization package.
  That CGAL package is GPL-licensed, so this standalone executable and its
  source are distributed under GPL-3.0-or-later, separately from the root MIT
  Python packages. Its license is included in `native/curve_skeleton/LICENSE`.
- [Robust Skin Weights Transfer](https://github.com/rin-23/RobustSkinWeightsTransferCode),
  revision `a60263c17f38625b98a62bea09c91f61f7b865f0`, is the reference for
  `experiments/mesh/robust_weights.py`. Copyright (c) 2024 Rinat
  Abdrashitov; MIT License.

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to
  deal in the Software without restriction, including without limitation the
  rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
  sell copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in
  all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
  IN THE SOFTWARE.

These native sources are not copied into the Python wheel. They are built by
`scripts/install_geometry.sh`; Python dependencies installed from package
indexes retain their respective licenses.

## Reproduction adapters

- The Skelebone example reads the locally installed AMA `T_samba` sequence;
  the standalone PartMM example reads the HOOD/VTO
  `tshirt_shape00_91_33` case with SMPL motion and SSDR outer bones. The bundled
  previews show those respective cases. Raw sequences, body models and trained
  checkpoints are not redistributed by this package; users remain responsible
  for the terms governing their dataset and model copies.

- [DressRecon](https://github.com/jefftan969/dressrecon), local base revision
  `91b3b79`, is Copyright (c) 2023 Jeff Tan and Gengshan Yang and distributed
  under the MIT License. The pose-conditioned BoB modules under
  `experiments/reproductions/dressrecon_bob/` are adapted from this local
  reproduction; its MIT license is included beside the files.
- [Dynamic3DGaussians](https://github.com/JonathonLuiten/Dynamic3DGaussians),
  local base revision `7dbbd4d`, is distributed under the MIT License. The BoB
  batch and evaluation adapters derive from the local experiment code.
- [Deform-2DGS](https://github.com/Net-Maker/Deform-2DGS), local base revision
  `16a7093`, is represented only by normalized numerical results and source
  hashes.

The BoB rendering evaluator requires the Dynamic3DGaussians Gaussian rasterizer.
That rasterizer is not copied here and retains the Inria/MPII research-only
Gaussian-Splatting License.

## Excluded legacy snapshot

The working research tree may contain upstream or adapted code that is
explicitly excluded from the public wheel and ignored by the release manifest:

- Gaussian Splatting rasterization and nearest-neighbor submodules use the
  bundled Inria/MPII research-only Gaussian-Splatting License.
- `skeleton_utils/tree_utils.py` and `skeleton_utils/mst_utils.py` identify
  themselves as copied from RigNet and governed by GPLv3 or a commercial
  license.
- RigGS-derived and Dynamic3DGaussians-derived reconstruction code retains the
  notices and licenses supplied by those upstream projects.

Do not distribute those excluded files under the root MIT license. The public
experiment adapters consume their exported point trajectories through documented
file interfaces instead of bundling their implementations.
