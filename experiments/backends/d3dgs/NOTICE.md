# D3DGS Experiment Extensions

The trainer, dataset converters and metric evaluator were copied from the
author's local Dynamic3DGaussians experiment checkout. They are custom
extensions, not files guaranteed to exist in the official upstream checkout.
Packaging changes preserve the training and evaluation computations.

The underlying [Dynamic3DGaussians](https://github.com/JonathonLuiten/Dynamic3DGaussians)
code is distributed under the included MIT license. Its `helpers.py` and
`external.py` modules remain external engine dependencies. The CUDA Gaussian
rasterizer is not copied here and retains its own research-only license.

DNA-Rendering and ActorsHQ data and SMPL/SMPL-X models must be obtained
separately under their own terms. No checkpoints or model data are included.
