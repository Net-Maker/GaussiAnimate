"""Load stage-one models through the bundled monocular engine API.

Source: https://github.com/yaoyx689/RigGS
Inherited reconstruction: https://github.com/CVMI-Lab/SC-GS
CUDA extensions are installed separately; no private model loader is needed.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any


def bundled_engine_root():
    return Path(__file__).resolve().parents[1] / "backends" / "monocular"


@dataclass
class Reconstruction:
    gaussians: Any
    scene: Any
    deform: Any
    background: Any


def load_base_models(dataset, iteration=-1, shuffle=False):
    import torch
    from gaussian_renderer import GaussianModel
    from scene import DeformModel, Scene

    deformation_fields = (
        "K", "deform_type", "is_blender", "skinning", "hyper_dim",
        "node_num", "pred_opacity", "pred_color", "use_hash", "hash_time",
        "d_rot_as_res", "local_frame", "progressive_brand_time", "max_d_scale",
    )
    options = {name: getattr(dataset, name) for name in deformation_fields}
    deform = DeformModel(**options)
    deform.load_weights(dataset.model_path, iteration=iteration)
    feature_dimension = (
        deform.deform.node_num
        if dataset.skinning and deform.name == "node"
        else dataset.hyper_dim
    )
    gaussians = GaussianModel(
        dataset.sh_degree,
        fea_dim=feature_dimension,
        with_motion_mask=dataset.gs_with_motion_mask,
    )
    gaussians.use_isotropic_gs = bool(dataset.use_isotropic_gs)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=shuffle)
    background = torch.tensor(
        [1, 1, 1] if dataset.white_background else [0, 0, 0],
        dtype=torch.float32,
        device="cuda",
    )
    return Reconstruction(gaussians, scene, deform, background)
