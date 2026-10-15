"""Load stage-one models through the bundled monocular engine API.

Source: https://github.com/yaoyx689/RigGS
Inherited reconstruction: https://github.com/CVMI-Lab/SC-GS
CUDA extensions are installed separately; no private model loader is needed.
"""

from dataclasses import dataclass
from argparse import ArgumentParser, Namespace
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


def complete_model_config(dataset):
    """Restore omitted engine defaults without overwriting saved settings."""
    from arguments import ModelParams

    parser = ArgumentParser(add_help=False)
    ModelParams(parser)
    values = vars(parser.parse_args([]))
    saved = vars(dataset)
    if "deform_type" not in saved and "is_6dof" in saved:
        if saved["is_6dof"]:
            raise ValueError("legacy 6-DoF checkpoints require their original exporter")
        # The original D-NeRF MLP predates node deformation and feature vectors.
        values.update(deform_type="mlp", hyper_dim=0)
    values.update(saved)
    return Namespace(**values)


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
    if not deform.load_weights(dataset.model_path, iteration=iteration):
        raise FileNotFoundError(f"no deformation checkpoint at iteration {iteration}: {dataset.model_path}")
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
