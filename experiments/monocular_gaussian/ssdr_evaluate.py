"""Evaluate a reconstructed RigGS-compatible 4DGS after direct SSDR fitting.

This adapter intentionally stops at the paper's training-free boundary:

    reconstructed Gaussian trajectories -> SSDR/LBS -> novel-view metrics

It does not load, train, or evaluate a RigGS stage-2 skeleton model.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import yaml


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))


def _expand(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if not isinstance(value, str):
        return value
    return os.path.expanduser(os.path.expandvars(value))


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    required = {"riggs_root", "model_path", "iteration", "num_bones", "output_dir"}
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"configuration is missing: {', '.join(missing)}")
    return _expand(config)


def _parse_saved_namespace(path: Path) -> Namespace:
    expression = ast.parse(path.read_text(encoding="utf-8"), mode="eval").body
    if not isinstance(expression, ast.Call) or not isinstance(expression.func, ast.Name):
        raise ValueError(f"unsupported cfg_args format: {path}")
    if expression.func.id != "Namespace" or expression.args:
        raise ValueError(f"unsupported cfg_args format: {path}")
    values = {}
    for keyword in expression.keywords:
        if keyword.arg is None:
            raise ValueError(f"unsupported cfg_args expansion: {path}")
        values[keyword.arg] = ast.literal_eval(keyword.value)
    return Namespace(**values)


def _fid(view: Any) -> float:
    value = view.fid.detach().cpu().reshape(-1)
    if value.numel() != 1:
        raise ValueError(f"camera fid must be scalar, got {tuple(value.shape)}")
    return float(value.item())


def ordered_unique_views(views: Iterable[Any], tolerance: float = 1e-7) -> List[Any]:
    ordered = sorted(views, key=_fid)
    unique: List[Any] = []
    for view in ordered:
        if not unique or abs(_fid(view) - _fid(unique[-1])) > tolerance:
            unique.append(view)
    return unique


def nearest_frame_index(frame_fids: Sequence[float], fid: float) -> int:
    values = np.asarray(frame_fids, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("frame_fids must be a non-empty vector")
    return int(np.abs(values - float(fid)).argmin())


def deformation_attributes_for_render(
    rotation: Any,
    scaling: Any,
    *,
    use_isotropic_gaussians: bool,
) -> Tuple[Any, Any]:
    """Keep temporal scale changes even when the base splats are isotropic."""

    if not use_isotropic_gaussians:
        return rotation, scaling
    if hasattr(rotation, "new_zeros"):
        rotation = rotation.new_zeros(rotation.shape)
    else:
        rotation = np.zeros_like(rotation)
    return rotation, scaling


def select_source_indices(
    opacities: np.ndarray,
    max_points: Optional[int],
    strategy: str = "highest_opacity",
    seed: int = 42,
    require_num_bones: bool = True,
) -> np.ndarray:
    values = np.asarray(opacities, dtype=np.float32).reshape(-1)
    if max_points is None or max_points >= len(values):
        return np.arange(len(values), dtype=np.int64)
    if max_points < 1:
        raise ValueError("ssdr_max_points must be positive")
    if strategy == "highest_opacity":
        selected = np.argpartition(values, -max_points)[-max_points:]
    elif strategy == "random":
        selected = np.random.default_rng(seed).choice(
            len(values), size=max_points, replace=False
        )
    else:
        raise ValueError(f"unsupported SSDR selection strategy: {strategy}")
    return np.sort(selected.astype(np.int64, copy=False))


def transfer_weights_nearest(
    full_rest_pose: np.ndarray,
    source_rest_pose: np.ndarray,
    source_weights: np.ndarray,
) -> np.ndarray:
    full_rest_pose = np.asarray(full_rest_pose, dtype=np.float32)
    source_rest_pose = np.asarray(source_rest_pose, dtype=np.float32)
    weights = np.asarray(source_weights)
    if weights.shape[1] != len(source_rest_pose):
        if weights.shape[0] == len(source_rest_pose):
            weights = weights.T
        else:
            raise ValueError("SSDR weights do not match the source rest pose")
    try:
        from scipy.spatial import cKDTree

        nearest = cKDTree(source_rest_pose).query(full_rest_pose, k=1)[1]
    except (ImportError, ValueError):
        nearest_parts = []
        source_norm = np.einsum("nd,nd->n", source_rest_pose, source_rest_pose)
        for start in range(0, len(full_rest_pose), 256):
            query = full_rest_pose[start : start + 256]
            distances = (
                np.einsum("nd,nd->n", query, query)[:, None]
                + source_norm[None]
                - 2.0 * query @ source_rest_pose.T
            )
            nearest_parts.append(np.argmin(distances, axis=1))
        nearest = np.concatenate(nearest_parts)
    return np.asarray(weights[:, nearest], dtype=np.float32)


def _activate_riggs(riggs_root: Path) -> None:
    root = str(riggs_root.resolve())
    if root in sys.path:
        sys.path.remove(root)
    sys.path.insert(0, root)


def _load_models(riggs_root: Path, model_path: Path, iteration: int):
    _activate_riggs(riggs_root)
    from model_loader import load_base_models

    dataset = _parse_saved_namespace(model_path / "cfg_args")
    dataset.model_path = str(model_path.resolve())
    dataset.source_path = str(Path(dataset.source_path).expanduser().resolve())
    models = load_base_models(dataset, iteration=iteration, shuffle=False)

    # Older RigGS loaders omit this constructor argument. The PLY remains valid,
    # but the flag must be restored so isotropic rendering matches training.
    models.gaussians.use_isotropic_gs = bool(dataset.use_isotropic_gs)
    models.deform.deform.cached_nn_weight = True
    models.deform.deform.eval()
    return dataset, models


def extract_trajectory(models: Any) -> Tuple[np.ndarray, np.ndarray]:
    import torch

    gaussians = models.gaussians
    deform = models.deform
    xyz = gaussians.get_xyz.detach()
    views = ordered_unique_views(models.scene.getTrainCameras())
    frames: List[np.ndarray] = []
    frame_fids: List[float] = []

    with torch.inference_mode():
        for index, view in enumerate(views):
            fid = view.fid
            time_input = (
                fid.unsqueeze(0).expand(xyz.shape[0], -1)
                if deform.name == "mlp"
                else deform.deform.expand_time(fid)
            )
            values = deform.step(
                xyz,
                time_input,
                feature=gaussians.feature,
                motion_mask=gaussians.motion_mask,
                is_training=False,
            )
            frames.append((xyz + values["d_xyz"]).cpu().numpy().astype(np.float32))
            frame_fids.append(_fid(view))
            if (index + 1) % 25 == 0 or index + 1 == len(views):
                print(f"Extracted trajectory frame {index + 1}/{len(views)}")

    return np.stack(frames), np.asarray(frame_fids, dtype=np.float32)


def run_ssdr(
    trajectory: np.ndarray,
    *,
    num_bones: int,
    rest_frame: int,
    max_iterations: int,
    tolerance: float,
    patience: int,
    max_influences: int,
    protocol: str = "motion_pointcloud",
    initialization: str = "auto",
    initialization_frames: int = 40,
    initialization_components: int = 16,
    initialization_spatial_weight: float = 0.35,
    seed: int = 42,
    require_num_bones: bool = True,
    backend: Optional[Any] = None,
) -> Mapping[str, Any]:
    from skelebone.extraction import SSDRConfig, decompose_ssdr
    from skelebone.types import DynamicPointSequence

    result = decompose_ssdr(
        DynamicPointSequence(trajectory),
        SSDRConfig(
            num_bones=num_bones,
            rest_frame=rest_frame,
            tolerance=tolerance,
            patience=patience,
            max_iterations=max_iterations,
            max_influences=max_influences,
            protocol=protocol,
            initialization=initialization,
            initialization_frames=initialization_frames,
            initialization_components=initialization_components,
            initialization_spatial_weight=initialization_spatial_weight,
            random_seed=seed,
            require_num_bones=require_num_bones,
        ),
        backend=backend,
    )
    return {
        "weights": result.weights,
        "transformations": result.transformations,
        "bone_positions": result.bone_positions,
        "reconstruction": result.reconstruction,
        "rest_pose": result.rest_pose,
        "rmse": result.rmse,
        "num_bones": result.num_bones,
        "protocol": result.protocol,
        "requested_bones": result.requested_bones,
        "initialized_bones": result.initialized_bones,
        "effective_bones": result.effective_bones,
        "hard_bone_population": result.hard_bone_population,
        "soft_bone_mass": result.soft_bone_mass,
        "backend_name": result.backend_name,
        "backend_module": result.backend_module,
        "backend_path": result.backend_path or "",
        "backend_sha256": result.backend_sha256 or "",
        "rest_frame": rest_frame,
    }


def _json_number(value: Any) -> float:
    if hasattr(value, "detach"):
        value = value.detach().cpu().item()
    return float(value)


def render_and_evaluate(
    models: Any,
    ssdr: Mapping[str, Any],
    frame_fids: np.ndarray,
    output_dir: Path,
    source_indices: np.ndarray,
) -> Dict[str, Any]:
    import torch
    import torchvision
    from gaussian_renderer import render
    from utils.image_utils import lpips, psnr, ssim

    render_dir = output_dir / "renders"
    gt_dir = output_dir / "gt"
    render_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    pipeline = SimpleNamespace(
        compute_cov3D_python=False,
        convert_SHs_python=True,
        debug=False,
    )
    gaussian = models.gaussians
    deform = models.deform
    xyz = gaussian.get_xyz.detach()
    train_views = ordered_unique_views(models.scene.getTrainCameras())
    rest_frame = int(np.asarray(ssdr.get("rest_frame", 0)).item())
    rest_view = train_views[
        nearest_frame_index([_fid(view) for view in train_views], frame_fids[rest_frame])
    ]
    rest_time = (
        rest_view.fid.unsqueeze(0).expand(xyz.shape[0], -1)
        if deform.name == "mlp"
        else deform.deform.expand_time(rest_view.fid)
    )
    with torch.inference_mode():
        rest_values = deform.step(
            xyz,
            rest_time,
            feature=gaussian.feature,
            motion_mask=gaussian.motion_mask,
            is_training=False,
        )
    full_rest = (xyz + rest_values["d_xyz"]).detach().cpu().numpy()
    source_rest = np.asarray(ssdr["rest_pose"], dtype=np.float32)
    source_error = np.max(
        np.abs(full_rest[source_indices] - source_rest)
    )
    if source_error > 1e-4:
        raise ValueError(
            f"SSDR source/rest mismatch: max error {source_error:.6g}"
        )
    full_weights = transfer_weights_nearest(
        full_rest,
        source_rest,
        np.asarray(ssdr["weights"]),
    )
    weights_torch = torch.from_numpy(full_weights).to(xyz.device)
    rest_h = torch.cat(
        [
            torch.from_numpy(full_rest).to(device=xyz.device, dtype=xyz.dtype),
            torch.ones((len(full_rest), 1), device=xyz.device, dtype=xyz.dtype),
        ],
        dim=1,
    )
    transforms_torch = torch.from_numpy(
        np.asarray(ssdr["transformations"], dtype=np.float32)
    ).to(xyz.device)
    metric_values = {"psnr": [], "ssim": [], "lpips": []}
    frame_records = []
    views = ordered_unique_views(models.scene.getTestCameras())

    with torch.inference_mode():
        for index, view in enumerate(views):
            fid = _fid(view)
            frame_index = nearest_frame_index(frame_fids, fid)
            time_input = (
                view.fid.unsqueeze(0).expand(xyz.shape[0], -1)
                if deform.name == "mlp"
                else deform.deform.expand_time(view.fid)
            )
            values = deform.step(
                xyz,
                time_input,
                feature=gaussian.feature,
                motion_mask=gaussian.motion_mask,
                is_training=False,
            )
            transformed = torch.einsum(
                "bij,nj->bni", transforms_torch[frame_index], rest_h
            )
            ssdr_xyz = torch.einsum(
                "bn,bni->ni", weights_torch, transformed[..., :3]
            )
            d_xyz = ssdr_xyz - xyz
            d_rotation, d_scaling = deformation_attributes_for_render(
                values["d_rotation"],
                values["d_scaling"],
                use_isotropic_gaussians=gaussian.use_isotropic_gs,
            )

            result = render(
                view,
                gaussian,
                pipeline,
                models.background,
                d_xyz,
                d_rotation,
                d_scaling,
                d_opacity=values["d_opacity"],
                d_color=values["d_color"],
                d_rot_as_res=deform.d_rot_as_res,
            )
            image = torch.clamp(result["render"], 0.0, 1.0)
            gt = torch.clamp(view.original_image[:3].to(image.device), 0.0, 1.0)
            per_frame = {
                "frame": index,
                "fid": fid,
                "ssdr_frame": frame_index,
                "psnr": _json_number(psnr(image[None], gt[None]).mean()),
                "ssim": _json_number(ssim(image[None], gt[None], data_range=1.0).mean()),
                "lpips": _json_number(lpips(image[None], gt[None]).mean()),
            }
            for metric in metric_values:
                metric_values[metric].append(per_frame[metric])
            frame_records.append(per_frame)
            torchvision.utils.save_image(image, render_dir / f"{index:05d}.png")
            torchvision.utils.save_image(gt, gt_dir / f"{index:05d}.png")
            print(
                f"Rendered test frame {index + 1}/{len(views)} "
                f"PSNR={per_frame['psnr']:.4f}"
            )

    return {
        "protocol": "reconstruction_to_ssdr_lbs",
        "test_frames": len(views),
        "mean": {
            name: float(np.mean(values)) for name, values in metric_values.items()
        },
        "frames": frame_records,
    }


def fit_trajectory(
    config: Mapping[str, Any],
    trajectory: np.ndarray,
    frame_fids: np.ndarray,
    opacities: np.ndarray,
    output_dir: Path,
) -> Tuple[Path, Mapping[str, Any], np.ndarray]:
    max_points = config.get("ssdr_max_points")
    source_indices = select_source_indices(
        opacities,
        None if max_points is None else int(max_points),
        strategy=str(config.get("ssdr_selection", "highest_opacity")),
        seed=int(config.get("seed", 42)),
        require_num_bones=bool(config.get("ssdr_require_num_bones", True)),
    )
    print(f"SSDR points: {len(source_indices)}/{trajectory.shape[1]}")
    ssdr = run_ssdr(
        trajectory[:, source_indices],
        num_bones=int(config["num_bones"]),
        rest_frame=int(config.get("rest_frame", 0)),
        max_iterations=int(config.get("ssdr_max_iterations", 100)),
        tolerance=float(config.get("ssdr_tolerance", 1e-3)),
        patience=int(config.get("ssdr_patience", 3)),
        max_influences=int(config.get("ssdr_max_influences", 4)),
        protocol=str(config.get("ssdr_protocol", "motion_pointcloud")),
        initialization=str(config.get("ssdr_initialization", "auto")),
        initialization_frames=int(config.get("ssdr_initialization_frames", 40)),
        initialization_components=int(config.get("ssdr_initialization_components", 16)),
        initialization_spatial_weight=float(
            config.get("ssdr_initialization_spatial_weight", 0.35)
        ),
        seed=int(config.get("seed", 42)),
        require_num_bones=bool(config.get("ssdr_require_num_bones", True)),
    )
    ssdr_path = output_dir / f"ssdr_results_bones{int(ssdr['num_bones'])}.npz"
    np.savez_compressed(
        ssdr_path,
        frame_fids=frame_fids,
        source_indices=source_indices,
        trajectory_shape=np.asarray(trajectory.shape, dtype=np.int64),
        **ssdr,
    )
    return ssdr_path, ssdr, source_indices


def evaluate(
    config: Mapping[str, Any],
    *,
    stage: str = "all",
    ssdr_result: Optional[Path] = None,
    trajectory_cache: Optional[Path] = None,
) -> Path:
    riggs_root = Path(config["riggs_root"])
    model_path = Path(config["model_path"])
    output_dir = Path(config["output_dir"])
    iteration = int(config["iteration"])
    output_dir.mkdir(parents=True, exist_ok=True)

    if stage == "fit":
        if trajectory_cache is None:
            raise ValueError("--trajectory-cache is required for the fit stage")
        cached = np.load(trajectory_cache, allow_pickle=False)
        trajectory = np.asarray(cached["trajectory"], dtype=np.float32)
        frame_fids = np.asarray(cached["frame_fids"], dtype=np.float32)
        opacities = np.asarray(
            cached["opacity"] if "opacity" in cached else np.ones(trajectory.shape[1])
        ).reshape(-1)
        ssdr_path, _, _ = fit_trajectory(
            config, trajectory, frame_fids, opacities, output_dir
        )
        return ssdr_path

    _, models = _load_models(riggs_root, model_path, iteration)
    trajectory_shape = None
    if stage in {"all", "ssdr", "extract"}:
        trajectory, frame_fids = extract_trajectory(models)
        trajectory_shape = list(trajectory.shape)
        print(f"Trajectory: {trajectory.shape}")

        opacities = models.gaussians.get_opacity.detach().cpu().numpy().reshape(-1)
        if stage == "extract":
            cache_path = output_dir / "trajectory.npz"
            np.savez_compressed(
                cache_path,
                trajectory=trajectory,
                frame_fids=frame_fids,
                opacity=opacities,
            )
            return cache_path
        ssdr_path, ssdr, source_indices = fit_trajectory(
            config, trajectory, frame_fids, opacities, output_dir
        )
        if stage == "ssdr":
            return ssdr_path
    else:
        if ssdr_result is None:
            raise ValueError("--ssdr-result is required for the render stage")
        ssdr_path = Path(ssdr_result)
        data = np.load(ssdr_path, allow_pickle=False)
        ssdr = {key: data[key] for key in data.files}
        frame_fids = np.asarray(ssdr["frame_fids"])
        source_indices = np.asarray(ssdr["source_indices"], dtype=np.int64)
        if "trajectory_shape" in ssdr:
            trajectory_shape = np.asarray(ssdr["trajectory_shape"]).tolist()
        else:
            trajectory_shape = [
                len(frame_fids),
                len(models.gaussians.get_xyz),
                3,
            ]

    metrics = render_and_evaluate(
        models,
        ssdr,
        frame_fids,
        output_dir,
        source_indices,
    )
    metrics.update(
        {
            "model_path": str(model_path.resolve()),
            "iteration": iteration,
            "trajectory_shape": trajectory_shape,
            "ssdr_point_count": len(source_indices),
            "ssdr_selection": str(
                config.get("ssdr_selection", "highest_opacity")
            ),
            "ssdr_bones": int(ssdr["num_bones"]),
            "ssdr_protocol": str(np.asarray(ssdr.get("protocol", "unknown")).item()),
            "ssdr_requested_bones": int(
                np.asarray(ssdr.get("requested_bones", ssdr["num_bones"])).item()
            ),
            "ssdr_initialized_bones": int(
                np.asarray(ssdr.get("initialized_bones", ssdr["num_bones"])).item()
            ),
            "ssdr_effective_bones": int(
                np.asarray(ssdr.get("effective_bones", ssdr["num_bones"])).item()
            ),
            "ssdr_backend": {
                "class": str(np.asarray(ssdr.get("backend_name", "unknown")).item()),
                "module": str(np.asarray(ssdr.get("backend_module", "unknown")).item()),
                "path": str(np.asarray(ssdr.get("backend_path", "")).item()),
                "sha256": str(np.asarray(ssdr.get("backend_sha256", "")).item()),
            },
            "ssdr_rmse": float(ssdr["rmse"]),
            "ssdr_result": str(ssdr_path.resolve()),
        }
    )
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics["mean"], indent=2))
    return metrics_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=["all", "extract", "fit", "ssdr", "render"], default="all"
    )
    parser.add_argument("--ssdr-result", type=Path)
    parser.add_argument("--trajectory-cache", type=Path)
    args = parser.parse_args()
    print(
        evaluate(
            load_config(args.config),
            stage=args.stage,
            ssdr_result=args.ssdr_result,
            trajectory_cache=args.trajectory_cache,
        )
    )


if __name__ == "__main__":
    main()
