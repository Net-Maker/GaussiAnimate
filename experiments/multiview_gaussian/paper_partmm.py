"""Adapter for the exact PartMM research implementation used by the paper."""

import argparse
import hashlib
import json
import sys
from dataclasses import fields, replace
from pathlib import Path
from typing import Any, Dict

import numpy as np

from experiments.methods import (
    HeldOutPointProtocol,
    SSDRCache,
    point_error_metrics,
    save_prediction_diagnostic,
    write_prediction_artifacts,
)
from partmm import PartMMConfig
from .human_partmm import run_human_partmm


_PAPER_SPEC_PARTMM_CONFIG = PartMMConfig(
    num_parts=5,
    patch_size=7,
    num_neighbors=7,
    pyramid_levels=5,
    pyramid_blend=0.7,
    overlap_rings=1,
    align_parts=True,
    alignment_mode="edge_wise",
    partition_mode="humanoid_five",
    bone_overlap_ratio=1.5,
)


def _source_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted((root / "smpl_to_ssdr").rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _smpl_joints(poses: np.ndarray, model_dir: Path) -> np.ndarray:
    import smplx
    import torch

    model = smplx.SMPL(
        str(model_dir),
        gender="neutral",
        batch_size=1,
    )
    joints = []
    for start in range(0, len(poses), 64):
        batch = torch.from_numpy(poses[start : start + 64]).float()
        output = model(
            global_orient=batch[:, :3],
            body_pose=batch[:, 3:],
            return_verts=False,
        )
        joints.append(output.joints[:, :24].detach().cpu().numpy())
    return np.concatenate(joints)


def run_paper_partmm(
    params_path: Path,
    cache_path: Path,
    motion_path: Path,
    output_dir: Path,
    *,
    implementation_root: Path,
    motion2motion_root: Path,
    smpl_model_dir: Path,
    train_ratio: float = 0.8,
    device: str = "cpu",
) -> dict:
    sys.path.insert(0, str(Path(motion2motion_root).resolve()))
    sys.path.insert(0, str(Path(implementation_root).resolve()))
    import torch
    from smpl_to_ssdr.core import SMPLMotionData, SSDRMotionData
    from smpl_to_ssdr.core.part_matcher import PartBasedSMPLToSSDRMatcher
    from smpl_to_ssdr.reconstruction import (
        reconstruct_vertices,
        result_to_lbs_transforms,
    )

    with np.load(params_path, allow_pickle=False) as params:
        ground_truth = np.asarray(params["means3D"], dtype=np.float32)
    protocol = HeldOutPointProtocol.from_shape(
        ground_truth.shape,
        train_ratio,
    )
    frames = protocol.frames
    points = protocol.points
    training_frames = protocol.training_frames
    cache = SSDRCache.load(cache_path)
    fitted = cache.training_frames
    if fitted != training_frames:
        raise ValueError(
            f"SSDR cache contains {fitted} frames, "
            f"expected {training_frames}"
        )
    rest_pose = cache.rest_pose.astype(np.float32)
    weights = cache.weights.T.astype(np.float32)
    transforms = cache.transformations.astype(np.float32)
    frame_means = cache.frame_means.astype(np.float32)
    if weights.shape != (transforms.shape[1], points):
        raise ValueError(
            "SSDR weights and transforms do not share the point set"
        )
    with np.load(motion_path, allow_pickle=False) as motion:
        poses = np.asarray(motion["axis_angle"], dtype=np.float32).reshape(frames, -1)
        translations = np.asarray(motion["root_translation"], dtype=np.float32)
        case = str(motion["case"]) if "case" in motion else None
        dataset = str(motion["dataset"]) if "dataset" in motion else None

    source_smpl = SMPLMotionData(
        {
            "poses": torch.from_numpy(poses[:training_frames]),
            "trans": torch.from_numpy(translations[:training_frames]),
        }
    )
    source_ssdr = SSDRMotionData(
        {
            "transforms": torch.from_numpy(
                transforms.reshape(training_frames, -1)
            ),
            "trans": torch.from_numpy(frame_means),
        }
    )
    matcher = PartBasedSMPLToSSDRMatcher(
        patch_size=7,
        alpha=0.85,
        matching_mode="full",
        device=device,
        skeleton_guided=True,
        skeleton_transform_weight=1.0,
        skeleton_mode="edge_wise",
        skeleton_debug=False,
        gender="neutral",
        is_cloth3d=False,
        use_motion_binding=True,
        motion_weight=0.5,
        distance_weight=0.5,
    )
    matcher.build_part_binding(source_smpl, source_ssdr, rest_pose, weights)
    matcher.build_part_databases(source_smpl, source_ssdr)

    novel_poses = poses[training_frames:]
    novel_smpl = SMPLMotionData(
        {
            "poses": torch.from_numpy(novel_poses),
            "trans": torch.from_numpy(translations[training_frames:]),
        }
    )
    database_joints = _smpl_joints(
        poses[:training_frames],
        smpl_model_dir,
    )
    novel_joints = _smpl_joints(novel_poses, smpl_model_dir)
    result_ssdr = matcher.retarget_motion(
        novel_smpl,
        num_steps=2,
        matching_alpha=0.3,
        coarse_ratio=1.0,
        pyr_factor=0.75,
        novel_joint_positions=torch.from_numpy(novel_joints.astype(np.float32)),
        db_joint_positions=torch.from_numpy(database_joints.astype(np.float32)),
        novel_poses=novel_poses,
        db_poses=poses[:training_frames],
    )
    lbs_reference = {
        "weights": weights,
        "rest_pose": rest_pose,
    }
    lbs_data = result_to_lbs_transforms(result_ssdr, lbs_reference)
    predicted_centered = reconstruct_vertices(lbs_data).astype(np.float32)
    predicted_translation = (
        result_ssdr[0, -3:, :].permute(1, 0).detach().cpu().numpy()
    )
    predicted = predicted_centered + predicted_translation[:, None]
    target = ground_truth[training_frames:]
    metrics = point_error_metrics(predicted, target)

    output_dir = Path(output_dir)
    artifacts = write_prediction_artifacts(
        output_dir,
        protocol,
        test_prediction=predicted,
        training_prefix=ground_truth[:training_frames],
        bone_transforms=lbs_data["transformations"],
    )
    save_prediction_diagnostic(
        output_dir / "diagnostic.png",
        ground_truth[0],
        predicted[0],
        target[0],
    )
    artifacts["diagnostic"] = "diagnostic.png"
    metadata = {
        "schema_version": 1,
        "algorithm": "paper_partmm",
        "case": case,
        "dataset": dataset,
        **protocol.metadata(),
        "warped_sequence_training_payload": (
            "ground_truth_context_not_partmm_predictions"
        ),
        "ssdr_fitted_frames": fitted,
        "implementation": {
            "root": str(Path(implementation_root).resolve()),
            "python_source_sha256": _source_hash(Path(implementation_root)),
            "revision": "uncommitted_local_research_snapshot",
        },
        "partmm": {
            "patch_size": 7,
            "alpha": 0.85,
            "matching_mode": "full",
            "skeleton_guided": True,
            "skeleton_transform_weight": 1.0,
            "alignment_mode": "edge_wise",
            "use_motion_binding": True,
            "motion_weight": 0.5,
            "distance_weight": 0.5,
            "num_steps": 2,
            "matching_alpha": 0.3,
            "coarse_ratio": 1.0,
            "pyramid_factor": 0.75,
            "device": device,
        },
        "metrics": metrics,
        "artifacts": artifacts,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def _select_experiment_method(
    config: Dict[str, Any],
    method_id: str,
    variant: str,
) -> Dict[str, Any]:
    """Pick the unique experiment method matching the CLI selection."""

    matches = [
        method
        for method in config.get("methods", [])
        if method.get("id") == method_id
    ]
    if not matches:
        raise ValueError(
            f"experiment config has no method with id {method_id!r}"
        )
    if len(matches) > 1:
        raise ValueError(
            f"experiment config has {len(matches)} methods with "
            f"id {method_id!r}"
        )
    method = matches[0]
    if method.get("variant") != variant:
        raise ValueError(
            f"method {method_id!r} has variant {method.get('variant')!r}, "
            f"but the CLI requested variant {variant!r}"
        )
    return method


def _load_experiment_method(
    config_path: Path,
    method_id: str,
    variant: str,
) -> Dict[str, Any]:
    # Imported lazily: the engine imports multiview_gaussian stages, so a
    # module-level import would create a cycle.
    from experiments.engine import load_paper_config

    config = load_paper_config(config_path)
    return _select_experiment_method(config, method_id, variant)


def _paper_spec_config_from_protocol(protocol: Dict[str, Any]) -> PartMMConfig:
    """Override paper-spec defaults with PartMM protocol fields.

    Protocol keys outside PartMMConfig (train_ratio, ids, status) are
    ignored; omitted fields keep the published paper defaults.
    """

    overrides = {
        field.name: protocol[field.name]
        for field in fields(PartMMConfig)
        if field.name in protocol
    }
    return replace(_PAPER_SPEC_PARTMM_CONFIG, **overrides)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=("paper_spec", "historical"),
        required=True,
    )
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--ssdr-cache", type=Path, required=True)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--implementation-root", type=Path)
    parser.add_argument("--motion2motion-root", type=Path)
    parser.add_argument("--smpl-model-dir", type=Path)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--experiment-config", type=Path)
    parser.add_argument("--method-id")
    args = parser.parse_args()
    if args.variant == "historical" and None in (
        args.implementation_root,
        args.motion2motion_root,
        args.smpl_model_dir,
    ):
        parser.error(
            "--implementation-root, --motion2motion-root, and "
            "--smpl-model-dir are required for the historical variant"
        )
    if args.method_id and args.experiment_config is None:
        parser.error("--method-id requires --experiment-config")
    train_ratio = args.train_ratio
    partmm_config = _PAPER_SPEC_PARTMM_CONFIG
    if args.experiment_config is not None:
        if not args.method_id:
            parser.error("--method-id is required with --experiment-config")
        try:
            method = _load_experiment_method(
                args.experiment_config,
                args.method_id,
                args.variant,
            )
            if args.variant == "paper_spec":
                protocol = method.get("protocol")
                if (
                    not isinstance(protocol, dict)
                    or "train_ratio" not in protocol
                ):
                    raise ValueError(
                        f"method {args.method_id!r} protocol must "
                        "define train_ratio"
                    )
                train_ratio = float(protocol["train_ratio"])
                partmm_config = _paper_spec_config_from_protocol(protocol)
            # The historical variant keeps its own matcher and algorithm;
            # the experiment config only validated the method identity.
        except ValueError as error:
            parser.error(str(error))
    if args.variant == "paper_spec":
        result = run_human_partmm(
            args.params,
            args.ssdr_cache,
            args.motion,
            args.output_dir,
            train_ratio=train_ratio,
            config=partmm_config,
        )
        result["algorithm"] = "paper_spec_partmm"
        metadata_path = args.output_dir / "metadata.json"
        metadata_path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    else:
        result = run_paper_partmm(
            args.params,
            args.ssdr_cache,
            args.motion,
            args.output_dir,
            implementation_root=args.implementation_root,
            motion2motion_root=args.motion2motion_root,
            smpl_model_dir=args.smpl_model_dir,
            train_ratio=train_ratio,
            device=args.device,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
