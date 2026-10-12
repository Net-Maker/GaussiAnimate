"""Canonical validators for multiview release artifacts.

This module validates persisted D3DGS checkpoints and derived experiment
outputs. It is intentionally separate from ``experiments.methods.artifacts``,
which writes method results, and from the typed SSDR runtime representation.
Both the release pipeline and resource audit consume these validators so that
an artifact has one readiness definition.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np

from experiments.reproductions.dressrecon_bob.pose_contract import (
    BOB_ADAPTER_ID,
    BOB_EVALUATION_ATTRIBUTE_SOURCE,
    BOB_ORACLE_EVALUATION_ATTRIBUTES,
    BOB_PREDICTED_GAUSSIAN_ATTRIBUTES,
    SMPL_JOINT_MAPPING_ID,
    SMPL_ROOT_HANDLING,
    SMPL_ROOT_JOINT_ID,
    SMPL_ROOT_TRANSLATION_HANDLING,
    SMPL_ROOTLESS_JOINTS,
    SMPL_ROOTLESS_POSE_DIM,
)


PARAM_KEYS = (
    "means3D",
    "rgb_colors",
    "unnorm_rotations",
    "log_scales",
    "logit_opacities",
    "seg_colors",
    "cam_m",
    "cam_c",
)

METHOD_EXTRA_FILES = {
    "lbs": (
        "frame0_surface.off",
        "robust_weights.npz",
        "diagnostic.png",
    ),
    "bob": (
        "bob_model.pt",
        "loss_history.npy",
        "skinning_weights.npy",
        "skeleton_skinning_weights.npy",
        "diagnostic.png",
    ),
    "partmm": (
        "bone_transforms.npy",
        "diagnostic.png",
    ),
}

PARTMM_ALGORITHM = "paper_spec_partmm"
PARTMM_PROTOCOL_KEYS = (
    "num_parts",
    "patch_size",
    "num_neighbors",
    "pyramid_levels",
    "pyramid_blend",
    "overlap_rings",
    "align_parts",
    "alignment_mode",
    "partition_mode",
    "bone_overlap_ratio",
)


def _file(path: Path, minimum_bytes: int = 1) -> Dict[str, Any]:
    exists = path.is_file()
    size = path.stat().st_size if exists else 0
    return {
        "ready": exists and size >= minimum_bytes,
        "bytes": size,
    }


def _finite(array: np.ndarray, full: bool) -> bool:
    if array.size == 0:
        return False
    if full or array.ndim == 0:
        return bool(np.isfinite(array).all())
    step = max(1, len(array) // 5)
    return bool(np.isfinite(array[::step]).all())


def validate_d3dgs_params(
    path: Path,
    expected_frames: Optional[int],
    *,
    expected_scale_dims: Optional[int] = None,
    expected_cameras: Optional[int] = None,
    full: bool = False,
) -> Dict[str, Any]:
    """Validate the checkpoint schema used by every multiview method."""

    path = Path(path)
    record: Dict[str, Any] = {
        "path": str(path),
        "validation_level": "full" if full else "fast",
        **_file(path),
    }
    if not record["ready"]:
        return record
    try:
        with np.load(path, allow_pickle=False) as archive:
            missing = [
                key for key in PARAM_KEYS if key not in archive.files
            ]
            arrays = {
                key: archive[key]
                for key in PARAM_KEYS
                if key in archive.files
            }
        shapes = {
            key: list(value.shape)
            for key, value in arrays.items()
        }
        dtypes = {
            key: str(value.dtype)
            for key, value in arrays.items()
        }
        means_shape = shapes.get("means3D", [])
        frame_count = means_shape[0] if len(means_shape) == 3 else 0
        point_count = means_shape[1] if len(means_shape) == 3 else 0
        scale_dims = (
            shapes.get("log_scales", [None, None])[1]
            if len(shapes.get("log_scales", [])) == 2
            else None
        )
        shape_valid = bool(
            not missing
            and means_shape == [frame_count, point_count, 3]
            and frame_count > 0
            and (
                expected_frames is None
                or frame_count == expected_frames
            )
            and point_count > 0
            and shapes.get("rgb_colors")
            == [frame_count, point_count, 3]
            and shapes.get("unnorm_rotations")
            == [frame_count, point_count, 4]
            and shapes.get("seg_colors") == [point_count, 3]
            and shapes.get("logit_opacities") == [point_count, 1]
            and scale_dims in (1, 3)
            and shapes.get("log_scales") == [point_count, scale_dims]
            and len(shapes.get("cam_m", [])) == 2
            and shapes.get("cam_m", [None, None])[1] == 3
            and shapes.get("cam_c") == shapes.get("cam_m")
            and (
                expected_scale_dims is None
                or scale_dims == expected_scale_dims
            )
            and (
                expected_cameras is None
                or shapes.get("cam_m", [None])[0] == expected_cameras
            )
        )
        floating = bool(
            arrays
            and all(
                np.issubdtype(value.dtype, np.floating)
                for value in arrays.values()
            )
        )
        finite = bool(
            arrays
            and all(_finite(value, full) for value in arrays.values())
        )
        rotations = arrays.get("unnorm_rotations")
        if rotations is None:
            rotations_nonzero = False
        else:
            rotation_sample = (
                rotations
                if full
                else rotations[::max(1, len(rotations) // 5)]
            )
            rotations_nonzero = bool(
                (np.linalg.norm(rotation_sample, axis=-1) > 1e-8).all()
            )
        record.update(
            readable=True,
            missing_keys=missing,
            array_shapes=shapes,
            array_dtypes=dtypes,
            means3d_shape=means_shape,
            frame_count=frame_count,
            point_count=point_count,
            scale_dims=scale_dims,
            shape_valid=shape_valid,
            floating=floating,
            finite=finite,
            finite_sample=finite,
            rotations_nonzero=rotations_nonzero,
        )
        record["ready"] = bool(
            shape_valid
            and floating
            and finite
            and rotations_nonzero
        )
    except Exception as exc:
        record.update(
            ready=False,
            readable=False,
            error=repr(exc),
        )
    return record


def _load_d3dgs_training_state(path: Path) -> Tuple[Any, str, Any]:
    """Load a trusted local trainer checkpoint across PyTorch versions."""

    import torch

    try:
        state = torch.load(
            str(path),
            map_location="cpu",
            weights_only=False,
        )
        return state, "pickle", torch
    except TypeError:
        state = torch.load(str(path), map_location="cpu")
        return state, "legacy_pickle", torch


def _tensor_finite(torch_module: Any, value: Any, full: bool) -> bool:
    flattened = value.detach().reshape(-1)
    if flattened.numel() == 0:
        return False
    if not full:
        step = max(1, int(flattened.numel()) // 5)
        flattened = flattened[::step]
    return bool(torch_module.isfinite(flattened).all().item())


def validate_d3dgs_resume_checkpoint(
    directory: Path,
    expected_frames: int,
    *,
    expected_scale_dims: Optional[int] = None,
    expected_cameras: Optional[int] = None,
    full: bool = False,
) -> Dict[str, Any]:
    """Validate the two-file rolling checkpoint as one resume artifact."""

    directory = Path(directory)
    history_path = directory / "params_ckpt_latest.npz"
    state_path = directory / "training_state_latest.pt"
    history = validate_d3dgs_params(
        history_path,
        None,
        expected_scale_dims=expected_scale_dims,
        expected_cameras=expected_cameras,
        full=full,
    )
    history_frames = int(history.get("frame_count", 0))
    history_range_valid = bool(
        0 < history_frames <= int(expected_frames)
    )
    state_record: Dict[str, Any] = {
        "path": str(state_path),
        **_file(state_path),
    }
    if state_record["ready"]:
        try:
            state, load_mode, torch_module = _load_d3dgs_training_state(
                state_path
            )
            required_state_keys = (
                "schema_version",
                "completed_timestep",
                "params",
                "variables",
                "optimizer",
            )
            missing_state_keys = [
                key
                for key in required_state_keys
                if not isinstance(state, dict) or key not in state
            ]
            completed_timestep = (
                state.get("completed_timestep")
                if isinstance(state, dict)
                else None
            )
            params = (
                state.get("params")
                if isinstance(state, dict)
                else None
            )
            variables = (
                state.get("variables")
                if isinstance(state, dict)
                else None
            )
            optimizer = (
                state.get("optimizer")
                if isinstance(state, dict)
                else None
            )
            optimizer_state = (
                optimizer.get("state")
                if isinstance(optimizer, dict)
                else None
            )
            optimizer_groups = (
                optimizer.get("param_groups")
                if isinstance(optimizer, dict)
                else None
            )
            schema_valid = bool(
                not missing_state_keys
                and state.get("schema_version") == 1
                and isinstance(completed_timestep, int)
                and not isinstance(completed_timestep, bool)
                and completed_timestep >= 0
                and isinstance(params, dict)
                and isinstance(variables, dict)
                and isinstance(optimizer_state, dict)
                and isinstance(optimizer_groups, list)
            )
            missing_params = [
                key
                for key in PARAM_KEYS
                if not isinstance(params, dict) or key not in params
            ]
            param_items = (
                params.items()
                if isinstance(params, dict)
                else ()
            )
            param_shapes = {
                key: [int(dimension) for dimension in value.shape]
                for key, value in param_items
                if torch_module.is_tensor(value)
            }
            tensor_params = bool(
                isinstance(params, dict)
                and not missing_params
                and all(
                    torch_module.is_tensor(params[key])
                    for key in PARAM_KEYS
                )
            )
            point_count = int(history.get("point_count", 0))
            scale_dims = history.get("scale_dims")
            camera_shape = (
                history.get("array_shapes", {}).get("cam_m", [])
            )
            camera_count = (
                camera_shape[0]
                if len(camera_shape) == 2
                else 0
            )
            expected_param_shapes = {
                "means3D": [point_count, 3],
                "rgb_colors": [point_count, 3],
                "unnorm_rotations": [point_count, 4],
                "log_scales": [point_count, scale_dims],
                "logit_opacities": [point_count, 1],
                "seg_colors": [point_count, 3],
                "cam_m": [camera_count, 3],
                "cam_c": [camera_count, 3],
            }
            shape_valid = bool(
                tensor_params
                and history["ready"]
                and all(
                    param_shapes.get(key) == shape
                    for key, shape in expected_param_shapes.items()
                )
            )
            floating = bool(
                tensor_params
                and all(
                    torch_module.is_floating_point(params[key])
                    for key in PARAM_KEYS
                )
            )
            finite = bool(
                floating
                and all(
                    _tensor_finite(
                        torch_module,
                        params[key],
                        full,
                    )
                    for key in PARAM_KEYS
                )
            )
            state_record.update(
                readable=True,
                load_mode=load_mode,
                schema_version=(
                    state.get("schema_version")
                    if isinstance(state, dict)
                    else None
                ),
                completed_timestep=completed_timestep,
                missing_keys=missing_state_keys,
                schema_valid=schema_valid,
                missing_param_keys=missing_params,
                param_shapes=param_shapes,
                expected_param_shapes=expected_param_shapes,
                tensor_params=tensor_params,
                shape_valid=shape_valid,
                floating=floating,
                finite=finite,
            )
            state_record["ready"] = bool(
                schema_valid
                and shape_valid
                and floating
                and finite
            )
        except Exception as exc:
            state_record.update(
                ready=False,
                readable=False,
                error=repr(exc),
            )
    completed_timestep = state_record.get("completed_timestep")
    timesteps_match = bool(
        isinstance(completed_timestep, int)
        and not isinstance(completed_timestep, bool)
        and completed_timestep + 1 == history_frames
    )
    record = {
        "path": str(directory),
        "validation_level": "full" if full else "fast",
        "expected_frames": int(expected_frames),
        "history_frames": history_frames,
        "history_range_valid": history_range_valid,
        "timesteps_match": timesteps_match,
        "history": history,
        "state": state_record,
    }
    record["ready"] = bool(
        history["ready"]
        and history_range_valid
        and state_record["ready"]
        and timesteps_match
    )
    return record


def validate_d3dgs_provenance(
    path: Path,
    params_path: Path,
    expected_frames: int,
    *,
    expected_protocol: Optional[Dict[str, Any]] = None,
    full: bool = False,
) -> Dict[str, Any]:
    """Validate the checkpoint identity and source record used by the release."""

    path = Path(path)
    params_path = Path(params_path)
    record: Dict[str, Any] = {
        "path": str(path),
        "validation_level": "full" if full else "fast",
        **_file(path),
    }
    if not record["ready"]:
        return record
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        params = data.get("params", {})
        declared_path = Path(str(params.get("path", ""))).expanduser()
        declared_size = params.get("size_bytes")
        declared_sha256 = str(params.get("sha256", ""))
        path_matches = bool(
            declared_path.is_absolute()
            and declared_path.resolve() == params_path.resolve()
        )
        size_matches = bool(
            params_path.is_file()
            and declared_size == params_path.stat().st_size
        )
        sha256_declared = bool(
            len(declared_sha256) == 64
            and all(
                character in "0123456789abcdef"
                for character in declared_sha256
            )
        )
        sha256_matches = None
        if full and params_path.is_file() and sha256_declared:
            digest = hashlib.sha256()
            with params_path.open("rb") as handle:
                for chunk in iter(
                    lambda: handle.read(16 * 1024 * 1024), b""
                ):
                    digest.update(chunk)
            sha256_matches = digest.hexdigest() == declared_sha256
        command = data.get("command")
        source = data.get("source")
        execution = data.get("execution")
        protocol = data.get("protocol")
        inferred_protocol: Dict[str, Any] = {}
        if isinstance(command, list):
            def argument(name: str) -> Optional[str]:
                if name not in command:
                    return None
                index = command.index(name)
                return (
                    str(command[index + 1])
                    if index + 1 < len(command)
                    else None
                )

            if "--isotropic" in command:
                inferred_protocol["scale_mode"] = "isotropic"
            elif "--anisotropic" in command:
                inferred_protocol["scale_mode"] = "anisotropic"
            for flag, key in (
                ("--num-timesteps", "frames"),
                ("--num-iters", "first_frame_iterations"),
                ("--checkpoint-every", "checkpoint_every_frames"),
            ):
                value = argument(flag)
                if value is not None:
                    inferred_protocol[key] = int(value)
            inferred_protocol["mask_background"] = "--no-mask" not in command
            inferred_protocol["floor_loss"] = "--no-floor" not in command
        effective_protocol = (
            protocol
            if isinstance(protocol, dict)
            else inferred_protocol
        )
        protocol_matches = bool(
            expected_protocol is not None
            and all(
                effective_protocol.get(key) == value
                for key, value in expected_protocol.items()
            )
        )
        command_runs_training = bool(
            isinstance(command, list)
            and any(
                str(value).endswith("resume_d3dgs.py")
                for value in command
            )
            and not (
                isinstance(execution, dict)
                and execution.get("training_command_executed") is False
            )
        )
        record.update(
            readable=True,
            schema_version=data.get("schema_version"),
            method=data.get("method"),
            expected_frames=data.get("expected_frames"),
            params_path_matches=path_matches,
            params_size_matches=size_matches,
            sha256_declared=sha256_declared,
            sha256_matches=sha256_matches,
            command_recorded=isinstance(command, list) and bool(command),
            source_recorded=(
                isinstance(source, dict)
                and bool(source.get("path"))
                and bool(
                    source.get("revision")
                    or source.get("files")
                )
            ),
            protocol=effective_protocol,
            protocol_matches=protocol_matches,
            command_runs_training=command_runs_training,
        )
        record["ready"] = bool(
            record["schema_version"] == 1
            and record["method"] == "Dynamic3DGaussians"
            and record["expected_frames"] == int(expected_frames)
            and path_matches
            and size_matches
            and sha256_declared
            and (sha256_matches is not False)
            and record["command_recorded"]
            and record["source_recorded"]
        )
        record["strict_candidate"] = bool(
            record["ready"]
            and protocol_matches
            and command_runs_training
        )
        record["strict_ready"] = bool(
            record["strict_candidate"]
            and full
            and sha256_matches is True
        )
    except Exception as exc:
        record.update(
            ready=False,
            readable=False,
            error=repr(exc),
        )
    return record


def validate_prediction_artifact(
    directory: Path,
    expected_frames: int,
    *,
    metadata_name: str = "metadata.json",
    extra_required: Iterable[str] = (),
    require_test_prediction: bool = False,
    train_ratio: float = 0.8,
    expected_points: Optional[int] = None,
    full: bool = False,
) -> Dict[str, Any]:
    """Validate the common all-Gaussian prediction directory contract.

    ``frame_ids.npy`` is optional for compatibility with the first release
    outputs. When present, it must match the metadata and contiguous split.
    """

    directory = Path(directory)
    required = [
        metadata_name,
        "warped_sequence.npy",
        "point_indices.npy",
    ]
    if require_test_prediction:
        required.append("test_warped_sequence.npy")
    required.extend(extra_required)
    required = list(dict.fromkeys(required))
    files = {
        name: _file(directory / name)
        for name in required
    }
    record: Dict[str, Any] = {
        "path": str(directory),
        "files": files,
        "ready": False,
    }
    missing = [
        name for name, state in files.items() if not state["ready"]
    ]
    if missing:
        record["missing"] = missing
        return record

    try:
        metadata = json.loads(
            (directory / metadata_name).read_text(encoding="utf-8")
        )
        if not isinstance(metadata, dict):
            raise ValueError("prediction metadata must be a JSON object")
        prediction = np.load(
            directory / "warped_sequence.npy",
            mmap_mode="r",
        )
        point_indices = np.load(
            directory / "point_indices.npy",
            mmap_mode="r",
        )
        training_frames = int(expected_frames * train_ratio)
        test_frames = expected_frames - training_frames
        expected_test_ids = np.arange(
            training_frames,
            expected_frames,
            dtype=np.int64,
        )
        points = (
            int(prediction.shape[1])
            if prediction.ndim == 3
            else -1
        )
        finite = _finite(prediction, full)

        test_path = directory / "test_warped_sequence.npy"
        test_prediction = None
        if test_path.is_file():
            test_prediction = np.load(test_path, mmap_mode="r")
            finite = finite and _finite(test_prediction, full)

        frame_ids_path = directory / "frame_ids.npy"
        frame_ids = None
        if frame_ids_path.is_file():
            frame_ids = np.load(frame_ids_path, mmap_mode="r")

        metadata_test_ids = np.asarray(
            metadata["test_frame_ids"],
            dtype=np.int64,
        )
        ready = (
            prediction.shape == (expected_frames, points, 3)
            and point_indices.shape == (points,)
            and np.issubdtype(point_indices.dtype, np.integer)
            and np.array_equal(
                point_indices,
                np.arange(points, dtype=point_indices.dtype),
            )
            and int(metadata["training_frames"]) == training_frames
            and metadata_test_ids.shape == expected_test_ids.shape
            and np.array_equal(metadata_test_ids, expected_test_ids)
            and int(metadata["num_points"]) == points
            and (
                expected_points is None
                or points == expected_points
            )
            and finite
        )
        if test_prediction is not None:
            ready = ready and (
                test_prediction.shape == (test_frames, points, 3)
            )
        if frame_ids is not None:
            ready = (
                ready
                and frame_ids.shape == expected_test_ids.shape
                and np.issubdtype(frame_ids.dtype, np.integer)
                and np.array_equal(frame_ids, expected_test_ids)
            )

        record.update(
            ready=bool(ready),
            readable=True,
            frames=expected_frames,
            training_frames=training_frames,
            test_frames=test_frames,
            points=points,
            expected_points=expected_points,
            all_points=(
                expected_points is None
                or points == expected_points
            ),
            finite=finite,
            method=metadata.get(
                "method",
                metadata.get("algorithm"),
            ),
            metrics=metadata.get("metrics", {}),
            legacy_missing_frame_ids=not frame_ids_path.is_file(),
        )
    except Exception as exc:
        record.update(
            ready=False,
            readable=False,
            error=repr(exc),
        )
    return record


def validate_method_prediction(
    directory: Path,
    expected_frames: int,
    method: str,
    *,
    train_ratio: float = 0.8,
    expected_points: Optional[int] = None,
    expected_protocol: Optional[Dict[str, Any]] = None,
    full: bool = False,
) -> Dict[str, Any]:
    """Validate one LBS, BoB, or PartMM release result."""

    if method not in METHOD_EXTRA_FILES:
        raise ValueError(f"unknown multiview method artifact: {method}")
    record = validate_prediction_artifact(
        directory,
        expected_frames,
        extra_required=METHOD_EXTRA_FILES[method],
        require_test_prediction=True,
        train_ratio=train_ratio,
        expected_points=expected_points,
        full=full,
    )
    if not record["ready"]:
        return record

    directory = Path(directory)
    points = int(record["points"])
    test_frames = int(record["test_frames"])
    try:
        if method == "bob":
            metadata = json.loads(
                (directory / "metadata.json").read_text(
                    encoding="utf-8"
                )
            )
            protocol = metadata.get("protocol", {})
            implementation = metadata.get("implementation", {})
            pose_contract_matches = bool(
                metadata.get("method")
                == "smpl_conditioned_bag_of_bones"
                and protocol.get("adapter_id") == BOB_ADAPTER_ID
                and implementation.get("adapter_id") == BOB_ADAPTER_ID
                and protocol.get("joint_mapping_id")
                == SMPL_JOINT_MAPPING_ID
                and protocol.get("pose_joint_range") == [1, 24]
                and protocol.get("pose_dimensions")
                == SMPL_ROOTLESS_POSE_DIM
                and protocol.get("skeleton_links")
                == SMPL_ROOTLESS_JOINTS
                and protocol.get("root_joint_id")
                == SMPL_ROOT_JOINT_ID
                and protocol.get("root_pose_dimensions") == 3
                and protocol.get("root_handling")
                == SMPL_ROOT_HANDLING
                and protocol.get("root_translation_handling")
                == SMPL_ROOT_TRANSLATION_HANDLING
                and protocol.get("predicted_gaussian_attributes")
                == list(BOB_PREDICTED_GAUSSIAN_ATTRIBUTES)
                and protocol.get("evaluation_attribute_source")
                == BOB_EVALUATION_ATTRIBUTE_SOURCE
                and protocol.get("oracle_evaluation_attributes")
                == list(BOB_ORACLE_EVALUATION_ATTRIBUTES)
            )
            protocol_matches = bool(
                expected_protocol is None
                or (
                    metadata.get("method")
                    == expected_protocol.get(
                        "method",
                        metadata.get("method"),
                    )
                    and all(
                        protocol.get(key) == value
                        for key, value in expected_protocol.items()
                        if key != "method"
                    )
                )
            )
            loss = np.load(
                directory / "loss_history.npy",
                mmap_mode="r",
            )
            weights = np.load(
                directory / "skinning_weights.npy",
                mmap_mode="r",
            )
            skeleton_weights = np.load(
                directory / "skeleton_skinning_weights.npy",
                mmap_mode="r",
            )
            arrays_ready = (
                loss.ndim == 1
                and loss.size > 0
                and (
                    expected_protocol is None
                    or loss.size
                    == int(expected_protocol["num_iterations"])
                )
                and _finite(loss, full)
                and _valid_skinning_weights(weights, points, full)
                and (
                    expected_protocol is None
                    or weights.shape[1]
                    == int(expected_protocol["num_bones"])
                )
                and _valid_skinning_weights(
                    skeleton_weights,
                    points,
                    full,
                )
                and skeleton_weights.shape[1]
                == SMPL_ROOTLESS_JOINTS
                and pose_contract_matches
                and protocol_matches
            )
            record["method_artifacts"] = {
                "iterations": int(loss.size),
                "bones": int(weights.shape[1]),
                "weights_shape": list(weights.shape),
                "skeleton_weights_shape": list(
                    skeleton_weights.shape
                ),
                "pose_contract_matches": pose_contract_matches,
                "protocol_matches": protocol_matches,
            }
        elif method == "lbs":
            with np.load(
                directory / "robust_weights.npz",
                allow_pickle=False,
            ) as archive:
                missing = [
                    key
                    for key in (
                        "surface_weights",
                        "gaussian_weights",
                        "projection_squared_distance",
                    )
                    if key not in archive.files
                ]
                gaussian_weights = (
                    archive["gaussian_weights"]
                    if not missing
                    else np.asarray([])
                )
                projection_distance = (
                    archive["projection_squared_distance"]
                    if not missing
                    else np.asarray([])
                )
                arrays_ready = (
                    not missing
                    and _valid_skinning_weights(
                        gaussian_weights,
                        points,
                        full,
                    )
                    and projection_distance.shape == (points,)
                    and _finite(projection_distance, full)
                    and bool((projection_distance >= 0).all())
                )
                record["method_artifacts"] = {
                    "missing_weight_keys": missing,
                    "gaussian_weights_shape": list(
                        gaussian_weights.shape
                    ),
                    "projection_distance_shape": list(
                        projection_distance.shape
                    ),
                }
        else:
            metadata = json.loads(
                (directory / "metadata.json").read_text(
                    encoding="utf-8"
                )
            )
            partmm_protocol = metadata.get("partmm", {})
            protocol_matches = bool(
                expected_protocol is None
                or (
                    set(PARTMM_PROTOCOL_KEYS).issubset(
                        expected_protocol
                    )
                    and expected_protocol.get("algorithm")
                    == PARTMM_ALGORITHM
                    and metadata.get("algorithm")
                    == PARTMM_ALGORITHM
                    and all(
                        partmm_protocol.get(key) == value
                        for key, value in expected_protocol.items()
                        if key != "algorithm"
                    )
                )
            )
            transforms = np.load(
                directory / "bone_transforms.npy",
                mmap_mode="r",
            )
            arrays_ready = (
                transforms.ndim == 4
                and transforms.shape[0] == test_frames
                and transforms.shape[1] > 0
                and transforms.shape[2:] == (4, 4)
                and _finite(transforms, full)
                and np.allclose(
                    transforms[..., 3, :],
                    np.asarray([0.0, 0.0, 0.0, 1.0]),
                    atol=1e-5,
                )
                and protocol_matches
            )
            record["method_artifacts"] = {
                "bone_transforms_shape": list(transforms.shape),
                "algorithm": metadata.get("algorithm"),
                "protocol_matches": protocol_matches,
            }
        record["ready"] = bool(arrays_ready)
        record["method_artifacts"]["ready"] = bool(arrays_ready)
    except Exception as exc:
        record.update(
            ready=False,
            method_artifacts={
                "ready": False,
                "error": repr(exc),
            },
        )
    return record


def _valid_skinning_weights(
    weights: np.ndarray,
    points: int,
    full: bool,
) -> bool:
    if (
        weights.ndim != 2
        or weights.shape[0] != points
        or weights.shape[1] < 1
    ):
        return False
    sample = weights
    if not full:
        step = max(1, len(weights) // 5)
        sample = weights[::step]
    return bool(
        np.isfinite(sample).all()
        and (sample >= -1e-6).all()
        and np.allclose(sample.sum(axis=1), 1.0, atol=1e-5)
    )


def _ssdr_transform(
    archive,
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    if "world_transformations" in archive.files:
        return archive["world_transformations"], "world_transformations"
    if "transformations" in archive.files:
        return archive["transformations"], "transformations"
    return None, None


def validate_ssdr_cache(
    path: Path,
    expected_frames: int,
    *,
    train_ratio: float = 0.8,
    full: bool = False,
) -> Dict[str, Any]:
    """Validate new and historical training-only SSDR cache layouts."""

    path = Path(path)
    record: Dict[str, Any] = {
        "path": str(path),
        **_file(path),
    }
    if not record["ready"]:
        return record
    try:
        with np.load(path, allow_pickle=False) as archive:
            missing = [
                key
                for key in ("rest_pose", "weights")
                if key not in archive.files
            ]
            transforms, transform_key = _ssdr_transform(archive)
            if transforms is None:
                missing.append(
                    "world_transformations|transformations"
                )
            rest = (
                archive["rest_pose"]
                if "rest_pose" in archive.files
                else np.asarray([])
            )
            weights = (
                archive["weights"]
                if "weights" in archive.files
                else np.asarray([])
            )
            fitted = (
                int(archive["training_frames"])
                if "training_frames" in archive.files
                else (
                    len(transforms)
                    if transforms is not None
                    else -1
                )
            )
            frame_means = (
                archive["frame_means"]
                if (
                    transform_key == "transformations"
                    and "frame_means" in archive.files
                )
                else None
            )

            expected_training = int(expected_frames * train_ratio)
            points = (
                int(rest.shape[0])
                if rest.ndim == 2
                else -1
            )
            bones = (
                int(transforms.shape[1])
                if transforms is not None
                and transforms.ndim == 4
                else -1
            )
            if weights.shape == (points, bones):
                weights_layout = "points_bones"
            elif weights.shape == (bones, points):
                weights_layout = "bones_points"
            else:
                weights_layout = "invalid"

            finite = (
                _finite(rest, full)
                and _finite(weights, full)
                and transforms is not None
                and _finite(transforms, full)
            )
            frame_means_ready = True
            if transform_key == "transformations":
                frame_means_ready = (
                    frame_means is not None
                    and frame_means.shape == (fitted, 3)
                    and _finite(frame_means, full)
                )
            ready = (
                not missing
                and fitted == expected_training
                and transforms is not None
                and transforms.shape == (fitted, bones, 4, 4)
                and rest.shape == (points, 3)
                and points > 0
                and bones > 0
                and weights_layout != "invalid"
                and frame_means_ready
                and finite
            )
            record.update(
                ready=bool(ready),
                readable=True,
                missing_keys=missing,
                training_frames=fitted,
                training_frames_inferred=(
                    "training_frames" not in archive.files
                ),
                points=points,
                bones=bones,
                weights_layout=weights_layout,
                transform_key=transform_key,
                finite=bool(finite),
            )
    except Exception as exc:
        record.update(
            ready=False,
            readable=False,
            error=repr(exc),
        )
        return record

    metadata_path = path.with_suffix(".json")
    if metadata_path.is_file():
        try:
            metadata = json.loads(
                metadata_path.read_text(encoding="utf-8")
            )
            record["requested_bones"] = metadata.get(
                "requested_bones"
            )
            record["fit_rmse_mm"] = metadata.get(
                "full_reconstruction_rmse_mm"
            )
        except Exception as exc:
            record["metadata_error"] = repr(exc)
    return record
