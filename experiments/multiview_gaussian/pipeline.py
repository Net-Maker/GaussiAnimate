"""Manage the long-running multiview computation and release resource tree."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import yaml

from .configuration import expand_environment
from .release_validation import (
    validate_d3dgs_params,
    validate_d3dgs_provenance,
    validate_d3dgs_resume_checkpoint,
    validate_method_prediction,
    validate_prediction_artifact,
    validate_ssdr_cache,
)
from .resources import load_resource_config


_PARTMM_PROTOCOL_FIELDS = (
    ("partmm_num_parts", "num_parts", int),
    ("partmm_patch_size", "patch_size", int),
    ("partmm_num_neighbors", "num_neighbors", int),
    ("partmm_pyramid_levels", "pyramid_levels", int),
    ("partmm_pyramid_blend", "pyramid_blend", float),
    ("partmm_overlap_rings", "overlap_rings", int),
    ("partmm_align_parts", "align_parts", bool),
    ("partmm_alignment_mode", "alignment_mode", str),
    ("partmm_partition_mode", "partition_mode", str),
    ("partmm_bone_overlap_ratio", "bone_overlap_ratio", float),
)

_BOB_PROTOCOL_FIELDS = (
    ("bob_iterations", "num_iterations", int),
    ("bob_bones", "num_bones", int),
    ("bob_learning_rate", "learning_rate", float),
    ("bob_batch_frames", "batch_frames", int),
    ("bob_adapter_id", "adapter_id", str),
    ("bob_joint_mapping_id", "joint_mapping_id", str),
    ("bob_pose_joint_range", "pose_joint_range", list),
    ("bob_pose_dimensions", "pose_dimensions", int),
    ("bob_skeleton_links", "skeleton_links", int),
    ("bob_root_joint_id", "root_joint_id", int),
    ("bob_root_pose_dimensions", "root_pose_dimensions", int),
    ("bob_root_handling", "root_handling", str),
    (
        "bob_root_translation_handling",
        "root_translation_handling",
        str,
    ),
    (
        "bob_predicted_gaussian_attributes",
        "predicted_gaussian_attributes",
        list,
    ),
    (
        "bob_evaluation_attribute_source",
        "evaluation_attribute_source",
        str,
    ),
    (
        "bob_oracle_evaluation_attributes",
        "oracle_evaluation_attributes",
        list,
    ),
)


def load_pipeline_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = expand_environment(yaml.safe_load(handle))
    if config.get("schema_version") != 1:
        raise ValueError("multiview pipeline config requires schema_version: 1")
    config["_path"] = str(path.resolve())
    release_root = Path(config["release"]["root"]).resolve()
    config["release"]["root"] = str(release_root)
    for section, keys in (
        ("d3dgs", ("actorshq_output", "dna_output", "logs")),
        ("derived", ("runs",)),
        ("diagnostics", ("historical_lbs",)),
        ("evaluation", ("root",)),
    ):
        for key in keys:
            value = Path(config[section][key])
            if not value.is_absolute():
                value = release_root / value
            config[section][key] = str(value.resolve())
    resource_path = Path(config["resources"])
    if not resource_path.is_absolute():
        resource_path = path.parent / resource_path
    config["_resources"] = load_resource_config(resource_path.resolve())
    return config


def _cases(config: Dict[str, Any], selected: Iterable[str]) -> List[Dict[str, Any]]:
    selected_ids = set(selected)
    cases = []
    for dataset in config["_resources"]["datasets"]:
        for case in dataset["cases"]:
            record = dict(case)
            record["dataset"] = dataset["id"]
            record["expected_frames"] = int(dataset["expected_frames"])
            record["expected_cameras"] = int(dataset["expected_cameras"])
            cases.append(record)
    known = {case["id"] for case in cases}
    unknown = selected_ids - known
    if unknown:
        raise ValueError(f"unknown multiview cases: {', '.join(sorted(unknown))}")
    return [
        case
        for case in cases
        if not selected_ids or case["id"] in selected_ids
    ]


def _params_record(
    path: Path,
    expected_frames: int,
    *,
    expected_scale_dims: int = None,
    expected_cameras: int = None,
    full: bool = False,
) -> Dict[str, Any]:
    """Compatibility alias for the canonical checkpoint validator."""

    return validate_d3dgs_params(
        path,
        expected_frames,
        expected_scale_dims=expected_scale_dims,
        expected_cameras=expected_cameras,
        full=full,
    )


def _scale_mode(config: Dict[str, Any], case: Dict[str, Any]) -> str:
    modes = config["d3dgs"]["protocol"]["scale_mode_by_dataset"]
    mode = modes.get(case["dataset"])
    if mode not in ("isotropic", "anisotropic"):
        raise ValueError(
            f"{case['id']} has no valid D3DGS scale mode: {mode!r}"
        )
    return mode


def _scale_dims(config: Dict[str, Any], case: Dict[str, Any]) -> int:
    return 1 if _scale_mode(config, case) == "isotropic" else 3


def _training_protocol(
    config: Dict[str, Any],
    case: Dict[str, Any],
) -> Dict[str, Any]:
    protocol = config["d3dgs"]["protocol"]
    return {
        "scale_mode": _scale_mode(config, case),
        "frames": int(case["expected_frames"]),
        "first_frame_iterations": int(
            protocol["first_frame_iterations"]
        ),
        "checkpoint_every_frames": int(
            protocol["checkpoint_every_frames"]
        ),
        "mask_background": bool(protocol["mask_background"]),
        "floor_loss": bool(protocol["floor_loss"]),
    }


def _partmm_protocol(
    config: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    protocol = config.get("derived", {}).get("protocol", {})
    keys = ("partmm_algorithm",) + tuple(
        config_key
        for config_key, _, _ in _PARTMM_PROTOCOL_FIELDS
    )
    present = [key for key in keys if key in protocol]
    if not present:
        return None
    missing = [key for key in keys if key not in protocol]
    if missing:
        raise ValueError(
            "incomplete PartMM protocol configuration: "
            + ", ".join(missing)
        )
    return {
        "algorithm": str(protocol["partmm_algorithm"]),
        **{
            metadata_key: cast(protocol[config_key])
            for config_key, metadata_key, cast
            in _PARTMM_PROTOCOL_FIELDS
        },
    }


def _bob_protocol(config: Dict[str, Any]) -> Dict[str, Any]:
    protocol = config["derived"]["protocol"]
    required = (
        "bob_max_points",
        "bob_opacity_threshold",
    ) + tuple(
        config_key
        for config_key, _, _ in _BOB_PROTOCOL_FIELDS
    )
    missing = [key for key in required if key not in protocol]
    if missing:
        raise ValueError(
            "incomplete BoB protocol configuration: "
            + ", ".join(missing)
        )
    if protocol["bob_max_points"] is not None:
        raise ValueError("strict BoB protocol requires all Gaussians")
    if float(protocol["bob_opacity_threshold"]) != 0.0:
        raise ValueError("strict BoB protocol disables opacity filtering")
    return {
        "method": "smpl_conditioned_bag_of_bones",
        "driven_points": "all_gaussians",
        "opacity_filter": None,
        "point_subsampling": None,
        **{
            metadata_key: cast(protocol[config_key])
            for config_key, metadata_key, cast in _BOB_PROTOCOL_FIELDS
        },
    }


def _source_params(config: Dict[str, Any], case: Dict[str, Any]) -> Path:
    roots = config["_resources"]["roots"]
    root = roots[
        "dna_params" if case["dataset"] == "dna_rendering" else "actorshq_params"
    ]
    return Path(root) / case["params"] / "params.npz"


def _release_params(config: Dict[str, Any], case: Dict[str, Any]) -> Path:
    key = "dna_output" if case["dataset"] == "dna_rendering" else "actorshq_output"
    return Path(config["d3dgs"][key]) / case["id"] / "params.npz"


def _method_root(
    config: Dict[str, Any],
    case: Dict[str, Any],
    method: str,
) -> Path:
    return (
        Path(config["derived"]["runs"])
        / case["dataset"]
        / case["id"]
        / method
    )


def _prediction_record(
    directory: Path,
    expected_frames: int,
    *,
    metadata_name: str = "metadata.json",
    extra_required=(),
) -> Dict[str, Any]:
    """Compatibility alias for the canonical prediction validator."""

    return validate_prediction_artifact(
        directory,
        expected_frames,
        metadata_name=metadata_name,
        extra_required=extra_required,
    )


def _ssdr_record(path: Path, expected_frames: int) -> Dict[str, Any]:
    """Compatibility alias for the canonical SSDR cache validator."""

    return validate_ssdr_cache(path, expected_frames)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _historical_lbs_record(
    directory: Path,
    expected_frames: int,
    *,
    require_provenance: bool,
) -> Dict[str, Any]:
    record = {"path": str(directory), "ready": False}
    required = (
        "metadata.json",
        "driven_sequence.npy",
        "full_driven_sequence.npy",
        "point_indices.npy",
        "gaussian_weights.npy",
    )
    missing = [name for name in required if not (directory / name).is_file()]
    if require_provenance and not (directory / "provenance.json").is_file():
        missing.append("provenance.json")
    record["missing"] = missing
    if missing:
        return record
    try:
        metadata = json.loads(
            (directory / "metadata.json").read_text(encoding="utf-8")
        )
        driven = np.load(directory / "driven_sequence.npy", mmap_mode="r")
        full = np.load(directory / "full_driven_sequence.npy", mmap_mode="r")
        indices = np.load(directory / "point_indices.npy", mmap_mode="r")
        weights = np.load(directory / "gaussian_weights.npy", mmap_mode="r")
        expected_test = expected_frames - int(expected_frames * 0.8)
        points = len(indices)
        valid = (
            driven.shape == (expected_test, points, 3)
            and full.shape == (expected_frames, points, 3)
            and indices.shape == (points,)
            and np.issubdtype(indices.dtype, np.integer)
            and weights.shape == (points, 24)
            and np.isfinite(driven).all()
            and np.isfinite(full).all()
            and np.isfinite(weights).all()
            and (weights >= -1e-6).all()
            and np.allclose(weights.sum(axis=1), 1.0, atol=1e-5)
            and int(metadata["n_total_frames"]) == expected_frames
            and int(metadata["n_test_frames"]) == expected_test
            and int(metadata["n_gaussians"]) == points
            and abs(float(metadata["train_ratio"]) - 0.8) < 1e-8
        )
        record.update(
            ready=valid,
            frames=expected_frames,
            test_frames=expected_test,
            points=points,
            implementation="nearest_surface_barycentric_weight_copy",
        )
    except Exception as exc:
        record["error"] = repr(exc)
    return record


def status(
    config: Dict[str, Any],
    selected: Iterable[str],
    *,
    full: bool = False,
) -> Dict[str, Any]:
    cases = {}
    for case in _cases(config, selected):
        release_path = _release_params(config, case)
        release_params = _params_record(
            release_path,
            case["expected_frames"],
            expected_scale_dims=_scale_dims(config, case),
            expected_cameras=case["expected_cameras"],
            full=full,
        )
        source_params = _params_record(
            _source_params(config, case),
            case["expected_frames"],
            expected_scale_dims=_scale_dims(config, case),
            expected_cameras=case["expected_cameras"],
            full=full,
        )
        params_record = (
            release_params
            if release_params["ready"]
            else source_params
        )
        expected_points = (
            int(params_record["means3d_shape"][1])
            if params_record["ready"]
            else None
        )
        case_root = Path(config["d3dgs"][
            "dna_output" if case["dataset"] == "dna_rendering" else "actorshq_output"
        ]) / case["id"]
        lbs = validate_method_prediction(
            _method_root(config, case, "smpl_lbs") / "lbs",
            case["expected_frames"],
            "lbs",
            expected_points=expected_points,
        )
        partmm = validate_method_prediction(
            _method_root(config, case, "smpl_partmm")
            / "partmm"
            / case["id"],
            case["expected_frames"],
            "partmm",
            expected_points=expected_points,
            expected_protocol=_partmm_protocol(config),
        )
        bob = validate_method_prediction(
            _method_root(config, case, "smpl_bob")
            / "bob"
            / case["id"],
            case["expected_frames"],
            "bob",
            expected_points=expected_points,
            expected_protocol=_bob_protocol(config),
        )
        ssdr = validate_ssdr_cache(
            _method_root(config, case, "smpl_partmm")
            / "ssdr"
            / "train_only.npz",
            case["expected_frames"],
        )
        provenance = case_root / "training_provenance.json"
        provenance_record = validate_d3dgs_provenance(
            provenance,
            release_path,
            case["expected_frames"],
            expected_protocol=_training_protocol(config, case),
            full=full,
        )
        strict_ready = bool(
            release_params["ready"]
            and release_params.get("validation_level") == "full"
            and provenance_record.get("strict_ready")
        )
        resume_record = validate_d3dgs_resume_checkpoint(
            case_root,
            case["expected_frames"],
            expected_scale_dims=_scale_dims(config, case),
            expected_cameras=case["expected_cameras"],
            full=full,
        )
        cases[case["id"]] = {
            "dataset": case["dataset"],
            "d3dgs": {
                "release": release_params,
                "source": source_params,
                "provenance": str(provenance),
                "provenance_record": provenance_record,
                "ready": (
                    release_params["ready"]
                    and provenance_record["ready"]
                ),
                "strict_ready": strict_ready,
                "resume_record": resume_record,
                "resumable": resume_record["ready"],
            },
            "lbs": lbs,
            "bob": bob,
            "ssdr": ssdr,
            "partmm": partmm,
            "historical_lbs": _historical_lbs_record(
                Path(config["diagnostics"]["historical_lbs"]) / case["id"],
                case["expected_frames"],
                require_provenance=True,
            ),
        }
    summary = {
        stage: sum(record[stage]["ready"] for record in cases.values())
        for stage in (
            "d3dgs",
            "lbs",
            "bob",
            "ssdr",
            "partmm",
            "historical_lbs",
        )
    }
    summary["d3dgs_strict"] = sum(
        record["d3dgs"]["strict_ready"]
        for record in cases.values()
    )
    return {
        "schema_version": 1,
        "release_id": config["release"]["id"],
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "cases": cases,
        "summary": summary,
    }


def adopt_existing(config: Dict[str, Any], selected: Iterable[str]) -> None:
    for case in _cases(config, selected):
        source = _source_params(config, case)
        destination = _release_params(config, case)
        if _params_record(destination, case["expected_frames"])["ready"]:
            print(f"ready: {case['id']}")
            continue
        if not _params_record(source, case["expected_frames"])["ready"]:
            print(f"missing: {case['id']}")
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".npz.partial")
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
        print(f"adopted: {case['id']} -> {destination}")


def adopt_historical_lbs(
    config: Dict[str, Any],
    selected: Iterable[str],
) -> None:
    source_root = Path(config["_resources"]["roots"]["lbs"])
    output_root = Path(config["diagnostics"]["historical_lbs"])
    for case in _cases(config, selected):
        source = source_root / case["id"]
        destination = output_root / case["id"]
        if _historical_lbs_record(
            destination,
            case["expected_frames"],
            require_provenance=True,
        )["ready"]:
            print(f"ready: {case['id']}")
            continue
        source_record = _historical_lbs_record(
            source,
            case["expected_frames"],
            require_provenance=False,
        )
        if not source_record["ready"]:
            print(f"missing: {case['id']}")
            continue
        source_params = _source_params(config, case)
        release_params = _release_params(config, case)
        if not _params_record(
            source_params, case["expected_frames"]
        )["ready"] or not _params_record(
            release_params, case["expected_frames"]
        )["ready"]:
            print(f"unverified params: {case['id']}")
            continue
        source_hash = _sha256(source_params)
        release_hash = _sha256(release_params)
        if source_hash != release_hash:
            raise RuntimeError(
                f"refusing {case['id']}: historical and release params differ"
            )
        destination.mkdir(parents=True, exist_ok=True)
        for name in (
            "metadata.json",
            "driven_sequence.npy",
            "full_driven_sequence.npy",
            "point_indices.npy",
            "gaussian_weights.npy",
        ):
            temporary = destination / f"{name}.partial"
            shutil.copy2(source / name, temporary)
            os.replace(temporary, destination / name)
        provenance = {
            "schema_version": 1,
            "case": case["id"],
            "adopted_utc": datetime.now(timezone.utc).isoformat(),
            "classification": "diagnostic_not_paper_robust_lbs",
            "implementation": "nearest_surface_barycentric_weight_copy",
            "source_params_sha256": source_hash,
            "release_params_sha256": release_hash,
            "source_directory": str(source.resolve()),
        }
        _write_json(destination / "provenance.json", provenance)
        verified = _historical_lbs_record(
            destination,
            case["expected_frames"],
            require_provenance=True,
        )
        if not verified["ready"]:
            raise RuntimeError(f"adopted LBS did not validate: {case['id']}")
        print(f"adopted diagnostic: {case['id']} -> {destination}")


def _d3dgs_train_command(
    config: Dict[str, Any],
    case: Dict[str, Any],
) -> List[str]:
    protocol = config["d3dgs"]["protocol"]
    d3dgs_root = Path(config["d3dgs"]["root"])
    python = str(Path(config["d3dgs"]["python"]).resolve())
    data_root = config["_resources"]["roots"]["data"]
    if int(case["expected_frames"]) != int(protocol["frames"]):
        raise ValueError(
            f"{case['id']} expects {case['expected_frames']} frames, "
            f"but the D3DGS protocol declares {protocol['frames']}"
        )
    return [
        python,
        str(Path(__file__).with_name("train_d3dgs.py").resolve()),
        "--d3dgs-root",
        str(d3dgs_root),
        "--data-dir",
        data_root,
        "--output-dir",
        str(Path(config["d3dgs"]["actorshq_output"])),
        "--train-id",
        case["data"],
        "--resource-id",
        case["id"],
        "--frames",
        str(case["expected_frames"]),
        "--iterations",
        str(protocol["first_frame_iterations"]),
        "--checkpoint-every",
        str(protocol["checkpoint_every_frames"]),
        "--experiment",
        config["release"]["id"],
        "--scale-mode",
        _scale_mode(config, case),
    ]


def train_d3dgs(
    config: Dict[str, Any],
    selected: Iterable[str],
    dry_run: bool,
) -> None:
    d3dgs_root = Path(config["d3dgs"]["root"])
    for case in _cases(config, selected):
        if case["dataset"] != "actorshq":
            continue
        destination = _release_params(config, case)
        checkpoint = _params_record(
            destination,
            case["expected_frames"],
            expected_scale_dims=_scale_dims(config, case),
            expected_cameras=case["expected_cameras"],
        )
        provenance = destination.parent / "training_provenance.json"
        provenance_record = validate_d3dgs_provenance(
            provenance,
            destination,
            case["expected_frames"],
            expected_protocol=_training_protocol(config, case),
            full=True,
        )
        if checkpoint["ready"] and not provenance_record["ready"]:
            raise RuntimeError(
                f"valid D3DGS checkpoint has missing or invalid training "
                f"provenance: "
                f"{provenance}; inspect the source and run train_d3dgs.py "
                "--adopt-only explicitly"
            )
        if checkpoint["ready"]:
            print(f"ready: {case['id']}")
            continue
        output_root = Path(config["d3dgs"]["actorshq_output"])
        case_root = output_root / case["id"]
        resume_record = validate_d3dgs_resume_checkpoint(
            case_root,
            case["expected_frames"],
            expected_scale_dims=_scale_dims(config, case),
            expected_cameras=case["expected_cameras"],
            full=True,
        )
        resumable = resume_record["ready"]
        if case_root.exists() and not resumable:
            raise RuntimeError(
                f"incomplete non-resumable output must be inspected: {case_root}"
            )
        command = _d3dgs_train_command(config, case)
        print("$ " + " ".join(command))
        if dry_run:
            continue
        log_path = Path(config["d3dgs"]["logs"]) / f"{case['id']}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log:
            log.write("\n$ " + " ".join(command) + "\n")
            log.flush()
            completed = subprocess.run(
                command,
                cwd=str(d3dgs_root),
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode:
            recovered = _params_record(
                destination,
                case["expected_frames"],
                expected_scale_dims=_scale_dims(config, case),
                expected_cameras=case["expected_cameras"],
                full=True,
            )
            if not recovered["ready"]:
                raise RuntimeError(
                    f"D3DGS failed for {case['id']}; see {log_path}"
                )
            finalize_command = command + ["--adopt-only"]
            with log_path.open("a", encoding="utf-8") as log:
                log.write(
                    "\n# training produced a valid checkpoint; "
                    "retrying provenance finalization\n$ "
                    + " ".join(finalize_command)
                    + "\n"
                )
                log.flush()
                finalized = subprocess.run(
                    finalize_command,
                    cwd=str(d3dgs_root),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            if finalized.returncode:
                raise RuntimeError(
                    f"D3DGS checkpoint is valid but provenance finalization "
                    f"failed for {case['id']}; see {log_path}"
                )
        verified = _params_record(
            destination,
            case["expected_frames"],
            expected_scale_dims=_scale_dims(config, case),
            expected_cameras=case["expected_cameras"],
            full=True,
        )
        if not verified["ready"]:
            raise RuntimeError(
                f"D3DGS completed without a valid params.npz: {destination}"
            )
        provenance_record = validate_d3dgs_provenance(
            provenance,
            destination,
            case["expected_frames"],
            expected_protocol=_training_protocol(config, case),
            full=True,
        )
        if not provenance_record["ready"]:
            raise RuntimeError(
                f"D3DGS completed without valid training provenance: "
                f"{provenance}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage multiview release computation")
    parser.add_argument(
        "command",
        choices=("status", "adopt", "adopt-historical-lbs", "train-d3dgs"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("pipeline.yaml"),
    )
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--full",
        action="store_true",
        help="fully scan arrays and verify checkpoint SHA-256",
    )
    args = parser.parse_args()
    config = load_pipeline_config(args.config)
    if args.command == "status":
        result = status(config, args.case, full=args.full)
        if args.output:
            _write_json(args.output, result)
        print(json.dumps(result["summary"], indent=2, sort_keys=True))
    elif args.command == "adopt":
        adopt_existing(config, args.case)
    elif args.command == "adopt-historical-lbs":
        adopt_historical_lbs(config, args.case)
    else:
        train_d3dgs(config, args.case, args.dry_run)


if __name__ == "__main__":
    main()
