"""Audit every raw and derived resource used by the multiview experiments."""

import argparse
import json
import os
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np
import yaml

from .configuration import expand_environment
from .release_validation import (
    validate_d3dgs_params,
    validate_method_prediction,
    validate_prediction_artifact,
    validate_ssdr_cache,
)


def load_resource_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = expand_environment(yaml.safe_load(handle))
    if config.get("schema_version") != 1:
        raise ValueError("multiview resource config requires schema_version: 1")
    roots = config.get("roots")
    datasets = config.get("datasets")
    if not isinstance(roots, dict) or not isinstance(datasets, list):
        raise ValueError("resource config requires roots and datasets")
    for key, value in roots.items():
        roots[key] = str(Path(value).resolve())
    return config


def _link_state(path: Path, target: Path) -> Dict[str, Any]:
    target_ready = target.exists()
    path_ready = path.exists()
    same_target = (
        path_ready
        and target_ready
        and path.resolve() == target.resolve()
    )
    existing_directory = (
        path_ready
        and target_ready
        and path.is_dir()
        and target.is_dir()
        and not path.is_symlink()
    )
    return {
        "ready": same_target or existing_directory,
        "path_exists": path_ready,
        "target_exists": target_ready,
        "is_symlink": path.is_symlink(),
        "same_target": same_target,
        "existing_directory": existing_directory,
    }


def _ensure_link(path: Path, target: Path) -> Dict[str, Any]:
    before = _link_state(path, target)
    if before["ready"]:
        return {**before, "action": "unchanged"}
    if path.exists() or path.is_symlink():
        return {**before, "action": "conflict"}
    if not target.exists():
        return {**before, "action": "missing_target"}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.symlink_to(target, target_is_directory=target.is_dir())
    return {**_link_state(path, target), "action": "created"}


def prepare_resource_links(config: Dict[str, Any]) -> Dict[str, Any]:
    roots = config["roots"]
    links = {
        "d3dgs_data": _ensure_link(
            Path(roots["data"]),
            Path(roots["dna_raw"]),
        ),
        "actorshq_raw": _ensure_link(
            Path(roots["actorshq_raw_alias"]),
            Path(roots["actorshq_raw"]),
        ),
        "actorshq_smplx_legacy": _ensure_link(
            Path(roots["actorshq_smplx_legacy"]),
            Path(roots["actorshq_smplx"]),
        ),
    }
    smplx_root = Path(roots["actorshq_smplx"])
    for dataset in config["datasets"]:
        if dataset["id"] != "actorshq":
            continue
        for case in dataset["cases"]:
            links[f"smplx/{case['id']}"] = _ensure_link(
                smplx_root / case["id"],
                Path(roots["actorshq_raw"]) / case["raw"] / "smplx",
            )
    source_keys = ("dna_ssdr_source", "actorshq_ssdr_source")
    if "ssdr" in roots and all(key in roots for key in source_keys):
        ssdr_root = Path(roots["ssdr"])
        for source_key in source_keys:
            for source in sorted(Path(roots[source_key]).glob("*.npz")):
                links[f"ssdr/{source.name}"] = _ensure_link(
                    ssdr_root / source.name,
                    source,
                )
    return {
        "links": links,
        "all_ready": all(record["ready"] for record in links.values()),
    }


def _file(path: Path, minimum_bytes: int = 1) -> Dict[str, Any]:
    exists = path.is_file()
    size = path.stat().st_size if exists else 0
    return {
        "ready": exists and size >= minimum_bytes,
        "bytes": size,
    }


def _json_schema(path: Path, required: Iterable[str]) -> Dict[str, Any]:
    record = _file(path)
    if not record["ready"]:
        return record
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list) and data:
            data = data[0]
        missing = [key for key in required if key not in data]
        record.update(readable=True, missing_keys=missing)
        record["ready"] = not missing
    except Exception as exc:
        record.update(ready=False, readable=False, error=repr(exc))
    return record


def _missing_paths(paths: Iterable[Path]) -> int:
    grouped = defaultdict(set)
    for path in paths:
        grouped[path.parent].add(path.name)
    missing = 0
    for directory, expected in grouped.items():
        try:
            available = {entry.name for entry in os.scandir(directory)}
        except OSError:
            missing += len(expected)
            continue
        missing += len(expected - available)
    return missing


def _metadata(
    case_dir: Path,
    expected_frames: int,
    expected_cameras: int,
    full: bool,
) -> Dict[str, Any]:
    metadata_path = case_dir / "train_meta.json"
    record = _file(metadata_path)
    if not record["ready"]:
        return record
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        filenames = metadata["fn"]
        frame_count = len(filenames)
        camera_counts = [len(frame) for frame in filenames]
        structural = (
            frame_count == expected_frames
            and camera_counts
            and min(camera_counts) == expected_cameras
            and max(camera_counts) == expected_cameras
            and len(metadata["k"]) == frame_count
            and len(metadata["w2c"]) == frame_count
        )
        pairs = [
            (case_dir / "ims" / name, case_dir / "seg" / Path(name).with_suffix(".png"))
            for frame in filenames
            for name in frame
        ]
        checked = pairs if full else pairs[:expected_cameras] + pairs[-expected_cameras:]
        if full:
            missing_images = _missing_paths(image for image, _ in checked)
            missing_masks = _missing_paths(mask for _, mask in checked)
        else:
            missing_images = sum(not image.is_file() for image, _ in checked)
            missing_masks = sum(not mask.is_file() for _, mask in checked)
        init_path = case_dir / "init_pt_cld.npz"
        init_record = _pointcloud(init_path)
        record.update(
            readable=True,
            frames=frame_count,
            cameras=min(camera_counts) if camera_counts else 0,
            width=metadata.get("w"),
            height=metadata.get("h"),
            referenced_pairs=len(pairs),
            checked_pairs=len(checked),
            missing_images=missing_images,
            missing_masks=missing_masks,
            init_pointcloud=init_record,
        )
        record["ready"] = (
            structural
            and missing_images == 0
            and missing_masks == 0
            and init_record["ready"]
        )
    except Exception as exc:
        record.update(ready=False, readable=False, error=repr(exc))
    return record


def _pointcloud(path: Path) -> Dict[str, Any]:
    record = _file(path)
    if not record["ready"]:
        return record
    try:
        with np.load(path, allow_pickle=False) as data:
            points = data["data"]
            sample = np.asarray(points[: min(len(points), 1024)])
        record.update(
            readable=True,
            shape=list(points.shape),
            finite_sample=bool(np.isfinite(sample).all()),
        )
        record["ready"] = (
            points.ndim == 2
            and points.shape[1] >= 7
            and record["finite_sample"]
        )
    except Exception as exc:
        record.update(ready=False, readable=False, error=repr(exc))
    return record


def _params(path: Path, expected_frames: int) -> Dict[str, Any]:
    """Compatibility alias for the canonical checkpoint validator."""

    return validate_d3dgs_params(path, expected_frames)


def _actorshq_raw(
    raw_dir: Path,
    smplx_dir: Path,
    start_frame: int,
    frames: int,
    cameras: int,
    full: bool,
) -> Dict[str, Any]:
    scene = _json_schema(raw_dir / "scene.json", ("num_frames",))
    mesh = _file(raw_dir / "meshes.abc", minimum_bytes=1024)
    calibration = _file(raw_dir / "4x" / "calibration.csv", minimum_bytes=128)
    expected_frames = range(start_frame, start_frame + frames)
    camera_ids = range(1, cameras + 1)
    if full:
        pairs = [(camera, frame) for camera in camera_ids for frame in expected_frames]
    else:
        pairs = [
            (camera, frame)
            for camera in (1, cameras)
            for frame in (start_frame, start_frame + frames - 1)
        ]
    rgb_paths = []
    mask_paths = []
    for camera, frame in pairs:
        camera_name = f"Cam{camera:03d}"
        rgb = (
            raw_dir
            / "4x"
            / "rgbs"
            / camera_name
            / f"{camera_name}_rgb{frame:06d}.jpg"
        )
        mask = (
            raw_dir
            / "4x"
            / "masks"
            / camera_name
            / f"{camera_name}_mask{frame:06d}.png"
        )
        rgb_paths.append(rgb)
        mask_paths.append(mask)
    if full:
        missing_rgb = _missing_paths(rgb_paths)
        missing_masks = _missing_paths(mask_paths)
    else:
        missing_rgb = sum(not path.is_file() for path in rgb_paths)
        missing_masks = sum(not path.is_file() for path in mask_paths)
    pose_records = []
    for frame in expected_frames:
        pose_records.append(
            _json_schema(
                smplx_dir / f"{frame:06d}.json",
                ("poses", "Rh", "Th"),
            )
        )
    ready_poses = sum(record["ready"] for record in pose_records)
    record = {
        "scene": scene,
        "mesh": mesh,
        "calibration": calibration,
        "checked_raw_pairs": len(pairs),
        "missing_rgb": missing_rgb,
        "missing_masks": missing_masks,
        "smplx_frames": frames,
        "smplx_ready": ready_poses,
    }
    record["ready"] = (
        scene["ready"]
        and mesh["ready"]
        and calibration["ready"]
        and missing_rgb == 0
        and missing_masks == 0
        and ready_poses == frames
    )
    return record


def _dna_raw(raw_path: Path, annotation_path: Path) -> Dict[str, Any]:
    raw = _file(raw_path, minimum_bytes=1024 * 1024)
    annotation = _file(annotation_path, minimum_bytes=1024 * 1024)
    return {
        "raw_smc": raw,
        "annotation_smc": annotation,
        "ready": raw["ready"] and annotation["ready"],
    }


def _runtime(config: Dict[str, Any]) -> Dict[str, Any]:
    roots = config["roots"]
    runtime = config.get("runtime", {})
    python = Path(runtime.get("python", ""))
    records = {
        "smpl_model": _file(Path(roots["smpl_model"]), minimum_bytes=1024 * 1024),
        "dna_reader": _file(Path(roots["dna_reader"]) / "SMCReader.py"),
        "scripts": {},
        "imports": {},
    }
    for script in ("helpers.py", "external.py"):
        records["scripts"][script] = _file(Path(roots["d3dgs"]) / script)

    experiments_root = Path(__file__).resolve().parents[1]
    for script in (
        "backends/d3dgs/train_dna.py",
        "backends/d3dgs/eval_lbs_metrics.py",
        "multiview_gaussian/human_lbs.py",
        "multiview_gaussian/human_bob.py",
        "reproductions/dressrecon_bob/fit_bob.py",
    ):
        records["scripts"][script] = _file(experiments_root / script)

    environment = os.environ.copy()
    pythonpath = [roots["dressrecon"]]
    if environment.get("PYTHONPATH"):
        pythonpath.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(pythonpath)
    for name, modules in runtime.get("imports", {}).items():
        record = {"modules": list(modules), "ready": False}
        if not python.is_file():
            record["error"] = f"python executable not found: {python}"
        else:
            source = "\n".join(f"import {module}" for module in modules)
            completed = subprocess.run(
                [str(python), "-c", source],
                cwd=roots["d3dgs"],
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            record.update(
                ready=completed.returncode == 0,
                returncode=completed.returncode,
            )
            if completed.returncode:
                record["output"] = completed.stdout[-4000:]
        records["imports"][name] = record
    records["python"] = {
        "path": str(python),
        "ready": python.is_file() and os.access(python, os.X_OK),
    }
    records["ready"] = (
        records["python"]["ready"]
        and records["smpl_model"]["ready"]
        and records["dna_reader"]["ready"]
        and all(item["ready"] for item in records["scripts"].values())
        and all(item["ready"] for item in records["imports"].values())
    )
    return records


def _prediction_files(
    directory: Path,
    expected_frames: int,
    extra_required: Iterable[str],
    full: bool,
) -> Dict[str, Any]:
    """Compatibility alias for the canonical prediction validator."""

    return validate_prediction_artifact(
        directory,
        expected_frames,
        extra_required=extra_required,
        require_test_prediction=True,
        full=full,
    )


def _ssdr_files(path: Path, expected_frames: int) -> Dict[str, Any]:
    """Compatibility alias for the canonical SSDR cache validator."""

    return validate_ssdr_cache(path, expected_frames)


def _derived_files(
    dataset_id: str,
    case_id: str,
    expected_frames: int,
    roots: Mapping[str, str],
    full: bool,
    expected_points: Optional[int] = None,
    bob_protocol: Optional[Dict[str, Any]] = None,
    partmm_protocol: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    case_root = Path(roots["runs"]) / dataset_id / case_id
    lbs = validate_method_prediction(
        case_root / "smpl_lbs" / "lbs",
        expected_frames,
        "lbs",
        expected_points=expected_points,
        full=full,
    )
    bob = validate_method_prediction(
        case_root / "smpl_bob" / "bob" / case_id,
        expected_frames,
        "bob",
        expected_points=expected_points,
        expected_protocol=bob_protocol,
        full=full,
    )
    partmm = validate_method_prediction(
        case_root / "smpl_partmm" / "partmm" / case_id,
        expected_frames,
        "partmm",
        expected_points=expected_points,
        expected_protocol=partmm_protocol,
        full=full,
    )
    ssdr = validate_ssdr_cache(
        case_root / "smpl_partmm" / "ssdr" / "train_only.npz",
        expected_frames,
        full=full,
    )
    return {
        "lbs": lbs,
        "bob": bob,
        "ssdr": ssdr,
        "partmm": partmm,
    }


def audit_resources(
    config: Dict[str, Any],
    full: bool = False,
    selected_cases: Iterable[str] = (),
) -> Dict[str, Any]:
    roots = config["roots"]
    selected = set(selected_cases)
    known_cases = {
        case["id"]
        for dataset in config["datasets"]
        for case in dataset["cases"]
    }
    unknown = selected - known_cases
    if unknown:
        raise ValueError(f"unknown multiview cases: {', '.join(sorted(unknown))}")
    cases: Dict[str, Any] = {}
    for dataset in config["datasets"]:
        dataset_id = dataset["id"]
        expected_frames = int(dataset["expected_frames"])
        expected_cameras = int(dataset["expected_cameras"])
        params_root = Path(
            roots["dna_params"] if dataset_id == "dna_rendering" else roots["actorshq_params"]
        )
        for case in dataset["cases"]:
            case_id = case["id"]
            if selected and case_id not in selected:
                continue
            data_record = _metadata(
                Path(roots["data"]) / case["data"],
                expected_frames,
                expected_cameras,
                full,
            )
            if dataset_id == "dna_rendering":
                raw_record = _dna_raw(
                    Path(roots["dna_raw"]) / case["raw"],
                    Path(roots["dna_annotations"]) / case["annotation"],
                )
            else:
                raw_record = _actorshq_raw(
                    Path(roots["actorshq_raw"]) / case["raw"],
                    Path(roots["actorshq_smplx"]) / case_id,
                    int(case["start_frame"]),
                    expected_frames,
                    expected_cameras,
                    full,
                )
            params_record = _params(
                params_root / case["params"] / "params.npz",
                expected_frames,
            )
            raw_ready = raw_record["ready"] and data_record["ready"]
            cases[case_id] = {
                "dataset": dataset_id,
                "raw": raw_record,
                "prepared_data": data_record,
                "params": params_record,
                "derived": _derived_files(
                    dataset_id,
                    case_id,
                    expected_frames,
                    roots,
                    full,
                    expected_points=(
                        int(params_record["means3d_shape"][1])
                        if params_record["ready"]
                        else None
                    ),
                    bob_protocol=config.get(
                        "derived_protocols", {}
                    ).get("bob"),
                    partmm_protocol=config.get(
                        "derived_protocols", {}
                    ).get("partmm"),
                ),
                "raw_ready": raw_ready,
                "reconstruction_ready": raw_ready,
                "novel_pose_base_ready": raw_ready and params_record["ready"],
            }

    by_dataset = {}
    for dataset in config["datasets"]:
        dataset_id = dataset["id"]
        subset = [record for record in cases.values() if record["dataset"] == dataset_id]
        if not subset:
            continue
        by_dataset[dataset_id] = {
            "cases": len(subset),
            "raw_ready": sum(record["raw_ready"] for record in subset),
            "params_ready": sum(record["params"]["ready"] for record in subset),
            "novel_pose_base_ready": sum(
                record["novel_pose_base_ready"] for record in subset
            ),
            "lbs_ready": sum(record["derived"]["lbs"]["ready"] for record in subset),
            "bob_ready": sum(record["derived"]["bob"]["ready"] for record in subset),
            "ssdr_ready": sum(record["derived"]["ssdr"]["ready"] for record in subset),
            "partmm_ready": sum(
                record["derived"]["partmm"]["ready"] for record in subset
            ),
        }
    link_state = {
        "d3dgs_data": _link_state(
            Path(roots["data"]), Path(roots["dna_raw"])
        ),
        "actorshq_raw": _link_state(
            Path(roots["actorshq_raw_alias"]), Path(roots["actorshq_raw"])
        ),
        "actorshq_smplx_legacy": _link_state(
            Path(roots["actorshq_smplx_legacy"]),
            Path(roots["actorshq_smplx"]),
        ),
    }
    for dataset in config["datasets"]:
        if dataset["id"] != "actorshq":
            continue
        for case in dataset["cases"]:
            if selected and case["id"] not in selected:
                continue
            link_state[f"smplx/{case['id']}"] = _link_state(
                Path(roots["actorshq_smplx"]) / case["id"],
                Path(roots["actorshq_raw"]) / case["raw"] / "smplx",
            )
    for source_key in ("dna_ssdr_source", "actorshq_ssdr_source"):
        for source in sorted(Path(roots[source_key]).glob("*.npz")):
            link_state[f"ssdr/{source.name}"] = _link_state(
                Path(roots["ssdr"]) / source.name,
                source,
            )
    runtime = _runtime(config)
    derived_stages = ("lbs", "bob", "ssdr", "partmm")
    derived_ready = {
        stage: all(
            record["derived"][stage]["ready"] for record in cases.values()
        )
        for stage in derived_stages
    }
    return {
        "schema_version": 1,
        "resource_set": config["name"],
        "verification": "full" if full else "sampled",
        "roots": roots,
        "cases": cases,
        "by_dataset": by_dataset,
        "links": link_state,
        "runtime": runtime,
        "all_links_ready": all(record["ready"] for record in link_state.values()),
        "all_runtime_ready": runtime["ready"],
        "all_raw_ready": all(record["raw_ready"] for record in cases.values()),
        "all_novel_pose_base_ready": all(
            record["novel_pose_base_ready"] for record in cases.values()
        ),
        "all_derived_ready": all(derived_ready.values()),
        "derived_ready": derived_ready,
    }


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit multiview experiment resources")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("resources.yaml"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--prepare-links", action="store_true")
    parser.add_argument("--require-raw", action="store_true")
    parser.add_argument("--require-runtime", action="store_true")
    parser.add_argument("--require-novel-pose-base", action="store_true")
    parser.add_argument("--require-derived", action="store_true")
    args = parser.parse_args()
    config = load_resource_config(args.config)
    if args.prepare_links:
        prepared = prepare_resource_links(config)
        if not prepared["all_ready"]:
            print(json.dumps(prepared, indent=2, sort_keys=True))
            raise SystemExit(4)
    result = audit_resources(
        config,
        full=args.full,
        selected_cases=args.case,
    )
    if args.output:
        _write(args.output, result)
    print(json.dumps(result["by_dataset"], indent=2, sort_keys=True))
    if args.require_raw and not result["all_raw_ready"]:
        raise SystemExit(2)
    if args.require_runtime and not result["all_runtime_ready"]:
        raise SystemExit(5)
    if args.require_novel_pose_base and not result["all_novel_pose_base_ready"]:
        raise SystemExit(3)
    if args.require_derived and not result["all_derived_ready"]:
        raise SystemExit(6)


if __name__ == "__main__":
    main()
