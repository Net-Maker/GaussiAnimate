"""Reproducible skelebone + PartMM runner over monocular GaussianAnimator tracks.

Stages per scene (input ``dnerf_{case}_gaussiananimator``, output ``{case}``):
  prepare  registered ``pointcloud_t*.ply`` frames -> ``sequence.npz``
  extract  ``SkelebonePipeline`` -> ``result/skelebones.npz``
  animate  real PartMM source replay plus a deterministic 15-degree sinusoid
           edit on a non-root, non-leaf joint -> ``source_animation.npz`` and
           ``animation.npz``

Existing complete stage outputs are reused unless ``--force``. Case failures
are collected into ``summary.json`` under the output root and the process
exits non-zero; failures are never masked by partial results. Each animation
NPZ stores both PartMM query fields (``local_rotations``, ``root_translation``,
``fps``) and output fields (``points``, ``bone_transforms``,
``joint_positions``, ``matched_*``) with the exact key names used by
``partmm.load_query`` / ``partmm.load_output``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (REPO_ROOT, REPO_ROOT / ".native" / "python"):
    if _path.is_dir() and str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from partmm import PartMMConfig, PartMMInput, PartMMQuery, animate
from skelebone import (
    ExtractionConfig,
    SkelebonePipeline,
    load_bundle,
    load_point_sequence,
)
from skelebone.kinematics import forward_kinematics, hierarchy_order
from skelebone.pointcloud import load_registered_clouds, save_registered_clouds

STAGES = ("prepare", "extract", "animate")
SUMMARY_NAME = "summary.json"
METRICS_NAME = "metrics.json"
MANIFEST_NAME = "source_manifest.json"
MANIFEST_REQUIRED_FIELDS = ("source_ply", "iteration", "frames", "points")
EXTRACTION_FIELDS = (
    "num_bones",
    "ssdr_max_iterations",
    "mesh_poisson_depth",
    "mesh_target_triangles",
)
STAGE_OUTPUTS = {
    "prepare": {"sequence": "sequence.npz", "source_manifest": MANIFEST_NAME},
    "extract": {
        "skelebones": "result/skelebones.npz",
        "pipeline_manifest": "result/manifest.json",
    },
    "animate": {
        "source_animation": "source_animation.npz",
        "animation": "animation.npz",
    },
}
ANIMATION_KEYS = (
    "local_rotations",
    "root_translation",
    "fps",
    "points",
    "bone_transforms",
    "joint_positions",
)
BONE_LENGTH_RTOL = 1e-6
EDIT_AXIS = (0.0, 0.0, 1.0)
EDIT_CYCLES = 1.0


class RunnerError(RuntimeError):
    """A case-level failure that must be reported, never masked."""


# ---------------------------------------------------------------------------
# configuration


def load_config(path):
    """Load the experiment YAML and recursively expand ``${GA_VAR}`` values."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RunnerError(f"config {path} must be a YAML mapping")
    config = _expand_env(raw)
    _validate_config(config)
    return config


def _expand_env(value):
    if isinstance(value, str):
        previous = None
        while previous != value:
            previous = value
            value = os.path.expandvars(value)
        return value
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    return value


def _validate_config(config):
    mesh = config.get("mesh", {})
    if mesh.get("method", "poisson") not in ("tsdf", "poisson"):
        raise RunnerError("mesh.method must be tsdf or poisson")
    if mesh.get("method") == "tsdf" and not mesh.get("filename"):
        raise RunnerError("mesh.filename is required for TSDF extraction")
    scenes = config.get("scenes")
    if (
        not isinstance(scenes, list)
        or not scenes
        or not all(isinstance(name, str) and name for name in scenes)
    ):
        raise RunnerError("config must list at least one scene name")
    if len(set(scenes)) != len(scenes):
        raise RunnerError("config scenes must be unique")
    paths = config.get("paths")
    if not isinstance(paths, dict):
        raise RunnerError("config must define a 'paths' mapping")
    for key in ("point_track_root", "output_root", "folder_pattern"):
        value = paths.get(key)
        if not isinstance(value, str) or not value or "${" in value:
            raise RunnerError(
                f"paths.{key} must resolve to a concrete value "
                "(is its $GA_ environment variable set?)"
            )
    extraction = config.get("extraction")
    if not isinstance(extraction, dict):
        raise RunnerError("config must define an 'extraction' mapping")
    for key in EXTRACTION_FIELDS:
        value = extraction.get(key)
        if not isinstance(value, (int, float)) or value < 1:
            raise RunnerError(f"extraction.{key} must be a positive number")
    limit = extraction.get("max_normalized_rmse")
    if not isinstance(limit, (int, float)) or not np.isfinite(limit) or limit <= 0:
        raise RunnerError("extraction.max_normalized_rmse must be a positive number")
    animation = config.get("animation") or {}
    amplitude = animation.get("amplitude_deg")
    if (
        not isinstance(amplitude, (int, float))
        or not np.isfinite(amplitude)
        or not 0 < amplitude < 180
    ):
        raise RunnerError("animation.amplitude_deg must be within (0, 180)")
    if not isinstance(config.get("seed"), int) or config["seed"] < 0:
        raise RunnerError("seed must be a non-negative integer")
    fps = config.get("fps")
    if not isinstance(fps, (int, float)) or not np.isfinite(fps) or fps <= 0:
        raise RunnerError("fps must be finite and positive")
    max_points = config.get("max_points")
    if max_points is not None and (
        not isinstance(max_points, int) or max_points < 1
    ):
        raise RunnerError("max_points must be a positive integer or null")
    min_opacity = config.get("min_opacity")
    if (
        not isinstance(min_opacity, (int, float))
        or not np.isfinite(min_opacity)
        or not 0 <= min_opacity < 1
    ):
        raise RunnerError("min_opacity must be within [0, 1)")
    if not isinstance(config.get("pointcloud_pattern"), str) or not config["pointcloud_pattern"]:
        raise RunnerError("pointcloud_pattern must be a non-empty string")


def case_paths(config, case):
    """Return (input point-track dir, output dir) for one scene."""
    folder = config["paths"]["folder_pattern"].format(case=case)
    return (
        Path(config["paths"]["point_track_root"]) / folder,
        Path(config["paths"]["output_root"]) / case,
    )


# ---------------------------------------------------------------------------
# shared validation helpers


def _require_finite(**arrays):
    for name, array in arrays.items():
        array = np.asarray(array)
        if array.size and not np.isfinite(array).all():
            raise RunnerError(f"{name} contains nonfinite values")


def _bbox_diagonal(points):
    points = np.asarray(points, dtype=np.float64)
    diagonal = float(np.linalg.norm(np.ptp(points, axis=0)))
    if not np.isfinite(diagonal) or diagonal <= 0:
        raise RunnerError(
            "rest points have a degenerate bounding box; cannot normalize SSDR rmse"
        )
    return diagonal


def _check_fk_bone_lengths(skeleton, pose):
    """Bone lengths from forward kinematics must stay constant over the clip."""
    _, positions = forward_kinematics(skeleton, pose)
    parents = np.asarray(skeleton.parents, dtype=np.int64)
    child = parents >= 0
    lengths = np.linalg.norm(
        positions[:, child] - positions[:, parents[child]], axis=-1
    )
    if not lengths.size:
        return
    spread = float(np.max(lengths.max(axis=0) - lengths.min(axis=0)))
    tolerance = BONE_LENGTH_RTOL * max(float(lengths.max()), 1.0)
    if not np.isfinite(lengths).all() or spread > tolerance:
        raise RunnerError(
            f"FK bone lengths vary by {spread:.3g} over the clip; "
            "the rig must keep constant bone lengths"
        )


def _validate_bundle(bundle):
    """Skin weights, finiteness, and FK consistency of an extracted rig."""
    _require_finite(weights=bundle.weights, source_points=bundle.source_points)
    row_sums = bundle.weights.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        raise RunnerError("skin weights are not normalized to unit row sums")
    if bundle.motion is None or bundle.bone_transforms is None:
        raise RunnerError(
            "bundle lacks source motion or bone transforms; PartMM cannot run"
        )
    _require_finite(motion=bundle.motion.local_rotations)
    _check_fk_bone_lengths(bundle.skeleton, bundle.motion)


# ---------------------------------------------------------------------------
# novel edit query


def _select_edit_joint(parents):
    """Deterministic non-root, non-leaf joint with the largest subtree."""
    parents = np.asarray(parents, dtype=np.int64)
    sizes = np.ones(len(parents), dtype=np.int64)
    for joint in reversed(hierarchy_order(parents)):
        parent = int(parents[joint])
        if parent >= 0:
            sizes[parent] += sizes[joint]
    has_child = np.zeros(len(parents), dtype=bool)
    child_mask = parents >= 0
    has_child[parents[child_mask]] = True
    candidates = [
        joint for joint in range(len(parents)) if parents[joint] >= 0 and has_child[joint]
    ]
    if not candidates:
        raise RunnerError("extracted skeleton has no non-root, non-leaf joint to edit")
    return int(max(candidates, key=lambda joint: (int(sizes[joint]), -joint)))


def _axis_angle_quaternions(axis, angles):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    half = 0.5 * np.asarray(angles, dtype=np.float64)
    quaternions = np.zeros((len(half), 4), dtype=np.float64)
    quaternions[:, 0] = np.cos(half)
    quaternions[:, 1:] = np.sin(half)[:, None] * axis
    return quaternions


def _quaternion_multiply(left, right):
    """Scalar-first quaternion product."""
    lw, lx, ly, lz = np.moveaxis(np.asarray(left, dtype=np.float64), -1, 0)
    rw, rx, ry, rz = np.moveaxis(np.asarray(right, dtype=np.float64), -1, 0)
    return np.stack(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        axis=-1,
    )


def _build_edit_query(partmm_input, joint, amplitude_deg):
    """Sinusoidal rotation of one joint; every other joint keeps the source motion.

    The edit is composed onto the source local rotation, so the relative joint
    angle over the clip is exactly ``amplitude_deg * sin(2*pi*t/T)``.
    """
    motion = partmm_input.motion
    frames = motion.num_frames
    angles = np.deg2rad(float(amplitude_deg)) * np.sin(
        2.0 * np.pi * EDIT_CYCLES * np.arange(frames) / frames
    )
    edit = _axis_angle_quaternions(EDIT_AXIS, angles)
    rotations = np.array(motion.local_rotations, dtype=np.float64, copy=True)
    rotations[:, joint, :] = _quaternion_multiply(rotations[:, joint, :], edit)
    return PartMMQuery(rotations, np.array(motion.root_translation, copy=True), motion.fps)


# ---------------------------------------------------------------------------
# stage: prepare


def _load_source_manifest(input_dir, expected_frames, pattern):
    """Read the dataset provenance manifest; it is required for reproducibility."""
    path = input_dir / MANIFEST_NAME
    if not path.is_file():
        raise RunnerError(f"missing {MANIFEST_NAME} under {input_dir}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RunnerError(f"{path} is not valid JSON") from exc
    if not isinstance(manifest, dict):
        raise RunnerError(f"{path} must contain a JSON object")
    missing = [field for field in MANIFEST_REQUIRED_FIELDS if field not in manifest]
    if missing:
        raise RunnerError(f"{path} is missing required fields {missing}")
    try:
        recorded_frames = int(manifest["frames"])
    except (TypeError, ValueError) as exc:
        raise RunnerError(f"{path} records a non-integer frame count") from exc
    if recorded_frames != expected_frames:
        raise RunnerError(
            f"{path} records {recorded_frames} frames but {expected_frames} "
            f"inputs match {pattern!r}"
        )
    return manifest


def _prepare_info_from_npz(path):
    with np.load(path, allow_pickle=False) as data:
        points = data["points"]
        info = {"frames": int(points.shape[0]), "points": int(points.shape[1])}
        if "source_point_count" in data:
            info["native_input_count"] = int(
                np.asarray(data["source_point_count"]).reshape(-1)[0]
            )
        if "fps" in data:
            info["fps"] = float(np.asarray(data["fps"]).reshape(-1)[0])
    return info


def run_prepare(config, case, input_dir, output_dir, force):
    """Stage 1: import registered point-cloud frames into ``sequence.npz``."""
    pattern = config["pointcloud_pattern"]
    frame_paths = sorted(input_dir.glob(pattern))
    if not frame_paths:
        raise RunnerError(f"no inputs matching {pattern!r} under {input_dir}")
    manifest = _load_source_manifest(input_dir, len(frame_paths), pattern)
    _write_json(output_dir / MANIFEST_NAME, manifest)
    sequence_path = output_dir / "sequence.npz"
    if sequence_path.is_file() and not force:
        info = _prepare_info_from_npz(sequence_path)
        info.update(source_manifest=manifest, outputs=dict(STAGE_OUTPUTS["prepare"]))
        return info
    gaussian_ply = Path(manifest["source_ply"])
    if not gaussian_ply.is_absolute():
        gaussian_ply = input_dir / gaussian_ply
    sequence, attributes = load_registered_clouds(
        input_dir,
        pattern=pattern,
        max_points=config["max_points"],
        seed=config["seed"],
        fps=config["fps"],
        min_opacity=config["min_opacity"],
        gaussian_ply=gaussian_ply,
    )
    if manifest.get("use_isotropic_gs") and "scales" in sequence.point_attributes:
        scales = np.repeat(sequence.point_attributes["scales"][:, :1], 3, axis=1)
        sequence.point_attributes["scales"] = attributes["scales"] = scales
    _require_finite(points=sequence.points)
    save_registered_clouds(sequence_path, sequence, attributes)
    return {
        "frames": int(sequence.num_frames),
        "points": int(sequence.num_points),
        "native_input_count": int(
            np.asarray(attributes["source_point_count"]).reshape(-1)[0]
        ),
        "fps": float(sequence.fps),
        "source_manifest": manifest,
        "outputs": dict(STAGE_OUTPUTS["prepare"]),
    }


# ---------------------------------------------------------------------------
# stage: extract


def _extract_rmse(manifest_path):
    """Raw SSDR rmse recorded by the pipeline (sqrt of mean squared error)."""
    try:
        data = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        return float(data["ssdr"]["rmse"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise RunnerError(
            f"{manifest_path} does not record an SSDR rmse: {exc}"
        ) from exc


def _extract_info(bundle, raw_rmse, config):
    raw_rmse = float(raw_rmse)
    scale = _bbox_diagonal(bundle.source_points)
    normalized = raw_rmse / scale
    limit = float(config["extraction"]["max_normalized_rmse"])
    if not np.isfinite(normalized) or normalized > limit:
        raise RunnerError(
            f"normalized SSDR rmse {normalized:.6g} exceeds "
            f"max_normalized_rmse {limit:.6g}"
        )
    _validate_bundle(bundle)
    return {
        "bones": int(bundle.num_bones),
        "joints": int(bundle.skeleton.num_joints),
        "raw_rmse": raw_rmse,
        "normalization_scale": scale,
        "normalized_rmse": normalized,
        "max_normalized_rmse": limit,
        "outputs": dict(STAGE_OUTPUTS["extract"]),
    }


def _canonical_mesh(config, input_dir):
    settings = config.get("mesh", {})
    if settings.get("method", "poisson") == "poisson":
        return None
    path = Path(input_dir) / settings["filename"]
    if not path.is_file():
        raise RunnerError(
            f"missing TSDF mesh {path}; export it with "
            "python -m experiments.monocular_gaussian.tsdf MODEL OUTPUT"
        )
    return path


def run_extract(config, case, input_dir, output_dir, force):
    """Stage 2: extract the skelebone rig into ``result/skelebones.npz``."""
    sequence_path = output_dir / "sequence.npz"
    result_dir = output_dir / "result"
    bundle_path = result_dir / "skelebones.npz"
    manifest_path = result_dir / "manifest.json"
    if not sequence_path.is_file():
        raise RunnerError(f"missing {sequence_path}; run the prepare stage first")
    mesh = _canonical_mesh(config, input_dir)
    if bundle_path.is_file() and manifest_path.is_file() and not force:
        if mesh is not None:
            surface = json.loads(manifest_path.read_text()).get("surface", {})
            if (surface.get("source") != "tsdf"
                    or surface.get("canonical_mesh") != str(mesh.resolve())
                    or mesh.stat().st_mtime_ns > bundle_path.stat().st_mtime_ns):
                raise RunnerError("cached rig does not match the TSDF mesh; use a new output root or --force")
        bundle = load_bundle(bundle_path)
        return _extract_info(bundle, _extract_rmse(manifest_path), config)
    sequence = load_point_sequence(sequence_path)
    settings = {key: config["extraction"][key] for key in EXTRACTION_FIELDS}
    pipeline = SkelebonePipeline(ExtractionConfig(**settings))
    if mesh is None:
        result = pipeline.run(sequence, result_dir)
    else:
        result = pipeline.run(sequence, result_dir, mesh_for_skeleton=mesh)
        manifest = json.loads(result.manifest_path.read_text())
        manifest["surface"].update(source="tsdf", canonical_mesh=str(mesh.resolve()))
        _write_json(result.manifest_path, manifest)
    return _extract_info(result.bundle, _extract_rmse(result.manifest_path), config)


# ---------------------------------------------------------------------------
# stage: animate


def _pack_matches(matches):
    part_ids = np.asarray(sorted(matches), dtype=np.int64)
    if not len(part_ids):
        return part_ids, np.empty((0, 0, 0), dtype=np.int64)
    packed = [np.asarray(matches[int(part)], dtype=np.int64) for part in part_ids]
    return part_ids, np.stack(packed)


def _save_animation(path, query, output):
    """Write query + output into one NPZ using partmm's exact key names."""
    part_ids, matches = _pack_matches(output.matched_indices)
    np.savez_compressed(
        path,
        partmm_query_schema_version=np.asarray(1),
        partmm_output_schema_version=np.asarray(1),
        local_rotations=query.local_rotations,
        root_translation=query.root_translation,
        points=output.points,
        bone_transforms=output.bone_transforms,
        joint_positions=output.joint_positions,
        matched_part_ids=part_ids,
        matched_source_frames=matches,
        fps=np.asarray(query.fps),
    )
    return path


def _validate_output(output):
    _require_finite(
        points=output.points,
        bone_transforms=output.bone_transforms,
        joint_positions=output.joint_positions,
    )
    if len(output.points) < 1:
        raise RunnerError("animation produced no frames")


def _require_animation_npz(path):
    with np.load(path, allow_pickle=False) as data:
        missing = [key for key in ANIMATION_KEYS if key not in data]
        if missing:
            raise RunnerError(
                f"{path} is missing animation fields {missing}; rerun with --force"
            )
        _require_finite(**{key: data[key] for key in ANIMATION_KEYS})


def _animation_info(skeleton, joint, frames, config):
    return {
        "edit_joint": str(skeleton.names[joint]),
        "edit_joint_index": int(joint),
        "amplitude_deg": float(config["animation"]["amplitude_deg"]),
        "frames": int(frames),
        "outputs": dict(STAGE_OUTPUTS["animate"]),
    }


def run_animate(config, case, input_dir, output_dir, force):
    """Stage 3: real PartMM source replay plus a deterministic sinusoid edit."""
    bundle_path = output_dir / "result" / "skelebones.npz"
    source_path = output_dir / "source_animation.npz"
    animation_path = output_dir / "animation.npz"
    if not bundle_path.is_file():
        raise RunnerError(f"missing {bundle_path}; run the extract stage first")
    if source_path.is_file() and animation_path.is_file() and not force:
        _require_animation_npz(source_path)
        _require_animation_npz(animation_path)
        bundle = load_bundle(bundle_path)
        with np.load(animation_path, allow_pickle=False) as data:
            frames = int(data["local_rotations"].shape[0])
        joint = _select_edit_joint(bundle.skeleton.parents)
        return _animation_info(bundle.skeleton, joint, frames, config)
    bundle = load_bundle(bundle_path)
    _validate_bundle(bundle)
    partmm_input = PartMMInput.from_bundle(bundle)
    partmm_config = PartMMConfig()
    source_query = PartMMQuery.from_pose(partmm_input.motion)
    source_output = animate(partmm_input, source_query, partmm_config)
    _validate_output(source_output)
    joint = _select_edit_joint(partmm_input.parents)
    query = _build_edit_query(partmm_input, joint, config["animation"]["amplitude_deg"])
    _check_fk_bone_lengths(partmm_input.skeleton, query)
    edit_output = animate(partmm_input, query, partmm_config)
    _validate_output(edit_output)
    delta = float(np.max(np.linalg.norm(edit_output.points - source_output.points, axis=-1)))
    if delta <= 1e-8:
        raise RunnerError("joint edit did not change the animated points")
    _save_animation(source_path, source_query, source_output)
    _save_animation(animation_path, query, edit_output)
    info = _animation_info(partmm_input.skeleton, joint, query.num_frames, config)
    sequence = load_point_sequence(output_dir / "sequence.npz")
    info["source_replay_rmse"] = float(np.sqrt(np.mean((source_output.points - sequence.points) ** 2)))
    info["edit_max_displacement"] = delta
    return info


# ---------------------------------------------------------------------------
# orchestration


def _config_echo(config):
    return {
        "seed": config["seed"],
        "fps": config["fps"],
        "max_points": config["max_points"],
        "min_opacity": config["min_opacity"],
        "pointcloud_pattern": config["pointcloud_pattern"],
        "extraction": dict(config["extraction"]),
        "mesh": dict(config.get("mesh", {"method": "poisson"})),
        "animation": dict(config["animation"]),
        "paths": dict(config["paths"]),
    }


def _write_json(path, payload):
    path = Path(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _load_metrics(output_dir):
    path = output_dir / METRICS_NAME
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def run_case(config, case, stages, force):
    """Run the requested stages for one scene and update its metrics.json."""
    input_dir, output_dir = case_paths(config, case)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = _load_metrics(output_dir)
    metrics["case"] = case
    metrics["seed"] = config["seed"]
    metrics["config"] = _config_echo(config)
    if "prepare" in stages:
        metrics["prepare"] = run_prepare(config, case, input_dir, output_dir, force)
    if "extract" in stages:
        metrics["extract"] = run_extract(config, case, input_dir, output_dir, force)
    if "animate" in stages:
        metrics["animate"] = run_animate(config, case, input_dir, output_dir, force)
    metrics["source_manifest"] = (metrics.get("prepare") or {}).get("source_manifest")
    _write_json(output_dir / METRICS_NAME, metrics)
    return metrics


def _pass_summary(record):
    parts = []
    prepare = record.get("prepare") or {}
    if prepare:
        parts.append(f"frames={prepare.get('frames')} points={prepare.get('points')}")
    extract = record.get("extract") or {}
    if extract:
        parts.append(
            f"bones={extract.get('bones')} joints={extract.get('joints')} "
            f"normalized_rmse={float(extract.get('normalized_rmse', 0.0)):.4g}"
        )
    animate = record.get("animate") or {}
    if animate:
        parts.append(f"edit_joint={animate.get('edit_joint')}")
    return ", ".join(parts)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Reproducible skelebone extraction and PartMM animation runner"
    )
    parser.add_argument(
        "--config", required=True, type=Path, help="path to the experiment YAML config"
    )
    parser.add_argument(
        "--case",
        action="append",
        default=None,
        metavar="NAME",
        help="scene to process; repeatable (default: every configured scene)",
    )
    parser.add_argument(
        "--stage",
        choices=("prepare", "extract", "animate", "all"),
        default="all",
        help="pipeline stage to run (default: all)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="recompute stage outputs even when complete results already exist",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        config = load_config(args.config)
    except (RunnerError, OSError, yaml.YAMLError) as exc:
        print(f"[skelebone] CONFIG ERROR: {exc}", file=sys.stderr)
        return 2
    stages = STAGES if args.stage == "all" else (args.stage,)
    cases = list(args.case) if args.case else list(config["scenes"])
    unknown = [case for case in cases if case not in config["scenes"]]
    if unknown:
        print(
            f"[skelebone] CONFIG ERROR: unknown cases {unknown}; "
            f"configured scenes are {config['scenes']}",
            file=sys.stderr,
        )
        return 2
    output_root = Path(config["paths"]["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    results = {}
    for case in cases:
        try:
            record = run_case(config, case, stages, args.force)
            results[case] = {"status": "passed", "metrics": record}
            print(f"[skelebone] {case}: PASS ({_pass_summary(record)})")
        except Exception as exc:  # any failure is recorded, never masked
            results[case] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            print(f"[skelebone] {case}: FAIL ({exc})")
    summary = {
        "stage": args.stage,
        "force": bool(args.force),
        "cases": results,
        "passed": [case for case, result in results.items() if result["status"] == "passed"],
        "failed": [case for case, result in results.items() if result["status"] == "failed"],
    }
    summary_path = _write_json(output_root / SUMMARY_NAME, summary)
    print(f"[skelebone] summary: {summary_path}")
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
