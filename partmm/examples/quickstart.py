"""Run standalone PartMM on the HOOD/VTO T-shirt evaluation case."""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import pickle

import numpy as np

from partmm import (
    PartMMConfig,
    PartMMInput,
    PartMMOutput,
    PartMMQuery,
    animate,
    save_input,
    save_output,
    save_query,
)
from partmm.visualization import save_preview
from partmm.open3d_renderer import save_partmm_gif


DEFAULT_DATABASE_CASES = (
    "tshirt_shape00_01_01",
    "tshirt_shape00_02_04",
    "tshirt_shape00_02_05",
    "tshirt_shape00_05_20",
    "tshirt_shape00_07_02",
)
DEFAULT_QUERY_CASE = "tshirt_shape00_91_33"

SMPL_JOINT_NAMES = (
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee",
    "right_knee", "spine2", "left_ankle", "right_ankle", "spine3",
    "left_foot", "right_foot", "neck", "left_collar", "right_collar",
    "head", "left_shoulder", "right_shoulder", "left_elbow",
    "right_elbow", "left_wrist", "right_wrist", "left_hand", "right_hand",
)
SMPL_PARENTS = np.array(
    [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14,
     16, 17, 18, 19, 20, 21],
    dtype=np.int64,
)


def _require(path, description):
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"missing {description}: {path}")
    return path


def _load_trusted_pickle(path):
    with _require(path, "VTO simulation").open("rb") as handle:
        data = pickle.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"VTO simulation must contain a dictionary: {path}")
    return data


def axis_angle_to_quaternion(poses):
    """Convert `[frames, 72]` SMPL poses to scalar-first quaternions."""
    poses = np.asarray(poses, dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1] != 72:
        raise ValueError("VTO pose must have shape [frames, 72]")
    axis_angle = poses.reshape(len(poses), 24, 3).copy()
    axis_angle[:, 0] = 0.0  # VTO SSDR was computed in root-canonical space.
    angle = np.linalg.norm(axis_angle, axis=-1, keepdims=True)
    axis = np.divide(
        axis_angle,
        angle,
        out=np.zeros_like(axis_angle),
        where=angle > 1e-12,
    )
    quaternion = np.empty(axis_angle.shape[:-1] + (4,), dtype=np.float64)
    quaternion[..., :1] = np.cos(angle * 0.5)
    quaternion[..., 1:] = axis * np.sin(angle * 0.5)
    return quaternion


def load_hood_tshirt_case(
    hood_root,
    vto_root,
    query_case=DEFAULT_QUERY_CASE,
    database_cases=DEFAULT_DATABASE_CASES,
    fps=30.0,
):
    """Adapt trusted HOOD/VTO artifacts to the standalone PartMM contract."""
    hood_root = Path(hood_root).expanduser()
    vto_root = Path(vto_root).expanduser()
    ssdr_path = _require(
        hood_root / "output/vto_ssdr_results/tshirt_shape00_ALL_bones20.npz",
        "HOOD T-shirt SSDR database",
    )
    reference_path = _require(
        hood_root / "output/precomputed_tshirt" / f"{query_case}.npz",
        "HOOD T-shirt reference result",
    )
    simulation_root = vto_root / "tshirt" / "simulations"

    # This historical cache stores a dictionary as an object array. Only load
    # it from a trusted HOOD workspace.
    with np.load(ssdr_path, allow_pickle=True) as arrays:
        transforms_per_motion = arrays["transforms_per_motion"].item()
        source_points = np.asarray(arrays["rest_pose"], dtype=np.float64)
        stored_weights = np.asarray(arrays["weights"], dtype=np.float64)
        faces = np.asarray(arrays["faces"], dtype=np.int64)
        num_bones = int(arrays["num_bones"])
    weights = (
        stored_weights.T
        if stored_weights.shape == (num_bones, len(source_points))
        else stored_weights
    )
    if weights.shape != (len(source_points), num_bones):
        raise ValueError(
            "HOOD SSDR weights must have shape [bones, points] or [points, bones]"
        )

    source_rotations = []
    source_transforms = []
    for case in database_cases:
        if case not in transforms_per_motion:
            raise ValueError(f"HOOD SSDR database does not contain {case}")
        simulation = _load_trusted_pickle(simulation_root / f"{case}.pkl")
        if "pose" not in simulation:
            raise ValueError(f"VTO simulation is missing pose: {case}")
        rotations = axis_angle_to_quaternion(simulation["pose"])
        transforms = np.asarray(transforms_per_motion[case], dtype=np.float64)
        expected = (len(rotations), num_bones, 4, 4)
        if transforms.shape != expected:
            raise ValueError(
                f"SSDR transforms for {case} have shape {transforms.shape}, "
                f"expected {expected}"
            )
        source_rotations.append(rotations)
        source_transforms.append(transforms)

    query_simulation = _load_trusted_pickle(
        simulation_root / f"{query_case}.pkl"
    )
    if "pose" not in query_simulation:
        raise ValueError(f"VTO simulation is missing pose: {query_case}")
    query_rotations = axis_angle_to_quaternion(query_simulation["pose"])

    with np.load(reference_path, allow_pickle=False) as reference:
        required = {"skeleton_joints", "gt_vertices"}
        missing = required.difference(reference.files)
        if missing:
            raise ValueError(
                "HOOD reference result is missing: " + ", ".join(sorted(missing))
            )
        skeleton_joints = np.asarray(reference["skeleton_joints"], dtype=np.float64)
        target_points = np.asarray(reference["gt_vertices"], dtype=np.float64)
        historical_partmm_rmse = (
            float(reference["partmm_rmse"])
            if "partmm_rmse" in reference.files
            else None
        )
    if skeleton_joints.ndim != 3 or skeleton_joints.shape[1:] != (24, 3):
        raise ValueError("HOOD skeleton_joints must have shape [frames, 24, 3]")
    if target_points.shape != (len(query_rotations), len(source_points), 3):
        raise ValueError(
            "HOOD gt_vertices must match the query frames and SSDR point count"
        )

    database = PartMMInput(
        rest_joints=skeleton_joints[0],
        parents=SMPL_PARENTS,
        joint_names=SMPL_JOINT_NAMES,
        source_local_rotations=np.concatenate(source_rotations),
        source_root_translation=np.zeros((sum(map(len, source_rotations)), 3)),
        bone_transforms=np.concatenate(source_transforms),
        source_points=source_points,
        skinning_weights=weights,
        fps=fps,
    )
    query = PartMMQuery(
        query_rotations,
        np.zeros((len(query_rotations), 3)),
        fps,
    )
    return (
        database,
        query,
        target_points,
        skeleton_joints,
        faces,
        historical_partmm_rmse,
    )


def run(
    output_directory,
    hood_root,
    vto_root,
    query_case=DEFAULT_QUERY_CASE,
    database_cases=DEFAULT_DATABASE_CASES,
    config=PartMMConfig(partition_mode="humanoid_five"),
    fps=30.0,
    preview_format="gif",
    max_preview_frames=40,
    preview_fps=12.0,
):
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    (
        database,
        query,
        target_points,
        target_joints,
        faces,
        historical_partmm_rmse,
    ) = load_hood_tshirt_case(
        hood_root,
        vto_root,
        query_case=query_case,
        database_cases=tuple(database_cases),
        fps=fps,
    )
    retrieved = animate(database, query, config)
    result = PartMMOutput(
        points=retrieved.points,
        bone_transforms=retrieved.bone_transforms,
        joint_positions=target_joints,
        matched_indices=retrieved.matched_indices,
        fps=retrieved.fps,
    )
    rmse = float(np.sqrt(np.mean((result.points - target_points) ** 2)))

    preview_path = output_directory / f"partmm_preview.{preview_format}"
    if preview_format == "gif":
        preview_path = save_partmm_gif(
            preview_path,
            result,
            faces=faces,
            parents=database.parents,
            source_points=database.source_points,
            skinning_weights=database.skinning_weights,
            max_frames=max_preview_frames,
            fps=preview_fps,
        )
    elif preview_format == "png":
        preview_path = save_preview(
            preview_path,
            result,
            database.parents,
            frames=[0, query.num_frames // 2, query.num_frames - 1],
        )
    else:
        raise ValueError("preview_format must be 'gif' or 'png'")

    paths = {
        "input": save_input(output_directory / "partmm_input.npz", database),
        "query": save_query(output_directory / "partmm_query.npz", query),
        "output": save_output(output_directory / "partmm_output.npz", result),
        "preview": preview_path,
    }
    metrics = {
        "case": query_case,
        "database_cases": list(database_cases),
        "source_frames": database.motion.num_frames,
        "query_frames": query.num_frames,
        "points": len(database.source_points),
        "joints": database.skeleton.num_joints,
        "outer_bones": database.num_bones,
        "recomputed_rmse_native_units": rmse,
        "recomputed_rmse_mm": rmse * 1000.0,
        "historical_partmm_rmse_mm": historical_partmm_rmse,
        "config": asdict(config),
        "hood_root": str(Path(hood_root).expanduser()),
        "vto_root": str(Path(vto_root).expanduser()),
    }
    metrics_path = output_directory / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    paths["metrics"] = metrics_path
    return paths


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run standalone PartMM on HOOD/VTO T-shirt case "
            "tshirt_shape00_91_33."
        )
    )
    parser.add_argument("output", nargs="?", default="partmm_tshirt_demo")
    parser.add_argument(
        "--hood-root",
        default=os.environ.get("GA_HOOD_ROOT", "~/research/GA/hood"),
    )
    parser.add_argument(
        "--vto-root",
        default=os.environ.get("GA_VTO_ROOT", "~/research/data/vto-dataset"),
    )
    parser.add_argument("--case", default=DEFAULT_QUERY_CASE)
    parser.add_argument(
        "--database-cases", nargs="+", default=list(DEFAULT_DATABASE_CASES)
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--num-parts", type=int, default=5)
    parser.add_argument("--patch-size", type=int, default=7)
    parser.add_argument("--neighbors", type=int, default=7)
    parser.add_argument("--pyramid-levels", type=int, default=5)
    parser.add_argument("--pyramid-blend", type=float, default=0.7)
    parser.add_argument("--no-align", action="store_true")
    parser.add_argument(
        "--partition-mode",
        choices=("balanced_tree", "humanoid_five"),
        default="humanoid_five",
    )
    parser.add_argument(
        "--preview-format", choices=("gif", "png"), default="gif"
    )
    parser.add_argument("--max-preview-frames", type=int, default=40)
    parser.add_argument("--preview-fps", type=float, default=12.0)
    args = parser.parse_args()
    config = PartMMConfig(
        num_parts=args.num_parts,
        patch_size=args.patch_size,
        num_neighbors=args.neighbors,
        pyramid_levels=args.pyramid_levels,
        pyramid_blend=args.pyramid_blend,
        align_parts=not args.no_align,
        partition_mode=args.partition_mode,
    )
    paths = run(
        args.output,
        args.hood_root,
        args.vto_root,
        query_case=args.case,
        database_cases=args.database_cases,
        config=config,
        fps=args.fps,
        preview_format=args.preview_format,
        max_preview_frames=args.max_preview_frames,
        preview_fps=args.preview_fps,
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
