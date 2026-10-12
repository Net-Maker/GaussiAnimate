"""Migrate existing GaussiAnimate experiment caches into public bundles."""

import argparse
from pathlib import Path
from typing import Optional

import numpy as np

from skelebone.bvh import export_bvh
from skelebone.io import save_bundle
from skelebone.kinematics import matrix_to_quaternion
from skelebone.types import PoseSequence, Skeleton, SkeleboneBundle


def load_prediction_bundle(
    prediction_path,
    fps: float = 30.0,
    source_start: int = 0,
    source_end: Optional[int] = None,
) -> SkeleboneBundle:
    """Load the consolidated DT4D/VTO ``prediction_data.npz`` schema."""
    path = Path(prediction_path)
    data = np.load(path, allow_pickle=True)
    required = {
        "skeleton_rest_positions",
        "skeleton_parents",
        "skeleton_local_rotations",
        "skeleton_root_positions",
        "ssdr_weights",
        "ssdr_transformations",
        "ssdr_rest_pose",
    }
    missing = sorted(required - set(data.files))
    if missing:
        raise ValueError(f"prediction cache is missing: {', '.join(missing)}")
    skeleton = Skeleton(data["skeleton_rest_positions"], data["skeleton_parents"])
    root = int(np.flatnonzero(skeleton.parents == -1)[0])
    end = len(data["skeleton_local_rotations"]) if source_end is None else source_end
    frame_slice = slice(source_start, end)
    motion = PoseSequence(
        matrix_to_quaternion(data["skeleton_local_rotations"][frame_slice]),
        data["skeleton_root_positions"][frame_slice] - skeleton.rest_joints[root],
        fps=fps,
    )
    stored_weights = data["ssdr_weights"]
    if stored_weights.shape == () and stored_weights.dtype == object:
        stored_weights = stored_weights.item()
    weights = stored_weights.toarray() if hasattr(stored_weights, "toarray") else np.asarray(stored_weights)
    if weights.shape[0] != len(data["ssdr_rest_pose"]):
        weights = weights.T
    artifacts = {"legacy_prediction": str(path)}
    for name, relative in {
        "curve_skeleton": "curve_skeleton",
        "skinning_curve_skeleton": "curve_skeleton_skinning",
        "kinematic_tree": "inner_skeleton/inner_skeleton.npz",
    }.items():
        candidate = path.parent / relative
        if candidate.exists():
            artifacts[name] = str(candidate)
    return SkeleboneBundle(
        skeleton=skeleton,
        weights=weights,
        source_points=data["ssdr_rest_pose"],
        motion=motion,
        bone_transforms=data["ssdr_transformations"][frame_slice],
        artifacts=artifacts,
    )


def migrate_prediction(
    prediction_path,
    output_dir,
    fps: float = 30.0,
    source_start: int = 0,
    source_end: Optional[int] = None,
) -> SkeleboneBundle:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = load_prediction_bundle(prediction_path, fps, source_start, source_end)
    save_bundle(output_dir / "skelebones.npz", bundle)
    export_bvh(output_dir / "motion.bvh", bundle.skeleton, bundle.motion)
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate legacy prediction_data.npz")
    parser.add_argument("prediction")
    parser.add_argument("output")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--source-start", type=int, default=0)
    parser.add_argument("--source-end", type=int)
    args = parser.parse_args()
    bundle = migrate_prediction(
        args.prediction,
        args.output,
        fps=args.fps,
        source_start=args.source_start,
        source_end=args.source_end,
    )
    print(f"frames={bundle.motion.num_frames} joints={bundle.skeleton.num_joints} bones={bundle.num_bones}")


if __name__ == "__main__":
    main()
