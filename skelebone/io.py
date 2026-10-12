"""Portable NPZ I/O for public skelebone interfaces."""

import json
from pathlib import Path

import numpy as np

from .types import DynamicPointSequence, PoseSequence, Skeleton, SkeleboneBundle


POINT_SEQUENCE_SCHEMA_VERSION = 1


def load_point_sequence(path, fps: float = 30.0) -> DynamicPointSequence:
    """Load the public registered-point trajectory format."""

    path = Path(path)
    if path.suffix == ".npy":
        return DynamicPointSequence(np.load(path), fps=fps)
    if path.suffix != ".npz":
        raise ValueError("point sequence must be a .npy or .npz file")
    with np.load(path, allow_pickle=False) as data:
        if "points" not in data:
            raise ValueError("NPZ must contain 'points'")
        faces = data["faces"] if "faces" in data else None
        stored_fps = float(data["fps"]) if "fps" in data else fps
        return DynamicPointSequence(data["points"], faces=faces, fps=stored_fps)


def save_point_sequence(path, points, faces=None, fps: float = 30.0) -> Path:
    """Write the single portable input format accepted by the core pipeline."""

    sequence = (
        points
        if isinstance(points, DynamicPointSequence)
        else DynamicPointSequence(points, faces=faces, fps=fps)
    )
    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError("point sequence output must use the .npz extension")
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "schema_version": np.asarray(POINT_SEQUENCE_SCHEMA_VERSION),
        "points": sequence.points,
        "fps": np.asarray(sequence.fps),
    }
    if sequence.faces is not None:
        arrays["faces"] = sequence.faces
    np.savez_compressed(path, **arrays)
    return path


def point_sequence_summary(sequence: DynamicPointSequence) -> dict:
    """Return a concise, JSON-serializable description of an input sequence."""

    return {
        "schema_version": POINT_SEQUENCE_SCHEMA_VERSION,
        "frames": sequence.num_frames,
        "points_per_frame": sequence.num_points,
        "fps": float(sequence.fps),
        "registration_assumed": True,
        "faces": 0 if sequence.faces is None else int(len(sequence.faces)),
        "dtype": str(sequence.points.dtype),
    }


def save_bundle(path, bundle: SkeleboneBundle) -> Path:
    """Save the stable PartMM hand-off without pickled Python objects."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "rest_joints": bundle.skeleton.rest_joints,
        "parents": bundle.skeleton.parents,
        "names": np.asarray(bundle.skeleton.names),
        "weights": bundle.weights,
        "source_points": bundle.source_points,
        "artifacts_json": np.asarray(json.dumps(bundle.artifacts, sort_keys=True)),
    }
    if bundle.motion is not None:
        arrays.update(
            local_rotations=bundle.motion.local_rotations,
            root_translation=bundle.motion.root_translation,
            fps=np.asarray(bundle.motion.fps),
        )
    if bundle.bone_transforms is not None:
        arrays["bone_transforms"] = bundle.bone_transforms
    np.savez_compressed(path, **arrays)
    return path


def load_bundle(path) -> SkeleboneBundle:
    with np.load(path, allow_pickle=False) as data:
        skeleton = Skeleton(data["rest_joints"], data["parents"], data["names"].tolist())
        motion = None
        if "local_rotations" in data:
            motion = PoseSequence(
                data["local_rotations"], data["root_translation"], float(data["fps"])
            )
        bone_transforms = data["bone_transforms"] if "bone_transforms" in data else None
        artifacts = json.loads(str(data["artifacts_json"])) if "artifacts_json" in data else {}
        return SkeleboneBundle(
            skeleton=skeleton,
            weights=data["weights"],
            source_points=data["source_points"],
            motion=motion,
            bone_transforms=bone_transforms,
            artifacts=artifacts,
        )


def save_pose(path, pose: PoseSequence) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        local_rotations=pose.local_rotations,
        root_translation=pose.root_translation,
        fps=np.asarray(pose.fps),
    )
    return path


def load_pose(path) -> PoseSequence:
    with np.load(path, allow_pickle=False) as data:
        return PoseSequence(
            data["local_rotations"], data["root_translation"], float(data["fps"])
        )
