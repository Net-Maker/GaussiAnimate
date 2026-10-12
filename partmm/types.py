"""Dependency-light public representations for standalone PartMM use."""

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np


def _float_array(value, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return array


@dataclass
class PartMMSkeleton:
    rest_joints: np.ndarray
    parents: np.ndarray
    names: Optional[Sequence[str]] = None

    def __post_init__(self) -> None:
        self.rest_joints = _float_array(self.rest_joints, "rest_joints")
        self.parents = np.asarray(self.parents, dtype=np.int64)
        if self.rest_joints.ndim != 2 or self.rest_joints.shape[1] != 3:
            raise ValueError("rest_joints must have shape [joints, 3]")
        joint_count = len(self.rest_joints)
        if joint_count < 1 or self.parents.shape != (joint_count,):
            raise ValueError("parents must have shape [joints]")
        if np.any(self.parents < -1) or np.any(self.parents >= joint_count):
            raise ValueError("parents contain an out-of-range joint index")
        for joint in range(joint_count):
            seen = set()
            current = joint
            while current != -1:
                if current in seen:
                    raise ValueError("parents must define an acyclic hierarchy")
                seen.add(current)
                current = int(self.parents[current])
        if self.names is None:
            self.names = tuple(f"joint_{index:03d}" for index in range(joint_count))
        else:
            self.names = tuple(str(name) for name in self.names)
            if len(self.names) != joint_count or len(set(self.names)) != joint_count:
                raise ValueError("joint names must be unique and match the joint count")

    @property
    def num_joints(self) -> int:
        return int(len(self.rest_joints))

    @property
    def roots(self) -> np.ndarray:
        return np.flatnonzero(self.parents == -1)

    @property
    def offsets(self) -> np.ndarray:
        offsets = self.rest_joints.copy()
        children = self.parents >= 0
        offsets[children] -= self.rest_joints[self.parents[children]]
        return offsets


@dataclass
class PartMMQuery:
    """Novel inner-skeleton motion queried against a PartMM database."""

    local_rotations: np.ndarray
    root_translation: np.ndarray
    fps: float = 30.0

    def __post_init__(self) -> None:
        self.local_rotations = _float_array(
            self.local_rotations, "local_rotations"
        )
        self.root_translation = _float_array(
            self.root_translation, "root_translation"
        )
        if self.local_rotations.ndim != 3 or self.local_rotations.shape[2] != 4:
            raise ValueError("local_rotations must have shape [frames, joints, 4]")
        if self.root_translation.shape != (len(self.local_rotations), 3):
            raise ValueError("root_translation must have shape [frames, 3]")
        norms = np.linalg.norm(self.local_rotations, axis=-1, keepdims=True)
        if np.any(norms < 1e-12):
            raise ValueError("local_rotations contains a zero quaternion")
        self.local_rotations = self.local_rotations / norms
        self.fps = float(self.fps)
        if self.fps <= 0:
            raise ValueError("fps must be positive")

    @classmethod
    def identity(
        cls, num_frames: int, num_joints: int, fps: float = 30.0
    ) -> "PartMMQuery":
        rotations = np.zeros((num_frames, num_joints, 4), dtype=np.float64)
        rotations[..., 0] = 1.0
        return cls(rotations, np.zeros((num_frames, 3)), fps)

    @classmethod
    def from_pose(cls, pose) -> "PartMMQuery":
        if isinstance(pose, cls):
            return pose
        return cls(pose.local_rotations, pose.root_translation, pose.fps)

    @property
    def num_frames(self) -> int:
        return int(len(self.local_rotations))

    @property
    def num_joints(self) -> int:
        return int(self.local_rotations.shape[1])


@dataclass
class PartMMInput:
    """Minimal source database accepted by PartMM.

    The field names describe arrays directly so callers do not need any
    GaussiAnimate-specific Python type.
    """

    rest_joints: np.ndarray
    parents: np.ndarray
    source_local_rotations: np.ndarray
    source_root_translation: np.ndarray
    bone_transforms: np.ndarray
    source_points: np.ndarray
    skinning_weights: np.ndarray
    fps: float = 30.0
    joint_names: Optional[Sequence[str]] = None
    _skeleton: PartMMSkeleton = field(init=False, repr=False)
    _motion: PartMMQuery = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._skeleton = PartMMSkeleton(
            self.rest_joints, self.parents, self.joint_names
        )
        self.rest_joints = self._skeleton.rest_joints
        self.parents = self._skeleton.parents
        self.joint_names = self._skeleton.names
        self._motion = PartMMQuery(
            self.source_local_rotations,
            self.source_root_translation,
            self.fps,
        )
        self.source_local_rotations = self._motion.local_rotations
        self.source_root_translation = self._motion.root_translation
        self.fps = self._motion.fps
        self.source_points = _float_array(self.source_points, "source_points")
        self.skinning_weights = _float_array(
            self.skinning_weights, "skinning_weights"
        )
        self.bone_transforms = _float_array(
            self.bone_transforms, "bone_transforms"
        )
        if self.source_points.ndim != 2 or self.source_points.shape[1] != 3:
            raise ValueError("source_points must have shape [points, 3]")
        if self.skinning_weights.ndim != 2 or self.skinning_weights.shape[0] != len(
            self.source_points
        ):
            raise ValueError("skinning_weights must have shape [points, bones]")
        if np.any(self.skinning_weights < -1e-8):
            raise ValueError("skinning_weights must be non-negative")
        row_sum = self.skinning_weights.sum(axis=1, keepdims=True)
        if self.skinning_weights.shape[1] < 1 or np.any(row_sum <= 1e-12):
            raise ValueError("every point must have at least one bone influence")
        self.skinning_weights = self.skinning_weights / row_sum
        expected = (self._motion.num_frames, self.num_bones, 4, 4)
        if self.bone_transforms.shape != expected:
            raise ValueError(f"bone_transforms must have shape {expected}")

    @classmethod
    def from_bundle(cls, bundle) -> "PartMMInput":
        if isinstance(bundle, cls):
            return bundle
        if bundle.motion is None or bundle.bone_transforms is None:
            raise ValueError(
                "PartMM requires source skeleton motion and outer bone transforms"
            )
        return cls(
            rest_joints=bundle.skeleton.rest_joints,
            parents=bundle.skeleton.parents,
            joint_names=bundle.skeleton.names,
            source_local_rotations=bundle.motion.local_rotations,
            source_root_translation=bundle.motion.root_translation,
            bone_transforms=bundle.bone_transforms,
            source_points=bundle.source_points,
            skinning_weights=bundle.weights,
            fps=bundle.motion.fps,
        )

    @property
    def skeleton(self) -> PartMMSkeleton:
        return self._skeleton

    @property
    def motion(self) -> PartMMQuery:
        return self._motion

    @property
    def weights(self) -> np.ndarray:
        return self.skinning_weights

    @property
    def num_bones(self) -> int:
        return int(self.skinning_weights.shape[1])


@dataclass
class PartMMOutput:
    points: np.ndarray
    bone_transforms: np.ndarray
    joint_positions: np.ndarray
    matched_indices: Dict[int, np.ndarray]
    fps: float

    def __post_init__(self) -> None:
        self.points = _float_array(self.points, "points")
        self.bone_transforms = _float_array(
            self.bone_transforms, "bone_transforms"
        )
        self.joint_positions = _float_array(
            self.joint_positions, "joint_positions"
        )
        if self.points.ndim != 3 or self.points.shape[2] != 3:
            raise ValueError("points must have shape [frames, points, 3]")
        if self.bone_transforms.ndim != 4 or self.bone_transforms.shape[2:] != (
            4,
            4,
        ):
            raise ValueError("bone_transforms must have shape [frames, bones, 4, 4]")
        if self.joint_positions.ndim != 3 or self.joint_positions.shape[2] != 3:
            raise ValueError("joint_positions must have shape [frames, joints, 3]")
        frames = len(self.points)
        if len(self.bone_transforms) != frames or len(self.joint_positions) != frames:
            raise ValueError("all output arrays must have the same frame count")
        self.matched_indices = {
            int(part): np.asarray(indices, dtype=np.int64)
            for part, indices in self.matched_indices.items()
        }
        self.fps = float(self.fps)
        if self.fps <= 0:
            raise ValueError("fps must be positive")


__all__ = [
    "PartMMInput",
    "PartMMOutput",
    "PartMMQuery",
    "PartMMSkeleton",
]
