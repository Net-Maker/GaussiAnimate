"""Shared, dependency-light representations for skeleton extraction and animation."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np


def _float_array(value, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return array


def _point_attributes(attributes, count):
    result = {}
    for name, value in attributes.items():
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError("point attribute names must be identifiers")
        array = np.asarray(value)
        if array.ndim < 1 or array.shape[0] != count:
            raise ValueError(f"point attribute {name} must have {count} rows")
        if array.dtype.kind not in "biuf" or not np.all(np.isfinite(array)):
            raise ValueError(f"point attribute {name} must contain finite numeric values")
        result[name] = array
    return result


@dataclass
class DynamicPointSequence:
    """A temporally registered point sequence with shape ``[T, N, 3]``."""

    points: np.ndarray
    faces: Optional[np.ndarray] = None
    fps: float = 30.0
    point_attributes: Dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.points = _float_array(self.points, "points")
        if self.points.ndim != 3 or self.points.shape[2] != 3:
            raise ValueError("points must have shape [frames, points, 3]")
        if self.points.shape[0] < 1 or self.points.shape[1] < 1:
            raise ValueError("point sequence must contain at least one frame and point")
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if self.faces is not None:
            self.faces = np.asarray(self.faces, dtype=np.int64)
            if self.faces.ndim != 2 or self.faces.shape[1] != 3:
                raise ValueError("faces must have shape [triangles, 3]")
            if self.faces.size and (self.faces.min() < 0 or self.faces.max() >= self.points.shape[1]):
                raise ValueError("faces contain an out-of-range vertex index")
        self.point_attributes = _point_attributes(self.point_attributes, self.num_points)

    @property
    def num_frames(self) -> int:
        return int(self.points.shape[0])

    @property
    def num_points(self) -> int:
        return int(self.points.shape[1])


@dataclass
class Skeleton:
    """A bind skeleton whose parent array defines an acyclic forest."""

    rest_joints: np.ndarray
    parents: np.ndarray
    names: Optional[Sequence[str]] = None

    def __post_init__(self) -> None:
        self.rest_joints = _float_array(self.rest_joints, "rest_joints")
        self.parents = np.asarray(self.parents, dtype=np.int64)
        if self.rest_joints.ndim != 2 or self.rest_joints.shape[1] != 3:
            raise ValueError("rest_joints must have shape [joints, 3]")
        joint_count = self.rest_joints.shape[0]
        if self.parents.shape != (joint_count,):
            raise ValueError("parents must have shape [joints]")
        if joint_count == 0:
            raise ValueError("skeleton must contain at least one joint")
        if np.any(self.parents >= joint_count) or np.any(self.parents < -1):
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
        return int(self.rest_joints.shape[0])

    @property
    def roots(self) -> np.ndarray:
        return np.flatnonzero(self.parents == -1)

    @property
    def offsets(self) -> np.ndarray:
        offsets = self.rest_joints.copy()
        child_mask = self.parents >= 0
        offsets[child_mask] -= self.rest_joints[self.parents[child_mask]]
        return offsets


@dataclass
class PoseSequence:
    """Local joint rotations and root displacement over time.

    Quaternions use scalar-first ``[w, x, y, z]`` ordering. ``root_translation``
    is a displacement from the bind root, not an absolute world position.
    """

    local_rotations: np.ndarray
    root_translation: np.ndarray
    fps: float = 30.0

    def __post_init__(self) -> None:
        self.local_rotations = _float_array(self.local_rotations, "local_rotations")
        self.root_translation = _float_array(self.root_translation, "root_translation")
        if self.local_rotations.ndim != 3 or self.local_rotations.shape[2] != 4:
            raise ValueError("local_rotations must have shape [frames, joints, 4]")
        if self.root_translation.shape != (self.local_rotations.shape[0], 3):
            raise ValueError("root_translation must have shape [frames, 3]")
        norms = np.linalg.norm(self.local_rotations, axis=-1, keepdims=True)
        if np.any(norms < 1e-12):
            raise ValueError("local_rotations contains a zero quaternion")
        self.local_rotations = self.local_rotations / norms
        if self.fps <= 0:
            raise ValueError("fps must be positive")

    @classmethod
    def identity(cls, num_frames: int, num_joints: int, fps: float = 30.0) -> "PoseSequence":
        rotations = np.zeros((num_frames, num_joints, 4), dtype=np.float64)
        rotations[..., 0] = 1.0
        return cls(rotations, np.zeros((num_frames, 3), dtype=np.float64), fps=fps)

    @property
    def num_frames(self) -> int:
        return int(self.local_rotations.shape[0])

    @property
    def num_joints(self) -> int:
        return int(self.local_rotations.shape[1])


@dataclass
class SkeleboneBundle:
    """Stable hand-off from skeleton extraction to partmm.

    Inner kinematic joints (``J``) and outer SSDR bones (``B``) are separate:
    ``motion`` stores ``J`` local rotations, while ``bone_transforms`` stores
    the corresponding ``B`` free-form rigid motions.
    """

    skeleton: Skeleton
    weights: np.ndarray
    source_points: np.ndarray
    motion: Optional[PoseSequence] = None
    bone_transforms: Optional[np.ndarray] = None
    artifacts: Dict[str, str] = field(default_factory=dict)
    point_attributes: Dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.weights = _float_array(self.weights, "weights")
        self.source_points = _float_array(self.source_points, "source_points")
        if self.source_points.ndim != 2 or self.source_points.shape[1] != 3:
            raise ValueError("source_points must have shape [points, 3]")
        if self.weights.ndim != 2 or self.weights.shape[0] != self.source_points.shape[0]:
            raise ValueError("weights must have shape [points, outer_bones]")
        if self.weights.shape[1] < 1:
            raise ValueError("bundle must contain at least one outer bone")
        if np.any(self.weights < -1e-8):
            raise ValueError("weights must be non-negative")
        row_sum = self.weights.sum(axis=1, keepdims=True)
        if np.any(row_sum <= 1e-12):
            raise ValueError("every point must be influenced by at least one joint")
        self.weights = self.weights / row_sum
        if self.motion is not None and self.motion.num_joints != self.skeleton.num_joints:
            raise ValueError("motion and skeleton joint counts differ")
        if self.bone_transforms is not None:
            self.bone_transforms = _float_array(self.bone_transforms, "bone_transforms")
            expected_tail = (self.weights.shape[1], 4, 4)
            if self.bone_transforms.ndim != 4 or self.bone_transforms.shape[1:] != expected_tail:
                raise ValueError(f"bone_transforms must have shape [frames, {expected_tail[0]}, 4, 4]")
            if self.motion is None or self.bone_transforms.shape[0] != self.motion.num_frames:
                raise ValueError("bone transforms require a same-length source motion")
        self.artifacts = {str(key): str(Path(value)) for key, value in self.artifacts.items()}
        self.point_attributes = _point_attributes(self.point_attributes, len(self.source_points))

    @property
    def num_bones(self) -> int:
        return int(self.weights.shape[1])
