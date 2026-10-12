"""Typed loaders for method artifacts with legacy compatibility."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SSDRCache:
    """Canonical in-memory SSDR cache.

    Weights are always [points, bones] in memory. Legacy files that store
    [bones, points] remain readable.
    """

    rest_pose: np.ndarray
    weights: np.ndarray
    transformations: np.ndarray
    world_transformations: np.ndarray
    frame_means: np.ndarray
    training_frames: int
    schema_version: int

    @property
    def points(self) -> int:
        return int(self.rest_pose.shape[0])

    @property
    def bones(self) -> int:
        return int(self.weights.shape[1])

    @classmethod
    def load(cls, path) -> "SSDRCache":
        path = Path(path)
        with np.load(path, allow_pickle=False) as archive:
            rest_pose = np.asarray(
                archive["rest_pose"],
                dtype=np.float64,
            )
            weights = np.asarray(
                archive["weights"],
                dtype=np.float64,
            )
            has_local = "transformations" in archive
            has_world = "world_transformations" in archive
            if not has_local and not has_world:
                raise ValueError(
                    "SSDR cache has no transformations"
                )
            source_transforms = (
                archive["transformations"]
                if has_local
                else archive["world_transformations"]
            )
            transformations = np.asarray(
                source_transforms,
                dtype=np.float64,
            )
            frame_means = np.asarray(
                archive["frame_means"]
                if "frame_means" in archive
                else np.zeros((len(transformations), 3)),
                dtype=np.float64,
            )
            training_frames = int(
                archive["training_frames"]
                if "training_frames" in archive
                else len(transformations)
            )
            schema_version = int(
                archive["schema_version"]
                if "schema_version" in archive
                else 0
            )
            weights_layout = (
                str(archive["weights_layout"])
                if "weights_layout" in archive
                else None
            )
            if has_world:
                world_transformations = np.asarray(
                    archive["world_transformations"],
                    dtype=np.float64,
                )
            else:
                world_transformations = transformations.copy()
                world_transformations[:, :, :3, 3] += (
                    frame_means[:, None]
                )
            if not has_local:
                transformations = world_transformations.copy()
                transformations[:, :, :3, 3] -= (
                    frame_means[:, None]
                )

        if rest_pose.ndim != 2 or rest_pose.shape[1] != 3:
            raise ValueError("SSDR rest_pose must have shape [N, 3]")
        if transformations.ndim != 4 or transformations.shape[2:] != (4, 4):
            raise ValueError(
                "SSDR transformations must have shape [T, B, 4, 4]"
            )
        frames, bones = transformations.shape[:2]
        if weights_layout not in (None, "bones_points", "points_bones"):
            raise ValueError(
                f"unsupported SSDR weights layout: {weights_layout}"
            )
        if weights_layout == "bones_points":
            weights = weights.T
        elif weights_layout is None and weights.shape == (
            bones,
            len(rest_pose),
        ):
            weights = weights.T
        if weights.shape != (len(rest_pose), bones):
            raise ValueError(
                "SSDR weights must match rest points and transform bones"
            )
        if frame_means.shape != (frames, 3):
            raise ValueError(
                "SSDR frame_means must have shape [T, 3]"
            )
        if world_transformations.shape != transformations.shape:
            raise ValueError(
                "SSDR world transformations do not match local transforms"
            )
        if training_frames != frames:
            raise ValueError(
                "SSDR training_frames does not match transformations"
            )
        arrays = (
            rest_pose,
            weights,
            transformations,
            world_transformations,
            frame_means,
        )
        if not all(np.isfinite(value).all() for value in arrays):
            raise ValueError("SSDR cache contains non-finite values")
        weight_sum = weights.sum(axis=1)
        if np.any(weight_sum <= 0):
            raise ValueError("SSDR cache contains unbound points")
        weights = weights / weight_sum[:, None]
        return cls(
            rest_pose=rest_pose,
            weights=weights,
            transformations=transformations,
            world_transformations=world_transformations,
            frame_means=frame_means,
            training_frames=training_frames,
            schema_version=schema_version,
        )
