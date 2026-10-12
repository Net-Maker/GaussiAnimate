"""Partwise motion matching from inner skeleton motion to outer SSDR bones."""

from dataclasses import dataclass, replace
from typing import Dict, Tuple

import numpy as np

from .database import build_motion_patches, motion_patch_starts
from .kinematics import forward_kinematics
from .matching import (
    patch_distance as _patch_distance,
    select_patch_neighbors,
)
from .partition import (
    _children,
    _bone_influence_centroids,
    map_bones_to_edge_parents,
    map_bones_to_part_edges,
    map_bones_to_parts,
    overlapping_part_masks,
    partition_skeleton,
    reference_bone_positions,
)
from .transforms import (
    align_transform as _align_transform,
    blend_transforms as _blend_transforms,
    quaternion_slerp as _quaternion_slerp,
    quaternion_to_matrix as _quaternion_to_matrix,
    resample_transforms as _resample_transforms,
    rotation_matrix_to_quaternion as _rotation_matrix_to_quaternion,
    sample_indices as _sample_indices,
)
from .types import PartMMInput, PartMMOutput, PartMMQuery


@dataclass(frozen=True)
class PartMMConfig:
    """Partwise retrieval settings and explicit geometric conventions."""

    num_parts: int = 5
    patch_size: int = 7
    num_neighbors: int = 7
    pyramid_levels: int = 5
    pyramid_blend: float = 0.7
    blend_temperature: float = 0.1
    overlap_rings: int = 1
    align_parts: bool = True
    alignment_mode: str = "part_rigid"
    partition_mode: str = "balanced_tree"
    bone_overlap_ratio: float = 1.5
    alignment_frames: str = "per_frame"
    blend_space: str = "bone_center"

    def __post_init__(self) -> None:
        counts = (
            self.num_parts,
            self.patch_size,
            self.num_neighbors,
            self.pyramid_levels,
        )
        if min(counts) < 1:
            raise ValueError(
                "part, patch, neighbor, and pyramid counts must be positive"
            )
        if not 0.0 <= self.pyramid_blend <= 1.0:
            raise ValueError("pyramid_blend must be in [0, 1]")
        if self.blend_temperature <= 0 or self.overlap_rings < 0:
            raise ValueError(
                "blend_temperature must be positive and "
                "overlap_rings non-negative"
            )
        if self.alignment_mode not in {"part_rigid", "edge_wise"}:
            raise ValueError("alignment_mode must be part_rigid or edge_wise")
        if self.partition_mode not in {"balanced_tree", "humanoid_five"}:
            raise ValueError(
                "partition_mode must be balanced_tree or humanoid_five"
            )
        if self.partition_mode == "humanoid_five" and self.num_parts != 5:
            raise ValueError("humanoid_five requires exactly five parts")
        if self.bone_overlap_ratio < 1.0:
            raise ValueError("bone_overlap_ratio must be at least one")
        if self.alignment_frames not in {"per_frame", "patch_center"}:
            raise ValueError("alignment_frames must be per_frame or patch_center")
        if self.blend_space not in {"bone_center", "origin"}:
            raise ValueError("blend_space must be bone_center or origin")


AnimationResult = PartMMOutput


class PartMotionMatcher:
    """KVQ database for partwise skeleton-to-SSDR motion retrieval."""

    def __init__(
        self,
        bundle: PartMMInput,
        config: PartMMConfig = PartMMConfig(),
    ):
        self.bundle = PartMMInput.from_bundle(bundle)
        self.config = config
        self.blend_centers = (
            _bone_influence_centroids(self.bundle)
            if config.blend_space == "bone_center" else None
        )
        (
            self.source_global_transforms,
            self.source_joint_positions,
        ) = forward_kinematics(self.bundle.skeleton, self.bundle.motion)
        self.part_assignment, components = partition_skeleton(
            self.bundle.skeleton.parents,
            config.num_parts,
            self.bundle.skeleton.names,
            config.partition_mode,
        )
        self.part_masks = overlapping_part_masks(
            self.bundle.skeleton.parents,
            components,
            config.overlap_rings,
        )
        reference_bones = reference_bone_positions(self.bundle, frame=0)
        (
            self.part_bone_masks,
            self.part_edge_parents,
        ) = map_bones_to_part_edges(
            self.bundle,
            components,
            joint_positions=self.source_joint_positions[0],
            bone_positions=reference_bones,
            overlap_ratio=config.bone_overlap_ratio,
        )
        self.bone_to_part = self.part_bone_masks.argmax(axis=0)
        self.bone_to_edge_parent = self.part_edge_parents[
            self.bone_to_part,
            np.arange(self.bundle.num_bones),
        ]

    def match(
        self,
        novel_pose: PartMMQuery,
    ) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
        novel_pose = PartMMQuery.from_pose(novel_pose)
        if self.config.pyramid_levels == 1:
            return self._match_single(novel_pose)

        refined = None
        finest_indices = {}
        strides = [
            2 ** power
            for power in reversed(range(self.config.pyramid_levels))
        ]
        for stride in strides:
            source_indices = _sample_indices(self.bundle.motion.num_frames, stride)
            query_indices = _sample_indices(novel_pose.num_frames, stride)
            source_motion = PartMMQuery(
                self.bundle.motion.local_rotations[source_indices],
                self.bundle.motion.root_translation[source_indices],
                self.bundle.motion.fps / stride,
            )
            query_motion = PartMMQuery(
                novel_pose.local_rotations[query_indices],
                novel_pose.root_translation[query_indices],
                novel_pose.fps / stride,
            )
            level_bundle = PartMMInput(
                rest_joints=self.bundle.rest_joints,
                parents=self.bundle.parents,
                joint_names=self.bundle.joint_names,
                skinning_weights=self.bundle.weights,
                source_points=self.bundle.source_points,
                source_local_rotations=source_motion.local_rotations,
                source_root_translation=source_motion.root_translation,
                fps=source_motion.fps,
                bone_transforms=self.bundle.bone_transforms[source_indices],
            )
            level_config = replace(
                self.config,
                pyramid_levels=1,
                patch_size=min(
                    self.config.patch_size,
                    len(source_indices),
                    len(query_indices),
                ),
            )
            level_matcher = PartMotionMatcher(level_bundle, level_config)
            level_transform, indices = level_matcher._match_single(
                query_motion
            )
            level_transform = _resample_transforms(
                level_transform, query_indices, novel_pose.num_frames,
                centers=self.blend_centers,
            )
            if refined is None:
                refined = level_transform
            else:
                level_weights = np.array(
                    [1.0 - self.config.pyramid_blend, self.config.pyramid_blend]
                )
                stacked = np.stack([refined, level_transform])
                refined = np.empty_like(level_transform)
                for frame in range(novel_pose.num_frames):
                    refined[frame] = _blend_transforms(
                        stacked[:, frame],
                        level_weights,
                        centers=self.blend_centers,
                    )
            if stride == 1:
                finest_indices = indices
        return refined, finest_indices

    def _match_single(
        self,
        novel_pose: PartMMQuery,
    ) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
        if novel_pose.num_joints != self.bundle.skeleton.num_joints:
            raise ValueError("novel pose and source skeleton joint counts differ")
        source_pose = self.bundle.motion.local_rotations
        query_pose = novel_pose.local_rotations
        patch = min(self.config.patch_size, len(source_pose), len(query_pose))
        source_starts = motion_patch_starts(len(source_pose), patch)
        query_starts = motion_patch_starts(len(query_pose), patch)
        if not len(source_starts) or not len(query_starts):
            raise ValueError("source and query motions must contain at least one frame")
        query_global_transforms, query_joint_positions = forward_kinematics(
            self.bundle.skeleton, novel_pose
        )
        translation_sum = np.zeros(
            (len(query_pose), self.bundle.num_bones, 3), dtype=np.float64
        )
        quaternion_scatter = np.zeros(
            (len(query_pose), self.bundle.num_bones, 4, 4), dtype=np.float64
        )
        counts = np.zeros((len(query_pose), self.bundle.num_bones), dtype=np.float64)
        matched_indices = {}

        for part, joint_mask in enumerate(self.part_masks):
            bone_indices = np.flatnonzero(self.part_bone_masks[part])
            if not len(bone_indices):
                continue
            source_patches = build_motion_patches(
                source_pose,
                source_starts,
                patch,
                joint_mask,
            )
            part_matches = np.empty(
                (len(query_starts), self.config.num_neighbors),
                dtype=np.int64,
            )
            for query_index, query_start in enumerate(query_starts):
                query_patch = query_pose[
                    query_start : query_start + patch,
                    joint_mask,
                ]
                selected_starts, reported_starts, blend_weights = (
                    select_patch_neighbors(
                        query_patch,
                        source_patches,
                        source_starts,
                        self.config.num_neighbors,
                        self.config.blend_temperature,
                    )
                )
                neighbor_count = len(selected_starts)
                part_matches[query_index] = reported_starts
                alignments = np.broadcast_to(
                    np.eye(4),
                    (neighbor_count, 4, 4),
                ).copy()
                edge_alignments = None
                if (
                    self.config.align_parts
                    and self.config.alignment_mode == "part_rigid"
                    and self.config.alignment_frames == "patch_center"
                ):
                    target_alignment_points = query_joint_positions[
                        query_start:query_start + patch, joint_mask
                    ].reshape(-1, 3)
                    for candidate_index, source_start in enumerate(
                        selected_starts
                    ):
                        source_alignment_points = self.source_joint_positions[
                            source_start:source_start + patch, joint_mask
                        ].reshape(-1, 3)
                        alignments[candidate_index] = _align_transform(
                            source_alignment_points, target_alignment_points
                        )
                elif self.config.align_parts and self.config.alignment_frames == "patch_center":
                    center = patch // 2
                    target_frame = min(
                        query_start + center,
                        len(query_pose) - 1,
                    )
                    edge_alignments = np.empty(
                        (neighbor_count, len(bone_indices), 4, 4),
                        dtype=np.float64,
                    )
                    for candidate_index, source_start in enumerate(
                        selected_starts
                    ):
                        source_frame = min(
                            source_start + center, len(source_pose) - 1
                        )
                        for local_bone, bone in enumerate(bone_indices):
                            parent = self.part_edge_parents[part, bone]
                            edge_alignments[candidate_index, local_bone] = (
                                query_global_transforms[target_frame, parent]
                                @ np.linalg.inv(
                                    self.source_global_transforms[source_frame, parent]
                                )
                            )
                for offset in range(patch):
                    if self.config.align_parts and self.config.alignment_frames == "per_frame":
                        if self.config.alignment_mode == "edge_wise":
                            parents = self.part_edge_parents[part, bone_indices]
                            edge_alignments = (
                                query_global_transforms[query_start + offset, parents][None]
                                @ np.linalg.inv(self.source_global_transforms[
                                    (selected_starts + offset)[:, None], parents[None]
                                ])
                            )
                        else:
                            for candidate_index, source_start in enumerate(selected_starts):
                                alignments[candidate_index] = _align_transform(
                                    self.source_joint_positions[source_start + offset, joint_mask],
                                    query_joint_positions[query_start + offset, joint_mask],
                                )
                    candidates = np.stack([
                        self.bundle.bone_transforms[start + offset, bone_indices]
                        for start in selected_starts
                    ])
                    candidates = (
                        edge_alignments @ candidates
                        if edge_alignments is not None
                        else alignments[:, None] @ candidates
                    )
                    centers = (
                        self.blend_centers[bone_indices]
                        if self.blend_centers is not None else None
                    )
                    blended = _blend_transforms(candidates, blend_weights, centers=centers)
                    frame = query_start + offset
                    quaternions = _rotation_matrix_to_quaternion(
                        blended[:, :3, :3]
                    )
                    positions = blended[:, :3, 3]
                    if centers is not None:
                        positions = positions + np.einsum("bij,bj->bi", blended[:, :3, :3], centers)
                    translation_sum[frame, bone_indices] += positions
                    quaternion_scatter[frame, bone_indices] += np.einsum(
                        "bi,bj->bij", quaternions, quaternions
                    )
                    counts[frame, bone_indices] += 1
            matched_indices[part] = part_matches

        valid = counts > 0
        output = np.broadcast_to(
            np.eye(4), (len(query_pose), self.bundle.num_bones, 4, 4)
        ).copy()
        output[..., :3, 3][valid] = translation_sum[valid] / counts[valid, None]
        _, eigenvectors = np.linalg.eigh(quaternion_scatter[valid])
        averaged_quaternions = eigenvectors[..., -1]
        averaged_quaternions = np.where(
            averaged_quaternions[..., :1] < 0,
            -averaged_quaternions,
            averaged_quaternions,
        )
        output[..., :3, :3][valid] = _quaternion_to_matrix(averaged_quaternions)
        if self.blend_centers is not None:
            rotated_centers = np.einsum("tbij,bj->tbi", output[..., :3, :3], self.blend_centers)
            output[..., :3, 3][valid] -= rotated_centers[valid]
        return output, matched_indices

    def animate(self, novel_pose: PartMMQuery) -> AnimationResult:
        novel_pose = PartMMQuery.from_pose(novel_pose)
        transforms, indices = self.match(novel_pose)
        homogeneous = np.concatenate([
            self.bundle.source_points,
            np.ones((len(self.bundle.source_points), 1), dtype=np.float64),
        ], axis=1)
        points = np.einsum(
            "nb,tbij,nj->tni",
            self.bundle.weights,
            transforms,
            homogeneous,
            optimize=True,
        )[..., :3]
        _, joints = forward_kinematics(self.bundle.skeleton, novel_pose)
        return AnimationResult(points, transforms, joints, indices, novel_pose.fps)


def animate(
    bundle: PartMMInput,
    novel_pose: PartMMQuery,
    config: PartMMConfig = PartMMConfig(),
) -> AnimationResult:
    return PartMotionMatcher(bundle, config).animate(novel_pose)
