import unittest

import numpy as np

import partmm
import partmm.animator as animator
from partmm.database import build_motion_patches, motion_patch_starts
from partmm.matching import patch_distance, select_patch_neighbors
from partmm.partition import (
    _children as partition_children,
    map_bones_to_edge_parents,
    map_bones_to_part_edges,
    map_bones_to_parts,
    overlapping_part_masks,
    partition_skeleton,
    reference_bone_positions,
)
from partmm.transforms import (
    align_transform,
    blend_transforms,
    quaternion_slerp,
    quaternion_to_matrix,
    resample_transforms,
    rotation_matrix_to_quaternion,
    sample_indices,
)


def _quaternion_z(angle):
    return np.array(
        [np.cos(angle / 2.0), 0.0, 0.0, np.sin(angle / 2.0)]
    )


class PartMMModuleBoundaryTest(unittest.TestCase):
    def test_original_public_api_is_unchanged(self):
        self.assertTrue(
            {
                "AnimationResult",
                "PartMMConfig",
                "PartMotionMatcher",
                "animate",
            }.issubset(partmm.__all__)
        )
        self.assertIs(partmm.AnimationResult, animator.AnimationResult)
        self.assertIs(partmm.PartMMConfig, animator.PartMMConfig)
        self.assertIs(partmm.PartMotionMatcher, animator.PartMotionMatcher)
        self.assertIs(partmm.animate, animator.animate)

    def test_animator_reexports_extracted_helpers(self):
        self.assertIs(animator._children, partition_children)
        self.assertIs(animator.partition_skeleton, partition_skeleton)
        self.assertIs(
            animator.overlapping_part_masks,
            overlapping_part_masks,
        )
        self.assertIs(animator.map_bones_to_parts, map_bones_to_parts)
        self.assertIs(
            animator.map_bones_to_edge_parents,
            map_bones_to_edge_parents,
        )
        self.assertIs(
            animator.map_bones_to_part_edges,
            map_bones_to_part_edges,
        )
        self.assertIs(
            animator.reference_bone_positions,
            reference_bone_positions,
        )
        self.assertIs(animator._patch_distance, patch_distance)
        self.assertIs(animator._align_transform, align_transform)
        self.assertIs(animator._blend_transforms, blend_transforms)
        self.assertIs(
            animator._rotation_matrix_to_quaternion,
            rotation_matrix_to_quaternion,
        )
        self.assertIs(
            animator._quaternion_to_matrix,
            quaternion_to_matrix,
        )
        self.assertIs(animator._sample_indices, sample_indices)
        self.assertIs(
            animator._resample_transforms,
            resample_transforms,
        )
        self.assertIs(animator._quaternion_slerp, quaternion_slerp)

    def test_motion_patch_database_preserves_frame_and_joint_order(self):
        rotations = np.arange(5 * 3 * 4).reshape(5, 3, 4)
        starts = motion_patch_starts(5, 3)
        np.testing.assert_array_equal(starts, [0, 1, 2])
        patches = build_motion_patches(
            rotations,
            starts,
            patch_size=3,
            joint_mask=np.array([True, False, True]),
        )
        expected = np.stack(
            [
                rotations[0:3][:, [0, 2]],
                rotations[1:4][:, [0, 2]],
                rotations[2:5][:, [0, 2]],
            ]
        )
        np.testing.assert_array_equal(patches, expected)

    def test_neighbor_selection_matches_geodesic_knn_and_tie_break(self):
        query = _quaternion_z(0.0)[None, None]
        source = np.stack(
            [
                _quaternion_z(np.pi / 2.0)[None, None],
                _quaternion_z(0.0)[None, None],
                -_quaternion_z(0.0)[None, None],
                _quaternion_z(np.pi)[None, None],
            ]
        )
        starts = np.array([10, 3, 1, 7])
        selected, reported, weights = select_patch_neighbors(
            query,
            source,
            starts,
            num_neighbors=5,
            temperature=0.1,
        )
        np.testing.assert_array_equal(selected, [1, 3, 10, 7])
        np.testing.assert_array_equal(reported, [1, 3, 10, 7, 7])
        np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-15)
        np.testing.assert_allclose(weights[0], weights[1], atol=1e-15)
        self.assertGreater(weights[1], weights[2])
        self.assertGreater(weights[2], weights[3])
        np.testing.assert_array_equal(
            animator._patch_distance(query, source),
            patch_distance(query, source),
        )

    def test_transform_reexports_are_numerically_identical(self):
        rotations = quaternion_to_matrix(
            np.stack(
                [
                    _quaternion_z(0.0),
                    _quaternion_z(np.pi / 2.0),
                ]
            )
        )
        transforms = np.broadcast_to(np.eye(4), (2, 1, 4, 4)).copy()
        transforms[:, 0, :3, :3] = rotations
        transforms[:, 0, :3, 3] = np.array(
            [[0.0, 0.0, 0.0], [2.0, 4.0, 6.0]]
        )
        weights = np.array([0.25, 0.75])
        np.testing.assert_array_equal(
            animator._blend_transforms(transforms, weights),
            blend_transforms(transforms, weights),
        )
        sampled = np.array([0, 3])
        np.testing.assert_array_equal(
            animator._resample_transforms(transforms, sampled, 4),
            resample_transforms(transforms, sampled, 4),
        )


if __name__ == "__main__":
    unittest.main()
