import unittest
from unittest import mock

import numpy as np

from partmm.animator import (
    PartMMConfig,
    PartMotionMatcher,
    _align_transform,
    _blend_transforms,
    _patch_distance,
    _resample_transforms,
    map_bones_to_edge_parents,
    map_bones_to_part_edges,
    map_bones_to_parts,
    partition_skeleton,
    reference_bone_positions,
)
from skelebone import PoseSequence, Skeleton, SkeleboneBundle


def _rotation_z(angle):
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]
    )


def _quaternion_z(angle):
    return np.array([np.cos(angle / 2.0), 0.0, 0.0, np.sin(angle / 2.0)])


def _transform(rotation=None, translation=None):
    result = np.eye(4)
    if rotation is not None:
        result[:3, :3] = rotation
    if translation is not None:
        result[:3, 3] = translation
    return result


class PartMMTest(unittest.TestCase):
    def test_so3_patch_distance_is_sign_invariant_and_uses_rotation_angle(self):
        query = _quaternion_z(np.pi / 2.0)[None, None]
        sources = np.stack([-query, np.array([[[1.0, 0.0, 0.0, 0.0]]])])
        distances = _patch_distance(query, sources)
        np.testing.assert_allclose(distances[0], 0.0, atol=1e-14)
        np.testing.assert_allclose(distances[1], (np.pi / 2.0) ** 2, atol=1e-12)

    def test_partition_is_balanced_connected_and_deterministic(self):
        parents = np.array([-1, 0, 0, 1, 1, 2, 2])
        first_assignment, first_parts = partition_skeleton(parents, 3)
        second_assignment, second_parts = partition_skeleton(parents, 3)
        np.testing.assert_array_equal(first_assignment, second_assignment)
        self.assertEqual(first_parts, second_parts)
        self.assertEqual(sorted(map(len, first_parts)), [1, 3, 3])
        for component in first_parts:
            roots_in_component = [
                joint for joint in component if parents[joint] not in component
            ]
            self.assertEqual(len(roots_in_component), 1)

    def test_humanoid_partition_uses_five_semantic_parts(self):
        names = [
            "pelvis",
            "left_hip",
            "right_hip",
            "spine1",
            "left_knee",
            "right_knee",
            "spine2",
            "left_ankle",
            "right_ankle",
            "spine3",
            "left_foot",
            "right_foot",
            "neck",
            "left_collar",
            "right_collar",
            "head",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_hand",
            "right_hand",
        ]
        parents = np.array(
            [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12,
             13, 14, 16, 17, 18, 19, 20, 21]
        )
        assignment, components = partition_skeleton(
            parents,
            5,
            names,
            "humanoid_five",
        )
        lookup = {name: index for index, name in enumerate(names)}
        self.assertEqual(assignment[lookup["head"]], 0)
        self.assertEqual(assignment[lookup["left_wrist"]], 1)
        self.assertEqual(assignment[lookup["right_wrist"]], 2)
        self.assertEqual(assignment[lookup["left_ankle"]], 3)
        self.assertEqual(assignment[lookup["right_ankle"]], 4)
        self.assertEqual(set().union(*components), set(range(len(names))))

    def test_bones_map_by_weighted_influence_centroid(self):
        skeleton = Skeleton(
            np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
            np.array([-1, 0]),
        )
        bundle = SkeleboneBundle(
            skeleton,
            weights=np.eye(2),
            source_points=np.array([[0.1, 0.0, 0.0], [9.9, 0.0, 0.0]]),
        )
        np.testing.assert_array_equal(map_bones_to_parts(bundle, np.array([0, 1])), [0, 1])

    def test_bones_bind_to_nearest_skeleton_edge_parent(self):
        skeleton = Skeleton(
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 2.0, 0.0]]),
            np.array([-1, 0, 1]),
        )
        bundle = SkeleboneBundle(
            skeleton,
            weights=np.eye(2),
            source_points=np.array([[0.5, 0.1, 0.0], [1.1, 1.5, 0.0]]),
        )
        np.testing.assert_array_equal(map_bones_to_edge_parents(bundle), [0, 1])

    def test_reference_bone_positions_apply_observed_outer_transforms(self):
        skeleton = Skeleton(
            np.array([[0.0, 0.0, 0.0]]),
            np.array([-1]),
        )
        transform = _transform(translation=[3.0, 4.0, 5.0])[None, None]
        bundle = SkeleboneBundle(
            skeleton,
            weights=np.ones((1, 1)),
            source_points=np.array([[1.0, 2.0, 3.0]]),
            motion=PoseSequence.identity(1, 1),
            bone_transforms=transform,
        )
        np.testing.assert_allclose(
            reference_bone_positions(bundle),
            [[4.0, 6.0, 8.0]],
        )

    def test_boundary_bone_can_belong_to_two_parts_with_distinct_edges(self):
        skeleton = Skeleton(
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [-1.0, 1.0, 0.0],
                    [1.0, 1.0, 0.0],
                ]
            ),
            np.array([-1, 0, 0]),
        )
        bundle = SkeleboneBundle(
            skeleton,
            weights=np.ones((1, 1)),
            source_points=np.array([[0.0, 1.0, 0.0]]),
        )
        masks, edge_parents = map_bones_to_part_edges(
            bundle,
            [{1}, {2}],
            overlap_ratio=1.0,
        )
        np.testing.assert_array_equal(masks[:, 0], [True, True])
        np.testing.assert_array_equal(edge_parents[:, 0], [0, 0])

    def test_kabsch_recovers_rigid_transform_and_rejects_reflection(self):
        source = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]
        )
        rotation = _rotation_z(0.7)
        translation = np.array([2.0, -1.0, 0.5])
        target = source @ rotation.T + translation
        alignment = _align_transform(source, target)
        np.testing.assert_allclose(alignment[:3, :3], rotation, atol=1e-12)
        np.testing.assert_allclose(alignment[:3, 3], translation, atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(alignment[:3, :3]), 1.0, places=12)

        reflected = source.copy()
        reflected[:, 0] *= -1.0
        reflected_alignment = _align_transform(source, reflected)
        self.assertGreater(np.linalg.det(reflected_alignment[:3, :3]), 0.0)

    def test_se3_blend_uses_proper_rotation_and_linear_translation(self):
        transforms = np.stack(
            [
                _transform(_rotation_z(0.0), [0.0, 0.0, 0.0]),
                _transform(_rotation_z(np.pi / 2.0), [2.0, 4.0, 6.0]),
            ]
        )[:, None]
        blended = _blend_transforms(transforms, np.array([0.5, 0.5]))[0]
        np.testing.assert_allclose(blended[:3, :3], _rotation_z(np.pi / 4.0), atol=1e-12)
        np.testing.assert_allclose(blended[:3, 3], [1.0, 2.0, 3.0], atol=1e-12)
        np.testing.assert_allclose(blended[:3, :3].T @ blended[:3, :3], np.eye(3), atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(blended[:3, :3]), 1.0, places=12)

    def test_each_knn_candidate_is_aligned_before_blending(self):
        skeleton = Skeleton(
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            np.array([-1, 0, 0]),
        )
        source_pose = PoseSequence.identity(2, 3)
        source_pose.local_rotations[1, 0] = _quaternion_z(np.pi)
        source_transforms = np.stack(
            [_transform(_rotation_z(0.0)), _transform(_rotation_z(np.pi))]
        )[:, None]
        bundle = SkeleboneBundle(
            skeleton,
            np.ones((3, 1)),
            skeleton.rest_joints,
            source_pose,
            source_transforms,
        )
        query = PoseSequence.identity(1, 3)
        query.local_rotations[0, 0] = _quaternion_z(np.pi / 2.0)
        matcher = PartMotionMatcher(
            bundle,
            PartMMConfig(
                num_parts=1,
                patch_size=1,
                num_neighbors=2,
                pyramid_levels=1,
                overlap_rings=0,
                align_parts=True,
            ),
        )
        matched, _ = matcher.match(query)
        np.testing.assert_allclose(
            matched[0, 0, :3, :3], _rotation_z(np.pi / 2.0), atol=1e-7
        )

    def test_edge_wise_alignment_uses_bound_parent_transform(self):
        skeleton = Skeleton(
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            np.array([-1, 0]),
        )
        bundle = SkeleboneBundle(
            skeleton,
            np.ones((1, 1)),
            np.array([[0.5, 0.0, 0.0]]),
            PoseSequence.identity(1, 2),
            np.eye(4)[None, None],
        )
        query = PoseSequence.identity(1, 2)
        query.local_rotations[0, 0] = _quaternion_z(np.pi / 2.0)
        matched, _ = PartMotionMatcher(
            bundle,
            PartMMConfig(
                num_parts=1,
                patch_size=1,
                num_neighbors=1,
                pyramid_levels=1,
                overlap_rings=0,
                alignment_mode="edge_wise",
            ),
        ).match(query)
        np.testing.assert_allclose(
            matched[0, 0, :3, :3], _rotation_z(np.pi / 2.0), atol=1e-12
        )

    def test_pyramid_uses_recursive_paper_blend(self):
        skeleton = Skeleton(np.array([[0.0, 0.0, 0.0]]), np.array([-1]))
        source_pose = PoseSequence.identity(5, 1)
        bundle = SkeleboneBundle(
            skeleton,
            np.ones((1, 1)),
            np.zeros((1, 3)),
            source_pose,
            np.broadcast_to(np.eye(4), (5, 1, 4, 4)).copy(),
        )
        matcher = PartMotionMatcher(
            bundle,
            PartMMConfig(
                num_parts=1,
                patch_size=1,
                num_neighbors=1,
                pyramid_levels=2,
                pyramid_blend=0.7,
                align_parts=False,
            ),
        )

        def fake_single(level_matcher, novel_pose):
            angle = 0.0 if level_matcher.bundle.motion.num_frames == 3 else np.pi / 2.0
            transform = _transform(_rotation_z(angle))[None, None]
            return np.repeat(transform, novel_pose.num_frames, axis=0), {0: np.zeros((1, 1), dtype=int)}

        with mock.patch.object(PartMotionMatcher, "_match_single", fake_single):
            result, _ = matcher.match(PoseSequence.identity(5, 1))
        expected = _blend_transforms(
            np.stack(
                [
                    _transform(_rotation_z(0.0))[None],
                    _transform(_rotation_z(np.pi / 2.0))[None],
                ]
            ),
            np.array([0.3, 0.7]),
        )[0]
        np.testing.assert_allclose(result, np.broadcast_to(expected, result.shape), atol=1e-12)

    def test_transform_resampling_uses_slerp(self):
        transforms = np.stack(
            [_transform(_rotation_z(0.0)), _transform(_rotation_z(np.pi))]
        )[:, None]
        result = _resample_transforms(transforms, np.array([0, 2]), 3)
        np.testing.assert_allclose(result[1, 0, :3, :3], _rotation_z(np.pi / 2.0), atol=1e-12)

    def test_lbs_uses_vertex_by_bone_weights(self):
        skeleton = Skeleton(np.array([[0.0, 0.0, 0.0]]), np.array([-1]))
        pose = PoseSequence.identity(1, 1)
        transforms = np.stack(
            [_transform(translation=[1.0, 0.0, 0.0]), _transform(translation=[3.0, 0.0, 0.0])]
        )[None]
        bundle = SkeleboneBundle(
            skeleton,
            np.array([[0.25, 0.75]]),
            np.array([[2.0, 0.0, 0.0]]),
            pose,
            transforms,
        )
        result = PartMotionMatcher(
            bundle,
            PartMMConfig(
                num_parts=1,
                patch_size=1,
                num_neighbors=1,
                pyramid_levels=1,
                align_parts=False,
            ),
        ).animate(pose)
        np.testing.assert_allclose(result.points[0, 0], [4.5, 0.0, 0.0], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
