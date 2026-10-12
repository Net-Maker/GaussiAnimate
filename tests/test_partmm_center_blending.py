import unittest

import numpy as np

from partmm.transforms import blend_transforms, resample_transforms
from dataclasses import replace
from partmm import PartMMInput, PartMMQuery, PartMMConfig, PartMotionMatcher


def rotations_about(center):
    transforms = np.tile(np.eye(4), (2, 1, 1))
    for transform, angle in zip(transforms, [-np.pi / 3, np.pi / 3]):
        c, s = np.cos(angle), np.sin(angle)
        transform[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
        transform[:3, 3] = center - transform[:3, :3] @ center
    return transforms


class CenterBlendingTest(unittest.TestCase):
    def test_fixed_pivot_stays_fixed(self):
        center = np.array([2., 1., 0.])
        transforms = rotations_about(center)
        legacy = blend_transforms(transforms, [.5, .5])
        self.assertGreater(np.linalg.norm(legacy[:3, :3] @ center + legacy[:3, 3] - center), 1)
        blended = blend_transforms(transforms, [.5, .5], centers=center)
        np.testing.assert_allclose(blended[:3, :3] @ center + blended[:3, 3], center, atol=1e-12)

    def test_changing_canonical_origin_does_not_change_world_result(self):
        center = np.array([2., 1., 0.])
        offset = np.array([3., -4., .5])
        transforms = rotations_about(center)
        shifted = transforms.copy()
        shifted[:, :3, 3] -= np.einsum('kij,j->ki', transforms[:, :3, :3], offset)
        first = blend_transforms(transforms, [.3, .7], centers=center)
        second = blend_transforms(shifted, [.3, .7], centers=center + offset)
        point = np.array([.2, .3, .4])
        np.testing.assert_allclose(first[:3, :3] @ point + first[:3, 3],
                                   second[:3, :3] @ (point + offset) + second[:3, 3], atol=1e-12)

    def test_temporal_resampling_preserves_pivot(self):
        center = np.array([2., 1., 0.])
        result = resample_transforms(rotations_about(center)[:, None], np.array([0, 4]), 5, centers=center[None])
        images = np.einsum('tbij,bj->tbi', result[..., :3, :3], center[None]) + result[..., :3, 3]
        np.testing.assert_allclose(images, np.broadcast_to(center, images.shape), atol=1e-12)

    def test_full_matcher_is_independent_of_canonical_origin(self):
        frames = 17
        rotations = np.zeros((frames, 3, 4))
        rotations[..., 0] = 1
        transforms = np.tile(np.eye(4), (frames, 1, 1, 1))
        for frame, angle in enumerate(np.linspace(-1., 1., frames)):
            c, s = np.cos(angle), np.sin(angle)
            transforms[frame, 0, :3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
        bundle = PartMMInput(
            rest_joints=[[0, 0, 0], [1, 0, 0], [0, 1, 0]], parents=[-1, 0, 0],
            source_local_rotations=rotations, source_root_translation=np.zeros((frames, 3)),
            bone_transforms=transforms, source_points=[[2., 1., 0.], [2.1, .9, .1]],
            skinning_weights=np.ones((2, 1)),
        )
        offset = np.array([3., -4., .5])
        shifted_transforms = transforms.copy()
        shifted_transforms[..., :3, 3] -= np.einsum('tbij,j->tbi', transforms[..., :3, :3], offset)
        shifted = replace(bundle, source_points=bundle.source_points + offset, bone_transforms=shifted_transforms)
        query = PartMMQuery(rotations[:9], np.zeros((9, 3)))
        config = PartMMConfig(num_parts=1, patch_size=3, pyramid_levels=3,
                              alignment_mode='edge_wise', blend_space='bone_center')
        first = PartMotionMatcher(bundle, config).animate(query).points
        second = PartMotionMatcher(shifted, config).animate(query).points
        np.testing.assert_allclose(first, second, atol=1e-10)


if __name__ == '__main__':
    unittest.main()
