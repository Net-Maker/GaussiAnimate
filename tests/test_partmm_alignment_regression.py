import unittest

import numpy as np

from partmm import PartMMConfig, PartMMInput, PartMMQuery, PartMotionMatcher
from partmm.kinematics import forward_kinematics


class FrameAlignmentTest(unittest.TestCase):
    def test_rigid_transfer_tracks_every_frame_of_patch(self):
        n = 13
        rest = np.array([[2., 1., 0.], [3., 1., 0.], [2., 2., 0.]])
        rotations = np.zeros((n, 3, 4))
        rotations[..., 0] = 1
        source = PartMMQuery(rotations, np.zeros((n, 3)))
        query_rotations = rotations.copy()
        angle = np.linspace(0, 1.2, n) ** 2
        query_rotations[:, 0, 0] = np.cos(angle / 2)
        query_rotations[:, 0, 3] = np.sin(angle / 2)
        query = PartMMQuery(query_rotations, np.c_[angle ** 2, angle, angle * 0])
        bundle = PartMMInput(
            rest_joints=rest, parents=[-1, 0, 0],
            source_local_rotations=source.local_rotations,
            source_root_translation=source.root_translation,
            source_points=rest, skinning_weights=np.ones((3, 1)),
            bone_transforms=np.tile(np.eye(4), (n, 1, 1, 1)),
        )
        _, expected = forward_kinematics(bundle.skeleton, query)
        for mode in ['edge_wise', 'part_rigid']:
            with self.subTest(mode=mode):
                settings = dict(num_parts=1, patch_size=7, num_neighbors=3,
                                pyramid_levels=1, overlap_rings=0, alignment_mode=mode)
                old = PartMotionMatcher(bundle, PartMMConfig(**settings, alignment_frames='patch_center')).animate(query)
                self.assertGreater(np.linalg.norm(old.points - expected), .1)
                fixed = PartMotionMatcher(bundle, PartMMConfig(**settings, alignment_frames='per_frame')).animate(query)
                np.testing.assert_allclose(fixed.points, expected, atol=1e-8)

    def test_alignment_frames_validation(self):
        with self.assertRaises(ValueError):
            PartMMConfig(alignment_frames='invalid')


if __name__ == '__main__':
    unittest.main()
