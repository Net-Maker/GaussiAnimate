import tempfile
import unittest
from pathlib import Path

import numpy as np

from partmm import PartMMConfig, animate
from partmm.gui import PoseEditorState
from skelebone import (
    PoseSequence,
    Skeleton,
    SkeleboneBundle,
    export_bvh,
    fit_pose_to_joint_positions,
    forward_kinematics,
)
from skelebone.io import (
    load_bundle,
    load_point_sequence,
    point_sequence_summary,
    save_bundle,
    save_point_sequence,
)


class SkeleboneCoreTest(unittest.TestCase):
    def setUp(self):
        self.skeleton = Skeleton(
            rest_joints=np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float),
            parents=np.array([-1, 0, 1]),
            names=["root", "middle", "tip"],
        )

    def test_identity_fk_reproduces_bind_pose(self):
        pose = PoseSequence.identity(2, 3, fps=24)
        _, positions = forward_kinematics(self.skeleton, pose)
        np.testing.assert_allclose(positions, np.broadcast_to(self.skeleton.rest_joints, (2, 3, 3)))

    def test_pose_fit_recovers_simple_bend(self):
        expected = PoseSequence.identity(1, 3)
        angle = np.pi / 3
        expected.local_rotations[0, 1] = [np.cos(angle / 2), 0, 0, np.sin(angle / 2)]
        _, positions = forward_kinematics(self.skeleton, expected)
        fitted = fit_pose_to_joint_positions(self.skeleton, positions)
        _, reconstructed = forward_kinematics(self.skeleton, fitted)
        np.testing.assert_allclose(reconstructed, positions, atol=1e-7)

    def test_partmm_exact_motion_retrieval(self):
        angle = np.pi / 2
        source_pose = PoseSequence.identity(2, 3)
        source_pose.local_rotations[1, 0] = [np.cos(angle / 2), 0, 0, np.sin(angle / 2)]
        transforms = np.broadcast_to(np.eye(4), (2, 1, 4, 4)).copy()
        transforms[1, 0, :3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        bundle = SkeleboneBundle(
            self.skeleton,
            weights=np.ones((2, 1), dtype=float),
            source_points=np.array([[1, 0, 0], [0, 1, 0]], dtype=float),
            motion=source_pose,
            bone_transforms=transforms,
        )
        result = animate(
            bundle,
            PoseSequence(source_pose.local_rotations[1:], source_pose.root_translation[1:]),
            PartMMConfig(num_parts=1, patch_size=1, num_neighbors=1, pyramid_levels=1, align_parts=False),
        )
        np.testing.assert_allclose(result.points[0], [[0, 1, 0], [-1, 0, 0]], atol=1e-7)

    def test_bundle_round_trip_and_bvh(self):
        pose = PoseSequence.identity(3, 3, fps=25)
        bundle = SkeleboneBundle(
            self.skeleton,
            weights=np.eye(3),
            source_points=self.skeleton.rest_joints,
            motion=pose,
            bone_transforms=np.broadcast_to(np.eye(4), (3, 3, 4, 4)).copy(),
            artifacts={"curve_skeleton": "curve/curve.obj"},
        )
        with tempfile.TemporaryDirectory() as directory:
            bundle_path = save_bundle(Path(directory) / "bundle.npz", bundle)
            loaded = load_bundle(bundle_path)
            np.testing.assert_allclose(loaded.weights, bundle.weights)
            bvh_path = export_bvh(Path(directory) / "motion.bvh", loaded.skeleton, loaded.motion)
            text = bvh_path.read_text()
            self.assertIn("Frames: 3", text)
            self.assertIn("ROOT root", text)

    def test_pose_editor_changes_selected_joint(self):
        pose = PoseSequence.identity(2, 3)
        bundle = SkeleboneBundle(
            self.skeleton, np.eye(3), self.skeleton.rest_joints, pose,
            np.broadcast_to(np.eye(4), (2, 3, 4, 4)).copy(),
        )
        editor = PoseEditorState(bundle)
        editor.joint = 1
        editor.set_euler_degrees([0, 0, 90])
        np.testing.assert_allclose(editor.euler_degrees(), [0, 0, 90], atol=1e-7)

    def test_public_point_sequence_round_trip(self):
        points = np.stack([self.skeleton.rest_joints, self.skeleton.rest_joints + 1])
        with tempfile.TemporaryDirectory() as directory:
            path = save_point_sequence(
                Path(directory) / "sequence.npz",
                points,
                faces=np.array([[0, 1, 2]]),
                fps=24,
            )
            sequence = load_point_sequence(path)
        np.testing.assert_allclose(sequence.points, points)
        self.assertEqual(point_sequence_summary(sequence)["frames"], 2)
        self.assertEqual(point_sequence_summary(sequence)["points_per_frame"], 3)
        self.assertEqual(point_sequence_summary(sequence)["fps"], 24)


if __name__ == "__main__":
    unittest.main()
