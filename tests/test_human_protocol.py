import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.multiview_gaussian.human_partmm import run_human_partmm
from experiments.multiview_gaussian.human_pose import axis_angle_to_quaternion
from partmm import PartMMConfig


class HumanProtocolTest(unittest.TestCase):
    def test_axis_angle_conversion_is_scalar_first(self):
        rotations = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, np.pi],
            ]
        )
        quaternions = axis_angle_to_quaternion(rotations)
        np.testing.assert_allclose(
            quaternions[0], [1.0, 0.0, 0.0, 0.0], atol=1e-12
        )
        np.testing.assert_allclose(
            np.abs(quaternions[1]), [0.0, 0.0, 0.0, 1.0], atol=1e-12
        )

    def test_human_partmm_rejects_all_frame_ssdr_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frames = 10
            points = 2
            np.savez(
                root / "params.npz",
                means3D=np.zeros((frames, points, 3), dtype=np.float32),
            )
            np.savez(
                root / "cache.npz",
                rest_pose=np.zeros((points, 3), dtype=np.float32),
                weights=np.ones((1, points), dtype=np.float32),
                world_transformations=np.broadcast_to(
                    np.eye(4), (frames, 1, 4, 4)
                ),
                training_frames=np.asarray(frames),
            )
            np.savez(
                root / "motion.npz",
                rest_joints=np.zeros((1, 3), dtype=np.float32),
                parents=np.array([-1]),
                local_rotations=np.broadcast_to(
                    [1.0, 0.0, 0.0, 0.0], (frames, 1, 4)
                ),
                root_translation=np.zeros((frames, 3), dtype=np.float32),
            )
            with self.assertRaisesRegex(ValueError, "not training-only"):
                run_human_partmm(
                    root / "params.npz",
                    root / "cache.npz",
                    root / "motion.npz",
                    root / "output",
                )

    def test_human_partmm_runs_on_training_only_synthetic_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frames = 10
            training_frames = 8
            rest = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
            translations = np.arange(frames, dtype=np.float64)[:, None] * np.array(
                [[0.1, 0.0, 0.0]]
            )
            points = rest[None] + translations[:, None]
            transforms = np.broadcast_to(
                np.eye(4), (training_frames, 1, 4, 4)
            ).copy()
            transforms[:, 0, :3, 3] = translations[:training_frames]
            np.savez(root / "params.npz", means3D=points.astype(np.float32))
            np.savez(
                root / "cache.npz",
                rest_pose=rest.astype(np.float32),
                weights=np.ones((1, len(rest)), dtype=np.float32),
                world_transformations=transforms.astype(np.float32),
                training_frames=np.asarray(training_frames),
            )
            np.savez(
                root / "motion.npz",
                rest_joints=np.zeros((1, 3), dtype=np.float32),
                parents=np.array([-1]),
                local_rotations=np.broadcast_to(
                    [1.0, 0.0, 0.0, 0.0], (frames, 1, 4)
                ),
                root_translation=translations.astype(np.float32),
                fps=np.asarray(30.0),
            )
            metadata = run_human_partmm(
                root / "params.npz",
                root / "cache.npz",
                root / "motion.npz",
                root / "output",
                config=PartMMConfig(
                    num_parts=1,
                    patch_size=1,
                    num_neighbors=1,
                    pyramid_levels=1,
                    align_parts=True,
                ),
            )
            self.assertEqual(metadata["training_frames"], training_frames)
            self.assertEqual(metadata["test_frames"], 2)
            self.assertEqual(metadata["test_frame_ids"], [8, 9])
            self.assertEqual(metadata["num_points"], len(rest))
            full = np.load(root / "output/warped_sequence.npy")
            test = np.load(root / "output/test_warped_sequence.npy")
            self.assertEqual(full.shape, points.shape)
            self.assertEqual(test.shape, points[training_frames:].shape)
            np.testing.assert_allclose(
                full[training_frames:],
                test,
                atol=1e-6,
            )


if __name__ == "__main__":
    unittest.main()
