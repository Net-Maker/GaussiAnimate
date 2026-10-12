import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from experiments.multiview_gaussian.human_bob import run_human_bob
from experiments.reproductions.dressrecon_bob.pose_contract import (
    BOB_ADAPTER_ID,
    SMPL_ROOTLESS_JOINTS,
    SMPL_ROOTLESS_POSE_DIM,
    flatten_rootless_pose,
    reshape_rootless_pose,
    smplx_body_to_rootless_links,
    split_smpl_root_and_links,
)
from experiments.reproductions.dressrecon_bob.rotation import (
    axis_angle_to_quaternion,
)


class BoBJointContractTest(unittest.TestCase):
    def test_axis_angle_conversion_is_finite_at_zero_and_differentiable(self):
        rotations = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, np.pi],
                [1.0e-8, -2.0e-8, 3.0e-8],
            ],
            dtype=torch.float64,
            requires_grad=True,
        )

        quaternions = axis_angle_to_quaternion(rotations)

        expected = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [1.0, 0.5e-8, -1.0e-8, 1.5e-8],
            ],
            dtype=torch.float64,
        )
        torch.testing.assert_close(quaternions, expected, atol=1.0e-15, rtol=1.0e-7)
        quaternions.square().sum().backward()
        self.assertTrue(torch.isfinite(rotations.grad).all())

    def test_smpl_joint_ids_one_through_twenty_three_map_without_offset(self):
        axis_angle = np.empty((2, 24, 3), dtype=np.float32)
        for frame in range(2):
            for joint in range(24):
                axis_angle[frame, joint] = (
                    frame * 1000 + joint * 10 + np.arange(3)
                )

        root, links = split_smpl_root_and_links(
            axis_angle,
            expected_frames=2,
        )

        np.testing.assert_array_equal(root, axis_angle[:, 0])
        np.testing.assert_array_equal(
            links.reshape(2, SMPL_ROOTLESS_JOINTS, 3),
            axis_angle[:, 1:24],
        )
        np.testing.assert_array_equal(
            links[:, :3],
            axis_angle[:, 1],
        )
        np.testing.assert_array_equal(
            links[:, -3:],
            axis_angle[:, 23],
        )

    def test_legacy_sixty_six_dimensional_pose_is_rejected(self):
        legacy_pose = np.zeros((4, 66), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "exact shape"):
            reshape_rootless_pose(
                legacy_pose,
                SMPL_ROOTLESS_JOINTS,
                label="SMPL rootless FK pose",
            )
        with self.assertRaisesRegex(ValueError, "exact shape"):
            flatten_rootless_pose(
                legacy_pose,
                SMPL_ROOTLESS_JOINTS,
                label="SMPL fitter pose",
            )

    def test_sixty_nine_dimensional_and_structured_poses_are_equivalent(self):
        flat = np.arange(
            3 * SMPL_ROOTLESS_POSE_DIM,
            dtype=np.float32,
        ).reshape(3, SMPL_ROOTLESS_POSE_DIM)
        structured = flat.reshape(3, SMPL_ROOTLESS_JOINTS, 3)
        np.testing.assert_array_equal(
            reshape_rootless_pose(
                flat,
                SMPL_ROOTLESS_JOINTS,
            ),
            structured,
        )
        np.testing.assert_array_equal(
            flatten_rootless_pose(
                structured,
                SMPL_ROOTLESS_JOINTS,
            ),
            flat,
        )

    def test_smplx_body_adapter_appends_only_terminal_hand_placeholders(self):
        body = np.arange(2 * 21 * 3, dtype=np.float32).reshape(
            2,
            21,
            3,
        )
        links = smplx_body_to_rootless_links(body).reshape(2, 23, 3)
        np.testing.assert_array_equal(links[:, :21], body)
        np.testing.assert_array_equal(
            links[:, 21:],
            np.zeros((2, 2, 3), dtype=np.float32),
        )
        with self.assertRaisesRegex(ValueError, "exact shape"):
            smplx_body_to_rootless_links(
                np.zeros((2, 66), dtype=np.float32)
            )

    def test_complete_smpl_source_shape_and_finiteness_are_strict(self):
        with self.assertRaisesRegex(ValueError, "exact shape"):
            split_smpl_root_and_links(
                np.zeros((2, 24, 4), dtype=np.float32),
                expected_frames=2,
            )
        values = np.zeros((2, 24, 3), dtype=np.float32)
        values[0, 7, 1] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            split_smpl_root_and_links(values, expected_frames=2)

    def test_human_adapter_forwards_only_smpl_joints_one_through_twenty_three(
        self,
    ):
        # Keep torch modules in patch.dict's snapshot so restoring sys.modules
        # does not leave later tests with a partially unloaded torch package.
        import torch  # noqa: F401

        captured = {}

        class FakeFitter:
            def __init__(
                self,
                canonical_pts,
                target_pts,
                smpl_poses,
                num_bones,
                **_kwargs,
            ):
                captured["poses"] = np.asarray(smpl_poses).copy()
                self.target = np.asarray(target_pts)
                self.points = len(canonical_pts)
                self.num_bones = num_bones

            def fit(self, num_iters, **_kwargs):
                return [0.0] * num_iters

            def evaluate(self):
                return {"test_mean": 0.0}

            def get_warped_sequence(self):
                return self.target

            def save(self, path):
                Path(path).write_bytes(b"model")

            def get_skinning_weights(self):
                return np.full(
                    (self.points, self.num_bones),
                    1.0 / self.num_bones,
                    dtype=np.float32,
                )

            def get_skel_skinning_weights(self):
                return np.full(
                    (self.points, SMPL_ROOTLESS_JOINTS),
                    1.0 / SMPL_ROOTLESS_JOINTS,
                    dtype=np.float32,
                )

        fake_module = types.ModuleType(
            "experiments.reproductions.dressrecon_bob.fit_bob"
        )
        fake_module.BoBFitter = FakeFitter

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frames, points = 5, 3
            target = np.zeros((frames, points, 3), dtype=np.float32)
            axis_angle = np.arange(
                frames * 24 * 3,
                dtype=np.float32,
            ).reshape(frames, 24, 3)
            np.savez(root / "params.npz", means3D=target)
            np.savez(
                root / "motion.npz",
                axis_angle=axis_angle,
                case=np.asarray("case"),
                dataset=np.asarray("dataset"),
            )
            with mock.patch.dict(
                sys.modules,
                {fake_module.__name__: fake_module},
            ), mock.patch(
                "experiments.multiview_gaussian.human_bob._source_state",
                return_value={"adapter_id": BOB_ADAPTER_ID},
            ), mock.patch(
                "experiments.multiview_gaussian.human_bob.save_prediction_diagnostic",
                side_effect=lambda path, *_args, **_kwargs: Path(
                    path
                ).write_bytes(b"diagnostic"),
            ):
                metadata = run_human_bob(
                    root / "params.npz",
                    root / "motion.npz",
                    root / "output",
                    dressrecon_root=root,
                    num_bones=2,
                    num_iterations=2,
                    batch_frames=1,
                    device="cpu",
                )

        np.testing.assert_array_equal(
            captured["poses"].reshape(
                frames,
                SMPL_ROOTLESS_JOINTS,
                3,
            ),
            axis_angle[:, 1:24],
        )
        self.assertEqual(
            metadata["protocol"]["pose_joint_range"],
            [1, 24],
        )
        self.assertEqual(
            metadata["protocol"]["root_handling"],
            "separate_excluded_from_rootless_fk",
        )


if __name__ == "__main__":
    unittest.main()
