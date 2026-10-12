import tempfile
import unittest
import subprocess
from pathlib import Path
from unittest import mock

import numpy as np

from experiments.multiview_gaussian.pipeline import (
    _d3dgs_train_command,
    _historical_lbs_record,
    _params_record,
    _prediction_record,
    status,
    train_d3dgs,
)
from experiments.multiview_gaussian.oracle_eval import (
    resolve_params,
    sequence_spec,
)
from experiments.multiview_gaussian.ssdr_cache import (
    project_weights,
    select_points,
)
from experiments.multiview_gaussian.train_d3dgs import (
    adopt_checkpoint,
    build_training_command,
)
from experiments.multiview_gaussian.resume_d3dgs import load_output_history_once
from experiments.paper import load_paper_config, render_template


class MultiviewPipelineTest(unittest.TestCase):
    def test_pipeline_training_uses_provenance_wrapper(self):
        config = {
            "release": {"id": "release_v1"},
            "d3dgs": {
                "root": "/external/d3dgs",
                "python": "/env/bin/python",
                "actorshq_output": "/release/d3dgs/actorshq",
                "protocol": {
                    "frames": 300,
                    "first_frame_iterations": 10000,
                    "checkpoint_every_frames": 10,
                    "mask_background": True,
                    "floor_loss": False,
                    "scale_mode_by_dataset": {
                        "actorshq": "isotropic",
                    },
                },
            },
            "_resources": {"roots": {"data": "/prepared/data"}},
        }
        case = {
            "id": "actorshq_actor02_seq2",
            "data": "actorshq_actor02_seq2",
            "dataset": "actorshq",
            "expected_frames": 300,
        }
        command = _d3dgs_train_command(config, case)
        self.assertTrue(command[1].endswith("train_d3dgs.py"))
        self.assertNotIn("resume_d3dgs.py", command)
        self.assertEqual(
            command[command.index("--resource-id") + 1],
            case["id"],
        )
        self.assertEqual(command[command.index("--frames") + 1], "300")
        self.assertEqual(command[command.index("--experiment") + 1], "release_v1")
        self.assertEqual(
            command[command.index("--scale-mode") + 1], "isotropic"
        )

    def test_strict_d3dgs_command_records_protocol_flags(self):
        command = build_training_command(
            d3dgs_root=Path("/external/d3dgs"),
            data_dir=Path("/prepared/data"),
            output_dir=Path("/release/d3dgs"),
            train_id="actorshq_actor02_seq2",
            frames=300,
            iterations=10000,
            checkpoint_every=10,
            experiment="release_v1",
        )
        self.assertIn("--isotropic", command)
        self.assertIn("--no-floor", command)
        self.assertEqual(
            command[command.index("--checkpoint-every") + 1], "10"
        )
        self.assertEqual(command[command.index("--num-timesteps") + 1], "300")

    def test_pipeline_rejects_checkpoint_without_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "case/params.npz"
            destination.parent.mkdir()
            frames, points = 2, 3
            np.savez(
                destination,
                means3D=np.zeros((frames, points, 3), dtype=np.float32),
                rgb_colors=np.zeros((frames, points, 3), dtype=np.float32),
                unnorm_rotations=np.ones(
                    (frames, points, 4), dtype=np.float32
                ),
                log_scales=np.zeros((points, 1), dtype=np.float32),
                logit_opacities=np.zeros((points, 1), dtype=np.float32),
                seg_colors=np.zeros((points, 3), dtype=np.float32),
                cam_m=np.zeros((2, 3), dtype=np.float32),
                cam_c=np.zeros((2, 3), dtype=np.float32),
            )
            case = {
                "id": "case",
                "data": "case",
                "dataset": "actorshq",
                "expected_frames": frames,
                "expected_cameras": 2,
            }
            config = {
                "d3dgs": {
                    "root": str(root / "external"),
                    "python": "/env/bin/python",
                    "actorshq_output": str(root),
                    "logs": str(root / "logs"),
                    "protocol": {
                        "frames": frames,
                        "first_frame_iterations": 10,
                        "checkpoint_every_frames": 1,
                        "mask_background": True,
                        "floor_loss": False,
                        "scale_mode_by_dataset": {
                            "actorshq": "isotropic",
                        },
                    },
                },
                "release": {"id": "release_v1"},
                "_resources": {"roots": {"data": str(root / "data")}},
            }
            with mock.patch(
                "experiments.multiview_gaussian.pipeline._cases",
                return_value=[case],
            ), mock.patch(
                "experiments.multiview_gaussian.pipeline.subprocess.run"
            ) as run:
                with self.assertRaisesRegex(
                    RuntimeError, "missing or invalid training provenance"
                ):
                    train_d3dgs(config, [], dry_run=False)
            run.assert_not_called()

    def test_d3dgs_resume_loader_keeps_one_array_per_dynamic_key(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.npz"
            np.savez(
                path,
                means3D=np.arange(18, dtype=np.float32).reshape(2, 3, 3),
                rgb_colors=np.ones((2, 3, 3), dtype=np.float32),
                unnorm_rotations=np.ones((2, 3, 4), dtype=np.float32),
                log_scales=np.zeros((3, 1), dtype=np.float32),
            )
            history = load_output_history_once(path)
            self.assertEqual(len(history), 2)
            self.assertIn("log_scales", history[0])
            self.assertNotIn("log_scales", history[1])
            self.assertIs(
                history[0]["means3D"].base,
                history[1]["means3D"].base,
            )

    def test_pipeline_status_rejects_existing_but_invalid_resume_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case = {
                "id": "case",
                "data": "case",
                "params": "case",
                "dataset": "actorshq",
                "expected_frames": 5,
                "expected_cameras": 2,
            }
            case_root = root / "d3dgs/case"
            case_root.mkdir(parents=True)
            (case_root / "params_ckpt_latest.npz").write_bytes(b"invalid")
            (case_root / "training_state_latest.pt").write_bytes(b"invalid")
            config = {
                "release": {"id": "release_v1"},
                "d3dgs": {
                    "actorshq_output": str(root / "d3dgs"),
                    "dna_output": str(root / "dna"),
                    "protocol": {
                        "first_frame_iterations": 10,
                        "checkpoint_every_frames": 1,
                        "mask_background": True,
                        "floor_loss": False,
                        "scale_mode_by_dataset": {
                            "actorshq": "isotropic",
                        },
                    },
                },
                "derived": {
                    "runs": str(root / "derived"),
                    "protocol": {
                        "bob_max_points": None,
                        "bob_opacity_threshold": 0,
                        "bob_iterations": 2,
                        "bob_bones": 1,
                        "bob_learning_rate": 0.001,
                        "bob_batch_frames": 1,
                        "bob_adapter_id": (
                            "gaussianimate_dressrecon_composed_pose_bob_v2"
                        ),
                        "bob_joint_mapping_id": (
                            "smpl_joints_1_23_to_rootless_links_v1"
                        ),
                        "bob_pose_joint_range": [1, 24],
                        "bob_pose_dimensions": 69,
                        "bob_skeleton_links": 23,
                        "bob_root_joint_id": 0,
                        "bob_root_pose_dimensions": 3,
                        "bob_root_handling": (
                            "separate_excluded_from_rootless_fk"
                        ),
                        "bob_root_translation_handling": (
                            "not_consumed_by_adapter"
                        ),
                        "bob_predicted_gaussian_attributes": [
                            "means3D"
                        ],
                        "bob_evaluation_attribute_source": (
                            "target_frame"
                        ),
                        "bob_oracle_evaluation_attributes": [
                            "rgb_colors",
                            "unnorm_rotations",
                        ],
                    },
                },
                "diagnostics": {
                    "historical_lbs": str(root / "historical"),
                },
                "_resources": {
                    "roots": {
                        "actorshq_params": str(root / "source"),
                        "dna_params": str(root / "dna_source"),
                    },
                },
            }
            with mock.patch(
                "experiments.multiview_gaussian.pipeline._cases",
                return_value=[case],
            ):
                report = status(config, [])
            d3dgs = report["cases"]["case"]["d3dgs"]
            self.assertFalse(d3dgs["resumable"])
            self.assertFalse(d3dgs["resume_record"]["ready"])
            self.assertTrue(
                Path(
                    d3dgs["resume_record"]["history"]["path"]
                ).is_file()
            )

    def test_d3dgs_adoption_records_hash_and_source_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "train_id/params.npz"
            target = root / "resource_id/params.npz"
            source.parent.mkdir()
            source.write_bytes(b"checkpoint")
            d3dgs_root = root / "d3dgs"
            d3dgs_root.mkdir()
            for name in ("train_dna.py", "helpers.py", "external.py"):
                (d3dgs_root / name).write_text("# test\n", encoding="utf-8")
            subprocess.run(
                ["git", "init", "-q", str(d3dgs_root)],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(d3dgs_root), "add", "."],
                check=True,
            )
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(d3dgs_root),
                    "-c",
                    "user.name=Test",
                    "-c",
                    "user.email=test@example.com",
                    "commit",
                    "-qm",
                    "fixture",
                ],
                check=True,
            )
            record = adopt_checkpoint(
                source,
                target,
                d3dgs_root=d3dgs_root,
                command=["python", "train_dna.py"],
                frames=150,
            )
            self.assertEqual(target.read_bytes(), b"checkpoint")
            self.assertNotEqual(
                source.stat().st_ino,
                target.stat().st_ino,
            )
            self.assertEqual(record["expected_frames"], 150)
            self.assertEqual(len(record["params"]["sha256"]), 64)
            self.assertIn("revision", record["source"])
            self.assertIn("train_dna.py", record["source"]["files"])
            self.assertIn(
                "experiments/multiview_gaussian/resume_d3dgs.py",
                record["adapter"]["files"],
            )
            self.assertTrue(
                (target.parent / "training_provenance.json").is_file()
            )

    def test_ssdr_projection_is_normalized_and_deterministic(self):
        indices = select_points(20, 5, seed=42)
        np.testing.assert_array_equal(indices, select_points(20, 5, seed=42))
        rest = np.arange(15, dtype=np.float32).reshape(5, 3)
        weights = np.eye(5, dtype=np.float32)
        projected = project_weights(rest, rest, weights)
        np.testing.assert_allclose(projected, weights, atol=1e-5)
        np.testing.assert_allclose(projected.sum(axis=1), 1.0)

    def test_params_record_requires_complete_frame_sequence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "params.npz"
            values = {
                "means3D": np.zeros((2, 4, 3), dtype=np.float32),
                "rgb_colors": np.zeros((2, 4, 3), dtype=np.float32),
                "unnorm_rotations": np.ones((2, 4, 4), dtype=np.float32),
                "log_scales": np.zeros((4, 1), dtype=np.float32),
                "logit_opacities": np.zeros((4, 1), dtype=np.float32),
                "seg_colors": np.zeros((4, 3), dtype=np.float32),
                "cam_m": np.zeros((2, 3), dtype=np.float32),
                "cam_c": np.zeros((2, 3), dtype=np.float32),
            }
            np.savez(path, **values)
            self.assertTrue(_params_record(path, 2)["ready"])
            self.assertFalse(_params_record(path, 3)["ready"])

    def test_prediction_record_requires_all_points_and_exact_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frames, points = 10, 3
            (root / "metadata.json").write_text(
                """
{
  "training_frames": 8,
  "test_frame_ids": [8, 9],
  "num_points": 3
}
""".lstrip(),
                encoding="utf-8",
            )
            np.save(
                root / "warped_sequence.npy",
                np.zeros((frames, points, 3), dtype=np.float32),
            )
            np.save(root / "point_indices.npy", np.arange(points))
            self.assertTrue(_prediction_record(root, frames)["ready"])
            np.save(root / "point_indices.npy", np.array([0, 2, 1]))
            self.assertFalse(_prediction_record(root, frames)["ready"])

    def test_historical_lbs_record_validates_protocol_and_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frames = 10
            points = 3
            (root / "metadata.json").write_text(
                """
{
  "n_gaussians": 3,
  "n_total_frames": 10,
  "n_test_frames": 2,
  "train_ratio": 0.8
}
""".lstrip(),
                encoding="utf-8",
            )
            np.save(root / "driven_sequence.npy", np.zeros((2, points, 3)))
            np.save(root / "full_driven_sequence.npy", np.zeros((frames, points, 3)))
            np.save(root / "point_indices.npy", np.arange(points))
            np.save(
                root / "gaussian_weights.npy",
                np.full((points, 24), 1.0 / 24.0),
            )
            self.assertTrue(
                _historical_lbs_record(
                    root, frames, require_provenance=False
                )["ready"]
            )
            self.assertFalse(
                _historical_lbs_record(
                    root, frames, require_provenance=True
                )["ready"]
            )
            (root / "provenance.json").write_text("{}\n", encoding="utf-8")
            self.assertTrue(
                _historical_lbs_record(
                    root, frames, require_provenance=True
                )["ready"]
            )
            weights = np.load(root / "gaussian_weights.npy")
            weights[0, 0] = 2.0
            np.save(root / "gaussian_weights.npy", weights)
            self.assertFalse(
                _historical_lbs_record(
                    root, frames, require_provenance=True
                )["ready"]
            )

    def test_reconstruction_protocol_is_isotropic_and_resumable(self):
        root = Path(__file__).parents[1]
        config = load_paper_config(
            root / "experiments/multiview_gaussian/reconstruction.yaml"
        )
        stage = config["methods"][0]["stages"][0]
        case = config["datasets"][1]["cases"][0]
        rendered = render_template(
            stage,
            {
                "case": case,
                "dataset": config["datasets"][1],
                "run_dir": "/tmp/run",
            },
        )
        command = rendered["command"]
        self.assertIn(
            "experiments.multiview_gaussian.train_d3dgs",
            command,
        )
        self.assertEqual(
            command[command.index("--scale-mode") + 1],
            "isotropic",
        )
        self.assertEqual(
            rendered["outputs"],
            [
                "/mnt/local_nas/wjx/GaussiAnimate_release_v1/resources/"
                f"multiview/d3dgs/actorshq/{case['id']}/params.npz",
                "/mnt/local_nas/wjx/GaussiAnimate_release_v1/resources/"
                f"multiview/d3dgs/actorshq/{case['id']}/training_provenance.json",
            ],
        )

        dna_case = config["datasets"][0]["cases"][0]
        dna_rendered = render_template(
            stage,
            {
                "case": dna_case,
                "dataset": config["datasets"][0],
                "run_dir": "/tmp/run",
            },
        )
        self.assertEqual(
            dna_rendered["command"][
                dna_rendered["command"].index("--scale-mode") + 1
            ],
            "anisotropic",
        )

    def test_oracle_prefers_release_params_and_has_no_lbs_dependency(self):
        self.assertEqual(
            sequence_spec("dna_0012_11")["dataset"],
            "dna_rendering",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            release = (
                root
                / "release"
                / "actorshq"
                / "actorshq_actor01_seq1"
            )
            historical = root / "historical" / "actorshq_actor01_seq1"
            release.mkdir(parents=True)
            historical.mkdir(parents=True)
            (release / "params.npz").touch()
            (historical / "params.npz").touch()
            selected = resolve_params(
                "actorshq_actor01_seq1",
                release_root=root / "release",
                dna_results_dir=root / "dna",
                actorshq_results_dir=root / "historical",
            )
            self.assertEqual(selected, release / "params.npz")


if __name__ == "__main__":
    unittest.main()
