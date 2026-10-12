import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.multiview_gaussian.pipeline import (
    _params_record,
    _prediction_record,
    _ssdr_record,
)
from experiments.multiview_gaussian.release_validation import (
    validate_d3dgs_params,
    validate_d3dgs_provenance,
    validate_d3dgs_resume_checkpoint,
    validate_method_prediction,
    validate_prediction_artifact,
    validate_ssdr_cache,
)
from experiments.multiview_gaussian.resources import (
    _params,
    _prediction_files,
    _ssdr_files,
)


def _write_params(path: Path, frames: int = 5, points: int = 3) -> None:
    np.savez(
        path,
        means3D=np.zeros((frames, points, 3), dtype=np.float32),
        rgb_colors=np.zeros((frames, points, 3), dtype=np.float32),
        unnorm_rotations=np.ones(
            (frames, points, 4),
            dtype=np.float32,
        ),
        log_scales=np.zeros((points, 1), dtype=np.float32),
        logit_opacities=np.zeros((points, 1), dtype=np.float32),
        seg_colors=np.zeros((points, 3), dtype=np.float32),
        cam_m=np.zeros((2, 3), dtype=np.float32),
        cam_c=np.zeros((2, 3), dtype=np.float32),
    )


def _write_prediction(
    directory: Path,
    *,
    frames: int = 5,
    points: int = 3,
    frame_ids: bool = False,
) -> None:
    directory.mkdir(parents=True)
    training_frames = int(frames * 0.8)
    test_ids = list(range(training_frames, frames))
    (directory / "metadata.json").write_text(
        json.dumps(
            {
                "training_frames": training_frames,
                "test_frame_ids": test_ids,
                "num_points": points,
                "method": "fixture",
            }
        ),
        encoding="utf-8",
    )
    np.save(
        directory / "warped_sequence.npy",
        np.zeros((frames, points, 3), dtype=np.float32),
    )
    np.save(
        directory / "test_warped_sequence.npy",
        np.zeros(
            (frames - training_frames, points, 3),
            dtype=np.float32,
        ),
    )
    np.save(
        directory / "point_indices.npy",
        np.arange(points, dtype=np.int64),
    )
    if frame_ids:
        np.save(
            directory / "frame_ids.npy",
            np.asarray(test_ids, dtype=np.int64),
        )


def _write_ssdr(
    path: Path,
    *,
    transpose_weights: bool,
    raw_transforms: bool = False,
) -> None:
    frames, training_frames, points, bones = 5, 4, 3, 2
    weights = np.full(
        (points, bones),
        1.0 / bones,
        dtype=np.float32,
    )
    arrays = {
        "training_frames": np.asarray(training_frames),
        "rest_pose": np.zeros((points, 3), dtype=np.float32),
        "weights": weights.T if transpose_weights else weights,
    }
    transforms = np.broadcast_to(
        np.eye(4, dtype=np.float32),
        (training_frames, bones, 4, 4),
    )
    if raw_transforms:
        arrays["transformations"] = transforms
        arrays["frame_means"] = np.zeros(
            (training_frames, 3),
            dtype=np.float32,
        )
    else:
        arrays["world_transformations"] = transforms
    np.savez(path, **arrays)
    self_check = validate_ssdr_cache(path, frames)
    if not self_check["ready"]:
        raise AssertionError(self_check)


def _write_resume_state(
    path: Path,
    *,
    completed_timestep: int,
    points: int = 3,
    scale_dims: int = 1,
    cameras: int = 2,
    schema_version: int = 1,
) -> None:
    import torch

    params = {
        "means3D": torch.zeros((points, 3)),
        "rgb_colors": torch.zeros((points, 3)),
        "unnorm_rotations": torch.ones((points, 4)),
        "log_scales": torch.zeros((points, scale_dims)),
        "logit_opacities": torch.zeros((points, 1)),
        "seg_colors": torch.zeros((points, 3)),
        "cam_m": torch.zeros((cameras, 3)),
        "cam_c": torch.zeros((cameras, 3)),
    }
    torch.save(
        {
            "schema_version": schema_version,
            "completed_timestep": completed_timestep,
            "params": params,
            "variables": {},
            "optimizer": {
                "state": {},
                "param_groups": [],
            },
        },
        str(path),
    )


class MultiviewReleaseValidationTest(unittest.TestCase):
    def test_d3dgs_provenance_binds_checkpoint_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            params = root / "params.npz"
            params.write_bytes(b"checkpoint")
            digest = hashlib.sha256(params.read_bytes()).hexdigest()
            provenance = root / "training_provenance.json"
            provenance.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "method": "Dynamic3DGaussians",
                        "expected_frames": 300,
                        "params": {
                            "path": str(params),
                            "size_bytes": params.stat().st_size,
                            "sha256": digest,
                        },
                        "source": {
                            "path": "/source/params.npz",
                            "revision": "a" * 40,
                        },
                        "command": [
                            "python",
                            "/repo/resume_d3dgs.py",
                            "--num-timesteps",
                            "300",
                            "--num-iters",
                            "10000",
                            "--isotropic",
                            "--no-floor",
                            "--checkpoint-every",
                            "10",
                        ],
                    }
                ),
                encoding="utf-8",
            )
            record = validate_d3dgs_provenance(
                provenance,
                params,
                300,
                expected_protocol={
                    "scale_mode": "isotropic",
                    "frames": 300,
                    "first_frame_iterations": 10000,
                    "checkpoint_every_frames": 10,
                    "mask_background": True,
                    "floor_loss": False,
                },
                full=True,
            )
            self.assertTrue(record["ready"])
            self.assertTrue(record["sha256_matches"])
            self.assertTrue(record["strict_ready"])
            params.write_bytes(b"changed")
            self.assertFalse(
                validate_d3dgs_provenance(
                    provenance, params, 300, full=True
                )["ready"]
            )

    def test_pipeline_and_resource_checkpoint_records_are_identical(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "params.npz"
            _write_params(path)
            self.assertEqual(
                _params_record(path, 5),
                _params(path, 5),
            )

            with np.load(path, allow_pickle=False) as archive:
                values = {
                    key: np.asarray(archive[key])
                    for key in archive.files
                }
            values["means3D"][0, 0, 0] = np.nan
            np.savez(path, **values)
            self.assertFalse(_params_record(path, 5)["ready"])
            self.assertEqual(
                _params_record(path, 5),
                _params(path, 5),
            )

    def test_checkpoint_contract_checks_scale_cameras_and_rotations(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "params.npz"
            _write_params(path)
            self.assertTrue(
                validate_d3dgs_params(
                    path,
                    5,
                    expected_scale_dims=1,
                    expected_cameras=2,
                    full=True,
                )["ready"]
            )
            self.assertFalse(
                validate_d3dgs_params(
                    path,
                    5,
                    expected_scale_dims=3,
                    expected_cameras=2,
                    full=True,
                )["ready"]
            )
            with np.load(path, allow_pickle=False) as archive:
                values = {
                    key: np.asarray(archive[key])
                    for key in archive.files
                }
            values["unnorm_rotations"] = np.zeros_like(
                values["unnorm_rotations"]
            )
            np.savez(path, **values)
            self.assertFalse(
                validate_d3dgs_params(path, 5, full=True)["ready"]
            )

    def test_resume_checkpoint_validates_history_and_training_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_params(
                root / "params_ckpt_latest.npz",
                frames=3,
                points=3,
            )
            _write_resume_state(
                root / "training_state_latest.pt",
                completed_timestep=2,
            )
            record = validate_d3dgs_resume_checkpoint(
                root,
                5,
                expected_scale_dims=1,
                expected_cameras=2,
                full=True,
            )
            self.assertTrue(record["ready"], record)
            self.assertEqual(record["history_frames"], 3)
            self.assertTrue(record["timesteps_match"])
            self.assertTrue(record["state"]["shape_valid"])

    def test_resume_checkpoint_rejects_cross_file_and_shape_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_params(
                root / "params_ckpt_latest.npz",
                frames=3,
                points=3,
            )
            state_path = root / "training_state_latest.pt"
            _write_resume_state(
                state_path,
                completed_timestep=1,
            )
            mismatched_time = validate_d3dgs_resume_checkpoint(
                root,
                5,
                expected_scale_dims=1,
                expected_cameras=2,
            )
            self.assertFalse(mismatched_time["ready"])
            self.assertFalse(mismatched_time["timesteps_match"])

            _write_resume_state(
                state_path,
                completed_timestep=2,
                points=4,
            )
            mismatched_shape = validate_d3dgs_resume_checkpoint(
                root,
                5,
                expected_scale_dims=1,
                expected_cameras=2,
            )
            self.assertFalse(mismatched_shape["ready"])
            self.assertFalse(mismatched_shape["state"]["shape_valid"])

    def test_resume_checkpoint_rejects_training_state_schema_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_params(
                root / "params_ckpt_latest.npz",
                frames=3,
                points=3,
            )
            _write_resume_state(
                root / "training_state_latest.pt",
                completed_timestep=2,
                schema_version=2,
            )
            record = validate_d3dgs_resume_checkpoint(root, 5)
            self.assertFalse(record["ready"])
            self.assertTrue(record["state"]["readable"])
            self.assertFalse(record["state"]["schema_valid"])

    def test_resume_checkpoint_rejects_scale_camera_and_corrupt_npz(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            history_path = root / "params_ckpt_latest.npz"
            _write_params(history_path, frames=3, points=3)
            _write_resume_state(
                root / "training_state_latest.pt",
                completed_timestep=2,
            )
            self.assertFalse(
                validate_d3dgs_resume_checkpoint(
                    root,
                    5,
                    expected_scale_dims=3,
                    expected_cameras=2,
                )["ready"]
            )
            self.assertFalse(
                validate_d3dgs_resume_checkpoint(
                    root,
                    5,
                    expected_scale_dims=1,
                    expected_cameras=3,
                )["ready"]
            )
            history_path.write_bytes(b"not an npz archive")
            corrupt = validate_d3dgs_resume_checkpoint(root, 5)
            self.assertFalse(corrupt["ready"])
            self.assertFalse(corrupt["history"]["readable"])

    def test_pipeline_and_resource_prediction_records_are_identical(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "prediction"
            _write_prediction(root)
            pipeline = _prediction_record(
                root,
                5,
                extra_required=("test_warped_sequence.npy",),
            )
            resources = _prediction_files(root, 5, (), full=False)
            self.assertTrue(pipeline["ready"])
            self.assertEqual(pipeline, resources)
            self.assertTrue(pipeline["legacy_missing_frame_ids"])

    def test_optional_frame_ids_are_checked_when_present(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "prediction"
            _write_prediction(root, frame_ids=True)
            self.assertTrue(
                validate_prediction_artifact(
                    root,
                    5,
                    require_test_prediction=True,
                    full=True,
                )["ready"]
            )
            np.save(root / "frame_ids.npy", np.asarray([0]))
            self.assertFalse(
                validate_prediction_artifact(
                    root,
                    5,
                    require_test_prediction=True,
                    full=True,
                )["ready"]
            )

    def test_method_contract_requires_method_specific_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "bob"
            _write_prediction(root)
            missing = validate_method_prediction(root, 5, "bob")
            self.assertFalse(missing["ready"])
            self.assertEqual(
                set(missing["missing"]),
                {
                    "bob_model.pt",
                    "loss_history.npy",
                    "skinning_weights.npy",
                    "skeleton_skinning_weights.npy",
                    "diagnostic.png",
                },
            )
            for name in (
                "bob_model.pt",
                "diagnostic.png",
            ):
                (root / name).write_bytes(b"fixture")
            np.save(root / "loss_history.npy", np.ones(2))
            np.save(
                root / "skinning_weights.npy",
                np.ones((3, 1), dtype=np.float32),
            )
            np.save(
                root / "skeleton_skinning_weights.npy",
                np.full((3, 23), 1.0 / 23, dtype=np.float32),
            )
            metadata_path = root / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata.update(
                method="smpl_conditioned_bag_of_bones",
                protocol={
                    "num_iterations": 2,
                    "num_bones": 1,
                    "learning_rate": 0.001,
                    "batch_frames": 8,
                    "driven_points": "all_gaussians",
                    "opacity_filter": None,
                    "point_subsampling": None,
                    "adapter_id": (
                        "gaussianimate_dressrecon_composed_pose_bob_v2"
                    ),
                    "joint_mapping_id": (
                        "smpl_joints_1_23_to_rootless_links_v1"
                    ),
                    "pose_joint_range": [1, 24],
                    "pose_dimensions": 69,
                    "skeleton_links": 23,
                    "root_joint_id": 0,
                    "root_pose_dimensions": 3,
                    "root_handling": (
                        "separate_excluded_from_rootless_fk"
                    ),
                    "root_translation_handling": (
                        "not_consumed_by_adapter"
                    ),
                    "predicted_gaussian_attributes": ["means3D"],
                    "evaluation_attribute_source": "target_frame",
                    "oracle_evaluation_attributes": [
                        "rgb_colors",
                        "unnorm_rotations",
                    ],
                },
                implementation={
                    "adapter_id": (
                        "gaussianimate_dressrecon_composed_pose_bob_v2"
                    ),
                },
            )
            metadata_path.write_text(
                json.dumps(metadata),
                encoding="utf-8",
            )
            self.assertTrue(
                validate_method_prediction(root, 5, "bob")["ready"]
            )
            expected = {
                "method": "smpl_conditioned_bag_of_bones",
                **metadata["protocol"],
            }
            self.assertTrue(
                validate_method_prediction(
                    root,
                    5,
                    "bob",
                    expected_protocol=expected,
                )["ready"]
            )
            expected["num_iterations"] = 2000
            self.assertFalse(
                validate_method_prediction(
                    root,
                    5,
                    "bob",
                    expected_protocol=expected,
                )["ready"]
            )
            expected["num_iterations"] = 2
            metadata["protocol"]["pose_dimensions"] = 66
            metadata_path.write_text(
                json.dumps(metadata),
                encoding="utf-8",
            )
            invalid = validate_method_prediction(
                root,
                5,
                "bob",
                expected_protocol=expected,
            )
            self.assertFalse(invalid["ready"])
            self.assertFalse(
                invalid["method_artifacts"]["pose_contract_matches"]
            )

    def test_method_contract_rejects_subsampled_prediction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "bob"
            _write_prediction(root, points=3)
            for name in (
                "bob_model.pt",
                "diagnostic.png",
            ):
                (root / name).write_bytes(b"fixture")
            np.save(root / "loss_history.npy", np.ones(2))
            np.save(
                root / "skinning_weights.npy",
                np.ones((3, 1), dtype=np.float32),
            )
            np.save(
                root / "skeleton_skinning_weights.npy",
                np.ones((3, 1), dtype=np.float32),
            )
            record = validate_method_prediction(
                root,
                5,
                "bob",
                expected_points=4,
            )
            self.assertFalse(record["ready"])
            self.assertFalse(record["all_points"])
            self.assertEqual(record["points"], 3)
            self.assertEqual(record["expected_points"], 4)

    def test_partmm_contract_binds_algorithm_and_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "partmm"
            _write_prediction(root)
            (root / "diagnostic.png").write_bytes(b"fixture")
            np.save(
                root / "bone_transforms.npy",
                np.broadcast_to(
                    np.eye(4, dtype=np.float32),
                    (1, 2, 4, 4),
                ),
            )
            expected = {
                "algorithm": "paper_spec_partmm",
                "num_parts": 5,
                "patch_size": 7,
                "num_neighbors": 7,
                "pyramid_levels": 5,
                "pyramid_blend": 0.7,
                "overlap_rings": 1,
                "align_parts": True,
                "alignment_mode": "edge_wise",
                "partition_mode": "humanoid_five",
                "bone_overlap_ratio": 1.5,
            }
            metadata_path = root / "metadata.json"
            metadata = json.loads(
                metadata_path.read_text(encoding="utf-8")
            )
            metadata.update(
                algorithm=expected["algorithm"],
                partmm={
                    key: value
                    for key, value in expected.items()
                    if key != "algorithm"
                },
            )
            metadata_path.write_text(
                json.dumps(metadata),
                encoding="utf-8",
            )
            record = validate_method_prediction(
                root,
                5,
                "partmm",
                expected_protocol=expected,
            )
            self.assertTrue(record["ready"])
            self.assertTrue(
                record["method_artifacts"]["protocol_matches"]
            )
            incomplete = dict(expected)
            incomplete.pop("bone_overlap_ratio")
            self.assertFalse(
                validate_method_prediction(
                    root,
                    5,
                    "partmm",
                    expected_protocol=incomplete,
                )["ready"]
            )

            metadata["partmm"]["partition_mode"] = "balanced_tree"
            metadata_path.write_text(
                json.dumps(metadata),
                encoding="utf-8",
            )
            self.assertFalse(
                validate_method_prediction(
                    root,
                    5,
                    "partmm",
                    expected_protocol=expected,
                )["ready"]
            )

            metadata["partmm"]["partition_mode"] = "humanoid_five"
            metadata["algorithm"] = "paper_partmm"
            metadata_path.write_text(
                json.dumps(metadata),
                encoding="utf-8",
            )
            self.assertFalse(
                validate_method_prediction(
                    root,
                    5,
                    "partmm",
                    expected_protocol=expected,
                )["ready"]
            )

    def test_ssdr_accepts_both_weight_layouts_and_raw_transforms(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            modern = root / "modern.npz"
            historical = root / "historical.npz"
            raw = root / "raw.npz"
            _write_ssdr(modern, transpose_weights=False)
            _write_ssdr(historical, transpose_weights=True)
            _write_ssdr(
                raw,
                transpose_weights=True,
                raw_transforms=True,
            )
            self.assertEqual(
                validate_ssdr_cache(modern, 5)["weights_layout"],
                "points_bones",
            )
            self.assertEqual(
                validate_ssdr_cache(historical, 5)["weights_layout"],
                "bones_points",
            )
            self.assertEqual(
                validate_ssdr_cache(raw, 5)["transform_key"],
                "transformations",
            )

    def test_pipeline_and_resource_ssdr_records_are_identical(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.npz"
            _write_ssdr(path, transpose_weights=True)
            self.assertEqual(
                _ssdr_record(path, 5),
                _ssdr_files(path, 5),
            )


if __name__ == "__main__":
    unittest.main()
