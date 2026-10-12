import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.multiview_gaussian.resources import (
    _derived_files,
    _metadata,
    _params,
    audit_resources,
    prepare_resource_links,
)


class MultiviewResourceTest(unittest.TestCase):
    def test_metadata_checks_every_referenced_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "case"
            (root / "ims/0").mkdir(parents=True)
            (root / "seg/0").mkdir(parents=True)
            metadata = {
                "w": 8,
                "h": 8,
                "fn": [["0/000000.jpg"], ["0/000001.jpg"]],
                "k": [[np.eye(3).tolist()], [np.eye(3).tolist()]],
                "w2c": [[np.eye(4).tolist()], [np.eye(4).tolist()]],
            }
            (root / "train_meta.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            for frame in ("000000", "000001"):
                (root / f"ims/0/{frame}.jpg").write_bytes(b"image")
                (root / f"seg/0/{frame}.png").write_bytes(b"mask")
            np.savez(
                root / "init_pt_cld.npz",
                data=np.zeros((4, 7), dtype=np.float32),
            )
            result = _metadata(root, 2, 1, full=True)
            self.assertTrue(result["ready"])
            (root / "seg/0/000001.png").unlink()
            self.assertFalse(_metadata(root, 2, 1, full=True)["ready"])

    def test_params_require_the_multiview_schema_and_frame_count(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "params.npz"
            common = {
                "means3D": np.zeros((2, 3, 3), dtype=np.float32),
                "rgb_colors": np.zeros((2, 3, 3), dtype=np.float32),
                "unnorm_rotations": np.ones((2, 3, 4), dtype=np.float32),
                "log_scales": np.zeros((3, 1), dtype=np.float32),
                "logit_opacities": np.zeros((3, 1), dtype=np.float32),
                "seg_colors": np.zeros((3, 3), dtype=np.float32),
                "cam_m": np.zeros((2, 3), dtype=np.float32),
                "cam_c": np.zeros((2, 3), dtype=np.float32),
            }
            np.savez(path, **common)
            self.assertTrue(_params(path, 2)["ready"])
            self.assertFalse(_params(path, 3)["ready"])

    def test_link_preparation_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            data = root / "data_source"
            smplx = root / "smplx"
            case_raw = raw / "Actor01/Sequence1/smplx"
            for path in (data, smplx, case_raw):
                path.mkdir(parents=True)
            config = {
                "roots": {
                    "data": str(root / "data_link"),
                    "dna_raw": str(data),
                    "actorshq_raw": str(raw),
                    "actorshq_raw_alias": str(root / "raw_link"),
                    "actorshq_smplx": str(smplx),
                    "actorshq_smplx_legacy": str(root / "smplx_link"),
                },
                "datasets": [
                    {
                        "id": "actorshq",
                        "cases": [
                            {
                                "id": "actorshq_actor01_seq1",
                                "raw": "Actor01/Sequence1",
                            }
                        ],
                    }
                ],
            }
            first = prepare_resource_links(config)
            second = prepare_resource_links(config)
            self.assertTrue(first["all_ready"])
            self.assertTrue(second["all_ready"])
            self.assertEqual(
                second["links"]["d3dgs_data"]["action"], "unchanged"
            )

    def test_audit_rejects_unknown_case_filter(self):
        config = {
            "roots": {},
            "datasets": [
                {
                    "id": "actorshq",
                    "expected_frames": 1,
                    "expected_cameras": 1,
                    "cases": [{"id": "known"}],
                }
            ],
        }
        with self.assertRaisesRegex(ValueError, "unknown multiview cases"):
            audit_resources(config, selected_cases=["missing"])

    def test_derived_audit_uses_release_run_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case_id = "dna_sample"
            case_root = root / "dna_rendering" / case_id
            frames = 5
            training_frames = 4
            points = 3
            partmm_protocol = {
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
            metadata = {
                "training_frames": training_frames,
                "test_frame_ids": [4],
                "num_points": points,
                "method": "test",
                "algorithm": partmm_protocol["algorithm"],
                "partmm": {
                    key: value
                    for key, value in partmm_protocol.items()
                    if key != "algorithm"
                },
            }

            def prediction(directory_path, extras):
                directory_path.mkdir(parents=True)
                (directory_path / "metadata.json").write_text(
                    json.dumps(metadata), encoding="utf-8"
                )
                np.save(
                    directory_path / "warped_sequence.npy",
                    np.zeros((frames, points, 3), dtype=np.float32),
                )
                np.save(
                    directory_path / "test_warped_sequence.npy",
                    np.zeros((1, points, 3), dtype=np.float32),
                )
                np.save(
                    directory_path / "point_indices.npy",
                    np.arange(points),
                )
                for name in extras:
                    (directory_path / name).write_bytes(b"artifact")

            prediction(
                case_root / "smpl_lbs/lbs",
                ("frame0_surface.off", "robust_weights.npz", "diagnostic.png"),
            )
            np.savez(
                case_root / "smpl_lbs/lbs/robust_weights.npz",
                surface_weights=np.ones((2, 1), dtype=np.float32),
                gaussian_weights=np.ones(
                    (points, 1),
                    dtype=np.float32,
                ),
                projection_squared_distance=np.zeros(
                    points,
                    dtype=np.float32,
                ),
            )
            prediction(
                case_root / f"smpl_partmm/partmm/{case_id}",
                ("bone_transforms.npy", "diagnostic.png"),
            )
            np.save(
                case_root
                / f"smpl_partmm/partmm/{case_id}/bone_transforms.npy",
                np.broadcast_to(
                    np.eye(4, dtype=np.float32),
                    (1, 1, 4, 4),
                ),
            )
            ssdr = case_root / "smpl_partmm/ssdr"
            ssdr.mkdir(parents=True)
            np.savez(
                ssdr / "train_only.npz",
                training_frames=np.array(training_frames),
                rest_pose=np.zeros((points, 3), dtype=np.float32),
                weights=np.ones((points, 1), dtype=np.float32),
                world_transformations=np.tile(
                    np.eye(4, dtype=np.float32),
                    (training_frames, 1, 1, 1),
                ),
            )

            result = _derived_files(
                "dna_rendering",
                case_id,
                frames,
                {"runs": str(root)},
                full=True,
            )
            self.assertTrue(result["lbs"]["ready"])
            self.assertTrue(result["ssdr"]["ready"])
            self.assertTrue(result["partmm"]["ready"])
            self.assertFalse(result["bob"]["ready"])
            strict_result = _derived_files(
                "dna_rendering",
                case_id,
                frames,
                {"runs": str(root)},
                full=True,
                partmm_protocol=partmm_protocol,
            )
            self.assertTrue(strict_result["partmm"]["ready"])


if __name__ == "__main__":
    unittest.main()
