import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from experiments.mesh.database_scaling import (
    evaluate_database_scaling,
    minimum_pose_coverage,
)
from experiments.mesh.dt4d_suite import evaluate_dt4d_method
from experiments.mesh.patch_length import evaluate_patch_length
from experiments.mesh.runtime import benchmark_runtime
from experiments.mesh.vto_lbs import select_vto_paths
from experiments.mesh.vto_suite import evaluate_vto_method
from experiments.metrics import collect_json_metrics, evaluate_geometry, psnr
from experiments.paper import expand_jobs, load_paper_config, run_job
from experiments.protocols import (
    select_high_motion_frames,
    slice_point_sequence,
    split_frame_indices,
)
from skelebone.io import load_point_sequence, save_point_sequence


def _public_config_paths(root=Path("experiments")):
    excluded = {"runs", "reproductions"}
    return sorted(
        path for path in root.rglob("*.yaml")
        if excluded.isdisjoint(path.relative_to(root).parts[:-1])
    )


class PaperConfigDiscoveryTest(unittest.TestCase):
    def test_discovery_excludes_runs_and_reproductions_with_colliding_names(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "experiments"
            public = root / "mesh" / "runtime.yaml"
            ignored = [
                root / "runs" / "local" / "runtime.yaml",
                root / "reproductions" / "runtime.yaml",
            ]
            for path in [public, *ignored]:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("schema_version: 2\n", encoding="ascii")
            self.assertEqual(_public_config_paths(root), [public])


class PaperConfigTest(unittest.TestCase):
    def test_all_paper_configs_validate_and_render(self):
        configs = []
        for path in _public_config_paths():
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
            if raw.get("schema_version") == 2:
                configs.append(path)
        self.assertEqual(len(configs), 10)
        for path in configs:
            config = load_paper_config(path)
            jobs = expand_jobs(config)
            self.assertTrue(jobs, path)
            for job in jobs:
                rendered = run_job(job, dry_run=True)
                self.assertTrue(rendered["stages"])

    def test_paper_matrix_sizes(self):
        expected = {
            "novel_view.yaml": 77,
            "novel_pose.yaml": 90,
            "reconstruction.yaml": 18,
            "vto_animation.yaml": 10,
            "dt4d_animation.yaml": 5,
            "arap_ssdr_ablation.yaml": 3,
            "progressive_skeleton.yaml": 5,
            "runtime.yaml": 1,
            "patch_length.yaml": 5,
            "database_scaling.yaml": 3,
        }
        for filename, count in expected.items():
            path = next(path for path in _public_config_paths() if path.name == filename)
            self.assertEqual(len(expand_jobs(load_paper_config(path))), count, path)

    def test_camera_ready_non_novel_view_table_mapping(self):
        expected = {
            "novel_pose.yaml": "Table 3 / unseen-pose clothing rendering",
            "reconstruction.yaml": "Table 3 / shared D3DGS reconstruction",
            "vto_animation.yaml": "Table 4 / VTO garment mean vertex error",
            "dt4d_animation.yaml": "Table 4 / D4D animal mean vertex error",
            "patch_length.yaml": "Table 5 / PartMM patch length",
            "database_scaling.yaml": "Table R.1 / PartMM database coverage and cost",
        }
        for filename, paper_target in expected.items():
            path = next(path for path in _public_config_paths() if path.name == filename)
            config = load_paper_config(path)
            self.assertEqual(config["experiment"]["paper_target"], paper_target)

        novel_view = load_paper_config(
            Path("experiments/monocular_gaussian/novel_view.yaml")
        )
        self.assertEqual(
            novel_view["experiment"]["paper_target"],
            "Table 2 / unseen-view synthetic reconstruction",
        )

    def test_progressive_skeleton_matches_paper_columns(self):
        path = Path("experiments/monocular_gaussian/progressive_skeleton.yaml")
        config = load_paper_config(path)
        cases = config["datasets"][0]["cases"]
        self.assertEqual(
            [(case["id"], case["sequence"], case["percentage"], case["frame_count"])
             for case in cases],
            [
                ("p0", "trex", 0, 1),
                ("p30", "trex", 30, 60),
                ("p60", "trex", 60, 120),
                ("p80", "trex", 80, 160),
                ("p100", "trex", 100, 200),
            ],
        )
        export = config["methods"][0]["stages"][0]
        self.assertEqual(export["max_points"], 30000)

    def test_arap_ablation_is_the_three_paper_variants(self):
        path = Path("experiments/monocular_gaussian/arap_ssdr_ablation.yaml")
        config = load_paper_config(path)
        self.assertEqual(config["datasets"][0]["cases"], [{"id": "girlwalk"}])
        self.assertEqual(
            [method["id"] for method in config["methods"]],
            ["arap_ssdr", "no_arap_ssdr", "arap_no_ssdr"],
        )


class PaperMetricTest(unittest.TestCase):
    def test_json_selectors_support_indices_wildcards_and_reductions(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.json"
            output = Path(directory) / "metrics.json"
            source.write_text(
                json.dumps([{"score": 1.0}, {"score": 3.0}]),
                encoding="utf-8",
            )
            result = collect_json_metrics(
                source,
                output,
                {
                    "first": "0.score",
                    "mean": {"path": "*.score", "reduce": "mean"},
                    "count": {"path": "*.score", "reduce": "count"},
                },
            )
            self.assertEqual(result["metrics"], {"first": 1.0, "mean": 2.0, "count": 2})

    def test_geometry_metrics_and_psnr(self):
        with tempfile.TemporaryDirectory() as directory:
            reference = np.zeros((2, 3, 3), dtype=np.float32)
            prediction = np.ones_like(reference) * 0.01
            ref_path = Path(directory) / "reference.npy"
            pred_path = Path(directory) / "prediction.npy"
            np.save(ref_path, reference)
            np.save(pred_path, prediction)
            result = evaluate_geometry(
                pred_path,
                ref_path,
                Path(directory) / "geometry_metrics.json",
                unit_scale=1000,
            )
            self.assertAlmostEqual(result["metrics"]["rmse"], np.sqrt(300), places=5)
            self.assertTrue(np.isinf(psnr(reference, reference)))


class PaperProtocolTest(unittest.TestCase):
    def test_split_high_motion_and_slice_are_deterministic(self):
        train, test = split_frame_indices(10, 0.8)
        np.testing.assert_array_equal(train, np.arange(8))
        np.testing.assert_array_equal(test, np.arange(8, 10))
        points = np.zeros((10, 2, 3), dtype=np.float32)
        points[5:] = 10
        first = select_high_motion_frames(points, 4)
        second = select_high_motion_frames(points, 4)
        np.testing.assert_array_equal(first, second)
        with tempfile.TemporaryDirectory() as directory:
            source = save_point_sequence(Path(directory) / "full.npz", points)
            output = slice_point_sequence(
                source,
                Path(directory) / "half.npz",
                fraction=0.5,
            )
            self.assertEqual(load_point_sequence(output).num_frames, 5)

    def test_vto_lbs_uses_frozen_cases_in_declared_order(self):
        paths = [
            "/data/tshirt_shape00_91_62.pkl",
            "/data/tshirt_shape00_01_01.pkl",
            "/data/tshirt_shape00_91_33.pkl",
            "/data/tshirt_shape00_104_17.pkl",
        ]
        retained, test = select_vto_paths(
            paths,
            test_start=2,
            exclude="tshirt_shape00_104_17",
            expected_cases=[
                "tshirt_shape00_91_33",
                "tshirt_shape00_91_62",
            ],
        )
        self.assertEqual(Path(retained[0]).stem, "tshirt_shape00_01_01")
        self.assertEqual(
            [Path(path).stem for path in test],
            ["tshirt_shape00_91_33", "tshirt_shape00_91_62"],
        )
        with self.assertRaisesRegex(ValueError, "are missing"):
            select_vto_paths(
                paths,
                test_start=2,
                exclude="",
                expected_cases=["missing"],
            )


class HistoricalResultTest(unittest.TestCase):
    def test_patch_length_recomputes_frozen_case_mean(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "per_case.json"
            source.write_text(
                json.dumps(
                    [
                        {
                            "seq_name": "case_a",
                            "patch_size": 7,
                            "frames": 10,
                            "rmse_mm": 10.0,
                        },
                        {
                            "seq_name": "case_b",
                            "patch_size": 7,
                            "frames": 30,
                            "rmse_mm": 20.0,
                        },
                    ]
                ),
                encoding="utf-8",
            )
            result = evaluate_patch_length(
                7,
                source,
                root / "metrics.json",
                expected_cases=["case_a", "case_b"],
            )
        self.assertEqual(result["metrics"]["rmse_mm"], 15.0)
        self.assertEqual(result["metrics"]["frame_weighted_rmse_mm"], 17.5)
        self.assertEqual(len(result["source"]["sha256"]), 64)

    def test_database_scaling_requires_raw_measurements(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            accuracy = root / "accuracy.json"
            profile = root / "profile.json"
            accuracy.write_text(
                json.dumps(
                    {
                        "train_sizes": [5, 20, 50],
                        "mm_rmse": [17.36, 15.74, 15.27],
                    }
                ),
                encoding="utf-8",
            )
            profile.write_text(
                json.dumps(
                    {
                        "measurements": [
                            {
                                "sequences": 5,
                                "min_so3": 0.1,
                                "build_s": 10.0,
                                "predict_ms": 16.0,
                                "gpu_mb": 144,
                            },
                            {
                                "sequences": 20,
                                "min_so3": 0.09,
                                "build_s": 26.0,
                                "predict_ms": 22.0,
                                "gpu_mb": 356,
                            },
                            {
                                "sequences": 50,
                                "min_so3": 0.07,
                                "build_s": 66.0,
                                "predict_ms": 27.0,
                                "gpu_mb": 1001,
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            result = evaluate_database_scaling(
                20,
                accuracy,
                profile,
                root / "metrics.json",
                expected_sizes=[5, 20, 50],
            )
        self.assertEqual(
            result["metrics"],
            {
                "rmse_mm": 15.74,
                "min_so3": 0.09,
                "build_s": 26.0,
                "predict_ms": 22.0,
                "gpu_mb": 356.0,
            },
        )
        self.assertEqual(len(result["sources"]), 2)

    def test_database_coverage_matches_partmm_quaternion_geodesic(self):
        database = np.zeros((2, 72), dtype=np.float64)
        query = np.zeros((1, 72), dtype=np.float64)
        self.assertEqual(minimum_pose_coverage(database, query), 0.0)
        query[0, 3] = np.pi / 2
        self.assertAlmostEqual(
            minimum_pose_coverage(database, query),
            (np.pi / 4) / 23,
        )
        query[0, :3] = np.pi
        self.assertAlmostEqual(
            minimum_pose_coverage(database, query),
            (np.pi / 4) / 23,
        )

    def test_dt4d_table_result_matches_historical_benchmark(self):
        summary = Path(
            "/home/ubuntu/research/data/DeformingThings4D/"
            "skeleton2ssdr_experiment/benchmark_results/benchmark_results.json"
        )
        if not summary.is_file():
            self.skipTest("historical DT4D result is not installed")
        with tempfile.TemporaryDirectory() as directory:
            result = evaluate_dt4d_method(
                "fullmm",
                Path(directory) / "metrics.json",
                experiment_root=summary.parents[1],
                python="/home/ubuntu/anaconda3/envs/py38/bin/python",
                summary=summary,
                expected_instances=26,
                expected_summary_sha256=(
                    "2d6017baba956f3bcc191453409fa8baf"
                    "dfa238fea7529082fbc8da040daafa5"
                ),
            )
        self.assertEqual(result["metrics"]["evaluated_instances"], 26)
        self.assertAlmostEqual(result["metrics"]["rmse_mm"], 53.375852, places=5)

    def test_vto_partmm_result_matches_historical_benchmark(self):
        summary = Path("/home/ubuntu/research/GA/hood/output/precomputed_tshirt/summary.json")
        if not summary.is_file():
            self.skipTest("historical VTO result is not installed")
        with tempfile.TemporaryDirectory() as directory:
            result = evaluate_vto_method(
                "tshirt",
                "partmm",
                Path(directory) / "metrics.json",
                summary=summary,
                hood_root="/home/ubuntu/research/GA/hood",
                python="/home/ubuntu/anaconda3/envs/hood/bin/python",
                expected_cases=[
                    "tshirt_shape00_91_33",
                    "tshirt_shape00_91_36",
                    "tshirt_shape00_91_38",
                    "tshirt_shape00_91_61",
                    "tshirt_shape00_91_62",
                ],
            )
        self.assertEqual(len(result["per_sequence"]), 5)
        self.assertAlmostEqual(result["metrics"]["rmse_mm"], 17.3641979, places=5)

    def test_vto_frozen_protocol_rejects_a_different_case_set(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = root / "summary.json"
            summary.write_text(
                json.dumps(
                    [
                        {
                            "seq_name": "unexpected",
                            "frames": 1,
                            "PartMM_rmse": 1.0,
                        }
                    ]
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "frozen protocol"):
                evaluate_vto_method(
                    "tshirt",
                    "partmm",
                    root / "metrics.json",
                    summary=summary,
                    hood_root=root,
                    python="python",
                    expected_cases=["expected"],
                )

    def test_runtime_result_matches_executable_ablation_records(self):
        result_dir = Path(
            "/home/ubuntu/research/GA/hood/rebuttal_partmm_tshirt_shape00_91_33/"
            "results/ablation_tshirt_5cases_n5"
        )
        if not (result_dir / "mean_summary.json").is_file():
            self.skipTest("historical PartMM runtime result is not installed")
        with tempfile.TemporaryDirectory() as directory:
            result = benchmark_runtime(
                Path(directory) / "metrics.json",
                code_root=result_dir.parents[1] / "code",
                python="/home/ubuntu/anaconda3/envs/hood/bin/python",
                result_dir=result_dir,
                vto_root="/home/ubuntu/research/data/vto-dataset",
                smpl_model_path="unused",
                smpl_fk_model_dir="unused",
                ssdr_cache="unused",
            )
        self.assertEqual(result["config"]["query_frames"], 203)
        self.assertGreater(result["metrics"]["online_per_frame_ms"], 0)


if __name__ == "__main__":
    unittest.main()
