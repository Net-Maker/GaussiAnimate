"""Config plumbing tests for the paper_partmm experiment-config CLI."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from experiments.engine import expand_jobs
from experiments.multiview_gaussian import paper_partmm
from partmm import PartMMConfig

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_experiment_config(
    root: Path,
    methods,
) -> Path:
    config = {
        "schema_version": 2,
        "experiment": {
            "id": "partmm_config_test",
            "target_id": "partmm_config_test",
            "title": "PartMM config test",
            "paper_target": "test",
            "seed": 42,
        },
        "output_dir": str(root / "runs"),
        "datasets": [{"id": "data", "cases": ["case"]}],
        "methods": methods,
    }
    path = root / "experiment.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def _paper_spec_method(protocol):
    return {
        "id": "smpl_partmm",
        "variant": "paper_spec",
        "protocol": protocol,
        "stages": [
            {"id": "animate", "type": "command", "command": ["true"]}
        ],
    }


def _run_main(argv):
    argv = ["paper_partmm"] + argv
    with mock.patch.object(sys, "argv", argv):
        paper_partmm.main()


class PaperPartmmConfigTest(unittest.TestCase):
    def test_protocol_settings_reach_run_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = _write_experiment_config(
                root,
                [
                    _paper_spec_method(
                        {
                            "id": "smpl_partmm_test_protocol",
                            "train_ratio": 0.75,
                            "patch_size": 9,
                            "num_neighbors": 3,
                            "alignment_mode": "edge_wise",
                        }
                    )
                ],
            )
            recorded = {}

            def fake_run(params, cache, motion, output_dir, **kwargs):
                config = kwargs["config"]
                recorded["train_ratio"] = kwargs["train_ratio"]
                recorded["config"] = config
                output_dir.mkdir(parents=True, exist_ok=True)
                return {
                    "partmm": {
                        "patch_size": config.patch_size,
                        "num_neighbors": config.num_neighbors,
                    }
                }

            output_dir = root / "out"
            with mock.patch.object(
                paper_partmm, "run_human_partmm", fake_run
            ):
                _run_main(
                    [
                        "--variant",
                        "paper_spec",
                        "--params",
                        str(root / "params.npz"),
                        "--ssdr-cache",
                        str(root / "cache.npz"),
                        "--motion",
                        str(root / "motion.npz"),
                        "--output-dir",
                        str(output_dir),
                        "--experiment-config",
                        str(config_path),
                        "--method-id",
                        "smpl_partmm",
                    ]
                )
            self.assertEqual(recorded["train_ratio"], 0.75)
            self.assertEqual(recorded["config"].patch_size, 9)
            self.assertEqual(recorded["config"].num_neighbors, 3)
            self.assertEqual(recorded["config"].alignment_mode, "edge_wise")
            metadata = json.loads(
                (output_dir / "metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["algorithm"], "paper_spec_partmm")
            self.assertEqual(metadata["partmm"]["patch_size"], 9)
            self.assertEqual(metadata["partmm"]["num_neighbors"], 3)

    def test_protocol_overrides_keep_paper_defaults_for_omitted_fields(self):
        config = paper_partmm._paper_spec_config_from_protocol(
            {"patch_size": 9, "num_neighbors": 3}
        )
        self.assertEqual(config.patch_size, 9)
        self.assertEqual(config.num_neighbors, 3)
        self.assertEqual(config.num_parts, 5)
        self.assertEqual(config.pyramid_levels, 5)
        self.assertEqual(config.pyramid_blend, 0.7)
        self.assertEqual(config.overlap_rings, 1)
        self.assertEqual(config.alignment_mode, "edge_wise")
        self.assertEqual(config.partition_mode, "humanoid_five")
        self.assertEqual(config.bone_overlap_ratio, 1.5)

    def test_protocol_invalid_setting_rejected_by_partmm_config(self):
        with self.assertRaises(ValueError):
            paper_partmm._paper_spec_config_from_protocol(
                {"alignment_mode": "bogus"}
            )

    def test_select_method_rejects_unknown(self):
        with self.assertRaisesRegex(ValueError, "no method with id"):
            paper_partmm._select_experiment_method(
                {"methods": []},
                "smpl_partmm",
                "paper_spec",
            )

    def test_select_method_rejects_nonunique(self):
        with self.assertRaisesRegex(ValueError, "2 methods"):
            paper_partmm._select_experiment_method(
                {"methods": [{"id": "m"}, {"id": "m"}]},
                "m",
                "paper_spec",
            )

    def test_select_method_rejects_variant_mismatch(self):
        with self.assertRaisesRegex(ValueError, "variant"):
            paper_partmm._select_experiment_method(
                {"methods": [{"id": "m", "variant": "historical"}]},
                "m",
                "paper_spec",
            )

    def test_main_rejects_unknown_method_id(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = _write_experiment_config(
                root, [_paper_spec_method({"train_ratio": 0.8})]
            )
            argv = [
                "--variant",
                "paper_spec",
                "--params",
                "p.npz",
                "--ssdr-cache",
                "c.npz",
                "--motion",
                "m.npz",
                "--output-dir",
                "out",
                "--experiment-config",
                str(config_path),
                "--method-id",
                "missing_method",
            ]
            with self.assertRaises(SystemExit) as raised:
                _run_main(argv)
            self.assertEqual(raised.exception.code, 2)

    def test_main_rejects_variant_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = _write_experiment_config(
                root, [_paper_spec_method({"train_ratio": 0.8})]
            )
            argv = [
                "--variant",
                "historical",
                "--params",
                "p.npz",
                "--ssdr-cache",
                "c.npz",
                "--motion",
                "m.npz",
                "--output-dir",
                "out",
                "--implementation-root",
                "impl",
                "--motion2motion-root",
                "m2m",
                "--smpl-model-dir",
                "smpl",
                "--experiment-config",
                str(config_path),
                "--method-id",
                "smpl_partmm",
            ]
            with self.assertRaises(SystemExit) as raised:
                _run_main(argv)
            self.assertEqual(raised.exception.code, 2)

    def test_historical_with_config_keeps_cli_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = _write_experiment_config(
                root,
                [
                    {
                        "id": "smpl_partmm_historical",
                        "variant": "historical",
                        "protocol": {
                            "id": "historical_protocol",
                            "train_ratio": 0.8,
                            "patch_size": 11,
                        },
                        "stages": [
                            {
                                "id": "animate",
                                "type": "command",
                                "command": ["true"],
                            }
                        ],
                    }
                ],
            )
            recorded = {}

            def fake_historical(params, cache, motion, output_dir, **kwargs):
                recorded.update(kwargs)
                return {}

            with mock.patch.object(
                paper_partmm, "run_paper_partmm", fake_historical
            ):
                _run_main(
                    [
                        "--variant",
                        "historical",
                        "--params",
                        "p.npz",
                        "--ssdr-cache",
                        "c.npz",
                        "--motion",
                        "m.npz",
                        "--output-dir",
                        "out",
                        "--implementation-root",
                        str(root / "impl"),
                        "--motion2motion-root",
                        str(root / "m2m"),
                        "--smpl-model-dir",
                        str(root / "smpl"),
                        "--train-ratio",
                        "0.9",
                        "--device",
                        "cpu",
                        "--experiment-config",
                        str(config_path),
                        "--method-id",
                        "smpl_partmm_historical",
                    ]
                )
            self.assertEqual(recorded["train_ratio"], 0.9)
            self.assertEqual(
                recorded["implementation_root"], root / "impl"
            )

    def test_legacy_cli_without_config_keeps_paper_defaults(self):
        recorded = {}

        def fake_run(params, cache, motion, output_dir, **kwargs):
            recorded.update(kwargs)
            output_dir.mkdir(parents=True, exist_ok=True)
            return {}

        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            paper_partmm, "run_human_partmm", fake_run
        ):
            _run_main(
                [
                    "--variant",
                    "paper_spec",
                    "--params",
                    "p.npz",
                    "--ssdr-cache",
                    "c.npz",
                    "--motion",
                    "m.npz",
                    "--output-dir",
                    str(Path(directory) / "out"),
                ]
            )
        self.assertEqual(recorded["train_ratio"], 0.8)
        self.assertEqual(
            recorded["config"],
            PartMMConfig(
                num_parts=5,
                patch_size=7,
                num_neighbors=7,
                pyramid_levels=5,
                pyramid_blend=0.7,
                overlap_rings=1,
                align_parts=True,
                alignment_mode="edge_wise",
                partition_mode="humanoid_five",
                bone_overlap_ratio=1.5,
            ),
        )

    def test_historical_requires_implementation_options(self):
        argv = [
            "--variant",
            "historical",
            "--params",
            "p.npz",
            "--ssdr-cache",
            "c.npz",
            "--motion",
            "m.npz",
            "--output-dir",
            "out",
        ]
        with self.assertRaises(SystemExit) as raised:
            _run_main(argv)
        self.assertEqual(raised.exception.code, 2)

    def test_job_context_exposes_config_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.yaml"
            config = {
                "schema_version": 2,
                "experiment": {
                    "id": "engine_test",
                    "target_id": "engine_test",
                    "title": "Engine test",
                    "paper_target": "Test",
                    "seed": 42,
                },
                "output_dir": str(root / "runs"),
                "datasets": [{"id": "data", "cases": ["case"]}],
                "methods": [
                    {
                        "id": "method",
                        "mode": "native",
                        "stages": [
                            {
                                "id": "first",
                                "type": "command",
                                "command": ["true"],
                            }
                        ],
                    }
                ],
                "_config_path": str(config_path),
            }
            job = expand_jobs(config)[0]
            context = job.context()
            self.assertEqual(context["config_path"], str(config_path))
            self.assertEqual(context["config_dir"], str(root))

    def test_yaml_animate_stage_passes_experiment_config(self):
        component = yaml.safe_load(
            (
                REPO_ROOT
                / "experiments/multiview_gaussian/methods/smpl_partmm.yaml"
            ).read_text(encoding="utf-8")
        )
        stage = next(
            stage
            for stage in component["methods"][0]["stages"]
            if stage["id"] == "animate"
        )
        command = stage["command"]
        self.assertIn("--experiment-config", command)
        self.assertIn("{config_path}", command)
        self.assertIn("--method-id", command)
        self.assertIn("{method.id}", command)
        self.assertIn("{config_path}", stage["inputs"])
        self.assertIn(
            "{config_dir}/methods/smpl_partmm.yaml",
            stage["inputs"],
        )


if __name__ == "__main__":
    unittest.main()
