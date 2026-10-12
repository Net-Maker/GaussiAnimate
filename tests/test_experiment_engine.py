import json
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import yaml

from experiments.engine import stages as stage_handlers
from experiments.engine import (
    adopt_legacy_stages,
    expand_jobs,
    load_paper_config,
    run_job,
    stage_fingerprint,
)
from experiments.report import aggregate_run


def _write_command():
    return [
        sys.executable,
        "-c",
        (
            "from pathlib import Path; import sys; "
            "p=Path(sys.argv[1]); p.parent.mkdir(parents=True, exist_ok=True); "
            "p.write_text(sys.argv[2], encoding='utf-8')"
        ),
    ]


def _config(root: Path):
    return {
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
                        "command": _write_command()
                        + ["{run_dir}/first.txt", "first"],
                        "outputs": ["{run_dir}/first.txt"],
                    },
                    {
                        "id": "second",
                        "type": "command",
                        "command": _write_command()
                        + ["{run_dir}/second.txt", "second"],
                        "outputs": ["{run_dir}/second.txt"],
                    },
                ],
            }
        ],
        "_config_path": str(root / "config.yaml"),
    }


class ExperimentEngineTest(unittest.TestCase):
    def test_method_protocol_invalidates_stage_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            job = expand_jobs(_config(Path(directory)))[0]
            job.method['protocol'] = {'patch_size': 7}
            stage = job.method['stages'][0]
            before = stage_fingerprint(job, stage)
            job.method['protocol']['patch_size'] = 9
            self.assertNotEqual(before, stage_fingerprint(job, stage))

    def test_imported_partmm_implementation_invalidates_command(self):
        with tempfile.TemporaryDirectory() as directory:
            job = expand_jobs(_config(Path(directory)))[0]
            stage = dict(job.method['stages'][0], command=[
                sys.executable, '-m', 'experiments.multiview_gaussian.human_partmm'
            ])
            before = stage_fingerprint(job, stage)
            source = Path('partmm/matching.py').resolve()
            read_bytes = Path.read_bytes

            def changed(path):
                content = read_bytes(path)
                return content + b'\n# implementation changed\n' if path.resolve() == source else content

            with mock.patch.object(Path, 'read_bytes', changed):
                self.assertNotEqual(before, stage_fingerprint(job, stage))

    def test_imported_metric_implementation_invalidates_handler(self):
        with tempfile.TemporaryDirectory() as directory:
            job = expand_jobs(_config(Path(directory)))[0]
            stage = {'id': 'metrics', 'type': 'geometry_metrics', 'outputs': []}
            before = stage_fingerprint(job, stage)
            source = Path('experiments/metrics.py').resolve()
            read_bytes = Path.read_bytes

            def changed(path):
                content = read_bytes(path)
                return content + b'\n# metric changed\n' if path.resolve() == source else content

            with mock.patch.object(Path, 'read_bytes', changed):
                self.assertNotEqual(before, stage_fingerprint(job, stage))

    def test_command_log_format_is_python37_compatible(self):
        self.assertEqual(
            stage_handlers._shell_join(["python", "two words", "it's"]),
            "python 'two words' 'it'\"'\"'s'",
        )

    def test_selected_stages_are_partial_and_manifest_is_merged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = expand_jobs(_config(root))[0]

            first = run_job(job, selected_stages=["first"])
            self.assertEqual(first["status"], "partial")
            self.assertEqual(set(first["stages"]), {"first"})

            complete = run_job(job, selected_stages=["second"])
            self.assertEqual(complete["status"], "complete")
            self.assertEqual(
                set(complete["stages"]),
                {"first", "second"},
            )
            stored = json.loads(
                (job.run_dir / "job.json").read_text(encoding="utf-8")
            )
            self.assertEqual(stored["status"], "complete")

    def test_input_signature_invalidates_stage_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.txt"
            source.write_text("first", encoding="utf-8")
            config = _config(root)
            config["methods"][0]["stages"][0]["inputs"] = [str(source)]
            job = expand_jobs(config)[0]
            stage = job.method["stages"][0]
            before = stage_fingerprint(job, stage)
            source.write_text("second and longer", encoding="utf-8")
            after = stage_fingerprint(job, stage)
            self.assertNotEqual(before, after)

    def test_method_components_are_composed_before_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            component = root / "method.yaml"
            component.write_text(
                yaml.safe_dump(
                    {
                        "schema_version": 1,
                        "kind": "experiment_methods",
                        "methods": _config(root)["methods"],
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            config = _config(root)
            config.pop("methods")
            config["method_files"] = ["method.yaml"]
            config.pop("_config_path")
            config_path = root / "config.yaml"
            config_path.write_text(
                yaml.safe_dump(config, sort_keys=False),
                encoding="utf-8",
            )

            loaded = load_paper_config(config_path)
            self.assertEqual(
                [method["id"] for method in loaded["methods"]],
                ["method"],
            )
            self.assertEqual(len(expand_jobs(loaded)), 1)

    def test_legacy_stage_adoption_requires_current_downstream_consumer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = _config(root)
            first_output = "{run_dir}/first.txt"
            config["methods"][0]["stages"][1]["inputs"] = [first_output]
            job = expand_jobs(config)[0]
            complete = run_job(job)
            self.assertEqual(complete["status"], "complete")

            record_path = job.run_dir / ".stages" / "first.json"
            record = json.loads(record_path.read_text(encoding="utf-8"))
            record.pop("fingerprint")
            record_path.write_text(
                json.dumps(record),
                encoding="utf-8",
            )

            adopted = adopt_legacy_stages(job, ["first"])
            self.assertEqual(adopted["status"], "complete")
            record = json.loads(record_path.read_text(encoding="utf-8"))
            self.assertTrue(record["fingerprint"])
            self.assertEqual(
                record["adoption"]["validated_by"],
                ["second"],
            )

    def test_legacy_stage_adoption_rejects_unconsumed_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = expand_jobs(_config(root))[0]
            run_job(job, selected_stages=["first"])
            record_path = job.run_dir / ".stages" / "first.json"
            record = json.loads(record_path.read_text(encoding="utf-8"))
            record.pop("fingerprint")
            record_path.write_text(
                json.dumps(record),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError,
                "no current completed downstream stage",
            ):
                adopt_legacy_stages(job, ["first"])

    def test_report_does_not_average_incomplete_case_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = _config(root)
            config["datasets"][0]["cases"] = ["first", "missing"]
            config["methods"][0]["stages"] = [
                config["methods"][0]["stages"][0]
            ]
            jobs = expand_jobs(config)
            run_job(jobs[0])
            (jobs[0].run_dir / "image_metrics.json").write_text(
                json.dumps({"metrics": {"psnr": 30.0}}),
                encoding="utf-8",
            )

            report = aggregate_run(
                jobs[0].output_root,
                expected_jobs=jobs,
            )
            aggregate = report["averages"][0]
            self.assertEqual(aggregate["cases"], 2)
            self.assertEqual(aggregate["complete"], 1)
            self.assertEqual(aggregate["status"], "partial")
            self.assertNotIn("psnr", aggregate)


if __name__ == "__main__":
    unittest.main()
