import tempfile
import unittest
from pathlib import Path

from PIL import Image

from experiments.monocular_gaussian.resources import (
    audit_resources,
    package_evaluation_results,
    prepare_release_links,
)


class MonocularResourceTest(unittest.TestCase):
    def _fixture(self, root: Path):
        model = root / "history/stage/case_512_node"
        render = model / "test/ours_12"
        (render / "renders").mkdir(parents=True)
        (render / "gt").mkdir()
        for index in range(2):
            name = f"{index:05d}.png"
            image = Image.new("RGB", (2, 2), color=(index, 0, 0))
            image.save(render / "renders" / name)
            image.save(render / "gt" / name)
            image.save(render / "renders" / f"{index:05d}_rig.png")
        (model / "cfg_args").write_text(
            "Namespace(use_skinning_weight_mlp=False, use_template_offsets=False)",
            encoding="utf-8",
        )
        return {
            "roots": {
                "historical_riggs": str(root / "history"),
                "release": str(root / "release"),
            },
            "datasets": [
                {"id": "dnerf", "expected_test_frames": 2, "cases": ["case"]}
            ],
            "methods": [
                {
                    "id": "ours",
                    "classification": "reconstruction",
                    "source": "stage",
                    "case_patterns": ["{case}_512_node/test/ours_*"],
                    "cfg_patterns": ["{case}_512_node/cfg_args"],
                    "required_cfg": {
                        "use_skinning_weight_mlp": False,
                        "use_template_offsets": False,
                    },
                }
            ],
        }

    def test_audit_pairs_only_exact_ground_truth_stems(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self._fixture(Path(directory))
            report = audit_resources(config)
            record = report["records"][0]
            self.assertTrue(record["ready"])
            self.assertEqual(record["paired_frames"], 2)
            self.assertEqual(record["prediction_only_count"], 2)
            self.assertEqual(
                record["prediction_only_sample"], ["00000_rig", "00001_rig"]
            )

    def test_release_links_filter_auxiliary_rig_renders(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._fixture(root)
            report = audit_resources(config)
            result = prepare_release_links(report, root / "release")
            render_dir = root / "release/ours/dnerf/case/renders"
            self.assertEqual(result["created"], 4)
            self.assertEqual(
                sorted(path.name for path in render_dir.iterdir()),
                ["00000.png", "00001.png"],
            )
            self.assertTrue((root / "release/ours/dnerf/case/provenance.json").is_file())

    def test_audit_rejects_unknown_filter(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self._fixture(Path(directory))
            with self.assertRaisesRegex(ValueError, "unknown monocular cases"):
                audit_resources(config, selected_cases=["missing"])

    def test_evaluation_package_records_summary_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            (source / "summary.json").write_text("{}\n", encoding="utf-8")
            result = package_evaluation_results(source, root / "release")
            self.assertIn("summary.json", result["summaries"])
            self.assertTrue(Path(result["provenance"]).is_file())


if __name__ == "__main__":
    unittest.main()
