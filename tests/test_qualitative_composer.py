import json
import tempfile
import unittest
from pathlib import Path

import yaml
from PIL import Image

from experiments.qualitative import (
    FigureBlockedError,
    audit_figure_manifest,
    compose_figure,
    load_figure_manifest,
    validate_figure_artifact,
    validate_figure_manifest,
)


def _write_image(path: Path, color, alpha=False) -> None:
    mode = "RGBA" if alpha else "RGB"
    fill = color + ((180,) if alpha else ())
    Image.new(mode, (40, 30), fill).save(path)


def _manifest(root: Path):
    return {
        "schema_version": 1,
        "kind": "qualitative_figure",
        "figure": {
            "id": "fixture",
            "paper_target": "test figure",
            "source_policy": "explicit_external",
        },
        "methods": [
            {"id": "input", "label": "Input"},
            {"id": "prediction", "label": "Prediction"},
        ],
        "rows": [
            {"id": "appearance", "label": "Appearance"},
            {"id": "error", "label": "Error"},
        ],
        "layout": {
            "method_order": ["input", "prediction"],
            "row_order": ["appearance", "error"],
            "background": "white",
            "cell_size": [80, 60],
            "padding": 10,
            "gap": 4,
            "header_height": 20,
            "row_label_width": 40,
            "row_label_orientation": "horizontal",
            "header_font_size": 10,
            "row_font_size": 9,
        },
        "panels": {
            "appearance": {
                "input": str(root / "input.png"),
                "prediction": str(root / "prediction.png"),
            },
            "error": {
                "input": None,
                "prediction": str(root / "error.png"),
            },
        },
        "output": {
            "image": str(root / "figure.png"),
            "provenance": str(root / "provenance.json"),
        },
    }


class QualitativeComposerTest(unittest.TestCase):
    def test_compose_writes_white_figure_and_source_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_image(root / "input.png", (30, 40, 50))
            _write_image(
                root / "prediction.png",
                (20, 140, 150),
                alpha=True,
            )
            _write_image(root / "error.png", (180, 70, 45))
            manifest = _manifest(root)
            record = compose_figure(manifest)
            output = root / "figure.png"
            provenance = root / "provenance.json"
            self.assertTrue(output.is_file())
            self.assertEqual(
                record["method_order"],
                ["input", "prediction"],
            )
            self.assertEqual(len(record["sources"]), 3)
            self.assertTrue(
                all(source["sha256"] for source in record["sources"])
            )
            with Image.open(output) as image:
                self.assertEqual(image.getpixel((0, 0)), (255, 255, 255))
            self.assertTrue(
                validate_figure_artifact(output, provenance)["ready"]
            )
            saved = json.loads(provenance.read_text(encoding="utf-8"))
            self.assertEqual(saved["canvas"]["background"], "white")
            self.assertEqual(
                saved["sources"][0]["background"],
                "alpha_to_white",
            )
            (root / "input.png").write_bytes(b"changed")
            stale = validate_figure_artifact(output, provenance)
            self.assertFalse(stale["ready"])
            self.assertFalse(stale["sources_match"])

    def test_missing_input_is_explicitly_blocked_without_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _manifest(root)
            report = audit_figure_manifest(manifest)
            self.assertEqual(report["status"], "blocked")
            self.assertEqual(len(report["blockers"]), 3)
            with self.assertRaises(FigureBlockedError):
                compose_figure(manifest)
            self.assertFalse((root / "figure.png").exists())

    def test_manifest_requires_fixed_complete_method_order(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = _manifest(Path(directory))
            manifest["layout"]["method_order"].reverse()
            with self.assertRaisesRegex(
                ValueError,
                "method_order must match",
            ):
                validate_figure_manifest(manifest)
            manifest = _manifest(Path(directory))
            del manifest["panels"]["error"]["prediction"]
            with self.assertRaisesRegex(
                ValueError,
                "define every method exactly",
            ):
                validate_figure_manifest(manifest)

    def test_unresolved_external_root_is_reported_as_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _manifest(root)
            manifest["panels"]["appearance"]["input"] = (
                "${EXTERNAL_PANEL_ROOT}/input.png"
            )
            report = audit_figure_manifest(manifest, check_inputs=False)
            self.assertFalse(report["ready"])
            self.assertIn(
                "unresolved input",
                report["blockers"][0],
            )

    def test_repository_figure_manifests_are_structurally_valid(self):
        repository = Path(__file__).parents[1]
        paths = (
            repository / "experiments/mesh/vto_main_figure.yaml",
            repository / "experiments/mesh/dt4d_main_figure.yaml",
            repository
            / "experiments/multiview_gaussian/actorshq_appendix_figure.yaml",
        )
        manifests = [
            load_figure_manifest(path, require_environment=False)
            for path in paths
        ]
        self.assertEqual(
            manifests[0]["layout"]["method_order"],
            ["lbs", "gru", "mlp", "fullmm", "partmm", "gt"],
        )
        self.assertEqual(
            manifests[1]["layout"]["row_order"],
            ["mesh", "overlap", "error"],
        )
        self.assertEqual(
            manifests[2]["layout"]["method_order"],
            ["gt", "d3dgs", "bob", "partmm", "lbs"],
        )
        for manifest in manifests:
            validate_figure_manifest(manifest)

    def test_yaml_round_trip_preserves_explicit_null_panels(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = _manifest(Path(directory))
            serialized = yaml.safe_dump(manifest, sort_keys=False)
            restored = yaml.safe_load(serialized)
            self.assertIsNone(restored["panels"]["error"]["input"])


if __name__ == "__main__":
    unittest.main()
