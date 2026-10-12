import unittest
import subprocess
import sys
from pathlib import Path

from experiments.config import load_config, resolve_skelebone_config


class ExperimentConfigTest(unittest.TestCase):
    def test_cli_validates_presets_without_optional_visualization_dependencies(self):
        for path in sorted(Path("experiments").glob("*/config.yaml")):
            result = subprocess.run(
                [sys.executable, "-m", "experiments.run", str(path), "--validate-only"],
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_all_presets_validate_without_environment(self):
        paths = sorted(Path("experiments").glob("*/config.yaml"))
        self.assertEqual(len(paths), 3)
        for path in paths:
            config = load_config(path, require_environment=False)
            self.assertEqual(config["schema_version"], 1)

    def test_vto_variant_applies_twenty_bone_override(self):
        config = load_config(
            "experiments/mesh/config.yaml", require_environment=False
        )
        config["dataset"]["variant"] = "vto"
        self.assertEqual(resolve_skelebone_config(config)["num_bones"], 20)


if __name__ == "__main__":
    unittest.main()
