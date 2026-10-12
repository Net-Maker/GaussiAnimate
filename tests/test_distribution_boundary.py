"""Keep release wheels limited to the experiment configs the engine needs."""

import fnmatch
import importlib.util
import tempfile
import unittest
import zipfile
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib

REPO_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "wheel_check", REPO_ROOT / "tests/check_wheel.py"
)
_wheel_check = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_wheel_check)
REQUIRED_CONFIGS = _wheel_check.REQUIRED_CONFIGS
audit = _wheel_check.audit

EXPERIMENT_PACKAGE = "experiments"
# Only the catalog and the checked-in experiment/config trees the engine and the
# paper scripts resolve by path may ship; run outputs and diagnostics stay local.
ALLOWED_DATA_ROOTS = {
    "README.md",
    "catalog.yaml",
    "mesh",
    "monocular_gaussian",
    "multiview_gaussian",
    "reproductions",
}
# Canonical engine storage layouts under experiments/runs (see engine/storage.py)
# and the multiview diagnostics tree, as wheel member paths.
PRIVATE_MEMBERS = (
    "experiments/runs/audit_run/config.yaml",
    "experiments/runs/audit_run/resolved_config.yaml",
    "experiments/multiview_gaussian/diagnostics/dt4d_frames.json",
)
REQUIRED_PACKAGE_FILES = (
    "catalog.yaml",
    "reproductions/catalog.yaml",
)
REQUIRED_METHOD_DIR = "multiview_gaussian/methods"
DIST_INFO = "gaussianimate-0.1.0.dist-info"


def package_data_patterns():
    with open(REPO_ROOT / "pyproject.toml", "rb") as handle:
        project = tomllib.load(handle)
    return project["tool"]["setuptools"]["package-data"][EXPERIMENT_PACKAGE]


def collected_package_data():
    package_dir = REPO_ROOT / EXPERIMENT_PACKAGE
    matched = set()
    for pattern in package_data_patterns():
        matched.update(
            path.relative_to(package_dir).as_posix()
            for path in package_dir.glob(pattern)
            if path.is_file()
        )
    return matched


class PackageDataAllowlistTest(unittest.TestCase):
    def setUp(self):
        self.patterns = package_data_patterns()

    def test_patterns_stay_within_experiment_config_trees(self):
        for pattern in self.patterns:
            root = pattern.split("/", 1)[0]
            self.assertIn(
                root,
                ALLOWED_DATA_ROOTS,
                f"package-data pattern {pattern!r} leaves the public config trees",
            )

    def test_patterns_reject_run_outputs_and_diagnostics(self):
        for member in PRIVATE_MEMBERS:
            relative = member.split("/", 1)[1]
            for pattern in self.patterns:
                self.assertFalse(
                    fnmatch.fnmatch(relative, pattern),
                    f"pattern {pattern!r} matches private artifact {relative!r}",
                )

    def test_patterns_retain_engine_required_configs(self):
        for required in REQUIRED_PACKAGE_FILES + (REQUIRED_METHOD_DIR,):
            self.assertTrue(
                (REPO_ROOT / EXPERIMENT_PACKAGE / required).exists(),
                f"required engine data missing from the source tree: {required}",
            )
        method_configs = sorted(
            (REPO_ROOT / EXPERIMENT_PACKAGE / REQUIRED_METHOD_DIR).glob("*.yaml")
        )
        self.assertTrue(method_configs, "no multiview method configs in the source tree")
        retained = collected_package_data()
        expected = list(REQUIRED_PACKAGE_FILES) + [
            path.relative_to(REPO_ROOT / EXPERIMENT_PACKAGE).as_posix()
            for path in method_configs
        ]
        for member in expected:
            self.assertIn(member, retained, f"wheel packaging would drop {member}")


class WheelAuditTest(unittest.TestCase):
    def build_wheel(self, members):
        wheel = Path(tempfile.mkdtemp()) / "audit-0.1.0-py3-none-any.whl"
        with zipfile.ZipFile(wheel, "w") as archive:
            for name, payload in members.items():
                archive.writestr(name, payload)
        return wheel

    def minimal_members(self):
        members = {
            "experiments/__init__.py": b"",
            f"{DIST_INFO}/LICENSE": b"MIT",
            f"{DIST_INFO}/THIRD_PARTY_NOTICES.md": b"",
        }
        members.update({name: b"schema_version: 1\n" for name in sorted(REQUIRED_CONFIGS)})
        return members

    def test_audit_accepts_wheel_without_private_data(self):
        wheel = self.build_wheel(self.minimal_members())
        audit(wheel)

    def test_audit_rejects_run_outputs_despite_benign_content(self):
        members = self.minimal_members()
        members["experiments/runs/audit_run/config.yaml"] = b"schema_version: 1\n"
        with self.assertRaises(RuntimeError) as raised:
            audit(self.build_wheel(members))
        self.assertIn("experiments/runs", str(raised.exception))

    def test_audit_rejects_diagnostics(self):
        members = self.minimal_members()
        members["experiments/multiview_gaussian/diagnostics/dt4d_frames.json"] = b"[]"
        with self.assertRaises(RuntimeError) as raised:
            audit(self.build_wheel(members))
        self.assertIn("diagnostics", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
