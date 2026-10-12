import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
NATIVE_ROOT = ROOT / "native" / "curve_skeleton"


class NativeCurveSkeletonTest(unittest.TestCase):
    def test_native_source_and_installer_are_part_of_the_repository(self):
        self.assertTrue((NATIVE_ROOT / "CMakeLists.txt").is_file())
        self.assertTrue((NATIVE_ROOT / "main.cpp").is_file())
        self.assertTrue((NATIVE_ROOT / "LICENSE").is_file())
        installer = ROOT / "scripts" / "install_geometry.sh"
        self.assertTrue(installer.is_file())
        completed = subprocess.run(
            ["bash", str(installer), "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--curve-only", completed.stdout)
        self.assertIn("--pyssdr-only", completed.stdout)

    @unittest.skipUnless(
        os.environ.get("GAUSSIANIMATE_TEST_NATIVE") == "1",
        "set GAUSSIANIMATE_TEST_NATIVE=1 to compile the CGAL integration test",
    )
    def test_compiled_extractor_writes_the_public_protocol(self):
        self.assertIsNotNone(shutil.which("cmake"))
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            build = directory / "build"
            subprocess.run(
                ["cmake", "-S", str(NATIVE_ROOT), "-B", str(build)],
                check=True,
            )
            subprocess.run(
                ["cmake", "--build", str(build), "--parallel", "2"],
                check=True,
            )
            mesh = directory / "tetrahedron.off"
            mesh.write_text(
                "OFF\n"
                "4 4 0\n"
                "0 0 0\n1 0 0\n0 1 0\n0 0 1\n"
                "3 0 2 1\n3 0 1 3\n3 1 2 3\n3 2 0 3\n",
                encoding="ascii",
            )
            prefix = directory / "result" / "curve_skeleton"
            completed = subprocess.run(
                [
                    str(build / "curve_skeleton"),
                    str(mesh),
                    str(prefix),
                    "--quality",
                    "0.1",
                    "--medial",
                    "0.25",
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=120,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue(Path(str(prefix) + "_points.txt").is_file())
            self.assertTrue(Path(str(prefix) + "_edges.txt").is_file())
            self.assertTrue(Path(str(prefix) + ".obj").is_file())


if __name__ == "__main__":
    unittest.main()
