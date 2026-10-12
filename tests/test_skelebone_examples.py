from pathlib import Path
import importlib.util
import tempfile
import unittest

import numpy as np

from skelebone.examples.hood_tshirt import load_hood_tshirt
import skelebone.examples.ama as ama_example


class HoodTshirtExampleTest(unittest.TestCase):
    def test_ama_example_module_is_available(self):
        self.assertIsNotNone(importlib.util.find_spec("skelebone.examples.ama"))

    def test_ama_loader_reads_registered_obj_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            meshes = root / "T_samba" / "meshes"
            meshes.mkdir(parents=True)
            faces = "f 1 2 3\nf 1 3 4\n"
            (meshes / "mesh_0000.obj").write_text(
                "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n" + faces
            )
            (meshes / "mesh_0001.obj").write_text(
                "v 0 0 1\nv 1 0 1\nv 1 1 1\nv 0 1 1\n" + faces
            )

            sequence = ama_example.load_ama_sequence(root, case="T_samba", fps=25)

        self.assertEqual(sequence.points.shape, (2, 4, 3))
        np.testing.assert_array_equal(sequence.faces, [[0, 1, 2], [0, 2, 3]])
        self.assertEqual(sequence.fps, 25)

    def test_loader_reads_the_hood_export_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = root / "output" / "precomputed_tshirt"
            result.mkdir(parents=True)
            points = np.zeros((3, 8, 3), dtype=np.float32)
            faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
            np.savez(result / "tshirt_case.npz", gt_vertices=points, faces=faces)
            sequence = load_hood_tshirt(root, case="tshirt_case", fps=24)
        self.assertEqual(sequence.points.shape, (3, 8, 3))
        self.assertEqual(sequence.faces.shape, (2, 3))
        self.assertEqual(sequence.fps, 24)


if __name__ == "__main__":
    unittest.main()
