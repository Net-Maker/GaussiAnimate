import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.mesh.vto_lbs import _deform_cloth_with_lbs, _load_vto_sequence


class PublicVtoLbsTest(unittest.TestCase):
    def test_loader_canonicalizes_root_and_pose(self):
        data = {
            "pose": np.array([[0.0, 0.0, np.pi / 2, *([0.0] * 69)]]),
            "translation": np.array([[1.0, 2.0, 3.0]]),
            "shape": np.zeros((1, 10)),
            "vertices": np.array([[[1.0, 3.0, 3.0]]]),
            "faces": np.empty((0, 3), dtype=np.int32),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "case.pkl"
            with path.open("wb") as handle:
                pickle.dump(data, handle)
            result = _load_vto_sequence(path, np.zeros(3))
        np.testing.assert_allclose(result["poses"][0, :3], 0.0)
        np.testing.assert_allclose(
            result["vertices"][0, 0], [1.0, 0.0, 0.0], atol=1e-6
        )

    def test_lbs_identity_preserves_vertices(self):
        vertices = np.array([[1.0, 2.0, 3.0], [-1.0, 0.0, 2.0]], dtype=np.float32)
        weights = np.zeros((2, 24), dtype=np.float32)
        weights[:, 0] = 1.0
        transforms = np.broadcast_to(np.eye(4), (1, 24, 4, 4)).copy()
        result = _deform_cloth_with_lbs(vertices, weights, transforms)
        np.testing.assert_allclose(result[0], vertices)


if __name__ == "__main__":
    unittest.main()
