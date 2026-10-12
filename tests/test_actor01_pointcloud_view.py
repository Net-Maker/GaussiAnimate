import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np


class Actor01PointCloudTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[1] / 'scripts/visualize_actor01_gaussians.py'
        spec = importlib.util.spec_from_file_location('actor01_view', path)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)

    def test_camera_basis_matches_capture_orientation(self):
        basis = self.module.view_basis(np.eye(4), 0)
        np.testing.assert_allclose(basis, np.diag([1., -1., -1.]))
        rotated = self.module.view_basis(np.eye(4), 80)
        np.testing.assert_allclose(rotated.T @ rotated, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(rotated[:, 1], basis[:, 1])

    def test_loading_keeps_zero_opacity_and_outliers(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'points.npz'
            xyz = np.array([[0., 0., 0.], [100., 0., 0.]])
            rgb = np.array([[1., 0., 0.], [0., 1., 0.]])
            np.savez(path, xyz=xyz, rgb=rgb, opacity=np.array([1., 0.]))
            points, colors = self.module.load_cloud(path)
            np.testing.assert_array_equal(points, xyz)
            np.testing.assert_array_equal(colors, rgb)


if __name__ == '__main__':
    unittest.main()
