import importlib.util
from pathlib import Path
import unittest

import numpy as np
import open3d as o3d


class TemplateVolumeTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[1] / 'experiments/multiview_gaussian/template_volume.py'
        spec = importlib.util.spec_from_file_location('template_volume', path)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)

    def test_scale_is_exactly_about_bbox_center(self):
        vertices = np.array([[2., 3., 4.], [4., 5., 6.]])
        expanded, center = self.module.inflate(vertices, 1.1)
        np.testing.assert_allclose(center, [3, 4, 5])
        np.testing.assert_allclose(expanded, [[1.9, 2.9, 3.9], [4.1, 5.1, 6.1]])

    def test_volume_keeps_interior_and_removes_exterior(self):
        mesh = o3d.geometry.TriangleMesh.create_box(2, 2, 2).translate([-1, -1, -1])
        vertices, _ = self.module.inflate(np.asarray(mesh.vertices), 1.1)
        queries = np.array([[0., 0., 0.], [1.05, 0., 0.], [1.2, 0., 0.], [20., 20., 20.]])
        keep, _ = self.module.contains(vertices, np.asarray(mesh.triangles), queries, .5)
        np.testing.assert_array_equal(keep, [True, True, False, False])

    def test_surface_dilation_preserves_interior_and_adds_distance_shell(self):
        mesh = o3d.geometry.TriangleMesh.create_box(2, 2, 2).translate([-1, -1, -1])
        vertices = np.asarray(mesh.vertices)
        queries = np.array([[0., 0., 0.], [1.02, 0., 0.], [1.04, 0., 0.], [1.02, 1.02, 0.]])
        keep, distance = self.module.dilated_contains(vertices, np.asarray(mesh.triangles), queries, .03)
        np.testing.assert_array_equal(keep, [True, True, False, True])
        np.testing.assert_allclose(distance, [-1., .02, .04, np.sqrt(.0008)], atol=1e-6)
        shifted, _ = self.module.dilated_contains(vertices + 10, np.asarray(mesh.triangles), queries + 10, .03)
        np.testing.assert_array_equal(shifted, keep)
        larger, _ = self.module.dilated_contains(vertices, np.asarray(mesh.triangles), queries, .05)
        self.assertTrue(larger.all())

    def test_surface_dilation_does_not_move_separated_limbs(self):
        mesh = o3d.geometry.TriangleMesh.create_box().translate([4, 0, 0])
        mesh += o3d.geometry.TriangleMesh.create_box().translate([-5, 0, 0])
        vertices = np.asarray(mesh.vertices).copy()
        queries = np.vstack([vertices, [[4.01, .5, .5], [-4.01, .5, .5]]])
        keep, _ = self.module.dilated_contains(vertices, np.asarray(mesh.triangles), queries, .03)
        self.assertTrue(keep.all())
        np.testing.assert_array_equal(vertices, np.asarray(mesh.vertices))

    def test_fixed_gaussian_indices_are_identical_in_every_frame(self):
        params = {}
        for key, channels in [('means3D', 3), ('rgb_colors', 3), ('unnorm_rotations', 4)]:
            params[key] = np.arange(3 * 5 * channels).reshape(3, 5, channels)
        for key in ['seg_colors', 'logit_opacities', 'log_scales']:
            params[key] = np.arange(5)[:, None]
        params['cam_m'] = np.arange(12).reshape(4, 3)
        params['cam_c'] = params['cam_m'] + 1
        keep = np.array([True, False, True, False, True])
        filtered = self.module.subset_gaussians(params, keep)
        for key in ['means3D', 'rgb_colors', 'unnorm_rotations']:
            np.testing.assert_array_equal(filtered[key], params[key][:, [0, 2, 4]])
        for key in ['seg_colors', 'logit_opacities', 'log_scales']:
            np.testing.assert_array_equal(filtered[key], params[key][[0, 2, 4]])
        for key in ['cam_m', 'cam_c']:
            np.testing.assert_array_equal(filtered[key], params[key])
        self.assertEqual(params['means3D'].shape[1], 5)


if __name__ == '__main__':
    unittest.main()
