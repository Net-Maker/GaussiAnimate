import unittest
import inspect
from pathlib import Path
import tempfile

import numpy as np

import skelebone.visualization as visualization
from skelebone import open3d_renderer
from scripts import visualize_historical_horse_ssdr as horse_visualization


class SkeleboneVisualizationTest(unittest.TestCase):
    def test_headless_open3d_uses_surfaceless_egl(self):
        environment = {}

        open3d_renderer.configure_render_environment(environment)

        self.assertEqual(environment["EGL_PLATFORM"], "surfaceless")

    def test_offscreen_open3d_does_not_trust_an_inaccessible_display(self):
        environment = {"DISPLAY": ":99"}

        open3d_renderer.configure_render_environment(environment)

        self.assertEqual(environment["EGL_PLATFORM"], "surfaceless")

    def test_preview_api_accepts_surface_faces(self):
        parameters = inspect.signature(
            visualization.save_skelebone_figure
        ).parameters

        self.assertIn("faces", parameters)

    def test_hood_camera_preserves_y_up_and_views_from_positive_z(self):
        points = np.array(
            [
                [-1.0, -2.0, -0.2],
                [1.0, 2.0, 0.2],
            ]
        )

        camera = visualization.compute_hood_camera(points, azimuth_degrees=12.0)

        np.testing.assert_allclose(camera.up, [0.0, 1.0, 0.0])
        self.assertGreater(camera.eye[2], camera.center[2])
        self.assertLess(abs(camera.eye[0] - camera.center[0]), camera.eye[2] - camera.center[2])
        self.assertGreater(camera.distance, 0.0)

    def test_horse_camera_places_the_motion_y_axis_horizontally(self):
        points = np.array(
            [
                [-0.2, -2.0, -1.0],
                [0.2, 2.0, 1.0],
            ]
        )

        camera = horse_visualization.compute_horizontal_horse_camera(points)
        view = camera.center - camera.eye
        view /= np.linalg.norm(view)
        screen_horizontal = np.cross(view, camera.up)
        screen_horizontal /= np.linalg.norm(screen_horizontal)

        np.testing.assert_allclose(camera.up, [0.0, 0.0, 1.0])
        self.assertGreater(abs(screen_horizontal[1]), 0.95)

    def test_parent_edges_omits_roots(self):
        parents = np.array([-1, 0, 1, -1, 3])

        edges = visualization.parent_edges(parents)

        np.testing.assert_array_equal(edges, [[0, 1], [1, 2], [3, 4]])

    def test_animation_frame_selection_spans_the_sequence(self):
        frames = open3d_renderer.select_animation_frames(101, max_frames=5)

        np.testing.assert_array_equal(frames, [0, 25, 50, 75, 100])

    def test_outer_bone_positions_apply_ssdr_transforms(self):
        points = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        weights = np.eye(2)
        transforms = np.broadcast_to(np.eye(4), (1, 2, 4, 4)).copy()
        transforms[0, 1, 2, 3] = 3.0

        positions = open3d_renderer.outer_bone_positions(
            points, weights, transforms
        )

        np.testing.assert_allclose(
            positions, [[[0.0, 0.0, 0.0], [2.0, 0.0, 3.0]]]
        )

    def test_dominant_bone_colors_are_deterministic_and_shared_by_label(self):
        weights = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.9, 0.1, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

        colors = open3d_renderer.dominant_bone_colors(weights)

        self.assertEqual(colors.shape, (4, 3))
        np.testing.assert_allclose(colors[0], colors[1])
        self.assertFalse(np.allclose(colors[0], colors[2]))
        self.assertTrue(np.all((colors >= 0.0) & (colors <= 1.0)))

    def test_blended_weight_colors_mix_every_bone_influence(self):
        weights = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.75, 0.25],
                [0.5, 0.5],
            ]
        )
        bone_colors = open3d_renderer.dominant_bone_colors(np.eye(2))

        colors = open3d_renderer.blended_weight_colors(weights)

        np.testing.assert_allclose(colors[0], bone_colors[0])
        np.testing.assert_allclose(colors[1], bone_colors[1])
        np.testing.assert_allclose(
            colors[2], 0.75 * bone_colors[0] + 0.25 * bone_colors[1]
        )
        np.testing.assert_allclose(colors[3], bone_colors.mean(axis=0))

    def test_stratified_bone_sample_keeps_every_dominant_bone(self):
        labels = np.array([0] * 20 + [1] * 3 + [2])

        indices = open3d_renderer.stratified_bone_sample(labels, max_points=8)

        self.assertLessEqual(len(indices), 8)
        np.testing.assert_array_equal(np.unique(labels[indices]), [0, 1, 2])
        np.testing.assert_array_equal(
            indices,
            open3d_renderer.stratified_bone_sample(labels, max_points=8),
        )

    def test_animation_writer_creates_a_looping_multiframe_gif(self):
        frames = [
            np.full((8, 8, 3), 255, dtype=np.uint8),
            np.zeros((8, 8, 3), dtype=np.uint8),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = open3d_renderer.save_animation_gif(
                Path(directory) / "preview.gif", frames, fps=10
            )
            from PIL import Image

            with Image.open(path) as image:
                self.assertEqual(image.n_frames, 2)
                self.assertEqual(image.info["loop"], 0)


if __name__ == "__main__":
    unittest.main()
