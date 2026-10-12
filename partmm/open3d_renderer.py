"""Standalone Open3D animation renderer for the HOOD PartMM example."""

from dataclasses import dataclass
import math
import os
from pathlib import Path

import numpy as np

from .types import PartMMOutput
from .visualization import outer_bone_positions, save_animation_gif


SURFACE_COLOR = (0.74, 0.78, 0.80)
OUTER_COLOR = (0.04, 0.48, 0.52)
BONE_COLOR = (0.91, 0.89, 0.84)
JOINT_COLOR = (0.82, 0.25, 0.16)


@dataclass(frozen=True)
class CameraView:
    eye: np.ndarray
    center: np.ndarray
    up: np.ndarray
    fov_degrees: float


def _open3d():
    os.environ.setdefault("EGL_PLATFORM", "surfaceless")
    try:
        import open3d as o3d
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "PartMM GIF rendering requires Open3D 0.19 and Pillow."
        ) from error
    return o3d


def _fit_camera(points, azimuth_degrees=12.0, elevation_degrees=3.0):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    low, high = points.min(axis=0), points.max(axis=0)
    center = (low + high) * 0.5
    extent = max(float(np.max(high - low)), 1e-6)
    fov = 30.0
    distance = extent / (2.0 * math.tan(math.radians(fov) * 0.5)) * 1.12
    azimuth = math.radians(azimuth_degrees)
    elevation = math.radians(elevation_degrees)
    direction = np.array(
        [
            math.sin(azimuth) * math.cos(elevation),
            math.sin(elevation),
            math.cos(azimuth) * math.cos(elevation),
        ]
    )
    return CameraView(
        eye=center + direction * distance,
        center=center,
        up=np.array([0.0, 1.0, 0.0]),
        fov_degrees=fov,
    )


def _material(o3d, shader="defaultLit", alpha=1.0):
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = shader
    material.base_color = [1.0, 1.0, 1.0, float(alpha)]
    material.base_roughness = 0.72
    material.base_reflectance = 0.18
    return material


def _surface(o3d, points, faces):
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(points),
        o3d.utility.Vector3iVector(faces),
    )
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(SURFACE_COLOR)
    return mesh


def _rotation_from_z(direction):
    direction = np.asarray(direction, dtype=np.float64)
    direction /= np.linalg.norm(direction)
    z_axis = np.array([0.0, 0.0, 1.0])
    cross = np.cross(z_axis, direction)
    sine = np.linalg.norm(cross)
    cosine = float(np.dot(z_axis, direction))
    if sine < 1e-10:
        return np.eye(3) if cosine > 0 else np.diag([1.0, -1.0, -1.0])
    axis = cross / sine
    skew = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )
    angle = np.arctan2(sine, cosine)
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def _segment(o3d, start, end, radius, color, resolution=12):
    direction = np.asarray(end) - np.asarray(start)
    length = float(np.linalg.norm(direction))
    if length <= 1e-9:
        return o3d.geometry.TriangleMesh()
    cylinder = o3d.geometry.TriangleMesh.create_cylinder(
        radius=radius, height=length, resolution=resolution, split=1
    )
    cylinder.rotate(_rotation_from_z(direction), center=(0.0, 0.0, 0.0))
    cylinder.translate((np.asarray(start) + np.asarray(end)) * 0.5)
    cylinder.paint_uniform_color(color)
    return cylinder


def _skeleton(o3d, joints, parents, extent):
    mesh = o3d.geometry.TriangleMesh()
    for child, parent in enumerate(np.asarray(parents, dtype=np.int64)):
        if parent >= 0:
            mesh += _segment(
                o3d, joints[int(parent)], joints[child], extent * 0.005, BONE_COLOR
            )
    for joint in joints:
        sphere = o3d.geometry.TriangleMesh.create_sphere(
            radius=extent * 0.012, resolution=14
        )
        sphere.translate(joint)
        sphere.paint_uniform_color(JOINT_COLOR)
        mesh += sphere
    mesh.compute_vertex_normals()
    return mesh


def _outer_bones(o3d, positions, rotations, extent):
    mesh = o3d.geometry.TriangleMesh()
    axis_colors = ((0.84, 0.16, 0.13), (0.13, 0.62, 0.25), (0.12, 0.34, 0.78))
    axis_length = extent * 0.060
    for position, rotation in zip(positions, rotations):
        sphere = o3d.geometry.TriangleMesh.create_sphere(
            radius=extent * 0.014, resolution=12
        )
        sphere.translate(position)
        sphere.paint_uniform_color(OUTER_COLOR)
        mesh += sphere
        for axis, color in enumerate(axis_colors):
            mesh += _segment(
                o3d,
                position,
                position + rotation[:, axis] * axis_length,
                extent * 0.0024,
                color,
                resolution=10,
            )
    mesh.compute_vertex_normals()
    return mesh


def _renderer(o3d, width, height):
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])
    renderer.scene.view.set_post_processing(False)
    renderer.scene.scene.set_sun_light(
        [0.35, -0.55, -0.75], [1.0, 0.98, 0.95], 62000
    )
    renderer.scene.scene.enable_sun_light(True)
    renderer.scene.scene.set_indirect_light_intensity(18000)
    return renderer


def _render(renderer, geometries, camera):
    renderer.scene.clear_geometry()
    for name, geometry, material in geometries:
        renderer.scene.add_geometry(name, geometry, material)
    renderer.setup_camera(
        camera.fov_degrees,
        camera.center.astype(np.float32),
        camera.eye.astype(np.float32),
        camera.up.astype(np.float32),
    )
    return np.asarray(renderer.render_to_image())[:, :, :3].astype(np.uint8)


def _compose(images, titles):
    from PIL import Image, ImageDraw, ImageFont

    height, width = images[0].shape[:2]
    title_height, margin, gap = 56, 18, 10
    canvas = Image.new(
        "RGB",
        (margin * 2 + width * 3 + gap * 2, margin * 2 + title_height + height),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
    for index, (array, title) in enumerate(zip(images, titles)):
        left = margin + index * (width + gap)
        bounds = draw.textbbox((0, 0), title, font=font)
        text_width = bounds[2] - bounds[0]
        draw.text(
            (left + (width - text_width) * 0.5, margin + 13),
            title,
            fill=(35, 40, 43),
            font=font,
        )
        canvas.paste(Image.fromarray(array), (left, margin + title_height))
    return np.asarray(canvas)


def save_partmm_gif(
    path,
    output: PartMMOutput,
    *,
    faces,
    parents,
    source_points,
    skinning_weights,
    max_frames=40,
    fps=12.0,
    width=340,
    height=420,
):
    """Render SMPL query, retrieved SSDR bones, and PartMM cloth together."""
    if not isinstance(output, PartMMOutput):
        raise TypeError("output must be a PartMMOutput")
    faces = np.asarray(faces, dtype=np.int64)
    outer_positions = outer_bone_positions(
        source_points, skinning_weights, output.bone_transforms
    )
    visible = np.concatenate(
        [output.points.reshape(-1, 3), output.joint_positions.reshape(-1, 3)], axis=0
    )
    camera = _fit_camera(visible)
    extent = max(float(np.ptp(visible, axis=0).max()), 1e-6)
    o3d = _open3d()
    renderer = _renderer(o3d, int(width), int(height))
    solid_material = _material(o3d)
    ghost_material = _material(o3d, shader="defaultLitTransparency", alpha=0.20)
    colored_material = _material(o3d)
    frame_indices = np.linspace(
        0, len(output.points) - 1, min(len(output.points), int(max_frames)), dtype=np.int64
    )
    frames = []
    for frame in frame_indices:
        surface = _surface(o3d, output.points[frame], faces)
        skeleton = _skeleton(o3d, output.joint_positions[frame], parents, extent)
        outer = _outer_bones(
            o3d,
            outer_positions[frame],
            output.bone_transforms[frame, :, :3, :3],
            extent,
        )
        panels = [
            _render(
                renderer,
                [("surface", surface, ghost_material), ("smpl", skeleton, colored_material)],
                camera,
            ),
            _render(
                renderer,
                [("surface", surface, ghost_material), ("ssdr", outer, colored_material)],
                camera,
            ),
            _render(renderer, [("surface", surface, solid_material)], camera),
        ]
        frames.append(_compose(panels, ["SMPL Query", "SSDR Outer Bones", "PartMM Output"]))
    return save_animation_gif(Path(path), frames, fps=fps)


__all__ = ["save_partmm_gif"]
