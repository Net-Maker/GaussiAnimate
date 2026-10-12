"""Open3D renderer for publication-quality Skelebone previews."""

import os
from pathlib import Path

import numpy as np


SURFACE_COLOR = (0.74, 0.78, 0.80)
CURVE_COLOR = (0.04, 0.48, 0.52)
BONE_COLOR = (0.91, 0.89, 0.84)
JOINT_COLOR = (0.82, 0.25, 0.16)

_BONE_PALETTE = np.asarray(
    [
        (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40),
        (148, 103, 189), (140, 86, 75), (227, 119, 194), (127, 127, 127),
        (188, 189, 34), (23, 190, 207), (57, 106, 177), (218, 124, 48),
        (62, 150, 81), (204, 37, 41), (107, 76, 154), (146, 36, 40),
        (148, 139, 61), (83, 81, 84), (231, 186, 82), (117, 107, 177),
        (141, 211, 199), (190, 186, 218), (251, 128, 114), (128, 177, 211),
        (253, 180, 98), (179, 222, 105), (252, 205, 229), (217, 217, 217),
        (204, 235, 197), (255, 237, 111),
    ],
    dtype=np.float64,
) / 255.0


def select_animation_frames(frame_count, max_frames=48):
    """Select uniformly spaced source frames, including both endpoints."""
    frame_count = int(frame_count)
    max_frames = int(max_frames)
    if frame_count < 1 or max_frames < 1:
        raise ValueError("frame_count and max_frames must be positive")
    count = min(frame_count, max_frames)
    return np.linspace(0, frame_count - 1, count, dtype=np.int64)


def outer_bone_positions(source_points, skinning_weights, bone_transforms):
    """Track SSDR influence centroids through the extracted transforms."""
    source_points = np.asarray(source_points, dtype=np.float64)
    weights = np.asarray(skinning_weights, dtype=np.float64)
    transforms = np.asarray(bone_transforms, dtype=np.float64)
    weight_sums = weights.sum(axis=0)
    if weights.shape[0] != len(source_points) or np.any(weight_sums <= 1e-12):
        raise ValueError("skinning weights must be valid [points,bones] weights")
    rest_centers = (weights.T @ source_points) / weight_sums[:, None]
    centers_h = np.concatenate(
        [rest_centers, np.ones((len(rest_centers), 1), dtype=np.float64)], axis=1
    )
    if transforms.ndim != 4 or transforms.shape[1:] != (len(rest_centers), 4, 4):
        raise ValueError("bone transforms must have shape [frames,bones,4,4]")
    return np.einsum("tbij,bj->tbi", transforms, centers_h)[..., :3]


def dominant_bone_colors(skinning_weights):
    """Map each point's strongest SSDR influence to a fixed qualitative color."""

    weights = np.asarray(skinning_weights, dtype=np.float64)
    if weights.ndim != 2 or weights.shape[1] < 1 or not np.all(np.isfinite(weights)):
        raise ValueError("skinning_weights must be a finite [points,bones] matrix")
    labels = np.argmax(weights, axis=1)
    return _BONE_PALETTE[labels % len(_BONE_PALETTE)].copy()


def blended_weight_colors(skinning_weights):
    """Blend the qualitative bone palette using every SSDR influence weight."""

    weights = np.asarray(skinning_weights, dtype=np.float64)
    if (
        weights.ndim != 2
        or weights.shape[1] < 1
        or not np.all(np.isfinite(weights))
        or np.any(weights < -1e-12)
    ):
        raise ValueError("skinning_weights must be a finite non-negative matrix")
    row_sums = weights.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 1e-12):
        raise ValueError("every point must have positive total skinning weight")
    normalized = np.maximum(weights, 0.0) / row_sums
    bone_colors = _BONE_PALETTE[np.arange(weights.shape[1]) % len(_BONE_PALETTE)]
    return normalized @ bone_colors


def stratified_bone_sample(labels, max_points):
    """Deterministically reduce a cloud while retaining every dominant bone."""

    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    max_points = int(max_points)
    if len(labels) == 0 or max_points < 1 or np.any(labels < 0):
        raise ValueError("labels must be non-empty/non-negative and max_points positive")
    if len(labels) <= max_points:
        return np.arange(len(labels), dtype=np.int64)

    unique, counts = np.unique(labels, return_counts=True)
    if len(unique) > max_points:
        raise ValueError("max_points is smaller than the number of dominant bones")
    quotas = np.ones(len(unique), dtype=np.int64)
    remaining = max_points - len(unique)
    capacity = counts - 1
    if remaining and capacity.sum():
        shares = remaining * capacity / capacity.sum()
        additions = np.minimum(np.floor(shares).astype(np.int64), capacity)
        quotas += additions
        unassigned = remaining - int(additions.sum())
        order = np.argsort(-(shares - additions), kind="stable")
        for index in order:
            if unassigned == 0:
                break
            available = counts[index] - quotas[index]
            if available > 0:
                quotas[index] += 1
                unassigned -= 1

    sampled = []
    for label, quota in zip(unique, quotas):
        candidates = np.flatnonzero(labels == label)
        positions = np.linspace(0, len(candidates) - 1, int(quota), dtype=np.int64)
        sampled.append(candidates[positions])
    return np.sort(np.concatenate(sampled)).astype(np.int64, copy=False)


def save_animation_gif(path, frames, fps=12.0):
    """Encode RGB arrays as a looping GIF without changing their geometry."""
    from PIL import Image

    if fps <= 0 or not frames:
        raise ValueError("fps must be positive and frames must be non-empty")
    images = [Image.fromarray(np.asarray(frame, dtype=np.uint8), mode="RGB") for frame in frames]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=max(1, round(1000.0 / fps)),
        loop=0,
        disposal=2,
        optimize=False,
    )
    return path


def configure_render_environment(environment=None):
    """Select Open3D's surfaceless EGL backend for offscreen rendering."""
    environment = os.environ if environment is None else environment
    environment.setdefault("EGL_PLATFORM", "surfaceless")
    return environment


def _open3d():
    configure_render_environment()
    try:
        import open3d as o3d
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "Open3D preview rendering requires a compatible Open3D environment. "
            "Install the visualization extra with numpy<1.26."
        ) from error
    return o3d


def _material(o3d, *, color, shader="defaultLit", alpha=1.0):
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = shader
    material.base_color = [*color, float(alpha)]
    material.base_roughness = 0.72
    material.base_reflectance = 0.18
    return material


def _surface_geometry(o3d, points, faces):
    if faces is None or len(faces) == 0:
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)
        cloud.paint_uniform_color(SURFACE_COLOR)
        return cloud, "defaultUnlit"
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(points),
        o3d.utility.Vector3iVector(faces),
    )
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(SURFACE_COLOR)
    return mesh, "defaultLit"


def _colored_point_cloud(o3d, points, colors):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    cloud.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))
    return cloud


def _bone_marker_mesh(o3d, positions, colors, radius):
    mesh = o3d.geometry.TriangleMesh()
    for position, color in zip(positions, colors):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=12)
        sphere.translate(position)
        sphere.paint_uniform_color(color)
        mesh += sphere
    mesh.compute_vertex_normals()
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


def _tube_mesh(o3d, points, edges, *, radius, joint_radius, color, sparse_joints=False):
    mesh = o3d.geometry.TriangleMesh()
    degree = np.zeros(len(points), dtype=np.int64)
    for start_index, end_index in np.asarray(edges, dtype=np.int64):
        start = points[int(start_index)]
        end = points[int(end_index)]
        direction = end - start
        length = float(np.linalg.norm(direction))
        if length <= 1e-9:
            continue
        cylinder = o3d.geometry.TriangleMesh.create_cylinder(
            radius=radius, height=length, resolution=16, split=2
        )
        cylinder.rotate(_rotation_from_z(direction), center=(0.0, 0.0, 0.0))
        cylinder.translate((start + end) * 0.5)
        mesh += cylinder
        degree[int(start_index)] += 1
        degree[int(end_index)] += 1

    joint_indices = np.flatnonzero(degree != 2) if sparse_joints else np.arange(len(points))
    for index in joint_indices:
        sphere = o3d.geometry.TriangleMesh.create_sphere(
            radius=joint_radius, resolution=18
        )
        sphere.translate(points[int(index)])
        mesh += sphere
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(color)
    return mesh


def _segment_mesh(o3d, start, end, radius, color, resolution=10):
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


def _outer_bone_mesh(o3d, positions, rotations, *, extent):
    mesh = o3d.geometry.TriangleMesh()
    sphere_radius = extent * 0.0105
    axis_length = extent * 0.042
    axis_radius = extent * 0.0018
    axis_colors = ((0.84, 0.16, 0.13), (0.13, 0.62, 0.25), (0.12, 0.34, 0.78))
    for position, rotation in zip(positions, rotations):
        sphere = o3d.geometry.TriangleMesh.create_sphere(
            radius=sphere_radius, resolution=12
        )
        sphere.translate(position)
        sphere.paint_uniform_color(CURVE_COLOR)
        mesh += sphere
        for axis, color in enumerate(axis_colors):
            mesh += _segment_mesh(
                o3d,
                position,
                position + rotation[:, axis] * axis_length,
                axis_radius,
                color,
            )
    mesh.compute_vertex_normals()
    return mesh


def _new_renderer(o3d, width, height):
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])
    renderer.scene.view.set_post_processing(False)
    renderer.scene.scene.set_sun_light(
        [0.35, -0.55, -0.75], [1.0, 0.98, 0.95], 62000
    )
    renderer.scene.scene.enable_sun_light(True)
    renderer.scene.scene.set_indirect_light_intensity(18000)
    return renderer


def _render_panel(o3d, renderer, geometries, camera):
    renderer.scene.clear_geometry()
    for name, geometry, material in geometries:
        renderer.scene.add_geometry(name, geometry, material)
    renderer.setup_camera(
        camera.fov_degrees,
        camera.center.astype(np.float32),
        camera.eye.astype(np.float32),
        camera.up.astype(np.float32),
    )
    image = np.asarray(renderer.render_to_image())
    return image[:, :, :3].astype(np.uint8)


def _compose_panel_frame(images, titles, *, title_height=70, font_size=27):
    from PIL import Image, ImageDraw, ImageFont

    panel_height, panel_width = images[0].shape[:2]
    outer_margin = 18
    gap = 10
    canvas_width = outer_margin * 2 + panel_width * len(images) + gap * (len(images) - 1)
    canvas_height = outer_margin * 2 + title_height + panel_height
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size
        )
    except OSError:
        font = ImageFont.load_default()

    for index, (array, title) in enumerate(zip(images, titles)):
        left = outer_margin + index * (panel_width + gap)
        bounds = draw.textbbox((0, 0), title, font=font)
        text_width = bounds[2] - bounds[0]
        draw.text(
            (left + (panel_width - text_width) * 0.5, outer_margin + 17),
            title,
            fill=(35, 40, 43),
            font=font,
        )
        canvas.paste(Image.fromarray(array), (left, outer_margin + title_height))
    return np.asarray(canvas)


def _compose_panels(images, titles, output_path):
    from PIL import Image

    composed = _compose_panel_frame(images, titles)
    Image.fromarray(composed).save(output_path, optimize=True)
    return Path(output_path)


def render_skelebone_preview(
    *,
    points,
    faces,
    curve_points,
    curve_edges,
    joints,
    joint_edges,
    camera,
    output_path,
    width,
    height,
):
    """Render input, curve skeleton, and final rig from one calibrated view."""
    o3d = _open3d()
    extent = max(float(np.ptp(points, axis=0).max()), 1e-6)
    surface, surface_shader = _surface_geometry(o3d, points, faces)
    curve = _tube_mesh(
        o3d,
        curve_points,
        curve_edges,
        radius=extent * 0.0032,
        joint_radius=extent * 0.0075,
        color=CURVE_COLOR,
        sparse_joints=True,
    )
    bones = _tube_mesh(
        o3d,
        joints,
        joint_edges,
        radius=extent * 0.0052,
        joint_radius=extent * 0.0125,
        color=BONE_COLOR,
    )
    joint_spheres = _tube_mesh(
        o3d,
        joints,
        np.empty((0, 2), dtype=np.int64),
        radius=extent * 0.001,
        joint_radius=extent * 0.0135,
        color=JOINT_COLOR,
    )

    neutral = (1.0, 1.0, 1.0)
    surface_material = _material(o3d, color=neutral, shader=surface_shader)
    ghost_shader = "defaultLitTransparency" if faces is not None else "defaultUnlit"
    ghost_material = _material(
        o3d, color=neutral, shader=ghost_shader, alpha=0.24
    )
    curve_material = _material(o3d, color=neutral)
    bone_material = _material(o3d, color=neutral)
    joint_material = _material(o3d, color=neutral)

    renderer = _new_renderer(o3d, width, height)
    panels = [
        _render_panel(
            o3d, renderer, [("surface", surface, surface_material)], camera
        ),
        _render_panel(
            o3d,
            renderer,
            [("surface", surface, ghost_material), ("curve", curve, curve_material)],
            camera,
        ),
        _render_panel(
            o3d,
            renderer,
            [
                ("surface", surface, ghost_material),
                ("bones", bones, bone_material),
                ("joints", joint_spheres, joint_material),
            ],
            camera,
        ),
    ]
    return _compose_panels(
        panels,
        ["Input Surface", "Curve Skeleton", "Skelebone Rig"],
        output_path,
    )


def render_skelebone_animation(
    *,
    points,
    faces,
    outer_positions,
    outer_rotations,
    joints,
    joint_edges,
    camera,
    output_path,
    width=340,
    height=500,
    max_frames=40,
    fps=12.0,
):
    """Render a shared-camera AMA animation of surface, outer bones, and rig."""
    points = np.asarray(points, dtype=np.float64)
    outer_positions = np.asarray(outer_positions, dtype=np.float64)
    outer_rotations = np.asarray(outer_rotations, dtype=np.float64)
    joints = np.asarray(joints, dtype=np.float64)
    frame_count = len(points)
    if outer_positions.shape[:2] != outer_rotations.shape[:2]:
        raise ValueError("outer positions and rotations must share [frames,bones]")
    if len(outer_positions) != frame_count or len(joints) != frame_count:
        raise ValueError("surface, outer bones, and joints must share frame count")

    o3d = _open3d()
    extent = max(float(np.ptp(points.reshape(-1, 3), axis=0).max()), 1e-6)
    neutral = (1.0, 1.0, 1.0)
    surface_material = _material(o3d, color=neutral)
    ghost_material = _material(
        o3d, color=neutral, shader="defaultLitTransparency", alpha=0.20
    )
    colored_material = _material(o3d, color=neutral)
    renderer = _new_renderer(o3d, int(width), int(height))
    images = []
    for frame in select_animation_frames(frame_count, max_frames=max_frames):
        surface, _ = _surface_geometry(o3d, points[frame], faces)
        outer = _outer_bone_mesh(
            o3d, outer_positions[frame], outer_rotations[frame], extent=extent
        )
        bones = _tube_mesh(
            o3d,
            joints[frame],
            joint_edges,
            radius=extent * 0.0040,
            joint_radius=extent * 0.0095,
            color=BONE_COLOR,
        )
        joint_spheres = _tube_mesh(
            o3d,
            joints[frame],
            np.empty((0, 2), dtype=np.int64),
            radius=extent * 0.001,
            joint_radius=extent * 0.0105,
            color=JOINT_COLOR,
        )
        panels = [
            _render_panel(
                o3d, renderer, [("surface", surface, surface_material)], camera
            ),
            _render_panel(
                o3d,
                renderer,
                [("surface", surface, ghost_material), ("outer", outer, colored_material)],
                camera,
            ),
            _render_panel(
                o3d,
                renderer,
                [
                    ("surface", surface, ghost_material),
                    ("bones", bones, colored_material),
                    ("joints", joint_spheres, colored_material),
                ],
                camera,
            ),
        ]
        images.append(
            _compose_panel_frame(
                panels,
                ["Dynamic Surface", "SSDR Outer Bones", "Inner Skeleton"],
                title_height=56,
                font_size=22,
            )
        )
    return save_animation_gif(output_path, images, fps=fps)


def render_ssdr_comparison_animation(
    *,
    observed_points,
    reconstructed_points,
    rest_pose,
    skinning_weights,
    bone_transforms,
    camera,
    output_path,
    width=500,
    height=560,
    max_frames=48,
    max_points=14000,
    fps=12.0,
):
    """Render registered observations beside their colored SSDR reconstruction."""

    observed = np.asarray(observed_points, dtype=np.float64)
    reconstructed = np.asarray(reconstructed_points, dtype=np.float64)
    rest_pose = np.asarray(rest_pose, dtype=np.float64)
    weights = np.asarray(skinning_weights, dtype=np.float64)
    transforms = np.asarray(bone_transforms, dtype=np.float64)
    if observed.ndim != 3 or observed.shape[-1] != 3:
        raise ValueError("observed_points must have shape [frames,points,3]")
    if reconstructed.shape != observed.shape:
        raise ValueError("reconstructed_points must match observed_points")
    if rest_pose.shape != observed.shape[1:] or weights.shape[0] != observed.shape[1]:
        raise ValueError("rest pose and weights must match the point count")
    if transforms.shape != (len(observed), weights.shape[1], 4, 4):
        raise ValueError("bone_transforms must have shape [frames,bones,4,4]")

    labels = np.argmax(weights, axis=1)
    indices = stratified_bone_sample(labels, max_points=max_points)
    colors = blended_weight_colors(weights)[indices]
    bone_colors = _BONE_PALETTE[np.arange(weights.shape[1]) % len(_BONE_PALETTE)]
    bone_positions = outer_bone_positions(rest_pose, weights, transforms)
    extent = max(float(np.ptp(observed.reshape(-1, 3), axis=0).max()), 1e-6)

    o3d = _open3d()
    point_material = _material(o3d, color=(1.0, 1.0, 1.0), shader="defaultUnlit")
    point_material.point_size = 3.2
    marker_material = _material(o3d, color=(1.0, 1.0, 1.0))
    renderer = _new_renderer(o3d, int(width), int(height))
    images = []

    for frame in select_animation_frames(len(observed), max_frames=max_frames):
        input_cloud = _colored_point_cloud(o3d, observed[frame, indices], colors)
        reconstructed_cloud = _colored_point_cloud(
            o3d, reconstructed[frame, indices], colors
        )
        markers = _bone_marker_mesh(
            o3d,
            bone_positions[frame],
            bone_colors,
            radius=extent * 0.009,
        )
        frame_rmse = float(
            np.sqrt(np.mean((reconstructed[frame] - observed[frame]) ** 2))
        )
        panels = [
            _render_panel(
                o3d,
                renderer,
                [("input", input_cloud, point_material)],
                camera,
            ),
            _render_panel(
                o3d,
                renderer,
                [
                    ("reconstruction", reconstructed_cloud, point_material),
                    ("bones", markers, marker_material),
                ],
                camera,
            ),
        ]
        images.append(
            _compose_panel_frame(
                panels,
                [
                    f"Input | frame {int(frame):03d}",
                    f"SSDR | {weights.shape[1]} bones | RMSE {frame_rmse:.4f}",
                ],
                title_height=58,
                font_size=21,
            )
        )
    return save_animation_gif(output_path, images, fps=fps)
