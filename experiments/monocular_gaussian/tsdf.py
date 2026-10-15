"""Fuse rendered Gaussian depth into a canonical TSDF mesh.

Restores the author's RigGS skeleton-extraction TSDF protocol. Camera and
fusion conventions follow GaussianAnimator's mesh utilities and
https://github.com/yaoyx689/RigGS. CUDA dependencies are imported lazily.
"""

import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path

import numpy as np


def validate_parameters(fid, voxel_size, sdf_trunc, depth_trunc, n_cameras, target_triangles):
    if not np.isfinite(fid) or not 0 <= fid <= 1:
        raise ValueError("fid must be within [0,1]")
    for name, value in (("voxel_size", voxel_size), ("sdf_trunc", sdf_trunc),
                        ("depth_trunc", depth_trunc)):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not isinstance(n_cameras, int) or n_cameras < 0:
        raise ValueError("n_cameras must be a nonnegative integer")
    if not isinstance(target_triangles, int) or target_triangles < 100:
        raise ValueError("target_triangles must be at least 100")


def spherical_camera_poses(count, radius):
    """Original Fibonacci cameras: stored R is camera-to-world, T is world-to-camera."""
    poses = []
    golden_angle = np.pi * (3 - np.sqrt(5))
    for index in range(count):
        y = 1 - 2 * index / (count - 1) if count > 1 else 0.
        ring = np.sqrt(max(0., 1 - y * y))
        position = radius * np.array([np.cos(golden_angle * index) * ring,
                                      y, np.sin(golden_angle * index) * ring])
        forward = -position / np.linalg.norm(position)
        up = np.array([0., -1., 0.])
        if abs(np.dot(forward, up)) > .9:
            up = np.array([0., 0., 1.])
        right = np.cross(up, forward)
        right /= np.linalg.norm(right)
        rotation = np.column_stack((right, np.cross(forward, right), forward))
        poses.append((rotation, -rotation.T @ position))
    return poses


def _generated_cameras(cameras, count, fid):
    if count == 0:
        return
    from scene.cameras import Camera

    reference = cameras[0]
    radius = float(np.mean([np.linalg.norm(cam.camera_center.detach().cpu().numpy())
                            for cam in cameras]))
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("camera radius must be finite and positive")
    for index, (rotation, translation) in enumerate(spherical_camera_poses(count, radius)):
        yield Camera(colmap_id=index, R=rotation, T=translation,
                     FoVx=reference.FoVx, FoVy=reference.FoVy,
                     image=reference.original_image, gt_alpha_mask=None,
                     image_name=f"tsdf_{index:03d}", uid=index, fid=fid)


def _canonical_surface(mesh, target_triangles):
    from skelebone.extraction.mesh import (
        TriangleMesh, close_boundary_loops, is_watertight,
        keep_largest_component, split_nonmanifold_vertices,
    )

    if mesh.is_empty() or not len(mesh.triangles):
        raise ValueError("TSDF fusion produced an empty surface")
    if len(mesh.triangles) > target_triangles:
        mesh = mesh.simplify_quadric_decimation(target_triangles)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    mesh.orient_triangles()
    surface = keep_largest_component(TriangleMesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles)))
    surface = keep_largest_component(split_nonmanifold_vertices(surface))
    capped = not is_watertight(surface)
    if capped:
        surface = close_boundary_loops(surface)
    if not is_watertight(surface):
        raise ValueError("TSDF surface is not watertight; no Poisson fallback is applied")
    return surface, capped


def extract_tsdf_mesh(model_path, output_path, *, source_path=None, iteration=40000,
                      fid=0., voxel_size=.01, sdf_trunc=.05, depth_trunc=12.,
                      n_cameras=300, target_triangles=8000):
    """Render one fixed time from all cameras, fuse depth, and export a closed proxy."""
    validate_parameters(fid, voxel_size, sdf_trunc, depth_trunc, n_cameras, target_triangles)
    output_path = Path(output_path)
    if output_path.suffix.lower() != ".ply":
        raise ValueError("TSDF output must use .ply")
    import itertools
    import open3d as o3d
    import torch
    from .riggs_loader import bundled_engine_root
    from .ssdr_evaluate import _load_models, deformation_attributes_for_render

    with redirect_stdout(io.StringIO()):
        dataset, models = _load_models(bundled_engine_root(), Path(model_path), iteration,
                                      source_path=source_path)
    from arguments import PipelineParams
    from gaussian_renderer import render

    parser = argparse.ArgumentParser(add_help=False)
    params = PipelineParams(parser)
    pipe = params.extract(parser.parse_args([]))
    cameras = models.scene.getTrainCameras() + models.scene.getTestCameras()
    if not cameras:
        raise ValueError("TSDF extraction requires calibrated cameras")
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size, sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    with torch.inference_mode():
        gaussians, deform = models.gaussians, models.deform
        xyz = gaussians.get_xyz
        time = torch.tensor([fid], dtype=xyz.dtype, device=xyz.device)
        time_input = time.unsqueeze(0).expand(len(xyz), -1) if deform.name == "mlp" else deform.deform.expand_time(time)
        values = deform.step(xyz, time_input, feature=gaussians.feature,
                             motion_mask=gaussians.motion_mask, is_training=False)
        rotation, scaling = deformation_attributes_for_render(
            values["d_rotation"], values["d_scaling"],
            use_isotropic_gaussians=gaussians.use_isotropic_gs)
        background = torch.zeros(3, device=xyz.device, dtype=xyz.dtype)
        views = itertools.chain(_generated_cameras(cameras, n_cameras, fid), cameras)
        for camera in views:
            result = render(camera, gaussians, pipe, background, values["d_xyz"],
                            rotation, scaling, d_opacity=values.get("d_opacity"),
                            d_color=values.get("d_color"), d_rot_as_res=dataset.d_rot_as_res)
            depth = result["depth"].detach().cpu().numpy().squeeze()
            rgb = result["render"].detach().cpu().numpy().transpose(1, 2, 0)
            if not np.isfinite(depth).all() or not np.isfinite(rgb).all():
                raise ValueError("nonfinite Gaussian RGB/depth in TSDF input")
            depth = np.ascontiguousarray(np.where(depth > 0, depth, 0), dtype=np.float32)
            rgb = np.ascontiguousarray(np.clip(rgb, 0, 1) * 255, dtype=np.uint8)
            intrinsic = o3d.camera.PinholeCameraIntrinsic(
                camera.image_width, camera.image_height,
                camera.image_width / (2 * np.tan(camera.FoVx / 2)),
                camera.image_height / (2 * np.tan(camera.FoVy / 2)),
                camera.image_width / 2, camera.image_height / 2)
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(rgb), o3d.geometry.Image(depth), depth_scale=1.,
                depth_trunc=depth_trunc, convert_rgb_to_intensity=False)
            extrinsic = camera.world_view_transform.T.detach().cpu().numpy().astype(np.float64)
            volume.integrate(rgbd, intrinsic, extrinsic)
    mesh = volume.extract_triangle_mesh()
    raw_vertices, raw_triangles = len(mesh.vertices), len(mesh.triangles)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path = output_path.with_name(output_path.stem + "_raw.ply")
    if not o3d.io.write_triangle_mesh(str(raw_path), mesh):
        raise OSError(f"could not write {raw_path}")
    surface, capped = _canonical_surface(mesh, target_triangles)
    canonical = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(surface.vertices), o3d.utility.Vector3iVector(surface.faces))
    if not o3d.io.write_triangle_mesh(str(output_path), canonical):
        raise OSError(f"could not write {output_path}")
    metadata = dict(surface="tsdf", model=str(Path(model_path).resolve()),
                    source_path=str(dataset.source_path), iteration=int(models.scene.loaded_iter),
                    fid=fid, voxel_size=voxel_size, sdf_trunc=sdf_trunc, depth_trunc=depth_trunc,
                    generated_cameras=n_cameras, dataset_cameras=len(cameras),
                    target_triangles=target_triangles, raw_mesh=raw_path.name,
                    raw_vertices=raw_vertices, raw_triangles=raw_triangles,
                    vertices=len(surface.vertices), triangles=len(surface.faces),
                    boundary_capped=capped, watertight=True)
    output_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--source-path", type=Path)
    parser.add_argument("--iteration", type=int, default=40000)
    parser.add_argument("--fid", type=float, default=0.)
    parser.add_argument("--voxel-size", type=float, default=.01)
    parser.add_argument("--sdf-trunc", type=float, default=.05)
    parser.add_argument("--depth-trunc", type=float, default=12.)
    parser.add_argument("--n-cameras", type=int, default=300)
    parser.add_argument("--target-triangles", type=int, default=8000)
    args = parser.parse_args()
    metadata = extract_tsdf_mesh(**vars(args))
    print(f"TSDF: {metadata['vertices']} vertices, {metadata['triangles']} triangles -> {args.output_path}")


if __name__ == "__main__":
    main()
