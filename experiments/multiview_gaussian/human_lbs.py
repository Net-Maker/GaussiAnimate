"""Drive every human Gaussian with the paper's robust SMPL+LBS baseline."""

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.methods import (
    HeldOutPointProtocol,
    PointErrorAccumulator,
    save_prediction_diagnostic,
    write_prediction_artifacts,
)
from experiments.mesh.robust_weights import transfer_skin_weights
from experiments.mesh.vto_lbs import (
    _compute_body_vertices_and_transforms,
    _deform_cloth_with_lbs,
)
from skelebone.extraction.mesh import (
    MeshConfig,
    TriangleMesh,
    is_watertight,
    remove_small_components,
    write_off,
)


def _reconstruct_transfer_surface(points, config, seed=42):
    """Build the largest Poisson surface; robust transfer permits open boundaries."""

    import open3d as o3d

    o3d.utility.random.seed(int(seed))
    points = np.asarray(points, dtype=np.float64)
    extent = float(np.ptp(points, axis=0).max())
    radius = config.normal_radius or max(extent * 0.02, 1e-6)
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius,
            max_nn=config.normal_neighbors,
        )
    )
    cloud.orient_normals_consistent_tangent_plane(
        config.normal_consistency_neighbors
    )
    reconstructed, densities = (
        o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            cloud,
            depth=config.poisson_depth,
            n_threads=1,
        )
    )
    densities = np.asarray(densities)
    if len(densities) and config.density_quantile > 0:
        reconstructed.remove_vertices_by_mask(
            densities < np.quantile(densities, config.density_quantile)
        )
    reconstructed.orient_triangles()
    mesh = remove_small_components(
        TriangleMesh(
            np.asarray(reconstructed.vertices),
            np.asarray(reconstructed.triangles),
        ),
        config.minimum_component_triangles,
    )
    if len(mesh.faces) > config.target_triangles:
        decimation_mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(mesh.vertices),
            o3d.utility.Vector3iVector(mesh.faces.astype(np.int32)),
        )
        decimated = decimation_mesh.simplify_quadric_decimation(
            target_number_of_triangles=config.target_triangles
        )
        decimated.orient_triangles()
        mesh = remove_small_components(
            TriangleMesh(
                np.asarray(decimated.vertices),
                np.asarray(decimated.triangles),
            ),
            config.minimum_component_triangles,
        )
    if not len(mesh.faces):
        raise RuntimeError("Poisson reconstruction produced no transfer surface")
    return mesh


def _project_mesh_weights(points, vertices, faces, weights):
    """Interpolate target-mesh weights at arbitrary Gaussian centers."""

    import igl

    squared_distance, face_ids, closest = igl.point_mesh_squared_distance(
        np.asarray(points, dtype=np.float64),
        np.asarray(vertices, dtype=np.float64),
        np.asarray(faces, dtype=np.int32),
    )
    triangles = np.asarray(faces, dtype=np.int32)[face_ids]
    triangle_vertices = np.asarray(vertices, dtype=np.float64)[triangles]
    barycentric_fn = (
        igl.barycentric_coordinates_tri
        if hasattr(igl, "barycentric_coordinates_tri")
        else igl.barycentric_coordinates
    )
    barycentric = barycentric_fn(
        np.ascontiguousarray(closest),
        np.ascontiguousarray(triangle_vertices[:, 0]),
        np.ascontiguousarray(triangle_vertices[:, 1]),
        np.ascontiguousarray(triangle_vertices[:, 2]),
    )
    projected = np.einsum(
        "vi,vib->vb",
        barycentric,
        np.asarray(weights, dtype=np.float64)[triangles],
    )
    projected = np.maximum(projected, 0.0)
    projected /= np.maximum(projected.sum(axis=1, keepdims=True), 1e-12)
    return projected.astype(np.float32), squared_distance.astype(np.float32)


def run_human_lbs(
    params,
    motion,
    smpl_model_path,
    output_dir,
    *,
    train_ratio=0.8,
    poisson_depth=9,
    target_triangles=20000,
    seed=42,
):
    """Fit robust weights on frame zero and drive the held-out pose interval."""

    import torch
    from smplx import SMPL

    params = Path(params)
    motion = Path(motion)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(params, allow_pickle=False) as archive:
        means = archive["means3D"]
        frames, points, dimensions = means.shape
        if dimensions != 3:
            raise ValueError("means3D must have shape [frames, points, 3]")
        frame_zero = np.asarray(means[0], dtype=np.float32)
    protocol = HeldOutPointProtocol.from_shape(
        (frames, points, dimensions),
        train_ratio,
    )
    with np.load(motion, allow_pickle=False) as archive:
        poses = np.asarray(archive["axis_angle"], dtype=np.float32)
        translations = np.asarray(
            archive["root_translation"], dtype=np.float32
        )
        if "betas" not in archive.files:
            raise ValueError(
                "motion archive predates the public Robust LBS protocol; "
                "re-export it with experiments.multiview_gaussian.human_pose"
            )
        betas = np.asarray(archive["betas"], dtype=np.float32)
        body_scale = float(archive["body_scale"])
    if len(poses) != frames or len(translations) != frames:
        raise ValueError("motion and Gaussian sequence frame counts differ")
    training_frames = protocol.training_frames

    mesh_config = MeshConfig(
        poisson_depth=int(poisson_depth),
        target_triangles=int(target_triangles),
    )
    target_mesh = _reconstruct_transfer_surface(
        frame_zero,
        mesh_config,
        seed=seed,
    )
    write_off(output_dir / "frame0_surface.off", target_mesh)

    model = SMPL(
        str(smpl_model_path),
        gender="neutral",
        batch_size=frames,
    )
    body_vertices, transforms, body_faces, body_weights = (
        _compute_body_vertices_and_transforms(
            model,
            poses.reshape(frames, -1),
            betas,
        )
    )
    body_vertices *= body_scale
    transforms[:, :, :3, 3] *= body_scale
    body_vertices += translations[:, None, :]

    surface_weights, matched, smoothed, transfer_diagnostics = transfer_skin_weights(
        body_vertices[0],
        body_faces,
        body_weights,
        target_mesh.vertices,
        target_mesh.faces,
        search_radius_scale=0.05,
        normal_threshold_degrees=30.0,
        smooth_steps=10,
        smooth_alpha=0.2,
        return_diagnostics=True,
    )
    gaussian_weights, projection_squared_distance = _project_mesh_weights(
        frame_zero,
        target_mesh.vertices,
        target_mesh.faces,
        surface_weights,
    )

    relative_input = frame_zero - translations[0][None, :]
    prediction = _deform_cloth_with_lbs(
        relative_input,
        gaussian_weights,
        transforms,
    )
    prediction += translations[:, None, :]

    error = PointErrorAccumulator()
    with np.load(params, allow_pickle=False) as archive:
        means = archive["means3D"]
        diagnostic_target = np.asarray(
            means[training_frames],
            dtype=np.float32,
        )
        for frame in range(training_frames, frames):
            error.update(
                prediction[frame],
                means[frame],
            )
    artifacts = write_prediction_artifacts(
        output_dir,
        protocol,
        test_prediction=prediction[training_frames:],
        full_prediction=prediction,
    )
    save_prediction_diagnostic(
        output_dir / "diagnostic.png",
        frame_zero,
        prediction[training_frames],
        diagnostic_target,
        seed=seed,
    )
    np.savez_compressed(
        output_dir / "robust_weights.npz",
        surface_weights=surface_weights,
        gaussian_weights=gaussian_weights,
        matched=matched,
        vertices_to_smooth=smoothed,
        component_anchors=transfer_diagnostics["component_anchors"],
        projection_squared_distance=projection_squared_distance,
    )
    artifacts.update(
        {
            "surface": "frame0_surface.off",
            "weights": "robust_weights.npz",
            "diagnostic": "diagnostic.png",
        }
    )

    metadata = {
        "schema_version": 1,
        "case": str(np.load(motion, allow_pickle=False)["case"]),
        **protocol.metadata(),
        "method": "smpl_robust_lbs",
        "protocol": {
            "split": "contiguous_prefix_80_20",
            "fit_frames": [0],
            "driven_points": "all_gaussians",
            "source_weights": "SMPL_NEUTRAL_lbs_weights",
            "point_cloud_adapter": (
                "poisson_surface_then_robust_transfer_then_surface_projection"
            ),
            "robust_transfer": (
                "closest_surface_normal_filter_biharmonic_inpainting_smoothing"
            ),
            "search_radius_scale": 0.05,
            "normal_threshold_degrees": 30.0,
            "smooth_steps": 10,
            "smooth_alpha": 0.2,
            "poisson_depth": int(poisson_depth),
            "target_triangles": int(target_triangles),
            "seed": int(seed),
        },
        "surface": {
            "vertices": len(target_mesh.vertices),
            "triangles": len(target_mesh.faces),
            "watertight": is_watertight(target_mesh),
            "components": transfer_diagnostics["component_count"],
            "component_anchors": int(
                np.count_nonzero(transfer_diagnostics["component_anchors"])
            ),
            "matched_vertices": int(np.count_nonzero(matched)),
            "smoothed_vertices": int(np.count_nonzero(smoothed)),
        },
        "metrics": error.metrics(),
        "artifacts": artifacts,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", required=True)
    parser.add_argument("--motion", required=True)
    parser.add_argument("--smpl-model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--poisson-depth", type=int, default=9)
    parser.add_argument("--target-triangles", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = run_human_lbs(
        args.params,
        args.motion,
        args.smpl_model,
        args.output_dir,
        train_ratio=args.train_ratio,
        poisson_depth=args.poisson_depth,
        target_triangles=args.target_triangles,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
