"""Fit a posed clothing template and retain Gaussian centers in its 1.1x volume."""

import argparse
import json
from pathlib import Path

import igl
import numpy as np
import open3d as o3d
import yaml


def inflate(vertices, factor):
    vertices = np.asarray(vertices, dtype=np.float64)
    center = (vertices.min(0) + vertices.max(0)) / 2
    return center + factor * (vertices - center), center


def contains(vertices, faces, queries, threshold):
    vertices = np.asarray(vertices, dtype=np.float64)
    queries = np.asarray(queries, dtype=np.float64)
    candidate = ((queries >= vertices.min(0)) & (queries <= vertices.max(0))).all(-1)
    winding = np.zeros(len(queries), dtype=np.float64)
    if candidate.any():
        winding[candidate] = np.asarray(igl.fast_winding_number_for_meshes(
            vertices, np.asarray(faces, dtype=np.int32), queries[candidate])).reshape(-1)
    return np.abs(winding) >= threshold, winding


def dilated_contains(vertices, faces, queries, offset, threshold=0.5):
    """Membership in the original volume plus a Euclidean surface-distance shell."""
    if offset < 0:
        raise ValueError('Outward offset must be nonnegative')
    inside, _ = contains(vertices, faces, queries, threshold)
    squared, _, _ = igl.point_mesh_squared_distance(
        np.asarray(queries, dtype=np.float64), np.asarray(vertices, dtype=np.float64),
        np.asarray(faces, dtype=np.int32))
    distance = np.sqrt(np.maximum(np.asarray(squared).reshape(-1), 0))
    distance[inside] *= -1
    return distance <= offset, distance


def subset_gaussians(params, keep):
    """Apply one point-index selection to all frames, leaving camera data intact."""
    dynamic = {'means3D', 'rgb_colors', 'unnorm_rotations'}
    static = {'seg_colors', 'logit_opacities', 'log_scales'}
    return {key: value[:, keep] if key in dynamic else value[keep] if key in static else value
            for key, value in params.items()}


def align_template(vertices, faces, points, options, seed):
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices),
                                    o3d.utility.Vector3iVector(faces))
    o3d.utility.random.seed(seed)
    source = mesh.sample_points_uniformly(options['samples'])
    target = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    target = target.voxel_down_sample(options['voxel_size'])
    target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=options['voxel_size'] * 4, max_nn=30))
    distance = options['max_correspondence_distance']
    initial = o3d.pipelines.registration.evaluate_registration(source, target, distance, np.eye(4))
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(
            o3d.pipelines.registration.TukeyLoss(options['tukey_scale'])),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=options['iterations']))
    transform = np.asarray(result.transformation)
    aligned = vertices @ transform[:3, :3].T + transform[:3, 3]
    angle = np.rad2deg(np.arccos(np.clip((np.trace(transform[:3, :3]) - 1) / 2, -1, 1)))
    record = dict(initial_fitness=initial.fitness, final_fitness=result.fitness,
                  initial_inlier_rmse=initial.inlier_rmse, final_inlier_rmse=result.inlier_rmse,
                  rotation_degrees=float(angle), translation=transform[:3, 3].tolist())
    return aligned, transform, record


def main(path):
    options = yaml.safe_load(path.read_text())
    output = Path(options['output'])
    frame_ids = np.load(output / 'frame_ids.npy').tolist()
    vertices = np.load(output / 'posed_template.npy', mmap_mode='r')
    faces = np.load(output / 'faces.npy')
    with np.load(options['d3dgs_params']) as archive:
        points = archive['means3D'][frame_ids]
        opacity = 1 / (1 + np.exp(-archive['logit_opacities'].reshape(-1)))
    masks = np.lib.format.open_memmap(output / 'keep_mask.npy', mode='w+', dtype=bool, shape=points.shape[:2])
    volumes = np.lib.format.open_memmap(output / 'volume_vertices.npy', mode='w+', dtype=np.float32, shape=vertices.shape)
    transforms = np.zeros((len(frame_ids), 4, 4))
    records = []
    for index, frame in enumerate(frame_ids):
        aligned, transforms[index], registration = align_template(
            np.asarray(vertices[index], dtype=np.float64), faces,
            np.asarray(points[index], dtype=np.float64), options['registration'], options['seed'])
        volume, center = inflate(aligned, options['scale'])
        keep, winding = contains(volume, faces, points[index], options['winding_threshold'])
        masks[index], volumes[index] = keep, volume
        if frame in [240, 254, 269, 299]:
            mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(volume), o3d.utility.Vector3iVector(faces))
            mesh.compute_vertex_normals()
            o3d.io.write_triangle_mesh(str(output / f'volume_{frame:04d}.ply'), mesh)
        record = dict(frame=frame, total=len(keep), kept=int(keep.sum()), removed=int((~keep).sum()),
                      removed_fraction=float((~keep).mean()), scale_center=center.tolist(),
                      removed_opacity_mean=float(opacity[~keep].mean()) if (~keep).any() else None,
                      kept_opacity_mean=float(opacity[keep].mean()) if keep.any() else None,
                      uncertain_winding_count=int((np.abs(np.abs(winding) - .5) < .05).sum()),
                      registration=registration)
        records.append(record)
        print(json.dumps(record), flush=True)
    masks.flush()
    volumes.flush()
    np.save(output / 'alignment.npy', transforms)
    record = dict(config=str(path.resolve()), source=options['d3dgs_params'], frames=frame_ids,
                  scale=options['scale'], volume_definition='abs(generalized winding number) >= 0.5',
                  scale_center=options['scale_center'], selection='Gaussian center, not ellipsoid overlap',
                  temporal_policy='Per-frame masks; original point indices preserved; no global correspondence deletion',
                  mean_removed_fraction=float(np.mean([r['removed_fraction'] for r in records])), records=records)
    (output / 'filter_summary.json').write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    main(parser.parse_args().config)
