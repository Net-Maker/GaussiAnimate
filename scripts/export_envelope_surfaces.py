"""Extract actual outward-distance envelope boundaries for scientific display."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import open3d as o3d
from skimage.measure import marching_cubes

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments/multiview_gaussian'))
from template_volume import dilated_contains


def main(args):
    summary = json.loads((args.experiment / 'filter_summary.json').read_text())
    mesh = o3d.io.read_triangle_mesh(str(args.experiment / 'selection_template_aligned.ply'))
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    factors = args.factors
    offsets = [(factor - 1) * summary['reference_span_m'] / 2 for factor in factors]
    padding = max(offsets) + 2 * args.spacing
    low, high = vertices.min(0) - padding, vertices.max(0) + padding
    axes = [np.arange(low[i], high[i] + args.spacing, args.spacing) for i in range(3)]
    shape = tuple(len(axis) for axis in axes)
    field = np.empty(int(np.prod(shape)), np.float32)
    for start in range(0, len(field), 150000):
        stop = min(start + 150000, len(field))
        indices = np.unravel_index(np.arange(start, stop), shape)
        points = np.column_stack([axes[i][indices[i]] for i in range(3)])
        _, field[start:stop] = dilated_contains(vertices, faces, points, 0)
        print(f'Sampled distance field {stop}/{len(field)}', flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for factor, offset in zip(factors, offsets):
        v, f, _, _ = marching_cubes(field.reshape(shape), level=offset,
                                   spacing=(args.spacing,) * 3)
        v += low
        envelope = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v), o3d.utility.Vector3iVector(f))
        envelope.compute_vertex_normals()
        path = args.output / f'envelope_{factor:.1f}.ply'
        o3d.io.write_triangle_mesh(str(path), envelope)
        # The surface is sampled for display; the Gaussian selection uses exact triangle distances.
        _, measured = dilated_contains(vertices, faces, v[::37], offset)
        error = float(np.abs(measured - offset).max())
        assert error < args.spacing * 2
        records.append(dict(factor=factor, offset_m=offset, mesh=str(path.resolve()),
                            display_bounds_min=v.min(0).tolist(), display_bounds_max=v.max(0).tolist(),
                            sampled_surface_max_distance_error_m=error))
    result = dict(source=str(args.experiment.resolve()), selection_frame=0, spacing_m=args.spacing,
                  grid_shape=shape, template_bounds_min=vertices.min(0).tolist(),
                  template_bounds_max=vertices.max(0).tolist(), reference_span_m=summary['reference_span_m'],
                  definition='Signed distance <= (factor - 1) * reference_span / 2; no geometric scaling', records=records)
    (args.output / 'surfaces.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--spacing', type=float, default=.0125)
    parser.add_argument('--factors', type=float, nargs='+', default=[1.3, 1.5])
    main(parser.parse_args())
