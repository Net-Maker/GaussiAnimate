"""Select Gaussians once in sequence frame zero and preserve that subset over time."""

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import open3d as o3d
import yaml

from template_volume import align_template, dilated_contains, subset_gaussians


def main(path):
    config = yaml.safe_load(path.read_text())
    output = Path(config['output'])
    preparation = output / 'selection_template'
    frame = config['selection_frame']
    assert np.load(preparation / 'frame_ids.npy').tolist() == [frame]
    vertices = np.load(preparation / 'posed_template.npy')[0].astype(np.float64)
    faces = np.load(preparation / 'faces.npy')
    with np.load(config['d3dgs_params']) as archive:
        params = {key: archive[key] for key in archive.files}
    aligned, transform, registration = align_template(
        vertices, faces, params['means3D'][frame], config['registration'], config['seed'])
    reference = float(np.ptp(aligned, axis=0).max())
    offset = (config['envelope_factor'] - 1) * reference / 2
    keep, distance = dilated_contains(aligned, faces, params['means3D'][frame], offset,
                                     config['winding_threshold'])
    np.save(output / 'fixed_keep_mask.npy', keep)
    np.save(output / 'keep_indices.npy', np.flatnonzero(keep))
    np.save(output / 'removed_indices.npy', np.flatnonzero(~keep))
    np.save(output / 'selection_signed_distance_m.npy', distance)
    np.save(output / 'selection_alignment.npy', transform)
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(aligned), o3d.utility.Vector3iVector(faces))
    o3d.io.write_triangle_mesh(str(output / 'selection_template_aligned.ply'), mesh)
    frames = np.arange(*config['frames'])
    np.save(output / 'frame_ids.npy', frames)
    np.save(output / 'keep_mask.npy', np.broadcast_to(keep, (len(frames), len(keep))))
    # Later posed meshes are visual references only; no later geometry affects the mask.
    references = Path(config['preview_template_source'])
    assert np.array_equal(np.load(references / 'frame_ids.npy'), frames)
    shutil.copyfile(references / 'faces.npy', output / 'faces.npy')
    shutil.copyfile(references / 'aligned_template.npy', output / 'aligned_template.npy')
    resolved = dict(config, reference_span_m=reference, offset_m=offset)
    (output / 'resolved_config.yaml').write_text(yaml.safe_dump(resolved, sort_keys=False))
    filtered = subset_gaussians(params, keep)
    checkpoint = output / 'checkpoint' / 'params.npz'
    checkpoint.parent.mkdir(exist_ok=True)
    np.savez(checkpoint, **filtered)
    summary = dict(config=str(path.resolve()), source=config['d3dgs_params'], checkpoint=str(checkpoint),
                   selection_frame=frame, applied_frames=int(params['means3D'].shape[0]), preview_frames=frames.tolist(),
                   envelope_factor=config['envelope_factor'], reference_span_m=reference, offset_m=offset,
                   reference_definition='Maximum axis span of aligned template at selection frame only',
                   volume_definition='Original interior UNION Euclidean surface-distance shell',
                   temporal_policy='One immutable Gaussian index subset selected in frame 0; no later reclassification',
                   total=len(keep), kept=int(keep.sum()), removed=int((~keep).sum()),
                   removed_fraction=float((~keep).mean()), registration=registration)
    (output / 'filter_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    main(parser.parse_args().config)
