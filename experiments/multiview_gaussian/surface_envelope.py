"""Filter Gaussian centers by a true outward offset of an aligned posed template."""

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import yaml

from template_volume import dilated_contains


def main(path):
    config = yaml.safe_load(path.read_text())
    output, source = Path(config['output']), Path(config['alignment_source'])
    output.mkdir(parents=True, exist_ok=True)
    ids = np.load(source / 'frame_ids.npy')
    assert np.array_equal(ids, np.arange(*config['frames']))
    vertices = np.load(source / 'posed_template.npy', mmap_mode='r')
    faces = np.load(source / 'faces.npy')
    transforms = np.load(source / 'alignment.npy')
    prior = json.loads((source / 'filter_summary.json').read_text())
    for name in ['frame_ids.npy', 'faces.npy', 'alignment.npy']:
        shutil.copyfile(source / name, output / name)
    with np.load(config['d3dgs_params']) as archive:
        points = archive['means3D'][ids]
    shape = points.shape[:2]
    masks = np.lib.format.open_memmap(output / 'keep_mask.npy', mode='w+', dtype=bool, shape=shape)
    distances = np.lib.format.open_memmap(output / 'signed_distance_m.npy', mode='w+', dtype=np.float32, shape=shape)
    aligned_meshes = np.lib.format.open_memmap(output / 'aligned_template.npy', mode='w+', dtype=np.float32, shape=vertices.shape)
    records = []
    for index, frame in enumerate(ids):
        transform = transforms[index]
        aligned = np.asarray(vertices[index], dtype=np.float64) @ transform[:3, :3].T + transform[:3, 3]
        keep, signed_distance = dilated_contains(aligned, faces, points[index], config['offset_m'], config['winding_threshold'])
        masks[index], distances[index], aligned_meshes[index] = keep, signed_distance, aligned
        record = dict(frame=int(frame), total=len(keep), kept=int(keep.sum()), removed=int((~keep).sum()),
                      removed_fraction=float((~keep).mean()), inside_count=int((signed_distance <= 0).sum()),
                      shell_count=int(((signed_distance > 0) & keep).sum()),
                      registration=prior['records'][index]['registration'])
        records.append(record)
        print(json.dumps(record), flush=True)
    for data in [masks, distances, aligned_meshes]:
        data.flush()
    summary = dict(config=str(path.resolve()), source=config['d3dgs_params'], frames=ids.tolist(),
                   mode='surface_offset', offset_m=config['offset_m'], alignment_source=str(source),
                   volume_definition='Original generalized-winding interior UNION points within offset_m of the triangle surface',
                   selection='Gaussian centers; exact triangle distance; no vertex scaling or normal displacement',
                   temporal_policy='Per-frame masks with original Gaussian indices preserved',
                   display_mesh='Aligned original template, NOT the offset boundary',
                   mean_removed_fraction=float(np.mean([r['removed_fraction'] for r in records])), records=records)
    (output / 'filter_summary.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    main(parser.parse_args().config)
