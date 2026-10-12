"""Show a frame-zero-only Gaussian selection and later distant-point diagnostics."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import yaml

from compose_actor01_comparison import render_points, projection_error, error_image
from visualize_template_filter import compose
from partmm.open3d_renderer import _open3d, _renderer
from partmm.visualization import save_animation_gif


def main(args):
    config = yaml.safe_load(args.config.read_text())
    root = Path(config['output'])
    output = root / 'visualization'
    output.mkdir(exist_ok=True)
    summary = json.loads((root / 'filter_summary.json').read_text())
    mask = np.load(root / 'fixed_keep_mask.npy')
    frame_ids = np.load(root / 'frame_ids.npy')
    metadata = json.loads((Path(config['data_dir']) / config['case'] / 'train_meta.json').read_text())
    diagnostic_root = Path(config['preview_template_source'])
    np.testing.assert_array_equal(np.load(diagnostic_root / 'frame_ids.npy'), frame_ids)
    distances = np.load(diagnostic_root / 'signed_distance_m.npy', mmap_mode='r')
    with np.load(config['d3dgs_params']) as archive:
        xyz, rgb = archive['means3D'], archive['rgb_colors']
    # This experiment retained all points; cached original renders are exactly equivalent.
    assert mask.all(), 'Nontrivial fixed selections require fresh filtered Gaussian renders'
    camera = config['camera']
    width, height = metadata['w'], metadata['h']
    intr = np.asarray(metadata['k'][0][camera], np.float64)
    extr = np.asarray(metadata['w2c'][0][camera], np.float64)
    o3d = _open3d()
    renderer = _renderer(o3d, width, height)
    first = render_points(o3d, renderer, xyz[0], rgb[0], intr, extr, width, height)
    first.save(output / 'first_frame_points.png')
    heading = f"Actor01Seq2 | {config['envelope_factor']:g}x envelope | Selection fixed at frame 0"
    footer = 'Red: outside the current posed envelope, diagnostic only. Fixed mask: 119,506 kept; 0 deleted.'
    animation, records, calibration = [], [], []
    for i, frame in enumerate(frame_ids):
        intr = np.asarray(metadata['k'][frame][camera], np.float64)
        extr = np.asarray(metadata['w2c'][frame][camera], np.float64)
        current = render_points(o3d, renderer, xyz[frame][mask], rgb[frame][mask], intr, extr, width, height)
        calibration.append(projection_error(renderer, xyz[frame], intr, extr, width, height))
        outside = distances[i] > summary['offset_m']
        diagnostic_colors = np.tile([.67, .69, .70], (len(mask), 1))
        diagnostic_colors[outside] = [.80, .10, .13]
        diagnostic = render_points(o3d, renderer, xyz[frame], diagnostic_colors, intr, extr, width, height)
        observed = Image.open(args.render_cache / f'gt_{frame:04d}.png').convert('RGB')
        predicted = Image.open(args.render_cache / f'original_{frame:04d}.png').convert('RGB')
        panels = [observed, predicted, predicted, first, current, diagnostic]
        titles = ['Observation', 'Original | Gaussian render', 'Fixed subset | Gaussian render',
                  'Frame 0 | all centers', f'Frame {frame} | retained centers', f'Outside now | {outside.sum():,} centers']
        subtitle = f"Camera {camera} | Frame {frame} | Outward margin {summary['offset_m'] * 100:.1f} cm | No later pruning"
        animation.append(np.asarray(compose(panels, titles, subtitle, 360, footer, heading)))
        if frame in [240, 254, 270, 299]:
            compose(panels, titles, subtitle, width, footer, heading).save(output / f'frame_{frame:04d}.png')
            compose([observed, error_image(observed, predicted), error_image(predicted, predicted)],
                    ['Observation', 'Reconstruction error x4', 'Before / after difference = 0'], subtitle, width,
                    'Errors are absolute RGB differences. No Gaussian was removed by the frame-zero selection.', heading
                    ).save(output / f'error_{frame:04d}.png')
        records.append(dict(frame=int(frame), diagnostic_outside_count=int(outside.sum()), removed=0))
        if i % 10 == 0:
            print(f'Composed {i + 1}/{len(frame_ids)}', flush=True)
    save_animation_gif(output / 'comparison.gif', animation, fps=15)
    manifest = dict(config=str(args.config.resolve()), checkpoint=summary['checkpoint'],
                    camera=camera, selection_frame=0, frames=frame_ids.tolist(),
                    point_downsampling=False, opacity_filter=False, point_size=2.,
                    diagnostic='Red points are outside the current posed template plus the same fixed outward margin; NOT deleted',
                    render_cache=str(args.render_cache.resolve()), render_reuse_reason='All checkpoint parameters are identical to original',
                    projection_error_pixels=max(calibration), fps=15, records=records)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('--render-cache', type=Path, required=True)
    main(parser.parse_args())
