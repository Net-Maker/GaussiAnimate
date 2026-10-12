"""Locate deleted points and trace a retained distant cluster back to frame zero."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import yaml

from compose_actor01_comparison import render_points, projection_error
from partmm.open3d_renderer import _open3d, _renderer


def project(points, intr, extr):
    camera = points @ extr[:3, :3].T + extr[:3, 3]
    pixels = camera @ intr.T
    return pixels[:, :2] / pixels[:, 2:3], camera[:, 2]


def main(args):
    config = yaml.safe_load((args.experiment / 'resolved_config.yaml').read_text())
    output = args.experiment / 'visualization/point_tracks'
    output.mkdir(exist_ok=True)
    meta = json.loads((Path(config['data_dir']) / config['case'] / 'train_meta.json').read_text())
    keep = np.load(args.experiment / 'fixed_keep_mask.npy')
    d0 = np.load(args.experiment / 'selection_signed_distance_m.npy')
    diagnostic_root = Path(config['preview_template_source'])
    diagnostic_frames = np.load(diagnostic_root / 'frame_ids.npy')
    d_later = np.load(diagnostic_root / 'signed_distance_m.npy', mmap_mode='r')[np.flatnonzero(diagnostic_frames == args.frame)[0]]
    with np.load(config['d3dgs_params']) as archive:
        points = archive['means3D'][[0, args.frame]]
    width, height, cam = meta['w'], meta['h'], config['camera']
    intr = [np.asarray(meta['k'][frame][cam], np.float64) for frame in [0, args.frame]]
    extr = [np.asarray(meta['w2c'][frame][cam], np.float64) for frame in [0, args.frame]]
    pixels, depth = project(points[1], intr[1], extr[1])
    # Select the conspicuous left-hand cluster in the existing capture-camera preview.
    x0, y0, x1, y1 = args.roi
    cluster = ((d_later > args.far_distance) & (depth > 1) & (depth < 100) &
               (pixels[:, 0] >= x0) & (pixels[:, 0] < x1) &
               (pixels[:, 1] >= y0) & (pixels[:, 1] < y1))
    assert cluster.any() and keep[cluster].all()
    o3d = _open3d()
    renderer = _renderer(o3d, width, height)
    bases, errors = [], []
    for i in range(2):
        bases.append(render_points(o3d, renderer, points[i], np.tile([.76, .78, .79], (len(keep), 1)), intr[i], extr[i], width, height))
        errors.append(projection_error(renderer, points[i], intr[i], extr[i], width, height))
    selections, marker_colors = [~keep, cluster], ['#d02639', '#087f9c']
    panels = []
    for selected, color in zip(selections, marker_colors):
        for i in range(2):
            panel = bases[i].copy()
            draw = ImageDraw.Draw(panel)
            uv, z = project(points[i][selected], intr[i], extr[i])
            for (u, v), distance in zip(uv, z):
                if 1 < distance < 100 and 0 <= u < width and 0 <= v < height:
                    draw.ellipse((u - 2.5, v - 2.5, u + 2.5, v + 2.5), fill=color)
            panels.append(panel)
    margin, gap, label, header = 20, 16, 58, 80
    page = Image.new('RGB', (2 * width + 2 * margin + gap, header + 2 * (height + label + gap) + 38), 'white')
    draw = ImageDraw.Draw(page)
    font = lambda size: ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', size)
    draw.text((margin, 12), 'Actor01Seq2 | Track the same Gaussian IDs', font=font(27), fill='#263238')
    draw.text((margin, 47), f'Red: {(~keep).sum():,} deleted IDs. Teal: {cluster.sum():,} retained IDs from the later distant cluster.', font=font(18), fill='#59636a')
    titles = [('Deleted points | Frame 0', f'Distance outside template: {d0[~keep].min()*100:.2f}-{d0[~keep].max()*100:.2f} cm'),
              (f'Same deleted IDs | Frame {args.frame}', 'Shown at their original, unfiltered locations'),
              ('Retained cluster | Frame 0', f'Originally inside or at most {d0[cluster].max()*100:.2f} cm outside the template'),
              (f'Same retained IDs | Frame {args.frame}', f'Now {d_later[cluster].min()*100:.1f}-{d_later[cluster].max()*100:.1f} cm outside the posed template')]
    for i, (panel, (title, detail)) in enumerate(zip(panels, titles)):
        row, col = divmod(i, 2)
        x, y = margin + col * (width + gap), header + row * (height + label + gap)
        color = marker_colors[row]
        draw.text((x, y + 2), title, font=font(23), fill=color)
        draw.text((x, y + 32), detail, font=font(16), fill='#59636a')
        page.paste(panel, (x, y + label))
    draw.text((margin, page.height - 30), 'Colored markers ignore depth occlusion to reveal hidden points. No new filtering is applied.', font=font(17), fill='#59636a')
    page.save(output / 'point_tracks.png')
    # A crop exposes the small deleted patch without hiding the full-frame context above.
    uv, _ = project(points[0][~keep], intr[0], extr[0])
    low = np.maximum(np.floor(uv.min(0)).astype(int) - 45, 0)
    high = np.minimum(np.ceil(uv.max(0)).astype(int) + 45, [width, height])
    panels[0].crop((*low, *high)).resize((600, 600), Image.Resampling.NEAREST).save(output / 'deleted_closeup.png')
    np.save(output / 'retained_cluster_ids.npy', np.flatnonzero(cluster))
    np.save(output / 'deleted_ids.npy', np.flatnonzero(~keep))
    groups = {}
    for name, selected in [('deleted', ~keep), ('retained_cluster', cluster)]:
        groups[name] = dict(count=int(selected.sum()), first_frame_signed_distance_range_m=[float(d0[selected].min()), float(d0[selected].max())],
                            later_signed_distance_range_m=[float(d_later[selected].min()), float(d_later[selected].max())],
                            xyz0_min=points[0][selected].min(0).tolist(), xyz0_max=points[0][selected].max(0).tolist())
    record = dict(config=str((args.experiment / 'resolved_config.yaml').resolve()), camera=cam, frames=[0, args.frame],
                  diagnostic_roi_pixels=args.roi, diagnostic_far_distance_m=args.far_distance,
                  projection_error_pixels=max(errors), markers='x-ray projected centers; radius 2.5 pixels', groups=groups)
    (output / 'analysis.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment', type=Path)
    parser.add_argument('--frame', type=int, default=240)
    parser.add_argument('--roi', type=int, nargs=4, default=[0, 400, 180, 600])
    parser.add_argument('--far-distance', type=float, default=.43752381932085654)
    main(parser.parse_args())
