"""Calibrated render and point-cloud comparisons for template volume filtering."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import yaml

from compose_actor01_comparison import render_points, projection_error, error_image
from partmm.open3d_renderer import _open3d, _renderer
from partmm.visualization import save_animation_gif


def compose(panels, titles, subtitle, tile, footer='Same camera / point size. Red: removed centers. Gray wireframe: filtering volume.', heading='Actor01Seq2 | Template volume x1.1'):
    margin, gap, header, label = 18, 12, 68, 34
    rows = (len(panels) + 2) // 3
    page = Image.new('RGB', (margin * 2 + tile * 3 + gap * 2, header + rows * (tile + label + gap) + 30), 'white')
    draw = ImageDraw.Draw(page)
    font = lambda size: ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', size)
    draw.text((margin, 8), heading, font=font(25), fill='#252d32')
    draw.text((margin, 41), subtitle, font=font(16), fill='#59636a')
    for index, (panel, title) in enumerate(zip(panels, titles)):
        row, col = divmod(index, 3)
        x, y = margin + col * (tile + gap), header + row * (tile + label + gap)
        draw.text((x, y + 5), title, font=font(18), fill='#252d32')
        page.paste(panel.resize((tile, tile), Image.Resampling.LANCZOS), (x, y + label))
    draw.text((margin, page.height - 25), footer, font=font(14), fill='#59636a')
    return page


def main(path):
    config = yaml.safe_load(path.read_text())
    output = Path(config['output'])
    destination = output / 'visualization'
    destination.mkdir(exist_ok=True)
    metadata = json.loads((Path(config['data_dir']) / config['case'] / 'train_meta.json').read_text())
    frames = np.load(output / 'frame_ids.npy')
    masks = np.load(output / 'keep_mask.npy')
    offset_mode = config.get('mode') == 'surface_offset'
    volumes = np.load(output / ('aligned_template.npy' if offset_mode else 'volume_vertices.npy'), mmap_mode='r')
    heading = (f"Actor01Seq2 | Surface dilation +{config['offset_m'] * 100:g} cm" if offset_mode
               else f"Actor01Seq2 | Template volume x{config['scale']:g}")
    footer = ('Red: removed centers. Gray: original aligned template, NOT the expanded boundary.' if offset_mode
              else 'Same camera / point size. Red: removed centers. Gray wireframe: filtering volume.')
    if 'selection_frame' in config:
        heading = f"Actor01Seq2 | {config['envelope_factor']:g}x envelope | Fixed frame-{config['selection_frame']} selection"
        footer = f"Red: permanently removed indices selected at frame {config['selection_frame']}. Gray: retained centers for reference."
    faces = np.load(output / 'faces.npy')
    with np.load(config['d3dgs_params']) as archive:
        points, colors = archive['means3D'][frames], archive['rgb_colors'][frames]
    metrics = json.loads((output / 'render_metrics.json').read_text())
    worst = min(metrics['records'], key=lambda r: r['delta_psnr'])['frame']
    selected = {int(frames[0]), int(frames[len(frames) // 2]), int(frames[-1]), worst}
    width, height, camera = metadata['w'], metadata['h'], config['camera']
    o3d = _open3d()
    renderer = _renderer(o3d, width, height)
    animation, errors = [], []
    for index, frame in enumerate(frames):
        intr = np.asarray(metadata['k'][frame][camera], np.float64)
        extr = np.asarray(metadata['w2c'][frame][camera], np.float64)
        keep = masks[index]
        original = render_points(o3d, renderer, points[index], colors[index], intr, extr, width, height)
        errors.append(projection_error(renderer, points[index], intr, extr, width, height))
        retained = render_points(o3d, renderer, points[index][keep], colors[index][keep], intr, extr, width, height)
        if 'selection_frame' in config:
            diagnostic_colors = np.tile([.67, .69, .70], (len(keep), 1))
            diagnostic_colors[~keep] = [.78, .16, .18]
            render_points(o3d, renderer, points[index], diagnostic_colors, intr, extr, width, height)
        else:
            render_points(o3d, renderer, points[index][~keep], np.tile([.78, .16, .18], ((~keep).sum(), 1)), intr, extr, width, height)
            mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(volumes[index]), o3d.utility.Vector3iVector(faces))
            outline = o3d.geometry.LineSet.create_from_triangle_mesh(mesh.simplify_quadric_decimation(900))
            outline.paint_uniform_color([.65, .68, .69])
            material = o3d.visualization.rendering.MaterialRecord()
            material.shader = 'unlitLine'
            material.line_width = 1.
            renderer.scene.add_geometry('volume', outline, material)
        removed = Image.fromarray(np.asarray(renderer.render_to_image())[:, :, :3])
        rendered = [Image.open(output / 'renders' / f'{name}_{frame:04d}.png').convert('RGB') for name in ['gt', 'original', 'filtered']]
        panels = rendered + [original, retained, removed]
        titles = ['Observation', 'Original | Gaussian render', 'Filtered | Gaussian render',
                  f'Original | {len(keep):,} centers', f'Retained | {keep.sum():,} centers', f'Removed | {(~keep).sum():,} centers']
        subtitle = f'Camera {camera} | Frame {frame} | Removed {(~keep).mean():.1%} | No retraining'
        if 'selection_frame' in config:
            subtitle += f" | Outward +{config['offset_m'] * 100:.2f} cm"
        animation.append(np.asarray(compose(panels, titles, subtitle, 360, footer, heading)))
        if frame in selected:
            compose(panels, titles, subtitle, width, footer, heading).save(destination / f'frame_{frame:04d}.png')
            error_panels = [rendered[0], error_image(rendered[0], rendered[1]), error_image(rendered[0], rendered[2])]
            compose(error_panels, ['Observation', 'Original | absolute error x4', 'Filtered | absolute error x4'], subtitle, width,
                    'Error: mean absolute RGB difference x4. Black = zero; white >= 0.25.', heading).save(destination / f'error_{frame:04d}.png')
            for name, selection in [('retained', keep), ('removed', ~keep)]:
                cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[index][selection]))
                cloud.colors = o3d.utility.Vector3dVector(np.clip(colors[index][selection], 0, 1))
                o3d.io.write_point_cloud(str(destination / f'{name}_{frame:04d}.ply'), cloud)
                np.save(destination / f'{name}_indices_{frame:04d}.npy', np.flatnonzero(selection))
        if index % 10 == 0:
            print(f'Visualized {index + 1}/{len(frames)}', flush=True)
    save_animation_gif(destination / 'comparison.gif', animation, fps=15)
    manifest = dict(frames=frames.tolist(), camera=camera, selected=sorted(selected),
                    point_size=2., opacity_filter=False, point_downsampling=False,
                    wireframe_display_triangles=900, selection_mesh_triangles=len(faces),
                    projection_error_pixels=max(errors), metrics=str(output / 'render_metrics.json'),
                    mask=str(output / 'keep_mask.npy'), fps=15, heading=heading,
                    wireframe_represents=None if 'selection_frame' in config else 'original aligned template' if offset_mode else 'scaled volume boundary',
                    selection_frame=config.get('selection_frame'))
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    main(parser.parse_args().config)
