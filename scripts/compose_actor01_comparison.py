"""Combine observed images, Gaussian renders, and calibrated Open3D point views."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from partmm.open3d_renderer import _open3d, _renderer
from partmm.visualization import save_animation_gif


def error_image(observed, predicted):
    error = np.abs(np.asarray(observed, dtype=np.float32) - np.asarray(predicted, dtype=np.float32)).mean(-1)
    return Image.fromarray(np.rint(np.clip(error * 4, 0, 255)).astype(np.uint8)).convert('RGB')


def compose(images, frame, tile):
    margin, gap, header, label, footer = 18, 12, 64, 36, 69
    page = Image.new('RGB', (2 * margin + 3 * tile + 2 * gap, header + 2 * (label + tile + gap) + footer), 'white')
    draw = ImageDraw.Draw(page)
    font = lambda size: ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', size)
    draw.text((margin, 8), f'Actor01Seq2 | Camera 126 | Frame {frame}', font=font(25), fill='#252d32')
    draw.text((margin, 39), 'AG: official template, 10k training | Synchronized capture camera', font=font(16), fill='#59636a')
    titles = ['Observation', 'D3DGS | Gaussian render', 'AG (10k) | Gaussian render',
              'AG absolute RGB error (x4)', 'D3DGS | 119,506 centers', 'AG | 337,500 centers']
    boxes = []
    for i, (image, title) in enumerate(zip(images, titles)):
        row, col = divmod(i, 3)
        x, y = margin + col * (tile + gap), header + row * (label + tile + gap)
        draw.text((x, y + 5), title, font=font(19), fill='#252d32')
        page.paste(image.resize((tile, tile), Image.Resampling.LANCZOS), (x, y + label))
        boxes.append((x, y + label, tile, tile))
    y = header + 2 * (label + tile + gap)
    for text in ['Points: RGB, fixed size, no opacity filtering; off-camera points are outside this view.',
                 'Error: black = 0, white >= 0.25 MAE. D3DGS and AG use different training protocols.',
                 'Point display is not Gaussian splatting. Preview playback: 15 fps.']:
        draw.text((margin, y + 4), text, font=font(14), fill='#59636a')
        y += 21
    return page, boxes


def render_points(o3d, renderer, xyz, rgb, intr, extr, width, height):
    renderer.scene.clear_geometry()
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(xyz, dtype=np.float64)))
    cloud.colors = o3d.utility.Vector3dVector(np.clip(rgb, 0, 1))
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = 'defaultUnlit'
    material.point_size = 2.0
    renderer.scene.add_geometry('centers', cloud, material)
    renderer.setup_camera(intr, extr, width, height)
    renderer.scene.camera.set_projection(intr, 1.0, 100.0, float(width), float(height))
    return Image.fromarray(np.asarray(renderer.render_to_image())[:, :, :3])


def projection_error(renderer, xyz, intr, extr, width, height):
    world = np.column_stack([xyz[:1000], np.ones(min(len(xyz), 1000))])
    camera = world @ extr.T
    uv = camera[:, :3] @ intr.T
    uv = uv[:, :2] / uv[:, 2:3]
    clip = world @ np.asarray(renderer.scene.camera.get_view_matrix()).T @ np.asarray(renderer.scene.camera.get_projection_matrix()).T
    ndc = clip[:, :2] / clip[:, 3:4]
    pixels = np.column_stack([(ndc[:, 0] + 1) * width / 2, (1 - ndc[:, 1]) * height / 2])
    error = float(np.max(np.abs(pixels - uv)))
    if error > 1.0:
        raise ValueError(f'Open3D camera mismatch: {error:.4f} pixels')
    return error


def main(args):
    output = args.root / 'combined'
    output.mkdir(exist_ok=True)
    metadata = json.loads(args.metadata.read_text())
    intr = np.asarray(metadata['k'][0][126], np.float64)
    extr = np.asarray(metadata['w2c'][0][126], np.float64)
    width, height = metadata['w'], metadata['h']
    metrics = json.loads((args.root / 'ag_10000/metrics.json').read_text())
    worst = min(metrics['records'], key=lambda r: r['full_psnr_db'])['frame']
    selected = {240, 269, 299, worst}
    with np.load(args.d3dgs) as archive:
        reference = archive['means3D'][args.frames]
        colors = archive['rgb_colors'][args.frames]
    o3d = _open3d()
    renderer = _renderer(o3d, width, height)
    frames, calibration_error = [], None
    for i, frame in enumerate(args.frames):
        ag_root = args.root / 'ag_10000' / f'frame_{frame:04d}'
        observation = Image.open(ag_root / 'gt.png').convert('RGB')
        ag_render = Image.open(ag_root / 'prediction.png').convert('RGB')
        d3_render = Image.open(output / 'd3dgs' / f'frame_{frame:04d}.png').convert('RGB')
        with np.load(args.root / 'pointclouds/data' / f'frame_{frame:04d}.npz') as data:
            ag_points = render_points(o3d, renderer, data['xyz'], data['rgb'], intr, extr, width, height)
            if calibration_error is None:
                calibration_error = projection_error(renderer, data['xyz'], intr, extr, width, height)
        d3_points = render_points(o3d, renderer, reference[i], colors[i], intr, extr, width, height)
        panels = [observation, d3_render, ag_render, error_image(observation, ag_render), d3_points, ag_points]
        frames.append(np.asarray(compose(panels, frame, 360)[0]))
        if frame in selected:
            compose(panels, frame, width)[0].save(output / f'frame_{frame:04d}.png')
            ag_points.save(output / f'ag_points_{frame:04d}.png')
            d3_points.save(output / f'd3dgs_points_{frame:04d}.png')
        if i % 10 == 0:
            print(f'Composed {i + 1}/{len(args.frames)} frames', flush=True)
    save_animation_gif(output / 'comparison.gif', frames, fps=15)
    manifest = dict(frames=args.frames, camera=126, intrinsics=intr.tolist(), world_to_camera=extr.tolist(),
                    raw_panel_size=[width, height], gif_panel_size=[360, 360],
                    point_size=2.0, opacity_filter=False, downsampling=False,
                    near=1.0, far=100.0, projection_error_pixels=calibration_error,
                    selected_stills=sorted(selected.intersection(args.frames)),
                    ag_checkpoint=metrics['checkpoint'],
                    d3dgs_checkpoint=str(args.d3dgs),
                    scope='Same frame and capture camera; different training protocols')
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT / 'outputs/animatable_gaussians_actor01_seq2')
    parser.add_argument('--metadata', type=Path, default=Path('/home/ubuntu/research/data/DNA-Rendering/actorshq_actor01_seq2/train_meta.json'))
    parser.add_argument('--d3dgs', type=Path, default=Path('/mnt/local_nas/wjx/GaussiAnimate_release_v1/resources/multiview/d3dgs/actorshq/actorshq_actor01_seq2/params.npz'))
    parser.add_argument('--frames', type=int, nargs='+', default=list(range(240, 300)))
    main(parser.parse_args())
