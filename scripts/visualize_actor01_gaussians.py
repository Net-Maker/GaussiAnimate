"""Open3D inspection of unfiltered D3DGS and Animatable Gaussians centers."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from partmm.open3d_renderer import _open3d, _renderer
from visualize_partmm_errors import setup_camera, cloud_image
from partmm.visualization import save_animation_gif


def view_basis(extrinsic, azimuth):
    rotation = np.asarray(extrinsic, dtype=np.float64)[:3, :3]
    right, up, front = rotation[0], -rotation[1], -rotation[2]
    angle = np.deg2rad(azimuth)
    facing = front * np.cos(angle) + right * np.sin(angle)
    return np.stack([np.cross(up, facing), up, facing], axis=1)


def load_cloud(path):
    with np.load(path) as data:
        return data['xyz'].copy(), np.clip(data['rgb'], 0, 1)


def compose(images, frame, subtitle):
    h, w = images[0].shape[:2]
    page = Image.new('RGB', (2 * w + 40, h + 123), 'white')
    draw = ImageDraw.Draw(page)
    font = lambda size: ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', size)
    draw.text((20, 12), f'Actor01Seq2 | Frame {frame} | Gaussian centers', font=font(24), fill='#252d32')
    for i, (array, title) in enumerate(zip(images, ['D3DGS | 119,506 points', 'Animatable Gaussians | 337,500 points'])):
        x = 20 + i * w
        draw.text((x, 50), title, font=font(20), fill='#252d32')
        page.paste(Image.fromarray(array), (x, 83))
    draw.text((20, h + 95), subtitle, font=font(15), fill='#59636a')
    return np.asarray(page)


def main(args):
    files = sorted(args.points.glob('frame_*.npz'))
    ids = [int(f.stem.split('_')[1]) for f in files]
    if not ids:
        raise FileNotFoundError(f'No exported frames in {args.points}')
    ag = [load_cloud(f) for f in files]
    with np.load(args.d3dgs) as archive:
        reference = archive['means3D'][ids]
        reference_rgb = np.clip(archive['rgb_colors'][ids], 0, 1)
    metadata = json.loads(args.metadata.read_text())
    extrinsic = metadata['w2c'][0][126]
    sequences = [reference, [xyz for xyz, _ in ag]]
    o3d = _open3d()
    args.output.mkdir(parents=True, exist_ok=True)
    selected = [frame for frame in [240, 269, 299] if frame in ids]
    for frame in selected:
        index = ids.index(frame)
        for name, xyz, rgb in [('d3dgs', reference[index], reference_rgb[index]),
                               ('ag', *ag[index])]:
            cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz.astype(np.float64)))
            cloud.colors = o3d.utility.Vector3dVector(rgb)
            if not o3d.io.write_point_cloud(str(args.output / f'{name}_{frame:04d}.ply'), cloud):
                raise OSError('Could not write point cloud')
    if args.interactive:
        index = ids.index(args.frame)
        basis = view_basis(extrinsic, 15)
        spacing = np.ptp(np.concatenate([reference[index], ag[index][0]]) @ basis, axis=0)[0] * 1.2
        clouds = []
        for column, (xyz, rgb) in enumerate([(reference[index], reference_rgb[index]), ag[index]]):
            cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz + column * spacing * basis[:, 0]))
            cloud.colors = o3d.utility.Vector3dVector(rgb)
            clouds.append(cloud)
        o3d.visualization.draw_geometries(clouds, window_name='D3DGS (left) | AG (right): all centers',
                                          width=1400, height=900, front=basis[:, 2], up=basis[:, 1],
                                          lookat=np.concatenate([np.asarray(c.points) for c in clouds]).mean(0), zoom=0.65)
        return
    renderer = _renderer(o3d, args.width, args.height)
    cameras = {}
    for view, angle, body in [('overview', 15, False), ('body', 15, True), ('side', 90, False)]:
        fit_sequences = [sequences[1]] if body else sequences
        cameras[view] = setup_camera(o3d, renderer, fit_sequences, view_basis(extrinsic, angle), args.width, args.height)
        animation = []
        for index, frame in enumerate(ids):
            if view == 'side' and frame not in selected:
                continue
            pairs = [(reference[index], reference_rgb[index]), ag[index]]
            images = [cloud_image(o3d, renderer, xyz, rgb) for xyz, rgb in pairs]
            subtitle = 'All centers | Same camera, scale and point size | RGB | Preview: 15 fps'
            if body:
                subtitle = 'Body detail: outer points may be outside view | No point filtering | RGB'
            page = compose(images, frame, subtitle)
            if view != 'side':
                animation.append(page)
            if frame in selected:
                Image.fromarray(page).save(args.output / f'{view}_{frame:04d}.png')
                neutral = [cloud_image(o3d, renderer, xyz, np.tile([.12, .39, .43], (len(xyz), 1))) for xyz, _ in pairs]
                Image.fromarray(compose(neutral, frame, subtitle.replace('RGB', 'Uniform geometry color'))).save(
                    args.output / f'{view}_geometry_{frame:04d}.png')
        if animation:
            save_animation_gif(args.output / f'{view}.gif', animation, fps=15)
    mesh = o3d.io.read_triangle_mesh(str(args.template))
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([.66, .75, .77])
    renderer.scene.clear_geometry()
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = 'defaultLit'
    renderer.scene.add_geometry('official_template', mesh, material)
    # Canonical SMPL-X coordinates are y-up, facing positive z.
    basis = np.eye(3)
    template_camera = setup_camera(o3d, renderer, [[np.asarray(mesh.vertices)]], basis, args.width, args.height)
    Image.fromarray(np.asarray(renderer.render_to_image())).save(args.output / 'official_template.png')
    manifest = dict(frames=ids, filtered_points=0, downsampling=False, point_size=2.0,
                    gaussian_opacity_used_for_visibility=False, color='model RGB; AG BGR converted to RGB',
                    cameras=cameras, template_camera=template_camera, official_template=str(args.template.resolve()),
                    ag_source=str(args.points.resolve()), d3dgs_source=str(args.d3dgs),
                    point_counts=dict(d3dgs=int(reference.shape[1]), ag=int(len(ag[0][0]))),
                    note='Overview and side fit all centers. Body detail clips the view, not the exported data.')
    (args.output / 'visualization.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(args.output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--points', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--d3dgs', type=Path, default=Path('/mnt/local_nas/wjx/GaussiAnimate_release_v1/resources/multiview/d3dgs/actorshq/actorshq_actor01_seq2/params.npz'))
    parser.add_argument('--metadata', type=Path, default=Path('/home/ubuntu/research/data/DNA-Rendering/actorshq_actor01_seq2/train_meta.json'))
    parser.add_argument('--template', type=Path, default=ROOT / '.research/AnimatableGaussians/data/actor01_seq2/template.ply')
    parser.add_argument('--width', type=int, default=500)
    parser.add_argument('--height', type=int, default=680)
    parser.add_argument('--interactive', action='store_true')
    parser.add_argument('--frame', type=int, default=240)
    main(parser.parse_args())
