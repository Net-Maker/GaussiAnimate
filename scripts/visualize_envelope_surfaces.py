"""Show the actual 1.3x and 1.5x outward envelopes at a shared physical scale."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import yaml

from visualize_actor01_gaussians import view_basis
from visualize_partmm_errors import setup_camera
from partmm.open3d_renderer import _open3d, _renderer
from partmm.visualization import save_animation_gif


def compose(images, records, reference, angle):
    width, height = images[0].size
    page = Image.new('RGB', (len(images) * (width + 8) + 24, height + 156), 'white')
    draw = ImageDraw.Draw(page)
    font = lambda size: ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', size)
    draw.text((16, 10), 'Actor01Seq2 | Frame 0 | Outward envelopes', font=font(24), fill='#263238')
    draw.text((16, 42), f'Same camera and scale | Reference span {reference:.2f} m | View {angle:+.0f} deg', font=font(16), fill='#59636a')
    labels = [('Original centers', f'Reference span {reference:.2f} m')]
    labels += [(f"{r['factor']:.1f}x envelope", f"Outward +{100 * r['offset_m']:.2f} cm | Span {reference * r['factor']:.2f} m") for r in records]
    for i, (im, (title, detail)) in enumerate(zip(images, labels)):
        x = 16 + i * (width + 8)
        draw.text((x, 72), title, font=font(21), fill='#263238')
        draw.text((x, 100), detail, font=font(14), fill='#59636a')
        page.paste(im, (x, 123))
    draw.text((16, height + 132), 'Shell: outward envelope. Red centers: outside that envelope in frame 0. No body scaling.', font=font(14), fill='#59636a')
    return page


def main(args):
    config = yaml.safe_load((args.experiment / 'resolved_config.yaml').read_text())
    surfaces = json.loads((args.surfaces / 'surfaces.json').read_text())
    meta = json.loads((Path(config['data_dir']) / config['case'] / 'train_meta.json').read_text())
    with np.load(config['d3dgs_params']) as archive:
        points, colors = archive['means3D'][0], archive['rgb_colors'][0]
    o3d = _open3d()
    renderer = _renderer(o3d, 620, 850)
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points.astype(np.float64)))
    cloud.colors = o3d.utility.Vector3dVector(np.clip(colors, 0, 1))
    distances = np.load(args.experiment / 'selection_signed_distance_m.npy')
    clouds = [cloud]
    for record in surfaces['records']:
        diagnostic = o3d.geometry.PointCloud(cloud)
        diagnostic_colors = np.clip(colors, 0, 1).copy()
        diagnostic_colors[distances > record['offset_m']] = [.9, .02, .03]
        diagnostic.colors = o3d.utility.Vector3dVector(diagnostic_colors)
        clouds.append(diagnostic)
    point_material = o3d.visualization.rendering.MaterialRecord()
    point_material.shader = 'defaultUnlit'
    point_material.point_size = 2.2
    meshes = [o3d.io.read_triangle_mesh(r['mesh']) for r in surfaces['records']]
    for mesh in meshes:
        mesh.compute_vertex_normals()
    materials = []
    for color in [[.15, .55, .80, .22], [.10, .65, .47, .22]]:
        material = o3d.visualization.rendering.MaterialRecord()
        material.shader = 'defaultLitTransparency'
        material.base_color = color
        material.base_roughness = .65
        material.base_reflectance = .1
        materials.append(material)
    bounds = np.asarray(meshes[-1].vertices)
    animation, cameras = [], []
    for angle in np.linspace(-25, 65, 36):
        camera = setup_camera(o3d, renderer, [[bounds]], view_basis(meta['w2c'][0][config['camera']], angle), 620, 850)
        cameras.append(camera)
        images = []
        for i in range(len(meshes) + 1):
            renderer.scene.clear_geometry()
            renderer.scene.add_geometry('points', clouds[i], point_material)
            if i:
                renderer.scene.add_geometry('envelope', meshes[i - 1], materials[i - 1])
            images.append(Image.fromarray(np.asarray(renderer.render_to_image())[:, :, :3]))
        page = compose(images, surfaces['records'], surfaces['reference_span_m'], angle)
        animation.append(np.asarray(page.resize((1140, round(page.height * 1140 / page.width)), Image.Resampling.LANCZOS)))
        if len(animation) == 11:
            page.save(args.surfaces / 'comparison.png')
        if len(animation) == 30:
            page.save(args.surfaces / 'oblique.png')
        if len(animation) % 12 == 0:
            print(f'Rendered {len(animation)}/36 views', flush=True)
    save_animation_gif(args.surfaces / 'envelopes.gif', animation, fps=12)
    record = dict(selection_frame=0, point_count=len(points), factors=[r['factor'] for r in surfaces['records']],
                  projection='orthographic', same_scale_per_view=True, cameras=cameras,
                  shell_alpha=.22, source=str(args.experiment.resolve()),
                  shells=str((args.surfaces / 'surfaces.json').resolve()), fps=12)
    (args.surfaces / 'visualization.json').write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment', type=Path)
    parser.add_argument('--surfaces', type=Path, required=True)
    main(parser.parse_args())
