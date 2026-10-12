"""Open3D point-cloud inspection of the high-error PartMM cases."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from partmm.kinematics import forward_kinematics
from partmm.types import PartMMSkeleton, PartMMQuery
from partmm.open3d_renderer import _open3d, _renderer
from partmm.visualization import save_animation_gif


def error_colors(error_mm, maximum):
    anchors = np.array([[.78, .85, .88], [.94, .64, .34], [.65, .10, .16]])
    values = np.clip(error_mm / maximum, 0, 1)
    return np.stack([np.interp(values, [0, .5, 1], anchors[:, c]) for c in range(3)], axis=-1)


def camera_basis(joints, side=False):
    up = joints[:, 15].mean(axis=0) - joints[:, 0].mean(axis=0)
    up /= np.linalg.norm(up)
    right = (joints[:, 17] - joints[:, 16]).mean(axis=0)
    right -= up * np.dot(right, up)
    right /= np.linalg.norm(right)
    front = np.cross(right, up)
    # A small oblique angle separates overlapping limb point clouds.
    angle = np.deg2rad(100 if side else 15)
    facing = front * np.cos(angle) + right * np.sin(angle)
    horizontal = np.cross(up, facing)
    return np.stack([horizontal, up, facing], axis=1)


def setup_camera(o3d, renderer, arrays, basis, width, height):
    low, high = np.full(3, np.inf), np.full(3, -np.inf)
    for sequence in arrays:
        for points in sequence:
            projected = points @ basis
            low = np.minimum(low, projected.min(axis=0))
            high = np.maximum(high, projected.max(axis=0))
    center = ((low + high) / 2) @ basis.T
    span = high - low
    half_height = max(span[1], span[0] * height / width) * .55
    half_width = half_height * width / height
    distance = max(span.max(), 1.) * 3
    renderer.scene.camera.look_at(center, center + basis[:, 2] * distance, basis[:, 1])
    renderer.scene.camera.set_projection(
        o3d.visualization.rendering.Camera.Projection.Ortho,
        -half_width, half_width, -half_height, half_height, .01, distance * 3,
    )
    return dict(center=center.tolist(), basis=basis.tolist(), half_height=float(half_height),
                bounds_low=low.tolist(), bounds_high=high.tolist(), projection='orthographic')


def cloud_image(o3d, renderer, points, colors):
    renderer.scene.clear_geometry()
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64)))
    cloud.colors = o3d.utility.Vector3dVector(np.clip(colors, 0, 1))
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = 'defaultUnlit'
    material.point_size = 2.0
    renderer.scene.add_geometry('points', cloud, material)
    return np.asarray(renderer.render_to_image())[:, :, :3].copy()


def font(size):
    return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', size)


def compose(images, case, frame_id, errors, maximum, view, compare_methods=False):
    h, w = images[0].shape[:2]
    canvas = Image.new('RGB', (4 * w + 40, h + 152), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 12), f'{case}  |  frame {frame_id}  |  {view}', font=font(23), fill='#252d32')
    titles = ['D3DGS reference', 'PartMM + fallback', 'Robust Skin Transfer', 'PartMM point error']
    subtitles = ['Reference-frame RGB', f'RMSE {errors[0]:.1f} mm', f'RMSE {errors[1]:.1f} mm', 'Error on predicted points']
    if compare_methods:
        titles[3] = 'BoB (SMPL-conditioned)'
        subtitles[3] = f'RMSE {errors[2]:.1f} mm'
    for index, (array, title, subtitle) in enumerate(zip(images, titles, subtitles)):
        x = 20 + index * w
        canvas.paste(Image.fromarray(array), (x, 103))
        draw.text((x + 10, 52), title, font=font(19), fill='#252d32')
        draw.text((x + 10, 79), subtitle, font=font(14), fill='#59636a')
    draw.text((25, h + 117), 'All points | shared camera and scale | 0.5x playback', font=font(14), fill='#59636a')
    if compare_methods:
        return np.asarray(canvas)
    ramp = (error_colors(np.linspace(0, maximum, 220), maximum) * 255).astype(np.uint8)
    canvas.paste(Image.fromarray(np.repeat(ramp[None], 10, axis=0)), (3 * w + 80, h + 119))
    draw.text((3 * w + 30, h + 114), '0', font=font(13), fill='#59636a')
    draw.text((3 * w + 308, h + 114), f'{maximum:.0f}+ mm', font=font(13), fill='#59636a')
    return np.asarray(canvas)


def run(args):
    name = args.case
    dataset = 'dna_rendering' if name.startswith('dna_') else 'actorshq'
    output = args.output / name
    output.mkdir(parents=True, exist_ok=True)
    pred_dir = args.predictions / name
    metadata = json.loads((pred_dir / 'metadata.json').read_text())
    ids = np.asarray(metadata['test_frame_ids'])
    params_path = args.release / 'resources/multiview/d3dgs' / dataset / name / 'params.npz'
    with np.load(params_path) as archive:
        reference = archive['means3D'][ids]
        colors = np.clip(archive['rgb_colors'][ids], 0, 1)
    prediction = np.load(pred_dir / 'test_warped_sequence.npy', mmap_mode='r')
    baseline_dir = args.release / 'runs/multiview_novel_pose' / dataset / name / 'smpl_lbs/lbs'
    baseline = np.load(baseline_dir / 'test_warped_sequence.npy', mmap_mode='r')
    if not (reference.shape == prediction.shape == baseline.shape):
        raise ValueError('reference and prediction point indexing/shapes differ')
    sequences = [reference, prediction, baseline]
    bob_dir = baseline_dir.parent.parent / 'smpl_bob/bob' / name
    if args.compare_methods:
        bob = np.load(bob_dir / 'test_warped_sequence.npy', mmap_mode='r')
        if bob.shape != reference.shape:
            raise ValueError('BoB prediction shape differs from reference')
        for directory in [pred_dir, baseline_dir, bob_dir]:
            method_metadata = json.loads((directory / 'metadata.json').read_text())
            if not np.array_equal(method_metadata['test_frame_ids'], ids):
                raise ValueError(f'Test frame order differs: {directory}')
            if not np.array_equal(np.load(directory / 'point_indices.npy'), np.arange(reference.shape[1])):
                raise ValueError(f'Point order differs: {directory}')
        sequences.append(bob)
        bob_error = np.linalg.norm(bob - reference, axis=-1) * 1000
        bob_rmse = np.sqrt(np.mean(bob_error ** 2, axis=1))
    with np.load(baseline_dir.parent.parent / 'smpl_partmm/motion/smpl_motion.npz') as archive:
        skeleton = PartMMSkeleton(archive['rest_joints'], archive['parents'])
        pose = PartMMQuery(archive['local_rotations'][ids], archive['root_translation'][ids])
    _, joints = forward_kinematics(skeleton, pose)
    point_error = np.linalg.norm(prediction - reference, axis=-1) * 1000
    baseline_error = np.linalg.norm(baseline - reference, axis=-1) * 1000
    rmse = np.sqrt(np.mean(point_error ** 2, axis=1))
    baseline_rmse = np.sqrt(np.mean(baseline_error ** 2, axis=1))
    poster_index = int(np.argmax(rmse))
    o3d = _open3d()
    renderer = _renderer(o3d, args.width, args.height)
    camera = setup_camera(o3d, renderer, sequences,
                          camera_basis(joints), args.width, args.height)
    def render_frame(index, view):
        last_cloud = (bob[index], colors[index]) if args.compare_methods else (
            prediction[index], error_colors(point_error[index], args.error_max_mm))
        images = [cloud_image(o3d, renderer, points, rgb) for points, rgb in [
            (reference[index], colors[index]), (prediction[index], colors[index]),
            (baseline[index], colors[index]),
            last_cloud,
        ]]
        errors = [rmse[index], baseline_rmse[index]]
        if args.compare_methods:
            errors.append(bob_rmse[index])
        return compose(images, name, int(ids[index]), errors, args.error_max_mm, view, args.compare_methods)
    frames = []
    for index in range(len(ids)):
        frame = render_frame(index, 'front-oblique')
        frames.append(frame)
        if index == poster_index:
            Image.fromarray(frame).save(output / 'front.png')
        if index % 10 == 0:
            print(f'{name}: {index + 1}/{len(ids)}', flush=True)
    save_animation_gif(output / 'pointcloud_comparison.gif', frames, fps=15)
    side_camera = setup_camera(o3d, renderer, sequences,
                               camera_basis(joints, side=True), args.width, args.height)
    Image.fromarray(render_frame(poster_index, 'side')).save(output / 'side.png')
    # Export all points of the worst frame for independent Open3D inspection.
    exports = [
        ('reference', reference[poster_index], colors[poster_index]),
        ('partmm', prediction[poster_index], colors[poster_index]),
        ('robust', baseline[poster_index], colors[poster_index]),
        ('error', prediction[poster_index], error_colors(point_error[poster_index], args.error_max_mm)),
    ]
    if args.compare_methods:
        exports.append(('bob', bob[poster_index], colors[poster_index]))
    for label, points, rgb in exports:
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64)))
        cloud.colors = o3d.utility.Vector3dVector(rgb)
        o3d.io.write_point_cloud(str(output / f'{label}.ply'), cloud)
    record = dict(case=name, point_count=int(reference.shape[1]), frames=ids.tolist(),
        source_params=str(params_path), prediction=str(pred_dir), robust_baseline=str(baseline_dir),
        color_source='reference_frame_rgb_shared_by_all_methods', error_max_mm=args.error_max_mm,
        error_metric='euclidean_point_displacement_mm', filtered_points=0,
        camera=camera, side_camera=side_camera, poster_frame=int(ids[poster_index]),
        frame_rmse_mm=rmse.tolist(), baseline_frame_rmse_mm=baseline_rmse.tolist(),
        aggregate_rmse_mm=float(np.sqrt(np.mean(point_error ** 2))),
        playback_fps=15, source_fps=30)
    record['method_metrics'] = {
        label: dict(rmse_mm=float(np.sqrt(np.mean(error.astype(np.float64) ** 2))),
                    mean_error_mm=float(np.mean(error.astype(np.float64))))
        for label, error in [('partmm', point_error), ('robust', baseline_error)]
    }
    if args.compare_methods:
        record['bob_baseline'] = str(bob_dir)
        record['bob_frame_rmse_mm'] = bob_rmse.tolist()
        record['method_metrics']['bob'] = dict(
            rmse_mm=float(np.sqrt(np.mean(bob_error.astype(np.float64) ** 2))),
            mean_error_mm=float(np.mean(bob_error.astype(np.float64))))
    (output / 'visualization.json').write_text(json.dumps(record, indent=2) + '\n')
    print(str(output), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', required=True)
    parser.add_argument('--release', type=Path, default=Path('/mnt/local_nas/wjx/GaussiAnimate_release_v1'))
    parser.add_argument('--predictions', type=Path, default=ROOT / 'experiments/runs/partmm_repair_20260913_final')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/partmm_error_pointclouds')
    parser.add_argument('--width', type=int, default=440)
    parser.add_argument('--height', type=int, default=620)
    parser.add_argument('--error-max-mm', type=float, default=2000.)
    parser.add_argument('--compare-methods', action='store_true', help='Show BoB in the fourth column instead of PartMM error')
    run(parser.parse_args())
