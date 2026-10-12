"""Save actual Gaussian renders using the existing held-out evaluation protocol."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
METHODS = [('gt', 'Ground truth'), ('d3dgs', 'D3DGS'), ('robust', 'Robust Skin Transfer'),
           ('partmm', 'PartMM (repaired)'), ('bob', 'BoB (adapted)')]


def font(size):
    return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', size)


def to_image(tensor):
    array = tensor.detach().cpu().permute(1, 2, 0).numpy()
    return Image.fromarray(np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8))


def contact_sheet(rows, records, case, camera_id, output):
    width = 360
    original_width, original_height = rows[0][0].size
    height = round(original_height * (width - 6) / original_width)
    canvas = Image.new('RGB', (5 * width + 40, 116 + len(rows) * (height + 46)), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 12), f'{case} | camera {camera_id}', font=font(25), fill='#252d32')
    draw.text((20, 48), 'Gaussian splatting | full camera frame | per-image PSNR before resizing',
              font=font(15), fill='#59636a')
    for column, (_, title) in enumerate(METHODS):
        draw.text((20 + column * width, 82), title, font=font(18), fill='#252d32')
    for row, (images, record) in enumerate(zip(rows, records)):
        y = 116 + row * (height + 46)
        for column, ((key, _), img) in enumerate(zip(METHODS, images)):
            x = 20 + column * width
            canvas.paste(img.resize((width - 6, height), Image.Resampling.LANCZOS), (x, y))
            score = record['psnr_db'].get(key)
            label = f"Frame {record['frame']}" if score is None else f'{score:.2f} dB'
            draw.text((x, y + height + 9), label, font=font(16), fill='#59636a')
    canvas.save(output)


def run(args):
    sys.path.insert(0, str(args.d3dgs_root))
    from eval_lbs_metrics import (compute_psnr, compute_fg_psnr, load_gt_image,
                                  load_seg_mask, render_gaussians, select_evenly_spaced_cameras)
    from helpers import setup_camera
    import torch

    torch.cuda.set_device(0)
    case = args.case
    dataset = 'dna_rendering' if case.startswith('dna_') else 'actorshq'
    data_subdir = case + '_1x' if dataset == 'dna_rendering' else case
    run_root = args.release_root / 'runs/multiview_novel_pose' / dataset / case
    sources = {
        'robust': run_root / 'smpl_lbs/lbs',
        'partmm': ROOT / 'experiments/runs/partmm_repair_20260913_final' / case,
        'bob': run_root / 'smpl_bob/bob' / case,
    }
    predictions = {}
    ids = None
    for method, directory in sources.items():
        metadata = json.loads((directory / 'metadata.json').read_text())
        method_ids = metadata['test_frame_ids']
        if ids is not None and method_ids != ids:
            raise ValueError(f'Test frame order differs: {directory}')
        ids = method_ids
        predictions[method] = np.load(directory / 'test_warped_sequence.npy', mmap_mode='r')
        count = predictions[method].shape[1]
        if not np.array_equal(np.load(directory / 'point_indices.npy'), np.arange(count)):
            raise ValueError(f'Point order differs: {directory}')
    selected = [ids[index] for index in np.linspace(0, len(ids) - 1, 3, dtype=int)]
    camera_path = args.data_dir / data_subdir / 'train_meta.json'
    camera_metadata = json.loads(camera_path.read_text())
    camera_ids = select_evenly_spaced_cameras(len(camera_metadata['fn'][0]), 4)
    params_path = args.release_root / 'resources/multiview/d3dgs' / dataset / case / 'params.npz'
    output = args.output / case
    output.mkdir(parents=True, exist_ok=True)
    all_records = []
    with np.load(params_path) as archive:
        means = archive['means3D']
        colors = archive['rgb_colors']
        rotations = archive['unnorm_rotations']
        scales = archive['log_scales']
        opacities = archive['logit_opacities']
        cam_m = archive['cam_m'] if 'cam_m' in archive else None
        cam_c = archive['cam_c'] if 'cam_c' in archive else None
        for method, prediction in predictions.items():
            if prediction.shape != (len(ids), means.shape[1], 3):
                raise ValueError(f'Prediction shape differs: {method}')
        for camera_id in camera_ids:
            rows, records = [], []
            for frame in selected:
                camera = setup_camera(camera_metadata['w'], camera_metadata['h'],
                                      camera_metadata['k'][frame][camera_id],
                                      camera_metadata['w2c'][frame][camera_id], near=1., far=100.)
                gt = load_gt_image(str(args.data_dir), data_subdir, camera_metadata, frame, camera_id)
                if gt is None:
                    raise FileNotFoundError(f'Missing GT: {case} frame {frame} camera {camera_id}')
                mask = load_seg_mask(str(args.data_dir), data_subdir, camera_metadata, frame, camera_id)
                target = gt * mask.unsqueeze(0) if mask is not None else gt
                images = [to_image(target)]
                record = dict(frame=frame, camera=camera_id, psnr_db={}, foreground_psnr_db={})
                with torch.no_grad():
                    for method, _ in METHODS[1:]:
                        points = means[frame] if method == 'd3dgs' else predictions[method][ids.index(frame)]
                        rendered = render_gaussians(np.array(points), colors[frame], rotations[frame],
                                                    scales, opacities, camera, cam_m, cam_c, camera_id)
                        record['psnr_db'][method] = compute_psnr(rendered, target)
                        if mask is not None:
                            record['foreground_psnr_db'][method] = compute_fg_psnr(rendered, gt, mask)
                        images.append(to_image(rendered))
                frame_dir = output / f'camera_{camera_id:03d}' / f'frame_{frame:04d}'
                frame_dir.mkdir(parents=True, exist_ok=True)
                for (method, _), image in zip(METHODS, images):
                    image.save(frame_dir / f'{method}.png')
                rows.append(images)
                records.append(record)
                print(f'{case}: camera {camera_id}, frame {frame}', flush=True)
            contact_sheet(rows, records, case, camera_id, output / f'camera_{camera_id:03d}.png')
            all_records.extend(records)
    manifest = dict(case=case, source_params=str(params_path), cameras=str(camera_path),
                    predictions={k: str(v) for k, v in sources.items()},
                    attribute_source='target_frame', filtered_points=0, crop=None,
                    selected_frames=selected, camera_ids=camera_ids,
                    metrics_scope='selected_images_only_not_full_sequence', records=all_records)
    (output / 'render_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', required=True)
    parser.add_argument('--release-root', type=Path, default=Path('/mnt/local_nas/wjx/GaussiAnimate_release_v1'))
    parser.add_argument('--d3dgs-root', type=Path, default=Path('/home/ubuntu/research/GA/Dynamic3DGaussians'))
    parser.add_argument('--data-dir', type=Path, default=Path('/home/ubuntu/research/data/DNA-Rendering'))
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/human_render_comparison')
    run(parser.parse_args())
