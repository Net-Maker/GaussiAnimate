"""Render original and volume-filtered D3DGS with the native rasterizer."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
import yaml


def main(args):
    sys.path.insert(0, str(args.renderer_root))
    from eval_lbs_metrics import render_gaussians, load_gt_image, load_seg_mask, compute_psnr, compute_fg_psnr
    from helpers import setup_camera

    config = yaml.safe_load(args.config.read_text())
    output = Path(config['output'])
    destination = output / 'renders'
    destination.mkdir(exist_ok=True)
    meta = json.loads((Path(config['data_dir']) / config['case'] / 'train_meta.json').read_text())
    frames = np.load(output / 'frame_ids.npy')
    masks = np.load(output / 'keep_mask.npy')
    with np.load(config['d3dgs_params']) as archive:
        params = {key: archive[key] for key in archive.files}
    camera_id = config['camera']
    records = []
    with torch.no_grad():
        for i, frame in enumerate(frames):
            frame = int(frame)
            cam = setup_camera(meta['w'], meta['h'], meta['k'][frame][camera_id], meta['w2c'][frame][camera_id], near=1., far=100.)
            gt = load_gt_image(config['data_dir'], config['case'], meta, frame, camera_id)
            foreground = load_seg_mask(config['data_dir'], config['case'], meta, frame, camera_id)
            if gt is None or foreground is None:
                raise ValueError('Missing ground truth or segmentation for frame {}'.format(frame))
            gt = gt * foreground
            record = dict(frame=frame, kept=int(masks[i].sum()), total=int(len(masks[i])))
            for name, keep in [('original', np.ones_like(masks[i])), ('filtered', masks[i])]:
                if not keep.any():
                    raise ValueError('Empty Gaussian selection')
                prediction = render_gaussians(
                    params['means3D'][frame][keep], params['rgb_colors'][frame][keep],
                    params['unnorm_rotations'][frame][keep], params['log_scales'][keep],
                    params['logit_opacities'][keep], cam, params.get('cam_m'), params.get('cam_c'), camera_id)
                record[name + '_psnr'] = compute_psnr(prediction, gt)
                record[name + '_fg_psnr'] = compute_fg_psnr(prediction, gt, foreground)
                array = prediction.cpu().permute(1, 2, 0).numpy()
                Image.fromarray(np.rint(array * 255).astype(np.uint8)).save(destination / '{}_{:04d}.png'.format(name, frame))
            Image.fromarray(np.rint(gt.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)).save(destination / 'gt_{:04d}.png'.format(frame))
            record['delta_psnr'] = record['filtered_psnr'] - record['original_psnr']
            record['delta_fg_psnr'] = record['filtered_fg_psnr'] - record['original_fg_psnr']
            records.append(record)
            print(json.dumps(record), flush=True)
    means = {key: float(np.mean([r[key] for r in records])) for key in records[0] if 'psnr' in key}
    result = dict(config=str(args.config.resolve()), camera=camera_id, frame_count=len(records),
                  checkpoint=config['d3dgs_params'], mask=str(output / 'keep_mask.npy'),
                  evaluation='Masked GT; full image and foreground-only PSNR, averaged per frame; no retraining',
                  means=means, records=records)
    (output / 'render_metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(means), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('--renderer-root', type=Path, required=True)
    main(parser.parse_args())
