#!/usr/bin/env python3
# Local experiment adapter built on Dynamic3DGaussians (MIT License).
# The external Gaussian rasterizer keeps its research-only license.
"""
Rigorous BoB rendering evaluation: PSNR, SSIM, LPIPS.

For each sequence, on test frames (last 20%):
  1. "Oracle": Render with GT Gaussian positions -> metrics (upper bound)
  2. "BoB":    Render with BoB-predicted positions -> metrics

Cameras: 4 evenly-spaced cameras with maximum angular separation.

Supports multi-GPU parallel execution via --sequences and --gpu.

Usage (single GPU):
    python eval_bob_metrics.py --gpu 0 --sequences dna_0012_11

Usage (multi-GPU, launch separately):
    python eval_bob_metrics.py --gpu 0 --sequences dna_0012_11 dna_0013_01 ...
    python eval_bob_metrics.py --gpu 1 --sequences dna_0013_09 dna_0013_11 ...

Merge results:
    python eval_bob_metrics.py --merge_only --output_dir output/bob_eval_v2
"""

import argparse
import json
import os
import sys
import glob
import time

import numpy as np
import torch
from PIL import Image

from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from helpers import setup_camera

from experiments.scheduling import configure_visible_gpu


# ========================== Metrics ==========================

def compute_psnr(img1, img2):
    """PSNR between two [C,H,W] tensors in [0,1]."""
    mse = ((img1 - img2) ** 2).mean()
    if mse < 1e-10:
        return 50.0
    return -10 * torch.log10(mse).item()


def compute_ssim(img1, img2, window_size=11):
    """SSIM between two [C,H,W] tensors in [0,1]. Uses skimage for reliability."""
    from skimage.metrics import structural_similarity
    i1 = img1.cpu().permute(1, 2, 0).numpy()
    i2 = img2.cpu().permute(1, 2, 0).numpy()
    return structural_similarity(i1, i2, channel_axis=2, data_range=1.0,
                                 win_size=min(window_size, min(i1.shape[0], i1.shape[1])))


def compute_lpips(img1, img2, lpips_fn):
    """LPIPS between two [C,H,W] tensors in [0,1]."""
    # lpips expects [B,C,H,W] in [-1,1]
    i1 = img1.unsqueeze(0) * 2 - 1
    i2 = img2.unsqueeze(0) * 2 - 1
    with torch.no_grad():
        return lpips_fn(i1, i2).item()


def compute_fg_psnr(img1, img2, mask):
    """FG-only PSNR: only on foreground pixels."""
    mask3 = mask.unsqueeze(0).expand_as(img1)
    n = mask.sum() * img1.shape[0]
    if n < 1:
        return 0.0
    mse = ((img1 - img2) ** 2 * mask3).sum() / n
    if mse < 1e-10:
        return 50.0
    return -10 * torch.log10(mse).item()


# ========================== Rendering ==========================

def render_gaussians(means3D, rgb_colors, unnorm_rotations, log_scales,
                     logit_opacities, cam, cam_m=None, cam_c=None, cam_id=0):
    """Render Gaussians and return image [C,H,W] in [0,1]."""
    means3D = torch.as_tensor(means3D, device="cuda", dtype=torch.float32)
    rgb_colors = torch.as_tensor(rgb_colors, device="cuda", dtype=torch.float32)
    unnorm_rotations = torch.as_tensor(unnorm_rotations, device="cuda", dtype=torch.float32)
    log_scales = torch.as_tensor(log_scales, device="cuda", dtype=torch.float32)
    logit_opacities = torch.as_tensor(logit_opacities, device="cuda", dtype=torch.float32)

    if log_scales.shape[-1] == 1:
        log_scales = log_scales.repeat(1, 3)

    rendervar = {
        'means3D': means3D,
        'colors_precomp': rgb_colors,
        'rotations': torch.nn.functional.normalize(unnorm_rotations),
        'opacities': torch.sigmoid(logit_opacities),
        'scales': torch.exp(log_scales),
        'means2D': torch.zeros_like(means3D, requires_grad=False, device="cuda"),
    }

    im, _, _ = Renderer(raster_settings=cam)(**rendervar)

    if cam_m is not None and cam_c is not None:
        cam_m_t = torch.tensor(cam_m[cam_id], device="cuda", dtype=torch.float32)
        cam_c_t = torch.tensor(cam_c[cam_id], device="cuda", dtype=torch.float32)
        im = torch.exp(cam_m_t)[:, None, None] * im + cam_c_t[:, None, None]

    im = torch.clamp(im, 0, 1)
    return im


# ========================== Data Loading ==========================

def load_gt_image(data_dir, data_subdir, md, t, cam_id):
    """Load GT image as [C,H,W] in [0,1]."""
    fn = md['fn'][t][cam_id]
    path = os.path.join(data_dir, data_subdir, "ims", fn)
    if not os.path.exists(path):
        return None
    img = np.array(Image.open(path)).astype(np.float32) / 255.0
    return torch.tensor(img, device="cuda").permute(2, 0, 1)


def load_seg_mask(data_dir, data_subdir, md, t, cam_id):
    """Load segmentation mask as [H,W] binary."""
    fn = md['fn'][t][cam_id]
    seg_fn = fn.replace('.jpg', '.png').replace('.jpeg', '.png')
    path = os.path.join(data_dir, data_subdir, "seg", seg_fn)
    if not os.path.exists(path):
        return None
    mask = np.array(Image.open(path))
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]
    if mask.dtype == bool:
        return torch.tensor(mask.astype(np.float32), device="cuda")
    return torch.tensor((mask > 128).astype(np.float32), device="cuda")


def select_evenly_spaced_cameras(num_cams, n_select=4):
    """Select n_select cameras with maximum angular separation."""
    if num_cams <= n_select:
        return list(range(num_cams))
    step = num_cams / n_select
    return [int(round(i * step)) % num_cams for i in range(n_select)]


# ========================== Sequence Config ==========================

def get_sequence_config():
    """Return mapping: seq_name -> (params_dir_key, params_subdir, data_subdir)."""
    dna_seqs = {
        "dna_0012_11": ("dna", "dna_0012_11_1x", "dna_0012_11_1x"),
        "dna_0013_01": ("dna", "dna_0013_01_1x", "dna_0013_01_1x"),
        "dna_0013_03": ("dna", "dna_0013_03_1x", "dna_0013_03_1x"),
        "dna_0013_09": ("dna", "dna_0013_09_1x", "dna_0013_09_1x"),
        "dna_0013_11": ("dna", "dna_0013_11_1x", "dna_0013_11_1x"),
        "dna_0019_08": ("dna", "dna_0019_08_1x", "dna_0019_08_1x"),
        "dna_0021_03": ("dna", "dna_0021_03_1x", "dna_0021_03_1x"),
        "dna_0023_06": ("dna", "dna_0023_06_1x", "dna_0023_06_1x"),
    }
    ahq_seqs = {
        "actorshq_actor01_seq1": ("ahq", "actorshq_actor01_seq1", "actorshq_actor01_seq1"),
        "actorshq_actor01_seq2": ("ahq", "actorshq_actor01_seq2", "actorshq_actor01_seq2"),
        "actorshq_actor02_seq1": ("ahq", "actorshq_actor02_seq1", "actorshq_actor02_seq1"),
        "actorshq_actor02_seq2": ("ahq", "actorshq_actor02_seq2", "actorshq_actor02_seq2"),
        "actorshq_actor04_seq2": ("ahq", "actorshq_actor04_seq2", "actorshq_actor04_seq2"),
        "actorshq_actor05_seq1": ("ahq", "actorshq_actor05_seq1", "actorshq_actor05_seq1"),
        "actorshq_actor06_seq1": ("ahq", "actorshq_actor06_seq1", "actorshq_actor06_seq1"),
        "actorshq_actor06_seq2": ("ahq", "actorshq_actor06_seq2", "actorshq_actor06_seq2"),
        "actorshq_actor07_seq1": ("ahq", "actorshq_actor07_seq1", "actorshq_actor07_seq1"),
        "actorshq_actor08_seq1": ("ahq", "actorshq_actor08_seq1", "actorshq_actor08_seq1"),
    }
    all_seqs = {}
    all_seqs.update(dna_seqs)
    all_seqs.update(ahq_seqs)
    return all_seqs


# ========================== Per-Sequence Eval ==========================

def eval_sequence(seq_name, params_path, bob_dir, data_dir, md_path,
                  data_subdir, lpips_fn, output_dir=None):
    """Evaluate one sequence: Oracle + BoB on test frames x 4 cameras."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {seq_name}")
    print(f"{'='*60}")

    md = json.load(open(md_path))
    w, h = md['w'], md['h']
    num_cams = len(md['fn'][0])

    params = np.load(params_path)
    means3D_all = params['means3D']
    rgb_colors_all = params['rgb_colors']
    unnorm_rots_all = params['unnorm_rotations']
    log_scales = params['log_scales']
    logit_opacities = params['logit_opacities']
    cam_m = params.get('cam_m', None)
    cam_c = params.get('cam_c', None)
    T = means3D_all.shape[0]

    bob_meta = json.load(open(os.path.join(bob_dir, "metadata.json")))
    test_frame_ids = bob_meta["test_frame_ids"]
    n_bob_pts = bob_meta["num_points"]

    warped_seq = np.load(os.path.join(bob_dir, "warped_sequence.npy"))

    pt_indices_path = os.path.join(bob_dir, "point_indices.npy")
    has_indices = False
    if os.path.exists(pt_indices_path):
        pt_indices = np.load(pt_indices_path)
        has_indices = True
    elif n_bob_pts == means3D_all.shape[1]:
        pt_indices = np.arange(n_bob_pts)
        has_indices = True

    # Select 4 evenly-spaced cameras
    cam_ids = select_evenly_spaced_cameras(num_cams, 4)
    print(f"  {T} timesteps, {means3D_all.shape[1]} Gaussians, {num_cams} cameras")
    print(f"  Test frames: {len(test_frame_ids)}, BoB points: {n_bob_pts}, has_indices: {has_indices}")
    print(f"  Eval cameras (evenly spaced): {cam_ids}")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Per-frame metrics
    oracle_metrics = {"psnr": [], "ssim": [], "lpips": [], "fg_psnr": []}
    bob_metrics = {"psnr": [], "ssim": [], "lpips": [], "fg_psnr": []}

    total_evals = len(test_frame_ids) * len(cam_ids)
    eval_count = 0
    t_start = time.time()

    for t_idx, t in enumerate(test_frame_ids):
        if t >= T or t >= len(md['fn']):
            continue

        for cam_id in cam_ids:
            if cam_id >= num_cams:
                continue

            k = md['k'][t][cam_id]
            w2c = md['w2c'][t][cam_id]
            cam = setup_camera(w, h, k, w2c, near=1.0, far=100)

            gt_img = load_gt_image(data_dir, data_subdir, md, t, cam_id)
            if gt_img is None:
                continue
            mask = load_seg_mask(data_dir, data_subdir, md, t, cam_id)

            # Apply mask to GT (consistent with training: bg=black)
            gt_masked = gt_img * mask.unsqueeze(0) if mask is not None else gt_img

            # --- Oracle ---
            with torch.no_grad():
                oracle_im = render_gaussians(
                    means3D_all[t], rgb_colors_all[t], unnorm_rots_all[t],
                    log_scales, logit_opacities, cam, cam_m, cam_c, cam_id
                )
            oracle_metrics["psnr"].append(compute_psnr(oracle_im, gt_masked))
            oracle_metrics["ssim"].append(compute_ssim(oracle_im, gt_masked))
            oracle_metrics["lpips"].append(compute_lpips(oracle_im, gt_masked, lpips_fn))
            if mask is not None:
                oracle_metrics["fg_psnr"].append(compute_fg_psnr(oracle_im, gt_img, mask))

            # --- BoB ---
            if has_indices:
                means3d_bob = means3D_all[t].copy()
                means3d_bob[pt_indices] = warped_seq[t]
                with torch.no_grad():
                    bob_im = render_gaussians(
                        means3d_bob, rgb_colors_all[t], unnorm_rots_all[t],
                        log_scales, logit_opacities, cam, cam_m, cam_c, cam_id
                    )
                bob_metrics["psnr"].append(compute_psnr(bob_im, gt_masked))
                bob_metrics["ssim"].append(compute_ssim(bob_im, gt_masked))
                bob_metrics["lpips"].append(compute_lpips(bob_im, gt_masked, lpips_fn))
                if mask is not None:
                    bob_metrics["fg_psnr"].append(compute_fg_psnr(bob_im, gt_img, mask))

            eval_count += 1
            if eval_count % 20 == 0 or eval_count == total_evals:
                elapsed = time.time() - t_start
                eta = elapsed / eval_count * (total_evals - eval_count)
                print(f"  [{eval_count}/{total_evals}] "
                      f"Oracle PSNR={np.mean(oracle_metrics['psnr']):.2f} "
                      f"SSIM={np.mean(oracle_metrics['ssim']):.4f} "
                      f"LPIPS={np.mean(oracle_metrics['lpips']):.4f} | "
                      f"BoB PSNR={np.mean(bob_metrics['psnr']):.2f} "
                      f"SSIM={np.mean(bob_metrics['ssim']):.4f} "
                      f"LPIPS={np.mean(bob_metrics['lpips']):.4f} "
                      f"(ETA {eta:.0f}s)")

    # Compile
    def mean_or_na(lst):
        return float(np.mean(lst)) if lst else -1.0

    results = {
        "seq_name": seq_name,
        "num_test_frames": len(test_frame_ids),
        "num_eval_cams": len(cam_ids),
        "cam_ids": cam_ids,
        "total_evals": eval_count,
        "oracle": {
            "psnr": mean_or_na(oracle_metrics["psnr"]),
            "ssim": mean_or_na(oracle_metrics["ssim"]),
            "lpips": mean_or_na(oracle_metrics["lpips"]),
            "fg_psnr": mean_or_na(oracle_metrics["fg_psnr"]),
        },
        "bob": {
            "psnr": mean_or_na(bob_metrics["psnr"]),
            "ssim": mean_or_na(bob_metrics["ssim"]),
            "lpips": mean_or_na(bob_metrics["lpips"]),
            "fg_psnr": mean_or_na(bob_metrics["fg_psnr"]),
        },
    }

    print(f"\n  {seq_name} Results:")
    print(f"    Oracle:  PSNR={results['oracle']['psnr']:.2f}  "
          f"SSIM={results['oracle']['ssim']:.4f}  "
          f"LPIPS={results['oracle']['lpips']:.4f}  "
          f"FG_PSNR={results['oracle']['fg_psnr']:.2f}")
    print(f"    BoB:     PSNR={results['bob']['psnr']:.2f}  "
          f"SSIM={results['bob']['ssim']:.4f}  "
          f"LPIPS={results['bob']['lpips']:.4f}  "
          f"FG_PSNR={results['bob']['fg_psnr']:.2f}")

    # Save per-sequence result
    if output_dir:
        with open(os.path.join(output_dir, f"{seq_name}.json"), "w") as f:
            json.dump(results, f, indent=2)

    return results


# ========================== Merge & Summary ==========================

def merge_results(output_dir):
    """Merge all per-sequence JSON results and print summary table."""
    json_files = sorted(glob.glob(os.path.join(output_dir, "*.json")))
    json_files = [f for f in json_files if os.path.basename(f) != "summary.json"]

    all_results = {}
    for jf in json_files:
        with open(jf) as f:
            data = json.load(f)
        seq = data["seq_name"]
        all_results[seq] = data

    if not all_results:
        print("No results found to merge.")
        return

    # Print table
    header = (f"{'Sequence':<30s} "
              f"{'Orac_PSNR':>10s} {'Orac_SSIM':>10s} {'Orac_LPIPS':>11s} {'Orac_FG':>10s} "
              f"{'BoB_PSNR':>10s} {'BoB_SSIM':>10s} {'BoB_LPIPS':>11s} {'BoB_FG':>10s}")
    print(f"\n{'='*135}")
    print("RENDERING EVALUATION RESULTS (Test Frames, 4 Evenly-Spaced Cameras)")
    print(f"{'='*135}")
    print(header)
    print("-" * 135)

    dna_keys = sorted([k for k in all_results if k.startswith("dna_")])
    ahq_keys = sorted([k for k in all_results if k.startswith("actorshq_")])

    def print_row(seq, res):
        o = res["oracle"]
        b = res["bob"]
        print(f"{seq:<30s} "
              f"{o['psnr']:>10.2f} {o['ssim']:>10.4f} {o['lpips']:>11.4f} {o['fg_psnr']:>10.2f} "
              f"{b['psnr']:>10.2f} {b['ssim']:>10.4f} {b['lpips']:>11.4f} {b['fg_psnr']:>10.2f}")

    def compute_avg(keys):
        if not keys:
            return None
        avg = {"oracle": {}, "bob": {}}
        for metric in ["psnr", "ssim", "lpips", "fg_psnr"]:
            o_vals = [all_results[k]["oracle"][metric] for k in keys if all_results[k]["oracle"][metric] > -0.5]
            b_vals = [all_results[k]["bob"][metric] for k in keys if all_results[k]["bob"][metric] > -0.5]
            avg["oracle"][metric] = float(np.mean(o_vals)) if o_vals else -1.0
            avg["bob"][metric] = float(np.mean(b_vals)) if b_vals else -1.0
        return avg

    # DNA
    if dna_keys:
        for k in dna_keys:
            print_row(k, all_results[k])
        dna_avg = compute_avg(dna_keys)
        print("-" * 135)
        print(f"{'DNA Average':<30s} "
              f"{dna_avg['oracle']['psnr']:>10.2f} {dna_avg['oracle']['ssim']:>10.4f} {dna_avg['oracle']['lpips']:>11.4f} {dna_avg['oracle']['fg_psnr']:>10.2f} "
              f"{dna_avg['bob']['psnr']:>10.2f} {dna_avg['bob']['ssim']:>10.4f} {dna_avg['bob']['lpips']:>11.4f} {dna_avg['bob']['fg_psnr']:>10.2f}")
        print()

    # ActorsHQ
    if ahq_keys:
        for k in ahq_keys:
            print_row(k, all_results[k])
        ahq_avg = compute_avg(ahq_keys)
        print("-" * 135)
        print(f"{'ActorsHQ Average':<30s} "
              f"{ahq_avg['oracle']['psnr']:>10.2f} {ahq_avg['oracle']['ssim']:>10.4f} {ahq_avg['oracle']['lpips']:>11.4f} {ahq_avg['oracle']['fg_psnr']:>10.2f} "
              f"{ahq_avg['bob']['psnr']:>10.2f} {ahq_avg['bob']['ssim']:>10.4f} {ahq_avg['bob']['lpips']:>11.4f} {ahq_avg['bob']['fg_psnr']:>10.2f}")
        print()

    # Overall
    all_keys = dna_keys + ahq_keys
    overall_avg = compute_avg(all_keys)
    print("=" * 135)
    print(f"{'Overall Average':<30s} "
          f"{overall_avg['oracle']['psnr']:>10.2f} {overall_avg['oracle']['ssim']:>10.4f} {overall_avg['oracle']['lpips']:>11.4f} {overall_avg['oracle']['fg_psnr']:>10.2f} "
          f"{overall_avg['bob']['psnr']:>10.2f} {overall_avg['bob']['ssim']:>10.4f} {overall_avg['bob']['lpips']:>11.4f} {overall_avg['bob']['fg_psnr']:>10.2f}")
    print("=" * 135)

    # Save summary
    summary = {
        "per_sequence": all_results,
        "dna_average": dna_avg if dna_keys else None,
        "actorshq_average": ahq_avg if ahq_keys else None,
        "overall_average": overall_avg,
    }
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {os.path.join(output_dir, 'summary.json')}")


# ========================== Main ==========================

def main():
    parser = argparse.ArgumentParser(description="Rigorous BoB rendering evaluation")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--sequences", nargs="+", default=None,
                        help="Sequences to evaluate (default: all)")
    parser.add_argument("--bob_root", type=str, default="output/bob_results_full")
    parser.add_argument("--dna_results_dir", type=str, default="output/dna_exp_1x")
    parser.add_argument("--actorshq_results_dir", type=str, default="output/actorshq_exp")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="output/bob_eval_v2")
    parser.add_argument("--merge_only", action="store_true",
                        help="Only merge existing per-sequence results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.merge_only:
        merge_results(args.output_dir)
        return

    configure_visible_gpu(args.gpu)

    # Initialize LPIPS
    import lpips
    lpips_fn = lpips.LPIPS(net="vgg").cuda().eval()

    seq_config = get_sequence_config()

    if args.sequences is None:
        args.sequences = list(seq_config.keys())

    for seq_name in args.sequences:
        if seq_name not in seq_config:
            print(f"Unknown sequence: {seq_name}, skipping.")
            continue

        dtype, params_subdir, data_subdir = seq_config[seq_name]
        if dtype == "dna":
            params_path = os.path.join(args.dna_results_dir, params_subdir, "params.npz")
        else:
            params_path = os.path.join(args.actorshq_results_dir, params_subdir, "params.npz")

        md_path = os.path.join(args.data_dir, data_subdir, "train_meta.json")
        bob_dir = os.path.join(args.bob_root, seq_name)

        if not os.path.exists(params_path):
            print(f"SKIP {seq_name}: no params at {params_path}")
            continue
        if not os.path.exists(bob_dir) or not os.path.exists(os.path.join(bob_dir, "metadata.json")):
            print(f"SKIP {seq_name}: no bob results at {bob_dir}")
            continue
        if not os.path.exists(md_path):
            print(f"SKIP {seq_name}: no metadata at {md_path}")
            continue

        eval_sequence(
            seq_name=seq_name,
            params_path=params_path,
            bob_dir=bob_dir,
            data_dir=args.data_dir,
            md_path=md_path,
            data_subdir=data_subdir,
            lpips_fn=lpips_fn,
            output_dir=args.output_dir,
        )
        torch.cuda.empty_cache()

    # Merge at the end
    print("\n\nMerging all available results...")
    merge_results(args.output_dir)


if __name__ == "__main__":
    main()
