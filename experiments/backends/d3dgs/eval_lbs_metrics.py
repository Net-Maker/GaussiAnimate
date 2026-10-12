"""
Evaluate LBS-driven Gaussians: PSNR, SSIM, LPIPS.
Consistent with eval_bob_metrics.py and eval_ssdr_metrics.py:
  - LPIPS net="vgg"
  - 4 evenly-spaced cameras
  - Test frames = last 20%
  - GT masked with seg mask
"""
import argparse, json, os, sys, glob, time
import numpy as np
import torch
from PIL import Image
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from helpers import setup_camera


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
    i1 = img1.unsqueeze(0) * 2 - 1
    i2 = img2.unsqueeze(0) * 2 - 1
    with torch.no_grad():
        return lpips_fn(i1, i2).item()


def compute_fg_psnr(img1, img2, mask):
    """FG-only PSNR: only on foreground pixels. Same as BoB/SSDR eval."""
    mask3 = mask.unsqueeze(0).expand_as(img1)
    n = mask.sum() * img1.shape[0]
    if n < 1:
        return 0.0
    mse = ((img1 - img2) ** 2 * mask3).sum() / n
    if mse < 1e-10:
        return 50.0
    return -10 * torch.log10(mse).item()


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
    return torch.clamp(im, 0, 1)


def load_gt_image(data_dir, data_subdir, md, t, cam_id):
    """Load GT image as [C,H,W] in [0,1]. Same path logic as BoB/SSDR."""
    fn = md["fn"][t][cam_id]
    path = os.path.join(data_dir, data_subdir, "ims", fn)
    if not os.path.exists(path):
        return None
    img = np.array(Image.open(path)).astype(np.float32) / 255.0
    return torch.tensor(img, device="cuda").permute(2, 0, 1)


def load_seg_mask(data_dir, data_subdir, md, t, cam_id):
    """Load segmentation mask as [H,W] binary. Same as BoB/SSDR."""
    fn = md["fn"][t][cam_id]
    seg_fn = fn.replace(".jpg", ".png").replace(".jpeg", ".png")
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
    """Select n_select cameras with maximum angular separation. Same as BoB/SSDR."""
    if num_cams <= n_select:
        return list(range(num_cams))
    step = num_cams / n_select
    return [int(round(i * step)) % num_cams for i in range(n_select)]


def get_sequence_config():
    """Return mapping: seq_name -> (dtype, params_subdir, data_subdir). Same as BoB/SSDR."""
    dna_seqs = {}
    for s in ["0012_11", "0013_01", "0013_03", "0013_09", "0013_11", "0019_08", "0021_03", "0023_06"]:
        dna_seqs["dna_" + s] = ("dna", "dna_" + s + "_1x", "dna_" + s + "_1x")

    ahq_seqs = {}
    for s in ["actor01_seq1", "actor01_seq2", "actor02_seq1", "actor02_seq2",
              "actor04_seq2", "actor05_seq1", "actor06_seq1", "actor06_seq2",
              "actor07_seq1", "actor08_seq1"]:
        ahq_seqs["actorshq_" + s] = ("ahq", "actorshq_" + s, "actorshq_" + s)

    all_seqs = {}
    all_seqs.update(dna_seqs)
    all_seqs.update(ahq_seqs)
    return all_seqs


def eval_sequence(seq_name, params_path, lbs_dir, data_dir, data_subdir,
                  md_path, lpips_fn, train_ratio=0.8, output_dir=None):
    """Evaluate one sequence: Oracle + LBS on test frames x 4 cameras."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {seq_name}")
    print(f"{'='*60}")

    md = json.load(open(md_path))
    w, h = md["w"], md["h"]
    num_cams = len(md["fn"][0])

    params = np.load(params_path, allow_pickle=True)
    means3D_all = params["means3D"]
    rgb_colors_all = params["rgb_colors"]
    unnorm_rots_all = params["unnorm_rotations"]
    log_scales = params["log_scales"]
    logit_opacities = params["logit_opacities"]
    cam_m = params.get("cam_m", None)
    cam_c = params.get("cam_c", None)
    T, N = means3D_all.shape[:2]

    # Load LBS results
    lbs_driven_path = os.path.join(lbs_dir, "full_driven_sequence.npy")
    lbs_indices_path = os.path.join(lbs_dir, "point_indices.npy")
    if not os.path.exists(lbs_driven_path) or not os.path.exists(lbs_indices_path):
        print(f"  SKIP: LBS results not found at {lbs_dir}")
        return None
    lbs_driven = np.load(lbs_driven_path)  # (T, N_subset, 3)
    lbs_indices = np.load(lbs_indices_path)  # (N_subset,)
    print(f"  LBS loaded: driven={lbs_driven.shape}, indices={lbs_indices.shape}")

    # Train/test split: last 20% as test (same as SSDR and BoB)
    n_train = int(T * train_ratio)
    test_frame_ids = list(range(n_train, T))

    # Select 4 evenly-spaced cameras (same as BoB/SSDR)
    cam_ids = select_evenly_spaced_cameras(num_cams, 4)
    print(f"  T={T}, N={N}, cams={num_cams}")
    print(f"  Train: 0-{n_train-1} ({n_train}), Test: {n_train}-{T-1} ({len(test_frame_ids)})")
    print(f"  Eval cameras (evenly spaced): {cam_ids}")

    oracle_metrics = {"psnr": [], "ssim": [], "lpips": [], "fg_psnr": []}
    lbs_metrics = {"psnr": [], "ssim": [], "lpips": [], "fg_psnr": []}

    total = len(test_frame_ids) * len(cam_ids)
    count = 0
    t0 = time.time()

    for t in test_frame_ids:
        if t >= T or t >= len(md["fn"]):
            continue

        # Construct LBS means3D for this frame
        lbs_means = means3D_all[t].copy()
        if t < lbs_driven.shape[0]:
            lbs_means[lbs_indices] = lbs_driven[t]

        for cam_id in cam_ids:
            if cam_id >= num_cams:
                continue

            k = md["k"][t][cam_id]
            w2c = md["w2c"][t][cam_id]
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

            # --- LBS ---
            with torch.no_grad():
                lbs_im = render_gaussians(
                    lbs_means, rgb_colors_all[t], unnorm_rots_all[t],
                    log_scales, logit_opacities, cam, cam_m, cam_c, cam_id
                )
            lbs_metrics["psnr"].append(compute_psnr(lbs_im, gt_masked))
            lbs_metrics["ssim"].append(compute_ssim(lbs_im, gt_masked))
            lbs_metrics["lpips"].append(compute_lpips(lbs_im, gt_masked, lpips_fn))
            if mask is not None:
                lbs_metrics["fg_psnr"].append(compute_fg_psnr(lbs_im, gt_img, mask))

            count += 1
            if count % 20 == 0:
                elapsed = time.time() - t0
                eta = elapsed / count * (total - count)
                print(f"    {count}/{total} ETA: {int(eta)}s")

    def safe_mean(lst):
        return float(np.mean(lst)) if lst else -1.0

    result = {
        "oracle": {k: safe_mean(v) for k, v in oracle_metrics.items()},
        "lbs": {k: safe_mean(v) for k, v in lbs_metrics.items()},
    }

    # Load vertex RMSE from metadata
    meta_path = os.path.join(lbs_dir, "metadata.json")
    if os.path.exists(meta_path):
        meta = json.load(open(meta_path))
        result["lbs_vertex_rmse_mm"] = meta.get("test_rmse_mm", -1)

    print(f"\n  Oracle: PSNR={result['oracle']['psnr']:.2f} "
          f"SSIM={result['oracle']['ssim']:.4f} LPIPS={result['oracle']['lpips']:.4f}")
    print(f"  LBS:    PSNR={result['lbs']['psnr']:.2f} "
          f"SSIM={result['lbs']['ssim']:.4f} LPIPS={result['lbs']['lpips']:.4f}")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, seq_name + ".json"), "w") as f:
            json.dump(result, f, indent=2)

    return result


def merge_results(output_dir):
    all_results = {}
    for f in sorted(glob.glob(os.path.join(output_dir, "*.json"))):
        if "summary" in f:
            continue
        name = os.path.basename(f).replace(".json", "")
        all_results[name] = json.load(open(f))

    dna_keys = sorted([k for k in all_results if k.startswith("dna_")])
    ahq_keys = sorted([k for k in all_results if k.startswith("actorshq_")])

    def avg_metrics(keys):
        res = {}
        for src in ["oracle", "lbs"]:
            res[src] = {}
            for m in ["psnr", "ssim", "lpips", "fg_psnr"]:
                vals = [all_results[k][src][m] for k in keys if all_results[k][src].get(m, -1) > -0.5]
                res[src][m] = float(np.mean(vals)) if vals else -1
        rmse_vals = [all_results[k].get("lbs_vertex_rmse_mm", -1) for k in keys]
        rmse_vals = [v for v in rmse_vals if v > 0]
        res["rmse"] = float(np.mean(rmse_vals)) if rmse_vals else -1
        return res

    header = f"{'Sequence':<35s} {'RMSE_mm':>8s}  {'O_PSNR':>8s} {'O_SSIM':>8s} {'O_LPIPS':>8s}  {'L_PSNR':>8s} {'L_SSIM':>8s} {'L_LPIPS':>8s}"
    sep = "-" * 110

    print(f"\n{'='*110}")
    print("LBS Evaluation Results (LPIPS=vgg, consistent with BoB/SSDR)")
    print(f"{'='*110}")
    print(header)
    print(sep)

    for k in dna_keys:
        r = all_results[k]
        rmse = r.get("lbs_vertex_rmse_mm", -1)
        o, l = r["oracle"], r["lbs"]
        print(f"{k:<35s} {rmse:>8.1f}  {o['psnr']:>8.2f} {o['ssim']:>8.4f} {o['lpips']:>8.4f}  {l['psnr']:>8.2f} {l['ssim']:>8.4f} {l['lpips']:>8.4f}")

    if dna_keys:
        da = avg_metrics(dna_keys)
        print(sep)
        print(f"{'DNA Average ('+str(len(dna_keys))+' seqs)':<35s} {da['rmse']:>8.1f}  {da['oracle']['psnr']:>8.2f} {da['oracle']['ssim']:>8.4f} {da['oracle']['lpips']:>8.4f}  {da['lbs']['psnr']:>8.2f} {da['lbs']['ssim']:>8.4f} {da['lbs']['lpips']:>8.4f}")

    print()
    for k in ahq_keys:
        r = all_results[k]
        rmse = r.get("lbs_vertex_rmse_mm", -1)
        o, l = r["oracle"], r["lbs"]
        print(f"{k:<35s} {rmse:>8.1f}  {o['psnr']:>8.2f} {o['ssim']:>8.4f} {o['lpips']:>8.4f}  {l['psnr']:>8.2f} {l['ssim']:>8.4f} {l['lpips']:>8.4f}")

    if ahq_keys:
        aa = avg_metrics(ahq_keys)
        print(sep)
        print(f"{'ActorsHQ Average ('+str(len(ahq_keys))+' seqs)':<35s} {aa['rmse']:>8.1f}  {aa['oracle']['psnr']:>8.2f} {aa['oracle']['ssim']:>8.4f} {aa['oracle']['lpips']:>8.4f}  {aa['lbs']['psnr']:>8.2f} {aa['lbs']['ssim']:>8.4f} {aa['lbs']['lpips']:>8.4f}")

    all_keys = dna_keys + ahq_keys
    if all_keys:
        oa = avg_metrics(all_keys)
        print(f"\n{'='*110}")
        print(f"{'Overall Average ('+str(len(all_keys))+' seqs)':<35s} {oa['rmse']:>8.1f}  {oa['oracle']['psnr']:>8.2f} {oa['oracle']['ssim']:>8.4f} {oa['oracle']['lpips']:>8.4f}  {oa['lbs']['psnr']:>8.2f} {oa['lbs']['ssim']:>8.4f} {oa['lbs']['lpips']:>8.4f}")
        print(f"{'='*110}")

    summary = {"per_sequence": all_results}
    if dna_keys:
        summary["dna_average"] = avg_metrics(dna_keys)
    if ahq_keys:
        summary["actorshq_average"] = avg_metrics(ahq_keys)
    if all_keys:
        summary["overall_average"] = avg_metrics(all_keys)

    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {os.path.join(output_dir, 'summary.json')}")


def main():
    parser = argparse.ArgumentParser(description="LBS rendering evaluation (consistent with BoB/SSDR)")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--sequences", nargs="+", default=None)
    parser.add_argument("--output_dir", type=str, default="output/lbs_eval_v2")
    parser.add_argument("--merge_only", action="store_true")
    parser.add_argument("--lbs_root", type=str, default="output/lbs_results")
    parser.add_argument("--dna_results_dir", type=str, default="server_output/dna_exp_1x")
    parser.add_argument("--actorshq_results_dir", type=str, default="output/actorshq_exp")
    parser.add_argument("--data_dir", type=str, default="./data")
    args = parser.parse_args()

    if args.merge_only:
        merge_results(args.output_dir)
        return

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    torch.cuda.set_device(0)

    # LPIPS with VGG backbone - same as BoB and SSDR eval
    import lpips
    lpips_fn = lpips.LPIPS(net="vgg").cuda().eval()

    os.makedirs(args.output_dir, exist_ok=True)

    seq_config = get_sequence_config()

    if args.sequences is None:
        args.sequences = sorted(seq_config.keys())

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
        lbs_dir = os.path.join(args.lbs_root, seq_name)

        if not os.path.exists(params_path):
            print(f"SKIP {seq_name}: no params at {params_path}")
            continue
        if not os.path.exists(md_path):
            print(f"SKIP {seq_name}: no metadata at {md_path}")
            continue

        eval_sequence(seq_name, params_path, lbs_dir, args.data_dir, data_subdir,
                      md_path, lpips_fn, output_dir=args.output_dir)
        torch.cuda.empty_cache()

    merge_results(args.output_dir)


if __name__ == "__main__":
    main()
