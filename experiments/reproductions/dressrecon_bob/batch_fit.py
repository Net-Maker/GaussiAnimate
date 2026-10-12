#!/usr/bin/env python3
# Local experiment adapter built on DressRecon (MIT License) and
# Dynamic3DGaussians (MIT License); see THIRD_PARTY_NOTICES.md.
"""
Batch run fit_bob on all DNA-Rendering and ActorsHQ sequences.

For each sequence:
  1. Load trained params.npz (4D point clouds)
  2. Load SMPL-X poses from annotation files
  3. Fit BoB model using first 80% frames
  4. Infer point cloud positions for last 20% frames using SMPL poses
  5. Save all results

Usage:
    python batch_fit_bob.py [--gpu GPU_ID] [--sequences SEQ1 SEQ2 ...]
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

from experiments.scheduling import configure_visible_gpu

from .fit_bob import BoBFitter, load_dna_rendering_data
from .pose_contract import smplx_body_to_rootless_links


def load_actorshq_data_with_offset(params_path, smplx_dir, start_frame=0,
                                     opacity_threshold=0.5, max_points=None,
                                     pose_slice=(3, 66), include_global_rot=False):
    """Load ActorsHQ data with correct frame offset for SMPL-X JSON files.

    Args:
        params_path: Path to params.npz from Dynamic3DGaussians training
        smplx_dir: Path to directory with per-frame SMPL-X JSON files
        start_frame: Starting frame index in original sequence
        opacity_threshold: Filter Gaussians with opacity below this
        max_points: Subsample to this many points (None = keep all)
        pose_slice: (start, end) slice of 'poses' vector for body pose
        include_global_rot: Unsupported legacy option. Root rotation must stay
            separate from the rootless FK pose.

    Returns:
        canonical_pts: (V, 3) First-frame point cloud
        target_pts: (T, V, 3) Full point cloud sequence
        smpl_poses: (T, pose_dim) SMPL-X body poses
        point_indices: Indices of selected points
    """
    import json as json_mod

    # Load Gaussian parameters
    print(f"Loading point clouds from {params_path} ...")
    params = np.load(params_path)
    means3d = params["means3D"]  # (T, V_all, 3)
    T = means3d.shape[0]

    # Filter by opacity
    point_indices = np.arange(means3d.shape[1])
    if opacity_threshold > 0 and "logit_opacities" in params:
        logit_op = params["logit_opacities"].squeeze()
        opacity = 1.0 / (1.0 + np.exp(-logit_op))
        mask = opacity > opacity_threshold
        means3d = means3d[:, mask]
        point_indices = point_indices[mask]
        print(f"  Opacity filter (>{opacity_threshold}): {mask.sum()}/{len(mask)} points kept")

    if max_points is not None and means3d.shape[1] > max_points:
        idx = np.random.choice(means3d.shape[1], max_points, replace=False)
        idx.sort()
        means3d = means3d[:, idx]
        point_indices = point_indices[idx]
        print(f"  Subsampled to {max_points} points")

    canonical_pts = means3d[0].astype(np.float32)
    target_pts = means3d.astype(np.float32)

    # Load SMPL-X poses from per-frame JSON files
    # Frame indices: start_frame, start_frame+1, ..., start_frame+T-1
    print(f"Loading SMPL-X poses from {smplx_dir} (start_frame={start_frame}, T={T}) ...")

    poses_list = []
    rh_list = []
    pstart, pend = pose_slice
    missing_count = 0
    for t in range(T):
        fidx = start_frame + t
        json_path = os.path.join(smplx_dir, f"{fidx:06d}.json")
        if not os.path.exists(json_path):
            missing_count += 1
            poses_list.append(np.zeros(pend - pstart, dtype=np.float32))
            rh_list.append(np.zeros(3, dtype=np.float32))
            continue
        with open(json_path) as f:
            data = json_mod.load(f)
        full_pose = np.array(data[0]["poses"]).squeeze()  # (87,)
        body_pose = full_pose[pstart:pend]
        poses_list.append(body_pose.astype(np.float32))
        rh = np.array(data[0]["Rh"]).squeeze()  # (3,)
        rh_list.append(rh.astype(np.float32))

    if missing_count > 0:
        print(f"  WARNING: {missing_count}/{T} frames missing SMPL-X JSON files")

    body_poses = np.stack(poses_list)  # (T, 63)
    rh_all = np.stack(rh_list)         # (T, 3)

    if include_global_rot:
        raise ValueError(
            "global/pelvis rotation must remain separate from the "
            "rootless 23-link FK pose"
        )
    smpl_poses = smplx_body_to_rootless_links(body_poses)
    print("  Global rotation kept separate; rootless pose_dim=69")

    print(f"  Loaded {T} frames, pose_dim={smpl_poses.shape[1]}")
    print(f"  Pose range: [{smpl_poses.min():.4f}, {smpl_poses.max():.4f}]")

    return canonical_pts, target_pts, smpl_poses, point_indices


def run_fit_bob_for_sequence(seq_name, params_path, canonical_pts, target_pts,
                              smpl_poses, output_dir, train_ratio=0.8,
                              num_bones=25, num_iters=2000, lr=1e-3,
                              batch_frames=8, max_points=None, device="cuda",
                              point_indices=None):
    """Run fit_bob for a single sequence and save all results."""

    os.makedirs(output_dir, exist_ok=True)
    print(f"\n{'='*70}")
    print(f"Fitting BoB for: {seq_name}")
    print(f"  Points: {target_pts.shape[1]}, Frames: {target_pts.shape[0]}")
    print(f"  Train ratio: {train_ratio}")
    print(f"  Output: {output_dir}")
    print(f"{'='*70}")

    pose_dim = smpl_poses.shape[1]

    # Create fitter
    fitter = BoBFitter(
        canonical_pts=canonical_pts,
        target_pts=target_pts,
        smpl_poses=smpl_poses,
        num_bones=num_bones,
        pose_dim=pose_dim,
        init_gauss_scale=0.1,
        delta_skin=True,
        device=device,
        train_ratio=train_ratio,
        skel_type="smpl",
        skel_init_gauss_scale=0.03,
    )

    # Fit
    print(f"\nStarting optimization: {num_iters} iterations, lr={lr}")
    t0 = time.time()
    history = fitter.fit(
        num_iters=num_iters,
        lr=lr,
        batch_frames=batch_frames,
        reg_delta_skin=0.01,
    )
    fit_time = time.time() - t0
    print(f"Fitting done in {fit_time:.1f}s")

    # Evaluate
    metrics = fitter.evaluate()

    # Save model
    fitter.save(os.path.join(output_dir, "bob_model.pt"))

    # Save loss history
    np.save(os.path.join(output_dir, "loss_history.npy"), np.array(history))

    # Save warped sequence (all frames - both train and test)
    warped_seq = fitter.get_warped_sequence()
    np.save(os.path.join(output_dir, "warped_sequence.npy"), warped_seq)

    # Save target (GT) sequence
    np.save(os.path.join(output_dir, "target_sequence.npy"), target_pts.astype(np.float32))

    # Save skinning weights
    skin_weights = fitter.get_skinning_weights()
    np.save(os.path.join(output_dir, "skinning_weights.npy"), skin_weights)

    # Save skeleton skinning weights
    skel_skin_weights = fitter.get_skel_skinning_weights()
    np.save(os.path.join(output_dir, "skel_skinning_weights.npy"), skel_skin_weights)

    # Save per-frame error
    np.save(os.path.join(output_dir, "per_frame_error.npy"), metrics["per_frame_mean"])

    # Save train/test split info and inferred point clouds for test frames
    n_train = fitter.n_train
    test_frame_ids = fitter.test_frame_ids

    # The "result" = warped point clouds for test frames (inferred from SMPL pose)
    if test_frame_ids:
        test_warped = warped_seq[test_frame_ids]  # (T_test, V, 3)
        test_target = target_pts[test_frame_ids]  # (T_test, V, 3)
        np.save(os.path.join(output_dir, "test_warped.npy"), test_warped)
        np.save(os.path.join(output_dir, "test_target.npy"), test_target)
        print(f"\nTest results saved: {test_warped.shape}")

    # Save point indices (mapping from subsampled to original params.npz Gaussians)
    if point_indices is not None:
        np.save(os.path.join(output_dir, "point_indices.npy"), point_indices)
        print(f"  Point indices saved: {len(point_indices)} entries")

    # Save metadata
    meta = {
        "seq_name": seq_name,
        "num_frames": int(target_pts.shape[0]),
        "num_points": int(target_pts.shape[1]),
        "num_bones": num_bones,
        "pose_dim": pose_dim,
        "train_ratio": train_ratio,
        "n_train": n_train,
        "n_test": len(test_frame_ids),
        "train_frame_ids": fitter.train_frame_ids,
        "test_frame_ids": test_frame_ids,
        "num_iters": num_iters,
        "lr": lr,
        "fit_time_seconds": fit_time,
        "overall_mean_error": float(metrics["overall_mean"]),
        "train_mean_error": float(metrics.get("train_mean", metrics["overall_mean"])),
        "test_mean_error": float(metrics.get("test_mean", -1)),
        "train_max_error": float(metrics.get("train_max", -1)),
        "test_max_error": float(metrics.get("test_max", -1)),
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nAll outputs saved to {output_dir}/")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Batch fit BoB on all sequences")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--sequences", nargs="+", default=None,
                        help="Specific sequences to process (default: all)")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="Fraction of frames for training (default: 0.8)")
    parser.add_argument("--num_iters", type=int, default=2000,
                        help="Number of fitting iterations")
    parser.add_argument("--num_bones", type=int, default=25,
                        help="Number of BoB bones")
    parser.add_argument("--opacity_threshold", type=float, default=0.5,
                        help="Opacity threshold for filtering Gaussians")
    parser.add_argument("--max_points", type=int, default=20000,
                        help="Max points per sequence (0=no limit)")
    parser.add_argument("--output_root", type=str, default="output/bob_results",
                        help="Output root directory")
    # Paths
    parser.add_argument("--dna_results_dir", type=str,
                        default="output/dna_exp_1x",
                        help="Directory with DNA training results")
    parser.add_argument("--actorshq_results_dir", type=str,
                        default="output/actorshq_exp",
                        help="Directory with ActorsHQ training results")
    parser.add_argument("--annotations_dir", type=str,
                        default="annotations",
                        help="Directory with DNA annotation .smc files")
    parser.add_argument("--actorshq_smplx_dir", type=str,
                        default="actorshq_smplx",
                        help="Directory with ActorsHQ SMPL-X JSON dirs")
    args = parser.parse_args()

    configure_visible_gpu(args.gpu)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_points = args.max_points if args.max_points > 0 else None

    # Define all sequences
    # DNA sequences: seq_id -> (annot_filename, result_dirname)
    dna_sequences = {
        "dna_0012_11": ("0012_11_annots.smc", "dna_0012_11_1x"),
        "dna_0013_01": ("0013_01_annots.smc", "dna_0013_01_1x"),
        "dna_0013_03": ("0013_03_annots.smc", "dna_0013_03_1x"),
        "dna_0013_09": ("0013_09_annots.smc", "dna_0013_09_1x"),
        "dna_0013_11": ("0013_11_annots.smc", "dna_0013_11_1x"),
        "dna_0019_08": ("0019_08_annots.smc", "dna_0019_08_1x"),
        "dna_0021_03": ("0021_03_annots.smc", "dna_0021_03_1x"),
        "dna_0023_06": ("0023_06_annots.smc", "dna_0023_06_1x"),
    }

    # ActorsHQ sequences: seq_name -> (smplx_subdir, result_dirname, start_frame)
    actorshq_sequences = {
        "actorshq_actor01_seq1": ("actorshq_actor01_seq1", "actorshq_actor01_seq1", 822),
        "actorshq_actor01_seq2": ("actorshq_actor01_seq2", "actorshq_actor01_seq2", 758),
        "actorshq_actor02_seq1": ("actorshq_actor02_seq1", "actorshq_actor02_seq1", 743),
        "actorshq_actor02_seq2": ("actorshq_actor02_seq2", "actorshq_actor02_seq2", 1013),
        "actorshq_actor04_seq2": ("actorshq_actor04_seq2", "actorshq_actor04_seq2", 717),
        "actorshq_actor05_seq1": ("actorshq_actor05_seq1", "actorshq_actor05_seq1", 870),
        "actorshq_actor06_seq1": ("actorshq_actor06_seq1", "actorshq_actor06_seq1", 916),
        "actorshq_actor06_seq2": ("actorshq_actor06_seq2", "actorshq_actor06_seq2", 206),
        "actorshq_actor07_seq1": ("actorshq_actor07_seq1", "actorshq_actor07_seq1", 966),
        "actorshq_actor08_seq1": ("actorshq_actor08_seq1", "actorshq_actor08_seq1", 858),
    }

    # Determine which sequences to process
    all_seq_names = list(dna_sequences.keys()) + list(actorshq_sequences.keys())
    if args.sequences:
        seq_to_run = args.sequences
    else:
        seq_to_run = all_seq_names

    results_summary = {}

    # Process DNA sequences
    for seq_name in seq_to_run:
        if seq_name in dna_sequences:
            annot_file, result_dir = dna_sequences[seq_name]
            params_path = os.path.join(args.dna_results_dir, result_dir, "params.npz")
            annot_path = os.path.join(args.annotations_dir, annot_file)

            if not os.path.exists(params_path):
                print(f"\nSKIP {seq_name}: params.npz not found at {params_path}")
                continue
            if not os.path.exists(annot_path):
                print(f"\nSKIP {seq_name}: annotation not found at {annot_path}")
                continue

            # Load data
            canonical_pts, target_pts, smpl_poses, pt_indices = load_dna_rendering_data(
                params_path=params_path,
                annot_path=annot_path,
                opacity_threshold=args.opacity_threshold,
                max_points=max_points,
            )

            output_dir = os.path.join(args.output_root, seq_name)
            metrics = run_fit_bob_for_sequence(
                seq_name=seq_name,
                params_path=params_path,
                canonical_pts=canonical_pts,
                target_pts=target_pts,
                smpl_poses=smpl_poses,
                output_dir=output_dir,
                train_ratio=args.train_ratio,
                num_bones=args.num_bones,
                num_iters=args.num_iters,
                max_points=max_points,
                device=device,
                point_indices=pt_indices,
            )
            results_summary[seq_name] = {
                "train_err": metrics.get("train_mean", -1),
                "test_err": metrics.get("test_mean", -1),
            }

            # Free GPU memory
            torch.cuda.empty_cache()

        elif seq_name in actorshq_sequences:
            smplx_subdir, result_dir, start_frame = actorshq_sequences[seq_name]
            params_path = os.path.join(args.actorshq_results_dir, result_dir, "params.npz")
            smplx_dir = os.path.join(args.actorshq_smplx_dir, smplx_subdir)

            if not os.path.exists(params_path):
                print(f"\nSKIP {seq_name}: params.npz not found at {params_path}")
                continue
            if not os.path.exists(smplx_dir):
                print(f"\nSKIP {seq_name}: SMPL-X dir not found at {smplx_dir}")
                continue

            # Load data with frame offset
            canonical_pts, target_pts, smpl_poses, pt_indices = load_actorshq_data_with_offset(
                params_path=params_path,
                smplx_dir=smplx_dir,
                start_frame=start_frame,
                opacity_threshold=args.opacity_threshold,
                max_points=max_points,
            )

            output_dir = os.path.join(args.output_root, seq_name)
            metrics = run_fit_bob_for_sequence(
                seq_name=seq_name,
                params_path=params_path,
                canonical_pts=canonical_pts,
                target_pts=target_pts,
                smpl_poses=smpl_poses,
                output_dir=output_dir,
                train_ratio=args.train_ratio,
                num_bones=args.num_bones,
                num_iters=args.num_iters,
                max_points=max_points,
                device=device,
                point_indices=pt_indices,
            )
            results_summary[seq_name] = {
                "train_err": metrics.get("train_mean", -1),
                "test_err": metrics.get("test_mean", -1),
            }

            torch.cuda.empty_cache()

        else:
            print(f"\nSKIP {seq_name}: unknown sequence")

    # Print summary
    print(f"\n{'='*70}")
    print("BATCH FIT BOB SUMMARY")
    print(f"{'='*70}")
    print(f"{'Sequence':<30s}  {'Train Err':>10s}  {'Test Err':>10s}")
    print("-" * 55)
    for seq_name, res in results_summary.items():
        print(f"{seq_name:<30s}  {res['train_err']:>10.6f}  {res['test_err']:>10.6f}")

    # Save summary
    summary_path = os.path.join(args.output_root, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
