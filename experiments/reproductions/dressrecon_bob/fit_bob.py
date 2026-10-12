# Adapted from the local DressRecon reproduction at revision 91b3b79.
# DressRecon copyright (c) 2023 Jeff Tan, Gengshan Yang; MIT License.
#
# Fit Composed BoB to 4D Point Clouds
#
# Given:
#   - A sequence of point clouds: (T, V, 3) where T=num_frames, V=num_vertices
#   - Corresponding SMPL poses: (T, 23, 3) or (T, 69)
#
# Architecture (ComposedPoseBoB):
#   Inner skeleton FK uses GT SMPL joint angles directly (no MLP).
#   Outer BoB learns only clothing/residual deformation.
#
# Usage:
#   python -m lab4d.nnutils.fit_bob \
#       --pointclouds /path/to/pointclouds.npy \
#       --smpl_poses /path/to/smpl_poses.npy \
#       --num_bones 25 --num_iters 2000 --lr 1e-3

import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from .bob_warp import ComposedPoseBoB
from .pose_contract import (
    SMPL_ROOTLESS_JOINTS,
    flatten_rootless_pose,
    smplx_body_to_rootless_links,
)


class BoBFitter:
    """Fit a ComposedPoseBoB model to a 4D point cloud sequence.

    Architecture (ComposedPoseBoB):
        Inner skeleton FK uses GT SMPL joint angles directly for body articulation.
        Outer BoB (pose-conditioned) learns only clothing/residual deformation.

    Args:
        canonical_pts: (V, 3) Point cloud at frame 0 (canonical/rest frame)
        target_pts: (T, V, 3) Point clouds at all frames (including frame 0)
        smpl_poses: (T, 69) or (T, 23, 3) SMPL joint angles for all frames
        num_bones (int): Number of BoB bones for clothing deformation layer
        pose_dim (int): Dimension of SMPL pose
        init_gauss_scale (float): Initial Gaussian bone scale for BoB
        delta_skin (bool): Whether to use delta skinning MLP in BoB
        device (str): Device to use
        train_ratio (float): Fraction of frames for training
        skel_type (str): Skeleton type for FK ("smpl", "human", "quad")
        skel_init_gauss_scale (float): Initial Gaussian scale for skeleton bones
    """

    def __init__(
        self,
        canonical_pts,
        target_pts,
        smpl_poses,
        num_bones=25,
        pose_dim=69,
        init_gauss_scale=0.03,
        delta_skin=True,
        device="cuda",
        train_ratio=1.0,
        skel_type="smpl",
        skel_init_gauss_scale=0.03,
    ):
        self.device = device
        self.skel_type = skel_type

        # Data: move to device
        self.canonical_pts = torch.as_tensor(
            canonical_pts, dtype=torch.float32, device=device
        )  # (V, 3)
        self.target_pts = torch.as_tensor(
            target_pts, dtype=torch.float32, device=device
        )  # (T, V, 3)
        smpl_poses = torch.as_tensor(
            smpl_poses, dtype=torch.float32, device=device
        )
        expected_joints = (
            SMPL_ROOTLESS_JOINTS
            if skel_type == "smpl"
            else pose_dim // 3
        )
        self.smpl_poses = flatten_rootless_pose(
            smpl_poses,
            expected_joints,
            label=f"{skel_type} fitter pose",
        )
        if self.smpl_poses.shape[1] != pose_dim:
            raise ValueError(
                f"pose_dim={pose_dim} does not match validated pose width "
                f"{self.smpl_poses.shape[1]}"
            )

        self.num_frames = self.target_pts.shape[0]
        self.num_verts = self.canonical_pts.shape[0]

        # Train/test split
        self.train_ratio = train_ratio
        n_train = max(2, int(self.num_frames * train_ratio))
        self.train_frame_ids = list(range(1, n_train))  # skip frame 0 (identity warp)
        self.test_frame_ids = list(range(n_train, self.num_frames))
        self.n_train = n_train

        # Rest pose = frame 0 SMPL pose
        self.rest_pose = self.smpl_poses[0:1]  # (1, 69)

        # Build model
        self.model = ComposedPoseBoB(
            skel_type=skel_type,
            num_bob_bones=num_bones,
            pose_dim=pose_dim,
            bob_init_gauss_scale=init_gauss_scale,
            skel_init_gauss_scale=skel_init_gauss_scale,
            delta_skin=delta_skin,
        ).to(device)

        skel_params = sum(p.numel() for p in self.model.skeleton.parameters())
        bob_params = sum(p.numel() for p in self.model.bob.parameters())

        print(f"BoBFitter initialized:")
        print(f"  Mode: ComposedPoseBoB (skel={skel_type}, bob_bones={num_bones})")
        print(f"  Frames: {self.num_frames}, Vertices: {self.num_verts}")
        print(f"  Train frames: {len(self.train_frame_ids)} (idx 1..{n_train-1})")
        print(f"  Test  frames: {len(self.test_frame_ids)} (idx {n_train}..{self.num_frames-1})")
        print(f"  Pose dim: {pose_dim}")
        print(f"  Skeleton params: {skel_params:,}, BoB params: {bob_params:,}")
        print(f"  Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def compute_loss(self, frame_ids):
        """Compute per-vertex L2 loss for given frame indices.

        Args:
            frame_ids: list or tensor of frame indices to compute loss for
        Returns:
            loss (scalar): Mean per-vertex L2 error
            loss_dict (Dict): Detailed loss breakdown
        """
        M = len(frame_ids)

        # canonical points: broadcast to (M, V, 3)
        xyz = self.canonical_pts[None].expand(M, -1, -1)

        # target SMPL poses for selected frames
        smpl_pose = self.smpl_poses[frame_ids]  # (M, 69)

        # rest pose: broadcast to (M, 69)
        rest_pose = self.rest_pose.expand(M, -1)

        # Forward warp: canonical -> deformed
        warped, warp_dict = self.model(xyz, smpl_pose, rest_pose=rest_pose, type="forward")

        # Target point cloud for selected frames
        target = self.target_pts[frame_ids]  # (M, V, 3)

        # Per-vertex L2 loss
        vertex_error = (warped - target).pow(2).sum(-1)  # (M, V)
        loss_l2 = vertex_error.mean()

        loss_dict = {
            "l2": loss_l2,
            "rmse": vertex_error.mean().sqrt(),
            "max_error": vertex_error.max().sqrt(),
        }

        # Optional: delta skinning regularization
        if "delta_skin" in warp_dict:
            loss_delta = warp_dict["delta_skin"].mean()
            loss_dict["delta_skin"] = loss_delta

        return loss_l2, loss_dict

    def fit(
        self,
        num_iters=2000,
        lr=1e-3,
        batch_frames=8,
        reg_delta_skin=0.01,
        log_interval=100,
        schedule_milestones=None,
    ):
        """Run optimization to fit BoB to 4D point cloud.

        Args:
            num_iters (int): Number of optimization iterations
            lr (float): Learning rate
            batch_frames (int): Number of frames per batch (0 = all frames)
            reg_delta_skin (float): Weight for delta skinning regularization
            log_interval (int): Print loss every N iterations
            schedule_milestones (list): LR decay milestones (fractions of num_iters)
        Returns:
            history (list): Per-iteration loss values
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        if schedule_milestones is None:
            schedule_milestones = [0.5, 0.75, 0.9]
        milestones = [int(m * num_iters) for m in schedule_milestones]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=0.5
        )

        # Only train on train split (skip frame 0 since warp is identity there)
        all_frame_ids = self.train_frame_ids
        if batch_frames <= 0 or batch_frames >= len(all_frame_ids):
            batch_frames = len(all_frame_ids)

        history = []
        t0 = time.time()

        for it in range(num_iters):
            optimizer.zero_grad()

            # Sample batch of frames
            if batch_frames < len(all_frame_ids):
                frame_ids = np.random.choice(all_frame_ids, batch_frames, replace=False)
            else:
                frame_ids = all_frame_ids
            frame_ids = torch.tensor(frame_ids, dtype=torch.long, device=self.device)

            # Compute loss
            loss, loss_dict = self.compute_loss(frame_ids)

            # Add regularization
            total_loss = loss
            if reg_delta_skin > 0 and "delta_skin" in loss_dict:
                total_loss = total_loss + reg_delta_skin * loss_dict["delta_skin"]

            total_loss.backward()
            optimizer.step()
            scheduler.step()

            history.append(loss.item())

            if (it + 1) % log_interval == 0 or it == 0:
                elapsed = time.time() - t0
                lr_now = optimizer.param_groups[0]["lr"]
                msg = (
                    f"[Iter {it+1:5d}/{num_iters}] "
                    f"L2={loss_dict['l2'].item():.6f}  "
                    f"RMSE={loss_dict['rmse'].item():.6f}  "
                    f"MaxErr={loss_dict['max_error'].item():.6f}  "
                    f"LR={lr_now:.2e}  "
                    f"Time={elapsed:.1f}s"
                )
                if "delta_skin" in loss_dict:
                    msg += f"  DeltaSkin={loss_dict['delta_skin'].item():.6f}"
                print(msg)

        return history

    @torch.no_grad()
    def evaluate(self):
        """Evaluate per-frame reconstruction error, with train/test split.

        Returns:
            metrics (Dict): Per-frame and overall metrics, with train/test breakdown
        """
        all_errors = []

        for t in range(self.num_frames):
            xyz = self.canonical_pts[None]  # (1, V, 3)
            smpl_pose = self.smpl_poses[t:t+1]  # (1, 69)
            rest_pose = self.rest_pose  # (1, 69)

            warped, _ = self.model(xyz, smpl_pose, rest_pose=rest_pose, type="forward")
            target = self.target_pts[t:t+1]  # (1, V, 3)

            per_vertex_error = (warped - target).pow(2).sum(-1).sqrt()  # (1, V)
            all_errors.append(per_vertex_error[0])

        all_errors = torch.stack(all_errors)  # (T, V)
        per_frame_mean = all_errors.mean(dim=1).cpu().numpy()  # (T,)
        per_frame_max = all_errors.max(dim=1).values.cpu().numpy()  # (T,)

        metrics = {
            "per_frame_mean": per_frame_mean,
            "per_frame_max": per_frame_max,
            "overall_mean": all_errors.mean().item(),
            "overall_max": all_errors.max().item(),
        }

        print(f"\nEvaluation results (all {self.num_frames} frames):")
        print(f"  Overall mean vertex error: {metrics['overall_mean']:.6f}")
        print(f"  Overall max vertex error:  {metrics['overall_max']:.6f}")
        print(f"  Per-frame mean error range: "
              f"[{per_frame_mean.min():.6f}, {per_frame_mean.max():.6f}]")

        # Train/test breakdown
        if self.test_frame_ids:
            train_ids = [0] + self.train_frame_ids
            test_ids = self.test_frame_ids
            train_err = per_frame_mean[train_ids]
            test_err = per_frame_mean[test_ids]
            metrics["train_mean"] = float(train_err.mean())
            metrics["test_mean"] = float(test_err.mean())
            metrics["train_max"] = float(per_frame_max[train_ids].max())
            metrics["test_max"] = float(per_frame_max[test_ids].max())

            print(f"\n  Train split ({len(train_ids)} frames, idx 0..{self.n_train-1}):")
            print(f"    Mean vertex error: {metrics['train_mean']:.6f}")
            print(f"    Max  vertex error: {metrics['train_max']:.6f}")
            print(f"    Per-frame range:   [{train_err.min():.6f}, {train_err.max():.6f}]")
            print(f"  Test  split ({len(test_ids)} frames, idx {self.n_train}..{self.num_frames-1}):")
            print(f"    Mean vertex error: {metrics['test_mean']:.6f}")
            print(f"    Max  vertex error: {metrics['test_max']:.6f}")
            print(f"    Per-frame range:   [{test_err.min():.6f}, {test_err.max():.6f}]")

        return metrics

    @torch.no_grad()
    def get_warped_sequence(self):
        """Get the full warped point cloud sequence.

        Returns:
            warped_pts: (T, V, 3) numpy array of warped point clouds
        """
        warped_all = []
        for t in range(self.num_frames):
            xyz = self.canonical_pts[None]  # (1, V, 3)
            smpl_pose = self.smpl_poses[t:t+1]
            rest_pose = self.rest_pose
            warped, _ = self.model(xyz, smpl_pose, rest_pose=rest_pose, type="forward")
            warped_all.append(warped[0])
        return torch.stack(warped_all).cpu().numpy()  # (T, V, 3)

    @torch.no_grad()
    def get_skinning_weights(self):
        """Get per-vertex BoB skinning weights in canonical space.

        Returns:
            weights: (V, B) numpy array of BoB skinning weights
        """
        xyz = self.canonical_pts[None]  # (1, V, 3)
        rest_pose = self.rest_pose
        # Use rest pose as both target and rest -> identity warp, but we get skin_prob
        _, warp_dict = self.model(xyz, rest_pose, rest_pose=rest_pose, type="forward")
        return warp_dict["skin_prob"][0].cpu().numpy()  # (V, B)

    @torch.no_grad()
    def get_skel_skinning_weights(self):
        """Get per-vertex skeleton skinning weights.

        Returns:
            weights: (V, K) numpy array of skeleton skinning weights
        """
        xyz = self.canonical_pts[None]  # (1, V, 3)
        rest_pose = self.rest_pose
        _, warp_dict = self.model(xyz, rest_pose, rest_pose=rest_pose, type="forward")
        return warp_dict["skel_skin_prob"][0].cpu().numpy()  # (V, K)

    def save(self, path):
        """Save model and fitting metadata.

        Args:
            path (str): Output file path (.pt)
        """
        save_dict = {
            "model_state_dict": self.model.state_dict(),
            "canonical_pts": self.canonical_pts.cpu(),
            "rest_pose": self.rest_pose.cpu(),
            "num_bones": self.model.num_se3,
            "pose_dim": self.model.articulation.pose_dim,
            "skel_type": self.skel_type,
        }
        torch.save(save_dict, path)
        print(f"Model saved to {path}")

    def load(self, path):
        """Load model from checkpoint.

        Args:
            path (str): Input file path (.pt)
        """
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        print(f"Model loaded from {path}")


def load_dna_rendering_data(
    params_path,
    annot_path,
    opacity_threshold=0.5,
    max_points=None,
):
    """Load 4D point cloud and SMPL-X poses from DNA-Rendering / Dynamic3DGaussians.

    Args:
        params_path: Path to params.npz (contains means3D, logit_opacities, etc.)
        annot_path: Path to annotation .smc file (HDF5, contains SMPLx/fullpose)
        opacity_threshold: Filter Gaussians with opacity below this (0 = keep all)
        max_points: Further subsample to this many points (None = keep all after filtering)

    Returns:
        canonical_pts: (V, 3) First-frame point cloud
        target_pts: (T, V, 3) Full point cloud sequence
        smpl_poses: (T, pose_dim) SMPL-X body poses
        point_indices: Indices of selected points (for visualization)
    """
    import h5py

    # Load Gaussian parameters
    print(f"Loading point clouds from {params_path} ...")
    params = np.load(params_path)
    means3d = params["means3D"]  # (T, V_all, 3)

    # Filter by opacity
    point_indices = np.arange(means3d.shape[1])
    if opacity_threshold > 0 and "logit_opacities" in params:
        logit_op = params["logit_opacities"].squeeze()  # (V_all,)
        opacity = 1.0 / (1.0 + np.exp(-logit_op))
        mask = opacity > opacity_threshold
        means3d = means3d[:, mask]
        point_indices = point_indices[mask]
        print(f"  Opacity filter (>{opacity_threshold}): {mask.sum()}/{len(mask)} points kept")

    # Further subsample if needed
    if max_points is not None and means3d.shape[1] > max_points:
        idx = np.random.choice(means3d.shape[1], max_points, replace=False)
        idx.sort()
        means3d = means3d[:, idx]
        point_indices = point_indices[idx]
        print(f"  Subsampled to {max_points} points")

    canonical_pts = means3d[0].astype(np.float32)   # (V, 3)
    target_pts = means3d.astype(np.float32)          # (T, V, 3)

    # Load SMPL-X poses
    print(f"Loading SMPL-X poses from {annot_path} ...")
    with h5py.File(annot_path, "r") as f:
        fullpose = f["SMPLx/fullpose"][:]  # (T, 55, 3)
        transl = f["SMPLx/transl"][:]      # (T, 3)

    body_pose = smplx_body_to_rootless_links(fullpose[:, 1:22, :])
    print("  Using SMPL joints 1..23 (root excluded), pose_dim=69")
    print(f"  Frames: {fullpose.shape[0]}, Translation range: [{transl.min():.3f}, {transl.max():.3f}]")

    # Sanity check frame count
    assert means3d.shape[0] == fullpose.shape[0], (
        f"Frame count mismatch: pointclouds={means3d.shape[0]}, poses={fullpose.shape[0]}"
    )

    return canonical_pts, target_pts, body_pose.astype(np.float32), point_indices


def load_actorshq_data(
    params_path,
    smplx_dir,
    opacity_threshold=0.5,
    max_points=None,
    frame_stride=11,
    pose_slice=(3, 66),
    include_global_rot=False,
):
    """Load 4D point cloud and SMPL-X poses from ActorsHQ dataset.

    ActorsHQ stores SMPL-X as per-frame JSON files, each containing:
      poses (1, 87): global_orient(3) + body_pose(63) + jaw(3) + eyes(6) + hands(12)
      Rh (1, 3): global/pelvis rotation (stored separately from poses)
      Th (1, 3): global translation

    NOTE: In ActorsHQ, poses[:3] (global orient) is always zero.
    The actual global rotation is in the separate 'Rh' field.
    Global/pelvis rotation is not part of the rootless 23-link FK input.

    Args:
        params_path: Path to params.npz from Dynamic3DGaussians training
        smplx_dir: Path to directory with per-frame SMPL-X JSON files
        opacity_threshold: Filter Gaussians with opacity below this
        max_points: Subsample to this many points (None = keep all)
        frame_stride: Stride between original frame indices (default 11 for 4x)
        pose_slice: (start, end) slice of 'poses' vector to use as body pose
        include_global_rot: Unsupported legacy option. Root rotation must stay
            separate from the rootless FK pose.

    Returns:
        canonical_pts: (V, 3) First-frame point cloud
        target_pts: (T, V, 3) Full point cloud sequence
        smpl_poses: (T, 69) rootless SMPL link rotations
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
    print(f"Loading SMPL-X poses from {smplx_dir} (stride={frame_stride}) ...")
    frame_indices = list(range(0, T * frame_stride, frame_stride))

    poses_list = []
    rh_list = []
    pstart, pend = pose_slice
    for i, fidx in enumerate(frame_indices):
        json_path = os.path.join(smplx_dir, f"{fidx:06d}.json")
        if not os.path.exists(json_path):
            print(f"  WARNING: Missing {json_path}, using zeros")
            poses_list.append(np.zeros(pend - pstart, dtype=np.float32))
            rh_list.append(np.zeros(3, dtype=np.float32))
            continue
        with open(json_path) as f:
            data = json_mod.load(f)
        full_pose = np.array(data[0]["poses"]).squeeze()  # (87,)
        body_pose = full_pose[pstart:pend]  # body joints only
        poses_list.append(body_pose.astype(np.float32))
        rh = np.array(data[0]["Rh"]).squeeze()  # (3,)
        rh_list.append(rh.astype(np.float32))

    body_poses = np.stack(poses_list)  # (T, 63)
    rh_all = np.stack(rh_list)         # (T, 3)

    if include_global_rot:
        raise ValueError(
            "global/pelvis rotation must remain separate from the "
            "rootless 23-link FK pose"
        )
    smpl_poses = smplx_body_to_rootless_links(body_poses)
    print("  Global rotation kept separate; rootless pose_dim=69")

    print(f"  Loaded {len(poses_list)} frames, pose_dim={smpl_poses.shape[1]}")
    print(f"  Pose range: [{smpl_poses.min():.4f}, {smpl_poses.max():.4f}]")

    assert T == smpl_poses.shape[0], (
        f"Frame count mismatch: pointclouds={T}, poses={smpl_poses.shape[0]}"
    )

    return canonical_pts, target_pts, smpl_poses, point_indices


def generate_synthetic_data(num_frames=30, num_verts=500, device="cpu"):
    """Generate synthetic 4D point cloud data for testing.

    Creates a unit sphere that deforms based on synthetic SMPL-like poses.

    Returns:
        canonical_pts: (V, 3) Rest pose point cloud
        target_pts: (T, V, 3) Point cloud sequence
        smpl_poses: (T, 69) Synthetic SMPL poses
    """
    # Generate sphere vertices
    phi = np.random.uniform(0, np.pi, num_verts)
    theta = np.random.uniform(0, 2 * np.pi, num_verts)
    r = 0.1
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    canonical_pts = np.stack([x, y, z], axis=1).astype(np.float32)  # (V, 3)

    # Generate synthetic poses (gradually increasing rotation)
    smpl_poses = np.zeros((num_frames, 69), dtype=np.float32)
    for t in range(num_frames):
        phase = t / num_frames * 2 * np.pi
        # A few joints rotate sinusoidally
        smpl_poses[t, 0:3] = [0.1 * np.sin(phase), 0, 0]  # SMPL joint 1
        smpl_poses[t, 3:6] = [0, 0.2 * np.sin(phase), 0]  # SMPL joint 2
        smpl_poses[t, 9:12] = [0, 0, 0.15 * np.cos(phase)]  # SMPL joint 4

    # Generate target point clouds with simple deformation
    target_pts = np.zeros((num_frames, num_verts, 3), dtype=np.float32)
    for t in range(num_frames):
        phase = t / num_frames * 2 * np.pi
        pts = canonical_pts.copy()
        # Apply pose-dependent deformation
        # Upper hemisphere bends one way, lower the other
        upper = pts[:, 2] > 0
        pts[upper, 0] += 0.02 * np.sin(phase)
        pts[~upper, 0] -= 0.01 * np.sin(phase)
        pts[:, 1] += 0.015 * np.sin(phase) * pts[:, 2]
        target_pts[t] = pts

    return canonical_pts, target_pts, smpl_poses


def parse_args():
    parser = argparse.ArgumentParser(description="Fit BoB to 4D point clouds")
    # Data
    parser.add_argument("--pointclouds", type=str, default="",
                        help="Path to point cloud sequence (T,V,3) .npy file")
    parser.add_argument("--smpl_poses", type=str, default="",
                        help="Path to SMPL poses (T,69) or (T,23,3) .npy file")
    # DNA-Rendering data format
    parser.add_argument("--dna_params", type=str, default="",
                        help="Path to Dynamic3DGaussians params.npz")
    parser.add_argument("--dna_annot", type=str, default="",
                        help="Path to DNA-Rendering annotation .smc file")
    # ActorsHQ data format
    parser.add_argument("--actorshq_params", type=str, default="",
                        help="Path to ActorsHQ Dynamic3DGaussians params.npz")
    parser.add_argument("--actorshq_smplx_dir", type=str, default="",
                        help="Path to ActorsHQ per-frame SMPL-X JSON directory")
    parser.add_argument("--frame_stride", type=int, default=11,
                        help="Frame stride for ActorsHQ (original frames per training step)")
    parser.add_argument("--no_global_rot", action="store_true",
                        help="Deprecated: root rotation is always kept separate")
    # Common
    parser.add_argument("--opacity_threshold", type=float, default=0.5,
                        help="Opacity threshold for filtering Gaussians")
    parser.add_argument("--max_points", type=int, default=0,
                        help="Max points to use (0=all after opacity filter)")
    # Model
    parser.add_argument("--num_bones", type=int, default=25, help="Number of BoB bones")
    parser.add_argument("--pose_dim", type=int, default=0,
                        help="SMPL pose dimension (0=auto from data)")
    parser.add_argument("--init_gauss_scale", type=float, default=0.1,
                        help="Initial Gaussian bone scale")
    parser.add_argument("--no_delta_skin", action="store_true",
                        help="Disable delta skinning MLP")
    # Optimization
    parser.add_argument("--num_iters", type=int, default=2000, help="Number of iterations")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_frames", type=int, default=8,
                        help="Frames per batch (0=all)")
    parser.add_argument("--reg_delta_skin", type=float, default=0.01,
                        help="Delta skinning regularization weight")
    # Train/test split
    parser.add_argument("--train_ratio", type=float, default=1.0,
                        help="Fraction of frames for training (e.g. 0.8 = first 80%% train, last 20%% test)")
    # Skeleton FK layer
    parser.add_argument("--skel_type", type=str, default="smpl",
                        choices=["smpl", "human", "quad"],
                        help="Skeleton type for FK layer")
    parser.add_argument("--skel_init_gauss_scale", type=float, default=0.03,
                        help="Initial Gaussian scale for skeleton bones")
    # Output
    parser.add_argument("--output_dir", type=str, default="bob_output",
                        help="Output directory")
    parser.add_argument("--use_synthetic", action="store_true",
                        help="Use synthetic data for testing")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load or generate data
    if args.use_synthetic:
        print("Using synthetic test data...")
        canonical_pts, target_pts, smpl_poses = generate_synthetic_data(
            num_frames=30, num_verts=500
        )
    elif args.dna_params and args.dna_annot:
        # DNA-Rendering + Dynamic3DGaussians format
        canonical_pts, target_pts, smpl_poses, pt_indices = load_dna_rendering_data(
            params_path=args.dna_params,
            annot_path=args.dna_annot,
            opacity_threshold=args.opacity_threshold,
            max_points=args.max_points if args.max_points > 0 else None,
        )
    elif args.actorshq_params and args.actorshq_smplx_dir:
        # ActorsHQ format
        canonical_pts, target_pts, smpl_poses, pt_indices = load_actorshq_data(
            params_path=args.actorshq_params,
            smplx_dir=args.actorshq_smplx_dir,
            opacity_threshold=args.opacity_threshold,
            max_points=args.max_points if args.max_points > 0 else None,
            frame_stride=args.frame_stride,
            include_global_rot=False,
        )
    else:
        assert args.pointclouds, "--pointclouds required (or use --use_synthetic / --dna_params)"
        assert args.smpl_poses, "--smpl_poses required (or use --use_synthetic / --dna_annot)"
        target_pts = np.load(args.pointclouds)  # (T, V, 3)
        smpl_poses = np.load(args.smpl_poses)  # (T, 69) or (T, 23, 3)
        canonical_pts = target_pts[0]  # Frame 0 = canonical

    # Auto-detect pose_dim from data
    pose_dim = args.pose_dim if args.pose_dim > 0 else smpl_poses.reshape(smpl_poses.shape[0], -1).shape[1]

    print(f"Data loaded: pointclouds={target_pts.shape}, smpl_poses={smpl_poses.shape}, pose_dim={pose_dim}")

    # Create fitter
    fitter = BoBFitter(
        canonical_pts=canonical_pts,
        target_pts=target_pts,
        smpl_poses=smpl_poses,
        num_bones=args.num_bones,
        pose_dim=pose_dim,
        init_gauss_scale=args.init_gauss_scale,
        delta_skin=not args.no_delta_skin,
        device=device,
        train_ratio=args.train_ratio,
        skel_type=args.skel_type,
        skel_init_gauss_scale=args.skel_init_gauss_scale,
    )

    # Fit
    print(f"\nStarting optimization: {args.num_iters} iterations, lr={args.lr}")
    history = fitter.fit(
        num_iters=args.num_iters,
        lr=args.lr,
        batch_frames=args.batch_frames,
        reg_delta_skin=args.reg_delta_skin,
    )

    # Evaluate
    metrics = fitter.evaluate()

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    fitter.save(os.path.join(args.output_dir, "bob_model.pt"))

    # Save loss history
    np.save(os.path.join(args.output_dir, "loss_history.npy"), np.array(history))

    # Save warped sequence
    warped_seq = fitter.get_warped_sequence()
    np.save(os.path.join(args.output_dir, "warped_sequence.npy"), warped_seq)

    # Save target (GT) sequence (filtered points, same order as warped)
    np.save(os.path.join(args.output_dir, "target_sequence.npy"), target_pts.astype(np.float32))

    # Save skinning weights
    skin_weights = fitter.get_skinning_weights()
    np.save(os.path.join(args.output_dir, "skinning_weights.npy"), skin_weights)

    # Save skeleton skinning weights
    skel_skin_weights = fitter.get_skel_skinning_weights()
    np.save(os.path.join(args.output_dir, "skel_skinning_weights.npy"), skel_skin_weights)

    # Save metrics
    np.save(os.path.join(args.output_dir, "per_frame_error.npy"), metrics["per_frame_mean"])

    # Save point_indices (for matching back to original params.npz)
    if 'pt_indices' in dir():
        np.save(os.path.join(args.output_dir, "point_indices.npy"), pt_indices)

    # Save metadata for reproducibility and visualization
    import json
    meta = {
        "num_frames": int(target_pts.shape[0]),
        "num_points": int(target_pts.shape[1]),
        "num_bones": args.num_bones,
        "pose_dim": pose_dim,
        "train_ratio": args.train_ratio,
        "n_train": fitter.n_train,
        "opacity_threshold": args.opacity_threshold,
        "num_iters": args.num_iters,
        "lr": args.lr,
        "delta_skin": not args.no_delta_skin,
        "init_gauss_scale": args.init_gauss_scale,
        "skel_type": args.skel_type,
        "overall_mean_error": float(metrics["overall_mean"]),
        "train_mean_error": float(metrics.get("train_mean", metrics["overall_mean"])),
        "test_mean_error": float(metrics.get("test_mean", -1)),
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nAll outputs saved to {args.output_dir}/")
    print(f"  bob_model.pt         - Model checkpoint")
    print(f"  loss_history.npy     - Training loss curve")
    print(f"  warped_sequence.npy  - Reconstructed point cloud sequence ({warped_seq.shape})")
    print(f"  target_sequence.npy  - GT filtered point cloud sequence ({target_pts.shape})")
    print(f"  skinning_weights.npy - Per-vertex BoB skinning weights ({skin_weights.shape})")
    print(f"  skel_skinning_weights.npy - Per-vertex skeleton skinning weights ({skel_skin_weights.shape})")
    print(f"  per_frame_error.npy  - Per-frame mean vertex error")
    print(f"  metadata.json        - Fitting configuration and metrics")
