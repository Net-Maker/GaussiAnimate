# Copyright (c) 2025 Jeff Tan, Gengshan Yang, Carnegie Mellon University.
# Pose-Conditioned Bag of Bones (BoB) Module
#
# SMPL pose -> MLP -> per-bone transforms -> Gaussian skinning -> DQ blend -> warped points
#
# Unlike the original ArticulationFlatMLP which is conditioned on time (frame_id),
# this module is conditioned purely on SMPL joint angles.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import trimesh

from lab4d.nnutils.base import BaseMLP, ScaleLayer
from lab4d.nnutils.skinning import SkinningField
from lab4d.utils.quat_transform import (
    quaternion_translation_to_dual_quaternion,
    dual_quaternion_mul,
    dual_quaternion_inverse,
    dual_quaternion_to_se3,
)
from lab4d.utils.geom_utils import dual_quaternion_skinning
from lab4d.utils.transforms import get_xyz_bone_distance
from lab4d.utils.skel_utils import (
    get_predefined_skeleton,
    rest_joints_to_local,
    fk_se3,
    shift_joints_to_bones_dq,
    apply_root_offset,
)

from .pose_contract import flatten_rootless_pose, reshape_rootless_pose
from .rotation import axis_angle_to_quaternion


class ArticulationPoseMLP(nn.Module):
    """SMPL-pose-conditioned bag-of-bones articulation.

    Maps SMPL pose (M, pose_dim) -> per-bone dual quaternion transforms ((M,B,4), (M,B,4)).

    Unlike ArticulationFlatMLP which uses TimeEmbedding(frame_id) as input,
    this module takes SMPL joint angles directly as input.

    Args:
        num_se3 (int): Number of bones
        pose_dim (int): Dimension of input pose vector (e.g. 69 for 23 SMPL joints * 3)
        D (int): Number of linear layers in the pose encoder
        W (int): Number of hidden units in each MLP layer
    """

    def __init__(self, num_se3=25, pose_dim=69, D=5, W=256):
        super().__init__()
        self.num_se3 = num_se3
        self.pose_dim = pose_dim
        self.edges = None  # BoB: no skeleton topology

        activation = nn.ReLU(True)

        # Pose encoder: SMPL pose -> hidden features
        self.pose_encoder = BaseMLP(
            D=D,
            W=W,
            in_channels=pose_dim,
            out_channels=W,
            skips=[D // 2] if D > 2 else [],
            activation=activation,
            final_act=True,
        )

        # Output heads: hidden features -> per-bone (so3, trans)
        # Same structure as ArticulationFlatMLP
        self.trans = nn.Sequential(
            nn.Linear(W, W // 2),
            activation,
            nn.Linear(W // 2, 3 * num_se3),
            ScaleLayer(0.1),
        )
        self.so3 = nn.Sequential(
            nn.Linear(W, W // 2),
            activation,
            nn.Linear(W // 2, 3 * num_se3),
        )

    def forward(self, smpl_pose):
        """
        Args:
            smpl_pose: (M, 69) or (M, 23, 3) SMPL joint angles
        Returns:
            dq: ((M, B, 4), (M, B, 4)) Per-bone dual quaternion transforms
        """
        if self.pose_dim % 3:
            raise ValueError(
                f"pose_dim must be divisible by 3; received {self.pose_dim}"
            )
        pose = flatten_rootless_pose(
            smpl_pose,
            self.pose_dim // 3,
            label="BoB conditioning pose",
        )

        feat = self.pose_encoder(pose)  # (M, W)
        trans = self.trans(feat).reshape(-1, self.num_se3, 3)  # (M, B, 3)
        so3 = self.so3(feat).reshape(-1, self.num_se3, 3)  # (M, B, 3)

        # Convert to dual quaternion representation
        qr = axis_angle_to_quaternion(so3)
        dq = quaternion_translation_to_dual_quaternion(qr, trans)
        return dq

    def get_vals(self, smpl_pose):
        """Compute bone transforms for given SMPL pose.

        Args:
            smpl_pose: (M, 69) or (M, 23, 3) SMPL joint angles
        Returns:
            dq: ((M, B, 4), (M, B, 4)) Per-bone dual quaternion transforms
        """
        return self.forward(smpl_pose)

    def get_rest_vals(self, batch_size=1, device=None):
        """Compute bone transforms at rest (zero) SMPL pose.

        Args:
            batch_size (int): Batch size
            device: Torch device
        Returns:
            dq: ((M, B, 4), (M, B, 4)) Per-bone dual quaternion transforms at rest
        """
        if device is None:
            device = next(self.parameters()).device
        rest_pose = torch.zeros(batch_size, self.pose_dim, device=device)
        return self.forward(rest_pose)


class PoseConditionedBoB(nn.Module):
    """Bag of Bones (BoB) warping conditioned on SMPL pose.

    Architecture:
        1. ArticulationPoseMLP: SMPL pose -> per-bone dual quaternion transforms
        2. SkinningField: 3D point positions + bone transforms -> Gaussian skinning weights
        3. DQ Skinning: blend bone transforms by skinning weights -> warped points

    Forward warp (canonical -> deformed):
        se3 = bone_transforms(target_pose) * inverse(bone_transforms(rest_pose))
        skinning weights computed in canonical space (using rest bone positions)
        warped_xyz = DQ_skinning(se3, xyz, skin_prob)

    Backward warp (deformed -> canonical):
        se3 = bone_transforms(rest_pose) * inverse(bone_transforms(target_pose))
        skinning weights computed in deformed space (using target bone positions)
        warped_xyz = DQ_skinning(se3, xyz, skin_prob)

    Args:
        num_se3 (int): Number of bones
        pose_dim (int): Dimension of input pose vector (69 for SMPL)
        D (int): Number of linear layers in the pose encoder
        W (int): Number of hidden units in each MLP layer
        init_gauss_scale (float): Initial scale/variance of Gaussian bones
        delta_skin (bool): Whether to apply learned MLP delta on top of Gaussian skinning
    """

    def __init__(
        self,
        num_se3=25,
        pose_dim=69,
        D=5,
        W=256,
        init_gauss_scale=0.03,
        delta_skin=True,
    ):
        super().__init__()
        self.num_se3 = num_se3

        # Articulation: SMPL pose -> bone transforms
        self.articulation = ArticulationPoseMLP(
            num_se3=num_se3,
            pose_dim=pose_dim,
            D=D,
            W=W,
        )

        # Skinning field: computes Gaussian skinning weights
        # NOTE: frame_info is not used inside SkinningField, safe to pass None
        self.skinning_model = SkinningField(
            num_se3,
            frame_info=None,
            num_inst=1,
            init_scale=init_gauss_scale,
            delta_skin=delta_skin,
            symm_idx=None,
        )

        # Beta: transparency for bone density rendering
        beta = torch.tensor([0.01])
        self.logibeta = nn.Parameter(-beta.log())

    def forward(self, xyz, smpl_pose, rest_pose=None, type="forward"):
        """Warp 3D points conditioned on SMPL pose.

        Args:
            xyz: (M, N, D, 3) or (M, N, 3) Points in space.
                For "forward": points in canonical space.
                For "backward": points in deformed (posed) space.
            smpl_pose: (M, 69) or (M, 23, 3) Target SMPL joint angles
            rest_pose: (M, 69) or (M, 23, 3) or None. Rest SMPL pose that defines
                the canonical frame. If None, uses zero pose.
            type (str): "forward" (canonical -> deformed) or
                        "backward" (deformed -> canonical)
        Returns:
            out: Same shape as xyz, warped points
            warp_dict (Dict): Auxiliary outputs with keys:
                "skin_prob": (M,...,B) skinning weights
                "delta_skin": (M,...,1) delta skinning regularization
                "t_articulation": ((M,B,4),(M,B,4)) target bone transforms
                "rest_articulation": ((M,B,4),(M,B,4)) rest bone transforms
        """
        # Handle input shape: ensure (M, N, D, 3)
        squeeze_d = False
        if xyz.ndim == 3:
            xyz = xyz.unsqueeze(2)  # (M, N, 1, 3)
            squeeze_d = True

        M = xyz.shape[0]

        # Step 1: Get bone transforms from SMPL pose
        t_articulation = self.articulation.get_vals(smpl_pose)  # ((M,B,4), (M,B,4))
        if rest_pose is not None:
            rest_articulation = self.articulation.get_vals(rest_pose)
        else:
            rest_articulation = self.articulation.get_rest_vals(M, xyz.device)

        # Step 2: Compute relative per-bone SE3 transform
        if type == "backward":
            # deformed -> canonical: rest * inv(target)
            se3 = dual_quaternion_mul(
                rest_articulation, dual_quaternion_inverse(t_articulation)
            )
            # Skinning weights computed in deformed space
            articulation = t_articulation
        elif type == "forward":
            # canonical -> deformed: target * inv(rest)
            se3 = dual_quaternion_mul(
                t_articulation, dual_quaternion_inverse(rest_articulation)
            )
            # Skinning weights computed in canonical space
            articulation = rest_articulation
        else:
            raise ValueError(f"Unknown warp type: {type}")

        # Step 3: Compute Gaussian skinning weights
        # Expand articulation for SkinningField: (M, 1, 1, B, 4)
        articulation_expanded = (
            articulation[0][:, None, None, :, :],
            articulation[1][:, None, None, :, :],
        )
        skin, delta_skin = self.skinning_model(
            xyz, articulation_expanded, None, None
        )
        skin_prob = skin.softmax(-1)  # (M, N, D, B)

        # Step 4: Apply dual quaternion skinning
        out, dual_quat = dual_quaternion_skinning(se3, xyz, skin_prob)

        # Restore original shape
        if squeeze_d:
            out = out.squeeze(2)
            skin_prob = skin_prob.squeeze(2)

        # Collect auxiliary outputs
        warp_dict = {
            "skin_prob": skin_prob,
            "dual_quat": dual_quat,
            "t_articulation": t_articulation,
            "rest_articulation": rest_articulation,
        }
        if delta_skin is not None:
            warp_dict["delta_skin"] = delta_skin.pow(2).mean(-1, keepdims=True)

        return out, warp_dict

    def get_gauss_density(self, xyz, smpl_pose=None):
        """Compute Gaussian bone density at given points.

        Args:
            xyz: (N, 3) Points in object space
            smpl_pose: (1, 69) SMPL pose. If None, use rest pose.
        Returns:
            density: (N, 1) Volumetric density of bones at each point
        """
        if smpl_pose is None:
            bone2obj = self.articulation.get_rest_vals(1, xyz.device)
        else:
            bone2obj = self.articulation.get_vals(smpl_pose)

        dist2 = get_xyz_bone_distance(xyz, bone2obj)  # (N, B)
        dist2 = dist2 / (0.01) ** 2  # normalize by sphere radius

        score = (-0.5 * dist2).exp()  # (N, B)
        mix_prob = F.softmax(-dist2, -1)  # (N, B)
        density = (score * mix_prob).sum(-1, keepdim=True)  # (N, 1)
        return density

    def get_gauss_sdf(self, xyz, smpl_pose=None, bias=0.0):
        """Compute signed distance to Gaussian bones.

        Args:
            xyz: (N, 3) Points in object space
            smpl_pose: (1, 69) SMPL pose. If None, use rest pose.
            bias (float): Value to add to signed distances
        Returns:
            sdf: (N,) Signed distance, -inf to +inf
        """
        density = self.get_gauss_density(xyz, smpl_pose)
        density = density.clamp(1e-6, 1 - 1e-6)
        sdf = -density.logit()  # (N, 1)
        sdf = sdf.squeeze(-1) + bias
        return sdf

    def get_gauss_vis(self, smpl_pose=None):
        """Visualize Gaussian bones as a mesh.

        Args:
            smpl_pose: (1, 69) SMPL pose. If None, use rest pose.
        Returns:
            mesh (Trimesh): Gaussian bone visualization
        """
        if smpl_pose is None:
            articulation = self.articulation.get_rest_vals(1)
        else:
            articulation = self.articulation.get_vals(smpl_pose)
        # Remove batch dim: (1, B, 4) -> (B, 4)
        articulation = (articulation[0][0], articulation[1][0])
        mesh = self.skinning_model.draw_gaussian(
            articulation, edges=None, show_joints=False
        )
        return mesh


class SkeletonFK(nn.Module):
    """Direct forward kinematics from GT SMPL joint angles.

    No time-conditioned MLP needed. Uses learnable bone lengths and Gaussian
    skinning weights. Designed as the body articulation layer in ComposedPoseBoB.

    Args:
        skel_type (str): Skeleton type ("smpl", "human", "quad")
        init_gauss_scale (float): Initial Gaussian bone scale for skinning
        delta_skin (bool): Whether to use delta skinning MLP for the skeleton
    """

    def __init__(self, skel_type="smpl", init_gauss_scale=0.03, delta_skin=False):
        super().__init__()
        rest_joints, edges, symm_idx = get_predefined_skeleton(skel_type)
        self.register_buffer("rest_joints", rest_joints)  # (K, 3)
        self.edges = edges
        self.symm_idx = symm_idx
        self.num_se3 = len(rest_joints)

        # Learnable bone length scaling
        self.log_bone_len = nn.Parameter(torch.zeros(self.num_se3))
        self.logscale = nn.Parameter(torch.zeros(1))

        # Learnable root offset
        self.shift = nn.Parameter(torch.zeros(3))
        self.register_buffer("orient", torch.tensor([1.0, 0.0, 0.0, 0.0]))

        # Skinning field (Gaussian-based)
        self.skinning_model = SkinningField(
            self.num_se3,
            frame_info=None,
            num_inst=1,
            init_scale=init_gauss_scale,
            delta_skin=delta_skin,
            symm_idx=symm_idx,
        )

    def compute_rel_rest_joints(self):
        """Compute relative rest joints with learnable bone lengths.

        Returns:
            rel: (K, 3) Local joint translations scaled by bone lengths
        """
        rel = rest_joints_to_local(self.rest_joints, self.edges)
        bone_len = (self.log_bone_len + self.logscale).exp()
        bone_len = (bone_len + bone_len[self.symm_idx]) / 2  # enforce symmetry
        rel = rel * bone_len[:, None]
        return rel

    def compute_articulation(self, so3):
        """Run forward kinematics with given joint angles.

        Args:
            so3: (M, K, 3) Joint angles (axis-angle representation)
        Returns:
            dq: ((M, K, 4), (M, K, 4)) Per-bone transforms as dual quaternions
        """
        if (
            so3.ndim != 3
            or tuple(so3.shape[1:]) != (self.num_se3, 3)
        ):
            raise ValueError(
                "skeleton axis angles must have exact shape "
                f"[batch, {self.num_se3}, 3]; "
                f"received {tuple(so3.shape)}"
            )
        M = so3.shape[0]
        local_rest_joints = self.compute_rel_rest_joints()  # (K, 3)
        local_rest_joints = local_rest_joints[None].expand(M, -1, -1)  # (M, K, 3)

        dq = fk_se3(local_rest_joints, so3, self.edges)
        dq = shift_joints_to_bones_dq(dq, self.edges)
        dq = apply_root_offset(dq, self.shift, self.orient)
        return dq


class ComposedPoseBoB(nn.Module):
    """Two-layer warping: Skeleton FK (GT SMPL) + Pose-Conditioned BoB (clothing).

    Architecture:
        Inner layer (skeleton): GT SMPL joint angles -> FK -> body bone transforms
            No time MLP. Joint angles come directly from GT SMPL data.
        Outer layer (BoB): SMPL pose -> MLP -> clothing bone transforms
            Pose-conditioned, learns residual clothing deformation.

    Forward warp (canonical -> deformed):
        1. BoB forward: canonical -> clothing-deformed (at mean body pose)
        2. Skeleton forward: clothing-deformed -> final posed space

    Backward warp (deformed -> canonical):
        1. Skeleton backward: posed -> clothing-deformed
        2. BoB backward: clothing-deformed -> canonical

    Args:
        skel_type (str): Skeleton type ("smpl", "human", "quad")
        num_bob_bones (int): Number of BoB bones for clothing deformation
        pose_dim (int): Dimension of input SMPL pose vector
        bob_D (int): Number of MLP layers in BoB articulation
        bob_W (int): Hidden dimension of BoB articulation MLP
        bob_init_gauss_scale (float): Initial Gaussian scale for BoB bones
        skel_init_gauss_scale (float): Initial Gaussian scale for skeleton bones
        delta_skin (bool): Whether to use delta skinning MLP in BoB
        skel_delta_skin (bool): Whether to use delta skinning MLP in skeleton
    """

    def __init__(
        self,
        skel_type="smpl",
        num_bob_bones=25,
        pose_dim=69,
        bob_D=5,
        bob_W=256,
        bob_init_gauss_scale=0.03,
        skel_init_gauss_scale=0.03,
        delta_skin=True,
        skel_delta_skin=False,
    ):
        super().__init__()
        self.skel_type = skel_type
        self._pose_dim = pose_dim

        # Inner skeleton layer: FK from GT SMPL joint angles
        self.skeleton = SkeletonFK(
            skel_type, init_gauss_scale=skel_init_gauss_scale, delta_skin=skel_delta_skin
        )
        self.num_skel_joints = self.skeleton.num_se3  # 23 for SMPL
        expected_pose_dim = self.num_skel_joints * 3
        if pose_dim != expected_pose_dim:
            raise ValueError(
                f"{skel_type} rootless skeleton has "
                f"{self.num_skel_joints} links and requires "
                f"pose_dim={expected_pose_dim}; received {pose_dim}"
            )

        # Outer BoB layer: pose-conditioned clothing deformation
        self.bob = PoseConditionedBoB(
            num_se3=num_bob_bones,
            pose_dim=pose_dim,
            D=bob_D,
            W=bob_W,
            init_gauss_scale=bob_init_gauss_scale,
            delta_skin=delta_skin,
        )
        # For compatibility with BoBFitter.save
        self.num_se3 = num_bob_bones

    @property
    def articulation(self):
        """Access BoB articulation for compatibility with save/load."""
        return self.bob.articulation

    def _parse_skel_so3(self, pose):
        """Parse one axis angle per rootless skeleton link.

        Args:
            pose: (M, pose_dim) Full pose vector
        Returns:
            so3: (M, num_skel_joints, 3) Joint angles for skeleton FK
        """
        return reshape_rootless_pose(
            pose,
            self.num_skel_joints,
            label=f"{self.skel_type} rootless FK pose",
        )

    def forward(self, xyz, smpl_pose, rest_pose=None, type="forward"):
        """Warp 3D points with composed skeleton + BoB.

        Args:
            xyz: (M, N, 3) Points. For "forward": canonical space.
                For "backward": posed/deformed space.
            smpl_pose: (M, pose_dim) Target SMPL joint angles
            rest_pose: (M, pose_dim) or None. Rest SMPL pose defining canonical.
            type (str): "forward" (canonical -> deformed) or "backward"
        Returns:
            out: (M, N, 3) Warped points
            warp_dict (Dict): Auxiliary outputs
        """
        # Parse skeleton joint angles from pose vectors
        smpl_so3 = self._parse_skel_so3(smpl_pose)
        if rest_pose is not None:
            rest_so3 = self._parse_skel_so3(rest_pose)
        else:
            rest_so3 = torch.zeros_like(smpl_so3)

        if type == "forward":
            # Step 1: BoB forward (clothing deformation in canonical space)
            bob_out, bob_dict = self.bob(
                xyz, smpl_pose, rest_pose=rest_pose, type="forward"
            )

            # Step 2: Skeleton forward (body articulation)
            out, skel_dict = self._skeleton_warp(
                bob_out, smpl_so3, rest_so3, type="forward"
            )

        elif type == "backward":
            # Step 1: Skeleton backward (undo body articulation)
            skel_out, skel_dict = self._skeleton_warp(
                xyz, smpl_so3, rest_so3, type="backward"
            )

            # Step 2: BoB backward (undo clothing deformation)
            out, bob_dict = self.bob(
                skel_out, smpl_pose, rest_pose=rest_pose, type="backward"
            )

        else:
            raise ValueError(f"Unknown warp type: {type}")

        # Collect auxiliary outputs
        warp_dict = {
            "skin_prob": bob_dict.get("skin_prob"),
            "delta_skin": bob_dict.get("delta_skin"),
            "skel_skin_prob": skel_dict.get("skin_prob"),
            "t_articulation": bob_dict.get("t_articulation"),
            "rest_articulation": bob_dict.get("rest_articulation"),
        }

        return out, warp_dict

    def _skeleton_warp(self, xyz, smpl_so3, rest_so3, type="forward"):
        """Apply skeleton FK + Gaussian skinning warp.

        Args:
            xyz: (M, N, 3) Points to warp
            smpl_so3: (M, K, 3) Target joint angles
            rest_so3: (M, K, 3) Rest joint angles
            type (str): "forward" or "backward"
        Returns:
            out: (M, N, 3) Warped points
            skel_dict (Dict): Skeleton auxiliary outputs
        """
        squeeze_d = False
        if xyz.ndim == 3:
            xyz = xyz.unsqueeze(2)  # (M, N, 1, 3)
            squeeze_d = True

        # Compute skeleton articulation via FK
        t_art = self.skeleton.compute_articulation(smpl_so3)
        rest_art = self.skeleton.compute_articulation(rest_so3)

        # Compute relative bone transforms
        if type == "forward":
            # canonical -> deformed: target * inv(rest)
            se3 = dual_quaternion_mul(t_art, dual_quaternion_inverse(rest_art))
            articulation = rest_art  # skinning weights computed in canonical space
        else:
            # deformed -> canonical: rest * inv(target)
            se3 = dual_quaternion_mul(rest_art, dual_quaternion_inverse(t_art))
            articulation = t_art  # skinning weights computed in deformed space

        # Compute Gaussian skinning weights
        art_expanded = (
            articulation[0][:, None, None, :, :],
            articulation[1][:, None, None, :, :],
        )
        skin, delta_skin = self.skeleton.skinning_model(
            xyz, art_expanded, None, None
        )
        skin_prob = skin.softmax(-1)

        # Apply dual quaternion skinning
        out, dual_quat = dual_quaternion_skinning(se3, xyz, skin_prob)

        if squeeze_d:
            out = out.squeeze(2)
            skin_prob = skin_prob.squeeze(2)

        skel_dict = {
            "skin_prob": skin_prob,
            "dual_quat": dual_quat,
        }
        if delta_skin is not None:
            skel_dict["delta_skin"] = delta_skin.pow(2).mean(-1, keepdims=True)

        return out, skel_dict


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- Test PoseConditionedBoB (original) ----
    print("=" * 60)
    print("Testing PoseConditionedBoB (single-layer)")
    print("=" * 60)
    bob = PoseConditionedBoB(num_se3=25, pose_dim=69).to(device)

    M, N = 2, 100
    xyz = torch.randn(M, N, 3, device=device) * 0.1
    smpl_pose = torch.randn(M, 69, device=device) * 0.1

    out_fwd, dict_fwd = bob(xyz, smpl_pose, type="forward")
    print(f"Forward warp: {xyz.shape} -> {out_fwd.shape}")
    print(f"  Skinning weights shape: {dict_fwd['skin_prob'].shape}")

    out_bwd, dict_bwd = bob(out_fwd, smpl_pose, type="backward")
    cyc_error = (out_bwd - xyz).norm(dim=-1).mean()
    print(f"Cycle consistency error: {cyc_error.item():.6f}")

    total_params = sum(p.numel() for p in bob.parameters())
    print(f"Total parameters: {total_params:,}")

    # ---- Test ComposedPoseBoB (two-layer) ----
    print()
    print("=" * 60)
    print("Testing ComposedPoseBoB (skeleton + BoB)")
    print("=" * 60)
    composed = ComposedPoseBoB(
        skel_type="smpl", num_bob_bones=25, pose_dim=69
    ).to(device)

    out_fwd_c, dict_fwd_c = composed(xyz, smpl_pose, type="forward")
    print(f"Forward warp: {xyz.shape} -> {out_fwd_c.shape}")
    print(f"  BoB skin_prob shape: {dict_fwd_c['skin_prob'].shape}")
    print(f"  Skel skin_prob shape: {dict_fwd_c['skel_skin_prob'].shape}")

    out_bwd_c, dict_bwd_c = composed(out_fwd_c, smpl_pose, type="backward")
    cyc_error_c = (out_bwd_c - xyz).norm(dim=-1).mean()
    print(f"Cycle consistency error: {cyc_error_c.item():.6f}")

    # Gradient check
    composed.zero_grad()
    loss = out_fwd_c.sum()
    loss.backward()
    params_with_grad = sum(
        1 for p in composed.parameters() if p.requires_grad and p.grad is not None
    )
    params_total = sum(1 for p in composed.parameters() if p.requires_grad)
    print(f"Gradient flow: {params_with_grad}/{params_total} parameters have gradients")

    skel_params = sum(p.numel() for p in composed.skeleton.parameters())
    bob_params = sum(p.numel() for p in composed.bob.parameters())
    total_params_c = sum(p.numel() for p in composed.parameters())
    print(f"Skeleton parameters: {skel_params:,}")
    print(f"BoB parameters: {bob_params:,}")
    print(f"Total parameters: {total_params_c:,}")
