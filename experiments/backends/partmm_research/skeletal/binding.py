"""
Skeletal binding between SMPL joints and SSDR bones

Implements multi-parent soft binding using weighted forward kinematics
"""

import torch
import numpy as np


class SkeletalBinding:
    """
    SMPL-SSDR skeletal binding using soft multi-parent FK

    Each SSDR bone is influenced by k nearest SMPL joints
    """

    def __init__(self, device='cuda:0', k_nearest=4):
        self.device = torch.device(device)
        self.k_nearest = k_nearest
        self.binding_data = None
        self.smpl_model = None

    def build(self, smpl_motion, ssdr_motion, ssdr_rest_vertices, ssdr_skinning_weights):
        """
        Build skeletal binding from rest pose

        Args:
            smpl_motion: SMPLMotionData
            ssdr_motion: SSDRMotionData
            ssdr_rest_vertices: np.ndarray (n_vertices, 3)
            ssdr_skinning_weights: np.ndarray or sparse matrix (n_bones, n_vertices)

        Returns:
            binding_data: dict with indices, weights, local_transforms
        """
        print("\n" + "="*70)
        print("  Building Skeletal Binding (SMPL ↔ SSDR)")
        print("="*70)
        print(f"  Mode: SOFT multi-parent binding (k={self.k_nearest} parents per bone)")

        # Get rest pose
        smpl_data = smpl_motion.sample().to(self.device)
        ssdr_data = ssdr_motion.sample().to(self.device)

        smpl_frame0 = smpl_data[:, :, 0]
        ssdr_frame0 = ssdr_data[:, :, 0]

        # Compute SMPL joint transforms
        from .smpl_fk import compute_smpl_joint_transforms
        smpl_joint_positions, smpl_joint_rotations = compute_smpl_joint_transforms(
            smpl_frame0, device=self.device
        )

        # Extract SSDR bone positions from skinning weights
        rest_vertices = torch.from_numpy(ssdr_rest_vertices).float().to(self.device)

        if hasattr(ssdr_skinning_weights, 'toarray'):
            skinning_weights_np = ssdr_skinning_weights.toarray()
        else:
            skinning_weights_np = ssdr_skinning_weights

        skinning_weights = torch.from_numpy(skinning_weights_np).float().to(self.device)

        n_bones = skinning_weights.shape[0]
        ssdr_bone_positions = torch.zeros(n_bones, 3, device=self.device)

        for bone_idx in range(n_bones):
            bone_weights = skinning_weights[bone_idx]
            max_weight_idx = torch.argmax(bone_weights)
            ssdr_bone_positions[bone_idx] = rest_vertices[max_weight_idx]

        # Extract bone rotations
        ssdr_transforms = ssdr_frame0[:, :-3].reshape(n_bones, 4, 4)
        ssdr_bone_rotations = ssdr_transforms[:, :3, :3]

        # Build SSDR bone rest pose transformations
        ssdr_rest_transforms = torch.zeros(n_bones, 4, 4, device=self.device)
        for bone_idx in range(n_bones):
            T_rest = torch.eye(4, device=self.device)
            T_rest[:3, :3] = ssdr_bone_rotations[bone_idx]
            T_rest[:3, 3] = ssdr_bone_positions[bone_idx]
            ssdr_rest_transforms[bone_idx] = T_rest

        # Compute multi-parent binding
        binding_indices, binding_weights, local_transforms = self._compute_binding(
            ssdr_bone_positions, ssdr_bone_rotations,
            smpl_joint_positions, smpl_joint_rotations
        )

        # Store binding data
        self.binding_data = {
            'indices': binding_indices,
            'weights': binding_weights,
            'local_transforms': local_transforms,
            'k_nearest': self.k_nearest,
            'n_smpl_joints': smpl_joint_positions.shape[0],
            'n_ssdr_bones': n_bones,
            'smpl_rest_joints': smpl_joint_positions,
            'ssdr_rest_bones': ssdr_bone_positions,
            'ssdr_rest_transforms': ssdr_rest_transforms,
            'database_root_rotation': smpl_frame0[0, :4].clone(),
            'database_root_translation': smpl_frame0[0, -3:].clone()
        }

        print(f"  ✓ Bound {n_bones} SSDR bones to {smpl_joint_positions.shape[0]} SMPL joints")
        print("="*70 + "\n")

        return self.binding_data

    def _compute_binding(self, ssdr_positions, ssdr_rotations, smpl_positions, smpl_rotations):
        """
        Compute multi-parent binding weights and local transforms
        """
        n_ssdr_bones = ssdr_positions.shape[0]
        n_smpl_joints = smpl_positions.shape[0]

        binding_indices = torch.zeros(n_ssdr_bones, self.k_nearest, dtype=torch.long, device=self.device)
        binding_weights = torch.zeros(n_ssdr_bones, self.k_nearest, device=self.device)
        local_transforms = torch.zeros(n_ssdr_bones, self.k_nearest, 4, 4, device=self.device)

        for bone_idx in range(n_ssdr_bones):
            bone_pos = ssdr_positions[bone_idx:bone_idx+1]

            # Find k nearest SMPL joints
            distances = torch.cdist(bone_pos, smpl_positions).squeeze(0)
            k = min(self.k_nearest, n_smpl_joints)
            k_nearest_dists, k_nearest_indices = torch.topk(distances, k, largest=False)

            # Compute Gaussian weights
            sigma = k_nearest_dists.median()
            if sigma < 1e-6:
                sigma = 1.0

            weights = torch.exp(-k_nearest_dists**2 / (2 * sigma**2))
            weights = weights / weights.sum()

            binding_indices[bone_idx] = k_nearest_indices
            binding_weights[bone_idx] = weights

            # Compute local transforms
            for i in range(k):
                parent_idx = k_nearest_indices[i].item()

                T_parent = torch.eye(4, device=self.device)
                T_parent[:3, :3] = smpl_rotations[parent_idx]
                T_parent[:3, 3] = smpl_positions[parent_idx]

                T_child = torch.eye(4, device=self.device)
                T_child[:3, :3] = ssdr_rotations[bone_idx]
                T_child[:3, 3] = ssdr_positions[bone_idx]

                local_transforms[bone_idx, i] = torch.inverse(T_parent) @ T_child

        return binding_indices, binding_weights, local_transforms
