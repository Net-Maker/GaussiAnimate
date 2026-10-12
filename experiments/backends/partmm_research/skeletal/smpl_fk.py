"""SMPL Forward Kinematics"""
import torch

from ._smpl_config import get_smpl_model_dir

# Global SMPL model cache (per gender)
_SMPL_MODEL_CACHE = {}


def get_smpl_model(device='cpu', gender='female'):
    """Get or create SMPL model (cached per gender)"""
    global _SMPL_MODEL_CACHE

    gender_key = gender.lower()
    if gender_key not in _SMPL_MODEL_CACHE:
        import smplx
        print(f"  ✓ Initializing SMPL model (gender={gender_key})...")

        model_dir = get_smpl_model_dir()
        # Choose appropriate model file
        if gender_key == 'female':
            model_path = str(model_dir / "SMPL_FEMALE.pkl")
        else:
            model_path = str(model_dir / "SMPL_MALE.pkl")

        _SMPL_MODEL_CACHE[gender_key] = smplx.create(
            model_path=model_path,
            model_type="smpl",
            gender=gender_key.upper(),
            batch_size=1
        )

    return _SMPL_MODEL_CACHE[gender_key]


def compute_smpl_joint_transforms(smpl_frame, device='cpu', gender='female', is_cloth3d=True):
    """
    Compute SMPL joint positions using cached SMPL model

    Args:
        smpl_frame: [1, 99] quaternions or [1, 75] axis-angle
        device: torch device
        gender: 'female' or 'male' (default: 'female')
        is_cloth3d: if True, subtract root joint position (Cloth3D data is centered at pelvis)

    Returns:
        joint_positions: [24, 3]
        joint_rotations: [24, 3, 3]
    """
    from ..math import quaternion_to_axis_angle, axis_angle_to_matrix

    smpl_model = get_smpl_model(device, gender)
    smpl_parents = smpl_model.parents.cpu().numpy()

    n_joints = 24
    data_without_pos = smpl_frame[:, :-3]

    if data_without_pos.shape[1] == 96:  # quaternion
        quaternions = data_without_pos.reshape(n_joints, 4)
        axis_angles = quaternion_to_axis_angle(quaternions)
    else:
        axis_angles = data_without_pos.reshape(n_joints, 3)

    global_orient = axis_angles[0:1]
    body_pose = axis_angles[1:].reshape(1, -1)

    with torch.no_grad():
        smpl_output = smpl_model(
            global_orient=global_orient.cpu(),
            body_pose=body_pose.cpu(),
            transl=None
        )

    joints = smpl_output.joints.squeeze(0)[:24].to(device)

    # Cloth3D data: root is at origin, subtract pelvis position
    # Other data: keep absolute positions
    if is_cloth3d:
        joints = joints - joints[0, :]

    joint_rotations = torch.zeros(n_joints, 3, 3, device=device)
    for i in range(n_joints):
        joint_rotations[i] = axis_angle_to_matrix(axis_angles[i])

    parents = torch.from_numpy(smpl_parents).to(device)
    for i in range(n_joints):
        parent_idx = parents[i]
        if parent_idx >= 0:
            joint_rotations[i] = joint_rotations[parent_idx] @ joint_rotations[i]

    return joints, joint_rotations
