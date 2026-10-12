"""
Training script for DNA-Rendering data.
Modified from original train.py to work with converted DNA-Rendering data.
"""
import torch
import os
import json
import copy
import numpy as np
from PIL import Image
from random import randint
from tqdm import tqdm
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from helpers import setup_camera, l1_loss_v1, l1_loss_v2, weighted_l2_loss_v1, weighted_l2_loss_v2, quat_mult, \
    o3d_knn, params2rendervar, params2cpu, save_params
from external import calc_ssim, calc_psnr, build_rotation, densify, update_params_and_optimizer


def save_params_safe(output_params, seq, exp, output_base=None, npz_basename="params"):
    """
    Save params to npz file. Handles single timestep case.
    npz_basename: file name without extension (np.savez adds .npz).
    """
    to_save = {}
    if len(output_params) == 1:
        # Single timestep - save directly with added dimension for consistency
        for k in output_params[0].keys():
            to_save[k] = np.expand_dims(output_params[0][k], axis=0)
    else:
        # Multiple timesteps
        for k in output_params[0].keys():
            if k in output_params[1].keys():
                to_save[k] = np.stack([params[k] for params in output_params])
            else:
                to_save[k] = output_params[0][k]

    if output_base:
        output_dir = f"{output_base}/{seq}"
    else:
        output_dir = f"./output/{exp}/{seq}"
    os.makedirs(output_dir, exist_ok=True)
    np.savez(f"{output_dir}/{npz_basename}", **to_save)
    return output_dir


def load_output_history(path):
    """Restore the list representation expected by save_params_safe."""
    dynamic_keys = {"means3D", "rgb_colors", "unnorm_rotations"}
    with np.load(path, allow_pickle=False) as saved:
        frame_count = int(saved["means3D"].shape[0])
        history = []
        for frame in range(frame_count):
            record = {}
            for key in saved.files:
                value = saved[key]
                if key in dynamic_keys:
                    record[key] = value[frame]
                elif frame_count == 1:
                    record[key] = value[0]
                else:
                    record[key] = value
            if frame:
                record = {key: value for key, value in record.items() if key in dynamic_keys}
            history.append(record)
    return history


def save_training_state(path, timestep, params, variables, optimizer):
    """Save a rolling optimizer checkpoint that can continue the next frame."""
    transient = {"means2D", "seen"}
    persistent_variables = {}
    for key, value in variables.items():
        if key in transient:
            continue
        if torch.is_tensor(value):
            persistent_variables[key] = value.detach().cpu()
        else:
            persistent_variables[key] = value
    state = {
        "schema_version": 1,
        "completed_timestep": int(timestep),
        "params": {
            key: value.detach().cpu()
            for key, value in params.items()
        },
        "variables": persistent_variables,
        "optimizer": optimizer.state_dict(),
    }
    temporary = f"{path}.tmp"
    torch.save(state, temporary)
    os.replace(temporary, path)


def load_training_state(path):
    """Load parameters, temporal constraints, and Adam state onto CUDA."""
    state = torch.load(path, map_location="cpu")
    if state.get("schema_version") != 1:
        raise ValueError(f"unsupported training checkpoint: {path}")
    params = {
        key: torch.nn.Parameter(
            value.cuda().float().contiguous().requires_grad_(True)
        )
        for key, value in state["params"].items()
    }
    variables = {}
    for key, value in state["variables"].items():
        variables[key] = value.cuda() if torch.is_tensor(value) else value
    optimizer = initialize_optimizer(params, variables)
    optimizer.load_state_dict(state["optimizer"])
    return int(state["completed_timestep"]), params, variables, optimizer


def export_pointcloud_ply(params, output_path, timestep=None):
    """
    Export point cloud positions to PLY file.

    Args:
        params: Dictionary containing 'means3D' (and optionally 'rgb_colors')
        output_path: Path to save the PLY file
        timestep: If params contains multiple timesteps, specify which one to export
    """
    # Get means3D
    if isinstance(params['means3D'], torch.Tensor):
        means3D = params['means3D'].detach().cpu().numpy()
    else:
        means3D = np.array(params['means3D'])

    # Handle multi-timestep params
    if means3D.ndim == 3 and timestep is not None:
        means3D = means3D[timestep]

    # Get colors if available
    if 'rgb_colors' in params:
        if isinstance(params['rgb_colors'], torch.Tensor):
            rgb = params['rgb_colors'].detach().cpu().numpy()
        else:
            rgb = np.array(params['rgb_colors'])
        if rgb.ndim == 3 and timestep is not None:
            rgb = rgb[timestep]
        # Convert to 0-255 range
        rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        has_color = True
    else:
        has_color = False

    num_points = means3D.shape[0]

    # Write PLY file
    with open(output_path, 'w') as f:
        # Header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {num_points}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if has_color:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        f.write("end_header\n")

        # Data
        for i in range(num_points):
            x, y, z = means3D[i]
            if has_color:
                r, g, b = rgb[i]
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")
            else:
                f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")

    print(f"Exported {num_points} points to {output_path}")


def get_dataset(t, md, seq, data_dir="./data", mask_background=True):
    """
    Load dataset for a timestep.

    Args:
        mask_background: If True, set background pixels to black (only train on foreground/human)
    """
    dataset = []
    for c in range(len(md['fn'][t])):
        w, h, k, w2c = md['w'], md['h'], md['k'][t][c], md['w2c'][t][c]
        cam = setup_camera(w, h, k, w2c, near=1.0, far=100)
        fn = md['fn'][t][c]
        im = np.array(copy.deepcopy(Image.open(f"{data_dir}/{seq}/ims/{fn}")))
        im = torch.tensor(im).float().cuda().permute(2, 0, 1) / 255
        seg = np.array(copy.deepcopy(Image.open(f"{data_dir}/{seq}/seg/{fn.replace('.jpg', '.png')}"))).astype(np.float32)
        # Normalize seg to [0, 1] - handles both 0/1 and 0/255 masks
        if seg.max() > 1:
            seg = seg / 255.0
        seg = torch.tensor(seg).float().cuda()

        # Apply mask to image: set background to black
        if mask_background:
            mask = seg.unsqueeze(0)  # [1, H, W]
            im = im * mask  # Background becomes black (0)

        seg_col = torch.stack((seg, torch.zeros_like(seg), 1 - seg))
        dataset.append({'cam': cam, 'im': im, 'seg': seg_col, 'seg_mask': seg, 'id': c})
    return dataset


def get_batch(todo_dataset, dataset):
    if not todo_dataset:
        todo_dataset = dataset.copy()
    curr_data = todo_dataset.pop(randint(0, len(todo_dataset) - 1))
    return curr_data


def initialize_params_from_pointcloud(seq, md, data_dir="./data", pure_fg=False, isotropic=False):
    """Initialize from init_pt_cld.npz if available.

    Args:
        pure_fg: If True, only use foreground points (seg=1) for initialization
        isotropic: If True, use single scale value per Gaussian (spherical)
    """
    npz_path = f"{data_dir}/{seq}/init_pt_cld.npz"
    init_pt_cld = np.load(npz_path, allow_pickle=True)["data"]

    # Filter to foreground only if requested
    if pure_fg:
        fg_mask = init_pt_cld[:, 6] == 1
        init_pt_cld = init_pt_cld[fg_mask]
        print(f"  Pure FG mode: {fg_mask.sum()} / {len(fg_mask)} points")

    seg = init_pt_cld[:, 6]
    max_cams = len(md['fn'][0])  # Get actual number of cameras from metadata
    sq_dist, _ = o3d_knn(init_pt_cld[:, :3], 3)
    mean3_sq_dist = sq_dist.mean(-1).clip(min=0.0000001)

    # Scale: (N, 1) for isotropic, (N, 3) for anisotropic
    if isotropic:
        log_scales = np.log(np.sqrt(mean3_sq_dist))[..., None]  # (N, 1)
    else:
        log_scales = np.tile(np.log(np.sqrt(mean3_sq_dist))[..., None], (1, 3))  # (N, 3)

    params = {
        'means3D': init_pt_cld[:, :3],
        'rgb_colors': init_pt_cld[:, 3:6],
        'seg_colors': np.stack((seg, np.zeros_like(seg), 1 - seg), -1),
        'unnorm_rotations': np.tile([1, 0, 0, 0], (seg.shape[0], 1)),
        'logit_opacities': np.zeros((seg.shape[0], 1)),
        'log_scales': log_scales,
        'cam_m': np.zeros((max_cams, 3)),
        'cam_c': np.zeros((max_cams, 3)),
    }
    params = {k: torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True)) for k, v in
              params.items()}
    cam_centers = np.linalg.inv(md['w2c'][0])[:, :3, 3]  # Get scene radius
    scene_radius = 1.1 * np.max(np.linalg.norm(cam_centers - np.mean(cam_centers, 0)[None], axis=-1))
    variables = {'max_2D_radius': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'scene_radius': scene_radius,
                 'means2D_gradient_accum': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'denom': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'isotropic': isotropic}
    return params, variables


def initialize_params_random(md, num_points=10000):
    """Initialize with random points (no init_pt_cld needed)."""
    # Get camera centers to estimate scene bounds
    cam_centers = np.linalg.inv(md['w2c'][0])[:, :3, 3]
    center = cam_centers.mean(axis=0)
    scale = np.linalg.norm(cam_centers - center, axis=1).max()

    # Random points around scene center
    means3D = np.random.randn(num_points, 3) * scale * 0.5 + center
    rgb_colors = np.random.rand(num_points, 3)
    seg = np.ones(num_points)  # All foreground initially

    max_cams = len(md['fn'][0])  # Get actual number of cameras from metadata

    # Estimate appropriate initial scale based on pixel size
    # pixel_size_at_distance = distance / focal_length
    K = np.array(md['k'][0][0])
    fx = K[0, 0]
    avg_distance = scale  # Approximate distance to scene
    pixel_size = avg_distance / fx
    # Use a small multiple of pixel size as initial scale
    initial_scale = pixel_size * 3.0  # 3 pixels
    print(f"  Initial Gaussian scale: {initial_scale*1000:.2f}mm (based on {pixel_size*1000:.2f}mm pixel size)")

    params = {
        'means3D': means3D,
        'rgb_colors': rgb_colors,
        'seg_colors': np.stack((seg, np.zeros_like(seg), 1 - seg), -1),
        'unnorm_rotations': np.tile([1, 0, 0, 0], (num_points, 1)),
        'logit_opacities': np.zeros((num_points, 1)),
        'log_scales': np.full((num_points, 3), np.log(initial_scale)),
        'cam_m': np.zeros((max_cams, 3)),
        'cam_c': np.zeros((max_cams, 3)),
    }
    params = {k: torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True)) for k, v in
              params.items()}

    scene_radius = 1.1 * scale
    variables = {'max_2D_radius': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'scene_radius': scene_radius,
                 'means2D_gradient_accum': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'denom': torch.zeros(params['means3D'].shape[0]).cuda().float()}
    return params, variables


def initialize_optimizer(params, variables):
    # For isotropic Gaussians, disable rotation learning (spheres don't need rotation)
    rotation_lr = 0.0 if variables.get('isotropic', False) else 0.001

    lrs = {
        'means3D': 0.00016 * variables['scene_radius'],
        'rgb_colors': 0.0025,
        'seg_colors': 0.0,
        'unnorm_rotations': rotation_lr,
        'logit_opacities': 0.05,
        'log_scales': 0.001,
        'cam_m': 1e-4,
        'cam_c': 1e-4,
    }
    param_groups = [{'params': [v], 'name': k, 'lr': lrs[k]} for k, v in params.items()]
    return torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)


def params2rendervar_isotropic(params):
    """Convert params to rendervar for isotropic Gaussians (scales are (N, 1))."""
    # Expand (N, 1) scales to (N, 3)
    log_scales = params['log_scales']
    if log_scales.shape[-1] == 1:
        log_scales = log_scales.repeat(1, 3)

    rendervar = {
        'means3D': params['means3D'],
        'colors_precomp': params['rgb_colors'],
        'rotations': torch.nn.functional.normalize(params['unnorm_rotations']),
        'opacities': torch.sigmoid(params['logit_opacities']),
        'scales': torch.exp(log_scales),
        'means2D': torch.zeros_like(params['means3D'], requires_grad=True, device="cuda") + 0
    }
    return rendervar


def accumulate_mean2d_gradient(variables):
    """Accumulate 2D gradient for densification decisions."""
    variables['means2D_gradient_accum'][variables['seen']] += torch.norm(
        variables['means2D'].grad[variables['seen'], :2], dim=-1)
    variables['denom'][variables['seen']] += 1
    return variables


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


def cat_params_to_optimizer_isotropic(new_params, params, optimizer):
    """Concatenate new params to optimizer for isotropic Gaussians."""
    for k, v in new_params.items():
        group = [g for g in optimizer.param_groups if g['name'] == k][0]
        stored_state = optimizer.state.get(group['params'][0], None)
        if stored_state is not None:
            stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(v)), dim=0)
            stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(v)), dim=0)
            del optimizer.state[group['params'][0]]
            group["params"][0] = torch.nn.Parameter(torch.cat((group["params"][0], v), dim=0).requires_grad_(True))
            optimizer.state[group['params'][0]] = stored_state
            params[k] = group["params"][0]
        else:
            group["params"][0] = torch.nn.Parameter(torch.cat((group["params"][0], v), dim=0).requires_grad_(True))
            params[k] = group["params"][0]
    return params


def remove_points_isotropic(to_remove, params, variables, optimizer):
    """Remove points for isotropic Gaussians."""
    to_keep = ~to_remove
    keys = [k for k in params.keys() if k not in ['cam_m', 'cam_c']]
    for k in keys:
        group = [g for g in optimizer.param_groups if g['name'] == k][0]
        stored_state = optimizer.state.get(group['params'][0], None)
        if stored_state is not None:
            stored_state["exp_avg"] = stored_state["exp_avg"][to_keep]
            stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][to_keep]
            del optimizer.state[group['params'][0]]
            group["params"][0] = torch.nn.Parameter((group["params"][0][to_keep].requires_grad_(True)))
            optimizer.state[group['params'][0]] = stored_state
            params[k] = group["params"][0]
        else:
            group["params"][0] = torch.nn.Parameter(group["params"][0][to_keep].requires_grad_(True))
            params[k] = group["params"][0]
    variables['means2D_gradient_accum'] = variables['means2D_gradient_accum'][to_keep]
    variables['denom'] = variables['denom'][to_keep]
    variables['max_2D_radius'] = variables['max_2D_radius'][to_keep]
    return params, variables


def densify_isotropic(params, variables, optimizer, i):
    """Densify for isotropic Gaussians (log_scales is (N, 1))."""
    if i <= 5000:
        variables = accumulate_mean2d_gradient(variables)
        grad_thresh = 0.0002
        if (i >= 500) and (i % 100 == 0):
            grads = variables['means2D_gradient_accum'] / variables['denom']
            grads[grads.isnan()] = 0.0

            # For isotropic, log_scales is (N, 1)
            scales = torch.exp(params['log_scales'][:, 0])  # (N,)

            # Clone small Gaussians with high gradient
            to_clone = torch.logical_and(grads >= grad_thresh, scales <= 0.01 * variables['scene_radius'])
            new_params = {k: v[to_clone] for k, v in params.items() if k not in ['cam_m', 'cam_c']}
            params = cat_params_to_optimizer_isotropic(new_params, params, optimizer)
            num_pts = params['means3D'].shape[0]

            padded_grad = torch.zeros(num_pts, device="cuda")
            padded_grad[:grads.shape[0]] = grads

            # Split large Gaussians with high gradient
            scales = torch.exp(params['log_scales'][:, 0])
            to_split = torch.logical_and(padded_grad >= grad_thresh, scales > 0.01 * variables['scene_radius'])
            n = 2
            new_params = {k: v[to_split].repeat(n, 1) for k, v in params.items() if k not in ['cam_m', 'cam_c']}

            # For isotropic, use the single scale value for all 3 dimensions
            stds = scales[to_split].unsqueeze(-1).repeat(n, 3)  # (n*split, 3)
            means = torch.zeros((stds.size(0), 3), device="cuda")
            samples = torch.normal(mean=means, std=stds)
            rots = build_rotation(params['unnorm_rotations'][to_split]).repeat(n, 1, 1)
            new_params['means3D'] += torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)
            # Reduce scale for split Gaussians
            new_params['log_scales'] = torch.log(torch.exp(new_params['log_scales']) / (0.8 * n))
            params = cat_params_to_optimizer_isotropic(new_params, params, optimizer)
            num_pts = params['means3D'].shape[0]

            variables['means2D_gradient_accum'] = torch.zeros(num_pts, device="cuda")
            variables['denom'] = torch.zeros(num_pts, device="cuda")
            variables['max_2D_radius'] = torch.zeros(num_pts, device="cuda")
            to_remove = torch.cat((to_split, torch.zeros(n * to_split.sum(), dtype=torch.bool, device="cuda")))
            params, variables = remove_points_isotropic(to_remove, params, variables, optimizer)

            # Prune low opacity and big points
            remove_threshold = 0.25 if i == 5000 else 0.005
            to_remove = (torch.sigmoid(params['logit_opacities']) < remove_threshold).squeeze()
            if i >= 3000:
                scales = torch.exp(params['log_scales'][:, 0])
                big_points_ws = scales > 0.1 * variables['scene_radius']
                to_remove = torch.logical_or(to_remove, big_points_ws)
            params, variables = remove_points_isotropic(to_remove, params, variables, optimizer)

            torch.cuda.empty_cache()

        # Reset opacity periodically
        if i > 0 and i % 3000 == 0:
            new_logit = inverse_sigmoid(torch.ones_like(params['logit_opacities']) * 0.01)
            group = [g for g in optimizer.param_groups if g['name'] == 'logit_opacities'][0]
            stored_state = optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.zeros_like(new_logit)
                stored_state["exp_avg_sq"] = torch.zeros_like(new_logit)
                del optimizer.state[group['params'][0]]
            group["params"][0] = torch.nn.Parameter(new_logit.requires_grad_(True))
            if stored_state is not None:
                optimizer.state[group['params'][0]] = stored_state
            params['logit_opacities'] = group["params"][0]

    return params, variables


def masked_l1_loss(pred, gt, mask):
    """Compute L1 loss: pred is NOT masked, gt is already masked (bg=black).

    This way, Gaussians rendering outside the mask get penalized because
    their rendered color != 0 (black GT), providing gradient to push them away.
    """
    # mask: [H, W], pred: [C, H, W] (unmasked), gt: [C, H, W] (already masked)
    # Loss over entire image, normalized by total pixel count
    return torch.abs(pred - gt).mean()


def masked_ssim(pred, gt, mask):
    """Compute SSIM on the masked image (background already set to 0)."""
    # For SSIM, we just use the masked images directly
    # Background is 0, so SSIM will be computed with black background
    return calc_ssim(pred, gt)


def get_loss(params, curr_data, variables, is_initial_timestep, use_mask=True, use_floor_loss=True):
    losses = {}
    isotropic = variables.get('isotropic', False)

    # Get mask for foreground-only loss
    mask = curr_data.get('seg_mask', None)

    # Use appropriate rendervar based on isotropic setting
    if isotropic:
        rendervar = params2rendervar_isotropic(params)
    else:
        rendervar = params2rendervar(params)
    rendervar['means2D'].retain_grad()
    im, radius, _, = Renderer(raster_settings=curr_data['cam'])(**rendervar)
    curr_id = curr_data['id']
    im = torch.exp(params['cam_m'][curr_id])[:, None, None] * im + params['cam_c'][curr_id][:, None, None]

    # Loss computation: rendered (unmasked) vs GT (already masked, bg=black)
    # Do NOT mask rendered image - Gaussians outside mask get penalized against black GT
    if use_mask and mask is not None:
        gt_masked = curr_data['im']  # Already masked in get_dataset (bg=black)
        losses['im'] = 0.8 * masked_l1_loss(im, gt_masked, mask) + 0.2 * (1.0 - calc_ssim(im, gt_masked))
    else:
        losses['im'] = 0.8 * l1_loss_v1(im, curr_data['im']) + 0.2 * (1.0 - calc_ssim(im, curr_data['im']))

    variables['means2D'] = rendervar['means2D']  # Gradient only accum from colour render for densification

    # Segmentation loss - KEY for pruning points outside mask!
    if isotropic:
        segrendervar = params2rendervar_isotropic(params)
    else:
        segrendervar = params2rendervar(params)
    segrendervar['colors_precomp'] = params['seg_colors']
    seg, _, _, = Renderer(raster_settings=curr_data['cam'])(**segrendervar)
    losses['seg'] = 0.8 * l1_loss_v1(seg, curr_data['seg']) + 0.2 * (1.0 - calc_ssim(seg, curr_data['seg']))

    if not is_initial_timestep:
        is_fg = (params['seg_colors'][:, 0] > 0.5).detach()
        fg_pts = rendervar['means3D'][is_fg]
        fg_rot = rendervar['rotations'][is_fg]

        rel_rot = quat_mult(fg_rot, variables["prev_inv_rot_fg"])
        rot = build_rotation(rel_rot)
        neighbor_pts = fg_pts[variables["neighbor_indices"]]
        curr_offset = neighbor_pts - fg_pts[:, None]
        curr_offset_in_prev_coord = (rot.transpose(2, 1)[:, None] @ curr_offset[:, :, :, None]).squeeze(-1)
        losses['rigid'] = weighted_l2_loss_v2(curr_offset_in_prev_coord, variables["prev_offset"],
                                              variables["neighbor_weight"])

        losses['rot'] = weighted_l2_loss_v2(rel_rot[variables["neighbor_indices"]], rel_rot[:, None],
                                            variables["neighbor_weight"])

        curr_offset_mag = torch.sqrt((curr_offset ** 2).sum(-1) + 1e-20)
        losses['iso'] = weighted_l2_loss_v1(curr_offset_mag, variables["neighbor_dist"], variables["neighbor_weight"])

        if use_floor_loss:
            losses['floor'] = torch.clamp(fg_pts[:, 1], min=0).mean()
        else:
            losses['floor'] = torch.tensor(0.0).cuda()

        bg_pts = rendervar['means3D'][~is_fg]
        bg_rot = rendervar['rotations'][~is_fg]
        losses['bg'] = l1_loss_v2(bg_pts, variables["init_bg_pts"]) + l1_loss_v2(bg_rot, variables["init_bg_rot"])

        losses['soft_col_cons'] = l1_loss_v2(params['rgb_colors'], variables["prev_col"])

    loss_weights = {'im': 1.0, 'seg': 3.0, 'rigid': 4.0, 'rot': 4.0, 'iso': 2.0, 'floor': 2.0, 'bg': 20.0,
                    'soft_col_cons': 0.01}
    loss = sum([loss_weights[k] * v for k, v in losses.items()])
    seen = radius > 0
    variables['max_2D_radius'][seen] = torch.max(radius[seen], variables['max_2D_radius'][seen])
    variables['seen'] = seen
    return loss, variables


def initialize_per_timestep(params, variables, optimizer):
    pts = params['means3D']
    rot = torch.nn.functional.normalize(params['unnorm_rotations'])
    new_pts = pts + (pts - variables["prev_pts"])
    new_rot = torch.nn.functional.normalize(rot + (rot - variables["prev_rot"]))

    is_fg = params['seg_colors'][:, 0] > 0.5
    prev_inv_rot_fg = rot[is_fg]
    prev_inv_rot_fg[:, 1:] = -1 * prev_inv_rot_fg[:, 1:]
    fg_pts = pts[is_fg]
    prev_offset = fg_pts[variables["neighbor_indices"]] - fg_pts[:, None]
    variables['prev_inv_rot_fg'] = prev_inv_rot_fg.detach()
    variables['prev_offset'] = prev_offset.detach()
    variables["prev_col"] = params['rgb_colors'].detach()
    variables["prev_pts"] = pts.detach()
    variables["prev_rot"] = rot.detach()

    new_params = {'means3D': new_pts, 'unnorm_rotations': new_rot}
    params = update_params_and_optimizer(new_params, params, optimizer)

    return params, variables


def initialize_post_first_timestep(params, variables, optimizer, num_knn=20):
    is_fg = params['seg_colors'][:, 0] > 0.5
    init_fg_pts = params['means3D'][is_fg]
    init_bg_pts = params['means3D'][~is_fg]
    init_bg_rot = torch.nn.functional.normalize(params['unnorm_rotations'][~is_fg])
    neighbor_sq_dist, neighbor_indices = o3d_knn(init_fg_pts.detach().cpu().numpy(), num_knn)
    neighbor_weight = np.exp(-2000 * neighbor_sq_dist)
    neighbor_dist = np.sqrt(neighbor_sq_dist)
    variables["neighbor_indices"] = torch.tensor(neighbor_indices).cuda().long().contiguous()
    variables["neighbor_weight"] = torch.tensor(neighbor_weight).cuda().float().contiguous()
    variables["neighbor_dist"] = torch.tensor(neighbor_dist).cuda().float().contiguous()

    variables["init_bg_pts"] = init_bg_pts.detach()
    variables["init_bg_rot"] = init_bg_rot.detach()
    variables["prev_pts"] = params['means3D'].detach()
    variables["prev_rot"] = torch.nn.functional.normalize(params['unnorm_rotations']).detach()
    params_to_fix = ['logit_opacities', 'log_scales', 'cam_m', 'cam_c']
    for param_group in optimizer.param_groups:
        if param_group["name"] in params_to_fix:
            param_group['lr'] = 0.0
    return variables


def report_progress(params, data, i, progress_bar, every_i=100, use_mask=True, isotropic=False):
    if i % every_i == 0:
        if isotropic:
            rendervar = params2rendervar_isotropic(params)
        else:
            rendervar = params2rendervar(params)
        im, _, _, = Renderer(raster_settings=data['cam'])(**rendervar)
        curr_id = data['id']
        im = torch.exp(params['cam_m'][curr_id])[:, None, None] * im + params['cam_c'][curr_id][:, None, None]

        # Apply mask for PSNR calculation (compare only foreground pixels)
        if use_mask and 'seg_mask' in data:
            mask = data['seg_mask'].unsqueeze(0)  # [1, H, W]
            gt_masked = data['im']  # Already masked (bg=black)
            # Compute MSE only over foreground pixels to avoid inflated PSNR
            # Use unmasked rendered (same as loss) but evaluate FG PSNR on foreground only
            fg_count = mask.sum().clamp(min=1)
            mse = ((im * mask - gt_masked) ** 2).sum() / (fg_count * 3)  # 3 channels
            psnr = 20 * torch.log10(1.0 / torch.sqrt(mse.clamp(min=1e-10)))
        else:
            psnr = calc_psnr(im, data['im']).mean()

        progress_bar.set_postfix({"FG PSNR": f"{psnr:.{7}f}"})
        progress_bar.update(every_i)


def train(seq, exp, data_dir="./data", use_pointcloud=True, num_timesteps=None, mask_background=True,
          num_iters=10000, output_base=None, pure_fg=False, isotropic=False, use_floor_loss=True,
          checkpoint_every=0, resume=False):
    """
    Train Dynamic 3D Gaussians on a sequence.

    Args:
        seq: Sequence name (e.g., "dna_0013_01")
        exp: Experiment name
        data_dir: Data directory (default: "./data")
        use_pointcloud: If True, use init_pt_cld.npz; if False, use random initialization
        num_timesteps: Number of timesteps to train (None = all)
        mask_background: If True, only train on foreground (human), set background to black
        num_iters: Number of iterations for initial timestep (default: 10000)
        output_base: Override output directory base (default: ./output/{exp})
        pure_fg: If True, only use foreground points for initialization
        isotropic: If True, use isotropic (spherical) Gaussians
        checkpoint_every: If > 0, save params_ckpt_upto_XXX.npz every N completed timesteps
    """
    if output_base:
        output_dir = f"{output_base}/{seq}"
    else:
        output_dir = f"./output/{exp}/{seq}"
    state_path = f"{output_dir}/training_state_latest.pt"
    history_path = f"{output_dir}/params_ckpt_latest.npz"
    can_resume = resume and os.path.isfile(state_path) and os.path.isfile(history_path)
    if os.path.exists(output_dir) and not can_resume:
        print(f"Experiment '{exp}' for sequence '{seq}' already exists. Exiting.")
        return

    # Load metadata
    md = json.load(open(f"{data_dir}/{seq}/train_meta.json", 'r'))
    total_timesteps = len(md['fn'])
    if num_timesteps is not None:
        total_timesteps = min(num_timesteps, total_timesteps)

    print(f"Training on {seq}")
    print(f"  Data dir: {data_dir}/{seq}")
    print(f"  Image size: {md['w']}x{md['h']}")
    print(f"  Timesteps: {total_timesteps}")
    print(f"  Cameras per timestep: {len(md['fn'][0])}")
    print(f"  Mask background (foreground only): {mask_background}")
    print(f"  Pure FG init: {pure_fg}")
    print(f"  Isotropic Gaussians: {isotropic}")
    print(f"  Floor loss: {use_floor_loss}")
    if checkpoint_every > 0:
        print(f"  Checkpoint every {checkpoint_every} timestep(s)")

    if can_resume:
        completed_timestep, params, variables, optimizer = load_training_state(
            state_path
        )
        output_params = load_output_history(history_path)
        start_timestep = completed_timestep + 1
        print(
            f"  Resuming after timestep {completed_timestep}: "
            f"{len(output_params)} frames restored"
        )
    else:
        if use_pointcloud:
            print("  Initializing from point cloud...")
            params, variables = initialize_params_from_pointcloud(
                seq, md, data_dir, pure_fg=pure_fg, isotropic=isotropic
            )
        else:
            print("  Initializing with random points...")
            params, variables = initialize_params_random(md, num_points=10000)
        optimizer = initialize_optimizer(params, variables)
        output_params = []
        start_timestep = 0

    print(f"  Number of Gaussians: {params['means3D'].shape[0]}")

    for t in range(start_timestep, total_timesteps):
        dataset = get_dataset(t, md, seq, data_dir, mask_background=mask_background)
        todo_dataset = []
        is_initial_timestep = (t == 0)
        if not is_initial_timestep:
            params, variables = initialize_per_timestep(params, variables, optimizer)
        num_iter_per_timestep = num_iters if is_initial_timestep else (num_iters // 5)
        progress_bar = tqdm(range(num_iter_per_timestep), desc=f"timestep {t}")
        for i in range(num_iter_per_timestep):
            curr_data = get_batch(todo_dataset, dataset)
            loss, variables = get_loss(params, curr_data, variables, is_initial_timestep, use_mask=mask_background, use_floor_loss=use_floor_loss)
            loss.backward()
            with torch.no_grad():
                report_progress(params, dataset[0], i, progress_bar, use_mask=mask_background, isotropic=isotropic)
                if is_initial_timestep:
                    params, variables = densify_isotropic(params, variables, optimizer, i) if isotropic else densify(params, variables, optimizer, i)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        progress_bar.close()
        output_params.append(params2cpu(params, is_initial_timestep))
        if is_initial_timestep:
            variables = initialize_post_first_timestep(params, variables, optimizer)
        if checkpoint_every > 0 and (t + 1) % checkpoint_every == 0:
            save_params_safe(
                output_params,
                seq,
                exp,
                output_base=output_base,
                npz_basename="params_ckpt_latest",
            )
            save_training_state(
                state_path,
                t,
                params,
                variables,
                optimizer,
            )
            print(f"  Rolling checkpoint saved ({t + 1} timesteps)")

    save_params_safe(output_params, seq, exp, output_base=output_base)

    # Export point clouds for each timestep
    params_file = f"{output_dir}/params.npz"
    if os.path.exists(params_file):
        saved_params = np.load(params_file)
        # means3D is always 3D: (num_timesteps, num_points, 3)
        num_timesteps_saved = saved_params['means3D'].shape[0]

        for t in range(num_timesteps_saved):
            ply_path = f"{output_dir}/pointcloud_t{t:03d}.ply"
            export_pointcloud_ply(
                {k: saved_params[k] for k in saved_params.files},
                ply_path,
                timestep=t
            )

    print(f"Training complete! Output saved to {output_dir}")


def validate_data(seq, data_dir="./data"):
    """Validate that the data is correctly formatted."""
    print(f"Validating data for {seq}...")

    # Check train_meta.json
    meta_path = f"{data_dir}/{seq}/train_meta.json"
    if not os.path.exists(meta_path):
        print(f"  ERROR: {meta_path} not found!")
        return False

    md = json.load(open(meta_path, 'r'))
    print(f"  train_meta.json: OK")
    print(f"    - w={md['w']}, h={md['h']}")
    print(f"    - {len(md['fn'])} timesteps, {len(md['fn'][0])} cameras")

    # Check a few images and masks
    t, c = 0, 0
    fn = md['fn'][t][c]

    img_path = f"{data_dir}/{seq}/ims/{fn}"
    if not os.path.exists(img_path):
        print(f"  ERROR: Image not found: {img_path}")
        return False

    im = np.array(Image.open(img_path))
    print(f"  Image: shape={im.shape}, dtype={im.dtype}")

    seg_path = f"{data_dir}/{seq}/seg/{fn.replace('.jpg', '.png')}"
    if not os.path.exists(seg_path):
        print(f"  ERROR: Seg not found: {seg_path}")
        return False

    seg = np.array(Image.open(seg_path))
    print(f"  Seg: shape={seg.shape}, dtype={seg.dtype}, unique={np.unique(seg)}")

    # Check init_pt_cld.npz
    npz_path = f"{data_dir}/{seq}/init_pt_cld.npz"
    if os.path.exists(npz_path):
        data = np.load(npz_path, allow_pickle=True)['data']
        print(f"  init_pt_cld.npz: shape={data.shape}")
    else:
        print(f"  init_pt_cld.npz: NOT FOUND (will use random init)")

    # Test loading one batch
    print("  Testing data loading...")
    try:
        w, h, k, w2c = md['w'], md['h'], md['k'][0][0], md['w2c'][0][0]
        cam = setup_camera(w, h, k, w2c, near=1.0, far=100)

        im = np.array(Image.open(img_path))
        im_tensor = torch.tensor(im).float().cuda().permute(2, 0, 1) / 255

        seg = np.array(Image.open(seg_path)).astype(np.float32)
        seg_tensor = torch.tensor(seg).float().cuda()
        seg_col = torch.stack((seg_tensor, torch.zeros_like(seg_tensor), 1 - seg_tensor))

        print(f"    im_tensor: {im_tensor.shape}, range=[{im_tensor.min():.2f}, {im_tensor.max():.2f}]")
        print(f"    seg_col: {seg_col.shape}, range=[{seg_col.min():.2f}, {seg_col.max():.2f}]")
        print("  Data loading: OK")
    except Exception as e:
        print(f"  ERROR during data loading: {e}")
        return False

    print("Validation PASSED!")
    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Dynamic 3D Gaussians on DNA-Rendering data")
    parser.add_argument("--seq", type=str, default="dna_0013_01", help="Sequence name")
    parser.add_argument("--exp", type=str, default="dna_exp1", help="Experiment name")
    parser.add_argument("--data_dir", type=str, default="./data", help="Data directory")
    parser.add_argument("--validate_only", action="store_true", help="Only validate data, don't train")
    parser.add_argument("--num_timesteps", type=int, default=None, help="Number of timesteps to train")
    parser.add_argument("--no_pointcloud", action="store_true", help="Use random init instead of pointcloud")
    parser.add_argument("--no_mask", action="store_true", help="Don't mask background (train full scene)")
    parser.add_argument("--num_iters", type=int, default=10000, help="Number of iterations for first timestep")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--pure_fg", action="store_true", help="Only use foreground points for init")
    parser.add_argument("--isotropic", action="store_true", help="Use isotropic (spherical) Gaussians")
    parser.add_argument("--no_floor", action="store_true", help="Disable floor loss (for scenes not at Y=0)")
    parser.add_argument("--checkpoint_every", type=int, default=0,
                        help="Save params_ckpt_upto_XXX.npz every N timesteps (0 = disabled)")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from params_ckpt_latest.npz and training_state_latest.pt",
    )

    args = parser.parse_args()

    if args.validate_only:
        validate_data(args.seq, args.data_dir)
    else:
        # First validate
        if validate_data(args.seq, args.data_dir):
            print()
            train(
                args.seq,
                args.exp,
                args.data_dir,
                use_pointcloud=not args.no_pointcloud,
                num_timesteps=args.num_timesteps,
                mask_background=not args.no_mask,
                num_iters=args.num_iters,
                output_base=args.output_dir,
                pure_fg=args.pure_fg,
                isotropic=args.isotropic,
                use_floor_loss=not args.no_floor,
                checkpoint_every=args.checkpoint_every,
                resume=args.resume,
            )
            torch.cuda.empty_cache()
