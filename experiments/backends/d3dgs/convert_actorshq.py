#!/usr/bin/env python3
"""
Convert ActorsHQ dataset to Dynamic3DGaussians format.

ActorsHQ format:
- calibration.csv: Camera intrinsics and extrinsics
  - rotation (rx, ry, rz): axis-angle, camera-to-world
  - translation (tx, ty, tz): camera-to-world
  - focal (fx, fy): normalized (focal / image_dimension)
  - principal (px, py): normalized (~0.5)
- 4x/rgbs/CamXXX/CamXXX_rgbYYYYYY.jpg
- 4x/masks/CamXXX/CamXXX_maskYYYYYY.png

Dynamic3DGaussians format:
- ims/cam_id/frame_id.jpg
- seg/cam_id/frame_id.png
- train_meta.json: {w, h, k: [[[K]]], w2c: [[[w2c]]], fn: [[filenames]]}
"""

import argparse
import csv
import json
import lzma
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from tqdm import tqdm


def read_calibration_csv(csv_path: str) -> List[dict]:
    """Read ActorsHQ calibration CSV file."""
    cameras = []
    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cam = {
                'name': row['name'],
                'width': int(row['w']),
                'height': int(row['h']),
                'rotation_axisangle': np.array([
                    float(row['rx']), float(row['ry']), float(row['rz'])
                ]),
                'translation': np.array([
                    float(row['tx']), float(row['ty']), float(row['tz'])
                ]),
                'focal_length': np.array([float(row['fx']), float(row['fy'])]),
                'principal_point': np.array([float(row['px']), float(row['py'])]),
            }
            cameras.append(cam)
    return cameras


def cam2world_to_w2c(rotation_axisangle: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """
    Convert camera-to-world (c2w) parameters to world-to-camera (w2c) matrix.

    ActorsHQ stores extrinsics as c2w:
        world_point = R @ camera_point + t

    We need w2c:
        camera_point = R_inv @ (world_point - t)
                     = R_inv @ world_point - R_inv @ t

    So w2c = [R_inv | -R_inv @ t]
    """
    # Get rotation matrix from axis-angle
    R_c2w = Rotation.from_rotvec(rotation_axisangle).as_matrix()
    t_c2w = translation

    # Compute w2c
    R_w2c = R_c2w.T  # R_inv = R^T for rotation matrices
    t_w2c = -R_w2c @ t_c2w

    # Build 4x4 matrix
    w2c = np.eye(4)
    w2c[:3, :3] = R_w2c
    w2c[:3, 3] = t_w2c

    return w2c


def compute_intrinsic_matrix(cam: dict) -> np.ndarray:
    """
    Compute 3x3 intrinsic matrix K from normalized parameters.

    fx_pixel = fx * width
    fy_pixel = fy * height
    cx_pixel = px * width
    cy_pixel = py * height
    """
    w, h = cam['width'], cam['height']
    fx = cam['focal_length'][0] * w
    fy = cam['focal_length'][1] * h
    cx = cam['principal_point'][0] * w
    cy = cam['principal_point'][1] * h

    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    return K


def get_available_cameras(data_dir: str, scale: str = "4x") -> List[str]:
    """Get list of available camera directories."""
    rgb_dir = os.path.join(data_dir, scale, "rgbs")
    if not os.path.exists(rgb_dir):
        return []

    cameras = sorted([d for d in os.listdir(rgb_dir) if d.startswith("Cam")])
    return cameras


def get_frame_count(data_dir: str, camera_name: str, scale: str = "4x") -> int:
    """Get number of frames for a camera."""
    rgb_dir = os.path.join(data_dir, scale, "rgbs", camera_name)
    if not os.path.exists(rgb_dir):
        return 0

    files = [f for f in os.listdir(rgb_dir) if f.endswith('.jpg')]
    return len(files)


def check_camera_completeness(data_dir: str, camera_name: str, total_frames: int, scale: str = "4x") -> Tuple[bool, int]:
    """
    Check if a camera has complete data (all frames).
    Returns (is_complete, actual_frame_count).
    """
    rgb_dir = os.path.join(data_dir, scale, "rgbs", camera_name)
    mask_dir = os.path.join(data_dir, scale, "masks", camera_name)

    if not os.path.exists(rgb_dir) or not os.path.exists(mask_dir):
        return False, 0

    rgb_count = len([f for f in os.listdir(rgb_dir) if f.endswith('.jpg')])
    mask_count = len([f for f in os.listdir(mask_dir) if f.endswith('.png')])

    is_complete = (rgb_count == total_frames) and (mask_count == total_frames)
    return is_complete, min(rgb_count, mask_count)


def filter_cameras_by_resolution(cameras: List[dict], target_width: Optional[int] = None, target_height: Optional[int] = None) -> List[dict]:
    """
    Filter cameras to only include those with the specified resolution.
    If target not specified, uses the most common resolution.
    """
    if not cameras:
        return cameras

    # Count resolutions
    resolution_counts = {}
    for cam in cameras:
        res = (cam['width'], cam['height'])
        resolution_counts[res] = resolution_counts.get(res, 0) + 1

    # Determine target resolution
    if target_width is not None and target_height is not None:
        target_res = (target_width, target_height)
    else:
        # Use most common resolution
        target_res = max(resolution_counts.keys(), key=lambda r: resolution_counts[r])

    # Filter
    filtered = [cam for cam in cameras if (cam['width'], cam['height']) == target_res]

    print(f"  Resolution filter: {len(filtered)}/{len(cameras)} cameras with {target_res[0]}x{target_res[1]}")
    for res, count in sorted(resolution_counts.items()):
        marker = " <--" if res == target_res else ""
        print(f"    {res[0]}x{res[1]}: {count} cameras{marker}")

    return filtered


def crop_cameras_to_square(cameras: List[dict]) -> Tuple[List[dict], List[Tuple[int, int]]]:
    """
    Crop all cameras to square resolution (min(width, height) x min(width, height)).
    This allows using all cameras regardless of their original orientation.

    Based on humanrf's VolumetricDataset._crop_cameras() method.

    Returns:
        Tuple of (modified cameras list, list of (offset_w, offset_h) for each camera)
    """
    crop_offsets = []

    for cam in cameras:
        w, h = cam['width'], cam['height']
        offset = abs(h - w) // 2

        if w < h:
            # Portrait -> crop top/bottom
            offset_w = 0
            offset_h = offset
            new_size = w
        else:
            # Landscape -> crop left/right
            offset_w = offset
            offset_h = 0
            new_size = h

        crop_offsets.append((offset_w, offset_h))

        # Adjust principal point (it's normalized, so we need to convert, adjust, convert back)
        px, py = cam['principal_point']
        # Convert normalized to pixel, subtract offset, then re-normalize to new size
        px_pixel = px * w - offset_w
        py_pixel = py * h - offset_h
        cam['principal_point'] = np.array([px_pixel / new_size, py_pixel / new_size])

        # Adjust focal length (normalized focal needs to be scaled)
        fx, fy = cam['focal_length']
        # When cropping, the focal length in normalized coords changes because the image size changes
        cam['focal_length'] = np.array([fx * w / new_size, fy * h / new_size])

        # Update dimensions
        cam['width'] = new_size
        cam['height'] = new_size

    return cameras, crop_offsets


def generate_init_pointcloud_from_mesh(
    input_dir: str,
    output_dir: str,
    frame_idx: int = 0,
    num_samples: int = 100000,
    alembic_extractor_path: Optional[str] = None,
):
    """
    Generate init_pt_cld.npz from ActorsHQ mesh (meshes.abc).

    The point cloud format: [x, y, z, r, g, b, seg]
    where seg=1 for foreground (human), seg=0 for background.
    """
    alembic_extractor_path = alembic_extractor_path or os.environ.get("GA_ALEMBIC_EXTRACTOR")
    if not alembic_extractor_path:
        raise ValueError("provide the official HumanRF alembic_extractor path via --alembic-extractor or GA_ALEMBIC_EXTRACTOR")
    mesh_path = Path(input_dir) / "meshes.abc"
    mesh_xz_path = Path(input_dir) / "meshes.abc.xz"

    # Decompress if needed
    if not mesh_path.exists() and mesh_xz_path.exists():
        print(f"Decompressing {mesh_xz_path}...")
        with lzma.open(str(mesh_xz_path), 'rb') as f_in:
            with open(str(mesh_path), 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        print(f"  Decompressed to {mesh_path}")

    if not mesh_path.exists():
        print(f"Warning: No mesh file found at {mesh_path}, skipping init_pt_cld generation")
        return False

    # Extract mesh using alembic_extractor
    temp_obj_dir = Path("/tmp/actorshq_mesh_extract")
    if temp_obj_dir.exists():
        shutil.rmtree(temp_obj_dir)
    temp_obj_dir.mkdir(parents=True)

    print(f"Extracting mesh from {mesh_path}...")
    result = subprocess.run(
        [alembic_extractor_path, "--alembic", str(mesh_path), "--output", str(temp_obj_dir)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"Error extracting mesh: {result.stderr}")
        return False

    # Find the OBJ file for the target frame
    obj_path = temp_obj_dir / f"Frame{frame_idx:06d}.obj"
    if not obj_path.exists():
        # Try to find any available OBJ file
        obj_files = sorted(temp_obj_dir.glob("Frame*.obj"))
        if not obj_files:
            print(f"Error: No OBJ files found in {temp_obj_dir}")
            return False
        # Find closest frame
        obj_path = obj_files[min(frame_idx, len(obj_files) - 1)]
        print(f"  Using {obj_path.name} (closest to frame {frame_idx})")

    # Read vertices from OBJ file
    print(f"Reading vertices from {obj_path.name}...")
    vertices = []
    with open(obj_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    vertices = np.array(vertices)
    print(f"  Found {len(vertices)} vertices")

    # Subsample if too many vertices
    if len(vertices) > num_samples:
        indices = np.random.choice(len(vertices), num_samples, replace=False)
        vertices = vertices[indices]
        print(f"  Subsampled to {num_samples} vertices")

    # Create init_pt_cld format: [x, y, z, r, g, b, seg]
    num_pts = len(vertices)
    # Default color: gray (0.5, 0.5, 0.5)
    colors = np.full((num_pts, 3), 0.5)
    # All points are foreground (seg=1)
    seg = np.ones((num_pts, 1))

    init_pt_cld = np.concatenate([vertices, colors, seg], axis=1)  # (N, 7)

    # Save
    os.makedirs(output_dir, exist_ok=True)
    npz_path = os.path.join(output_dir, "init_pt_cld.npz")
    np.savez(npz_path, data=init_pt_cld)
    print(f"  Saved init_pt_cld.npz: {init_pt_cld.shape} to {npz_path}")

    # Cleanup
    shutil.rmtree(temp_obj_dir)

    return True


def convert_actorshq_to_dynamic3dgs(
    input_dir: str,
    output_dir: str,
    scale: str = "4x",
    num_frames: Optional[int] = None,
    num_cameras: Optional[int] = None,
    frame_step: int = 1,
    image_scale: float = 1.0,
    only_complete_cameras: bool = True,
    crop_to_square: bool = True,
    start_frame: int = 0,
):
    """
    Convert ActorsHQ dataset to Dynamic3DGaussians format.

    Args:
        input_dir: Path to ActorsHQ sequence directory (e.g., Actor01/Sequence1)
        output_dir: Output directory for converted data
        scale: Resolution scale ("1x", "2x", "4x")
        num_frames: Number of frames to process (None = all)
        num_cameras: Number of cameras to use (None = all)
        frame_step: Process every N-th frame
        image_scale: Additional scale factor for output images
        only_complete_cameras: Only use cameras with complete frame data
    """
    print("Loading ActorsHQ data...")

    # Read scene info
    scene_json = os.path.join(input_dir, "scene.json")
    with open(scene_json, 'r') as f:
        scene_info = json.load(f)
    total_frames = scene_info['num_frames']
    print(f"  Total frames in scene: {total_frames}")

    # Read calibration
    calib_path = os.path.join(input_dir, scale, "calibration.csv")
    all_cameras = read_calibration_csv(calib_path)
    print(f"  Total cameras in calibration: {len(all_cameras)}")

    # Filter cameras with complete data
    if only_complete_cameras:
        complete_cameras = []
        for cam in all_cameras:
            is_complete, frame_count = check_camera_completeness(
                input_dir, cam['name'], total_frames, scale
            )
            if is_complete:
                complete_cameras.append(cam)
        print(f"  Cameras with complete data: {len(complete_cameras)}/{len(all_cameras)}")
        cameras = complete_cameras
    else:
        cameras = all_cameras

    if len(cameras) == 0:
        print("Error: No cameras with complete data found!")
        sys.exit(1)

    # Handle mixed resolutions (ActorsHQ has landscape 1022x747 and portrait 747x1022 cameras)
    if crop_to_square:
        # Crop all cameras to square - use ALL cameras
        cameras, crop_offsets = crop_cameras_to_square(cameras)
        print(f"  Cropped all {len(cameras)} cameras to square: {cameras[0]['width']}x{cameras[0]['height']}")
    else:
        # Filter to same resolution - discard some cameras
        cameras = filter_cameras_by_resolution(cameras)
        crop_offsets = [(0, 0)] * len(cameras)

        if len(cameras) == 0:
            print("Error: No cameras after resolution filtering!")
            sys.exit(1)

    # Limit number of cameras
    if num_cameras is not None:
        cameras = cameras[:num_cameras]
    print(f"  Using {len(cameras)} cameras")

    # Determine frames to process
    end_frame = total_frames
    if num_frames is not None:
        end_frame = min(start_frame + num_frames, total_frames)
    frame_indices = list(range(start_frame, end_frame, frame_step))
    print(f"  Processing {len(frame_indices)} frames (range={start_frame}-{end_frame}, step={frame_step})")

    # Get image dimensions from first camera
    first_cam = cameras[0]
    orig_w, orig_h = first_cam['width'], first_cam['height']

    # Apply additional scaling
    if image_scale != 1.0:
        new_w = int(orig_w * image_scale)
        new_h = int(orig_h * image_scale)
    else:
        new_w, new_h = orig_w, orig_h
    print(f"  Original resolution: {orig_w}x{orig_h}")
    print(f"  Output resolution: {new_w}x{new_h}")

    # Create output directories
    ims_dir = os.path.join(output_dir, "ims")
    seg_dir = os.path.join(output_dir, "seg")
    os.makedirs(ims_dir, exist_ok=True)
    os.makedirs(seg_dir, exist_ok=True)

    # Process cameras
    k_all = []  # [num_frames][num_cameras]
    w2c_all = []  # [num_frames][num_cameras]
    fn_all = []  # [num_frames][num_cameras]

    # Pre-compute camera matrices (they don't change per frame)
    camera_matrices = []
    for cam in cameras:
        # Compute intrinsic matrix and scale if needed
        K = compute_intrinsic_matrix(cam)
        if image_scale != 1.0:
            K[0, :] *= image_scale  # fx, cx
            K[1, :] *= image_scale  # fy, cy

        # Compute w2c matrix
        w2c = cam2world_to_w2c(cam['rotation_axisangle'], cam['translation'])

        camera_matrices.append((K, w2c))

    # Process frames
    print("\nProcessing frames...")
    for t_idx, frame_idx in enumerate(tqdm(frame_indices, desc="Frames")):
        k_frame = []
        w2c_frame = []
        fn_frame = []

        for cam_idx, cam in enumerate(cameras):
            cam_name = cam['name']
            K, w2c = camera_matrices[cam_idx]

            # Create camera subdirectory
            cam_ims_dir = os.path.join(ims_dir, str(cam_idx))
            cam_seg_dir = os.path.join(seg_dir, str(cam_idx))

            if t_idx == 0:
                os.makedirs(cam_ims_dir, exist_ok=True)
                os.makedirs(cam_seg_dir, exist_ok=True)

            # Source file paths
            src_rgb = os.path.join(
                input_dir, scale, "rgbs", cam_name,
                f"{cam_name}_rgb{frame_idx:06d}.jpg"
            )
            src_mask = os.path.join(
                input_dir, scale, "masks", cam_name,
                f"{cam_name}_mask{frame_idx:06d}.png"
            )

            # Output file paths
            out_filename = f"{frame_idx:06d}.jpg"
            out_rgb = os.path.join(cam_ims_dir, out_filename)
            out_mask = os.path.join(cam_seg_dir, f"{frame_idx:06d}.png")

            # Get crop offset for this camera
            crop_w, crop_h = crop_offsets[cam_idx]
            crop_size = cameras[cam_idx]['width']  # After cropping, width == height

            # Load and process RGB image
            if os.path.exists(src_rgb):
                img = cv2.imread(src_rgb)
                if img is not None:
                    # Apply center crop if needed
                    if crop_w > 0 or crop_h > 0:
                        img = img[crop_h:crop_h+crop_size, crop_w:crop_w+crop_size]
                    # Apply additional scaling
                    if image_scale != 1.0:
                        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    Image.fromarray(img).save(out_rgb, quality=95)

            # Load and process mask
            if os.path.exists(src_mask):
                mask = cv2.imread(src_mask, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    # Apply center crop if needed
                    if crop_w > 0 or crop_h > 0:
                        mask = mask[crop_h:crop_h+crop_size, crop_w:crop_w+crop_size]
                    # Apply additional scaling
                    if image_scale != 1.0:
                        mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                    # Ensure binary mask
                    mask = (mask > 127).astype(np.uint8) * 255
                    Image.fromarray(mask).save(out_mask)

            # Store camera parameters
            k_frame.append(K.tolist())
            w2c_frame.append(w2c.tolist())
            fn_frame.append(f"{cam_idx}/{out_filename}")

        k_all.append(k_frame)
        w2c_all.append(w2c_frame)
        fn_all.append(fn_frame)

    # Create metadata
    train_meta = {
        'w': new_w,
        'h': new_h,
        'k': k_all,
        'w2c': w2c_all,
        'fn': fn_all,
    }

    # Save metadata
    train_meta_path = os.path.join(output_dir, "train_meta.json")
    with open(train_meta_path, 'w') as f:
        json.dump(train_meta, f)
    print(f"\nSaved train_meta.json")

    # Also save test_meta.json (same as train for now)
    test_meta_path = os.path.join(output_dir, "test_meta.json")
    with open(test_meta_path, 'w') as f:
        json.dump(train_meta, f)
    print(f"Saved test_meta.json")

    # Print summary
    print("\n" + "=" * 60)
    print("Conversion complete!")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Frames: {len(frame_indices)}")
    print(f"Cameras: {len(cameras)}")
    print(f"Resolution: {new_w}x{new_h}")

    # Verify principal point is centered
    K_sample = np.array(k_all[0][0])
    cx, cy = K_sample[0, 2], K_sample[1, 2]
    print(f"\nCamera parameters check:")
    print(f"  Principal point: ({cx:.1f}, {cy:.1f})")
    print(f"  Image center: ({new_w/2:.1f}, {new_h/2:.1f})")
    print(f"  Offset: ({cx - new_w/2:.1f}, {cy - new_h/2:.1f})")

    return frame_indices


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ActorsHQ to Dynamic3DGaussians format")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Path to ActorsHQ sequence directory (e.g., Actor01/Sequence1)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for converted data")
    parser.add_argument("--scale", type=str, default="4x",
                        choices=["1x", "2x", "4x"],
                        help="Resolution scale to use (default: 4x)")
    parser.add_argument("--num_frames", type=int, default=None,
                        help="Number of frames to process (default: all)")
    parser.add_argument("--num_cameras", type=int, default=None,
                        help="Number of cameras to use (default: all complete cameras)")
    parser.add_argument("--frame_step", type=int, default=1,
                        help="Process every N-th frame (default: 1)")
    parser.add_argument("--image_scale", type=float, default=1.0,
                        help="Additional scale factor for output images (default: 1.0)")
    parser.add_argument("--start_frame", type=int, default=0,
                        help="Starting frame index (default: 0)")
    parser.add_argument("--allow_incomplete", action="store_true",
                        help="Allow cameras with incomplete frame data")
    parser.add_argument("--no_crop", action="store_true",
                        help="Don't crop to square (filter by resolution instead)")
    parser.add_argument("--init_mesh", action="store_true",
                        help="Generate init_pt_cld.npz from mesh (meshes.abc)")
    parser.add_argument("--num_mesh_samples", type=int, default=100000,
                        help="Number of mesh vertices to sample for init_pt_cld (default: 100000)")
    parser.add_argument("--alembic-extractor", default=os.environ.get("GA_ALEMBIC_EXTRACTOR"),
                        help="Official HumanRF mesh extraction executable (required with --init_mesh)")

    args = parser.parse_args()

    frame_indices = convert_actorshq_to_dynamic3dgs(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        scale=args.scale,
        num_frames=args.num_frames,
        num_cameras=args.num_cameras,
        frame_step=args.frame_step,
        image_scale=args.image_scale,
        only_complete_cameras=not args.allow_incomplete,
        crop_to_square=not args.no_crop,
        start_frame=args.start_frame,
    )

    # Generate init_pt_cld from mesh if requested
    if args.init_mesh and frame_indices is not None:
        print("\nGenerating initial point cloud from mesh...")
        generate_init_pointcloud_from_mesh(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            frame_idx=frame_indices[0],  # Use first frame's mesh
            num_samples=args.num_mesh_samples,
            alembic_extractor_path=args.alembic_extractor,
        )
