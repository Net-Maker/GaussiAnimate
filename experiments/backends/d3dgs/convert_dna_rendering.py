"""
Convert DNA-Rendering data to Dynamic3DGaussians format.

DNA-Rendering format:
    - {sequence}.smc: Main file with RGB images (Camera_5mp)
    - {sequence}_annots.smc: Annotations with camera params, masks, SMPLX

Dynamic3DGaussians format:
    ./data/{sequence}/
        ├── train_meta.json      # Metadata: w, h, k[t][c], w2c[t][c], fn[t][c]
        ├── init_pt_cld.npz      # Initial point cloud: (N, 7) [x,y,z,r,g,b,seg]
        ├── ims/                 # RGB images
        │   └── {t:05d}_{c:02d}.jpg
        └── seg/                 # Segmentation masks
            └── {t:05d}_{c:02d}.png

Usage:
    python convert_dna_rendering.py --main_smc /path/to/sequence.smc --output_dir ./data/sequence_name
"""

import os
import sys
import json
import argparse
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

def _reader_class():
    reader_root = os.environ.get("GA_DNA_READER_ROOT")
    if reader_root:
        sys.path.insert(0, os.path.expanduser(reader_root))
    from SMCReader import SMCReader

    return SMCReader


def get_smplx_vertices(annot_reader, main_reader, frame_idx=0):
    """Get SMPLX vertices for initial point cloud."""
    smplx_model_path = os.environ.get("GA_SMPLX_MODEL_ROOT")
    if not smplx_model_path:
        raise ValueError("set GA_SMPLX_MODEL_ROOT to a directory containing smplx/")

    # Check if smplx is available
    smplx_available = False
    try:
        import torch
        from smplx import SMPLX
        smplx_available = True
    except ImportError:
        print("Warning: smplx not available, will use random point cloud")
        return None

    gender = main_reader.actor_info['gender']
    model = SMPLX(
        os.path.join(smplx_model_path, 'smplx'),
        gender=gender,
        use_face_contour=True,
        flat_hand_mean=False,
        use_pca=False,
        num_betas=10,
        num_expression_coeffs=10,
        ext='npz'
    )

    smplx_dict = annot_reader.get_SMPLx(Frame_id=frame_idx)
    if smplx_dict is None:
        print("Warning: SMPLx data not available")
        return None

    import torch
    betas = torch.from_numpy(smplx_dict["betas"]).unsqueeze(0).float()
    expression = torch.from_numpy(smplx_dict["expression"]).unsqueeze(0).float()
    fullpose = torch.from_numpy(smplx_dict["fullpose"]).unsqueeze(0).float()
    translation = torch.from_numpy(smplx_dict['transl']).unsqueeze(0).float()

    output = model(
        betas=betas,
        expression=expression,
        global_orient=fullpose[:, 0].clone(),
        body_pose=fullpose[:, 1:22].clone(),
        jaw_pose=fullpose[:, 22].clone(),
        leye_pose=fullpose[:, 23].clone(),
        reye_pose=fullpose[:, 24].clone(),
        left_hand_pose=fullpose[:, 25:40].clone(),
        right_hand_pose=fullpose[:, 40:55].clone(),
        transl=translation,
        return_verts=True
    )

    vertices = output.vertices.detach().cpu().numpy().squeeze()
    # No coordinate conversion needed since we use RT directly as w2c
    return vertices


def create_initial_pointcloud(annot_reader, main_reader, num_cameras=48, frame_idx=0):
    """
    Create initial point cloud from SMPLX vertices.
    Returns: (N, 7) array with [x, y, z, r, g, b, seg_label]
    """
    # Try to get SMPLX vertices
    vertices = get_smplx_vertices(annot_reader, main_reader, frame_idx)

    if vertices is None:
        # Fallback: create random point cloud around human center
        print("Using random point cloud as fallback")
        # Get human center from SMPLX translation if available
        import h5py
        try:
            annot_path = annot_reader.smc_path if hasattr(annot_reader, 'smc_path') else None
            if annot_path is None:
                # Try to find annotation file
                import os
                main_smc = main_reader.smc_path if hasattr(main_reader, 'smc_path') else None
                if main_smc:
                    base = os.path.dirname(main_smc)
                    annot_path = os.path.join(base, 'data_used_in_4K4D/annotations',
                                              os.path.basename(main_smc).replace('-018.smc', '_annots.smc'))
            if annot_path and os.path.exists(annot_path):
                with h5py.File(annot_path, 'r') as f:
                    human_center = f['SMPLx/transl'][0]
                    print(f"Human center from SMPLX: {human_center}")
            else:
                human_center = np.array([0.14, 0.3, 0.05])  # Default from DNA-Rendering
                print(f"Using default human center: {human_center}")
        except:
            human_center = np.array([0.14, 0.3, 0.05])
            print(f"Using default human center: {human_center}")

        num_pts = 100000  # Increased from 10000 for better coverage
        # Create ellipsoid around human (taller than wide)
        xyz = np.random.randn(num_pts, 3)
        xyz[:, 0] *= 0.35  # Width (slightly wider)
        xyz[:, 1] *= 1.0   # Height (full body height)
        xyz[:, 2] *= 0.25  # Depth (slightly deeper)
        xyz = xyz + human_center
        # Use skin-like default colors instead of random
        rgb = np.ones((num_pts, 3)) * 0.6 + np.random.rand(num_pts, 3) * 0.2
        seg = np.ones(num_pts)  # All foreground
        return np.concatenate([xyz, rgb, seg[:, None]], axis=1)

    # SMPLX vertices are available - densify by replication with jitter
    num_copies = 10  # Replicate each vertex 10 times
    num_pts = vertices.shape[0] * num_copies

    # Replicate vertices and add small jitter
    vertices_dense = np.tile(vertices, (num_copies, 1))
    jitter = np.random.randn(num_pts, 3) * 0.005  # Small noise (5mm)
    vertices_dense = vertices_dense + jitter

    print(f"Densified SMPLX: {vertices.shape[0]} -> {num_pts} points")

    # Initialize colors with slight variation
    rgb = np.ones((num_pts, 3)) * 0.6 + np.random.rand(num_pts, 3) * 0.2

    # All SMPLX vertices are foreground (seg=1)
    seg = np.ones(num_pts)

    # Combine: [x, y, z, r, g, b, seg]
    init_pt_cld = np.concatenate([vertices_dense, rgb, seg[:, None]], axis=1)

    return init_pt_cld.astype(np.float32)


def convert_dna_to_dynamic3dgs(
    main_smc_path: str,
    annot_smc_path: str,
    output_dir: str,
    image_scale: float = 0.25,
    num_frames: int = None,
    num_cameras: int = None,
    frame_step: int = 1,
    camera_ids: list = None
):
    """
    Convert DNA-Rendering SMC files to Dynamic3DGaussians format.

    Args:
        main_smc_path: Path to main SMC file with RGB images
        annot_smc_path: Path to annotation SMC file with camera params and masks
        output_dir: Output directory for converted data
        image_scale: Scale factor for images (default 0.25 for memory efficiency)
        num_frames: Number of frames to process (None = all)
        num_cameras: Number of cameras to use (None = all 48)
        frame_step: Process every N-th frame
        camera_ids: Specific camera IDs to use (None = use first num_cameras)
    """
    print(f"Loading SMC files...")
    SMCReader = _reader_class()
    main_reader = SMCReader(main_smc_path)
    annot_reader = SMCReader(annot_smc_path)

    # Get data info
    cam_info = main_reader.get_Camera_5mp_info()
    total_cameras = cam_info['num_device']
    total_frames = cam_info['num_frame']
    resolution = cam_info['resolution']  # [W, H]

    print(f"Data info:")
    print(f"  - Total cameras: {total_cameras}")
    print(f"  - Total frames: {total_frames}")
    print(f"  - Resolution: {resolution[0]}x{resolution[1]}")
    print(f"  - Actor: {main_reader.actor_info}")

    # Determine cameras to use
    if camera_ids is not None:
        cameras = camera_ids
    elif num_cameras is not None:
        cameras = list(range(min(num_cameras, total_cameras)))
    else:
        cameras = list(range(total_cameras))

    # Determine frames to process
    if num_frames is not None:
        frames = list(range(0, min(num_frames * frame_step, total_frames), frame_step))
    else:
        frames = list(range(0, total_frames, frame_step))

    print(f"Processing {len(frames)} frames with {len(cameras)} cameras")
    print(f"Image scale: {image_scale}")

    # Create output directories
    ims_dir = os.path.join(output_dir, 'ims')
    seg_dir = os.path.join(output_dir, 'seg')
    os.makedirs(ims_dir, exist_ok=True)
    os.makedirs(seg_dir, exist_ok=True)

    # Get scaled dimensions
    orig_w, orig_h = resolution
    new_w = int(orig_w * image_scale)
    new_h = int(orig_h * image_scale)

    # Create subdirectories for each camera (matching original D3DGS format)
    # Original format: ims/{camera_id}/{frame:06d}.jpg, seg/{camera_id}/{frame:06d}.png
    for c_idx in range(len(cameras)):
        os.makedirs(os.path.join(ims_dir, str(c_idx)), exist_ok=True)
        os.makedirs(os.path.join(seg_dir, str(c_idx)), exist_ok=True)

    # Initialize metadata structure
    # fn[t][c], k[t][c], w2c[t][c]
    all_fn = []
    all_k = []
    all_w2c = []

    # Process each frame (timestep)
    for t_idx, frame_id in enumerate(tqdm(frames, desc="Processing frames")):
        fn_t = []
        k_t = []
        w2c_t = []

        for c_idx, cam_id in enumerate(cameras):
            # Get camera parameters
            cam_params = annot_reader.get_Calibration(cam_id)
            K = cam_params['K'].copy()
            D = cam_params['D']
            RT = cam_params['RT']  # This is camera-to-world (c2w)

            # DNA-Rendering RT matrix handling:
            # RT is c2w (camera-to-world) in DNA-Rendering
            # We need w2c for Dynamic3DGaussians in standard 4x4 format
            c2w = np.array(RT, dtype=np.float64)
            w2c = np.linalg.inv(c2w)

            # FIX: Create centered K for undistortion
            # The 3DGS rasterizer assumes principal point at image center
            # We need to undistort such that the output has centered principal point
            K_centered = K.copy()
            K_centered[0, 2] = orig_w / 2  # cx at center
            K_centered[1, 2] = orig_h / 2  # cy at center

            # Create undistortion maps (computed once per camera, reused across frames)
            # This maps from distorted image (with original K) to undistorted image (with centered K)
            map1, map2 = cv2.initUndistortRectifyMap(
                K, D, None, K_centered, (orig_w, orig_h), cv2.CV_32FC1
            )

            # Scale K_centered for resized images (use centered K, not original K!)
            K_scaled = K_centered.copy()
            K_scaled[0, :] *= image_scale  # fx, cx
            K_scaled[1, :] *= image_scale  # fy, cy

            # Load and process image with CORRECT undistortion
            image = main_reader.get_img('Camera_5mp', cam_id, 'color', Frame_id=frame_id)
            image = cv2.remap(image, map1, map2, cv2.INTER_LINEAR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Load and process mask with same undistortion
            mask = annot_reader.get_mask(cam_id, Frame_id=frame_id)
            if mask is not None:
                mask = cv2.remap(mask, map1, map2, cv2.INTER_NEAREST)
            else:
                # If no mask, create all-foreground mask
                mask = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8) * 255

            # Resize
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

            # Save image using original D3DGS format: {camera_id}/{frame:06d}.jpg
            img_filename = f"{c_idx}/{t_idx:06d}.jpg"
            img_path = os.path.join(ims_dir, str(c_idx), f"{t_idx:06d}.jpg")
            Image.fromarray(image).save(img_path, quality=95)

            # Save segmentation mask using original D3DGS format: {camera_id}/{frame:06d}.png
            # Original D3DGS uses bool dtype for masks
            seg_path = os.path.join(seg_dir, str(c_idx), f"{t_idx:06d}.png")
            # Binarize mask and save as bool (matching original format)
            mask_binary = (mask > 127)  # bool type: True/False
            Image.fromarray(mask_binary).save(seg_path)

            # Collect metadata
            fn_t.append(img_filename)
            k_t.append(K_scaled.tolist())
            w2c_t.append(w2c.tolist())

        all_fn.append(fn_t)
        all_k.append(k_t)
        all_w2c.append(w2c_t)

    # Create train_meta.json
    train_meta = {
        'w': new_w,
        'h': new_h,
        'fn': all_fn,
        'k': all_k,
        'w2c': all_w2c
    }

    meta_path = os.path.join(output_dir, 'train_meta.json')
    with open(meta_path, 'w') as f:
        json.dump(train_meta, f)
    print(f"Saved train_meta.json")

    # Create initial point cloud
    print("Creating initial point cloud from SMPLX...")
    init_pt_cld = create_initial_pointcloud(annot_reader, main_reader, len(cameras), frames[0])

    npz_path = os.path.join(output_dir, 'init_pt_cld.npz')
    np.savez(npz_path, data=init_pt_cld)
    print(f"Saved init_pt_cld.npz with {init_pt_cld.shape[0]} points")

    # Clean up
    main_reader.release()
    annot_reader.release()

    print(f"\nConversion complete!")
    print(f"Output directory: {output_dir}")
    print(f"  - {len(frames)} timesteps")
    print(f"  - {len(cameras)} cameras per timestep")
    print(f"  - Image size: {new_w}x{new_h}")

    return output_dir


def find_annotation_file(main_smc_path):
    """Find the corresponding annotation SMC file for a main SMC file."""
    # Try data_used_in_4K4D structure first
    base_dir = os.path.dirname(main_smc_path)
    base_name = os.path.basename(main_smc_path)

    # Extract sequence ID (e.g., "0012_11" from "0012_11-014.smc")
    seq_id = base_name.split('-')[0]

    # Check data_used_in_4K4D/annotations/
    annot_dir = os.path.join(base_dir, 'data_used_in_4K4D', 'annotations')
    annot_file = os.path.join(annot_dir, f"{seq_id}_annots.smc")
    if os.path.exists(annot_file):
        return annot_file

    # Check parent directory
    parent_dir = os.path.dirname(base_dir)
    annot_dir = os.path.join(parent_dir, 'annotations')
    annot_file = os.path.join(annot_dir, f"{seq_id}_annots.smc")
    if os.path.exists(annot_file):
        return annot_file

    # Try same directory with _annots suffix
    annot_file = main_smc_path.replace('.smc', '_annots.smc')
    if os.path.exists(annot_file):
        return annot_file

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert DNA-Rendering to Dynamic3DGaussians format")
    parser.add_argument("--main_smc", type=str, required=True,
                        help="Path to main SMC file with RGB images")
    parser.add_argument("--annot_smc", type=str, default=None,
                        help="Path to annotation SMC file (auto-detected if not provided)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for converted data")
    parser.add_argument("--scale", type=float, default=0.25,
                        help="Image scale factor (default: 0.25)")
    parser.add_argument("--num_frames", type=int, default=None,
                        help="Number of frames to process (default: all)")
    parser.add_argument("--num_cameras", type=int, default=None,
                        help="Number of cameras to use (default: all 48)")
    parser.add_argument("--frame_step", type=int, default=1,
                        help="Process every N-th frame (default: 1)")
    parser.add_argument("--dna-reader-root", default=os.environ.get("GA_DNA_READER_ROOT"),
                        help="Directory containing the official SMCReader.py")
    parser.add_argument("--smplx-model-root", default=os.environ.get("GA_SMPLX_MODEL_ROOT"),
                        help="Licensed body-model directory containing smplx/")

    args = parser.parse_args()
    if not args.dna_reader_root or not args.smplx_model_root:
        parser.error("provide --dna-reader-root and --smplx-model-root, or their GA_* environment variables")
    os.environ["GA_DNA_READER_ROOT"] = os.path.expanduser(args.dna_reader_root)
    os.environ["GA_SMPLX_MODEL_ROOT"] = os.path.expanduser(args.smplx_model_root)

    # Auto-detect annotation file if not provided
    annot_smc = args.annot_smc
    if annot_smc is None:
        annot_smc = find_annotation_file(args.main_smc)
        if annot_smc is None:
            print("Error: Could not find annotation SMC file. Please specify with --annot_smc")
            sys.exit(1)
        print(f"Auto-detected annotation file: {annot_smc}")

    convert_dna_to_dynamic3dgs(
        main_smc_path=args.main_smc,
        annot_smc_path=annot_smc,
        output_dir=args.output_dir,
        image_scale=args.scale,
        num_frames=args.num_frames,
        num_cameras=args.num_cameras,
        frame_step=args.frame_step
    )
