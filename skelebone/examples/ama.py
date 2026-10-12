"""Extract and animate Skelebones from a registered AMA mesh sequence."""

import argparse
import os
from pathlib import Path

import numpy as np

from skelebone.types import DynamicPointSequence
from skelebone.io import save_point_sequence
from skelebone.kinematics import forward_kinematics
from skelebone.open3d_renderer import (
    outer_bone_positions,
    render_skelebone_animation,
)
from skelebone.pipeline import ExtractionConfig, SkelebonePipeline
from skelebone.visualization import compute_y_up_camera, parent_edges


DEFAULT_CASE = "T_samba"


def _load_obj(path):
    vertices = []
    faces = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("v "):
                vertices.append([float(value) for value in line.split()[1:4]])
            elif line.startswith("f "):
                indices = [int(token.split("/")[0]) - 1 for token in line.split()[1:]]
                if len(indices) != 3:
                    raise ValueError(f"AMA mesh faces must be triangles: {path}")
                faces.append(indices)
    if not vertices or not faces:
        raise ValueError(f"AMA OBJ contains no triangle mesh: {path}")
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def load_ama_sequence(ama_root, case=DEFAULT_CASE, fps=30.0):
    """Load a registered AMA OBJ sequence into the public point contract."""
    mesh_directory = Path(ama_root).expanduser() / case / "meshes"
    paths = sorted(mesh_directory.glob("mesh_*.obj"))
    if not paths:
        raise FileNotFoundError(f"missing AMA mesh sequence: {mesh_directory}")
    first_points, faces = _load_obj(paths[0])
    frames = [first_points]
    for path in paths[1:]:
        points, frame_faces = _load_obj(path)
        if points.shape != first_points.shape or not np.array_equal(frame_faces, faces):
            raise ValueError(f"AMA sequence is not registered at {path}")
        frames.append(points)
    return DynamicPointSequence(np.stack(frames), faces=faces, fps=fps)


def run(
    output_directory,
    ama_root,
    *,
    case=DEFAULT_CASE,
    fps=30.0,
    num_bones=50,
    max_iterations=50,
    max_preview_frames=40,
    preview_fps=12.0,
    curve_skeleton_bin=None,
):
    """Run the complete public extraction and produce an Open3D GIF."""
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    sequence = load_ama_sequence(ama_root, case=case, fps=fps)
    input_path = save_point_sequence(output_directory / "sequence.npz", sequence)
    result = SkelebonePipeline(
        ExtractionConfig(
            num_bones=num_bones,
            ssdr_max_iterations=max_iterations,
            curve_skeleton_bin=curve_skeleton_bin,
        )
    ).run(sequence, output_directory / "result")

    bundle = result.bundle
    if bundle.motion is None or bundle.bone_transforms is None:
        raise RuntimeError("AMA extraction did not produce motion and outer bones")
    _, joint_positions = forward_kinematics(bundle.skeleton, bundle.motion)
    outer_positions = outer_bone_positions(
        bundle.source_points, bundle.weights, bundle.bone_transforms
    )
    camera_points = np.concatenate(
        [sequence.points.reshape(-1, 3), joint_positions.reshape(-1, 3)], axis=0
    )
    camera = compute_y_up_camera(
        camera_points,
        azimuth_degrees=65.0,
        elevation_degrees=2.0,
        fit_scale=1.32,
    )
    preview_path = render_skelebone_animation(
        points=sequence.points,
        faces=sequence.faces,
        outer_positions=outer_positions,
        outer_rotations=bundle.bone_transforms[:, :, :3, :3],
        joints=joint_positions,
        joint_edges=parent_edges(bundle.skeleton.parents),
        camera=camera,
        output_path=output_directory / "ama_skelebone.gif",
        width=380,
        height=480,
        max_frames=max_preview_frames,
        fps=preview_fps,
    )
    return {
        "input": input_path,
        "bundle": result.bundle_path,
        "bvh": result.bvh_path,
        "manifest": result.manifest_path,
        "preview": preview_path,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract Skelebones from the registered AMA T_samba sequence."
    )
    parser.add_argument("output", nargs="?", default="outputs/ama_skelebone")
    parser.add_argument(
        "--ama-root",
        default=os.environ.get("GA_AMA_ROOT", "~/research/data/AMA"),
    )
    parser.add_argument("--case", default=DEFAULT_CASE)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--num-bones", type=int, default=50)
    parser.add_argument("--max-iterations", type=int, default=50)
    parser.add_argument("--max-preview-frames", type=int, default=40)
    parser.add_argument("--preview-fps", type=float, default=12.0)
    parser.add_argument("--curve-skeleton-bin")
    args = parser.parse_args()
    paths = run(
        args.output,
        args.ama_root,
        case=args.case,
        fps=args.fps,
        num_bones=args.num_bones,
        max_iterations=args.max_iterations,
        max_preview_frames=args.max_preview_frames,
        preview_fps=args.preview_fps,
        curve_skeleton_bin=args.curve_skeleton_bin,
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
