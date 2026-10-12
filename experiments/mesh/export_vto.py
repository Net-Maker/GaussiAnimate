"""Convert trusted VTO simulation pickles to registered point trajectories."""

import argparse
import pickle
from pathlib import Path
from typing import Optional, Sequence

import numpy as np


def _axis_angle_matrices(axis_angle: np.ndarray) -> np.ndarray:
    axis_angle = np.asarray(axis_angle, dtype=np.float64)
    if axis_angle.ndim != 2 or axis_angle.shape[1] != 3:
        raise ValueError("root axis-angle rotations must have shape [frames, 3]")
    angle = np.linalg.norm(axis_angle, axis=1)
    axis = np.divide(
        axis_angle,
        angle[:, None],
        out=np.zeros_like(axis_angle),
        where=angle[:, None] > 1e-12,
    )
    skew = np.zeros((len(axis), 3, 3), dtype=np.float64)
    skew[:, 0, 1] = -axis[:, 2]
    skew[:, 0, 2] = axis[:, 1]
    skew[:, 1, 0] = axis[:, 2]
    skew[:, 1, 2] = -axis[:, 0]
    skew[:, 2, 0] = -axis[:, 1]
    skew[:, 2, 1] = axis[:, 0]
    identity = np.broadcast_to(np.eye(3), skew.shape)
    return (
        identity
        + np.sin(angle)[:, None, None] * skew
        + (1.0 - np.cos(angle))[:, None, None] * (skew @ skew)
    )


def canonicalize_root_motion(
    vertices: np.ndarray,
    root_axis_angle: np.ndarray,
    translation: np.ndarray,
    pelvis_offset: Sequence[float],
) -> np.ndarray:
    """Remove SMPL root motion while preserving rotation around the pelvis."""

    vertices = np.asarray(vertices, dtype=np.float64)
    translation = np.asarray(translation, dtype=np.float64)
    pelvis = np.asarray(pelvis_offset, dtype=np.float64)
    if translation.shape != (len(vertices), 3):
        raise ValueError("translation must have shape [frames, 3]")
    if pelvis.shape != (3,) or not np.isfinite(pelvis).all():
        raise ValueError("pelvis_offset must contain three finite coordinates")
    rotation = _axis_angle_matrices(root_axis_angle)
    centered = vertices - translation[:, None] - pelvis
    return np.einsum("tni,tij->tnj", centered, rotation) + pelvis


def export_vto_points(
    pickle_path,
    output_path,
    frame_stride: int = 1,
    fps: float = 30.0,
    pelvis_offset: Optional[Sequence[float]] = None,
) -> Path:
    """Export one trusted VTO simulation; a pelvis offset enables canonicalization."""

    if frame_stride < 1:
        raise ValueError("frame_stride must be positive")
    if fps <= 0:
        raise ValueError("fps must be positive")
    source = Path(pickle_path)
    with source.open("rb") as handle:
        data = pickle.load(handle)
    if not isinstance(data, dict):
        raise ValueError("VTO pickle must contain a dictionary")
    missing = {"vertices", "faces"} - data.keys()
    if missing:
        raise ValueError(f"VTO pickle is missing fields: {', '.join(sorted(missing))}")
    points = np.asarray(data["vertices"], dtype=np.float64)
    raw_faces = np.asarray(data["faces"])
    if points.ndim != 3 or points.shape[2] != 3 or len(points) < 1:
        raise ValueError("VTO vertices must have shape [frames, vertices, 3]")
    if not np.isfinite(points).all():
        raise ValueError("VTO vertices contain NaN or infinite values")
    if raw_faces.ndim != 2 or raw_faces.shape[1] != 3:
        raise ValueError("VTO faces must have shape [triangles, 3]")
    if not np.isfinite(raw_faces).all() or not np.equal(raw_faces, np.round(raw_faces)).all():
        raise ValueError("VTO faces must contain finite integer indices")
    faces = raw_faces.astype(np.int64)
    if faces.size and (faces.min() < 0 or faces.max() >= points.shape[1]):
        raise ValueError("VTO faces contain an out-of-range vertex index")

    canonicalized = pelvis_offset is not None
    if canonicalized:
        if "pose" not in data or "translation" not in data:
            raise ValueError("root canonicalization requires VTO pose and translation fields")
        pose = np.asarray(data["pose"], dtype=np.float64)
        if pose.ndim != 2 or pose.shape[0] != len(points) or pose.shape[1] < 3:
            raise ValueError("VTO pose must have shape [frames, at_least_3]")
        points = canonicalize_root_motion(
            points, pose[:, :3], data["translation"], pelvis_offset
        )

    frame_indices = np.arange(0, len(points), frame_stride)
    if frame_indices[-1] != len(points) - 1:
        frame_indices = np.append(frame_indices, len(points) - 1)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        points=points[frame_indices].astype(np.float32),
        faces=faces,
        source_frame_indices=frame_indices,
        fps=np.asarray(fps / frame_stride),
        source=np.asarray(str(source)),
        sequence=np.asarray(str(data.get("sequence", ""))),
        subject=np.asarray(str(data.get("subject", ""))),
        root_canonicalized=np.asarray(canonicalized),
        pelvis_offset=np.asarray(
            pelvis_offset if pelvis_offset is not None else [0.0, 0.0, 0.0]
        ),
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a trusted VTO simulation pickle as registered cloth points"
    )
    parser.add_argument("pickle")
    parser.add_argument("output")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--pelvis-offset",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="Remove root motion around this SMPL pelvis; omit to keep world coordinates",
    )
    args = parser.parse_args()
    print(
        export_vto_points(
            args.pickle,
            args.output,
            frame_stride=args.frame_stride,
            fps=args.fps,
            pelvis_offset=args.pelvis_offset,
        )
    )


if __name__ == "__main__":
    main()
