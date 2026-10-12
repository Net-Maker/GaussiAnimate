"""Convert DeformingThings4D ANIME files to registered mesh trajectories."""

import argparse
from pathlib import Path

import numpy as np


def load_anime(path):
    with Path(path).open("rb") as handle:
        header = np.fromfile(handle, dtype=np.int32, count=3)
        if len(header) != 3:
            raise ValueError("ANIME file is missing its frame/vertex/triangle header")
        frame_count, vertex_count, triangle_count = map(int, header)
        if frame_count < 1 or vertex_count < 1 or triangle_count < 0:
            raise ValueError("ANIME header contains invalid counts")
        rest = np.fromfile(handle, dtype=np.float32, count=vertex_count * 3)
        faces = np.fromfile(handle, dtype=np.int32, count=triangle_count * 3)
        if len(rest) != vertex_count * 3 or len(faces) != triangle_count * 3:
            raise ValueError("ANIME rest mesh payload is truncated")
        rest = rest.reshape(vertex_count, 3)
        faces = faces.reshape(triangle_count, 3)
        expected_offsets = (frame_count - 1) * vertex_count * 3
        offsets = np.fromfile(handle, dtype=np.float32)
        if len(offsets) != expected_offsets:
            raise ValueError("ANIME deformation payload size does not match its header")
        offsets = offsets.reshape(frame_count - 1, vertex_count, 3)
    if not np.isfinite(rest).all() or not np.isfinite(offsets).all():
        raise ValueError("ANIME geometry contains NaN or infinite values")
    if faces.size and (faces.min() < 0 or faces.max() >= vertex_count):
        raise ValueError("ANIME faces contain an out-of-range vertex index")
    points = np.empty((frame_count, vertex_count, 3), dtype=np.float32)
    points[0] = rest
    points[1:] = rest[None] + offsets
    return points, faces


def export_dt4d_points(anime_path, output_path, frame_stride: int = 1, fps: float = 30.0) -> Path:
    if frame_stride < 1:
        raise ValueError("frame_stride must be positive")
    points, faces = load_anime(anime_path)
    frame_indices = np.arange(0, len(points), frame_stride)
    if frame_indices[-1] != len(points) - 1:
        frame_indices = np.append(frame_indices, len(points) - 1)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        points=points[frame_indices],
        faces=faces,
        source_frame_indices=frame_indices,
        fps=np.asarray(fps / frame_stride),
        source=np.asarray(str(Path(anime_path))),
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a DT4D .anime mesh sequence")
    parser.add_argument("anime")
    parser.add_argument("output")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()
    print(export_dt4d_points(args.anime, args.output, args.frame_stride, args.fps))


if __name__ == "__main__":
    main()
