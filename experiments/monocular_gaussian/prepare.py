"""Export a RigGS-style folder of registered frames to the common NPZ schema."""

import argparse
from pathlib import Path
from typing import Optional

import numpy as np


def _load_frame(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        points = np.load(path, allow_pickle=False)
    elif path.suffix.lower() == ".ply":
        try:
            import open3d as o3d
        except ImportError as exc:
            raise ImportError("reading RigGS PLY frames requires the 'geometry' extra") from exc
        cloud = o3d.io.read_point_cloud(str(path))
        if cloud.is_empty():
            raise ValueError(f"could not read point data from {path}")
        points = np.asarray(cloud.points)
    else:
        raise ValueError(f"unsupported point-frame extension: {path.suffix}")
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError(f"{path} must contain points with shape [N,3]")
    if not np.isfinite(points).all():
        raise ValueError(f"{path} contains NaN or infinite values")
    return points


def export_riggs_points(
    frame_dir,
    output_path,
    pattern: str = "frame_*.ply",
    max_points: Optional[int] = None,
    seed: int = 42,
    fps: float = 30.0,
) -> Path:
    if max_points is not None and max_points < 1:
        raise ValueError("max_points must be positive")
    if fps <= 0:
        raise ValueError("fps must be positive")
    directory = Path(frame_dir)
    files = sorted(directory.glob(pattern))
    if not files:
        raise ValueError(f"no frames matching {pattern!r} under {directory}")
    frames = [_load_frame(path) for path in files]
    point_count = len(frames[0])
    if any(len(frame) != point_count for frame in frames):
        raise ValueError("RigGS frames must have a constant registered point count")
    indices = np.arange(point_count)
    if max_points is not None and point_count > max_points:
        generator = np.random.default_rng(seed)
        indices = np.sort(generator.choice(indices, size=max_points, replace=False))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        points=np.stack([frame[indices] for frame in frames]),
        source_indices=indices,
        source_frames=np.asarray([path.name for path in files]),
        fps=np.asarray(fps),
        source=np.asarray(str(directory)),
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export registered RigGS deformed point-cloud frames"
    )
    parser.add_argument("frame_dir")
    parser.add_argument("output")
    parser.add_argument("--pattern", default="frame_*.ply")
    parser.add_argument("--max-points", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()
    print(
        export_riggs_points(
            args.frame_dir,
            args.output,
            pattern=args.pattern,
            max_points=args.max_points,
            seed=args.seed,
            fps=args.fps,
        )
    )


if __name__ == "__main__":
    main()
