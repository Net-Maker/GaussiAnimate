"""Export Dynamic3DGaussians checkpoints to the common point-sequence schema."""

import argparse
from pathlib import Path
from typing import Optional

import numpy as np


def export_d3dgs_points(
    params_path,
    output_path,
    foreground_only: bool = True,
    max_points: Optional[int] = None,
    seed: int = 42,
    fps: float = 30.0,
    frame_start: int = 0,
    frame_count: Optional[int] = None,
    frame_stride: int = 1,
    high_motion_window: Optional[int] = None,
) -> Path:
    if max_points is not None and max_points < 1:
        raise ValueError("max_points must be positive")
    if fps <= 0:
        raise ValueError("fps must be positive")
    if frame_start < 0 or frame_stride < 1:
        raise ValueError("frame_start must be non-negative and frame_stride positive")
    if frame_count is not None and frame_count < 1:
        raise ValueError("frame_count must be positive")
    if high_motion_window is not None and high_motion_window < 2:
        raise ValueError("high_motion_window must contain at least two frames")
    if high_motion_window is not None and (frame_start != 0 or frame_count is not None):
        raise ValueError("high_motion_window cannot be combined with frame_start/frame_count")
    data = np.load(params_path, allow_pickle=False)
    if "means3D" not in data:
        raise ValueError("Dynamic3DGaussians params must contain means3D")
    points = np.asarray(data["means3D"], dtype=np.float32)
    if points.ndim != 3 or points.shape[2] != 3:
        raise ValueError("means3D must have shape [frames, gaussians, 3]")
    if not np.isfinite(points).all():
        raise ValueError("means3D contains NaN or infinite values")
    indices = np.arange(points.shape[1])
    if foreground_only:
        if "seg_colors" not in data:
            raise ValueError("foreground export requires seg_colors")
        segmentation = data["seg_colors"]
        if segmentation.ndim == 3:
            segmentation = segmentation[0]
        if segmentation.ndim != 2 or len(segmentation) != points.shape[1]:
            raise ValueError("seg_colors must have one row per Gaussian")
        indices = indices[np.asarray(segmentation)[:, 0] > 0.5]
        if len(indices) == 0:
            raise ValueError("seg_colors selected no foreground Gaussians")
    if max_points is not None and len(indices) > max_points:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(indices, size=max_points, replace=False))
    selected_points = points[:, indices]
    if high_motion_window is not None:
        if high_motion_window > len(selected_points):
            raise ValueError("high_motion_window exceeds the checkpoint frame count")
        speed = np.linalg.norm(np.diff(selected_points, axis=0), axis=-1).mean(axis=1)
        cumulative = np.concatenate([[0.0], np.cumsum(speed)])
        scores = cumulative[high_motion_window - 1:] - cumulative[:1 - high_motion_window]
        frame_start = int(np.argmax(scores))
        frame_count = high_motion_window
    frame_stop = len(selected_points) if frame_count is None else min(
        len(selected_points), frame_start + frame_count
    )
    if frame_start >= len(selected_points):
        raise ValueError("frame_start is outside the checkpoint")
    frame_indices = np.arange(frame_start, frame_stop, frame_stride)
    if frame_indices[-1] != frame_stop - 1:
        frame_indices = np.append(frame_indices, frame_stop - 1)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        points=selected_points[frame_indices],
        source_indices=indices,
        source_frame_indices=frame_indices,
        fps=np.asarray(fps / frame_stride),
        source=np.asarray(str(Path(params_path))),
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Export D3DGS means3D as registered points")
    parser.add_argument("params")
    parser.add_argument("output")
    parser.add_argument("--include-background", action="store_true")
    parser.add_argument("--max-points", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-count", type=int)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--high-motion-window", type=int)
    args = parser.parse_args()
    path = export_d3dgs_points(
        args.params,
        args.output,
        foreground_only=not args.include_background,
        max_points=args.max_points,
        seed=args.seed,
        fps=args.fps,
        frame_start=args.frame_start,
        frame_count=args.frame_count,
        frame_stride=args.frame_stride,
        high_motion_window=args.high_motion_window,
    )
    print(path)


if __name__ == "__main__":
    main()
