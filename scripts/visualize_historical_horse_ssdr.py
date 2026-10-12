"""Render the historical GaussianAnimator Horse SSDR decomposition."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from skelebone.extraction.ssdr import linear_blend_skinning
from skelebone.open3d_renderer import render_ssdr_comparison_animation
from skelebone.visualization import compute_y_up_camera


DEFAULT_TRAJECTORY = Path("outputs/horse_ssdr_debug/historical_trajectory_40k.npy")
DEFAULT_SSDR = Path(
    "/home/ubuntu/research/GA/GaussianAnimator/tempexp/exp-lbs-horse/ssdr_result.npy"
)
DEFAULT_OUTPUT = Path("outputs/horse_historical_ssdr_visualization")


def compute_horizontal_horse_camera(
    points,
    *,
    azimuth_degrees=78.0,
    elevation_degrees=4.0,
    fov_degrees=30.0,
    fit_scale=1.22,
):
    """View along the thin axis while placing Horse motion Y horizontally."""

    camera = compute_y_up_camera(
        points,
        azimuth_degrees=azimuth_degrees,
        elevation_degrees=elevation_degrees,
        fov_degrees=fov_degrees,
        fit_scale=fit_scale,
    )
    return replace(camera, up=np.array([0.0, 0.0, 1.0], dtype=np.float64))


def load_ssdr(path, point_count):
    artifact = np.load(path, allow_pickle=True).item()
    weights = artifact["weights"]
    if hasattr(weights, "toarray"):
        weights = weights.toarray()
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape[0] != point_count and weights.shape[1] == point_count:
        weights = weights.T
    if weights.shape[0] != point_count:
        raise ValueError("historical SSDR weights do not match the trajectory")
    return (
        np.asarray(artifact["rest_pose"], dtype=np.float64),
        weights,
        np.asarray(artifact["transformations"], dtype=np.float64),
    )


def run(args):
    trajectory = np.load(args.trajectory).astype(np.float64, copy=False)
    rest_pose, weights, transforms = load_ssdr(args.ssdr, trajectory.shape[1])
    reconstruction = linear_blend_skinning(rest_pose, weights, transforms)
    error = reconstruction - trajectory
    coordinate_rmse = float(np.sqrt(np.mean(error * error)))
    vertex_rmse = float(np.sqrt(np.mean(np.sum(error * error, axis=-1))))

    camera = compute_horizontal_horse_camera(
        trajectory.reshape(-1, 3),
        azimuth_degrees=args.azimuth,
        elevation_degrees=args.elevation,
        fov_degrees=args.fov,
        fit_scale=args.fit_scale,
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    gif_path = render_ssdr_comparison_animation(
        observed_points=trajectory,
        reconstructed_points=reconstruction,
        rest_pose=rest_pose,
        skinning_weights=weights,
        bone_transforms=transforms,
        camera=camera,
        output_path=output / "horse_historical_ssdr_30bones.gif",
        width=args.width,
        height=args.height,
        max_frames=args.max_frames,
        max_points=args.max_points,
        fps=args.fps,
    )

    from PIL import Image

    poster_path = output / "horse_historical_ssdr_30bones.png"
    with Image.open(gif_path) as image:
        image.seek(min(12, image.n_frames - 1))
        image.convert("RGB").save(poster_path, optimize=True)

    metrics = {
        "trajectory": str(Path(args.trajectory).resolve()),
        "ssdr_artifact": str(Path(args.ssdr).resolve()),
        "frames": int(len(trajectory)),
        "points": int(trajectory.shape[1]),
        "bones": int(weights.shape[1]),
        "coordinate_rmse": coordinate_rmse,
        "vertex_rmse": vertex_rmse,
        "camera": {
            "azimuth_degrees": args.azimuth,
            "elevation_degrees": args.elevation,
            "fov_degrees": args.fov,
            "fit_scale": args.fit_scale,
            "up_axis": "z",
        },
        "preview_frames": min(int(args.max_frames), len(trajectory)),
        "preview_fps": args.fps,
        "gif": str(gif_path.resolve()),
        "poster": str(poster_path.resolve()),
    }
    metrics_path = output / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return gif_path, poster_path, metrics_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, default=DEFAULT_TRAJECTORY)
    parser.add_argument("--ssdr", type=Path, default=DEFAULT_SSDR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--width", type=int, default=500)
    parser.add_argument("--height", type=int, default=560)
    parser.add_argument("--max-frames", type=int, default=48)
    parser.add_argument("--max-points", type=int, default=14000)
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--azimuth", type=float, default=78.0)
    parser.add_argument("--elevation", type=float, default=4.0)
    parser.add_argument("--fov", type=float, default=30.0)
    parser.add_argument("--fit-scale", type=float, default=1.22)
    args = parser.parse_args()
    for path in run(args):
        print(path)


if __name__ == "__main__":
    main()
