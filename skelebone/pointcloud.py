"""Registered point-cloud import with a shared Gaussian attribute selection."""

import argparse
from pathlib import Path
import re

import numpy as np

from .types import DynamicPointSequence


def frame_sort_key(path):
    """Order numbered frames numerically, including unpadded filenames."""
    return tuple(int(part) if part.isdigit() else part for part in re.split(r"(\d+)", Path(path).name))


def _read_vertices(path):
    try:
        from plyfile import PlyData
    except ImportError as exc:
        raise ImportError("PLY input requires the 'geometry' extra (plyfile)") from exc
    return PlyData.read(str(path))["vertex"].data


def _points(path):
    if path.suffix.lower() == ".npy":
        values = np.load(path, allow_pickle=False)
    elif path.suffix.lower() == ".ply":
        vertices = _read_vertices(path)
        values = np.column_stack([vertices[name] for name in ("x", "y", "z")])
    else:
        raise ValueError("registered frames must be PLY or NPY files")
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 3 or not len(values) or not np.isfinite(values).all():
        raise ValueError(f"{path} must contain finite points with shape [N,3]")
    return values


def _appearance(path, count):
    vertices = _read_vertices(path)
    if len(vertices) != count:
        raise ValueError("Gaussian attribute count does not match registered point count")
    fields = set(vertices.dtype.names)
    attributes = {}
    if all(f"f_dc_{axis}" in fields for axis in range(3)):
        dc = np.column_stack([vertices[f"f_dc_{axis}"] for axis in range(3)])
        attributes["colors"] = np.clip(.5 + .28209479177387814 * dc, 0., 1.)
        attributes["sh_dc"] = dc
        rest = sorted((name for name in fields if name.startswith("f_rest_")), key=frame_sort_key)
        if rest:
            attributes["sh_rest"] = np.column_stack([vertices[name] for name in rest])
    elif all(name in fields for name in ("red", "green", "blue")):
        rgb = np.column_stack([vertices[name] for name in ("red", "green", "blue")])
        normalized = np.issubdtype(rgb.dtype, np.floating) and np.max(rgb) <= 1.
        attributes["colors"] = np.clip(rgb.astype(np.float32) / (1. if normalized else 255.), 0., 1.)
    if "opacity" in fields:
        raw = np.asarray(vertices["opacity"], dtype=np.float64)
        if not np.isfinite(raw).all():
            raise ValueError("Gaussian opacity contains nonfinite values")
        attributes["opacity"] = 1. / (1. + np.exp(-np.clip(raw, -80., 80.)))
    scales = sorted((name for name in fields if name.startswith("scale_")), key=frame_sort_key)
    if scales:
        with np.errstate(over="ignore", invalid="ignore"):
            attributes["scales"] = np.exp(np.column_stack([vertices[name] for name in scales]))
        if len(scales) == 1:
            attributes["scales"] = np.repeat(attributes["scales"], 3, axis=1)
        elif len(scales) != 3:
            raise ValueError("Gaussian scales must contain one or three components")
    rotations = [f"rot_{axis}" for axis in range(4)]
    if all(name in fields for name in rotations):
        quaternion = np.column_stack([vertices[name] for name in rotations])
        length = np.linalg.norm(quaternion, axis=1, keepdims=True)
        if np.any(length <= 1e-12):
            raise ValueError("Gaussian rotations contain a zero quaternion")
        attributes["rotations"] = quaternion / length
    for name, values in attributes.items():
        if not np.isfinite(values).all():
            raise ValueError(f"Gaussian {name} contains nonfinite values")
    return attributes


def load_registered_clouds(
    frame_dir, pattern="pointcloud_t*.ply", max_points=12000, seed=42,
    fps=30., gaussian_ply=None, min_opacity=.05,
):
    """Import stable vertex rows; select point IDs once for the entire sequence.

    Registration must be established by the reconstruction/exporter. Matching
    point counts alone cannot establish material-point correspondence.
    Gaussian PLY opacity is a logit, scales are logarithmic, and SH DC is decoded
    only for the RGB preview. The original SH coefficients are retained.
    """
    if max_points is not None and (not isinstance(max_points, (int, np.integer)) or max_points < 1):
        raise ValueError("max_points must be a positive integer or None")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    if not np.isfinite(min_opacity) or not 0 <= min_opacity < 1:
        raise ValueError("min_opacity must be in [0,1)")
    paths = sorted(Path(frame_dir).glob(pattern), key=frame_sort_key)
    if not paths:
        raise ValueError(f"no frames matching {pattern!r} under {frame_dir}")
    first = _points(paths[0])
    count = len(first)
    attributes = _appearance(Path(gaussian_ply), count) if gaussian_ply is not None else {}
    if not attributes and paths[0].suffix.lower() == ".ply":
        attributes = _appearance(paths[0], count)
    attributes.setdefault("colors", np.full((count, 3), [.22, .46, .49], dtype=np.float32))
    attributes.setdefault("opacity", np.ones(count, dtype=np.float32))
    ids = np.flatnonzero(attributes["opacity"].reshape(-1) >= min_opacity)
    if not len(ids):
        raise ValueError("no points remain after the opacity threshold")
    if max_points is not None and len(ids) > max_points:
        ids = np.sort(np.random.default_rng(seed).choice(ids, size=max_points, replace=False))
    points = np.empty((len(paths), len(ids), 3), dtype=np.float32)
    points[0] = first[ids]
    for index, path in enumerate(paths[1:], 1):
        frame = _points(path)
        if len(frame) != count:
            raise ValueError(f"registered point count changes in {path}")
        points[index] = frame[ids]
    selected = {name: np.asarray(values[ids], dtype=np.float32) for name, values in attributes.items()}
    selected.update(
        source_indices=ids, source_frames=np.asarray([path.name for path in paths]),
        source_point_count=np.asarray(count), seed=np.asarray(seed),
        min_opacity=np.asarray(min_opacity),
    )
    point_attributes = {name: value for name, value in selected.items() if name in attributes or name == "source_indices"}
    return DynamicPointSequence(points, fps=fps, point_attributes=point_attributes), selected


def save_registered_clouds(path, sequence, attributes):
    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError("registered point output must use .npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = dict(attributes)
    arrays.update(points=sequence.points.astype(np.float32), fps=np.asarray(sequence.fps), schema_version=np.asarray(1))
    if sequence.faces is not None:
        arrays["faces"] = sequence.faces
    np.savez_compressed(path, **arrays)
    return path


def main():
    parser = argparse.ArgumentParser(description="Import registered point-cloud or Gaussian-mean frames")
    parser.add_argument("frame_dir")
    parser.add_argument("output")
    parser.add_argument("--pattern", default="pointcloud_t*.ply")
    parser.add_argument("--gaussian-ply", type=Path)
    parser.add_argument("--max-points", type=int, default=12000)
    parser.add_argument("--min-opacity", type=float, default=.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=float, default=30.)
    args = parser.parse_args()
    sequence, attributes = load_registered_clouds(
        args.frame_dir, args.pattern, args.max_points, args.seed, args.fps,
        args.gaussian_ply, args.min_opacity,
    )
    print(save_registered_clouds(args.output, sequence, attributes))


if __name__ == "__main__":
    main()
