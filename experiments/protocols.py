"""Dataset selection protocols stated in the GaussiAnimate paper."""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from skelebone.io import load_point_sequence, save_point_sequence

from .mesh.export_dt4d import load_anime


def split_frame_indices(
    frame_count: int,
    train_ratio: float = 0.8,
    mode: str = "contiguous",
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return deterministic train/test indices for the paper's 80/20 protocol."""

    if frame_count < 2:
        raise ValueError("frame split requires at least two frames")
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be in (0,1)")
    train_count = min(frame_count - 1, max(1, int(round(frame_count * train_ratio))))
    indices = np.arange(frame_count, dtype=np.int64)
    if mode == "contiguous":
        return indices[:train_count], indices[train_count:]
    if mode == "interleaved":
        test_count = frame_count - train_count
        test = np.unique(
            np.round(np.linspace(0, frame_count - 1, test_count)).astype(np.int64)
        )
        train = np.setdiff1d(indices, test, assume_unique=True)
        return train, test
    if mode == "random":
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(indices)
        train = np.sort(shuffled[:train_count])
        test = np.sort(shuffled[train_count:])
        return train, test
    raise ValueError("split mode must be contiguous, interleaved, or random")


def motion_scores(points: np.ndarray, metric: str = "mean_displacement") -> np.ndarray:
    """Assign a motion score to every frame of a registered trajectory."""

    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 3 or points.shape[-1] != 3 or len(points) < 2:
        raise ValueError("motion scoring requires points with shape [T,N,3], T>=2")
    if metric == "mean_displacement":
        transition = np.linalg.norm(np.diff(points, axis=0), axis=-1).mean(axis=1)
    elif metric == "bbox":
        bounds = np.concatenate([points.min(axis=1), points.max(axis=1)], axis=1)
        transition = np.linalg.norm(np.diff(bounds, axis=0), axis=1)
    else:
        raise ValueError("motion metric must be mean_displacement or bbox")
    scores = np.empty(len(points), dtype=np.float64)
    scores[0] = transition[0]
    scores[-1] = transition[-1]
    if len(points) > 2:
        scores[1:-1] = 0.5 * (transition[:-1] + transition[1:])
    return scores


def select_high_motion_frames(
    points: np.ndarray,
    target_frames: int,
    metric: str = "mean_displacement",
) -> np.ndarray:
    """Select high-motion frames, retaining chronological order in the output."""

    if target_frames < 2:
        raise ValueError("target_frames must be at least two")
    if target_frames >= len(points):
        return np.arange(len(points), dtype=np.int64)
    scores = motion_scores(points, metric)
    ranked = np.lexsort((np.arange(len(scores)), -scores))
    selected = np.sort(ranked[:target_frames])
    return selected.astype(np.int64)


def select_high_motion_window(
    points: np.ndarray,
    window_frames: int,
    metric: str = "bbox",
) -> np.ndarray:
    """Select the contiguous ActorsHQ window with maximum accumulated motion."""

    points = np.asarray(points)
    if window_frames < 2 or window_frames > len(points):
        raise ValueError("window_frames must be in [2,T]")
    scores = motion_scores(points, metric)
    cumulative = np.concatenate([[0.0], np.cumsum(scores)])
    window_scores = cumulative[window_frames:] - cumulative[:-window_frames]
    start = int(np.argmax(window_scores))
    return np.arange(start, start + window_frames, dtype=np.int64)


def export_dt4d_high_motion(
    anime_path,
    output_path,
    target_frames: int = 200,
    fps: float = 30.0,
) -> Path:
    """Convert one DT4D motion using the paper's high-motion frame protocol."""

    points, faces = load_anime(anime_path)
    indices = select_high_motion_frames(points, target_frames)
    output = save_point_sequence(output_path, points[indices], faces=faces, fps=fps)
    # Preserve provenance not covered by the public three-field hand-off.
    data = dict(np.load(output, allow_pickle=False))
    data.update(
        source=np.asarray(str(Path(anime_path))),
        source_frame_indices=indices,
        selection=np.asarray("high_motion"),
    )
    np.savez_compressed(output, **data)
    return output


def slice_point_sequence(
    input_path,
    output_path,
    *,
    fraction: Optional[float] = None,
    frame_count: Optional[int] = None,
) -> Path:
    """Write the first requested frames of a registered point trajectory."""

    if (fraction is None) == (frame_count is None):
        raise ValueError("set exactly one of fraction or frame_count")
    sequence = load_point_sequence(input_path)
    if fraction is not None:
        if not 0.0 < fraction <= 1.0:
            raise ValueError("fraction must be in (0,1]")
        frame_count = max(1, int(round(sequence.num_frames * fraction)))
    frame_count = int(frame_count)
    if frame_count < 1 or frame_count > sequence.num_frames:
        raise ValueError("frame_count must be in [1,T]")
    return save_point_sequence(
        output_path,
        sequence.points[:frame_count],
        faces=sequence.faces,
        fps=sequence.fps,
    )
