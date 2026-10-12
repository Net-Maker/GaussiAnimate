"""Fit SSDR on a deterministic subset and project weights to all Gaussians."""

import argparse
import json
import os
import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from skelebone import DynamicPointSequence
from skelebone.extraction.ssdr import SSDRConfig, decompose_ssdr


def select_points(num_points: int, target: int, seed: int) -> np.ndarray:
    if target <= 0 or target >= num_points:
        return np.arange(num_points, dtype=np.int64)
    generator = np.random.RandomState(seed)
    return np.sort(generator.choice(num_points, target, replace=False))


def project_weights(
    subset_rest: np.ndarray,
    full_rest: np.ndarray,
    subset_weights: np.ndarray,
    neighbors: int = 4,
) -> np.ndarray:
    count = min(neighbors, len(subset_rest))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from scipy.spatial import cKDTree

        distances, indices = cKDTree(subset_rest).query(full_rest, k=count)
    except (ImportError, AttributeError, ValueError):
        distances, indices = _numpy_nearest_neighbors(
            subset_rest, full_rest, count
        )
    if count == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    inverse = 1.0 / np.maximum(distances, 1e-8)
    blend = inverse / inverse.sum(axis=1, keepdims=True)
    weights = np.zeros(
        (len(full_rest), subset_weights.shape[1]),
        dtype=np.float32,
    )
    for column in range(count):
        weights += (
            blend[:, column, None]
            * subset_weights[indices[:, column]]
        )
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-8)
    return weights


def _numpy_nearest_neighbors(
    reference: np.ndarray,
    queries: np.ndarray,
    neighbors: int,
    chunk_size: int = 1024,
) -> Tuple[np.ndarray, np.ndarray]:
    """Memory-bounded fallback used when a compatible SciPy is unavailable."""
    all_distances = []
    all_indices = []
    for start in range(0, len(queries), chunk_size):
        chunk = queries[start : start + chunk_size]
        squared = np.sum(
            (chunk[:, None, :] - reference[None, :, :]) ** 2,
            axis=-1,
        )
        candidate = np.argpartition(
            squared, kth=neighbors - 1, axis=1
        )[:, :neighbors]
        candidate_distance = np.take_along_axis(squared, candidate, axis=1)
        order = np.argsort(candidate_distance, axis=1)
        candidate = np.take_along_axis(candidate, order, axis=1)
        candidate_distance = np.take_along_axis(
            candidate_distance, order, axis=1
        )
        all_indices.append(candidate)
        all_distances.append(np.sqrt(candidate_distance))
    return np.concatenate(all_distances), np.concatenate(all_indices)


def reconstruct_full(
    rest_pose: np.ndarray,
    weights: np.ndarray,
    transforms: np.ndarray,
    frame_means: np.ndarray,
) -> np.ndarray:
    homogeneous = np.concatenate(
        [rest_pose, np.ones((len(rest_pose), 1), dtype=np.float32)],
        axis=1,
    )
    reconstruction = np.empty(
        (len(transforms), len(rest_pose), 3),
        dtype=np.float32,
    )
    for frame, frame_transforms in enumerate(transforms):
        positions = np.zeros_like(rest_pose, dtype=np.float32)
        for bone, transform in enumerate(frame_transforms):
            transformed = (homogeneous @ transform.T)[:, :3]
            positions += weights[:, bone, None] * transformed
        reconstruction[frame] = positions + frame_means[frame]
    return reconstruction


def infer_parents(bone_positions: np.ndarray) -> np.ndarray:
    mean_distance = np.linalg.norm(
        bone_positions[:, :, None] - bone_positions[:, None, :],
        axis=-1,
    ).mean(axis=0)
    count = len(mean_distance)
    parents = np.full(count, -2, dtype=np.int64)
    parents[0] = -1
    in_tree = np.zeros(count, dtype=bool)
    in_tree[0] = True
    best_distance = mean_distance[0].copy()
    best_parent = np.zeros(count, dtype=np.int64)
    for _ in range(1, count):
        candidates = np.where(in_tree, np.inf, best_distance)
        node = int(np.argmin(candidates))
        if not np.isfinite(candidates[node]):
            raise ValueError("bone positions do not define a connected finite graph")
        parents[node] = best_parent[node]
        in_tree[node] = True
        improve = (~in_tree) & (mean_distance[node] < best_distance)
        best_distance[improve] = mean_distance[node, improve]
        best_parent[improve] = node
    return parents


def fit_cache(
    params_path: Path,
    output_dir: Path,
    case_id: str,
    num_bones: int = 50,
    ssdr_points: int = 10000,
    seed: int = 42,
    train_ratio: float = 1.0,
    training_frames: Optional[int] = None,
    output_path: Optional[Path] = None,
) -> Tuple[Path, dict]:
    with np.load(params_path, allow_pickle=False) as params:
        means = np.asarray(params["means3D"], dtype=np.float32)
    frames, num_points = means.shape[:2]
    if training_frames is None:
        if not 0.0 < train_ratio <= 1.0:
            raise ValueError("train_ratio must be in (0, 1]")
        training_frames = int(frames * train_ratio)
    if training_frames < 1 or training_frames > frames:
        raise ValueError("training_frames must be between 1 and the sequence length")
    fitted_means = means[:training_frames]
    frame_means = fitted_means.mean(axis=1)
    point_indices = select_points(num_points, ssdr_points, seed)
    subset = fitted_means[:, point_indices] - frame_means[:, None]
    # The paper defines the canonical Gaussian shape empirically as frame 0.
    rest_frame = 0
    result = decompose_ssdr(
        DynamicPointSequence(subset),
        SSDRConfig(
            num_bones=num_bones,
            rest_frame=rest_frame,
            tolerance=1e-3,
            patience=3,
            max_iterations=100,
            max_influences=4,
            weight_smoothness=0.0,
            bind_update=2,
            protocol="motion_pointcloud",
            initialization="motion",
            random_seed=seed,
            require_num_bones=True,
        ),
    )
    full_rest = fitted_means[rest_frame] - frame_means[rest_frame]
    full_weights = project_weights(
        subset[rest_frame],
        full_rest,
        result.weights.astype(np.float32),
    )
    reconstruction = reconstruct_full(
        full_rest,
        full_weights,
        result.transformations.astype(np.float32),
        frame_means,
    )
    error = np.linalg.norm(reconstruction - fitted_means, axis=2)
    rmse_mm = float(error.mean() * 1000.0)
    bone_positions = result.bone_positions + frame_means[:, None]
    world_transformations = result.transformations.astype(np.float32).copy()
    world_transformations[:, :, :3, 3] += frame_means[:, None]
    parents = infer_parents(bone_positions)
    prefix = case_id[4:] if case_id.startswith("dna_") else case_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output = (
        Path(output_path)
        if output_path is not None
        else output_dir
        / f"{prefix}_bones{num_bones}_full{num_points}_ssdr{len(point_indices)}.npz"
    )
    if output.suffix != ".npz":
        raise ValueError("output_path must end in .npz")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".npz.partial")
    with temporary.open("wb") as handle:
        np.savez(
            handle,
            schema_version=np.int64(1),
            weights_layout=np.asarray("bones_points"),
            weights=full_weights.T,
            transformations=result.transformations.astype(np.float32),
            world_transformations=world_transformations,
            bone_positions=bone_positions.astype(np.float32),
            rest_pose=full_rest.astype(np.float32),
            reconstruction_original=reconstruction,
            frame_means=frame_means.astype(np.float32),
            full_recon_rmse=np.float64(rmse_mm),
            num_bones=np.int64(result.num_bones),
            requested_bones=np.int64(num_bones),
            initialized_bones=np.int64(result.initialized_bones),
            effective_bones=np.int64(result.effective_bones),
            hard_bone_population=result.hard_bone_population,
            soft_bone_mass=result.soft_bone_mass,
            ssdr_protocol=np.asarray(result.protocol),
            backend_name=np.asarray(result.backend_name),
            backend_module=np.asarray(result.backend_module),
            backend_path=np.asarray(result.backend_path or ""),
            backend_sha256=np.asarray(result.backend_sha256 or ""),
            rest_frame=np.int64(rest_frame),
            parents=parents,
            point_indices=point_indices,
            n_full=np.int64(num_points),
            n_ssdr=np.int64(len(point_indices)),
            total_frames=np.int64(frames),
            training_frames=np.int64(training_frames),
            train_ratio=np.float64(training_frames / frames),
        )
    os.replace(temporary, output)
    metadata = {
        "schema_version": 1,
        "case": case_id,
        "frames": frames,
        "training_frames": training_frames,
        "train_ratio": training_frames / frames,
        "split": "contiguous_prefix",
        "points": num_points,
        "ssdr_points": len(point_indices),
        "requested_bones": num_bones,
        "initialized_bones": result.initialized_bones,
        "actual_bones": result.num_bones,
        "effective_bones": result.effective_bones,
        "ssdr_protocol": result.protocol,
        "ssdr_backend": result.diagnostics["backend"],
        "rest_frame": rest_frame,
        "full_reconstruction_rmse_mm": rmse_mm,
        "params": str(params_path.resolve()),
        "cache": str(output.resolve()),
    }
    metadata_path = output.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit a release SSDR cache")
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--num-bones", type=int, default=50)
    parser.add_argument("--ssdr-points", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=1.0)
    parser.add_argument("--training-frames", type=int)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional deterministic cache path ending in .npz",
    )
    args = parser.parse_args()
    output, metadata = fit_cache(
        args.params,
        args.output_dir,
        args.case,
        args.num_bones,
        args.ssdr_points,
        args.seed,
        args.train_ratio,
        args.training_frames,
        args.output,
    )
    print(output)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
