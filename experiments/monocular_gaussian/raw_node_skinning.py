"""Export deformation-graph node weights without SSDR."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Tuple

import numpy as np


def compute_raw_node_weights(
    points: np.ndarray,
    features: np.ndarray,
    nodes: np.ndarray,
    node_radius: np.ndarray,
    node_weight: np.ndarray,
    *,
    k: int = 3,
    chunk_size: int = 4096,
) -> Tuple[np.ndarray, np.ndarray]:
    """Reproduce RigGS Gaussian-kernel KNN weights in xyz+hyperfeature space."""

    points = np.asarray(points, dtype=np.float32)
    features = np.asarray(features, dtype=np.float32)
    nodes = np.asarray(nodes, dtype=np.float32)
    node_radius = np.asarray(node_radius, dtype=np.float32).reshape(-1)
    node_weight = np.asarray(node_weight, dtype=np.float32).reshape(-1)
    queries = np.concatenate([points, features], axis=1)
    if queries.ndim != 2 or nodes.ndim != 2 or queries.shape[1] != nodes.shape[1]:
        raise ValueError("query and node feature dimensions must match")
    if len(node_radius) != len(nodes) or len(node_weight) != len(nodes):
        raise ValueError("node radius and weight arrays must match node count")
    if not 1 <= k <= len(nodes):
        raise ValueError("k must be in [1, node_count]")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")

    indices = np.empty((len(queries), k), dtype=np.int64)
    weights = np.empty((len(queries), k), dtype=np.float32)
    for start in range(0, len(queries), chunk_size):
        stop = min(start + chunk_size, len(queries))
        delta = queries[start:stop, None, :] - nodes[None, :, :]
        squared_distance = np.einsum("qnd,qnd->qn", delta, delta)
        nearest = np.argpartition(squared_distance, k - 1, axis=1)[:, :k]
        nearest_distance = np.take_along_axis(squared_distance, nearest, axis=1)
        order = np.argsort(nearest_distance, axis=1)
        nearest = np.take_along_axis(nearest, order, axis=1)
        nearest_distance = np.take_along_axis(nearest_distance, order, axis=1)
        raw = np.exp(
            -nearest_distance / (2.0 * np.square(node_radius[nearest]))
        ) * node_weight[nearest]
        raw += 1e-7
        indices[start:stop] = nearest
        weights[start:stop] = raw / raw.sum(axis=1, keepdims=True)
    return indices, weights


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_checkpoint(path: Path):
    try:
        import torch
    except ImportError as exc:
        raise ImportError("checkpoint loading requires PyTorch") from exc

    state = torch.load(path, map_location="cpu")
    required = {"nodes", "_node_radius", "_node_weight"}
    missing = required - set(state)
    if missing:
        raise KeyError(f"deformation checkpoint is missing {sorted(missing)}")
    nodes = state["nodes"].detach().numpy()
    radius = np.exp(state["_node_radius"].detach().numpy())
    weight = 1.0 / (1.0 + np.exp(-state["_node_weight"].detach().numpy()))
    return nodes, radius, weight


def _load_ply(path: Path, hyper_dim: int):
    try:
        from plyfile import PlyData
    except ImportError as exc:
        raise ImportError("PLY export requires the plyfile package") from exc

    vertex = PlyData.read(str(path))["vertex"].data
    points = np.column_stack([vertex[name] for name in ("x", "y", "z")])
    feature_names = [f"fea_{index}" for index in range(hyper_dim)]
    missing = [name for name in feature_names if name not in vertex.dtype.names]
    if missing:
        raise KeyError(f"point-cloud PLY is missing features {missing}")
    features = np.column_stack([vertex[name] for name in feature_names])
    return vertex, points, features


def _node_colors(nodes: np.ndarray) -> np.ndarray:
    xyz = nodes[:, :3]
    extent = xyz.max(axis=0) - xyz.min(axis=0)
    extent[extent == 0] = 1.0
    return np.clip((xyz - xyz.min(axis=0)) / extent, 0.0, 0.99)


def export_raw_node_skinning(
    point_cloud: Path,
    deform_checkpoint: Path,
    output_ply: Path,
    output_npz: Path,
    *,
    k: int = 3,
) -> Path:
    try:
        from plyfile import PlyData, PlyElement
    except ImportError as exc:
        raise ImportError("PLY export requires the plyfile package") from exc

    nodes, radius, node_weight = _load_checkpoint(Path(deform_checkpoint))
    hyper_dim = nodes.shape[1] - 3
    vertex, points, features = _load_ply(Path(point_cloud), hyper_dim)
    indices, weights = compute_raw_node_weights(
        points, features, nodes, radius, node_weight, k=k
    )
    colors = np.sum(
        weights[..., None] * _node_colors(nodes)[indices],
        axis=1,
    )
    output_vertex = np.empty(
        len(vertex),
        dtype=vertex.dtype.descr
        + [("red", "u1"), ("green", "u1"), ("blue", "u1")],
    )
    for name in vertex.dtype.names:
        output_vertex[name] = vertex[name]
    rgb = np.rint(colors * 255.0).astype(np.uint8)
    output_vertex["red"], output_vertex["green"], output_vertex["blue"] = rgb.T

    output_ply = Path(output_ply)
    output_npz = Path(output_npz)
    output_ply.parent.mkdir(parents=True, exist_ok=True)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(output_vertex, "vertex")], text=False).write(
        str(output_ply)
    )
    np.savez_compressed(
        output_npz,
        node_indices=indices,
        node_weights=weights,
        node_xyz=nodes[:, :3],
        point_colors=colors,
    )
    manifest = {
        "method": "raw_deformation_graph_node_weights_without_ssdr",
        "point_count": len(points),
        "node_count": len(nodes),
        "hyper_dim": hyper_dim,
        "k": k,
        "point_cloud_sha256": _sha256(Path(point_cloud)),
        "deform_checkpoint_sha256": _sha256(Path(deform_checkpoint)),
        "output_ply_sha256": _sha256(output_ply),
        "output_npz_sha256": _sha256(output_npz),
    }
    manifest_path = output_npz.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--point-cloud", required=True, type=Path)
    parser.add_argument("--deform-checkpoint", required=True, type=Path)
    parser.add_argument("--output-ply", required=True, type=Path)
    parser.add_argument("--output-npz", required=True, type=Path)
    parser.add_argument("-k", type=int, default=3)
    args = parser.parse_args()
    print(
        export_raw_node_skinning(
            args.point_cloud,
            args.deform_checkpoint,
            args.output_ply,
            args.output_npz,
            k=args.k,
        )
    )


if __name__ == "__main__":
    main()
