"""Joint detection and kinematic-tree construction on a skinned curve."""

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..types import Skeleton
from ._geometry import as_edges, as_points, normalize_rows
from .curve import CurveSkeleton


@dataclass
class JointDetection:
    positions: np.ndarray
    curve_indices: np.ndarray
    scores: np.ndarray
    all_scores: np.ndarray

    def __post_init__(self) -> None:
        self.positions = as_points(self.positions, "joint positions")
        self.curve_indices = np.asarray(self.curve_indices, dtype=np.int64)
        self.scores = np.asarray(self.scores, dtype=np.float64)
        self.all_scores = np.asarray(self.all_scores, dtype=np.float64)
        if self.curve_indices.shape != (len(self.positions),) or self.scores.shape != (len(self.positions),):
            raise ValueError("joint positions, indices, and scores must have the same length")


@dataclass(frozen=True)
class JointConfig:
    score_threshold: float = 0.3
    minimum_distance: float = 0.08
    peak_window_hops: int = 8
    max_joints: Optional[int] = None

    def __post_init__(self) -> None:
        if not 0 <= self.score_threshold <= 1:
            raise ValueError("score_threshold must be in [0, 1]")
        if self.minimum_distance < 0 or self.peak_window_hops < 1:
            raise ValueError("minimum_distance must be non-negative and peak_window_hops positive")
        if self.max_joints is not None and self.max_joints < 1:
            raise ValueError("max_joints must be positive")


@dataclass
class KinematicTree:
    skeleton: Skeleton
    edges: np.ndarray
    node_types: Tuple[str, ...]
    curve_indices: np.ndarray
    curve_paths: Tuple[Tuple[int, ...], ...]

    def __post_init__(self) -> None:
        self.edges = as_edges(self.edges, self.skeleton.num_joints, "kinematic edges")
        self.node_types = tuple(str(value) for value in self.node_types)
        self.curve_indices = np.asarray(self.curve_indices, dtype=np.int64)
        self.curve_paths = tuple(tuple(int(value) for value in path) for path in self.curve_paths)
        if len(self.node_types) != self.skeleton.num_joints:
            raise ValueError("node_types must match the kinematic joint count")
        if self.curve_indices.shape != (self.skeleton.num_joints,):
            raise ValueError("curve_indices must match the kinematic joint count")
        if len(self.curve_paths) != len(self.edges):
            raise ValueError("curve_paths must match the kinematic edge count")


def graph_adjacency(num_nodes: int, edges) -> Tuple[Tuple[int, ...], ...]:
    edges = as_edges(edges, num_nodes)
    adjacency = [set() for _ in range(num_nodes)]
    for first, second in edges:
        adjacency[int(first)].add(int(second))
        adjacency[int(second)].add(int(first))
    return tuple(tuple(sorted(neighbors)) for neighbors in adjacency)


def joint_scores(curve: CurveSkeleton, skinning_weights) -> np.ndarray:
    """Score balanced transitions between distinct dominant outer bones."""

    weights = normalize_rows(skinning_weights, "curve skinning weights")
    if len(weights) != len(curve.points):
        raise ValueError("curve point and skinning-weight counts differ")
    if weights.shape[1] < 2:
        return np.zeros(len(curve.points), dtype=np.float64)
    adjacency = graph_adjacency(len(curve.points), curve.edges)
    scores = np.zeros(len(curve.points), dtype=np.float64)
    for node, neighbors in enumerate(adjacency):
        if len(neighbors) < 2:
            continue
        top = np.argsort(weights[node])[-2:]
        pair_mass = weights[node, top].sum()
        if pair_mass <= 1e-12:
            continue
        balance = 1.0 - 2.0 * abs(weights[node, top[-1]] / pair_mass - 0.5)
        dominant = {int(np.argmax(weights[neighbor])) for neighbor in neighbors}
        if len(neighbors) == 2:
            if len(dominant) < 2:
                transition = 0.0
            else:
                overlap = len(set(int(value) for value in top) & dominant)
                transition = 1.0 if overlap == 2 else 0.5 if overlap == 1 else 0.0
        else:
            transition = min(1.0, len(dominant) / 3.0)
        scores[node] = np.clip(balance * transition, 0.0, 1.0)
    return scores


def _window(adjacency: Sequence[Sequence[int]], center: int, hops: int) -> set:
    visited = {center}
    queue = deque([(center, 0)])
    while queue:
        node, depth = queue.popleft()
        if depth == hops:
            continue
        for neighbor in adjacency[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, depth + 1))
    visited.remove(center)
    return visited


def detect_joints(
    curve: CurveSkeleton,
    skinning_weights,
    config: JointConfig = JointConfig(),
) -> JointDetection:
    scores = joint_scores(curve, skinning_weights)
    adjacency = graph_adjacency(len(curve.points), curve.edges)
    peaks = []
    for node, score in enumerate(scores):
        if score < config.score_threshold or not adjacency[node]:
            continue
        window = _window(adjacency, node, config.peak_window_hops)
        if all(scores[other] < score or (scores[other] == score and other > node) for other in window):
            peaks.append(node)
    peaks.sort(key=lambda node: (-scores[node], node))
    selected = []
    for node in peaks:
        if all(
            np.linalg.norm(curve.points[node] - curve.points[other]) >= config.minimum_distance
            for other in selected
        ):
            selected.append(node)
            if config.max_joints is not None and len(selected) == config.max_joints:
                break
    indices = np.asarray(selected, dtype=np.int64)
    return JointDetection(
        curve.points[indices] if len(indices) else np.empty((0, 3)),
        indices,
        scores[indices],
        scores,
    )


class _DisjointSet:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, first: int, second: int) -> bool:
        first, second = self.find(first), self.find(second)
        if first == second:
            return False
        self.parent[first] = second
        return True


def _minimum_spanning_forest(curve: CurveSkeleton) -> np.ndarray:
    lengths = np.linalg.norm(
        curve.points[curve.edges[:, 0]] - curve.points[curve.edges[:, 1]], axis=1
    ) if len(curve.edges) else np.empty(0)
    order = sorted(
        range(len(curve.edges)),
        key=lambda index: (lengths[index], int(curve.edges[index, 0]), int(curve.edges[index, 1])),
    )
    groups = _DisjointSet(len(curve.points))
    selected = [curve.edges[index] for index in order if groups.union(*map(int, curve.edges[index]))]
    return np.asarray(selected, dtype=np.int64).reshape(-1, 2)


def _compact_tree(
    curve: CurveSkeleton,
    tree_edges: np.ndarray,
    joint_indices: set,
) -> Tuple[List[int], List[List[int]], List[Tuple[int, ...]]]:
    adjacency = graph_adjacency(len(curve.points), tree_edges)
    keys = {node for node, neighbors in enumerate(adjacency) if len(neighbors) != 2}
    keys.update(joint_indices)
    key_list = sorted(keys)
    key_to_output = {node: index for index, node in enumerate(key_list)}
    output_edges: List[List[int]] = []
    paths: List[Tuple[int, ...]] = []
    visited_segments = set()
    for start in key_list:
        for neighbor in adjacency[start]:
            segment = tuple(sorted((start, neighbor)))
            if segment in visited_segments:
                continue
            visited_segments.add(segment)
            path = [start, neighbor]
            previous, current = start, neighbor
            while current not in keys:
                candidates = [value for value in adjacency[current] if value != previous]
                if not candidates:
                    break
                following = candidates[0]
                visited_segments.add(tuple(sorted((current, following))))
                path.append(following)
                previous, current = current, following
            if current in key_to_output:
                output_edges.append([key_to_output[start], key_to_output[current]])
                paths.append(tuple(path))
    return key_list, output_edges, paths


def _orient_forest(points: np.ndarray, edges: np.ndarray, node_types: Sequence[str]) -> np.ndarray:
    adjacency = graph_adjacency(len(points), edges)
    parents = np.full(len(points), -2, dtype=np.int64)
    remaining = set(range(len(points)))
    while remaining:
        seed = min(remaining)
        component = []
        queue = deque([seed])
        discovered = {seed}
        while queue:
            node = queue.popleft()
            component.append(node)
            for neighbor in adjacency[node]:
                if neighbor not in discovered:
                    discovered.add(neighbor)
                    queue.append(neighbor)
        centroid = points[component].mean(axis=0)
        preferred = [node for node in component if node_types[node] in ("joint", "branch")]
        candidates = preferred or component
        root = min(candidates, key=lambda node: (np.linalg.norm(points[node] - centroid), node))
        parents[root] = -1
        queue = deque([root])
        while queue:
            node = queue.popleft()
            for neighbor in adjacency[node]:
                if parents[neighbor] == -2:
                    parents[neighbor] = node
                    queue.append(neighbor)
        remaining.difference_update(component)
    return parents


def build_kinematic_tree(
    curve: CurveSkeleton,
    joints: JointDetection,
) -> KinematicTree:
    """Remove curve cycles, compress samples, then orient an acyclic hierarchy."""

    if np.any(joints.curve_indices < 0) or np.any(joints.curve_indices >= len(curve.points)):
        raise ValueError("joint detection references an out-of-range curve point")
    tree_edges = _minimum_spanning_forest(curve)
    curve_indices, compact_edges, paths = _compact_tree(
        curve, tree_edges, set(int(value) for value in joints.curve_indices)
    )
    if not curve_indices:
        raise ValueError("cannot build a kinematic tree from an empty curve")
    points = curve.points[curve_indices]
    compact_edges_array = np.asarray(compact_edges, dtype=np.int64).reshape(-1, 2)
    compact_adjacency = graph_adjacency(len(points), compact_edges_array)
    joint_set = set(int(value) for value in joints.curve_indices)
    node_types = []
    for output_index, curve_index in enumerate(curve_indices):
        if curve_index in joint_set:
            node_types.append("joint")
        elif len(compact_adjacency[output_index]) > 2:
            node_types.append("branch")
        elif len(compact_adjacency[output_index]) <= 1:
            node_types.append("endpoint")
        else:
            node_types.append("connector")
    parents = _orient_forest(points, compact_edges_array, node_types)
    curve_indices_array = np.asarray(curve_indices, dtype=np.int64)
    roots = np.flatnonzero(parents == -1)
    if len(roots) > 1:
        virtual_root = len(points)
        points = np.vstack([points, points[roots].mean(axis=0)])
        parents[roots] = virtual_root
        parents = np.append(parents, -1)
        root_edges = np.column_stack([
            np.full(len(roots), virtual_root, dtype=np.int64),
            roots,
        ])
        compact_edges_array = np.vstack([compact_edges_array, root_edges])
        paths.extend((int(curve_indices_array[root]),) for root in roots)
        curve_indices_array = np.append(curve_indices_array, -1)
        node_types.append("virtual_root")
    names = [f"{kind}_{index:03d}" for index, kind in enumerate(node_types)]
    return KinematicTree(
        Skeleton(points, parents, names),
        compact_edges_array,
        tuple(node_types),
        curve_indices_array,
        tuple(paths),
    )
