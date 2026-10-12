"""Skeleton partitioning and outer-bone assignment for PartMM."""

from typing import List, Optional, Sequence, Set, Tuple

import numpy as np

from .types import PartMMInput


HUMANOID_FIVE_PARTS = (
    ("pelvis", "spine1", "spine2", "spine3", "neck", "head"),
    ("left_collar", "left_shoulder", "left_elbow", "left_wrist", "left_hand"),
    (
        "right_collar",
        "right_shoulder",
        "right_elbow",
        "right_wrist",
        "right_hand",
    ),
    ("left_hip", "left_knee", "left_ankle", "left_foot"),
    ("right_hip", "right_knee", "right_ankle", "right_foot"),
)


def _children(parents: np.ndarray) -> List[List[int]]:
    children = [[] for _ in parents]
    for child, parent in enumerate(parents):
        if parent >= 0:
            children[int(parent)].append(child)
    return children


def partition_skeleton(
    parents: np.ndarray,
    num_parts: int,
    names: Optional[Sequence[str]] = None,
    mode: str = "balanced_tree",
) -> Tuple[np.ndarray, List[Set[int]]]:
    """Split a rooted tree into connected components.

    ``humanoid_five`` is the five-part protocol used by the human experiments:
    torso/head, left arm, right arm, left leg, and right leg. Other categories
    retain the topology-only balanced partition.
    """

    parents = np.asarray(parents, dtype=np.int64)
    roots = np.flatnonzero(parents == -1)
    if len(roots) != 1:
        raise ValueError("PartMM currently requires exactly one skeleton root")
    joint_count = len(parents)
    if mode == "humanoid_five":
        if num_parts != len(HUMANOID_FIVE_PARTS):
            raise ValueError("humanoid_five requires exactly five parts")
        if names is None or len(names) != joint_count:
            raise ValueError(
                "humanoid_five requires one canonical name per joint"
            )
        normalized = {
            str(name).strip().lower().replace("-", "_").replace(" ", "_"): index
            for index, name in enumerate(names)
        }
        required = {
            name for component in HUMANOID_FIVE_PARTS for name in component
        }
        missing = sorted(required - set(normalized))
        if missing:
            raise ValueError(
                "humanoid_five is missing joints: " + ", ".join(missing)
            )
        components = [
            {normalized[name] for name in component}
            for component in HUMANOID_FIVE_PARTS
        ]
        covered = set().union(*components)
        if covered != set(range(joint_count)):
            extras = sorted(set(range(joint_count)) - covered)
            raise ValueError(
                "humanoid_five does not classify joints: "
                + ", ".join(str(index) for index in extras)
            )
        assignment = np.empty(joint_count, dtype=np.int64)
        for part, component in enumerate(components):
            assignment[list(component)] = part
        return assignment, components
    if mode != "balanced_tree":
        raise ValueError(
            "partition mode must be balanced_tree or humanoid_five"
        )
    if num_parts >= joint_count:
        return np.arange(joint_count), [
            {joint} for joint in range(joint_count)
        ]
    children = _children(parents)
    components = [(int(roots[0]), set(range(joint_count)))]
    while len(components) < num_parts:
        component_index = max(
            range(len(components)),
            key=lambda index: len(components[index][1]),
        )
        root, joints = components[component_index]
        if len(joints) <= 1:
            break
        subtree_sizes = {}

        def size(joint: int) -> int:
            subtree_sizes[joint] = 1 + sum(
                size(child) for child in children[joint] if child in joints
            )
            return subtree_sizes[joint]

        size(root)
        candidates = sorted(joint for joint in joints if joint != root)
        cut = max(
            candidates,
            key=lambda joint: (
                min(
                    subtree_sizes.get(joint, 1),
                    len(joints) - subtree_sizes.get(joint, 1),
                ),
                -joint,
            ),
        )
        branch = set()

        def collect(joint: int) -> None:
            branch.add(joint)
            for child in children[joint]:
                if child in joints:
                    collect(child)

        collect(cut)
        components[component_index] = (root, joints - branch)
        components.append((cut, branch))
    assignment = np.zeros(joint_count, dtype=np.int64)
    sets = []
    for part, (_, joints) in enumerate(components):
        assignment[list(joints)] = part
        sets.append(joints)
    return assignment, sets


def _bone_influence_centroids(bundle: PartMMInput) -> np.ndarray:
    roots = bundle.skeleton.roots
    if len(roots) != 1:
        raise ValueError("PartMM currently requires exactly one skeleton root")
    centroids = np.empty((bundle.num_bones, 3), dtype=np.float64)
    for bone in range(bundle.num_bones):
        weight = bundle.weights[:, bone]
        total = weight.sum()
        centroids[bone] = (
            weight @ bundle.source_points / total
            if total > 1e-12
            else bundle.skeleton.rest_joints[int(roots[0])]
        )
    return centroids


def reference_bone_positions(
    bundle: PartMMInput,
    frame: int = 0,
) -> np.ndarray:
    """Return outer-bone influence centers in one observed world frame."""

    centroids = _bone_influence_centroids(bundle)
    if bundle.bone_transforms is None:
        return centroids
    if frame < 0 or frame >= len(bundle.bone_transforms):
        raise ValueError("reference bone frame is out of range")
    homogeneous = np.concatenate(
        [centroids, np.ones((len(centroids), 1), dtype=np.float64)],
        axis=1,
    )
    return np.einsum(
        "bij,bj->bi",
        bundle.bone_transforms[frame],
        homogeneous,
    )[:, :3]


def _reference_positions(
    bundle: PartMMInput,
    joint_positions: Optional[np.ndarray],
    bone_positions: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    joints = (
        bundle.skeleton.rest_joints
        if joint_positions is None
        else np.asarray(joint_positions, dtype=np.float64)
    )
    bones = (
        _bone_influence_centroids(bundle)
        if bone_positions is None
        else np.asarray(bone_positions, dtype=np.float64)
    )
    if joints.shape != (bundle.skeleton.num_joints, 3):
        raise ValueError("joint_positions must have shape [joints, 3]")
    if bones.shape != (bundle.num_bones, 3):
        raise ValueError("bone_positions must have shape [bones, 3]")
    if not np.isfinite(joints).all() or not np.isfinite(bones).all():
        raise ValueError("reference positions must be finite")
    return joints, bones


def overlapping_part_masks(
    parents: np.ndarray,
    components: List[Set[int]],
    rings: int,
) -> np.ndarray:
    """Expand connected parts by hierarchy rings for stable boundary matching."""
    adjacency = [set() for _ in parents]
    for child, parent in enumerate(parents):
        if parent >= 0:
            adjacency[child].add(int(parent))
            adjacency[int(parent)].add(child)
    masks = np.zeros((len(components), len(parents)), dtype=bool)
    for part, component in enumerate(components):
        expanded = set(component)
        frontier = set(component)
        for _ in range(rings):
            frontier = (
                set().union(*(adjacency[joint] for joint in frontier))
                - expanded
            )
            expanded.update(frontier)
        masks[part, list(expanded)] = True
    return masks


def map_bones_to_parts(
    bundle: PartMMInput,
    part_assignment: np.ndarray,
    *,
    joint_positions: Optional[np.ndarray] = None,
    bone_positions: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Assign each outer bone by its weighted influence centroid."""
    assignment = np.asarray(part_assignment, dtype=np.int64)
    if assignment.shape != (bundle.skeleton.num_joints,):
        raise ValueError(
            "part_assignment must have one entry per skeleton joint"
        )
    joints, centroids = _reference_positions(
        bundle,
        joint_positions,
        bone_positions,
    )
    nearest = np.linalg.norm(
        centroids[:, None] - joints[None],
        axis=-1,
    ).argmin(axis=1)
    return assignment[nearest]


def map_bones_to_edge_parents(
    bundle: PartMMInput,
    *,
    joint_positions: Optional[np.ndarray] = None,
    bone_positions: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Bind each outer bone to the parent joint of its closest skeleton edge."""

    parents = bundle.skeleton.parents
    children = np.flatnonzero(parents >= 0)
    root = int(bundle.skeleton.roots[0])
    if not len(children):
        return np.full(bundle.num_bones, root, dtype=np.int64)
    joints, centroids = _reference_positions(
        bundle,
        joint_positions,
        bone_positions,
    )
    starts = joints[parents[children]]
    segments = joints[children] - starts
    squared_length = np.sum(segments * segments, axis=1)
    relative = centroids[:, None] - starts[None]
    parameter = np.divide(
        np.einsum("bsi,si->bs", relative, segments),
        squared_length[None],
        out=np.zeros(
            (bundle.num_bones, len(children)),
            dtype=np.float64,
        ),
        where=squared_length[None] > 1e-12,
    )
    closest = (
        starts[None]
        + np.clip(parameter, 0.0, 1.0)[..., None] * segments[None]
    )
    edge = np.linalg.norm(
        centroids[:, None] - closest,
        axis=-1,
    ).argmin(axis=1)
    return parents[children[edge]].astype(np.int64)


def map_bones_to_part_edges(
    bundle: PartMMInput,
    components: Sequence[Set[int]],
    *,
    joint_positions: Optional[np.ndarray] = None,
    bone_positions: Optional[np.ndarray] = None,
    overlap_ratio: float = 1.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Map bones to overlapping parts and a closest edge inside each part."""

    if overlap_ratio < 1.0:
        raise ValueError("bone overlap ratio must be at least one")
    joints, bones = _reference_positions(
        bundle,
        joint_positions,
        bone_positions,
    )
    parents = bundle.skeleton.parents
    root = int(bundle.skeleton.roots[0])
    part_count = len(components)
    distances = np.full(
        (part_count, bundle.num_bones),
        np.inf,
        dtype=np.float64,
    )
    edge_parents = np.full(
        (part_count, bundle.num_bones),
        root,
        dtype=np.int64,
    )
    for part, component in enumerate(components):
        children = np.asarray(
            [
                child
                for child in range(len(parents))
                if parents[child] >= 0 and child in component
            ],
            dtype=np.int64,
        )
        if not len(children):
            candidates = np.asarray(sorted(component), dtype=np.int64)
            point_distance = np.linalg.norm(
                bones[:, None] - joints[candidates][None],
                axis=-1,
            )
            nearest = point_distance.argmin(axis=1)
            distances[part] = point_distance[
                np.arange(bundle.num_bones),
                nearest,
            ]
            edge_parents[part] = candidates[nearest]
            continue
        starts = joints[parents[children]]
        segments = joints[children] - starts
        squared_length = np.sum(segments * segments, axis=1)
        relative = bones[:, None] - starts[None]
        parameter = np.divide(
            np.einsum("bsi,si->bs", relative, segments),
            squared_length[None],
            out=np.zeros(
                (bundle.num_bones, len(children)),
                dtype=np.float64,
            ),
            where=squared_length[None] > 1e-12,
        )
        closest = (
            starts[None]
            + np.clip(parameter, 0.0, 1.0)[..., None] * segments[None]
        )
        edge_distance = np.linalg.norm(
            bones[:, None] - closest,
            axis=-1,
        )
        nearest = edge_distance.argmin(axis=1)
        distances[part] = edge_distance[
            np.arange(bundle.num_bones),
            nearest,
        ]
        edge_parents[part] = parents[children[nearest]]

    nearest_distance = distances.min(axis=0)
    masks = distances <= (
        nearest_distance[None] * overlap_ratio + 1e-12
    )
    for part in range(part_count):
        if not masks[part].any():
            masks[part, int(np.argmin(distances[part]))] = True
    return masks, edge_parents


__all__ = [
    "HUMANOID_FIVE_PARTS",
    "map_bones_to_edge_parents",
    "map_bones_to_part_edges",
    "map_bones_to_parts",
    "overlapping_part_masks",
    "partition_skeleton",
    "reference_bone_positions",
]
