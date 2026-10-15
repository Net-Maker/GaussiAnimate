"""Smooth skinning decomposition for registered dynamic point sequences."""

from dataclasses import dataclass
import hashlib
from pathlib import Path
import sys
import tempfile
from typing import Any, Optional

import numpy as np

from ..types import DynamicPointSequence
from ._geometry import as_points, normalize_rows


@dataclass(frozen=True)
class SSDRConfig:
    """Optimization controls passed to Dem Bones.

    Four non-zero weights provide locally blended motion without making the
    decomposition unnecessarily dense. Mesh weight smoothing is opt-in because
    point sequences have no reliable surface adjacency.
    """

    num_bones: int = 50
    rest_frame: int = 0
    tolerance: float = 1e-3
    patience: int = 3
    max_iterations: int = 100
    max_influences: int = 4
    weight_smoothness: float = 0.0
    bind_update: int = 2
    protocol: str = "motion_pointcloud"
    initialization: str = "auto"
    initialization_frames: int = 40
    initialization_components: int = 16
    initialization_spatial_weight: float = 0.35
    random_seed: int = 42
    require_num_bones: bool = False

    def __post_init__(self) -> None:
        if self.num_bones < 1:
            raise ValueError("num_bones must be positive")
        if self.rest_frame < 0:
            raise ValueError("rest_frame must be non-negative")
        if self.tolerance <= 0 or self.patience < 1 or self.max_iterations < 1:
            raise ValueError("SSDR convergence controls must be positive")
        if self.max_influences < 1:
            raise ValueError("max_influences must be positive")
        if self.weight_smoothness < 0:
            raise ValueError("weight_smoothness must be non-negative")
        if self.protocol not in {"motion_pointcloud", "paper_legacy"}:
            raise ValueError("protocol must be motion_pointcloud or paper_legacy")
        if self.initialization not in {"auto", "backend", "motion"}:
            raise ValueError("initialization must be auto, backend, or motion")
        if self.initialization_frames < 2 or self.initialization_components < 1:
            raise ValueError("motion initialization dimensions must be positive")
        if self.initialization_spatial_weight < 0:
            raise ValueError("initialization_spatial_weight must be non-negative")


@dataclass
class SSDRResult:
    weights: np.ndarray
    transformations: np.ndarray
    bone_positions: np.ndarray
    reconstruction: np.ndarray
    rest_pose: np.ndarray
    rmse: float
    protocol: str = "motion_pointcloud"
    requested_bones: Optional[int] = None
    initialized_bones: Optional[int] = None
    hard_bone_population: Optional[np.ndarray] = None
    soft_bone_mass: Optional[np.ndarray] = None
    backend_name: str = "unknown"
    backend_module: str = "unknown"
    backend_path: Optional[str] = None
    backend_sha256: Optional[str] = None

    def __post_init__(self) -> None:
        self.rest_pose = as_points(self.rest_pose, "rest_pose")
        self.weights = normalize_rows(self.weights)
        frames = np.asarray(self.transformations, dtype=np.float64)
        expected = (self.weights.shape[1], 4, 4)
        if frames.ndim != 4 or frames.shape[1:] != expected:
            raise ValueError(f"transformations must have shape [frames, {expected[0]}, 4, 4]")
        if not np.all(np.isfinite(frames)):
            raise ValueError("transformations contains NaN or infinite values")
        self.transformations = frames
        self.reconstruction = np.asarray(self.reconstruction, dtype=np.float64)
        if self.reconstruction.shape != (len(frames), len(self.rest_pose), 3):
            raise ValueError("reconstruction shape does not match transforms and rest pose")
        self.bone_positions = np.asarray(self.bone_positions, dtype=np.float64)
        if self.bone_positions.shape != (len(frames), self.weights.shape[1], 3):
            raise ValueError("bone_positions shape does not match transforms and weights")
        self.rmse = float(self.rmse)
        if self.protocol not in {"motion_pointcloud", "paper_legacy"}:
            raise ValueError("unsupported SSDR protocol")
        if self.requested_bones is None:
            self.requested_bones = self.num_bones
        if self.initialized_bones is None:
            self.initialized_bones = self.num_bones
        self.requested_bones = int(self.requested_bones)
        self.initialized_bones = int(self.initialized_bones)
        hard = np.bincount(
            np.argmax(self.weights, axis=1), minlength=self.num_bones
        ).astype(np.int64, copy=False)
        soft = self.weights.sum(axis=0, dtype=np.float64)
        if self.hard_bone_population is not None:
            provided = np.asarray(self.hard_bone_population, dtype=np.int64)
            if provided.shape != hard.shape:
                raise ValueError("hard_bone_population shape does not match weights")
            hard = provided
        if self.soft_bone_mass is not None:
            provided = np.asarray(self.soft_bone_mass, dtype=np.float64)
            if provided.shape != soft.shape:
                raise ValueError("soft_bone_mass shape does not match weights")
            soft = provided
        self.hard_bone_population = hard
        self.soft_bone_mass = soft

    @property
    def num_bones(self) -> int:
        return int(self.weights.shape[1])

    @property
    def effective_bones(self) -> int:
        return int(
            np.count_nonzero(
                (self.hard_bone_population > 0) & (self.soft_bone_mass >= 1.0)
            )
        )

    @property
    def diagnostics(self) -> dict:
        return {
            "protocol": self.protocol,
            "requested_bones": self.requested_bones,
            "initialized_bones": self.initialized_bones,
            "final_bones": self.num_bones,
            "effective_bones": self.effective_bones,
            "hard_bone_population": self.hard_bone_population.copy(),
            "soft_bone_mass": self.soft_bone_mass.copy(),
            "backend": {
                "class": self.backend_name,
                "module": self.backend_module,
                "path": self.backend_path,
                "sha256": self.backend_sha256,
            },
        }


def _backend_metadata(backend: Any) -> dict:
    backend_type = type(backend)
    module_name = backend_type.__module__
    module = sys.modules.get(module_name)
    raw_path = getattr(module, "__file__", None)
    path = Path(raw_path).resolve() if raw_path else None
    digest = None
    if path is not None and path.is_file():
        hasher = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(block)
        digest = hasher.hexdigest()
    return {
        "backend_name": backend_type.__name__,
        "backend_module": module_name,
        "backend_path": None if path is None else str(path),
        "backend_sha256": digest,
    }


def _dense_weights(weights: Any, num_points: int) -> np.ndarray:
    if hasattr(weights, "toarray"):
        weights = weights.toarray()
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 2:
        raise ValueError("SSDR backend returned non-matrix weights")
    if weights.shape[0] != num_points and weights.shape[1] == num_points:
        weights = weights.T
    if weights.shape[0] != num_points:
        raise ValueError("SSDR backend weight count does not match the point sequence")
    return normalize_rows(weights)


def _reshape_transforms(raw, num_frames: int, num_bones: int) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float64)
    if raw.shape == (num_frames, num_bones, 4, 4):
        return raw
    if raw.shape == (num_frames, 4, num_bones, 4):
        return raw.transpose(0, 2, 1, 3)
    if raw.size == num_frames * num_bones * 16:
        return raw.reshape(num_frames, 4, num_bones, 4).transpose(0, 2, 1, 3)
    raise ValueError("SSDR backend returned transformations with an unsupported shape")


def linear_blend_skinning(rest_pose, weights, transformations) -> np.ndarray:
    rest_pose = as_points(rest_pose, "rest_pose")
    weights = normalize_rows(weights)
    transforms = np.asarray(transformations, dtype=np.float64)
    if transforms.ndim != 4 or transforms.shape[1:] != (weights.shape[1], 4, 4):
        raise ValueError("transformations must have shape [frames, bones, 4, 4]")
    if len(rest_pose) != len(weights):
        raise ValueError("rest_pose and weights point counts differ")
    homogeneous = np.concatenate([rest_pose, np.ones((len(rest_pose), 1))], axis=1)
    output = np.empty((len(transforms), len(rest_pose), 3), dtype=np.float64)
    # Blend affine maps before applying them; never allocate [frames,bones,points].
    for frame, matrices in enumerate(transforms):
        blended = weights @ matrices[:, :3, :].reshape(weights.shape[1], 12)
        output[frame] = np.einsum("nij,nj->ni", blended.reshape(-1, 3, 4), homogeneous)
    return output


def bone_position_sequence(transformations, weights, rest_pose) -> np.ndarray:
    """Track one maximally influenced rest point as each outer bone anchor."""

    rest_pose = as_points(rest_pose, "rest_pose")
    weights = normalize_rows(weights)
    transforms = np.asarray(transformations, dtype=np.float64)
    if transforms.ndim != 4 or transforms.shape[1:] != (weights.shape[1], 4, 4):
        raise ValueError("transformations must have shape [frames, bones, 4, 4]")
    anchors = rest_pose[np.argmax(weights, axis=0)]
    homogeneous = np.concatenate([anchors, np.ones((len(anchors), 1))], axis=1)
    return np.einsum("tbij,bj->tbi", transforms, homogeneous)[..., :3]


def _assign_to_centers(features: np.ndarray, centers: np.ndarray) -> tuple:
    labels = np.empty(len(features), dtype=np.int32)
    distances = np.empty(len(features), dtype=np.float64)
    center_norm = np.einsum("kd,kd->k", centers, centers)
    for start in range(0, len(features), 4096):
        block = features[start : start + 4096]
        squared = (
            np.einsum("nd,nd->n", block, block)[:, None]
            + center_norm[None]
            - 2.0 * block @ centers.T
        )
        block_labels = np.argmin(squared, axis=1)
        labels[start : start + len(block)] = block_labels
        distances[start : start + len(block)] = squared[
            np.arange(len(block)), block_labels
        ]
    return labels, np.maximum(distances, 0.0)


def motion_cluster_labels(
    points: np.ndarray,
    num_clusters: int,
    *,
    rest_frame: int = 0,
    sampled_frames: int = 40,
    components: int = 16,
    spatial_weight: float = 0.35,
    seed: int = 42,
    iterations: int = 30,
) -> np.ndarray:
    """Initialize rigid parts from trajectories, independent of point density."""

    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 3 or points.shape[2] != 3:
        raise ValueError("points must have shape [frames, points, 3]")
    num_frames, num_points, _ = points.shape
    if not 0 <= rest_frame < num_frames:
        raise ValueError("rest_frame is outside the point sequence")
    if not 1 <= num_clusters <= num_points:
        raise ValueError("num_clusters must be between one and the point count")

    frame_indices = np.unique(
        np.linspace(0, num_frames - 1, min(sampled_frames, num_frames)).round().astype(int)
    )
    rest = points[rest_frame]
    displacement = points[frame_indices] - rest[None]
    motion = displacement.transpose(1, 0, 2).reshape(num_points, -1)
    motion -= motion.mean(axis=0, keepdims=True)
    covariance = motion.T @ motion / max(num_points - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    rank = min(components, motion.shape[1], num_points)
    basis = eigenvectors[:, np.argsort(eigenvalues)[-rank:]]
    motion_features = motion @ basis
    motion_features /= np.sqrt(
        np.mean(motion_features * motion_features, axis=0, keepdims=True) + 1e-12
    )
    spatial = (rest - rest.mean(axis=0)) / (rest.std(axis=0) + 1e-12)
    features = np.concatenate([motion_features, spatial_weight * spatial], axis=1)

    rng = np.random.default_rng(seed)
    centers = np.empty((num_clusters, features.shape[1]), dtype=np.float64)
    centers[0] = features[rng.integers(num_points)]
    nearest_squared = np.full(num_points, np.inf)
    for index in range(1, num_clusters):
        squared = np.sum((features - centers[index - 1]) ** 2, axis=1)
        nearest_squared = np.minimum(nearest_squared, squared)
        total = nearest_squared.sum()
        if total <= 0:
            chosen = int(np.argmax(nearest_squared))
        else:
            chosen = int(rng.choice(num_points, p=nearest_squared / total))
        centers[index] = features[chosen]

    previous = None
    for _ in range(iterations):
        labels, distances = _assign_to_centers(features, centers)
        if previous is not None and np.array_equal(labels, previous):
            break
        previous = labels.copy()
        counts = np.bincount(labels, minlength=num_clusters)
        sums = np.zeros_like(centers)
        np.add.at(sums, labels, features)
        nonempty = counts > 0
        centers[nonempty] = sums[nonempty] / counts[nonempty, None]
        for empty in np.flatnonzero(~nonempty):
            chosen = int(np.argmax(distances))
            centers[empty] = features[chosen]
            distances[chosen] = -1.0
    labels, _ = _assign_to_centers(features, centers)
    return labels


def _configure_backend_protocol(backend: Any, protocol: str) -> None:
    if protocol != "paper_legacy":
        return
    controls = {
        "clusterThreshold": 3,
        "labelAssignmentMode": 1,
        "splitSeedNeighbors": 0,
        "pruneDelay": 0,
        "zeroGrowthPatience": 0,
        "pointCloudSmoothMode": 0,
    }
    missing = [name for name in controls if not hasattr(backend, name)]
    if missing:
        raise RuntimeError(
            "paper_legacy requires the vendored pyssdr backend controls: "
            + ", ".join(missing)
        )
    for name, value in controls.items():
        setattr(backend, name, value)


def decompose_ssdr(
    sequence: DynamicPointSequence,
    config: SSDRConfig = SSDRConfig(),
    backend: Optional[Any] = None,
) -> SSDRResult:
    """Fit SSDR, accepting an injected Dem Bones-compatible backend for tests."""

    if not isinstance(sequence, DynamicPointSequence):
        sequence = DynamicPointSequence(sequence)
    if config.rest_frame >= sequence.num_frames:
        raise ValueError("rest_frame is outside the point sequence")
    if backend is None:
        try:
            import pyssdr
        except ImportError as exc:
            raise ImportError(
                "SSDR extraction requires pyssdr/DemBones; install the project SSDR backend"
            ) from exc
        backend = pyssdr.MyDemBones()
    backend_metadata = _backend_metadata(backend)

    backend.tolerance = config.tolerance
    backend.patience = config.patience
    backend.nIters = config.max_iterations
    backend.nnz = min(config.max_influences, config.num_bones)
    backend.weightsSmooth = config.weight_smoothness
    backend.bindUpdate = config.bind_update
    _configure_backend_protocol(backend, config.protocol)

    points = sequence.points
    frame_order = np.concatenate([
        np.asarray([config.rest_frame]),
        np.arange(config.rest_frame),
        np.arange(config.rest_frame + 1, sequence.num_frames),
    ])
    ordered_points = points[frame_order]
    trajectory = ordered_points.transpose(0, 2, 1).reshape(
        3 * sequence.num_frames, sequence.num_points
    )
    faces = [] if sequence.faces is None else sequence.faces.tolist()
    backend.load_data(trajectory, faces)
    backend.nB = config.num_bones
    supports_motion_init = all(
        hasattr(backend, name) for name in ("labelToWeights", "computeTransFromLabel")
    )
    use_motion_init = config.initialization == "motion" or (
        config.initialization == "auto"
        and config.protocol == "motion_pointcloud"
        and supports_motion_init
    )
    initialized_bones = None
    if use_motion_init:
        if not supports_motion_init:
            raise TypeError("SSDR backend does not support motion label initialization")
        labels = motion_cluster_labels(
            ordered_points,
            config.num_bones,
            rest_frame=0,
            sampled_frames=config.initialization_frames,
            components=config.initialization_components,
            spatial_weight=config.initialization_spatial_weight,
            seed=config.random_seed,
        )
        initialized_bones = int(len(np.unique(labels)))
        if initialized_bones != config.num_bones:
            raise RuntimeError(
                f"motion initialization returned {initialized_bones} nonempty clusters; "
                f"requested {config.num_bones}"
            )
        backend.label = labels
        backend.labelToWeights()
        backend.computeTransFromLabel()
    with tempfile.TemporaryDirectory(prefix="gaussianimate_ssdr_") as directory:
        raw_weights, raw_transforms, backend_rmse = backend.run_ssdr(
            config.num_bones, str(Path(directory) / "decomposition.fbx")
        )
    weights = _dense_weights(raw_weights, sequence.num_points)
    if initialized_bones is None:
        initialized_bones = int(getattr(backend, "initializedBones", weights.shape[1]))
    if config.require_num_bones and weights.shape[1] != config.num_bones:
        raise RuntimeError(
            f"SSDR backend returned {weights.shape[1]} bones; requested "
            f"{config.num_bones}. Use motion initialization or inspect the input trajectory."
        )
    ordered_transforms = _reshape_transforms(
        raw_transforms, sequence.num_frames, weights.shape[1]
    )
    transforms = np.empty_like(ordered_transforms)
    transforms[frame_order] = ordered_transforms
    reconstruction = linear_blend_skinning(points[config.rest_frame], weights, transforms)
    rmse = float(np.sqrt(np.mean((reconstruction - points) ** 2)))
    if not np.isfinite(rmse):
        rmse = float(backend_rmse)
    return SSDRResult(
        weights=weights,
        transformations=transforms,
        bone_positions=bone_position_sequence(transforms, weights, points[config.rest_frame]),
        reconstruction=reconstruction,
        rest_pose=points[config.rest_frame],
        rmse=rmse,
        protocol=config.protocol,
        requested_bones=config.num_bones,
        initialized_bones=initialized_bones,
        **backend_metadata,
    )
