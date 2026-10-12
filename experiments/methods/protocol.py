"""Shared contracts for registered-point human animation methods."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional

import numpy as np


@dataclass(frozen=True)
class HeldOutPointProtocol:
    """Contiguous training-prefix and held-out-tail point protocol."""

    frames: int
    points: int
    training_frames: int
    train_ratio: float

    @classmethod
    def from_shape(
        cls,
        shape,
        train_ratio: float = 0.8,
    ) -> "HeldOutPointProtocol":
        if len(shape) != 3 or int(shape[2]) != 3:
            raise ValueError(
                "registered point sequence must have shape [T, N, 3]"
            )
        if not 0.0 < train_ratio < 1.0:
            raise ValueError("train_ratio must be in (0, 1)")
        frames, points = int(shape[0]), int(shape[1])
        training_frames = int(frames * train_ratio)
        if not 0 < training_frames < frames:
            raise ValueError(
                "train_ratio produced an empty training or test split"
            )
        return cls(
            frames,
            points,
            training_frames,
            float(train_ratio),
        )

    @property
    def test_frames(self) -> int:
        return self.frames - self.training_frames

    @property
    def test_frame_ids(self):
        return np.arange(
            self.training_frames,
            self.frames,
            dtype=np.int64,
        )

    @property
    def full_shape(self):
        return (self.frames, self.points, 3)

    @property
    def test_shape(self):
        return (self.test_frames, self.points, 3)

    def validate_full(self, value, label: str) -> None:
        if tuple(value.shape) != self.full_shape:
            raise ValueError(
                f"{label} has shape {value.shape}, "
                f"expected {self.full_shape}"
            )

    def validate_test(self, value, label: str) -> None:
        if tuple(value.shape) != self.test_shape:
            raise ValueError(
                f"{label} has shape {value.shape}, "
                f"expected {self.test_shape}"
            )

    def metadata(self) -> Dict[str, object]:
        split = (
            "contiguous_prefix_80_20"
            if np.isclose(self.train_ratio, 0.8)
            else "contiguous_prefix"
        )
        return {
            "frames": self.frames,
            "training_frames": self.training_frames,
            "test_frames": self.test_frames,
            "test_frame_ids": self.test_frame_ids.tolist(),
            "num_points": self.points,
            "split": split,
            "train_ratio": self.train_ratio,
            "protocol_id": "contiguous_prefix_v1",
        }


class PointErrorAccumulator:
    """Numerically stable streaming point-distance metrics."""

    def __init__(self) -> None:
        self._squared_sum = 0.0
        self._distance_sum = 0.0
        self._count = 0

    def update(self, prediction, target) -> None:
        prediction = np.asarray(prediction)
        target = np.asarray(target)
        if prediction.shape != target.shape:
            raise ValueError(
                "prediction and target shapes differ: "
                f"{prediction.shape} != {target.shape}"
            )
        if prediction.shape[-1:] != (3,):
            raise ValueError("point arrays must end in xyz coordinates")
        distance = np.linalg.norm(prediction - target, axis=-1)
        if not np.isfinite(distance).all():
            raise ValueError("point error contains non-finite values")
        self._squared_sum += float(
            np.sum(distance * distance, dtype=np.float64)
        )
        self._distance_sum += float(
            np.sum(distance, dtype=np.float64)
        )
        self._count += int(distance.size)

    def metrics(self, unit_scale: float = 1000.0) -> Dict[str, float]:
        if self._count == 0:
            raise ValueError("cannot report metrics without points")
        return {
            "point_mean_error_mm": float(
                self._distance_sum / self._count * unit_scale
            ),
            "point_rmse_mm": float(
                np.sqrt(self._squared_sum / self._count)
                * unit_scale
            ),
        }


def point_error_metrics(
    prediction,
    target,
    unit_scale: float = 1000.0,
) -> Dict[str, float]:
    accumulator = PointErrorAccumulator()
    accumulator.update(prediction, target)
    return accumulator.metrics(unit_scale)


def write_prediction_artifacts(
    output_dir,
    protocol: HeldOutPointProtocol,
    *,
    test_prediction,
    full_prediction=None,
    training_prefix=None,
    bone_transforms=None,
    extra_arrays: Optional[Mapping[str, np.ndarray]] = None,
) -> Dict[str, str]:
    """Write the common all-point method artifact contract."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    test_prediction = np.asarray(test_prediction, dtype=np.float32)
    protocol.validate_test(test_prediction, "test prediction")
    if not np.isfinite(test_prediction).all():
        raise ValueError("test prediction contains non-finite values")

    if full_prediction is not None and training_prefix is not None:
        raise ValueError(
            "provide full_prediction or training_prefix, not both"
        )
    full_path = output_dir / "warped_sequence.npy"
    full = np.lib.format.open_memmap(
        full_path,
        mode="w+",
        dtype=np.float32,
        shape=protocol.full_shape,
    )
    if full_prediction is not None:
        full_prediction = np.asarray(full_prediction, dtype=np.float32)
        protocol.validate_full(full_prediction, "full prediction")
        if not np.isfinite(full_prediction).all():
            raise ValueError("full prediction contains non-finite values")
        full[:] = full_prediction
    else:
        if training_prefix is None:
            raise ValueError(
                "training_prefix is required without full_prediction"
            )
        training_prefix = np.asarray(training_prefix, dtype=np.float32)
        expected = (
            protocol.training_frames,
            protocol.points,
            3,
        )
        if tuple(training_prefix.shape) != expected:
            raise ValueError(
                f"training prefix has shape {training_prefix.shape}, "
                f"expected {expected}"
            )
        if not np.isfinite(training_prefix).all():
            raise ValueError("training prefix contains non-finite values")
        full[: protocol.training_frames] = training_prefix
        full[protocol.training_frames :] = test_prediction
    full.flush()
    del full

    np.save(
        output_dir / "test_warped_sequence.npy",
        test_prediction,
    )
    np.save(output_dir / "frame_ids.npy", protocol.test_frame_ids)
    np.save(
        output_dir / "point_indices.npy",
        np.arange(protocol.points, dtype=np.int64),
    )
    artifacts = {
        "warped_sequence": "warped_sequence.npy",
        "test_warped_sequence": "test_warped_sequence.npy",
        "frame_ids": "frame_ids.npy",
        "point_indices": "point_indices.npy",
    }
    if bone_transforms is not None:
        bone_transforms = np.asarray(
            bone_transforms,
            dtype=np.float32,
        )
        if len(bone_transforms) != protocol.test_frames:
            raise ValueError(
                "bone transforms must have one entry per test frame"
            )
        np.save(
            output_dir / "bone_transforms.npy",
            bone_transforms,
        )
        artifacts["bone_transforms"] = "bone_transforms.npy"
    for name, value in (extra_arrays or {}).items():
        if not name.endswith(".npy"):
            raise ValueError(
                f"extra prediction artifact must be .npy: {name}"
            )
        np.save(output_dir / name, np.asarray(value))
        artifacts[Path(name).stem] = name
    return artifacts
