"""Thread-safe resource assignment for experiment jobs."""

import os
import threading
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional, Sequence


def configure_visible_gpu(gpu_id: Any) -> str:
    """Select a GPU unless a parent scheduler already constrained visibility."""

    return os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(gpu_id))


def _enabled_gpu_value(value: Any) -> bool:
    normalized = str(value).strip().lower()
    return normalized not in {"", "-1", "none", "null", "cpu"}


def stage_requires_gpu(stage: Dict[str, Any]) -> bool:
    """Return whether a stage explicitly declares or selects CUDA execution."""

    if stage.get("resources", {}).get("gpu"):
        return True

    if stage.get("type") == "image_metrics":
        metrics = stage.get("metrics", ["psnr", "ssim", "lpips"])
        device = str(stage.get("device", "")).lower()
        return "lpips" in metrics and device != "cpu"

    environment = stage.get("env", {})
    if (
        "CUDA_VISIBLE_DEVICES" in environment
        and _enabled_gpu_value(environment["CUDA_VISIBLE_DEVICES"])
    ):
        return True

    command = [str(item) for item in stage.get("command", [])]
    for index, item in enumerate(command):
        normalized = item.strip().lower()
        if normalized == "cuda" or normalized.startswith("cuda:"):
            return True
        if normalized.startswith("--gpu="):
            if _enabled_gpu_value(normalized.split("=", 1)[1]):
                return True
            continue
        if normalized.startswith("--device="):
            if normalized.split("=", 1)[1].startswith("cuda"):
                return True
            continue
        if normalized == "--gpu":
            if (
                index + 1 >= len(command)
                or _enabled_gpu_value(command[index + 1])
            ):
                return True
            continue
        if normalized == "--device" and index + 1 < len(command):
            if command[index + 1].strip().lower().startswith("cuda"):
                return True
    return False


def method_requires_gpu(
    method: Dict[str, Any],
    selected_stages: Optional[Sequence[str]] = None,
) -> bool:
    """Inspect the selected declarative stages without executing the method."""

    if method.get("resources", {}).get("gpu"):
        return True
    selected = set(selected_stages or ())
    return any(
        stage_requires_gpu(stage)
        for stage in method.get("stages", [])
        if not selected or stage.get("id") in selected
    )


class GpuScheduler:
    """Assign jobs round-robin while allowing one active job per GPU."""

    def __init__(self, gpu_ids: Sequence[str]):
        self._gpu_ids = tuple(gpu_ids)
        self._locks = {
            gpu_id: threading.Lock()
            for gpu_id in self._gpu_ids
        }

    def gpu_for(self, job_index: int) -> Optional[str]:
        if job_index < 0:
            raise ValueError("job index must be non-negative")
        if not self._gpu_ids:
            return None
        return self._gpu_ids[job_index % len(self._gpu_ids)]

    @contextmanager
    def slot(self, job_index: int) -> Iterator[Optional[str]]:
        """Yield the assigned GPU after acquiring its exclusive job slot."""

        gpu_id = self.gpu_for(job_index)
        if gpu_id is None:
            yield None
            return
        with self._locks[gpu_id]:
            yield gpu_id
