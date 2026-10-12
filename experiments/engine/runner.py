"""Resumable execution and manifest lifecycle for experiment jobs."""

import hashlib
import importlib.util
import inspect
import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .config import ENV_PATTERN, render_template
from .model import Job
from .registry import STAGE_REGISTRY, StageContext
from .storage import write_json
from .source import local_source_signatures


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stage_outputs(stage: Dict[str, Any]) -> List[Path]:
    return [Path(path) for path in stage.get("outputs", [])]


def _path_signature(path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "exists": True,
        "kind": "directory" if path.is_dir() else "file",
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _source_signature(path) -> Dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        return _path_signature(path)
    return {
        **_path_signature(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "dependencies": local_source_signatures(path),
    }


def _command_source_signature(stage: Dict[str, Any]):
    if stage.get("type") != "command":
        return None
    command = [str(value) for value in stage.get("command", [])]
    if "-m" in command:
        index = command.index("-m")
        if index + 1 < len(command):
            module = command[index + 1]
            try:
                spec = importlib.util.find_spec(module)
            except (ImportError, ModuleNotFoundError, ValueError):
                spec = None
            if spec is not None and spec.origin:
                return {
                    "module": module,
                    "source": _source_signature(spec.origin),
                }
    for value in command[1:]:
        if value.endswith(".py") and Path(value).is_file():
            return {"source": _source_signature(value)}
    return None


def _handler_source_signature(stage_type: str) -> Dict[str, Any]:
    handler = STAGE_REGISTRY.handler(stage_type)
    source = inspect.getsourcefile(handler)
    if source is None:
        return {"callable": f"{handler.__module__}.{handler.__qualname__}"}
    return {
        "callable": f"{handler.__module__}.{handler.__qualname__}",
        "source": _source_signature(source),
    }


def stage_fingerprint(job: Job, stage: Dict[str, Any]) -> str:
    """Hash the fully rendered stage contract that controls its cache."""

    payload = {
        "schema_version": 3,
        "experiment": job.experiment["id"],
        "seed": job.experiment["seed"],
        "dataset": job.dataset["id"],
        "case": job.case["id"],
        "method": job.method["id"],
        "protocol": job.method.get("protocol", {}),
        "stage": stage,
        "inputs": [
            _path_signature(path)
            for path in stage.get("inputs", [])
        ],
        "handler": _handler_source_signature(stage["type"]),
        "command_source": _command_source_signature(stage),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _record_path(job: Job, stage_id: str) -> Path:
    return job.run_dir / ".stages" / f"{stage_id}.json"


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _stage_complete(job: Job, stage: Dict[str, Any]) -> bool:
    data = _load_json(_record_path(job, stage["id"]))
    return (
        data.get("status") == "complete"
        and data.get("fingerprint") == stage_fingerprint(job, stage)
        and all(output.exists() for output in _stage_outputs(stage))
    )


def _select_stages(
    job: Job,
    all_stages: Sequence[Dict[str, Any]],
    selected_stages: Optional[Sequence[str]],
) -> List[Dict[str, Any]]:
    if not selected_stages:
        return list(all_stages)
    selected = set(selected_stages)
    known = {stage["id"] for stage in all_stages}
    unknown = selected - known
    if unknown:
        raise ValueError(
            f"job {job.key} has no stages: {', '.join(sorted(unknown))}"
        )
    return [
        stage for stage in all_stages if stage["id"] in selected
    ]


def _require_inputs(stage: Dict[str, Any]) -> None:
    missing = [
        path
        for path in stage.get("inputs", [])
        if not Path(path).exists()
    ]
    if missing:
        raise FileNotFoundError(
            "stage inputs are missing:\n"
            + "\n".join(f"  {path}" for path in missing)
        )


def _require_resolved_environment(value: Any) -> None:
    if isinstance(value, dict):
        for item in value.values():
            _require_resolved_environment(item)
    elif isinstance(value, list):
        for item in value:
            _require_resolved_environment(item)
    elif isinstance(value, str):
        match = ENV_PATTERN.search(value)
        if match:
            raise ValueError(
                "required environment variable is unset for selected job: "
                f"{match.group(1)}"
            )


def _rendered_stages(job: Job) -> List[Dict[str, Any]]:
    context = job.context()
    return [
        render_template(stage, context)
        for stage in job.method["stages"]
    ]


def _new_manifest(job: Job) -> Dict[str, Any]:
    return {
        "schema_version": 2,
        "experiment": job.experiment["id"],
        "target_id": job.experiment["target_id"],
        "paper_target": job.experiment["paper_target"],
        "dataset": job.dataset["id"],
        "case": job.case["id"],
        "method": job.method["id"],
        "mode": job.method.get("mode", "native"),
        "config": str(job.config_path),
        "status": "pending",
        "created_utc": _utc_now(),
        "stages": {},
    }


def _load_manifest(job: Job) -> Dict[str, Any]:
    current = _load_json(job.run_dir / "job.json")
    if (
        current.get("experiment") != job.experiment["id"]
        or current.get("dataset") != job.dataset["id"]
        or current.get("case") != job.case["id"]
        or current.get("method") != job.method["id"]
    ):
        return _new_manifest(job)
    current["schema_version"] = 2
    current["target_id"] = job.experiment["target_id"]
    current["paper_target"] = job.experiment["paper_target"]
    current.setdefault("stages", {})
    return current


def _refresh_manifest_status(
    job: Job,
    manifest: Dict[str, Any],
    all_stages: Sequence[Dict[str, Any]],
) -> None:
    complete = []
    for stage in all_stages:
        stage_id = stage["id"]
        record = _load_json(_record_path(job, stage_id))
        if record:
            manifest["stages"][stage_id] = record
        complete.append(_stage_complete(job, stage))
    if all(complete):
        manifest["status"] = "complete"
        manifest["finished_utc"] = _utc_now()
    elif any(complete):
        manifest["status"] = "partial"
        manifest.pop("finished_utc", None)
    else:
        manifest["status"] = "pending"
        manifest.pop("finished_utc", None)


def adopt_legacy_stages(
    job: Job,
    selected_stages: Sequence[str],
) -> Dict[str, Any]:
    """Fingerprint legacy records whose outputs have a current consumer.

    Adoption is deliberately narrower than execution. A record must predate
    fingerprints, retain the exact declared outputs, and feed at least one
    downstream stage whose current fingerprint is complete. This proves that
    the unchanged legacy output was consumed by current experiment code.
    """

    if not selected_stages:
        raise ValueError("legacy adoption requires at least one selected stage")
    all_stages = _rendered_stages(job)
    stages = _select_stages(job, all_stages, selected_stages)
    if job.method.get("mode") == "blocked":
        raise RuntimeError(
            f"{job.key} is blocked: {job.method['blocked_reason']}"
        )

    job.run_dir.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(job)
    for stage in stages:
        stage_id = stage["id"]
        record_path = _record_path(job, stage_id)
        record = _load_json(record_path)
        expected_outputs = [str(path) for path in _stage_outputs(stage)]
        if record.get("status") != "complete":
            raise ValueError(
                f"cannot adopt {job.key}/{stage_id}: "
                "legacy record is not complete"
            )
        if record.get("fingerprint"):
            raise ValueError(
                f"cannot adopt {job.key}/{stage_id}: "
                "record already has a fingerprint"
            )
        if (
            record.get("id") != stage_id
            or record.get("type") != stage["type"]
            or record.get("outputs") != expected_outputs
        ):
            raise ValueError(
                f"cannot adopt {job.key}/{stage_id}: "
                "legacy record does not match the current stage outputs"
            )
        _require_resolved_environment(stage)
        _require_inputs(stage)
        missing = [
            output for output in expected_outputs
            if not Path(output).exists()
        ]
        if missing:
            raise FileNotFoundError(
                "cannot adopt stage with missing outputs:\n"
                + "\n".join(f"  {path}" for path in missing)
            )

        output_paths = {
            str(Path(path).resolve()) for path in expected_outputs
        }
        consumers = []
        for candidate in all_stages:
            if candidate["id"] == stage_id:
                continue
            candidate_inputs = {
                str(Path(path).resolve())
                for path in candidate.get("inputs", [])
            }
            if (
                output_paths & candidate_inputs
                and _stage_complete(job, candidate)
            ):
                consumers.append(candidate["id"])
        if not consumers:
            raise ValueError(
                f"cannot adopt {job.key}/{stage_id}: no current completed "
                "downstream stage consumes its outputs"
            )

        legacy = dict(record)
        record["fingerprint"] = stage_fingerprint(job, stage)
        record["adopted_utc"] = _utc_now()
        record["adoption"] = {
            "mode": "legacy_record_with_current_downstream_consumer",
            "validated_by": sorted(consumers),
            "legacy_started_utc": legacy.get("started_utc"),
            "legacy_finished_utc": legacy.get("finished_utc"),
            "legacy_elapsed_seconds": legacy.get("elapsed_seconds"),
        }
        write_json(record_path, record)
        manifest["stages"][stage_id] = record

    _refresh_manifest_status(job, manifest, all_stages)
    manifest["last_finished_utc"] = _utc_now()
    write_json(job.run_dir / "job.json", manifest)
    return manifest


def run_job(
    job: Job,
    selected_stages: Optional[Sequence[str]] = None,
    force: bool = False,
    dry_run: bool = False,
    gpu: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute selected stages while retaining the full job state."""

    all_stages = _rendered_stages(job)
    stages = _select_stages(job, all_stages, selected_stages)
    if not stages:
        raise ValueError(f"job {job.key} has no selected stages")
    if dry_run:
        return {
            "job": job.key,
            "mode": job.method.get("mode", "native"),
            "blocked_reason": job.method.get("blocked_reason"),
            "stages": stages,
        }
    if job.method.get("mode") == "blocked":
        raise RuntimeError(
            f"{job.key} is blocked: {job.method['blocked_reason']}"
        )

    job.run_dir.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(job)
    manifest["status"] = "running"
    manifest["last_started_utc"] = _utc_now()
    write_json(job.run_dir / "job.json", manifest)

    for stage in stages:
        stage_id = stage["id"]
        fingerprint = stage_fingerprint(job, stage)
        record_path = _record_path(job, stage_id)
        if not force and _stage_complete(job, stage):
            record = _load_json(record_path)
            record["cache_hit_utc"] = _utc_now()
            manifest["stages"][stage_id] = record
            write_json(job.run_dir / "job.json", manifest)
            continue

        started = time.time()
        record = {
            "id": stage_id,
            "type": stage["type"],
            "status": "running",
            "fingerprint": fingerprint,
            "started_utc": _utc_now(),
        }
        write_json(record_path, record)
        try:
            _require_resolved_environment(stage)
            _require_inputs(stage)
            STAGE_REGISTRY.execute(
                stage,
                StageContext(job=job, gpu=gpu),
            )
            missing = [
                str(output)
                for output in _stage_outputs(stage)
                if not output.exists()
            ]
            if missing:
                raise FileNotFoundError(
                    "stage completed but declared outputs are missing:\n"
                    + "\n".join(f"  {path}" for path in missing)
                )
            record.update(
                status="complete",
                elapsed_seconds=time.time() - started,
                outputs=[
                    str(path) for path in _stage_outputs(stage)
                ],
                finished_utc=_utc_now(),
            )
            write_json(record_path, record)
            manifest["stages"][stage_id] = record
            write_json(job.run_dir / "job.json", manifest)
        except Exception as exc:
            record.update(
                status="failed",
                elapsed_seconds=time.time() - started,
                error=repr(exc),
                traceback=traceback.format_exc(),
                finished_utc=_utc_now(),
            )
            write_json(record_path, record)
            manifest["status"] = "failed"
            manifest["stages"][stage_id] = record
            manifest["last_finished_utc"] = _utc_now()
            write_json(job.run_dir / "job.json", manifest)
            raise

    _refresh_manifest_status(job, manifest, all_stages)
    manifest["last_finished_utc"] = _utc_now()
    write_json(job.run_dir / "job.json", manifest)
    return manifest
