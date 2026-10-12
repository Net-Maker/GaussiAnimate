"""Coverage-aware aggregation for experiment results."""

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

from .engine.storage import write_json


def _flatten_metrics(value, prefix="") -> Dict[str, float]:
    flattened = {}
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else key
            flattened.update(_flatten_metrics(item, child))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        flattened[prefix] = float(value)
    return flattened


def _read_json(path: Path) -> Dict[str, object]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _metric_values(job_dir: Path) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    sources = {}
    for metric_path in sorted(job_dir.glob("*metrics.json")):
        data = _read_json(metric_path)
        for key, value in _flatten_metrics(
            data.get("metrics", {})
        ).items():
            if key in metrics:
                raise ValueError(
                    f"duplicate metric {key!r} in "
                    f"{sources[key]} and {metric_path}"
                )
            metrics[key] = value
            sources[key] = metric_path
    return metrics


def _row_from_job(job) -> Dict[str, object]:
    manifest = _read_json(job.run_dir / "job.json")
    status = manifest.get("status", "pending")
    if manifest and int(manifest.get("schema_version", 0)) < 2:
        status = "legacy_stale"
    row: Dict[str, object] = {
        "dataset": job.dataset["id"],
        "case": job.case["id"],
        "method": job.method["id"],
        "mode": job.method.get("mode", "native"),
        "status": status,
    }
    if status == "complete":
        row.update(_metric_values(job.run_dir))
    return row


def _row_from_manifest(manifest_path: Path) -> Dict[str, object]:
    manifest = _read_json(manifest_path)
    status = manifest.get("status", "pending")
    if manifest and int(manifest.get("schema_version", 0)) < 2:
        status = "legacy_stale"
    row: Dict[str, object] = {
        "dataset": manifest.get("dataset", "unknown"),
        "case": manifest.get("case", "unknown"),
        "method": manifest.get("method", "unknown"),
        "mode": manifest.get("mode"),
        "status": status,
    }
    if status == "complete":
        row.update(_metric_values(manifest_path.parent))
    return row


def _write_csv(path: Path, rows, columns) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=columns,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def aggregate_run(
    output_dir,
    expected_jobs: Optional[Iterable[object]] = None,
) -> Dict[str, object]:
    """Aggregate only complete, full-coverage method groups."""

    output_dir = Path(output_dir)
    if expected_jobs is None:
        rows = [
            _row_from_manifest(path)
            for path in sorted(output_dir.glob("*/*/*/job.json"))
        ]
        scope = "discovered_manifests"
    else:
        rows = [_row_from_job(job) for job in expected_jobs]
        scope = "expected_config_matrix"

    numeric_keys = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if isinstance(value, (int, float))
            and not isinstance(value, bool)
        }
    )
    groups = {}
    for row in rows:
        key = (row["dataset"], row["method"])
        groups.setdefault(key, []).append(row)

    averages = []
    for (dataset, method), items in sorted(groups.items()):
        complete = sum(
            item.get("status") == "complete" for item in items
        )
        aggregate = {
            "dataset": dataset,
            "method": method,
            "cases": len(items),
            "complete": complete,
            "status": (
                "complete" if complete == len(items) else "partial"
            ),
        }
        if aggregate["status"] == "complete":
            for metric in numeric_keys:
                values = [
                    item[metric] for item in items if metric in item
                ]
                if len(values) == len(items):
                    aggregate[metric] = float(np.mean(values))
        averages.append(aggregate)

    result = {
        "schema_version": 2,
        "scope": scope,
        "rows": rows,
        "averages": averages,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "summary.json", result)

    row_columns = [
        "dataset",
        "case",
        "method",
        "mode",
        "status",
    ] + numeric_keys
    _write_csv(output_dir / "results.csv", rows, row_columns)
    average_columns = [
        "dataset",
        "method",
        "cases",
        "complete",
        "status",
    ] + numeric_keys
    _write_csv(
        output_dir / "averages.csv",
        averages,
        average_columns,
    )

    lines = ["# Experiment Summary", ""]
    if averages:
        columns_md = [
            "Dataset",
            "Method",
            "Cases",
            "Complete",
            "Status",
        ] + numeric_keys
        lines.append("| " + " | ".join(columns_md) + " |")
        lines.append(
            "|" + "|".join(["---"] * len(columns_md)) + "|"
        )
        for row in averages:
            values = [
                row["dataset"],
                row["method"],
                str(row["cases"]),
                str(row["complete"]),
                row["status"],
            ]
            values.extend(
                f"{row[key]:.6g}" if key in row else ""
                for key in numeric_keys
            )
            lines.append("| " + " | ".join(values) + " |")
    (output_dir / "summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    return result
