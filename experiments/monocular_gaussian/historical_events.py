"""Recover paper-time test metrics from historical TensorBoard event files."""

import argparse
import hashlib
import json
from pathlib import Path

import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


TAGS = {
    "psnr": "test/loss_viewpoint - psnr",
    "ssim": "test/loss_viewpoint - ssim",
    "lpips": "test/loss_viewpoint - lpips",
    "ms_ssim": "test/loss_viewpoint - ms-ssim",
    "alex_lpips": "test/loss_viewpoint - alex-lpips",
}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _best_event(model_root):
    candidates = []
    errors = []
    event_files = sorted(Path(model_root).glob("events.out.tfevents*"))
    for event_file in event_files:
        try:
            accumulator = EventAccumulator(
                str(event_file), size_guidance={"scalars": 0}
            )
            accumulator.Reload()
        except Exception as exc:
            errors.append({"event_file": str(event_file.resolve()), "error": repr(exc)})
            continue
        available = set(accumulator.Tags()["scalars"])
        if not all(tag in available for tag in TAGS.values()):
            continue
        event_sha256 = _sha256(event_file)
        values = {
            name: accumulator.Scalars(tag) for name, tag in TAGS.items()
        }
        by_step = {
            name: {item.step: item for item in items}
            for name, items in values.items()
        }
        for psnr in values["psnr"]:
            if not all(psnr.step in by_step[name] for name in TAGS if name != "psnr"):
                continue
            candidates.append(
                {
                    "step": psnr.step,
                    "wall_time": psnr.wall_time,
                    "metrics": {
                        name: float(by_step[name][psnr.step].value)
                        for name in TAGS
                    },
                    "event_file": str(event_file.resolve()),
                    "event_sha256": event_sha256,
                }
            )
    if not candidates:
        return None, errors
    return (
        max(candidates, key=lambda item: (item["metrics"]["psnr"], item["wall_time"])),
        errors,
    )


def extract(audit_path, config_path, output):
    audit = json.loads(Path(audit_path).read_text(encoding="utf-8"))
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    expected_cases = {
        dataset["id"]: len(dataset["cases"]) for dataset in config["datasets"]
    }
    rows = []
    for record in audit["records"]:
        if not record["ready"]:
            continue
        model_root = Path(record["render_root"]).parents[1]
        best, event_errors = _best_event(model_root)
        rows.append(
            {
                "method": record["method"],
                "dataset": record["dataset"],
                "case": record["case"],
                "model_root": str(model_root.resolve()),
                "ready": best is not None,
                "best": best,
                "event_errors": event_errors,
            }
        )

    aggregates = []
    for method in sorted({row["method"] for row in rows}):
        for dataset in sorted({row["dataset"] for row in rows}):
            selected = [
                row
                for row in rows
                if row["ready"]
                and row["method"] == method
                and row["dataset"] == dataset
            ]
            if not selected:
                continue
            metrics = {
                name: sum(row["best"]["metrics"][name] for row in selected)
                / len(selected)
                for name in TAGS
            }
            paper = config.get("paper_metrics", {}).get(method, {}).get(dataset, {})
            aggregates.append(
                {
                    "method": method,
                    "dataset": dataset,
                    "cases": len(selected),
                    "expected_cases": expected_cases[dataset],
                    "complete": len(selected) == expected_cases[dataset],
                    "metrics": metrics,
                    "paper_metrics": paper,
                    "difference_from_paper": {
                        name: metrics[name] - value
                        for name, value in paper.items()
                        if name in metrics
                    },
                }
            )
    result = {
        "protocol": {
            "source": "TensorBoard test scalars written during historical training",
            "selection": "maximum test PSNR, with all metrics from the same step",
            "case_aggregation": "unweighted mean of per-sequence best-step metrics",
            "table_columns": {
                "psnr": TAGS["psnr"],
                "ssim": TAGS["ssim"],
                "lpips": TAGS["lpips"],
            },
        },
        "complete": bool(aggregates) and all(item["complete"] for item in aggregates),
        "aggregates": aggregates,
        "per_case": rows,
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = extract(args.audit, args.config, args.output)
    print(json.dumps(result["aggregates"], sort_keys=True))


if __name__ == "__main__":
    main()
