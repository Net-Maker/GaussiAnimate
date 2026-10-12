"""Audit and package historical render evidence for the synthetic table."""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import yaml


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if not isinstance(value, str):
        return value

    def replace(match):
        name, default = match.group(1), match.group(2)
        if name in os.environ:
            return os.environ[name]
        if default is not None:
            return default
        raise ValueError(f"required environment variable is unset: {name}")

    return os.path.expanduser(ENV_PATTERN.sub(replace, value))


def load_resource_config(path: Path) -> Dict[str, Any]:
    path = Path(path)
    config = _expand(yaml.safe_load(path.read_text(encoding="utf-8")))
    if config.get("schema_version") != 1:
        raise ValueError("monocular resource config requires schema_version: 1")
    if not isinstance(config.get("roots"), dict):
        raise ValueError("monocular resource config requires roots")
    if not isinstance(config.get("datasets"), list) or not config["datasets"]:
        raise ValueError("monocular resource config requires datasets")
    if not isinstance(config.get("methods"), list) or not config["methods"]:
        raise ValueError("monocular resource config requires methods")
    config["roots"] = {
        key: str(Path(value).resolve()) for key, value in config["roots"].items()
    }
    return config


def _image_files(directory: Path) -> Dict[str, Path]:
    if not directory.is_dir():
        return {}
    return {
        path.stem: path
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }


def _iteration(path: Path) -> int:
    match = re.fullmatch(r"ours_(\d+)", path.name)
    return int(match.group(1)) if match else -1


def _cfg_flags(path: Path, names: Iterable[str]) -> Dict[str, Optional[bool]]:
    if not path.is_file():
        return {name: None for name in names}
    text = path.read_text(encoding="utf-8", errors="replace")
    flags: Dict[str, Optional[bool]] = {}
    for name in names:
        matches = re.findall(rf"\b{re.escape(name)}=(True|False)\b", text)
        flags[name] = matches[-1] == "True" if matches else None
    return flags


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest_sha256(paths: Mapping[str, Path], content_hash: bool) -> str:
    digest = hashlib.sha256()
    for name, path in sorted(paths.items()):
        stat = path.stat()
        digest.update(name.encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        if content_hash:
            digest.update(_sha256(path).encode("ascii"))
    return digest.hexdigest()


def _find_cfg(source_root: Path, method: Mapping[str, Any], case: str) -> Optional[Path]:
    for pattern in method.get("cfg_patterns", []):
        path = source_root / pattern.format(case=case)
        if path.is_file():
            return path
    return None


def _case_record(
    historical_root: Path,
    dataset: str,
    case: str,
    expected_frames: int,
    method: Mapping[str, Any],
    content_hash: bool,
) -> Dict[str, Any]:
    source_root = historical_root / method["source"]
    candidates = []
    for pattern in method.get("case_patterns", []):
        candidates.extend(source_root.glob(pattern.format(case=case)))
    candidates = sorted(
        {path.resolve() for path in candidates if path.is_dir()},
        key=lambda path: (_iteration(path), str(path)),
        reverse=True,
    )
    cfg_path = _find_cfg(source_root, method, case)
    required_cfg = method.get("required_cfg", {})
    cfg_flags = _cfg_flags(cfg_path, required_cfg) if cfg_path else {
        name: None for name in required_cfg
    }
    cfg_matches = all(cfg_flags.get(name) == expected for name, expected in required_cfg.items())

    selected = None
    paired: Dict[str, Path] = {}
    reference: Dict[str, Path] = {}
    prediction_only = []
    reference_only = []
    for candidate in candidates:
        predictions = _image_files(candidate / "renders")
        references = _image_files(candidate / "gt")
        common = sorted(set(predictions) & set(references))
        if len(common) != expected_frames:
            continue
        selected = candidate
        paired = {stem: predictions[stem] for stem in common}
        reference = {stem: references[stem] for stem in common}
        prediction_only = sorted(set(predictions) - set(references))
        reference_only = sorted(set(references) - set(predictions))
        break

    ready = selected is not None and cfg_matches
    record: Dict[str, Any] = {
        "dataset": dataset,
        "case": case,
        "method": method["id"],
        "classification": method["classification"],
        "ready": ready,
        "expected_frames": expected_frames,
        "paired_frames": len(paired),
        "iteration": _iteration(selected) if selected else None,
        "render_root": str(selected) if selected else None,
        "cfg_args": str(cfg_path) if cfg_path else None,
        "cfg_flags": cfg_flags,
        "cfg_matches": cfg_matches,
        "prediction_only_count": len(prediction_only),
        "prediction_only_sample": prediction_only[:5],
        "reference_only_count": len(reference_only),
        "reference_only_sample": reference_only[:5],
    }
    if selected:
        all_paths = {
            **{f"renders/{stem}{path.suffix.lower()}": path for stem, path in paired.items()},
            **{f"gt/{stem}{path.suffix.lower()}": path for stem, path in reference.items()},
        }
        record["image_manifest_sha256"] = _manifest_sha256(all_paths, content_hash)
        record["hash_mode"] = "content" if content_hash else "name_and_size"
    if cfg_path:
        record["cfg_sha256"] = _sha256(cfg_path)
    return record


def audit_resources(
    config: Mapping[str, Any],
    selected_methods: Optional[Iterable[str]] = None,
    selected_cases: Optional[Iterable[str]] = None,
    content_hash: bool = False,
) -> Dict[str, Any]:
    method_filter = set(selected_methods or [])
    case_filter = set(selected_cases or [])
    methods = {method["id"]: method for method in config["methods"]}
    cases = {
        case if isinstance(case, str) else case["id"]
        for dataset in config["datasets"]
        for case in dataset["cases"]
    }
    unknown_methods = method_filter - set(methods)
    unknown_cases = case_filter - cases
    if unknown_methods:
        raise ValueError(f"unknown monocular methods: {sorted(unknown_methods)}")
    if unknown_cases:
        raise ValueError(f"unknown monocular cases: {sorted(unknown_cases)}")

    records = []
    historical_root = Path(config["roots"]["historical_riggs"])
    for dataset in config["datasets"]:
        expected = int(dataset["expected_test_frames"])
        for raw_case in dataset["cases"]:
            case = raw_case if isinstance(raw_case, str) else raw_case["id"]
            if case_filter and case not in case_filter:
                continue
            for method_id, method in methods.items():
                if method_filter and method_id not in method_filter:
                    continue
                records.append(
                    _case_record(
                        historical_root,
                        dataset["id"],
                        case,
                        expected,
                        method,
                        content_hash,
                    )
                )
    by_method = {
        method_id: {
            "ready": sum(record["ready"] for record in records if record["method"] == method_id),
            "total": sum(record["method"] == method_id for record in records),
        }
        for method_id in methods
        if not method_filter or method_id in method_filter
    }
    return {
        "historical_root": str(historical_root),
        "all_ready": bool(records) and all(record["ready"] for record in records),
        "by_method": by_method,
        "external_methods": list(config.get("external_methods", [])),
        "records": records,
    }


def _ensure_link(path: Path, target: Path) -> str:
    if path.is_symlink():
        return "unchanged" if path.resolve() == target.resolve() else "conflict"
    if path.exists():
        return "unchanged" if path.resolve() == target.resolve() else "conflict"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.symlink_to(target)
    return "created"


def prepare_release_links(report: Mapping[str, Any], release_root: Path) -> Dict[str, Any]:
    release_root = Path(release_root)
    links = []
    for record in report["records"]:
        if not record["ready"]:
            continue
        source = Path(record["render_root"])
        common = sorted(set(_image_files(source / "renders")) & set(_image_files(source / "gt")))
        destination = release_root / record["method"] / record["dataset"] / record["case"]
        for kind in ("renders", "gt"):
            files = _image_files(source / kind)
            for stem in common:
                target = files[stem]
                path = destination / kind / target.name
                links.append(
                    {
                        "path": str(path),
                        "target": str(target),
                        "action": _ensure_link(path, target),
                    }
                )
        provenance = {
            key: value
            for key, value in record.items()
            if key not in {
                "prediction_only_sample",
                "reference_only_sample",
            }
        }
        provenance_path = destination / "provenance.json"
        provenance_path.parent.mkdir(parents=True, exist_ok=True)
        provenance_path.write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    conflicts = [link for link in links if link["action"] == "conflict"]
    return {
        "release_root": str(release_root.resolve()),
        "links_total": len(links),
        "created": sum(link["action"] == "created" for link in links),
        "unchanged": sum(link["action"] == "unchanged" for link in links),
        "conflicts": sum(link["action"] == "conflict" for link in links),
        "conflict_sample": conflicts[:10],
    }


def evaluate_release(
    report: Mapping[str, Any],
    config: Mapping[str, Any],
    output_root: Path,
    metrics: Iterable[str],
    python: str,
    device: str,
) -> Dict[str, Any]:
    """Run the historical paper metric implementation on packaged image pairs."""

    metrics = tuple(metrics)
    release_root = Path(config["roots"]["release"])
    output_root = Path(output_root)
    script = Path(__file__).with_name("historical_metrics.py")
    rows = []
    for record in report["records"]:
        if not record["ready"]:
            continue
        source = (
            release_root
            / record["method"]
            / record["dataset"]
            / record["case"]
        )
        output = (
            output_root
            / record["method"]
            / record["dataset"]
            / record["case"]
            / "image_metrics.json"
        )
        command = [
            python,
            str(script),
            "--prediction-dir",
            str(source / "renders"),
            "--reference-dir",
            str(source / "gt"),
            "--output",
            str(output),
            "--metrics",
            *metrics,
            "--device",
            device,
            "--riggs-root",
            config["roots"]["historical_riggs"],
        ]
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
        result = json.loads(output.read_text(encoding="utf-8"))
        rows.append(
            {
                "method": record["method"],
                "dataset": record["dataset"],
                "case": record["case"],
                "iteration": record["iteration"],
                "frames": result["frames"],
                "metrics": result["metrics"],
                "result": str(output.resolve()),
            }
        )

    expected_cases = {
        dataset["id"]: len(dataset["cases"]) for dataset in config["datasets"]
    }
    aggregates = []
    for method in sorted({row["method"] for row in rows}):
        for dataset in sorted({row["dataset"] for row in rows}):
            selected = [
                row
                for row in rows
                if row["method"] == method and row["dataset"] == dataset
            ]
            if not selected:
                continue
            values = {
                metric: sum(row["metrics"][metric] for row in selected) / len(selected)
                for metric in metrics
            }
            paper = config.get("paper_metrics", {}).get(method, {}).get(dataset, {})
            aggregates.append(
                {
                    "method": method,
                    "dataset": dataset,
                    "cases": len(selected),
                    "expected_cases": expected_cases[dataset],
                    "complete": len(selected) == expected_cases[dataset],
                    "metrics": values,
                    "paper_metrics": {
                        metric: paper[metric] for metric in metrics if metric in paper
                    },
                    "difference_from_paper": {
                        metric: values[metric] - paper[metric]
                        for metric in metrics
                        if metric in paper
                    },
                }
            )
    summary = {
        "protocol": {
            "case_aggregation": "unweighted mean of per-sequence means",
            "metric_implementation": "historical RigGS metrics.py",
            "metrics": list(metrics),
            "device": device,
        },
        "complete": bool(aggregates) and all(item["complete"] for item in aggregates),
        "aggregates": aggregates,
        "per_case": rows,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary["summary"] = str(summary_path.resolve())
    return summary


def package_evaluation_results(source: Path, release_root: Path) -> Dict[str, Any]:
    source = Path(source)
    if not source.is_dir():
        raise FileNotFoundError(f"evaluation result directory does not exist: {source}")
    destination = Path(release_root) / "evaluation"
    shutil.copytree(source, destination, dirs_exist_ok=True)
    summaries = {}
    for name in ("summary.json", "tensorboard_summary.json"):
        path = destination / name
        if path.is_file():
            summaries[name] = {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
            }
    provenance = {
        "classification": "historical_metric_evidence",
        "source": str(source.resolve()),
        "summaries": summaries,
    }
    provenance_path = destination / "provenance.json"
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "destination": str(destination.resolve()),
        "summaries": summaries,
        "provenance": str(provenance_path.resolve()),
    }


def main() -> None:
    default_config = Path(__file__).with_name("resources.yaml")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--method", action="append")
    parser.add_argument("--case", action="append")
    parser.add_argument("--content-hash", action="store_true")
    parser.add_argument("--prepare-release", action="store_true")
    parser.add_argument("--evaluate-release", action="store_true")
    parser.add_argument("--extract-events", action="store_true")
    parser.add_argument("--package-evaluation", action="store_true")
    parser.add_argument("--metrics", nargs="+", default=["psnr", "ssim", "lpips"])
    parser.add_argument(
        "--metric-python",
        default="~/anaconda3/envs/riggs/bin/python",
    )
    parser.add_argument("--metric-device", default="cuda")
    parser.add_argument(
        "--metric-output",
        type=Path,
        default=Path("experiments/runs/monocular_historical"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-ready", action="store_true")
    args = parser.parse_args()
    config = load_resource_config(args.config)
    report = audit_resources(
        config,
        selected_methods=args.method,
        selected_cases=args.case,
        content_hash=args.content_hash,
    )
    if args.prepare_release:
        report["release"] = prepare_release_links(
            report, Path(config["roots"]["release"])
        )
    if args.evaluate_release:
        report["evaluation"] = evaluate_release(
            report,
            config,
            args.metric_output,
            args.metrics,
            args.metric_python,
            args.metric_device,
        )
    if args.extract_events:
        if not args.output:
            raise ValueError("--extract-events requires --output for stable provenance")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        event_output = args.metric_output / "tensorboard_summary.json"
        subprocess.run(
            [
                args.metric_python,
                str(Path(__file__).with_name("historical_events.py")),
                "--audit",
                str(args.output),
                "--config",
                str(args.config),
                "--output",
                str(event_output),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        report["tensorboard"] = json.loads(
            event_output.read_text(encoding="utf-8")
        )
    if args.package_evaluation:
        report["evaluation_package"] = package_evaluation_results(
            args.metric_output,
            Path(config["roots"]["release"]),
        )
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    if args.require_ready and not report["all_ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
