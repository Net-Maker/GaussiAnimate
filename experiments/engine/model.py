"""Typed experiment jobs expanded from declarative configs."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


@dataclass(frozen=True)
class Job:
    experiment: Dict[str, Any]
    dataset: Dict[str, Any]
    case: Dict[str, Any]
    method: Dict[str, Any]
    output_root: Path
    config_path: Path

    @property
    def run_dir(self) -> Path:
        return (
            self.output_root
            / self.dataset["id"]
            / self.case["id"]
            / self.method["id"]
        )

    @property
    def key(self) -> str:
        return (
            f"{self.dataset['id']}/"
            f"{self.case['id']}/"
            f"{self.method['id']}"
        )

    def context(self) -> Dict[str, Any]:
        return {
            "experiment": self.experiment,
            "dataset": self.dataset,
            "case": self.case,
            "method": self.method,
            "output_root": str(self.output_root),
            "run_dir": str(self.run_dir),
            "config_dir": str(self.config_path.parent),
            "config_path": str(self.config_path),
        }


def expand_jobs(
    config: Dict[str, Any],
    datasets: Optional[Sequence[str]] = None,
    cases: Optional[Sequence[str]] = None,
    methods: Optional[Sequence[str]] = None,
) -> List[Job]:
    """Expand the dataset x case x method matrix with optional filters."""

    dataset_filter = set(datasets or [])
    case_filter = set(cases or [])
    method_filter = set(methods or [])
    config_path = Path(config["_config_path"])
    output_root = Path(config["output_dir"])
    if not output_root.is_absolute():
        output_root = (config_path.parent / output_root).resolve()

    all_dataset_ids = [item["id"] for item in config["datasets"]]
    jobs = []
    for dataset in config["datasets"]:
        if dataset_filter and dataset["id"] not in dataset_filter:
            continue
        normalized_cases = [
            {"id": case} if isinstance(case, str) else case
            for case in dataset["cases"]
        ]
        for case in normalized_cases:
            if case_filter and case["id"] not in case_filter:
                continue
            for method in config["methods"]:
                if method_filter and method["id"] not in method_filter:
                    continue
                if dataset["id"] not in method.get(
                    "datasets",
                    all_dataset_ids,
                ):
                    continue
                jobs.append(
                    Job(
                        experiment=config["experiment"],
                        dataset=dataset,
                        case=case,
                        method=method,
                        output_root=output_root,
                        config_path=config_path,
                    )
                )
    return jobs
