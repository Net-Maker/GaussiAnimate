"""Loading, composition, templating, and validation for paper configs."""

import copy
import os
import re
from pathlib import Path
from typing import Any, Dict, Sequence

import yaml

from .registry import STAGE_REGISTRY
from . import stages as _builtin_stages  # noqa: F401


ENV_PATTERN = re.compile(
    r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}"
)
TEMPLATE_PATTERN = re.compile(
    r"(?<!\$)\{([A-Za-z_][A-Za-z0-9_.]*)\}"
)
TARGET_ID_PATTERN = re.compile(r"[a-z][a-z0-9_]*")


def _expand_environment(value: Any, require: bool) -> Any:
    if isinstance(value, dict):
        return {
            key: _expand_environment(item, require)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_expand_environment(item, require) for item in value]
    if not isinstance(value, str):
        return value

    def replace(match):
        name, default = match.group(1), match.group(2)
        if name in os.environ:
            return os.environ[name]
        if default is not None:
            return default
        if require:
            raise ValueError(
                f"required environment variable is unset: {name}"
            )
        return match.group(0)

    return os.path.expanduser(ENV_PATTERN.sub(replace, value))


def _lookup(context: Dict[str, Any], dotted: str):
    value: Any = context
    for component in dotted.split("."):
        if not isinstance(value, dict) or component not in value:
            raise KeyError(
                f"unknown experiment template field: {dotted}"
            )
        value = value[component]
    return value


def render_template(value: Any, context: Dict[str, Any]) -> Any:
    if isinstance(value, dict):
        return {
            key: render_template(item, context)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [render_template(item, context) for item in value]
    if not isinstance(value, str):
        return value
    full = TEMPLATE_PATTERN.fullmatch(value)
    if full:
        return copy.deepcopy(_lookup(context, full.group(1)))
    return TEMPLATE_PATTERN.sub(
        lambda match: str(_lookup(context, match.group(1))),
        value,
    )


def _ids(items: Sequence[Dict[str, Any]], label: str) -> None:
    values = [item.get("id") for item in items]
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError(f"every {label} must have a non-empty id")
    duplicates = sorted(
        {value for value in values if values.count(value) > 1}
    )
    if duplicates:
        raise ValueError(
            f"duplicate {label} ids: {', '.join(duplicates)}"
        )


def _load_method_components(
    config: Dict[str, Any],
    config_path: Path,
) -> Dict[str, Any]:
    references = config.pop("method_files", [])
    if not isinstance(references, list):
        raise ValueError("method_files must be a list")

    methods = list(config.get("methods", []))
    loaded_files = []
    for reference in references:
        component_path = (config_path.parent / reference).resolve()
        with component_path.open("r", encoding="utf-8") as handle:
            component = yaml.safe_load(handle)
        if not isinstance(component, dict):
            raise ValueError(
                f"method component must be a mapping: {component_path}"
            )
        if component.get("schema_version") != 1:
            raise ValueError(
                f"method component requires schema_version: 1: "
                f"{component_path}"
            )
        if component.get("kind") != "experiment_methods":
            raise ValueError(
                f"method component requires kind: experiment_methods: "
                f"{component_path}"
            )
        component_methods = component.get("methods")
        if not isinstance(component_methods, list) or not component_methods:
            raise ValueError(
                f"method component has no methods: {component_path}"
            )
        methods.extend(component_methods)
        loaded_files.append(str(component_path))
    config["methods"] = methods
    if loaded_files:
        config["_method_files"] = loaded_files
    return config


def validate_paper_config(config: Dict[str, Any]) -> None:
    if config.get("schema_version") != 2:
        raise ValueError("paper experiments require schema_version: 2")
    required = {"experiment", "output_dir", "datasets", "methods"}
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(
            f"paper config is missing: {', '.join(missing)}"
        )
    experiment = config["experiment"]
    for key in ("id", "target_id", "title", "paper_target", "seed"):
        if key not in experiment:
            raise ValueError(f"experiment.{key} is required")
    target_id = experiment["target_id"]
    if not isinstance(target_id, str) or not TARGET_ID_PATTERN.fullmatch(
        target_id
    ):
        raise ValueError(
            "experiment.target_id must use lowercase letters, digits, "
            "and underscores"
        )
    datasets = config["datasets"]
    methods = config["methods"]
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("datasets must be a non-empty list")
    if not isinstance(methods, list) or not methods:
        raise ValueError("methods must be a non-empty list")
    _ids(datasets, "dataset")
    _ids(methods, "method")

    dataset_ids = {dataset["id"] for dataset in datasets}
    for dataset in datasets:
        cases = dataset.get("cases")
        if not isinstance(cases, list) or not cases:
            raise ValueError(
                f"dataset {dataset['id']} must define cases"
            )
        normalized = [
            {"id": case} if isinstance(case, str) else case
            for case in cases
        ]
        _ids(normalized, f"{dataset['id']} case")

    for method in methods:
        mode = method.get("mode", "native")
        if mode not in {"native", "external", "artifact", "blocked"}:
            raise ValueError(
                f"method {method['id']} has invalid mode {mode!r}"
            )
        if mode == "blocked" and not method.get("blocked_reason"):
            raise ValueError(
                f"blocked method {method['id']} requires blocked_reason"
            )
        filters = set(method.get("datasets", dataset_ids))
        unknown = filters - dataset_ids
        if unknown:
            raise ValueError(
                f"method {method['id']} references unknown datasets: "
                f"{sorted(unknown)}"
            )
        stages = method.get("stages", [])
        if not stages:
            raise ValueError(
                f"method {method['id']} must define at least one stage"
            )
        _ids(stages, f"{method['id']} stage")
        for stage in stages:
            if stage.get("type") not in STAGE_REGISTRY.types:
                raise ValueError(
                    f"stage {method['id']}/{stage['id']} has unsupported "
                    f"type {stage.get('type')!r}"
                )
            if stage["type"] == "command":
                command = stage.get("command")
                if not isinstance(command, list) or not command:
                    raise ValueError(
                        f"command stage {stage['id']} requires argv list"
                    )
            outputs = stage.get("outputs", [])
            if not isinstance(outputs, list):
                raise ValueError(
                    f"stage {method['id']}/{stage['id']} outputs "
                    "must be a list"
                )
            if stage["type"] != "command" and not outputs:
                raise ValueError(
                    f"internal stage {method['id']}/{stage['id']} "
                    "must declare outputs"
                )


def validate_compiled_config(config: Dict[str, Any]) -> None:
    """Render every job to catch bad templates and output collisions."""

    from .model import expand_jobs

    for job in expand_jobs(config):
        outputs = {}
        for raw_stage in job.method["stages"]:
            stage = render_template(raw_stage, job.context())
            for field in ("inputs", "outputs"):
                value = stage.get(field, [])
                if not isinstance(value, list):
                    raise ValueError(
                        f"{job.key}/{stage['id']} {field} must be a list"
                    )
            for output in stage.get("outputs", []):
                output = str(output)
                previous = outputs.get(output)
                if previous is not None:
                    raise ValueError(
                        f"{job.key} stages {previous} and "
                        f"{stage['id']} declare the same output: {output}"
                    )
                outputs[output] = stage["id"]


def load_paper_config(
    path,
    require_environment: bool = False,
) -> Dict[str, Any]:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"paper config must be a mapping: {path}")
    config = _load_method_components(config, path)
    config = _expand_environment(config, require_environment)
    validate_paper_config(config)
    config["_config_path"] = str(path)
    validate_compiled_config(config)
    return config
