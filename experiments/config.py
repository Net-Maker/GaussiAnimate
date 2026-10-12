"""Validated experiment config loading with explicit environment expansion."""

import os
import re
from pathlib import Path
from typing import Any, Dict

import yaml

from partmm import PartMMConfig
from skelebone.pipeline import ExtractionConfig, SkelebonePipeline


ENV_PATTERN = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")


def _expand(value: Any, require_environment: bool) -> Any:
    if isinstance(value, dict):
        return {key: _expand(item, require_environment) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item, require_environment) for item in value]
    if isinstance(value, str):
        match = ENV_PATTERN.match(value)
        if match:
            variable = match.group(1)
            if variable in os.environ:
                return os.environ[variable]
            if require_environment:
                raise ValueError(f"required environment variable is unset: {variable}")
    return value


def validate_config(config: Dict[str, Any]) -> None:
    required = ["schema_version", "experiment", "dataset", "output_dir", "skelebone", "partmm", "evaluation"]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"missing config sections: {', '.join(missing)}")
    if config["schema_version"] != 1:
        raise ValueError("unsupported experiment schema_version")
    PartMMConfig(**config["partmm"])
    SkelebonePipeline(ExtractionConfig(**resolve_skelebone_config(config))).validate()


def resolve_skelebone_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Apply a dataset-variant override without mutating the loaded config."""

    resolved = dict(config["skelebone"])
    dataset = config.get("dataset", {})
    variant = dataset.get("variant")
    override = dataset.get("overrides", {}).get(variant, {}) if variant else {}
    resolved.update(override.get("skelebone", {}))
    return resolved


def load_config(path, require_environment: bool = False) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config = _expand(config, require_environment)
    validate_config(config)
    return config
