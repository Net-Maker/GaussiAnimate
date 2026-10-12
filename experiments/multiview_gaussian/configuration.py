"""Configuration expansion shared by multiview release tools."""

import os
import re
from typing import Any


ENV_PATTERN = re.compile(
    r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}"
)


def expand_environment(value: Any) -> Any:
    """Recursively expand ``${NAME}`` and ``${NAME:-default}`` values."""

    if isinstance(value, dict):
        return {
            key: expand_environment(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [expand_environment(item) for item in value]
    if not isinstance(value, str):
        return value

    def replace(match):
        name, default = match.group(1), match.group(2)
        if name in os.environ:
            return os.environ[name]
        if default is not None:
            return default
        raise ValueError(
            f"required environment variable is unset: {name}"
        )

    return os.path.expanduser(ENV_PATTERN.sub(replace, value))
