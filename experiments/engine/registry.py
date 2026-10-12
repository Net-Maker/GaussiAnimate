"""Registry for declarative experiment stage implementations."""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional


@dataclass(frozen=True)
class StageContext:
    """Runtime context passed to one stage implementation."""

    job: Any
    gpu: Optional[str] = None


StageHandler = Callable[[Dict[str, Any], StageContext], None]


class StageRegistry:
    """Map stable YAML stage type names to executable handlers."""

    def __init__(self) -> None:
        self._handlers: Dict[str, StageHandler] = {}

    @property
    def types(self):
        return frozenset(self._handlers)

    def register(self, stage_type: str):
        if not stage_type:
            raise ValueError("stage type must be non-empty")

        def decorator(handler: StageHandler) -> StageHandler:
            if stage_type in self._handlers:
                raise ValueError(f"duplicate stage type: {stage_type}")
            self._handlers[stage_type] = handler
            return handler

        return decorator

    def execute(
        self,
        stage: Dict[str, Any],
        context: StageContext,
    ) -> None:
        stage_type = stage["type"]
        try:
            handler = self._handlers[stage_type]
        except KeyError as exc:
            raise ValueError(
                f"unsupported experiment stage type: {stage_type}"
            ) from exc
        handler(stage, context)

    def handler(self, stage_type: str) -> StageHandler:
        try:
            return self._handlers[stage_type]
        except KeyError as exc:
            raise ValueError(
                f"unsupported experiment stage type: {stage_type}"
            ) from exc


STAGE_REGISTRY = StageRegistry()
