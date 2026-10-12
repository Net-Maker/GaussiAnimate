"""Standalone part-aware motion transfer for dynamic 3D assets."""

from .animator import AnimationResult, PartMMConfig, PartMotionMatcher, animate
from .fallback import PartMMFallback, SkinningBaseline
from .io import (
    load_input,
    load_output,
    load_query,
    save_input,
    save_output,
    save_query,
)
from .types import PartMMInput, PartMMOutput, PartMMQuery

__all__ = [
    "AnimationResult",
    "PartMMConfig",
    "PartMMFallback",
    "SkinningBaseline",
    "PartMMInput",
    "PartMMOutput",
    "PartMMQuery",
    "PartMotionMatcher",
    "animate",
    "load_input",
    "load_output",
    "load_query",
    "save_input",
    "save_output",
    "save_query",
]
