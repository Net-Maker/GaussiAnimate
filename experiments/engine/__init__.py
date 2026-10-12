"""Public API for the modular paper experiment engine."""

from .config import (
    load_paper_config,
    render_template,
    validate_compiled_config,
    validate_paper_config,
)
from .model import Job, expand_jobs
from .registry import STAGE_REGISTRY, StageContext, StageRegistry
from .runner import adopt_legacy_stages, run_job, stage_fingerprint

__all__ = [
    "Job",
    "STAGE_REGISTRY",
    "StageContext",
    "StageRegistry",
    "adopt_legacy_stages",
    "expand_jobs",
    "load_paper_config",
    "render_template",
    "run_job",
    "stage_fingerprint",
    "validate_compiled_config",
    "validate_paper_config",
]
