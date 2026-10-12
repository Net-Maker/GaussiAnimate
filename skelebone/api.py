"""Minimal public API for point-sequence to skelebone extraction."""

from pathlib import Path

from .io import load_point_sequence
from .pipeline import ExtractionConfig, ExtractionResult, SkelebonePipeline
from .types import DynamicPointSequence


def extract_skelebones(
    sequence,
    output_dir,
    *,
    mesh=None,
    config: ExtractionConfig = None,
) -> ExtractionResult:
    """Extract the full skelebone representation from one registered trajectory.

    ``sequence`` may be a :class:`DynamicPointSequence` or the path to the
    public ``sequence.npz`` format. The returned bundle is also stored as
    ``skelebones.npz`` for direct use by animation code.
    """

    if config is None:
        config = ExtractionConfig()
    if not isinstance(sequence, DynamicPointSequence):
        sequence = load_point_sequence(Path(sequence))
    return SkelebonePipeline(config).run(
        sequence,
        output_dir,
        mesh_for_skeleton=mesh,
    )
