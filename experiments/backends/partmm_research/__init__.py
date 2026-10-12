"""PartMM research runtime: packaged copy of the author's PartMM implementation.

This backend vendors the historical PartMM research code so the
experiment drivers import it as a normal packaged module
(``experiments.backends.partmm_research``) instead of relying on an external
implementation checkout on ``sys.path``.

Provenance and license status (see THIRD_PARTY_NOTICES.md for details):

- The ``core``, ``math``, ``patch``, ``pyramid``, ``reconstruction``,
  ``skeletal``, and ``utils`` subpackages are the author's own research code,
  taken from the author's uncommitted local research snapshot (the
  ``smpl_to_ssdr`` package). Unused io/visualization re-exports were
  dropped; computations are unchanged. No license file accompanies that
  snapshot; it is redistributable only with the author's permission.
- ``_thirdparty_motion.py`` and ``_thirdparty_nearest_neighbor.py`` come from
  the Motion2Motion code release (https://github.com/LinghaoChan/Motion2Motion_codes),
  which carries no license; redistribution status is unresolved. The nearest
  neighbor module additionally contains code borrowed from the Apache-2.0
  licensed Efficient-GPNN repository, per its original in-file notice.

The SMPL body model data used at runtime remains an external, licensed
dependency; its directory must be configured explicitly via
:func:`set_smpl_model_dir`.
"""

from .core.motion_data import SMPLMotionData, SSDRMotionData
from .core.part_matcher import PartBasedSMPLToSSDRMatcher
from .skeletal import set_smpl_model_dir
from .reconstruction import (
    reconstruct_vertices,
    result_to_lbs_transforms,
)

__all__ = [
    "SMPLMotionData",
    "SSDRMotionData",
    "PartBasedSMPLToSSDRMatcher",
    "reconstruct_vertices",
    "result_to_lbs_transforms",
    "set_smpl_model_dir",
]
