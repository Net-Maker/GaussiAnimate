"""Configuration for the external, licensed SMPL body model data.

The SMPL model files (``SMPL_FEMALE.pkl`` / ``SMPL_MALE.pkl``) are licensed
data that stays outside this package. Configure their directory explicitly
with :func:`set_smpl_model_dir` before any SMPL model is created; there is
no environment-variable or hard-coded fallback.
"""

from pathlib import Path

_SMPL_MODEL_DIR = None


def set_smpl_model_dir(model_dir) -> None:
    """Configure the directory that holds the licensed SMPL model data.

    Cached SMPL models are cleared when the directory changes, so models
    loaded from a previously configured directory are never reused.
    """
    global _SMPL_MODEL_DIR
    model_dir = Path(model_dir)
    if _SMPL_MODEL_DIR == model_dir:
        return
    _SMPL_MODEL_DIR = model_dir
    # Imported lazily: both modules import this module at their own module
    # level, so a module-level import here would be circular.
    from . import skeleton_transform, smpl_fk

    smpl_fk._SMPL_MODEL_CACHE.clear()
    skeleton_transform._smpl_fk_cache.clear()


def get_smpl_model_dir() -> Path:
    """Return the configured SMPL model directory.

    Raises:
        RuntimeError: if the directory has not been configured yet.
    """
    if _SMPL_MODEL_DIR is None:
        raise RuntimeError(
            "SMPL model directory is not configured. Call "
            "experiments.backends.partmm_research.set_smpl_model_dir(<dir>) "
            "with the directory containing SMPL_FEMALE.pkl / SMPL_MALE.pkl "
            "before running PartMM."
        )
    return _SMPL_MODEL_DIR
