"""Load locally licensed SMPL models, including legacy Chumpy pickles."""

import inspect
from pathlib import Path

import numpy as np


def load_smpl_model(model_path, gender="female", **kwargs):
    """Load a trusted model without permanently changing NumPy's API.

    Official older SMPL releases contain Chumpy objects. Chumpy imports
    scalar aliases removed in NumPy 1.24 and getargspec removed in Python 3.11.
    The aliases are needed only while deserializing and converting the model.
    """
    from smplx import SMPL

    aliases = {
        "bool": bool, "int": int, "float": float, "complex": complex,
        "object": object, "unicode": str, "str": str,
    }
    added = [name for name in aliases if name not in np.__dict__]
    add_argspec = "getargspec" not in inspect.__dict__
    try:
        for name in added:
            setattr(np, name, aliases[name])
        if add_argspec:
            inspect.getargspec = inspect.getfullargspec
        return SMPL(str(Path(model_path).expanduser()), gender=gender, **kwargs)
    except ModuleNotFoundError as exc:
        if exc.name == "chumpy":
            raise ImportError(
                "This SMPL file contains Chumpy arrays. Install the vto extra: "
                "python -m pip install -e '.[vto]'"
            ) from exc
        raise
    finally:
        for name in added:
            delattr(np, name)
        if add_argspec:
            delattr(inspect, "getargspec")
