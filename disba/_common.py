from collections import namedtuple

from numba import jit

__all__ = [
    "DispersionCurve",
]


DispersionCurve = namedtuple("DispersionCurve", ("period", "velocity", "mode", "type"),)


def jitted(*args, **kwargs):
    """Custom :func:`jit` with default options."""
    kwargs.update(
        {
            "nopython": True,
            "nogil": True,
            "fastmath": True,
            "boundscheck": False,
            "cache": True,
        }
    )
    return jit(*args, **kwargs)
