from numba import jit

ifunc = {
    "dunkin": {"love": 1, "rayleigh": 2},
    "fast-delta": {"love": 1, "rayleigh": 3},
}

ipar = {
    "thickness": 0,
    "velocity_p": 1,
    "velocity_s": 2,
    "density": 3,
}


def jitted(*args, **kwargs):
    """Custom :func:`jit` with default options."""
    kwargs.update(
        {
            "nopython": True,
            "nogil": True,
            "fastmath": True,
            # "boundscheck": False,
            "cache": True,
        }
    )
    return jit(*args, **kwargs)
