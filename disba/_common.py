from numba import jit


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
