from ._common import jitted

import numpy

__all__ = [
    "depthplot",
]


@jitted
def resample(thickness, velocity_p, velocity_s, density, dz):
    """Resample velocity model."""
    mmax = len(thickness)

    sizes = numpy.empty(mmax, dtype=numpy.int32)
    for i in range(mmax):
        sizes[i] = numpy.ceil(thickness[i] / dz) if thickness[i] > dz else 1     

    size = sizes.sum()
    d = numpy.empty(size, dtype=numpy.float64)
    a = numpy.empty(size, dtype=numpy.float64)
    b = numpy.empty(size, dtype=numpy.float64)
    rho = numpy.empty(size, dtype=numpy.float64)

    j = 0
    for i in range(mmax):
        dzi = thickness[i] / sizes[i]
        for _ in range(sizes[i]):
            d[j] = dzi
            a[j] = velocity_p[i]
            b[j] = velocity_s[i]
            rho[j] = density[i]
            j += 1

    return d, a, b, rho


def depthplot(x, z, zmax, ax=None, **kwargs):
    """
    Vertical step plot.

    Parameters
    ----------
    x : array_like
        X coordinates of data points.
    z : array_like
        Z coordinates of data points.
    zmax : scalar
        Depth of last data point.
    ax : matplotlib.pyplot.Axes or None, optional, default None
        Matplotlib axes. If `None`, a new figure and axe is created.

    Returns
    -------
    matplotlib.pyplot.Axes
        Plotted data axes.

    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "depthplot requires matplotlib to be installed."
        )

    n = len(x)
    if len(z) != n:
        raise ValueError()
    if zmax <= z[-1]:
        raise ValueError()

    xin = numpy.empty(2 * n)
    xin[:-1:2] = x
    xin[1::2] = x

    zin = numpy.empty(2 * n)
    zin[0] = z[0]
    zin[1:-2:2] = z[1:]
    zin[2:-1:2] = z[1:]
    zin[-1] = zmax

    if ax is None:
        _, ax = plt.subplots(1, 1)
    ax.plot(xin, zin, **kwargs)

    return ax
