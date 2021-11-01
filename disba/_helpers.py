import numpy

from ._common import jitted

__all__ = [
    "depthplot",
]


@jitted
def is_sorted(t):
    """Check if array is sorted."""
    for i in range(t.size - 1):
        if t[i + 1] < t[i]:
            return False

    return True


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


def depthplot(thickness, parameter, zmax=None, plot_args=None, ax=None):
    """
    Plot parameter against depth.

    Parameters
    ----------
    thickness : array_like
        Layer's thickness.
    parameter : array_like
        Parameter to plot against depth.
    zmax : scalar or None, optional, default None
        Depth of last data point.
    plot_args : dict or None, optional, default None
        Plot arguments passed to :func:`matplotlib.pyplot.plot`.
    ax : matplotlib.pyplot.Axes or None, optional, default None
        Matplotlib axes. If `None`, use current active plot.

    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("depthplot requires matplotlib to be installed")

    x = parameter
    z = numpy.cumsum(thickness)
    n = z.size

    if len(parameter) != n:
        raise ValueError()

    # Plot arguments
    plot_args = plot_args if plot_args is not None else {}
    _plot_args = {
        "color": "black",
        "linewidth": 2,
    }
    _plot_args.update(plot_args)

    # Determine zmax
    if zmax is None:
        tmp = numpy.array(thickness)
        tmp[-1] = tmp[:-1].min()
        zmax = tmp.sum()

    # Build layered model
    xin = numpy.empty(2 * n)
    xin[1::2] = x
    xin[2::2] = x[1:]
    xin[0] = xin[1]

    zin = numpy.zeros_like(xin)
    zin[1:-1:2] = z[:-1]
    zin[2::2] = z[:-1]
    zin[-1] = max(z[-1], zmax)

    # Plot
    plot = getattr(plt if ax is None else ax, "plot")
    plot(xin, zin, **_plot_args)

    ax = ax if ax is not None else plt.gca()
    ax.set_ylim(zmax, zin.min())
