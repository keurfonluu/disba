import numpy as np

from ._common import jitted

__all__ = [
    "resample",
    "depthplot",
]


@jitted
def is_sorted(t):
    """Check if array is sorted."""
    for i in range(t.size - 1):
        if t[i + 1] < t[i]:
            return False

    return True


def resample(thickness, parameters, dz):
    """
    Resample parameters.

    Parameters
    ----------
    thickness : array_like
        Layer thickness (in km).
    parameters : array_like
        Parameters to resample.
    dz : scalar
        Maximum layer thickness (in km).

    Returns
    -------
    array_like
        Resampled thickness.
    array_like
        Resampled parameters.

    """
    thickness = np.asarray(thickness)
    sizes = np.where(thickness > dz, np.ceil(thickness / dz), 1.0,).astype(int)

    size = sizes.sum()
    d = np.empty(size, dtype=np.float64)
    par = (
        np.empty(size, dtype=np.float64)
        if np.ndim(parameters) == 1
        else np.empty((size, np.shape(parameters)[1]), dtype=np.float64)
    )

    _resample(thickness, parameters, sizes, d, par)

    return d, par


@jitted
def _resample(thickness, parameters, sizes, d, par):
    """Compile loop in :func:resample."""
    mmax = len(thickness)

    j = 0
    for i in range(mmax):
        dzi = thickness[i] / sizes[i]
        for _ in range(sizes[i]):
            d[j] = dzi
            par[j] = parameters[i]
            j += 1


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
    z = np.cumsum(thickness)
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
        tmp = np.array(thickness)
        tmp[-1] = tmp[:-1].min()
        zmax = tmp.sum()

    # Build layered model
    xin = np.empty(2 * n)
    xin[1::2] = x
    xin[2::2] = x[1:]
    xin[0] = xin[1]

    zin = np.zeros_like(xin)
    zin[1:-1:2] = z[:-1]
    zin[2::2] = z[:-1]
    zin[-1] = max(z[-1], zmax)

    # Plot
    plot = getattr(plt if ax is None else ax, "plot")
    plot(xin, zin, **_plot_args)

    ax = ax if ax is not None else plt.gca()
    ax.set_ylim(zmax, zin.min())
