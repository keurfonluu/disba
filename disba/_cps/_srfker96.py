import numpy

from .._common import jitted
from ._surf96 import surf96

__all__ = [
    "srfker96",
]


@jitted
def srfker96(
    t, d, a, b, rho, mode=0, itype=0, ifunc=2, ipar=2, dc=0.005, dt=0.025, dp=0.025
):
    """
    Get phase or group velocity sensitivity kernel.

    Parameters
    ----------
    t : scalar
        Period (in s).
    d : array_like
        Layer thickness (in km).
    a : array_like
        Layer P-wave velocity (in km/s).
    b : array_like
        Layer S-wave velocity (in km/s).
    rho : array_like
        Layer density (in g/cm3).
    mode : int, optional, default 0
        Mode number (0 if fundamental).
    itype : int, optional, default 0
        Velocity type:
         - 0: phase velocity,
         - 1: group velocity.
    ifunc : int, optional, default 2
        Select wave type and algorithm for period equation:
         - 1: Love-wave (Thomson-Haskell method),
         - 2: Rayleigh-wave (Dunkin's matrix),
         - 3: Rayleigh-wave (fast delta matrix).
    ipar : int, optional, default 2
        Select parameter with respect to which sensitivity kernel is calculated:
         - 0: Layer thickness,
         - 1: P-wave velocity,
         - 2: S-wave velocity,
         - 3: Layer density.
    dc : scalar, optional, default 0.005
        Phase velocity increment for root finding.
    dt : scalar, optional, default 0.025
        Frequency increment (%) for calculating group velocity.
    dp : scalar, optional, default 0.025
        Parameter increment (%) for numerical partial derivatives.

    Returns
    -------
    array_like
        Phase or group velocity sensitivity kernel.

    """
    # Reference velocity
    period = numpy.empty(1, dtype=numpy.float64)
    period[0] = t
    c1 = surf96(period, d, a, b, rho, mode, itype, ifunc, dc, dt)

    # Initialize kernel
    mmax = len(d)
    kernel = numpy.zeros(mmax, dtype=numpy.float64)

    # Love-waves are not sensitive to P-wave
    if not (ipar == 1 and ifunc == 1):
        # Copy parameter array
        par = numpy.empty(mmax, dtype=numpy.float64)
        if ipar == 0:
            for i in range(mmax):
                par[i] = d[i]
        elif ipar == 1:
            for i in range(mmax):
                par[i] = a[i]
        elif ipar == 2:
            for i in range(mmax):
                par[i] = b[i]
        else:
            for i in range(mmax):
                par[i] = rho[i]

        # Ignore top and/or bottom layers depending on inputs
        ibeg = 1 if b[0] < 0.01 and (ipar == 2 or ifunc == 0) else 0
        iend = mmax - 1 if ipar == 0 else mmax

        # Loop over layers
        fac = 1.0 + dp
        for i in range(ibeg, iend):
            tmp = par[i]
            par[i] /= fac

            if ipar == 0:
                c2 = surf96(period, par, a, b, rho, mode, itype, ifunc, dc, dt)
            elif ipar == 1:
                c2 = surf96(period, d, par, b, rho, mode, itype, ifunc, dc, dt)
            elif ipar == 2:
                c2 = surf96(period, d, a, par, rho, mode, itype, ifunc, dc, dt)
            else:
                c2 = surf96(period, d, a, b, par, mode, itype, ifunc, dc, dt)

            kernel[i] = (c2[0] - c1[0]) / (par[i] - tmp)
            par[i] *= fac

    return c1[0], kernel
