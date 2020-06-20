import numpy

from ._common import DispersionCurve
from ._srfdis import srfdis

__all__ = [
    "ThomsonHaskell",
]


class ThomsonHaskell:
    def __init__(self, velocity_model):
        """
        Thomson-Haskell propagator.

        Parameters
        ----------
        velocity_model : array_like (num_layers, 4)
            Velocity model where each row corresponds to a layer:
             - P-wave velocity (km/s)
             - S-wave velocity (km/s)
             - Density (g/cm3)
             - Thickness (km)

        """
        self._velocity_model = numpy.asarray(velocity_model, dtype="float64")

    def __call__(self, t, mode=0, velocity_type="phase", dc=0.005, dt=0.005):
        """
        Calculate phase or group velocities for input period axis.

        Parameters
        ----------
        t : array_like
            Periods (s).
        mode : int, optional, default 0
            Mode number (0 if fundamental).
        velocity_type : str {'phase', 'group'}, optional, default 'phase'
            Velocity type.
        dc : scalar, optional, default 0.005
            Phase velocity increment for searching root.
        dt : scalar, optional, default 0.005
            Frequency increment (%) for calculating group velocity.

        Returns
        -------
        namedtuple
            Dispersion curve as a namedtuple (period, velocity, mode, type).

        Note
        ----
        This function does not perform any check to reduce overhead in case this function is called multiple times (e.g. inversion).

        """
        if velocity_type == "phase":
            t1 = numpy.asarray(t, dtype="float64")
        elif velocity_type == "group":
            t1 = t / (1.0 + dt)
            t1[:] = numpy.asarray(t1, dtype="float64")

        alpha, beta, rho, d = self._velocity_model.T
        mode = numpy.int(mode)
        dc = numpy.float(dc)
        c = srfdis(t1, d, alpha, beta, rho, mode + 1, dc)

        idx = c > 0.0
        t = t[idx]
        c = c[idx]

        if velocity_type == "group":
            t1 = t1[idx]
            t2 = t / (1.0 - dt)
            t2[:] = numpy.asarray(t2, dtype="float64")
            c2 = srfdis(t2, d, alpha, beta, rho, mode + 1, dc)
            t1[:] = 1.0 / t1
            t2[:] = 1.0 / t2
            c = (t1 - t2) / (t1 / c - t2 / c2)

        return DispersionCurve(t, c, mode, velocity_type)
