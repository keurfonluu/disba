from collections import namedtuple
import numpy as np

from ._base import BaseDispersion
from ._common import ifunc
from ._surf96 import surf96

__all__ = [
    "DispersionCurve",
    "PhaseDispersion",
    "GroupDispersion",
]


DispersionCurve = namedtuple(
    "DispersionCurve", ("x", "velocity", "mode", "wave", "type", "x_axis_type")
)

_XAXIS = ["period", "frequency"]

class PhaseDispersion(BaseDispersion):
    def __init__(
        self, thickness, velocity_p, velocity_s, density, algorithm="dunkin", dc=0.005,
    ):
        """
        Phase velocity dispersion class.

        Parameters
        ----------
        thickness : array_like
            Layer thickness (in km).
        velocity_p : array_like
            Layer P-wave velocity (in km/s).
        velocity_s : array_like
            Layer S-wave velocity (in km/s).
        density : array_like
            Layer density (in g/cm3).
        algorithm : str {'dunkin', 'fast-delta'}, optional, default 'dunkin'
            Algorithm to use for computation of Rayleigh-wave dispersion:
             - 'dunkin': Dunkin's matrix (adapted from surf96),
             - 'fast-delta': fast delta matrix (after Buchen and Ben-Hador, 1996).
        dc : scalar, optional, default 0.005
            Phase velocity increment for root finding.

        """
        super().__init__(thickness, velocity_p, velocity_s, density, algorithm, dc)

    def __call__(self, t, mode=0, wave="rayleigh", x_axis="period"):
        """
        Calculate phase velocities for input period axis.

        Parameters
        ----------
        t : array_like
            Periods (in s).
        mode : int, optional, default 0
            Mode number (0 if fundamental).
        wave : str {'love', 'rayleigh'}, optional, default 'rayleigh'
            Wave type.

        Returns
        -------
        namedtuple
            Dispersion curve as a namedtuple (period, velocity, mode, wave, type).

        Note
        ----
        This function does not perform any check to reduce overhead in case this function is called multiple times (e.g. inversion).

        """

        if x_axis not in _XAXIS:
            raise ValueError("Incorrect x-axis specified. Please choose either 'frequency' or 'period' as x-axis.")
        elif x_axis == "frequency":
            #Makes sure frequency is sorted and convert to sorted periods
            t = np.sort(t)
            t = 1 / t[::-1]

        c = surf96(
            t,
            self._thickness,
            self._velocity_p,
            self._velocity_s,
            self._density,
            mode,
            ifunc[self._algorithm][wave],
            self._dc,
        )

        idx = c > 0.0
        t = t[idx]
        c = c[idx]

        if x_axis == "frequency":
            t = 1 / t[::-1]

        return DispersionCurve(t, c, mode, wave, "phase", x_axis_type=x_axis)


class GroupDispersion(BaseDispersion):
    def __init__(
        self,
        thickness,
        velocity_p,
        velocity_s,
        density,
        algorithm="dunkin",
        dc=0.005,
        dt=0.025,
    ):
        """
        Group velocity dispersion class.

        Parameters
        ----------
        thickness : array_like
            Layer thickness (in km).
        velocity_p : array_like
            Layer P-wave velocity (in km/s).
        velocity_s : array_like
            Layer S-wave velocity (in km/s).
        density : array_like
            Layer density (in g/cm3).
        algorithm : str {'dunkin', 'fast-delta'}, optional, default 'dunkin'
            Algorithm to use for computation of Rayleigh-wave dispersion:
             - 'dunkin': Dunkin's matrix (adapted from surf96),
             - 'fast-delta': fast delta matrix (after Buchen and Ben-Hador, 1996).
        dc : scalar, optional, default 0.005
            Phase velocity increment for root finding.
        dt : scalar, optional, default 0.025
            Frequency increment (%) for calculating group velocity.

        """
        super().__init__(thickness, velocity_p, velocity_s, density, algorithm, dc)

        self._dt = dt

    def __call__(self, t, mode=0, wave="rayleigh", x_axis="period"):
        """
        Calculate group velocities for input period axis.

        Parameters
        ----------
        t : array_like
            Periods (in s).
        mode : int, optional, default 0
            Mode number (0 if fundamental).
        wave : str {'love', 'rayleigh'}, optional, default 'rayleigh'
            Wave type.

        Returns
        -------
        namedtuple
            Dispersion curve as a namedtuple (period, velocity, mode, wave, type).

        Note
        ----
        This function does not perform any check to reduce overhead in case this function is called multiple times (e.g. inversion).

        """
        if x_axis not in _XAXIS:
            raise ValueError("Incorrect x-axis specified. Please choose either 'frequency' or 'period' as x-axis.")
        elif x_axis == "frequency":
            #Makes sure frequency is sorted and convert to sorted periods
            t = np.sort(t)
            t = 1 / t[::-1]

        t1 = t / (1.0 + self._dt)
        c = surf96(
            t1,
            self._thickness,
            self._velocity_p,
            self._velocity_s,
            self._density,
            mode,
            ifunc[self._algorithm][wave],
            self._dc,
        )

        idx = c > 0.0
        t = t[idx]
        c = c[idx]

        t1 = t1[idx]
        t2 = t / (1.0 - self._dt)
        c2 = surf96(
            t2,
            self._thickness,
            self._velocity_p,
            self._velocity_s,
            self._density,
            mode,
            ifunc[self._algorithm][wave],
            self._dc,
        )

        idx = c2 > 0.0
        t = t[idx]
        t1 = 1.0 / t1[idx]
        t2 = 1.0 / t2[idx]
        c = (t1 - t2) / (t1 / c[idx] - t2 / c2[idx])

        if x_axis == "frequency":
            t = 1 / t[::-1]

        return DispersionCurve(t, c, mode, wave, "group", x_axis)

    @property
    def dt(self):
        """Return frequency increment (%) for calculating group velocity."""
        return self._dt
