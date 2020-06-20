from collections import namedtuple

import numpy

from ._base import BaseDispersion
from ._srfdis import srfdis

__all__ = [
    "DispersionCurve",
    "PhaseDispersion",
    "GroupDispersion",
]


DispersionCurve = namedtuple("DispersionCurve", ("period", "velocity", "mode", "wave", "type"))


iwave = {
    "love": 1,
    "rayleigh": 2,
}


class PhaseDispersion(BaseDispersion):

    def __init__(self, thickness, velocity_p, velocity_s, density):
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
        
        """
        super().__init__(thickness, velocity_p, velocity_s, density)

    def __call__(self, t, mode=0, wave="rayleigh", dc=0.005):
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
        dc : scalar, optional, default 0.005
            Phase velocity increment for searching root.

        Returns
        -------
        namedtuple
            Dispersion curve as a namedtuple (period, velocity, mode, type).
        
        Note
        ----
        This function does not perform any check to reduce overhead in case this function is called multiple times (e.g. inversion).

        """
        wave = iwave[wave]
        c = srfdis(
            t,
            self._thickness,
            self._velocity_p,
            self._velocity_s,
            self._density,
            mode + 1,
            wave,
            dc,
        )

        idx = c > 0.0
        t = t[idx]
        c = c[idx]

        return DispersionCurve(t, c, mode, wave, "phase")


class GroupDispersion(BaseDispersion):

    def __init__(self, thickness, velocity_p, velocity_s, density):
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
        
        """
        super().__init__(thickness, velocity_p, velocity_s, density)

    def __call__(self, t, mode=0, wave="rayleigh", dc=0.005, dt=0.005):
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
        wave = iwave[wave]

        t1 = t / (1.0 + dt)
        c = srfdis(
            t1,
            self._thickness,
            self._velocity_p,
            self._velocity_s,
            self._density,
            mode + 1,
            wave,
            dc,
        )

        idx = c > 0.0
        t = t[idx]
        c = c[idx]

        t1 = t1[idx]
        t2 = t / (1.0 - dt)
        c2 = srfdis(
            t2,
            self._thickness,
            self._velocity_p,
            self._velocity_s,
            self._density,
            mode + 1,
            wave,
            dc,
        )
        t1 = 1.0 / t1
        t2 = 1.0 / t2
        c = (t1 - t2) / (t1 / c - t2 / c2)

        return DispersionCurve(t, c, mode, wave, "group")
