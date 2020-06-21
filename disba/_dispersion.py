from collections import namedtuple

from ._base import BaseDispersion
from ._surf96 import surf96

__all__ = [
    "DispersionCurve",
    "PhaseDispersion",
    "GroupDispersion",
]


DispersionCurve = namedtuple(
    "DispersionCurve", ("period", "velocity", "mode", "wave", "type")
)


ifunc = {
    "dunkin": {"love": 1, "rayleigh": 2},
    "fast-delta": {"love": 1, "rayleigh": 3},
}


class PhaseDispersion(BaseDispersion):
    def __init__(
        self, thickness, velocity_p, velocity_s, density, algorithm="dunkin", dc=0.005
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

    def __call__(self, t, mode=0, wave="rayleigh"):
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
            Dispersion curve as a namedtuple (period, velocity, mode, type).

        Note
        ----
        This function does not perform any check to reduce overhead in case this function is called multiple times (e.g. inversion).

        """
        c = surf96(
            t,
            self._thickness,
            self._velocity_p,
            self._velocity_s,
            self._density,
            mode + 1,
            ifunc[self._algorithm][wave],
            self._dc,
        )

        idx = c > 0.0
        t = t[idx]
        c = c[idx]

        return DispersionCurve(t, c, mode, wave, "phase")


class GroupDispersion(BaseDispersion):
    def __init__(
        self,
        thickness,
        velocity_p,
        velocity_s,
        density,
        algorithm="dunkin",
        dc=0.005,
        dt=0.005,
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
        dt : scalar, optional, default 0.005
            Frequency increment (%) for calculating group velocity.

        """
        super().__init__(thickness, velocity_p, velocity_s, density, algorithm, dc)
        self._dt = dt

    def __call__(self, t, mode=0, wave="rayleigh"):
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
            Dispersion curve as a namedtuple (period, velocity, mode, type).

        Note
        ----
        This function does not perform any check to reduce overhead in case this function is called multiple times (e.g. inversion).

        """
        t1 = t / (1.0 + self._dt)
        c = surf96(
            t1,
            self._thickness,
            self._velocity_p,
            self._velocity_s,
            self._density,
            mode + 1,
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
            mode + 1,
            ifunc[self._algorithm][wave],
            self._dc,
        )
        t1 = 1.0 / t1
        t2 = 1.0 / t2
        c = (t1 - t2) / (t1 / c - t2 / c2)

        return DispersionCurve(t, c, mode, wave, "group")

    @property
    def dt(self):
        """Return frequency increment (%) for calculating group velocity."""
        return self._dt
