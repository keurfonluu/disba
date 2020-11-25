from collections import namedtuple

import numpy

from ._base import Base
from ._common import ifunc
from ._cps import swegn96

__all__ = [
    "LoveEigen",
    "RayleighEigen",
    "EigenFunction",
]


LoveEigen = namedtuple("LoveEigen", ("depth", "uu", "tt", "period", "mode"))

RayleighEigen = namedtuple(
    "RayleighEigen", ("depth", "ur", "uz", "tz", "tr", "period", "mode")
)


class EigenFunction(Base):
    def __init__(
        self,
        thickness,
        velocity_p,
        velocity_s,
        density,
        algorithm="dunkin",
        dc=0.005,
    ):
        """
        Eigenfunction class.

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
        Compute eigenfunctions for a given period and mode.

        Parameters
        ----------
        t : scalar
            Period (in s).
        mode : int, optional, default 0
            Mode number (0 if fundamental).
        wave : str {'love', 'rayleigh'}, optional, default 'rayleigh'
            Wave type.

        Returns
        -------
        :class:`disba.RayleighEigen` or :class:`disba.LoveEigen`
            Eigenfunction as a namedtuple:

             - If ``wave == 'love'``, (depth, uu, tt, period, mode),
             - If ``wave == 'rayleigh'``, (depth, ur, uz, tz, tr, period, mode).

        """
        if numpy.ndim(t) > 0:
            raise ValueError("Period t must be scalar.")

        egn = swegn96(
            t,
            self._thickness,
            self._velocity_p,
            self._velocity_s,
            self._density,
            mode,
            ifunc[self._algorithm][wave],
            self._dc,
        )

        depth = self._thickness.cumsum() - self._thickness[0]
        if wave == "love":
            uu, tt = egn.T
            wegn = LoveEigen(depth, uu, tt, t, mode)
        elif wave == "rayleigh":
            ur, uz, tz, tr = egn.T
            wegn = RayleighEigen(depth, ur, uz, tz, tr, t, mode)

        return wegn
