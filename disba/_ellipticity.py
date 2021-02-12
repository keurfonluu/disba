from collections import namedtuple

import numpy

from ._base import Base
from ._common import ifunc
from ._cps import swegn96
from ._exception import DispersionError

__all__ = [
    "RayleighEllipticity",
    "Ellipticity",
]


RayleighEllipticity = namedtuple(
    "RayleighEllipticity", ("period", "ellipticity", "mode")
)


class Ellipticity(Base):
    def __init__(
        self,
        thickness,
        velocity_p,
        velocity_s,
        density,
        algorithm="fast-delta",
        dc=0.005,
    ):
        """
        Ellipticity class (only Rayleigh-wave).

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
        algorithm : str {'dunkin', 'fast-delta'}, optional, default 'fast-delta'
            Algorithm to use for computation of Rayleigh-wave dispersion:
             - 'dunkin': Dunkin's matrix (adapted from surf96),
             - 'fast-delta': fast delta matrix (after Buchen and Ben-Hador, 1996).
        dc : scalar, optional, default 0.005
            Phase velocity increment for root finding.

        """
        super().__init__(thickness, velocity_p, velocity_s, density, algorithm, dc)

    def __call__(self, t, mode=0):
        """
        Compute Rayleigh-wave ellipticity for input period axis and mode.

        Parameters
        ----------
        t : array_like
            Periods (in s).
        mode : int, optional, default 0
            Mode number (0 if fundamental).

        Returns
        -------
        :class:`disba.RayleighEllipticity`
            Rayleigh-wave ellipticity as a namedtuple (period, ellipticity, mode).

        """
        ell = []
        for i, tt in enumerate(t):
            try:
                eig = swegn96(
                    tt,
                    self._thickness,
                    self._velocity_p,
                    self._velocity_s,
                    self._density,
                    mode,
                    ifunc[self._algorithm]["rayleigh"],
                    self._dc,
                )[:, :2]
                ell.append(eig[0, 0] / eig[0, 1])
            except DispersionError:
                i -= 1
                break

        return RayleighEllipticity(t[: i + 1], numpy.array(ell), mode)
