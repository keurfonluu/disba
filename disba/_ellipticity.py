from collections import namedtuple

import numpy

from ._base import Base
from ._eigen import EigenFunction

__all__ = [
    "RayleighEllipticity",
    "Ellipticity",
]


RayleighEllipticity = namedtuple("RayleighEllipticity", ("period", "ellipticity", "mode"))


class Ellipticity(Base):
    def __init__(
        self, thickness, velocity_p, velocity_s, density, algorithm="dunkin", dc=0.005,
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
        algorithm : str {'dunkin', 'fast-delta'}, optional, default 'dunkin'
            Algorithm to use for computation of Rayleigh-wave dispersion:
             - 'dunkin': Dunkin's matrix (adapted from surf96),
             - 'fast-delta': fast delta matrix (after Buchen and Ben-Hador, 1996).
        dc : scalar, optional, default 0.005
            Phase velocity increment for root finding.

        """
        super().__init__(thickness, velocity_p, velocity_s, density, algorithm, dc)

    def __call__(self, t, mode=0):
        """
        Compute Rayleigh-wave ellipticity for input period axis.

        Parameters
        ----------
        t : array_like
            Periods (in s).
        mode : int, optional, default 0
            Mode number (0 if fundamental).

        Returns
        -------
        namedtuple
            Rayleigh-wave ellipticity as a namedtuple (period, ellipticity, mode).

        """
        eigf = EigenFunction(
            self._thickness,
            self._velocity_p,
            self._velocity_s,
            self._density,
            self._algorithm,
            self._dc,
        )

        eigs = [eigf(tt, mode, wave="rayleigh") for tt in t]
        ell = [eig.ur[0] / eig.uz[0] for eig in eigs]

        return RayleighEllipticity(t, numpy.array(ell), mode)
