from collections import namedtuple

import numpy

from ._base import BaseSensitivity
from ._common import ifunc, ipar
from ._cps import srfker96, swegn96

__all__ = [
    "SensitivityKernel",
    "PhaseSensitivity",
    "GroupSensitivity",
    "EllipticitySensitivity",
]


SensitivityKernel = namedtuple(
    "SensitivityKernel",
    ("depth", "kernel", "period", "velocity", "mode", "wave", "type", "parameter"),
)


class PhaseSensitivity(BaseSensitivity):
    def __init__(
        self,
        thickness,
        velocity_p,
        velocity_s,
        density,
        algorithm="dunkin",
        dc=0.005,
        dp=0.025,
    ):
        """
        Phase velocity sensitivity kernel class.

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
        dp : scalar, optional, default 0.025
            Parameter increment (%) for numerical partial derivatives.

        """
        super().__init__(thickness, velocity_p, velocity_s, density, algorithm, dc, dp)

    def __call__(self, t, mode=0, wave="rayleigh", parameter="velocity_s"):
        """
        Calculate phase velocity sensitivity kernel for a given period and parameter.

        Parameters
        ----------
        t : scalar
            Period (in s).
        mode : int, optional, default 0
            Mode number (0 if fundamental).
        wave : str {'love', 'rayleigh'}, optional, default 'rayleigh'
            Wave type.
        parameter : str {'thickness', 'velocity_p', 'velocity_s', 'density'}, optional, default 'velocity_s'
            Parameter with respect to which sensitivity kernel is calculated.

        Returns
        -------
        namedtuple
            Sensitivity kernel as a namedtuple (depth, kernel, period, velocity, mode, wave, type, parameter).

        """
        c1, kernel = srfker96(
            t,
            self._thickness,
            self._velocity_p,
            self._velocity_s,
            self._density,
            mode=mode,
            itype=0,
            ifunc=ifunc[self._algorithm][wave],
            ipar=ipar[parameter],
            dc=self._dc,
            dp=self._dp,
        )

        return SensitivityKernel(
            self._thickness.cumsum() - self._thickness[0],
            kernel,
            t,
            c1,
            mode,
            wave,
            "phase",
            parameter,
        )


class GroupSensitivity(BaseSensitivity):
    def __init__(
        self,
        thickness,
        velocity_p,
        velocity_s,
        density,
        algorithm="dunkin",
        dc=0.005,
        dt=0.025,
        dp=0.025,
    ):
        """
        Phase velocity sensitivity kernel class.

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
        dp : scalar, optional, default 0.025
            Parameter increment (%) for numerical partial derivatives.

        """
        if not isinstance(dt, float):
            raise TypeError()

        super().__init__(thickness, velocity_p, velocity_s, density, algorithm, dc, dp)

        self._dt = dt

    def __call__(self, t, mode=0, wave="rayleigh", parameter="velocity_s"):
        """
        Calculate group velocity sensitivity kernel for a given period and parameter.

        Parameters
        ----------
        t : scalar
            Period (in s).
        mode : int, optional, default 0
            Mode number (0 if fundamental).
        wave : str {'love', 'rayleigh'}, optional, default 'rayleigh'
            Wave type.
        parameter : str {'thickness', 'velocity_p', 'velocity_s', 'density'}, optional, default 'velocity_s'
            Parameter with respect to which sensitivity kernel is calculated.

        Returns
        -------
        namedtuple
            Sensitivity kernel as a namedtuple (depth, kernel, period, velocity, mode, wave, type, parameter).

        """
        c1, kernel = srfker96(
            t,
            self._thickness,
            self._velocity_p,
            self._velocity_s,
            self._density,
            mode=mode,
            itype=1,
            ifunc=ifunc[self._algorithm][wave],
            ipar=ipar[parameter],
            dc=self._dc,
            dt=self._dt,
            dp=self._dp,
        )

        return SensitivityKernel(
            self._thickness.cumsum() - self._thickness[0],
            kernel,
            t,
            c1,
            mode,
            wave,
            "group",
            parameter,
        )

    @property
    def dt(self):
        """Return frequency increment (%) for calculating group velocity."""
        return self._dt


class EllipticitySensitivity(BaseSensitivity):
    def __init__(
        self,
        thickness,
        velocity_p,
        velocity_s,
        density,
        algorithm="dunkin",
        dc=0.005,
        dp=0.025,
    ):
        """
        Rayleigh-wave ellipticity sensitivity kernel class.

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
        dp : scalar, optional, default 0.025
            Parameter increment (%) for numerical partial derivatives.

        """
        super().__init__(thickness, velocity_p, velocity_s, density, algorithm, dc, dp)

    def __call__(self, t, mode=0, parameter="velocity_s"):
        """
        Calculate Rayleigh-wave ellipticity sensitivity kernel for a given period and parameter.

        Parameters
        ----------
        t : scalar
            Period (in s).
        mode : int, optional, default 0
            Mode number (0 if fundamental).
        parameter : str {'thickness', 'velocity_p', 'velocity_s', 'density'}, optional, default 'velocity_s'
            Parameter with respect to which sensitivity kernel is calculated.

        Returns
        -------
        namedtuple
            Sensitivity kernel as a namedtuple (depth, kernel, period, velocity, mode, wave, type, parameter).

        """
        # Reference ellipticity
        ell1 = self._ellipticity(t, mode)

        # Initialize kernel
        mmax = len(self._thickness)
        kernel = numpy.empty(mmax)

        # Loop over layers
        fac = 1.0 + self._dp
        par = getattr(self, parameter)
        for i in range(mmax):
            tmp = par[i]
            par[i] /= fac
            ell2 = self._ellipticity(t, mode)
            kernel[i] = (ell2 - ell1) / (par[i] - tmp)
            par[i] *= fac

        return SensitivityKernel(
            self._thickness.cumsum() - self._thickness[0],
            kernel,
            t,
            None,
            mode,
            "rayleigh",
            "ellipticity",
            parameter,
        )

    def _ellipticity(self, t, mode):
        """Compute Rayleigh-wave ellipticity for input period and mode."""
        eig = swegn96(
            t,
            self._thickness,
            self._velocity_p,
            self._velocity_s,
            self._density,
            mode,
            ifunc[self._algorithm]["rayleigh"],
            self._dc,
        )[:, :2]

        return eig[0, 0] / eig[0, 1]
