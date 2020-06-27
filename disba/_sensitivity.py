from collections import namedtuple

import numpy

from ._base import BaseSensitivity
from ._dispersion import GroupDispersion, PhaseDispersion

__all__ = [
    "SensitivityKernel",
    "PhaseSensitivity",
    "GroupSensitivity",
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
        dp=0.005,
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
        dp : scalar, optional, default 0.005
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
        if numpy.ndim(t) > 0:
            raise ValueError("Period t must be scalar.")

        period = numpy.array([t])
        pd = PhaseDispersion(
            self._thickness,
            self._velocity_p,
            self._velocity_s,
            self._density,
            self._algorithm,
            self._dc,
        )
        c1, kernel = surfker(pd, period, mode, wave, parameter, self._dp)

        return SensitivityKernel(
            self._thickness.cumsum() - self._thickness[0], kernel, t, c1, mode, wave, "phase", parameter,
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
        if numpy.ndim(t) > 0:
            raise ValueError("Period t must be scalar.")

        period = numpy.array([t])
        gd = GroupDispersion(
            self._thickness,
            self._velocity_p,
            self._velocity_s,
            self._density,
            self._algorithm,
            self._dc,
            self._dt,
        )
        c1, kernel = surfker(gd, period, mode, wave, parameter, self._dp)

        return SensitivityKernel(
            self._thickness.cumsum() - self._thickness[0], kernel, t, c1, mode, wave, "group", parameter,
        )

    @property
    def dt(self):
        """Return frequency increment (%) for calculating group velocity."""
        return self._dt


def surfker(dispersion, period, mode, wave, parameter, dp):
    """Compute sensitivity kernel."""
    # Reference velocity
    c1 = dispersion(period, mode, wave)

    # Initialize kernel
    nl = len(dispersion._thickness)
    kernel = numpy.zeros(nl)

    # Love-waves are not sensitive to compressional wave velocities
    if not (parameter == "velocity_p" and wave == "love"):
        # Ignore top and/or bottom layers depending on inputs
        cond = parameter == "velocity_s" or wave == "love"
        ibeg = int(dispersion._velocity_s[0] <= 0.0 and cond)
        iend = nl - 1 if parameter == "thickness" else nl

        # Loop over layers
        fac = 1.0 + dp
        par = getattr(dispersion, parameter)
        for i in range(ibeg, iend):
            tmp = par[i]
            par[i] /= fac
            c2 = dispersion(period, mode, wave)
            kernel[i] = (c2.velocity - c1.velocity) / (par[i] - tmp)
            par[i] *= fac

    return c1.velocity[0], kernel
