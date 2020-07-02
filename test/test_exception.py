import numpy
import pytest

from disba import PhaseDispersion, DispersionError


def test_exception():
    velocity_model = numpy.array(
        [
            [0.5, 1.0, 0.5, 2.00],
            [0.3, 2.0, 1.0, 2.00],
            [1.0, 1.0, 0.5, 2.00],
        ]
    )
    f = numpy.linspace(0.1, 10.0, 60)
    t = 1.0 / f[::-1]

    with pytest.raises(DispersionError):
        pd = PhaseDispersion(*velocity_model.T, algorithm="dunkin")
        pd(t, mode=0, wave="rayleigh")
