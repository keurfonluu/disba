import numpy
import pytest

import disba
import helpers


@pytest.mark.parametrize(
    "mode, wave, algorithm",
    [
        (0, "rayleigh", "dunkin"),
        (0, "rayleigh", "fast-delta"),
        (1, "rayleigh", "dunkin"),
        (1, "rayleigh", "fast-delta"),
        (0, "love", "dunkin"),
        (1, "love", "dunkin"),
    ],
)
def test_resample(mode, wave, algorithm):
    velocity_model = helpers.velocity_model(5)
    t = numpy.logspace(0.0, 1.0, 20)

    pd = disba.PhaseDispersion(*velocity_model, algorithm=algorithm)
    cref = pd(t, mode, wave)

    pd.resample(0.1)
    cp = pd(t, mode, wave)
    
    assert numpy.allclose(cref.velocity.sum(), cp.velocity.sum(), atol=0.1)
