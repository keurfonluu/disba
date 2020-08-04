import os

import matplotlib.pyplot as plt 
import numpy
import pytest

import disba
import helpers

if not os.environ.get("DISPLAY", ""):
    plt.switch_backend("Agg")


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


def test_depthplot(monkeypatch):
    velocity_model = helpers.velocity_model(5)
    x = velocity_model[1]
    z = velocity_model[0].cumsum() - velocity_model[0][0]

    monkeypatch.setattr(plt, "show", lambda: None)
    disba.depthplot(x, z, velocity_model[0].sum())
