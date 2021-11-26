import helpers
import numpy as np
import pytest

import disba


@pytest.mark.parametrize(
    "mode, wave, algorithm, cref",
    [
        (0, "rayleigh", "dunkin", 11.788),
        (0, "rayleigh", "fast-delta", 11.782),
        (1, "rayleigh", "dunkin", 8.032),
        (1, "rayleigh", "fast-delta", 8.819),
        (0, "love", "dunkin", 12.900),
        (1, "love", "dunkin", 7.295),
    ],
)
def test_phase(mode, wave, algorithm, cref):
    velocity_model = helpers.velocity_model(5)
    t = np.logspace(0.0, 1.0, 20)

    pd = disba.PhaseDispersion(*velocity_model, algorithm=algorithm)
    cp = pd(t, mode, wave)

    assert np.allclose(cref, cp.velocity.sum(), atol=0.001)


@pytest.mark.parametrize(
    "mode, wave, algorithm, cref",
    [
        (0, "rayleigh", "dunkin", 10.667),
        (0, "rayleigh", "fast-delta", 10.705),
        (1, "rayleigh", "dunkin", 6.931),
        (1, "rayleigh", "fast-delta", 6.914),
        (0, "love", "dunkin", 11.766),
        (1, "love", "dunkin", 6.255),
    ],
)
def test_group(mode, wave, algorithm, cref):
    velocity_model = helpers.velocity_model(5)
    t = np.logspace(0.0, 1.0, 20)

    gd = disba.GroupDispersion(*velocity_model, algorithm=algorithm)
    cg = gd(t, mode, wave)

    assert np.allclose(cref, cg.velocity.sum(), atol=0.001)


@pytest.mark.parametrize(
    "wave, algorithm, cref",
    [
        ("rayleigh", "dunkin", 11.596),
        ("rayleigh", "fast-delta", 11.581),
        ("love", "dunkin", 14.108),
    ],
)
def test_water_layer(wave, algorithm, cref):
    velocity_model = helpers.velocity_model(5, water_layer=True)
    t = np.logspace(0.0, 1.0, 20)

    pd = disba.PhaseDispersion(*velocity_model, algorithm=algorithm)
    cp = pd(t, wave=wave)

    assert np.allclose(cref, cp.velocity.sum(), atol=0.001)
