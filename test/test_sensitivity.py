import helpers
import numpy
import pytest

import disba


@pytest.mark.parametrize(
    "wave, parameter, kref, atol",
    [
        ("rayleigh", "thickness", -0.130, 1.0e-3),
        ("rayleigh", "velocity_p", 0.173, 1.0e-3),
        ("rayleigh", "velocity_s", 0.722, 1.0e-3),
        ("rayleigh", "density", 0.000245, 1.0e-6),
        ("love", "thickness", -0.197, 1.0e-3),
        ("love", "velocity_p", 0.0, 1.0e-3),
        ("love", "velocity_s", 1.194, 1.0e-3),
        ("love", "density", 0.000368, 1.0e-6),
    ],
)
def test_phase(wave, parameter, kref, atol):
    velocity_model = helpers.velocity_model(5)

    ps = disba.PhaseSensitivity(*velocity_model, dp=0.005)
    kp = ps(10.0, 0, wave, parameter)

    assert numpy.allclose(kref, kp.kernel.sum(), atol=atol)


@pytest.mark.parametrize(
    "wave, parameter, kref, atol",
    [
        ("rayleigh", "thickness", -0.252, 1.0e-3),
        ("rayleigh", "velocity_p", 0.254, 1.0e-3),
        ("rayleigh", "velocity_s", 0.804, 1.0e-3),
        ("rayleigh", "density", 0.0207, 1.0e-4),
        ("love", "thickness", -0.195, 1.0e-3),
        ("love", "velocity_p", 0.0, 1.0e-3),
        ("love", "velocity_s", 1.332, 1.0e-3),
        ("love", "density", 0.0479, 1.0e-4),
    ],
)
def test_group(wave, parameter, kref, atol):
    velocity_model = helpers.velocity_model(5)

    gs = disba.GroupSensitivity(*velocity_model, dt=0.005, dp=0.005)
    kg = gs(10.0, 0, wave, parameter)

    assert numpy.allclose(kref, kg.kernel.sum(), atol=atol)


@pytest.mark.parametrize(
    "parameter, kref, atol",
    [
        ("thickness", -0.293, 1.0e-3),
        ("velocity_p", -0.450, 1.0e-3),
        ("velocity_s", 0.731, 1.0e-3),
        ("density", -0.000230, 1.0e-6),
    ],
)
def test_ellipticity(parameter, kref, atol):
    velocity_model = helpers.velocity_model(5)

    es = disba.EllipticitySensitivity(*velocity_model, dp=0.005)
    ke = es(10.0, 0, parameter)

    assert numpy.allclose(kref, ke.kernel.sum(), atol=atol)
