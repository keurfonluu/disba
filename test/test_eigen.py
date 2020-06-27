import numpy
import pytest

import disba
import helpers


@pytest.mark.parametrize(
    "wave, water_layer, eref",
    [
        (
            "rayleigh",
            False,
            {
                "ur": 1.352,
                "uz": 4.702,
                "tz": -2.372,
                "tr": 2.038,
            },
        ),
        (
            "rayleigh",
            True,
            {
                "ur": 1.155,
                "uz": 4.965,
                "tz": -2.260,
                "tr": 1.638,
            },
        ),
        (
            "love",
            False,
            {
                "uu": 2.421,
                "tt": -1.857,
            },
        ),
        (
            "love",
            True,
            {
                "uu": 1.938,
                "tt": -2.507,
            },
        ),
    ],
)
def test_eigen(wave, water_layer, eref):
    velocity_model = helpers.velocity_model(5, water_layer=water_layer)

    eigf = disba.EigenFunction(*velocity_model)
    eig = eigf(10.0, 0, wave)

    for k, v in eref.items():
        assert numpy.allclose(v, getattr(eig, k).sum(), atol=1.0e-3)
