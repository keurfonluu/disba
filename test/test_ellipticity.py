import numpy
import pytest

import disba
import helpers


@pytest.mark.parametrize("mode, eref", [(0, 14.038), (1, 3.290)])
def test_ellipticity(mode, eref):
    velocity_model = helpers.velocity_model(5)
    t = numpy.logspace(0.0, 1.0, 20)

    ell = disba.Ellipticity(*velocity_model)
    rel = ell(t, mode=mode)

    assert numpy.allclose(eref, rel.ellipticity.sum(), atol=0.001)
