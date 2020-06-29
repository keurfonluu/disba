import numpy

import disba
import helpers


def test_ellipticity():
    velocity_model = helpers.velocity_model(5)
    t = numpy.logspace(0.0, 1.0, 20)

    ell = disba.Ellipticity(*velocity_model)
    rel = ell(t)

    assert numpy.allclose(14.038, rel.ellipticity.sum(), atol=0.001)
