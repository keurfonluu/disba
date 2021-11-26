import helpers
import numpy as np
import pytest

import disba


@pytest.mark.parametrize("mode, eref", [(0, 14.038), (1, 3.290)])
def test_ellipticity(mode, eref):
    velocity_model = helpers.velocity_model(5)
    t = np.logspace(0.0, 1.0, 20)

    ell = disba.Ellipticity(*velocity_model, algorithm="dunkin")
    rel = ell(t, mode=mode)

    assert np.allclose(eref, rel.ellipticity.sum(), atol=0.001)
