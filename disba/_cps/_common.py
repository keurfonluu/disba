import numpy

from .._common import jitted


@jitted
def normc(ee, nmat):
    """Normalize Haskell or Dunkin vectors."""
    t1 = 0.0
    for i in range(nmat):
        t1 = max(t1, numpy.abs(ee[i]))

    if t1 < 1.0e-40:
        t1 = 1.0

    for i in range(nmat):
        ee[i] /= t1

    ex = numpy.log(t1)
    return ee, ex
