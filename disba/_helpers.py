from ._common import jitted

import numpy


@jitted
def resample(thickness, velocity_p, velocity_s, density, dz):
    """Resample velocity model."""
    mmax = len(thickness)

    sizes = numpy.empty(mmax, dtype=numpy.int32)
    for i in range(mmax):
        sizes[i] = numpy.ceil(thickness[i] / dz) if thickness[i] > dz else 1     

    size = sizes.sum()
    d = numpy.empty(size, dtype=numpy.float64)
    a = numpy.empty(size, dtype=numpy.float64)
    b = numpy.empty(size, dtype=numpy.float64)
    rho = numpy.empty(size, dtype=numpy.float64)

    j = 0
    for i in range(mmax):
        dzi = thickness[i] / sizes[i]
        for _ in range(sizes[i]):
            d[j] = dzi
            a[j] = velocity_p[i]
            b[j] = velocity_s[i]
            rho[j] = density[i]
            j += 1

    return d, a, b, rho
