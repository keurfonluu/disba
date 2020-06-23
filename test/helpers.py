import numpy


def velocity_model(n):
    thickness = numpy.full(n, 0.5)
    velocity_p = 1.0 + 0.1 * numpy.arange(n)
    velocity_s = velocity_p / 1.73
    density = numpy.full(n, 2.0)

    return thickness, velocity_p, velocity_s, density
