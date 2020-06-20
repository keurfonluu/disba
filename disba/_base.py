from abc import ABC

import numpy


class BaseDispersion(ABC):

    def __init__(self, thickness, velocity_p, velocity_s, density):
        self._thickness = numpy.asarray(thickness)
        self._velocity_p = numpy.asarray(velocity_p)
        self._velocity_s = numpy.asarray(velocity_s)
        self._density = numpy.asarray(density)

    @property
    def thickness(self):
        return self._thickness

    @property
    def velocity_p(self):
        return self._velocity_p

    @property
    def velocity_s(self):
        return self._velocity_s

    @property
    def density(self):
        return self._density
