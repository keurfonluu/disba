import numpy
import perfplot

from disba import PhaseDispersion
from pysurf96 import surf96


def velocity_model(n):
    thickness = numpy.full(n, 0.5)
    velocity_p = 1.0 + 0.1 * numpy.arange(n)
    velocity_s = velocity_p / 1.74
    density = numpy.full(n, 2.0)

    return thickness, velocity_p, velocity_s, density


def perf_disba(n, periods, wave, algorithm="dunkin"):
    pd = PhaseDispersion(*velocity_model(n), algorithm=algorithm)
    return pd(periods, wave=wave).velocity


def perf_pysurf96(n, periods, wave):
    return surf96(*velocity_model(n), periods, wave, 1, "phase")


config = {
    "rayleigh": {
        "kernels": [
            lambda n: perf_pysurf96(n, periods, "rayleigh"),
            lambda n: perf_disba(n, periods, "rayleigh"),
            lambda n: perf_disba(n, periods, "rayleigh", "fast-delta"),
        ],
        "labels": ["pysurf96", "disba-dunkin", "disba-fast-delta"],
        "title": "Rayleigh-wave",
        "filename": "perf_rayleigh.png",
    },
    "love": {
        "kernels": [
            lambda n: perf_pysurf96(n, periods, "love"),
            lambda n: perf_disba(n, periods, "love"),
        ],
        "labels": ["pysurf96", "disba"],
        "title": "Love-wave",
        "filename": "perf_love.png",
    },
}
periods = numpy.logspace(0.0, 1.0, 60)

for wave in config.values():
    out = perfplot.bench(
        setup=lambda n: n,
        n_range=numpy.arange(19) + 2,
        kernels=wave["kernels"],
        equality_check=lambda a, b: numpy.allclose(a, b, atol=0.02),
        title=wave["title"],
        xlabel="Number of layers",
        labels=wave["labels"],
    )
    out.save(wave["filename"], transparent=True, bbox_inches="tight")
