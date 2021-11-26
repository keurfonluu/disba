import numpy as np
import perfplot

from disba import PhaseDispersion
from pysurf96 import surf96


def velocity_model(n):
    thickness = np.full(n, 0.5)
    velocity_p = 1.0 + 0.1 * np.arange(n)
    velocity_s = velocity_p / 1.73
    density = np.full(n, 2.0)

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
    },
    "love": {
        "kernels": [
            lambda n: perf_pysurf96(n, periods, "love"),
            lambda n: perf_disba(n, periods, "love"),
        ],
        "labels": ["pysurf96", "disba"],
        "title": "Love-wave",
    },
}
periods = np.logspace(0.0, 1.0, 60)

for k, v in config.items():
    out = perfplot.bench(
        setup=lambda n: n,
        n_range=np.arange(19) + 2,
        equality_check=lambda a, b: np.allclose(a, b, atol=0.02),
        xlabel="Number of layers",
        **v,
    )
    out.save(f"perf_{k}.svg", transparent=True, bbox_inches="tight")
