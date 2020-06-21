import numpy

from disba import PhaseDispersion

import dufte
import matplotlib.pyplot as plt

plt.style.use(dufte.style)


velocity_model = numpy.array([
    [0.5, 1.0, 0.5, 1.8],
    [0.3, 2.0, 1.0, 1.8],
    [10.0, 1.0, 0.5, 1.8],
])
pd = PhaseDispersion(*velocity_model.T, algorithm="fast-delta", dc=0.001)
f = numpy.linspace(0.1, 10.0, 100)
t = 1.0 / f[::-1]

for wave in ["rayleigh", "love"]:
    fig = plt.figure()

    for i in range(20):
        cp = pd(t, mode=i, wave=wave)
        plt.plot(1.0 / cp.period, cp.velocity, color="#1f77b4", linewidth=1)

    plt.title(f"{wave.capitalize()}-wave")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Phase velocity [km/s]")
    plt.xlim(0.0, 10.0)
    plt.ylim(0.45, 1.0)

    fig.savefig(f"sample_{wave}.svg", transparent=True, bbox_inches="tight")
