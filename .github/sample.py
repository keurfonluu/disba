import numpy

from disba import EigenFunction, PhaseDispersion, PhaseSensitivity

import dufte
import matplotlib.pyplot as plt

plt.style.use(dufte.style)


# Velocity model and periods
velocity_model = numpy.array(
    [
        [10.0, 7.00, 3.50, 2.00],
        [10.0, 6.80, 3.40, 2.00],
        [10.0, 7.00, 3.50, 2.00],
        [10.0, 7.60, 3.80, 2.00],
        [10.0, 8.40, 4.20, 2.00],
        [10.0, 9.00, 4.50, 2.00],
        [10.0, 9.40, 4.70, 2.00],
        [10.0, 9.60, 4.80, 2.00],
        [10.0, 9.50, 4.75, 2.00],
    ]
)
t = numpy.logspace(0.0, 3.0, 100)


# Dispersion curve
pd = PhaseDispersion(*velocity_model.T)
labels = {
    0: "Fundamental",
    1: "Mode 1",
    2: "Mode 2",
}
for wave in ["rayleigh", "love"]:
    fig = plt.figure()

    for i in range(3):
        cp = pd(t, mode=i, wave=wave)
        plt.semilogx(cp.period, cp.velocity, linewidth=1, label=labels[i])

    plt.title(f"{wave.capitalize()}-wave")
    plt.xlabel("Period [s]")
    plt.ylabel("Phase velocity [km/s]")
    plt.ylim(3.2, 4.8)
    plt.legend(loc=4, frameon=False)

    fig.savefig(f"sample_{wave}.svg", transparent=True, bbox_inches="tight")


# Resample velocity model with respect to depth
thickess = numpy.ones(161) * 0.5
idx = numpy.digitize(thickess.cumsum(), velocity_model[:, 0].cumsum(), right=True)
velocity_model = velocity_model[idx]
velocity_model[:, 0] = thickess


# Eigenfunction
eigf = EigenFunction(*velocity_model.T)
keys = {
    "rayleigh": ["ur", "uz", "tz", "tr"],
    "love": ["uu", "tt"],
}
for wave in ["rayleigh", "love"]:
    fig = plt.figure()

    eig = eigf(20.0, mode=0, wave=wave)
    for key in keys[wave]:
        plt.plot(getattr(eig, key), eig.depth, linewidth=1, label=key.upper())

    plt.title(f"{wave.capitalize()}-wave")
    plt.xlabel("Normalized eigenfunction")
    plt.ylabel("Depth [km]")
    plt.xlim(-2.0, 2.0)
    plt.ylim(0.0, 80.0)
    plt.gca().invert_yaxis()
    plt.legend(loc=4, frameon=False)

    fig.savefig(f"eigen_{wave}.svg", transparent=True, bbox_inches="tight")


# Sensitivity kernel
ps = PhaseSensitivity(*velocity_model.T)
labels = {
    "thickness": "$\\partial c / \\partial d$",
    "velocity_p": "$\\partial c / \\partial \\alpha$",
    "velocity_s": "$\\partial c / \\partial \\beta$",
    "density": "$\\partial c / \\partial \\rho$",
}
for wave in ["rayleigh", "love"]:
    fig = plt.figure()

    for parameter in ["thickness", "velocity_p", "velocity_s", "density"]:
        sk = ps(20.0, mode=0, wave=wave, parameter=parameter)
        plt.plot(sk.kernel, sk.depth, linewidth=1, label=labels[parameter])

    plt.title(f"{wave.capitalize()}-wave")
    plt.xlabel("Sensitivity kernel")
    plt.ylabel("Depth [km]")
    plt.xlim(-0.04, 0.04)
    plt.ylim(0.0, 80.0)
    plt.gca().invert_yaxis()
    plt.legend(loc=4, frameon=False)

    fig.savefig(f"kernel_{wave}.svg", transparent=True, bbox_inches="tight")
