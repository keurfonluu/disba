import numpy as np

from disba import EigenFunction, PhaseDispersion, PhaseSensitivity, Ellipticity, EllipticitySensitivity

import dufte
import matplotlib.pyplot as plt

plt.style.use(dufte.style)


# Velocity model and periods
velocity_model = np.array(
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
t = np.logspace(0.0, 3.0, 100)


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


# Eigenfunction
eigf = EigenFunction(*velocity_model.T)
eigf.resample(0.5)
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
    plt.ylim(0.0, 90.0)
    plt.gca().invert_yaxis()
    plt.legend(loc=4, frameon=False)

    fig.savefig(f"eigen_{wave}.svg", transparent=True, bbox_inches="tight")


# Sensitivity kernel
ps = PhaseSensitivity(*velocity_model.T)
ps.resample(0.5)
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
        plt.plot(sk.kernel * 1.0e2, sk.depth, linewidth=1, label=labels[parameter])

    plt.title(f"{wave.capitalize()}-wave")
    plt.xlabel("Sensitivity kernel [$\\times 10^{-2}$]")
    plt.ylabel("Depth [km]")
    plt.xlim(-2.0, 2.0)
    plt.ylim(0.0, 90.0)
    plt.gca().invert_yaxis()
    plt.legend(loc=4, frameon=False)

    fig.savefig(f"kernel_{wave}.svg", transparent=True, bbox_inches="tight")


# Ellipticity
ell = Ellipticity(*velocity_model.T)
rel = ell(t, mode=0)

fig = plt.figure()

plt.semilogx(rel.period, rel.ellipticity, linewidth=1)
plt.title("Ellipticity")
plt.xlabel("Period [s]")
plt.ylabel("Ellipticity [H/V]")
plt.ylim(0.55, 0.85)

fig.savefig("sample_ellipticity.svg", transparent=True, bbox_inches="tight")


# Ellipticity sensitivity kernel
es = EllipticitySensitivity(*velocity_model.T)
es.resample(0.5)
labels = {
    "thickness": "$\\partial \\chi / \\partial d$",
    "velocity_p": "$\\partial \\chi / \\partial \\alpha$",
    "velocity_s": "$\\partial \\chi / \\partial \\beta$",
    "density": "$\\partial \\chi / \\partial \\rho$",
}

fig = plt.figure()

for parameter in ["thickness", "velocity_p", "velocity_s", "density"]:
    ek = es(20.0, mode=0, parameter=parameter)
    plt.plot(ek.kernel * 1.0e2, ek.depth, linewidth=1, label=labels[parameter])

    plt.title("Ellipticity sensitivity")
    plt.xlabel("Sensitivity kernel [$\\times 10^{-2}$]")
    plt.ylabel("Depth [km]")
    plt.xlim(-2.0, 2,0)
    plt.ylim(0.0, 90.0)
    plt.gca().invert_yaxis()
    plt.legend(loc=4, frameon=False)

fig.savefig("kernel_ellipticity.svg", transparent=True, bbox_inches="tight")
