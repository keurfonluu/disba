import numpy as np


def velocity_model(n, water_layer=False):
    thickness = np.full(n, 0.5)
    velocity_p = 1.0 + 0.1 * np.arange(n)
    velocity_s = velocity_p / 1.73
    density = np.full(n, 2.0)

    if water_layer:
        velocity_s[0] = 0.0
        density[0] = 1.0

    return thickness, velocity_p, velocity_s, density
