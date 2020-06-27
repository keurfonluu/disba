# disba

[![License](https://img.shields.io/github/license/keurfonluu/disba)](https://github.com/keurfonluu/disba/blob/master/LICENSE)
[![Stars](https://img.shields.io/github/stars/keurfonluu/disba?logo=github)](https://github.com/keurfonluu/disba)
[![Pyversions](https://img.shields.io/pypi/pyversions/disba.svg?style=flat)](https://pypi.org/pypi/disba/)
[![Version](https://img.shields.io/pypi/v/disba.svg?style=flat)](https://pypi.org/project/disba)
[![Downloads](https://pepy.tech/badge/disba)](https://pepy.tech/project/disba)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat)](https://github.com/psf/black)
[![Codacy Badge](https://img.shields.io/codacy/grade/1d2218bb7d0e4e0fb2dec26fa32fe92e.svg?style=flat)](https://www.codacy.com/manual/keurfonluu/disba?utm_source=github.com&utm_medium=referral&utm_content=keurfonluu/disba&utm_campaign=Badge_Grade)
[![Codecov](https://img.shields.io/codecov/c/github/keurfonluu/disba.svg?style=flat)](https://codecov.io/gh/keurfonluu/disba)

**`disba`** is a computationally efficient Python library for the modeling of surface wave dispersion that implements a subset of codes from [Computer Programs in Seismology (CPS)](http://www.eas.slu.edu/eqc/eqccps.html) in Python compiled [_just-in-time_](https://en.wikipedia.org/wiki/Just-in-time_compilation) with [**`numba`**](https://numba.pydata.org/). Such implementation alleviates the usual prerequisite for a Fortran compiler needed by other libraries also based on CPS (e.g. [**`pysurf96`**](https://github.com/miili/pysurf96), [**`srfpython`**](https://github.com/obsmax/srfpython) and [**`PyLayeredModel`**](https://github.com/harrymd/PyLayeredModel)) which often leads to further installation troubleshooting, especially on Windows platform.

**`disba`** aims to be lightweight and portable without compromising on the performance. For instance, it yields similar speed compared to CPS's _surf96_ program compiled with [**`f2py`**](https://numpy.org/devdocs/f2py/index.html) for Rayleigh-wave but is significantly faster for Love-wave with increasing number of layers. **`disba`** also implements the _fast delta matrix_ algorithm for Rayleigh-wave which, albeit ironically slower, is more robust and handles reversion of phase velocity caused by low velocity zones.

| <img src="https://github.com/keurfonluu/disba/blob/master/.github/perf_rayleigh.svg"> | <img src="https://github.com/keurfonluu/disba/blob/master/.github/perf_love.svg"> |
| :-----------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------: |

## Features

Forward modeling:

-   Compute Rayleigh-wave dispersion curves using _Dunkin's matrix_ or _fast delta matrix_ algorithms,
-   Compute Love-wave dispersion curves using _Thomson-Haskell_ method,
-   Support phase and group dispersion velocity,
-   Support single top water layer.

Eigenfunctions and sensitivity kernels:

-   Compute Rayleigh- and Love- waves eigenfunctions,
-   Compute Rayleigh- and Love- waves phase or group sensitivity kernels with respect to layer thickness, P- and S- waves velocity, and density.

## Installation

The recommended way to install **`disba`** and all its dependencies is through the Python Package Index:

```bash
pip install disba --user
```

Otherwise, clone and extract the package, then run from the package location:

```bash
pip install . --user
```

## Usage

The following example computes the Rayleigh- and Love- waves phase velocity dispersion curves for the 3 first modes.

```python
import numpy
from disba import PhaseDispersion

# Velocity model
# thickness, Vp, Vs, density
# km, km/s, km/s, g/cm3
velocity_model = numpy.array([
    [10.0, 7.00, 3.50, 2.00],
    [10.0, 6.80, 3.40, 2.00],
    [10.0, 7.00, 3.50, 2.00],
    [10.0, 7.60, 3.80, 2.00],
    [10.0, 8.40, 4.20, 2.00],
    [10.0, 9.00, 4.50, 2.00],
    [10.0, 9.40, 4.70, 2.00],
    [10.0, 9.60, 4.80, 2.00],
    [10.0, 9.50, 4.75, 2.00],
])

# Periods must be sorted starting with low periods
t = numpy.logspace(0.0, 3.0, 100)

# Compute the 3 first Rayleigh- and Love- waves modal dispersion curves
# Fundamental mode corresponds to mode 0
pd = PhaseDispersion(*velocity_model.T)
cpr = [pd(t, mode=i, wave="rayleigh") for i in range(3)]
cpl = [pd(t, mode=i, wave="love") for i in range(3)]

# Returns a namedtuple (period, velocity, mode, wave, type)
```

| <img src="https://github.com/keurfonluu/disba/blob/master/.github/sample_rayleigh.svg"> | <img src="https://github.com/keurfonluu/disba/blob/master/.github/sample_love.svg"> |
| :-------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------: |

Likewise, `GroupDispersion` can be used for group velocity.

**`disba`**'s API is consistent between all its classes which are initialized and called in the same fashion. Eigenfunctions are calculated as follow:

```python
from disba import EigenFunction

eigf = EigenFunction(*velocity_model.T)
eigr = eigf(20.0, mode=0, wave="rayleigh")
eigl = eigf(20.0, mode=0, wave="love")

# Returns a namedtuple
#  - (depth, ur, uz, tz, tr, period, mode) for Rayleigh-waves
#  - (depth, uu, tt, period, mode) for Love-waves
```

| <img src="https://github.com/keurfonluu/disba/blob/master/.github/eigen_rayleigh.svg"> | <img src="https://github.com/keurfonluu/disba/blob/master/.github/eigen_love.svg"> |
| :------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------: |

And sensitivity kernels:

```python
from disba import PhaseSensitivity

ps = PhaseSensitivity(*velocity_model.T)
parameters = ["thickness", "velocity_p", "velocity_s", "density"]
skr = [ps(20.0, mode=0, wave="rayleigh", parameter=parameter) for parameter in parameters]
skl = [ps(20.0, mode=0, wave="love", parameter=parameter) for parameter in parameters]

# Returns a namedtuple (depth, kernel, period, velocity, mode, wave, type, parameter)
```

| <img src="https://github.com/keurfonluu/disba/blob/master/.github/kernel_rayleigh.svg"> | <img src="https://github.com/keurfonluu/disba/blob/master/.github/kernel_love.svg"> |
| :-------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------: |
