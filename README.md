# disba

[![License](https://img.shields.io/badge/license-MIT-green)](https://github.com/keurfonluu/disba/blob/master/LICENSE)
[![Stars](https://img.shields.io/github/stars/keurfonluu/disba?logo=github)](https://github.com/keurfonluu/toughio)
[![Pyversions](https://img.shields.io/pypi/pyversions/disba.svg?style=flat)](https://pypi.org/pypi/toughio/)
[![Version](https://img.shields.io/pypi/v/disba.svg?style=flat)](https://pypi.org/project/toughio)
[![Downloads](https://pepy.tech/badge/disba)](https://pepy.tech/project/toughio)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat)](https://github.com/psf/black)
[![Codacy Badge](https://img.shields.io/codacy/grade/1d2218bb7d0e4e0fb2dec26fa32fe92e.svg?style=flat)](https://www.codacy.com/manual/keurfonluu/disba?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=keurfonluu/disba&amp;utm_campaign=Badge_Grade)
[![Codecov](https://img.shields.io/codecov/c/github/keurfonluu/disba.svg?style=flat)](https://codecov.io/gh/keurfonluu/disba)

**`disba`** is a computationally efficient Python library for the modeling of surface wave dispersion curves that implements [_surf96_](http://www.eas.slu.edu/eqc/eqccps.html)'s code in Python compiled [_just-in-time_](https://en.wikipedia.org/wiki/Just-in-time_compilation) with [**`numba`**](https://numba.pydata.org/). Such implementation alleviates the usual prerequisite for a Fortran compiler needed by other libraries also based on _surf96_ (e.g. [**`pysurf96`**](https://github.com/miili/pysurf96) and [**`srfpython`**](https://github.com/obsmax/srfpython)) which often leads to further setup troubleshooting, especially on Windows platform.

**`disba`**'s speed is comparable to _surf96_ compiled with [**`f2py`**](https://numpy.org/devdocs/f2py/index.html) for Rayleigh-wave but significantly faster for Love-wave with increasing number of layers. **`disba`** also implements the _fast delta matrix_ algorithm for Rayleigh-wave which, albeit ironically slower, is more robust and handles reversion of phase velocity.

| <img src="https://github.com/keurfonluu/disba/blob/master/.github/perf_rayleigh.png"> | <img src="https://github.com/keurfonluu/disba/blob/master/.github/perf_love.png"> |
| :-----------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------: |

## Features

Forward modeling:

-   Compute Rayleigh-wave dispersion curves using _Dunkin's matrix_ or _fast delta matrix_ algorithms,
-   Compute Love-wave dispersion curves using _Thomson-Haskell_ method,
-   Support phase and group dispersion velocity.

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

The following example computes the Rayleigh- and Love- waves phase velocity dispersion curves for the 20 first modes using the _fast delta matrix_ algorithm for Rayleigh-wave (_surf96_ fails due to reverted phase velocity). The phase velocity increment is set to 1 m/s for root finding to avoid _modal jumps_ at higher frequencies.

```python
import numpy
from disba import PhaseDispersion

velocity_model = numpy.array([
    [0.5, 1.0, 0.5, 1.8],
    [0.3, 2.0, 1.0, 1.8],
    [10.0, 1.0, 0.5, 1.8],
])
pd = PhaseDispersion(*velocity_model.T, algorithm="fast-delta", dc=0.001)
f = numpy.linspace(0.1, 10.0, 100)
t = 1.0 / f[::-1]

cpr = [pd(t, mode=i, wave="rayleigh") for i in range(20)]
cpl = [pd(t, mode=i, wave="love") for i in range(20)]
```

| <img src="https://github.com/keurfonluu/disba/blob/master/.github/sample_rayleigh.png"> | <img src="https://github.com/keurfonluu/disba/blob/master/.github/sample_love.png"> |
| :-------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------: |
