disba
=====

|License| |Stars| |Pyversions| |Version| |Downloads| |Code style: black| |Codacy Badge| |Codecov|

**disba** is a computationally efficient Python library for the modeling of surface wave dispersion that implements a subset of codes from `Computer Programs in Seismology (CPS) <http://www.eas.slu.edu/eqc/eqccps.html>`__ in Python compiled `just-in-time <https://en.wikipedia.org/wiki/Just-in-time_compilation>`__ with `numba <https://numba.pydata.org/>`__. Such implementation alleviates the usual prerequisite for a Fortran compiler needed by other libraries also based on CPS (e.g. `pysurf96 <https://github.com/miili/pysurf96>`__, `srfpython <https://github.com/obsmax/srfpython>`__ and `PyLayeredModel <https://github.com/harrymd/PyLayeredModel>`__) which often leads to further installation troubleshooting, especially on Windows platform.

**disba** aims to be lightweight and portable without compromising on the performance. For instance, it yields similar speed compared to CPS's *surf96* program compiled with `f2py <https://numpy.org/devdocs/f2py/index.html>`__ for Rayleigh-wave but is significantly faster for Love-wave with increasing number of layers. **disba** also implements the *fast delta matrix* algorithm for Rayleigh-wave which, albeit ironically slower, is more robust and handles reversion of phase velocity caused by low velocity zones.

.. list-table::

   *  - |Perf Rayleigh|
      - |Perf Love|

Features
--------

Forward modeling:

-  Compute Rayleigh-wave phase or group dispersion curves using *Dunkin's matrix* or *fast delta matrix* algorithms,
-  Compute Love-wave phase or group dispersion curves using *Thomson-Haskell* method,
-  Compute Rayleigh-wave ellipticity.

Eigenfunctions and sensitivity kernels:

-  Compute Rayleigh- and Love- wave eigenfunctions,
-  Compute Rayleigh- and Love- wave phase or group sensitivity kernels with respect to layer thickness, P- and S- wave velocities, and density.

Installation
------------

The recommended way to install **disba** and all its dependencies is through the Python Package Index:

.. code:: bash

   pip install disba[full] --user

Otherwise, clone and extract the package, then run from the package location:

.. code:: bash

   pip install .[full] --user

Usage
-----

The following example computes the Rayleigh- and Love- wave phase velocity dispersion curves for the 3 first modes.

.. code:: python

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

   # Compute the 3 first Rayleigh- and Love- wave modal dispersion curves
   # Fundamental mode corresponds to mode 0
   pd = PhaseDispersion(*velocity_model.T)
   cpr = [pd(t, mode=i, wave="rayleigh") for i in range(3)]
   cpl = [pd(t, mode=i, wave="love") for i in range(3)]

   # pd returns a namedtuple (period, velocity, mode, wave, type)

.. list-table::

   *  - |Sample Rayleigh|
      - |Sample Love|

Likewise, ``GroupDispersion`` can be used for group velocity.

**disba**'s API is consistent across all its classes which are initialized and called in the same fashion. Thus, eigenfunctions are calculated as follow:

.. code:: python

   from disba import EigenFunction

   eigf = EigenFunction(*velocity_model.T)
   eigr = eigf(20.0, mode=0, wave="rayleigh")
   eigl = eigf(20.0, mode=0, wave="love")

   # eigf returns a namedtuple
   #  - (depth, ur, uz, tz, tr, period, mode) for Rayleigh-wave
   #  - (depth, uu, tt, period, mode) for Love-wave

.. list-table::

   *  - |Eigen Rayleigh|
      - |Eigen Love|

And sensitivity kernels:

.. code:: python

   from disba import PhaseSensitivity

   ps = PhaseSensitivity(*velocity_model.T)
   parameters = ["thickness", "velocity_p", "velocity_s", "density"]
   skr = [ps(20.0, mode=0, wave="rayleigh", parameter=parameter) for parameter in parameters]
   skl = [ps(20.0, mode=0, wave="love", parameter=parameter) for parameter in parameters]

   # ps returns a namedtuple (depth, kernel, period, velocity, mode,wave, type, parameter)

.. list-table::

   *  - |Kernel Rayleigh|
      - |Kernel Love|

.. |License| image:: https://img.shields.io/github/license/keurfonluu/disba
   :target: https://github.com/keurfonluu/disba/blob/master/LICENSE

.. |Stars| image:: https://img.shields.io/github/stars/keurfonluu/disba?logo=github
   :target: https://github.com/keurfonluu/disba

.. |Pyversions| image:: https://img.shields.io/pypi/pyversions/disba.svg?style=flat
   :target: https://pypi.org/pypi/disba/

.. |Version| image:: https://img.shields.io/pypi/v/disba.svg?style=flat
   :target: https://pypi.org/project/disba

.. |Downloads| image:: https://pepy.tech/badge/disba
   :target: https://pepy.tech/project/disba

.. |Code style: black| image:: https://img.shields.io/badge/code%20style-black-000000.svg?style=flat
   :target: https://github.com/psf/black

.. |Codacy Badge| image:: https://img.shields.io/codacy/grade/1d2218bb7d0e4e0fb2dec26fa32fe92e.svg?style=flat
   :target: https://www.codacy.com/manual/keurfonluu/disba?utm_source=github.com&utm_medium=referral&utm_content=keurfonluu/disba&utm_campaign=Badge_Grade

.. |Codecov| image:: https://img.shields.io/codecov/c/github/keurfonluu/disba.svg?style=flat
   :target: https://codecov.io/gh/keurfonluu/disba

.. |Perf Rayleigh| image:: https://raw.githubusercontent.com/keurfonluu/disba/5d23a8bb3967fd59c1a38b59ce1bf800749c7eb2/.github/perf_rayleigh.svg
   :alt: perf-rayleigh

.. |Perf Love| image:: https://raw.githubusercontent.com/keurfonluu/disba/5d23a8bb3967fd59c1a38b59ce1bf800749c7eb2/.github/perf_love.svg
   :alt: perf-love

.. |Sample Rayleigh| image:: https://raw.githubusercontent.com/keurfonluu/disba/5d23a8bb3967fd59c1a38b59ce1bf800749c7eb2/.github/sample_rayleigh.svg
   :alt: sample-rayleigh

.. |Sample Love| image:: https://raw.githubusercontent.com/keurfonluu/disba/5d23a8bb3967fd59c1a38b59ce1bf800749c7eb2/.github/sample_love.svg
   :alt: sample-love

.. |Eigen Rayleigh| image:: https://raw.githubusercontent.com/keurfonluu/disba/5d23a8bb3967fd59c1a38b59ce1bf800749c7eb2/.github/eigen_rayleigh.svg
   :alt: eigen-rayleigh

.. |Eigen Love| image:: https://raw.githubusercontent.com/keurfonluu/disba/5d23a8bb3967fd59c1a38b59ce1bf800749c7eb2/.github/eigen_love.svg
   :alt: eigen-love

.. |Kernel Rayleigh| image:: https://raw.githubusercontent.com/keurfonluu/disba/5d23a8bb3967fd59c1a38b59ce1bf800749c7eb2/.github/kernel_rayleigh.svg
   :alt: kernel-rayleigh

.. |Kernel Love| image:: https://raw.githubusercontent.com/keurfonluu/disba/5d23a8bb3967fd59c1a38b59ce1bf800749c7eb2/.github/kernel_love.svg
   :alt: kernel-love
