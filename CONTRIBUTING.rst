Contributing Guidelines
=======================

This project is mainly developed and maintained by `@keurfonluu <https://github.com/keurfonluu>`__. Everyone is welcomed to participate to the development of this project either by:

-  Submitting bug reports and/or feature requests via `issue tracker <https://github.com/keurfonluu/toughio/issues>`__,
-  Fixing bugs and/or adding new features through a `pull request <https://github.com/keurfonluu/toughio/pulls>`__,
-  Improving `documentation <https://toughio.readthedocs.io/>`__ by fixing typos and/or adding tutorials.

If you wish to participate but are new to GitHub, please read `GitHub's guide <https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/about-pull-requests>`__ and do not hesitate to open an issue.

Contributing to the code
------------------------

The first step to contributing is to fork and/or clone the repository. Additional Python libraries required for the development can be installed via the command:

.. code:: bash

    pip install -r requirements-dev.txt

The code is formatted with `black <https://github.com/psf/black>`__ so you don't have to worry about code formatting. Docstrings follow `NumPy's style guide <https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>`__ and all public functions, classes and methods must come with a full description of input parameters and return outputs. To format both code and docstrings, simply run the command:

.. code:: bash

    invoke format

Every new features must be unit tested in the directory ``test`` using the `pytest <https://docs.pytest.org/en/stable/>`__ framework. To run the test suite, run the command:

.. code:: bash

    pytest

Any change to the source code must be submitted via a pull request such that it goes through the continuous integration service that verifies that the code is up to standards and passes all the unit tests.
