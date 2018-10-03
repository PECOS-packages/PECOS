Getting Started
===============

First of all, you should have Python 3.5+ installed to run PECOS. PECOS was developed using the `Anaconda Distribution of Python <https://www.anaconda.com/download/>`_.

Requirements
------------

Package requirements include:

* NumPy
* SciPy
* Matplotlib 2.2+
* NetworkX 2.1+

Optional packages include:

* Cython (for compiling C and C++ extensions)
* PyTest (for running tests)

Installing and Uninstalling
---------------------------

PECOS has been developed to run on both Windows and Linux-based sytems. To install PECOS form source, you can cd into PECOS's root where ``setup.py`` is located and run:

>>> pip install .  # doctest: +SKIP

To develop PECOS on your machine, instead run:

>>> pip install -e .  # doctest: +SKIP

To uninstall run:

>>> pip uninstall quantum-pecos  # doctest: +SKIP

Tests
-----

PECOS comes with tests to verify that the package is running as expected. These tests can be used in the development process to determine if any expected functionality has been broken.

To run tests, the package PyTest is require. Once installed, simply navigate to your PECOS installation directory and run:

>>> py.test    # doctest: +SKIP

PyTest will automatically run all tests and inform you of any failures.


Importing
---------

The standard method for importing PECOS is:

.. code-block:: python

   import pecos as pc

It will be assumed throughout the documentation that PECOS has already been imported in this manner.
