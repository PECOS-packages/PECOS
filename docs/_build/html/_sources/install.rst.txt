Getting Started
===============

Language Requirement
--------------------

Python 3.5.2+ is need to run.

Package Requirements
--------------------

Package requirements include:

* NumPy 1.15+
* SciPy 1.1+
* Matplotlib 2.2+
* NetworkX 2.1+

Optional packages include:

* Cython (for compiling C and C++ extensions)
* PyTest (for running tests)


Note on Python Distributation/Environment
-----------------------------------------

PECOS was developed using the `Anaconda Distribution of Python <https://www.anaconda.com/download/>`_. If you decide to
use this distribution you may want to create an `environment <https://conda.io/docs/user-guide/tasks/manage-environments.html>`_
so that PECOS's package requirements do not restrict you when working on other projects.

To create an environment for PECOS using Anaconda run

>>> conda create -n pecos python=X # doctest: +SKIP

where `X` is whatever version of Python you wish to use with PECOS (e.g., ``X=3.5.2`` or ``X=3.6``).

To activate/use the environment in Windows run the command:

>>> activate pecos # doctest: +SKIP

In other operating systems you may need to run the following instead:

>>> source activate pecos # doctest: +SKIP

To deactivate/leave the PECOS environment run:

>>> deactivate # doctest: +SKIP

Installing and Uninstalling
---------------------------

PECOS has been developed to run on both Windows and Linux-based systems.

To install using pip type:

>>> pip install quantum-pecos   # doctest: +SKIP


Alternatively, the plackage can be cloned or downloaded from GitHub:

https://github.com/PECOS-packages/PECOS

To clone PECOS using git run:

>>> git clone https://github.com/PECOS-packages/PECOS.git # doctest: +SKIP

Then, download/unzip or clone the version of PECOS you would like to use. Next, navigate to the root of the package 
(where setup.py is located) and run:

>>> pip install .   # doctest: +SKIP


To install and continue to develop the version of PECOS located in the install folder, run:

>>> pip install -e .  # doctest: +SKIP

To uninstall run:

>>> pip uninstall quantum-pecos  # doctest: +SKIP

Development Branch
------------------

For the latest features, you may wish to clone/download the development version of PECOS found in the development
branch:

https://github.com/PECOS-packages/PECOS/tree/development

To clone using git run:

>>> git clone -b development https://github.com/PECOS-packages/PECOS.git # doctest: +SKIP

Be aware that as PECOS is in development in this branch, you may experience some bugs.

Tests
-----

PECOS comes with tests to verify that the package is running as expected. These tests can be used in the development process to determine if any expected functionality has been broken.

To run tests, the package PyTest is require. Once installed, simply navigate to your PECOS installation directory and run:

>>> py.test    # doctest: +SKIP

PyTest will automatically run all the PECOS's tests and inform you of any failures.


Importing
---------

The standard method for importing PECOS is:

.. code-block:: python

   import pecos as pc

It will be assumed throughout the documentation that PECOS has been imported in this manner.
