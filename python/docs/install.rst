Getting Started
===============

Language Requirement
--------------------

Python 3.9+ is need to run.

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


Note on Python Distribution/Environment
-----------------------------------------

PECOS was developed using the `Anaconda Distribution of Python <https://www.anaconda.com/download/>`_. If you decide to
use this distribution you may want to create an
`environment <https://conda.io/docs/user-guide/tasks/manage-environments.html>`_
so that PECOS's package requirements do not restrict you when working on other projects.

To create an environment for PECOS using Anaconda run:

.. code-block:: console

    $ conda create -n pecos python=X numpy scipy matplotlib networkx

where `X` is whatever version of Python you wish to use with PECOS (e.g., ``python=3.5.2``, ``python=3.6``,
``python=3.7``, etc.).

Alternatively, if you clone/download the package (see next section) and navigate to the root, you can create an
environment by running:

.. code-block:: console

    $ conda env create -f conda_environment.yml

This will create the environment ``pecos`` with the specific versions of Python and required packages that were used to
develop PECOS. Note, you will still need to install PECOS using one of the methods described in the following sections.

To activate/use the environment in Windows run the command:

.. code-block:: console

    $ activate pecos

In other operating systems you may need to run the following instead:

.. code-block:: console

    $ source activate pecos

To deactivate/leave the PECOS environment run:

.. code-block:: console

    $ deactivate

Installing and Uninstalling
---------------------------

PECOS has been developed to run on both Windows and Linux-based systems.

To install using pip run:

.. code-block:: console

    $ pip install quantum-pecos


Alternatively, the plackage can be cloned or downloaded from GitHub:

https://github.com/PECOS-packages/PECOS

To clone PECOS using git run:

.. code-block:: console

    $ git clone https://github.com/PECOS-packages/PECOS.git

Then, download/unzip or clone the version of PECOS you would like to use. Next, navigate to the root of the package
(where pyproject.toml is located) and run the command:

.. code-block:: console

    $ pip install .


To install and continue to develop the version of PECOS located in the install folder, run:

.. code-block:: console

    $ pip install -e .

To uninstall run:

.. code-block:: console

    $ pip uninstall quantum-pecos

Development Branch
------------------

For the latest features, you may wish to clone/download the version of PECOS found in the development branch:

https://github.com/PECOS-packages/PECOS/tree/development

To clone using git run:

.. code-block:: console

    $ git clone -b development https://github.com/PECOS-packages/PECOS.git

Be aware that as PECOS is in development in this branch, you may experience some bugs.

Tests
-----

PECOS comes with tests to verify that the package is running as expected. These tests can be used in the development
process to determine if any expected functionality has been broken.

To run tests, the package PyTest is require. Once installed, simply navigate to your PECOS installation directory and
run:

.. code-block:: console

    $ pytest

PyTest will automatically run all the PECOS's tests and inform you of any failures.


Importing
---------

The standard method for importing PECOS is:

.. code-block:: python

    import pecos as pc

It will be assumed throughout the documentation that PECOS has been imported in this manner.
