.. -*- coding: utf-8 -*-

.. image:: images/pecos_large_logo.png
    :class: dark-light
    :width: 400px

Welcome to the PECOS Docs!
==========================

PECOS stands for "Performance Estimator of Codes On Surfaces." PECOS is a Python framework for
studying, developing, and evaluating quantum error-correction (QEC).

PECOS balances functionality, simplicity, and extendability. Classes representing key elements in the study of
QEC--such as QEC protocols, decoders, error models, and simulators--are decoupled from each other. These elements of QEC
can be developed separately, allowing you to extend PECOS and study novel topics in QEC.

History
-------

The first incarnation of PECOS was created by Ciarán Ryan-Anderson in June 2014 to verify the lattice-surgery procedures
in :arxiv:`1407.5103` [LRA14]_. Since then, PECOS has been expanded to become framework for studying general QECCs and
has lead to the development of a fast stabilizer-simulation algorithm :arxiv:`1812.04735` [RA18]_.

Make this Documentation
-----------------------

To build this documentation go to the ``docs`` folder and run:

.. code-block:: console

   $ make clean
   $ make html

What's Next?
------------

To get started, check out the following:

.. toctree::
   :maxdepth: 2

   install
   api_guide/index
   examples/index
   reference/index
   development/index
   change_log
   bibliography

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
