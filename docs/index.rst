.. -*- coding: utf-8 -*-

.. figure:: images/pecos_large_logo.png
   :width: 400px


Welcome to the PECOS Docs!
==========================

PECOS, which stands for "Performance Estimator of Codes On Surfaces," is a Python package that provides a framework for studying, developing, and evaluating quantum error-correcting codes (QECCs).

PECOS is an attempt at balancing simplicity, usability, functionality, and extendibility as well as future-proofing. In the spirit of extendibility, PECOS is agnostic to quantum simulators, quantum operations, and QECCs. Of course, it is difficult to eloquently represent all QEC techniques. While agnostic to QECCs, the primary focus of PECOS has been the simulation and evaluation of lattice-surgery for topological stabilizer codes.

History
-------

The first incarnation of PECOS was created by Ciarán Ryan-Anderson in June 2014 to verify the lattice-surgery procedures in :arxiv:`1407.5103` [LRA14]_.

Make this Documentation
-----------------------

To build this documentation go to the ``docs`` folder and run:

.. code-block:: python

   >>> make clean
   >>> make html

What's Next?
------------

To get started, check out the following:

.. toctree::
   :maxdepth: 2

   install
   api_guide/index
   examples/index
   change_log
   bibliography
   todo_list



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
