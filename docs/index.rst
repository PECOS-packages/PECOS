.. -*- coding: utf-8 -*-

.. figure:: images/pecos_large_logo.png
   :width: 400px


Welcome to the PECOS Docs!
==========================

PECOS, which stands for "Performance Estimator of Codes On Surfaces," is a Python package that provides a framework for
studying, developing, and evaluating quantum error-correcting codes (QECCs).

PECOS attempts to balance simplicity, usability, functionality, and extendability while also attempting to be
future-proofed. The framework treats the main classes used to study/develop/evaluate QECCs like
interchangeable black boxes. Although, there are some necessary restrictions/specifications on inputs and outputs of the
classes. (This is essentially a pipeline approach.) These "black box" classes can easily be switched out for classed
created by the user. This flexible design approach has lead PECOS to be agnostic to quantum simulators, quantum
operations, QECCs, and evaluation methods. Therefore, PECOS can be used to encapsulate and study a wide variety of QECC
protocols; however, one of the main influences on design choice has been the study of lattice surgery for topological
stabilizer codes.


History
-------

The first incarnation of PECOS was created by Ciarán Ryan-Anderson in June 2014 to verify the lattice-surgery procedures
in :arxiv:`1407.5103` [LRA14]_. Since then, PECOS has been expanded to become framework for studying general QECCs and
has lead to the development of a fast stabilizer-simulation algorithm :arxiv:`1812.04735` [RA18]_.

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
