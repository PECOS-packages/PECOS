API Guide
=========

Concepts in PECOS are organized around the following namespaces:

=================== =================================================
``circuits``        Circuits of different abstraction levels.
``qeccs``           Represent QEC protocols.
``error_gens``      Used to specify error models and generate errors.
``simulators``      Simulate states and operations.
``circuit_runners`` Coordinate gates of ``circuits`` and ``error_gens`` with a ``simulator``.
``decoders``        Produce recovery operations given syndromes.
``tools``           Tools for studying and evaluating QEC protocols.
``misc``            A catch all namespace.
=================== =================================================

Classes and functions available in these namespaces are described in the following:

.. toctree::
   :maxdepth: 2

   quantum_circuits
   qeccs
   logical_circuits
   simulators
   standard_gates
   circuit_runners
   error_generators
   decoders
   tools
