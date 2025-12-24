.. _api-logical-circuits:

Logical Circuits
================

The class ``LogicalCircuit``, which is found in the ``circuits`` namespace, is a logical analog of the class
``QuantumCircuit``. The ``LogicalCircuit`` class has the same methods and attributes as ``QuantumCircuit``; however,
there are a few changes in the behavior of some of the methods. As the two classes are very similar, I will give a few
examples of using the ``LogicalCircuit`` class to illustrate their differences.

An instance of a ``LogicalCircuit`` can be created using the following lines:

>>> import pecos as pc
>>> logic = pc.circuits.LogicalCircuit()

Instead of gate symbols, the ``append`` method of the ``LogicalCircuit`` class accepts ``LogicalGates`` directly. Also,
if a ``LogicalCircuit`` contains a single ``qecc`` then a gate location is not needed:

>>> surface = pc.qeccs.Surface4444(distance=3)
>>> logic = pc.circuits.LogicalCircuit()
>>> logic.append(surface.gate("ideal init |0>"))
>>> logic.append(surface.gate("I"))
