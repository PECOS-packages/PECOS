.. _api-circ-run:

Circuit Runners
===============

Classes belonging to the ``circuit_runners`` namespace apply the gates of ``LogicalCircuits`` and ``QuantumCircuits`` to
states represented by simulators. ``circuit_runners`` are also responsible for applying error models to quantum circuit;
however, we will discus this in :ref:`error-gens`.

The main ``circuit_runner`` is simply called ``Standard``. There is another call ``TimingRunner``, which is essentially
the same as ``Standard`` except that it is used to time how long it takes simulators to apply gates and can be used to
compare the runtime of simulators. I will now discuss these two ``circuit_runners``.
 

Standard
--------

For convenience, the following tabels list the attributes and methods of ``Standard``:

Methods
~~~~~~~

=============== =========================================
``init``        Adds a collection of gates to the end of ``ticks``.
``run``         Combines a ``circuit``, ``error_gen``, ``simulator`` to run a simulation.
=============== =========================================



Attributes
~~~~~~~~~~

===================== ======================================
``seed``              The integer used as a seed for random number generators
===================== ======================================



Instance
~~~~~~~~

To create an instance of ``Standard`` one can simply write:

>>> import pecos as pc
>>> circ_runner = pc.circuit_runners.Standard()

By default, a ``Standard`` uses the ``StabSim`` as a simulator. This can be changed as follows:


>>> from somepackage import MyCustomSim  # doctest: +SKIP
>>> circ_runner = pc.circuit_runners.Standard(simulator=MyCustomSim)  # doctest: +SKIP

The ``init`` method is used to (re)initialize a ``simulator`` instance. An example of using this method to create a four-qubit registry is seen here:

>>> # Following from the previous code block.
>>> circ_runner = pc.circuit_runners.Standard()
>>> state = circ_runner.init(4)

The ``run_circuit`` method is used to apply a ``QuantumCircuit`` to a state in the following:

>>> # Continuing with the previous code block.
>>> qc = pc.circuits.QuantumCircuit()
>>> qc.append('X', {0, 1})
>>> qc.append('measure Z', {0, 1, 3})
>>> circ_runner.run(state, qc)
{1: {0: 1, 1: 1}}

In the last line of this code block, we see the measurement record produced by the ``circuit_runner``. The keys of the
outer dictionary are tick indices, while for the inner dictionary the keys are the indices of qubits with non-zero
measurements and the values are the measurement results.
   


The ``run_logic`` method is used to apply ``LogicalCircuits``:

>>> surface = pc.qeccs.Surface4444(distance=3)
>>> logic = pc.circuits.LogicalCircuit()
>>> logic.append(surface.gate('ideal init |0>'))
>>> logic.append(surface.gate('I'))
>>> state = circ_runner.init(surface.num_qudits)
>>> circ_runner.run(state, logic)
({}, {})



The final line is the output of ``run``. The first dictionary is a record measurement and the second is a record
of the errors generated. In this example, all the measurement results are zero and we have not applied any error models.
In :ref:`error-gens`, there are examples of where this is not the case; therefore, refer to that section if you are
curious about the output of ``run``.
   
TimingRunner
------------

As mention, ``TimingRunner`` is essentially the same as ``Standard`` except the runtime for applying gates is recorded.
The attribute ``total_time`` stores this value and is used in the following:

>>> circ_runner = pc.circuit_runners.TimingRunner()
>>> state = circ_runner.init(4)
>>> qc = pc.circuits.QuantumCircuit()
>>> qc.append('X', {0, 1, 2, 3})
>>> circ_runner.run(state, qc)
{}
>>> circ_runner.total_time   # doctest: +SKIP
7.22257152574457e-06

``TimingRunner`` times the execution of gates by using Python's ``perf_counter`` method. The time recorded by
``total_time`` continues to accumulate until it is reset by the ``reset_time`` method:

>>> # Continuing from previous Listing.
>>> circ_runner.reset_time()
>>> circ_runner.total_time
0.0