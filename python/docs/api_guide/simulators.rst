.. _simulators:

Simulators
==========

Quantum states and their dynamics are simulated by classes belonging to the namespace ``simulators``. PECOS contains a
stabilizer simulator called ``SparseSim``.


Expected Methods
----------------

The set of gates allowed by a simulator may differ (the standard set for PECOS is given in :ref:`api-standard-gates`;
however, each simulator is expected to have a set of standard methods. I will describe them in this section.

When initializing a simulator, the first argument is expected to be the number of qudits to be simulated. This reserves
the size of the quantum registry:



>>> from pecos.simulators import SparseSim
>>> state = SparseSim(4)

Note, for all simulators, the initial state of each qudit is the state :math:`|0\rangle`.

The only other method expected is the ``run_gate`` method. This method can be used to apply gates to a ``simulator``
instance by using the ``run_gate`` method:

>>> # Continuing from the previous code block.
>>> state.run_gate("X", {0, 1})
{}

Here the first argument is a gate symbol that is recognized by the ``simulator`` and the second argument is a set of
gate locations. Other keywords and arguments may be supplied if it is allowed by the ``simulator``. Such arguments could
be used to change the behavior of the gate. For example, arguments could be used to define gate rotation-angles.


If measurements are made then a dictionary indicating the measurement results is returned by ``run_gate``:

>>> # Continuing from the previous code block.
>>> state.run_gate("measure Z", {0, 1, 3})
{0: 1, 1: 1}

Here we see that the keys of the results dictionary are the qudit locations of the measurements, and the values are the
corresponding measurement results except that zero results are not returned.

Classes in the ``circuit_runners`` namespace combine ``QuantumCircuits`` and simulators to apply gates to simulated
uantum states. For a discussion about these classes see :ref:`api-circ-run`.

StabSim
-------

Methods that specific to ``SparseSim`` will now be described.

The ``print_stabs`` method prints a stabilizer table corresponding to the state currently store in the simulator:

>>> state = SparseSim(3)
>>> state.run_gate("CNOT", {(0, 1)})
{}
>>> state.run_gate("X", {0})
{}
>>> state.print_stabs(print_destabs=True)
 -ZII
 -ZZI
  IIZ
-------------------------------
  XXI
  IXI
  IIX
([' -ZII', ' -ZZI', '  IIZ'], ['  XXI', '  IXI', '  IIX'])

Here in the print output stabilizer generators are indicated by the strings above the dashed lines, while destabilizer
generators are indicated by the strings below. Note that the destabilizers are only given when ``print_destabs`` is set
to ``True`` (default value is ``False``).

The ``logical_sign`` method can be used to determine the sign of stabilizer generators. As the stabilizer simulators
represent stabilizer states, logical basis-states are stabilized by logical operators. Therefore, this method is useful
in Monte Carlo simulations to determine if logical errors have flipped the sign of logical operators.

 An example of using the ``logical_sign`` method is seen in the following:

>>> # Continuing with the following example:
>>> from pecos.circuits import QuantumCircuit
>>> stab = QuantumCircuit([{"Z": {0, 1}}])
>>> state.logical_sign(stab)
1
>>> stab = QuantumCircuit([{"Z": {2}}])
>>> state.logical_sign(stab)
0

A :math:`1` is returned if the phase of the stabilizer is :math:`-1`, and a :math:`0` is returned if the phase is
:math:`+1`. If the stabilizer supplied to ``logical_sign`` is not a stabilizer of the state, then an exception will be
raised.
