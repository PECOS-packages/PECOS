.. _example-gate-error-models:

Gate-wise Error Models
======================

The ``GatewiseGen`` is an ``error_gen``  that allows users to design error models where gates can be applied according
to classical probability distributions that are specified for individual ideal gates or groups of ideal gates. To being
we write the following:

>>> import pecos as pc
>>> myerrors = pc.error_gens.GatewiseModel()

To randomly add an
to classical probability distributions that are specified for individual ideal gates or groups of ideal gates. To being
we write the following:

>>> import pecos as pc
>>> myerrors = pc.error_gens.GatewiseModel()

To randomly add an
to classical probability distributions that are specified for individual ideal gates or groups of ideal gates. To being
we write the following:

>>> import pecos as pc
>>> myerrors = pc.error_gens.GatewiseModel()

To randomly add an
to classical probability distributions that are specified for individual ideal gates or groups of ideal gates. To being
we write the following:

>>> import pecos as pc
>>> myerrors = pc.error_gens.GatewiseModel()

To randomly add an
to classical probability distributions that are specified for individual ideal gates or groups of ideal gates. To being
we write the following:

>>> import pecos as pc
>>> myerrors = pc.error_gens.GatewiseGen()

To randomly add an :math:`X` error after every Hadamard we write:

>>> # Continuing from last example.
>>> myerrors.set_gate_error('H', 'X')

Here, the probability of a :math:`X` error occurring will, by default, equal to the value of the key ``'p'`` in an
``error_params`` dictionary that is passed to the ``run_logic`` method of a ``circuit_runner``.

To test the error model we are creating, we can use the ``get_gate_error`` method to generate errors. The first argument
of the method is the ideal gate-symbol. The second, is a set of qudit locations the errors may occur on. The third, is a
``error_params`` dictionary used to specify the probability of errors. An example of using this method is seen here:

>>> # Continuing from last example.
>>> myerrors.get_gate_error('H', {0, 1, 2, 3, 4}, error_params={'p':0.5})   # doctest: +SKIP
(QuantumCircuit([{'X': {0, 1, 3}}]), QuantumCircuit([]), set())

Here, the method returns a tuple. The first element is the error circuit that is applied after the ideal gates. The
second, before the ideal gates. The third element is the set of qudit locations corresponding to gate locations of ideal
gates to be removed from the ideal quantum-circuit.

Note, by default errors specified by the ``set_gate_error`` method will be generated after the ideal quantum-gates. To
generate errors before the gates, one can set the keyword ``after`` to ``False`` when using the ``add_gate_error``
method.


The probability-parameter used (default being ``'p'``) can be changed by using the keyword ``error_param``:

>>> # Continuing from last example.
>>> myerrors.set_gate_error('H', 'X', error_param='q', after=False)
>>> myerrors.get_gate_error('H', {0, 1, 2, 3, 4}, error_params={'q':0.5})   # doctest: +SKIP
(QuantumCircuit([]), QuantumCircuit([{'X': {0, 3}}]), set())

Here we used the keyword``error_param`` to declare that ``'q'`` will be used to set the probability of an :math:`X`
error occurring. We also see an example of the keyword ``after`` being used to indicate that errors should be applied
before the ideal gates rather than after.

Besides specifying errors of a single gate-type, we can declare a set of errors to be uniformly drawn from:

>>> # Continuing from last example.
>>> myerrors.set_gate_error('X', {'X', 'Y', 'S'}, error_param='r')
>>> myerrors.get_gate_error('X', {0, 1, 2, 3, 4}, error_params={'r':0.5})   # doctest: +SKIP
(QuantumCircuit([{'S': {3, 4}}]), QuantumCircuit([]), set())

Such uniform error-distributions can be made for two-qubit gates as well:

>>> # Continuing from last example.
>>> myerrors.set_gate_error('CNOT', {('I', 'X'), ('X', 'X'), 'CNOT'}, error_param='r')
>>> myerrors.get_gate_error('CNOT', {(0,1), (2,3), (4,5), (6,7), (8,9)}, error_params={'r':0.8})   # doctest: +SKIP
(QuantumCircuit([{'CNOT': {(0, 1), (4, 5), (8, 9)}, 'X': {3, 6, 7}, 'I': {2}}]), QuantumCircuit([]),
set())

Here we see that two-qubit gates or tuples of single-qubit gates can be supplied as errors.

Other distributions besides the uniform distribution can be specified by passing a callable, such as a function or a
method. An example is seen in the following:

>>> # Continuing from last example.
>>> import random
>>> def error_func(after, before, replace, location, error_params):
...    s = error_params['s']
...    rand = random.triangular(0, 1, 0.6)
...    if rand < 0.6:
...       err = 'Q'
...    elif rand < 0.7:
...       err = 'S'
...    else:
...       err ='R'
...    before.update(err, {location}, emptyappend=True)

Here, callables that are used to create unique error distributions must take the arguments ``after``, ``before``,
``replace``, ``location``, and ``error_params``. The variables ``after`` and ``before`` are ``QuantumCircuits``
representing the errors that are applied after and before the ideals gates of a single tick, respectively. The variable
``replace`` is the set of qubit gate-locations of the ideals gates to be removed from the ideal quantum-circuit. These
callables are called only if error occurs according to the probability of an associated error parameter, which we will
see later how to set. The ``location`` variable is the qudit index or tuple of qudit indices where the error has
occurred. The variable ``error_params`` is the dictionary of error parameters that are being used to determine the
probability distribution of errors. In the above callable, we see a triangular distribution being used to apply quantum
errors. Note that the callable is responsible for updating ``QuantumCircuits``  ``after``, ``before``, ``replace`` as
appropriate.


To use callables to generate errors, we can call the ``set_gate_error`` method in the following manner:

>>> # Continuing from last example.
>>> myerrors.set_gate_error('Y', error_func, error_param='s')
>>> myerrors.get_gate_error('Y', {0, 1, 2, 3, 4}, error_params={'s':0.5})   # doctest: +SKIP
(QuantumCircuit([]), QuantumCircuit([{'R': {0, 4}, 'Q': {1, 2}}]), set())

Here we set the probability of ``error_func`` being called to generate errors using the ``error_params`` keyword
argument.

There are two special gate-symbols for which error distributions can be assigned to. These special symbols are
``'data'`` and ``'idle'``. The error distribution associated with ``'data'`` is used to generate errors at the beginning
of each ``LogicalInstruction`` for each data qudit. An error distribution associated with the ``'idle'`` symbol is used
to generate errors whenever a qubit is not acted on by a quantum operation during a ``LogicalCircuit``.

An example of setting the errors of a ``'data'`` and ``'idle'`` can see here:


>>> # Continuing from last example.
>>> myerrors.set_gate_error('data', 'X', error_param='q')
>>> myerrors.set_gate_error('idle', 'Y', error_param='s')


Besides specifying errors for individual gate-types, one can specify errors for a group of gates. To do this one may
define a gate group and set the error distribution for this group:

>>> # Continuing from last example.
>>> myerrors.set_gate_group('measurements', {'measure X', 'measure Y', 'measure Z'})
>>> myerrors.set_group_error('measurements', {'X', 'Y', 'Z'}, error_param='m')

Note, ``set_group_error`` will override the error distribution of any gate belonging to the gate group.

The gate groups that are defined by default can be found by running:

>>> newerrors = pc.error_gens.GatewiseModel()
>>> newerrors.gate_groups   # doctest: +SKIP
{'measurements': {'measure X', 'measure Y', 'measure Z'},
 'inits': {'init |+>', 'init |+i>', 'init |->', 'init |-i>', 'init |0>', 'init |1>'},
 'two_qubits': {'CNOT', 'CZ', 'G', 'SWAP'},
 'one_qubits': {'F1', 'F1d', 'F2', 'F2d', 'F3', 'F3d', 'F4', 'F4d', 'H', 'H+y-z', 'H+z+x', 'H-x+y', 'H-x-y', 'H-y-z',
 'H-z-x', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'I', 'Q', 'Qd', 'R', 'Rd', 'S', 'Sd', 'X', 'Y', 'Z'}}

Here the keys are symbols representing the gate groups and the values are the set of gate symbols belong to the
corresponding gate group. These gate groups (

The gate groups that are defined by default can be found by running:

>>> newerrors = pc.error_gens.GatewiseModel()
>>> newerrors.gate_groups   # doctest: +SKIP
{'measurements': {'measure X', 'measure Y', 'measure Z'},
 'inits': {'init |+>', 'init |+i>', 'init |->', 'init |-i>', 'init |0>', 'init |1>'},
 'two_qubits': {'CNOT', 'CZ', 'G', 'SWAP'},
 'one_qubits': {'F1', 'F1d', 'F2', 'F2d', 'F3', 'F3d', 'F4', 'F4d', 'H', 'H+y-z', 'H+z+x', 'H-x+y', 'H-x-y', 'H-y-z',
 'H-z-x', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'I', 'Q', 'Qd', 'R', 'Rd', 'S', 'Sd', 'X', 'Y', 'Z'}}

Here the keys are symbols representing the gate groups and the values are the set of gate symbols belong to the
corresponding gate group. These gate groups (

The gate groups that are defined by default can be found by running:

>>> newerrors = pc.error_gens.GatewiseModel()
>>> newerrors.gate_groups   # doctest: +SKIP
{'measurements': {'measure X', 'measure Y', 'measure Z'},
 'inits': {'init |+>', 'init |+i>', 'init |->', 'init |-i>', 'init |0>', 'init |1>'},
 'two_qubits': {'CNOT', 'CZ', 'G', 'SWAP'},
 'one_qubits': {'F1', 'F1d', 'F2', 'F2d', 'F3', 'F3d', 'F4', 'F4d', 'H', 'H+y-z', 'H+z+x', 'H-x+y', 'H-x-y', 'H-y-z',
 'H-z-x', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'I', 'Q', 'Qd', 'R', 'Rd', 'S', 'Sd', 'X', 'Y', 'Z'}}

Here the keys are symbols representing the gate groups and the values are the set of gate symbols belong to the
corresponding gate group. These gate groups (

The gate groups that are defined by default can be found by running:

>>> newerrors = pc.error_gens.GatewiseModel()
>>> newerrors.gate_groups   # doctest: +SKIP
{'measurements': {'measure X', 'measure Y', 'measure Z'},
 'inits': {'init |+>', 'init |+i>', 'init |->', 'init |-i>', 'init |0>', 'init |1>'},
 'two_qubits': {'CNOT', 'CZ', 'G', 'SWAP'},
 'one_qubits': {'F1', 'F1d', 'F2', 'F2d', 'F3', 'F3d', 'F4', 'F4d', 'H', 'H+y-z', 'H+z+x', 'H-x+y', 'H-x-y', 'H-y-z',
 'H-z-x', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'I', 'Q', 'Qd', 'R', 'Rd', 'S', 'Sd', 'X', 'Y', 'Z'}}

Here the keys are symbols representing the gate groups and the values are the set of gate symbols belong to the
corresponding gate group. These gate groups (

The gate groups that are defined by default can be found by running:

>>> newerrors = pc.error_gens.GatewiseGen()
>>> newerrors.gate_groups   # doctest: +SKIP
{'measurements': {'measure X', 'measure Y', 'measure Z'},
 'inits': {'init |+>', 'init |+i>', 'init |->', 'init |-i>', 'init |0>', 'init |1>'},
 'two_qubits': {'CNOT', 'CZ', 'G', 'SWAP'},
 'one_qubits': {'F1', 'F1d', 'F2', 'F2d', 'F3', 'F3d', 'F4', 'F4d', 'H', 'H+y-z', 'H+z+x', 'H-x+y', 'H-x-y', 'H-y-z',
 'H-z-x', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'I', 'Q', 'Qd', 'R', 'Rd', 'S', 'Sd', 'X', 'Y', 'Z'}}

Here the keys are symbols representing the gate groups and the values are the set of gate symbols belong to the
corresponding gate group. These gate groups (``'measurements'``, ``'inits'``, ``'two_qubits'``, and ``'one_qubits'``)
can be redefined by the user.

Example: The Symmetric Depolarizing-channel
-------------------------------------------

As an example, the circuit-level symmetric depolarizing-channel is modeled by ``DepolarGen`` as discussed in
:ref:`this page <DepolarGen>`, can be represented by the ``GatewiseGen`` class as follows:

.. code-block:: python

   depolar_circuit = pc.error_gens.GatewiseGen()
   set_gate_group('Xinit', {'init |+>', 'init |->'})
   set_gate_group('Yinit', {'init |+i>', 'init |-i>'})
   set_gate_group('Zinit', {'init |0>', 'init |1>'})
   depolar_circuit.set_group_error('Xinit', 'Z')
   depolar_circuit.set_group_error('Yinit', 'Z')
   depolar_circuit.set_group_error('Zinit', 'X')
   depolar_circuit.set_gate_error('measure X', 'Z', after=False)
   depolar_circuit.set_gate_error('measure Y', 'Z', after=False)
   depolar_circuit.set_gate_error('measure Z', 'X', after=False)
   depolar_circuit.set_group_error('one_qubits', {'X', 'Y', 'Z'})
   depolar_circuit.set_group_error('two_qubits', {('I', 'X'), ('I', 'Y'), ('I', 'Z'),
   ('X', 'I'), ('X', 'X'), ('X', 'Y'), ('X', 'Z'), ('Y', 'I'), ('Y', 'X'), ('Y', 'Y'),
   ('Y', 'Z'), ('Z', 'I'), ('Z', 'X'), ('I', 'Y'), ('Z', 'Z')})

Example: The Amplitude-dampening Channel
----------------------------------------

The stochastic circuit-level amplitude-dampening channel can be described as:

.. code-block:: python

  amp_damp = pc.error_gens.GatewiseGen()
  amp_damp.set_group_error('inits', 'init |0>')
  amp_damp.set_gate_error('measurements', 'init |0>',   after=False)
  amp_damp.set_group_error('one_qubits', 'init |0>')
  amp_damp.set_group_error('two_qubits', {('I', 'init |0>'), ('init |0>', 'I'),
  ('init |0>', 'init |0>')})
