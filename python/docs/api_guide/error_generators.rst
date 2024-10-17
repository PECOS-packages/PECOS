.. _error-gens:

Error Generators
================

Error models are represented by classes called "error generators" that are in the ``error_models`` namespace. They are
called upon by ``circuit_runners`` to apply noise to ideal quantum circuits.

In this section I will discuss ``GatewiseGen`` and ``DepolarModel`` classes. Both represent ``stochastic error models``.
That is, error models that apply gates as noise according to classical probability distributions.

GatewiseGen
-----------

The ``GatewiseGen`` class allow one to define custom stochastic error-models where for each ideal gate-type the errors
applied to the ideal gate and the classical probability distribution for applying errors can be specified.

The follow section provides examples of how ``error_models`` are used in practice

.. _DepolarModel:

DepolarModel
------------

The ``DepolarModel`` class is used to represent the symmetric depolarizing channel, which is commonly studied in QEC. For
single-qubit gates, this class is used to apply errors at probability :math:`p` from set :math:`\{X, Y, Z\}`. For
two-qubit gates, errors also occur with probability :math:`p` but errors are chosen uniformally from the set
:math:`\{I, X, Y, Z\}^{\otimes 2} \; \setminus \; I\otimes I`. Errors are always applied after ideal gates except for
measurements. In which case, the errors are applied before.

An example of creating an instance of ``DepolarModel`` is seen here:

>>> import pecos as pc
>>> depolar = pc.error_models.DepolarModel(
...     model_level="code_capacity", has_idle_errors=False, perp_errors=True
... )

The

>>> import pecos as pc
>>> depolar = pc.error_models.DepolarModel(
...     model_level="code_capacity", has_idle_errors=False, perp_errors=True
... )

The

>>> import pecos as pc
>>> depolar = pc.error_models.DepolarModel(
...     model_level="code_capacity", has_idle_errors=False, perp_errors=True
... )

The

>>> import pecos as pc
>>> depolar = pc.error_models.DepolarModel(
...     model_level="code_capacity", has_idle_errors=False, perp_errors=True
... )

The

>>> import pecos as pc
>>> depolar = pc.error_models.DepolarModel(
...     model_level="code_capacity", has_idle_errors=False, perp_errors=True
... )

The ``model_level`` keyword is used to specify to what set of gates the ``DepolarModel`` is applied to. If ``model_level``
is set to the value of ``'code\_capacity'``, then the error model is applied before each ``LogicalInstruction`` to each
data qubits as if these qubits are acted on by ``'I'``. The error model is not applied to any other circuit element. If
``model_level`` is set to the value ``'phenomenological'``, then the error model applied to data qubits before each
``LogicalInstruction`` as well as to any measurement. If ``model_level`` is set to the value ``'circuit'``, then the
error model is applied to all the gates in the ``QuantumCircuit``. The default value of ``model_level`` is
``'circuit'``.

The ``has_idle_errors`` is a keyword that is only relevant when ``model_level == 'circuit'``. If ``has_idle_errors`` is
set to ``True``, then the error model is applied to inactive qubits as if the qubit is acted on by ``'I'``. If
``has_idle_errors`` is set to ``False``, then this does not occur. The default value of ``has_idle_errors`` is ``True``.

If the ``perp_errors`` keyword is set to ``True``, then errors that are applied to Pauli-basis initializations and
measurements are errors that do not include the Pauli-basis of the initializations or measurements. So, for example,
:math:`Z` is not applied as an error to the ``'init |0>'`` operation. If the ``perp_errors`` keyword is set to
``False``, then there is no restriction to the errors. The default value of ``perp_errors`` is ``True``.

An example of applying an error model using ``DepolarModel`` to a ``LogicalCircuit`` is seen in the following:


>>> depolar = pc.error_models.DepolarModel(model_level="code_capacity")
>>> surface = pc.qeccs.Surface4444(distance=3)
>>> logic = pc.circuits.LogicalCircuit()
>>> logic.append(surface.gate("ideal init |0>"))
>>> logic.append(surface.gate("I"))
>>> circ_runner = pc.circuit_runners.Standard(seed=1)
>>> state = pc.simulators.SparseSim(surface.num_qudits)
>>> meas, err = circ_runner.run(
...     state, logic, error_model=depolar, error_params={"p": 0.1}
... )

Note that the keyword argument


>>> depolar = pc.error_models.DepolarModel(model_level="code_capacity")
>>> surface = pc.qeccs.Surface4444(distance=3)
>>> logic = pc.circuits.LogicalCircuit()
>>> logic.append(surface.gate("ideal init |0>"))
>>> logic.append(surface.gate("I"))
>>> circ_runner = pc.circuit_runners.Standard(seed=1)
>>> state = pc.simulators.SparseSim(surface.num_qudits)
>>> meas, err = circ_runner.run(
...     state, logic, error_model=depolar, error_params={"p": 0.1}
... )

Note that the keyword argument


>>> depolar = pc.error_models.DepolarModel(model_level="code_capacity")
>>> surface = pc.qeccs.Surface4444(distance=3)
>>> logic = pc.circuits.LogicalCircuit()
>>> logic.append(surface.gate("ideal init |0>"))
>>> logic.append(surface.gate("I"))
>>> circ_runner = pc.circuit_runners.Standard(seed=1)
>>> state = pc.simulators.SparseSim(surface.num_qudits)
>>> meas, err = circ_runner.run(
...     state, logic, error_model=depolar, error_params={"p": 0.1}
... )

Note that the keyword argument


>>> depolar = pc.error_models.DepolarModel(model_level="code_capacity")
>>> surface = pc.qeccs.Surface4444(distance=3)
>>> logic = pc.circuits.LogicalCircuit()
>>> logic.append(surface.gate("ideal init |0>"))
>>> logic.append(surface.gate("I"))
>>> circ_runner = pc.circuit_runners.Standard(seed=1)
>>> state = pc.simulators.SparseSim(surface.num_qudits)
>>> meas, err = circ_runner.run(
...     state, logic, error_model=depolar, error_params={"p": 0.1}
... )

Note that the keyword argument


>>> depolar = pc.error_models.DepolarModel(model_level="code_capacity")
>>> surface = pc.qeccs.Surface4444(distance=3)
>>> logic = pc.circuits.LogicalCircuit()
>>> logic.append(surface.gate("ideal init |0>"))
>>> logic.append(surface.gate("I"))
>>> circ_runner = pc.circuit_runners.Standard(seed=1)
>>> state = pc.simulators.SparseSim(surface.num_qudits)
>>> meas, err = circ_runner.run(
...     state, logic, error_model=depolar, error_params={"p": 0.1}
... )

Note that the keyword argument ``error_params`` is used to pass a dictionary that indicates the probability :math:`p` of
the depolarizing error model.

The values returned by the ``run`` method is recorded in the variables ``meas`` and ``err``. These variables are
dictionaries that record the measurement output and applied errors.

An example of measurement outcomes is given here:

>>> # Following the previous example.
>>> meas  # doctest: +SKIP
{(1, 0): {7: {9: 1, 11: 1}}}

Here, in the last line, we see the measurement outcome. The key of the outer dictionary is a tuple where the first
element is the tick index of the ``LogicalGate`` and the second element is an index corresponding to a
``LogicalInstance``. That is, the tuple records at what point in the ``LogicalCircuit`` was the measurement made. The
value of the outer dictionary is just the measurement-outcome dictionary of a ``QuantumCircuit``.

We can see the errors that were generated by the ``DepolarModel`` in these lines:

>>> # Following the previous example.
>>> err  # doctest: +SKIP
{(1, 0): {0: {'after': QuantumCircuit([{'X': {4}, 'Z': {10}}])}}}

In the above code block, we see a dictionary that stores what errors were applied to the ``LogicalCircuit``. The key of the
outer dictionary, once again, is a tuple indicating the tick of a ``LogicalGate`` and the index of a
``LogicalInstance``. The key of the next inner dictionary is ``QuantumCircuit`` tick when the error occurred. The key
``'after'`` of the next inner dictionary indicates that the errors are applied after ideal gates. The key ``'before'``
is used when indicating that errors are applied before gates. The values of both the ``'after'`` and ``'before'`` keys
are ``QuantumCircuits``. These circuits are the errors that are applied.

The data structure used to describe the errors that are applied to a ``LogicalCircuit`` can be directly supplied to a
``run`` method of a ``circuit_runner``. Doing so will cause the ``logic`` method to apply the given error to a
``LogicalCircuit``. This can be seen in the following:

>>> # Continuing the previous examples.
>>> logic2 = pc.circuits.LogicalCircuit()
>>> logic2.append(surface.gate("ideal init |+>"))
>>> logic2.append(surface.gate("I"))
>>> state2 = pc.simulators.SparseSim(surface.num_qudits)
>>> meas2, err2 = circ_runner.run(state2, logic2, error_circuits=err)

One use for this is to apply the same error to a different logical basis-state. Doing so allows one to determine if a
logical error occurs for the logical operations that stabilizer the basis state.

Note that the ``circuit_runners`` can apply errors to both ``LogicalCircuits`` and ``QuantumCircuits``.
