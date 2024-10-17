.. _decoders:

Decoders
========

A decoder in PECOS is simply a function or other callable that takes the measurement outcomes from error extractions
(syndromes) as input and returns a ``QuantumCircuit``, which is used as a recovery operation to mitigate errors. Decoder
classes and functions are in the ``decoders`` namespace.

The ``MWPM2D`` class is an available decoder class, which I will discuss next.


MWPM2D
------

One of the standard decoders used for surface codes is the minimum-weight-perfect-matching (MWPM) decoder [Den+02]_. The
``MWPM2D`` class implements the 2D version of this decoder for ``Surface4444`` and ``SurfaceMedial4444``, that is, it
decodes syndromes for a single round of error extraction:

>>> import pecos as pc
>>> depolar = pc.error_gens.DepolarModel(model_level="code_capacity")
>>> surface = pc.qeccs.Surface4444(distance=3)
>>> logic = pc.circuits.LogicalCircuit()
>>> logic.append(surface.gate("ideal init |0>"))
>>> logic.append(surface.gate("I", num_syn_extract=1))
>>> circ_runner = pc.circuit_runners.Standard(seed=1)
>>> state = pc.simulators.SparseSim(surface.num_qudits)
>>> decode = pc.decoders.MWPM2D(surface).decode
>>> meas, err = circ_runner.run_logic(
...     state, logic, error_gen=depolar, error_params={"p": 0.1}
... )
>>> meas  # doctest: +SKIP
{(1, 0): {7: {3: 1, 5: 1, 9: 1, 15: 1}}}
>>> err  # doctest: +SKIP
{(1, 0): {0: {'after': QuantumCircuit([{'Y': {4}, 'X': {10}}])}}}
>>> decode(meas)  # doctest: +SKIP
QuantumCircuit([{'X': {10}, 'Y': {4}}])
decodes syndromes for a single round of error extraction:

>>> import pecos as pc
>>> depolar = pc.error_gens.DepolarModel(model_level="code_capacity")
>>> surface = pc.qeccs.Surface4444(distance=3)
>>> logic = pc.circuits.LogicalCircuit()
>>> logic.append(surface.gate("ideal init |0>"))
>>> logic.append(surface.gate("I", num_syn_extract=1))
>>> circ_runner = pc.circuit_runners.Standard(seed=1)
>>> state = pc.simulators.SparseSim(surface.num_qudits)
>>> decode = pc.decoders.MWPM2D(surface).decode
>>> meas, err = circ_runner.run_logic(
...     state, logic, error_gen=depolar, error_params={"p": 0.1}
... )
>>> meas  # doctest: +SKIP
{(1, 0): {7: {3: 1, 5: 1, 9: 1, 15: 1}}}
>>> err  # doctest: +SKIP
{(1, 0): {0: {'after': QuantumCircuit([{'Y': {4}, 'X': {10}}])}}}
>>> decode(meas)  # doctest: +SKIP
QuantumCircuit([{'X': {10}, 'Y': {4}}])
decodes syndromes for a single round of error extraction:

>>> import pecos as pc
>>> depolar = pc.error_gens.DepolarModel(model_level="code_capacity")
>>> surface = pc.qeccs.Surface4444(distance=3)
>>> logic = pc.circuits.LogicalCircuit()
>>> logic.append(surface.gate("ideal init |0>"))
>>> logic.append(surface.gate("I", num_syn_extract=1))
>>> circ_runner = pc.circuit_runners.Standard(seed=1)
>>> state = pc.simulators.SparseSim(surface.num_qudits)
>>> decode = pc.decoders.MWPM2D(surface).decode
>>> meas, err = circ_runner.run_logic(
...     state, logic, error_gen=depolar, error_params={"p": 0.1}
... )
>>> meas  # doctest: +SKIP
{(1, 0): {7: {3: 1, 5: 1, 9: 1, 15: 1}}}
>>> err  # doctest: +SKIP
{(1, 0): {0: {'after': QuantumCircuit([{'Y': {4}, 'X': {10}}])}}}
>>> decode(meas)  # doctest: +SKIP
QuantumCircuit([{'X': {10}, 'Y': {4}}])
decodes syndromes for a single round of error extraction:

>>> import pecos as pc
>>> depolar = pc.error_gens.DepolarModel(model_level="code_capacity")
>>> surface = pc.qeccs.Surface4444(distance=3)
>>> logic = pc.circuits.LogicalCircuit()
>>> logic.append(surface.gate("ideal init |0>"))
>>> logic.append(surface.gate("I", num_syn_extract=1))
>>> circ_runner = pc.circuit_runners.Standard(seed=1)
>>> state = pc.simulators.SparseSim(surface.num_qudits)
>>> decode = pc.decoders.MWPM2D(surface).decode
>>> meas, err = circ_runner.run_logic(
...     state, logic, error_gen=depolar, error_params={"p": 0.1}
... )
>>> meas  # doctest: +SKIP
{(1, 0): {7: {3: 1, 5: 1, 9: 1, 15: 1}}}
>>> err  # doctest: +SKIP
{(1, 0): {0: {'after': QuantumCircuit([{'Y': {4}, 'X': {10}}])}}}
>>> decode(meas)  # doctest: +SKIP
QuantumCircuit([{'X': {10}, 'Y': {4}}])
decodes syndromes for a single round of error extraction:

>>> import pecos as pc
>>> depolar = pc.error_gens.DepolarGen(model_level="code_capacity")
>>> surface = pc.qeccs.Surface4444(distance=3)
>>> logic = pc.circuits.LogicalCircuit()
>>> logic.append(surface.gate("ideal init |0>"))
>>> logic.append(surface.gate("I", num_syn_extract=1))
>>> circ_runner = pc.circuit_runners.Standard(seed=1)
>>> state = pc.simulators.SparseSim(surface.num_qudits)
>>> decode = pc.decoders.MWPM2D(surface).decode
>>> meas, err = circ_runner.run_logic(
...     state, logic, error_gen=depolar, error_params={"p": 0.1}
... )
>>> meas  # doctest: +SKIP
{(1, 0): {7: {3: 1, 5: 1, 9: 1, 15: 1}}}
>>> err  # doctest: +SKIP
{(1, 0): {0: {'after': QuantumCircuit([{'Y': {4}, 'X': {10}}])}}}
>>> decode(meas)  # doctest: +SKIP
QuantumCircuit([{'X': {10}, 'Y': {4}}])
