Analysis
========

``CircuitStatistics`` collects operation counts and durations for the maximum,
minimum, and actual runtime paths through a circuit. ``HybridEngine`` manages
the run lifecycle and reports its per-operation execution decisions directly:

.. code-block:: python

    import pecos as pc
    from pecos.analysis import CircuitStatistics
    from pecos.simulators import SparseStab

    circuit = pc.QuantumCircuit(num_qubits=1)
    circuit.append("H", {0}, duration=2.0)

    statistics = CircuitStatistics()
    pc.HybridEngine(seed=1).run(
        SparseStab(1),
        circuit,
        shot_id=0,
        circ_inspector=statistics,
    )

    assert statistics.results["total"]["runtime"] == [2.0]

Use a classifier to group circuit symbols or add parallel-layer counts without
changing the collector:

.. code-block:: python

    from pecos.analysis import OperationStatistic


    def classify(symbol, locations, metadata):
        yield OperationStatistic(
            key="one_qubit",
            count=len(locations),
            duration=metadata.get("duration"),
        )


    statistics = CircuitStatistics(classifier=classify)

Custom inspectors that opt into exact operation events implement
``pecos.protocols.OperationEventInspector``. Unmarked inspectors continue to
use the legacy per-tick ``analyze`` callback.
