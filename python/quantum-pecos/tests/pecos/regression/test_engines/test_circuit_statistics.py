# Copyright 2026 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Tests for condition-aware circuit execution statistics."""

from __future__ import annotations

import pecos as pc
import pytest
from pecos import BitInt
from pecos.analysis import CircuitStatistics, OperationStatistic
from pecos.engines.cvm import DefaultClassicalSemantics, UnsignedClassicalSemantics
from pecos.simulators import SparseStab


def _analyze_circuit(
    statistics: CircuitStatistics,
    circuit: pc.QuantumCircuit,
    output: dict,
) -> None:
    statistics.new()
    for tick_circuit, time, _metadata in circuit.iter_ticks():
        statistics.analyze(tick_circuit, time, output)
    statistics.finalize()


def test_default_classifier_records_counts_durations_and_summary() -> None:
    """Unconditional operations contribute to every execution view."""
    circuit = pc.QuantumCircuit()
    circuit.append("X", {0, 1}, duration=2.5)

    statistics = CircuitStatistics()
    _analyze_circuit(statistics, circuit, {})

    run = statistics.data["runs"][0]
    assert run["count"] == {
        "max": {"X": 2},
        "min": {"X": 2},
        "runtime": {"X": 2},
    }
    assert run["duration"] == {
        "max": {"X": 2.5},
        "min": {"X": 2.5},
        "runtime": {"X": 2.5},
    }
    assert statistics.data["total"] == {
        "runtime": [2.5],
        "avg_runtime": (2.5, 0.0),
    }


def test_classifier_can_emit_grouped_and_parallel_statistics() -> None:
    """A caller can adapt its gate vocabulary without changing aggregation."""

    def classify(symbol, locations, metadata):
        assert symbol == "H"
        return (
            OperationStatistic(
                "one_qubit",
                count=len(locations),
                duration=metadata["duration"],
            ),
            OperationStatistic("parallel_one_qubit", count=1),
        )

    circuit = pc.QuantumCircuit()
    circuit.append("H", {0, 1}, duration=3.0)
    statistics = CircuitStatistics(classifier=classify)

    _analyze_circuit(statistics, circuit, {})

    run = statistics.data["runs"][0]
    assert run["count"]["runtime"] == {
        "one_qubit": 2,
        "parallel_one_qubit": 1,
    }
    assert run["duration"]["runtime"] == {"one_qubit": 3.0}


def test_conditions_share_the_configured_classical_policy() -> None:
    """Maximum, minimum, and runtime views preserve conditional semantics."""
    condition = {"a": "word", "op": "<", "b": 0}
    circuit = pc.QuantumCircuit()
    circuit.append("X", {0}, duration=1.0, cond=condition)
    output = {"word": BitInt(8, -2)}
    statistics = CircuitStatistics(
        UnsignedClassicalSemantics(8),
        regwidth=8,
    )

    _analyze_circuit(statistics, circuit, output)

    run = statistics.data["runs"][0]
    assert run["count"]["max"] == {"X": 1}
    assert run["count"]["min"] == {}
    assert run["count"]["runtime"] == {}
    assert run["duration"]["max"] == {"X": 1.0}
    assert run["duration"]["min"] == {}
    assert run["duration"]["runtime"] == {}

    statistics.set_classical_semantics(
        DefaultClassicalSemantics(),
        regwidth=8,
    )
    _analyze_circuit(statistics, circuit, output)

    assert statistics.data["runs"][1]["count"]["runtime"] == {"X": 1}
    assert statistics.data["total"] == {
        "runtime": [0, 1.0],
        "avg_runtime": (0.5, 0.5),
    }


def test_hybrid_engine_calls_concrete_inspector() -> None:
    """The engine manages the concrete inspector's run lifecycle."""
    circuit = pc.QuantumCircuit(num_qubits=1)
    circuit.append("X", {0}, duration=4.0)
    statistics = CircuitStatistics()

    pc.HybridEngine(seed=1).run(
        SparseStab(1),
        circuit,
        shot_id=0,
        circ_inspector=statistics,
    )

    assert not statistics.run_active
    assert statistics.data["runs"][0]["count"]["runtime"] == {"X": 1}
    assert statistics.data["total"]["runtime"] == [4.0]


def test_engine_reports_pre_mutation_execution_decision() -> None:
    """Runtime statistics use the decision made before an operation executes."""
    circuit = pc.QuantumCircuit()
    circuit.append(
        "cop",
        set(),
        duration=2.0,
        cond={"a": "word", "op": "==", "b": 0},
        expr={"t": "word", "op": "=", "a": 1},
    )
    output = {"word": BitInt(8, 0)}
    statistics = CircuitStatistics()

    result, _errors = pc.HybridEngine(seed=1).run(
        SparseStab(1),
        circuit,
        shot_id=0,
        output=output,
        circ_inspector=statistics,
    )

    assert int(result["word"]) == 1
    run = statistics.data["runs"][0]
    assert run["count"]["runtime"] == {"cop": 1}
    assert run["duration"]["runtime"] == {"cop": 2.0}


def test_engine_decision_accounts_for_cond2_and_skip() -> None:
    """Operations rejected by the full engine predicate are not runtime work."""
    circuit = pc.QuantumCircuit(num_qubits=1)
    circuit.append(
        "X",
        {0},
        duration=3.0,
        cond2={"a": "word", "op": "==", "b": 1},
    )
    circuit.append("Y", {0}, duration=5.0, skip=True)
    statistics = CircuitStatistics()

    pc.HybridEngine(seed=1).run(
        SparseStab(1),
        circuit,
        shot_id=0,
        output={"word": BitInt(8, 0)},
        circ_inspector=statistics,
    )

    run = statistics.data["runs"][0]
    assert run["count"] == {
        "max": {"X": 1},
        "min": {},
        "runtime": {},
    }
    assert run["duration"] == {
        "max": {"X": 3.0},
        "min": {},
        "runtime": {},
    }


def test_empty_condition_is_unconditional() -> None:
    """An empty condition contributes to the minimum execution view."""
    circuit = pc.QuantumCircuit(num_qubits=1)
    circuit.append("X", {0}, duration=1.0, cond={})
    statistics = CircuitStatistics()

    pc.HybridEngine(seed=1).run(
        SparseStab(1),
        circuit,
        shot_id=0,
        circ_inspector=statistics,
    )

    run = statistics.data["runs"][0]
    assert run["count"]["min"] == {"X": 1}
    assert run["count"]["runtime"] == {"X": 1}


def test_failed_engine_run_is_aborted_and_collector_can_be_reused() -> None:
    """A failed engine-owned run does not leave partial active statistics."""
    attempts = 0

    def flaky_classifier(symbol, locations, metadata):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            msg = "classification failed"
            raise ValueError(msg)
        return (
            OperationStatistic(
                symbol,
                count=len(locations),
                duration=metadata["duration"],
            ),
        )

    circuit = pc.QuantumCircuit(num_qubits=1)
    circuit.append("X", {0}, duration=1.0)
    statistics = CircuitStatistics(classifier=flaky_classifier)
    engine = pc.HybridEngine(seed=1)

    with pytest.raises(ValueError, match="classification failed"):
        engine.run(
            SparseStab(1),
            circuit,
            shot_id=0,
            circ_inspector=statistics,
        )

    assert not statistics.run_active
    assert statistics.data["runs"] == []

    engine.run(
        SparseStab(1),
        circuit,
        shot_id=1,
        circ_inspector=statistics,
    )
    assert statistics.data["runs"][0]["count"]["runtime"] == {"X": 1}


def test_partial_lifecycle_inspector_uses_legacy_fallback() -> None:
    """Unmarked legacy inspectors are not opted into managed lifecycle."""

    class LegacyInspector:
        def __init__(self) -> None:
            self.started = False
            self.analyzed = 0

        def start_run(self) -> None:
            self.started = True

        def analyze(self, *_args) -> None:
            self.analyzed += 1

    circuit = pc.QuantumCircuit(num_qubits=1)
    circuit.append("X", {0})
    inspector = LegacyInspector()

    pc.HybridEngine(seed=1).run(
        SparseStab(1),
        circuit,
        shot_id=0,
        circ_inspector=inspector,
    )

    assert not inspector.started
    assert inspector.analyzed == 1


def test_marked_incomplete_event_inspector_fails_explicitly() -> None:
    """An explicit event opt-in must implement the complete protocol."""

    class IncompleteInspector:
        supports_operation_events = True
        run_active = False

        def start_run(self) -> None:
            pass

    circuit = pc.QuantumCircuit(num_qubits=1)
    circuit.append("X", {0})

    with pytest.raises(
        TypeError,
        match="finish_run, abort_run, analyze_operation",
    ):
        pc.HybridEngine(seed=1).run(
            SparseStab(1),
            circuit,
            shot_id=0,
            circ_inspector=IncompleteInspector(),
        )


def test_run_circuit_override_keeps_legacy_signature() -> None:
    """Engine subclasses need not accept an operation-analyzer keyword."""

    class LegacyEngine(pc.HybridEngine):
        def run_circuit(
            self,
            state,
            output,
            output_export,
            circuit,
            error_gen,
            removed_locations=None,
        ):
            return super().run_circuit(
                state,
                output,
                output_export,
                circuit,
                error_gen,
                removed_locations,
            )

    circuit = pc.QuantumCircuit(num_qubits=1)
    circuit.append("X", {0}, duration=1.0)
    statistics = CircuitStatistics()

    LegacyEngine(seed=1).run(
        SparseStab(1),
        circuit,
        shot_id=0,
        circ_inspector=statistics,
    )

    assert statistics.data["runs"][0]["count"]["runtime"] == {"X": 1}


def test_lifecycle_and_classifier_failures_are_explicit() -> None:
    """Invalid lifecycle and classifier results fail near their source."""
    circuit = pc.QuantumCircuit()
    circuit.append("X", {0})
    tick_circuit, time, _metadata = next(circuit.iter_ticks())

    with pytest.raises(TypeError, match="classifier must be callable"):
        CircuitStatistics(classifier=None)

    statistics = CircuitStatistics()
    with pytest.raises(RuntimeError, match=r"Call start_run\(\)"):
        statistics.analyze(tick_circuit, time, {})
    with pytest.raises(RuntimeError, match=r"Call start_run\(\)"):
        statistics.finish_run()

    statistics = CircuitStatistics(classifier=lambda *_args: (object(),))
    statistics.start_run()
    with pytest.raises(RuntimeError, match="Finish the active run"):
        statistics.start_run()
    with pytest.raises(TypeError, match="OperationStatistic"):
        statistics.analyze(tick_circuit, time, {})


def test_results_are_a_defensive_copy() -> None:
    """Consumers cannot mutate the collector through the typed result view."""
    circuit = pc.QuantumCircuit()
    circuit.append("X", {0}, duration=1.0)
    statistics = CircuitStatistics()
    _analyze_circuit(statistics, circuit, {})

    results = statistics.results
    results["runs"][0]["count"]["runtime"]["X"] = 99

    assert statistics.data["runs"][0]["count"]["runtime"]["X"] == 1


def test_manual_contributions_treat_empty_condition_as_unconditional() -> None:
    """Direct contribution methods match the engine's empty-condition policy."""
    statistics = CircuitStatistics()
    statistics.start_run()

    statistics.add_count("X", 1, {}, {})
    statistics.add_duration("X", 2.0, {}, {})
    statistics.finish_run()

    run = statistics.data["runs"][0]
    assert run["count"]["min"] == {"X": 1}
    assert run["duration"]["min"] == {"X": 2.0}


def test_default_classifier_rejects_non_numeric_duration() -> None:
    """Malformed duration metadata does not silently corrupt summaries."""
    circuit = pc.QuantumCircuit()
    circuit.append("X", {0}, duration="slow")
    statistics = CircuitStatistics()
    statistics.new()
    tick_circuit, time, _metadata = next(circuit.iter_ticks())

    with pytest.raises(TypeError, match="duration must be numeric"):
        statistics.analyze(tick_circuit, time, {})
