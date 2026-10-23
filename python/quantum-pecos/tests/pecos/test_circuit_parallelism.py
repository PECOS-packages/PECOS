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

"""Independent checks of static selected-operation packing metrics."""

from copy import deepcopy
from random import Random

import pytest
from pecos.analysis import circuit_parallelism
from pecos.circuits import QuantumCircuit


def test_compaction_preserves_per_qubit_order() -> None:
    """Disjoint gates move back, while shared wires retain their order."""
    circuit = QuantumCircuit()
    for pair in [(0, 1), (0, 2), (3, 4)]:
        circuit.append("CX", {pair})
    analysis = circuit_parallelism(circuit)
    assert [layer.locations for layer in analysis.layers] == [
        {(0, 1), (3, 4)},
        {(0, 2)},
    ]
    assert analysis.all_operations.operation_count == 3
    assert analysis.all_operations.mean_operations == 1.5
    assert analysis.all_operations.density(6) == 0.5
    assert analysis.unconditional == analysis.all_operations
    assert circuit_parallelism(circuit, compact=False).all_operations.layer_count == 3


def test_selection_barriers_are_explicit() -> None:
    """Projection may cross omitted operations only with explicit opt-in."""
    circuit = QuantumCircuit()
    circuit.append("CX", {(0, 1)})
    circuit.append("H", {1})
    circuit.append("CX", {(2, 3)})
    assert circuit_parallelism(circuit, gate_symbols={"CX"}).all_operations.layer_count == 2
    assert (
        circuit_parallelism(
            circuit,
            gate_symbols={"CX"},
            ignored_operations_are_barriers=False,
        ).all_operations.layer_count
        == 1
    )
    assert circuit_parallelism(circuit, gate_symbols=set()).all_operations.operation_count == 0
    with pytest.raises(TypeError, match="collection of symbols"):
        circuit_parallelism(circuit, gate_symbols="CX")


def test_zero_qubit_operations_are_barriers() -> None:
    """Classical operations cannot disappear as dependencies by default."""
    circuit = QuantumCircuit()
    circuit.append("X", {0})
    circuit.append("cop", set(), expr={"t": "x", "op": "=", "a": 1})
    circuit.append("X", {1})
    assert circuit_parallelism(circuit).all_operations.layer_count == 2
    assert circuit_parallelism(circuit, ignored_operations_are_barriers=False).all_operations.layer_count == 1


def test_conditions_are_scenarios_not_bounds() -> None:
    """Removing a sparse conditional layer can increase average parallelism."""
    circuit = QuantumCircuit()
    circuit.append("CX", {(0, 1), (2, 3)})
    circuit.append("CX", {(4, 5)}, cond={"a": "x", "op": "==", "b": 1})
    analysis = circuit_parallelism(circuit)
    assert analysis.all_operations.mean_operations == 1.5
    assert analysis.unconditional.mean_operations == 2.0
    assert analysis.all_operations.density(6) == 0.5
    assert analysis.unconditional.density(6) == pytest.approx(2 / 3)


def test_condition_transitions_stop_movement_and_input_is_unchanged() -> None:
    """Conditions are copied, and crossing a different condition is forbidden."""
    circuit = QuantumCircuit()
    for qubit, condition in [(0, {"a": "x"}), (1, {"a": "y"}), (2, {"a": "x"})]:
        circuit.append("X", {qubit}, cond=condition)
    before = deepcopy(list(circuit.items()))
    analysis = circuit_parallelism(circuit)
    assert len(analysis.layers) == 3
    analysis.layers[0].condition["a"] = "changed"
    assert list(circuit.items()) == before
    assert analysis.unconditional.mean_operations == 0
    assert analysis.unconditional.density(3) == 0


def test_empty_results_and_invalid_widths() -> None:
    """An empty circuit has finite zero metrics; invalid widths fail."""
    empty = circuit_parallelism(QuantumCircuit()).all_operations
    assert empty.mean_operations == 0
    assert empty.density(0) == 0
    circuit = QuantumCircuit([{"X": {3}}])
    summary = circuit_parallelism(circuit).all_operations
    for width in [-1, 0, 3]:
        with pytest.raises(ValueError, match="at least 4"):
            summary.density(width)
    with pytest.raises(TypeError, match="bool"):
        summary.density(True)
    with pytest.raises(TypeError, match="integer"):
        summary.density(4.5)


def test_mixed_arity_counts_occupied_qubits() -> None:
    """Density uses active qubits, not twice the number of operations."""
    circuit = QuantumCircuit()
    circuit.append("X", {0})
    circuit.append("CX", {(1, 2)})
    analysis = circuit_parallelism(circuit)
    assert analysis.all_operations.mean_operations == 2
    assert analysis.all_operations.mean_active_qubits == 3
    assert analysis.all_operations.density(4) == 0.75


def test_uncompacted_batches_do_not_claim_tick_depth() -> None:
    """Different symbol batches in one tick remain separate without compaction."""
    circuit = QuantumCircuit([{"X": {0}, "Z": {1}}])
    assert len(circuit) == 1
    assert circuit_parallelism(circuit, compact=False).all_operations.layer_count == 2
    assert circuit_parallelism(circuit).all_operations.layer_count == 1


def test_random_packing_matches_independent_wire_depth_oracle() -> None:
    """Selected gate multiplicities and wire depths match a separate scheduler."""
    random = Random(42)
    for _ in range(50):
        circuit = QuantumCircuit()
        expected: list[set[tuple[int, int]]] = []
        next_layer = [0] * 6
        for _ in range(20):
            pair = tuple(random.sample(range(6), 2))
            circuit.append("CX", {pair})
            depth = max(next_layer[q] for q in pair)
            while len(expected) <= depth:
                expected.append(set())
            expected[depth].add(pair)
            for qubit in pair:
                next_layer[qubit] = depth + 1
        analysis = circuit_parallelism(circuit)
        assert [layer.locations for layer in analysis.layers] == expected
        assert analysis.all_operations.operation_count == 20
        for layer in analysis.layers:
            assert len(layer.qubits) == 2 * len(layer.locations)
