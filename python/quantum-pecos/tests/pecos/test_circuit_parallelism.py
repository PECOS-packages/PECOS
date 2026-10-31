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


def test_batches_in_one_tick_share_a_layer() -> None:
    """Different symbol batches in one tick pack into one layer."""
    circuit = QuantumCircuit([{"X": {0}, "Z": {1}}])
    assert len(circuit) == 1
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


def test_barrier_ignores_batch_order_within_a_tick() -> None:
    """An omitted operation fences later ticks, whatever its position in its tick."""
    layer_counts = []
    for tick in [{"X": {0}, "H": {1}}, {"H": {1}, "X": {0}}]:
        circuit = QuantumCircuit()
        circuit.append(tick)
        circuit.append("X", {2})
        layer_counts.append(circuit_parallelism(circuit, gate_symbols={"X"}).all_operations.layer_count)
    assert layer_counts == [2, 2]


def test_results_are_unhashable_and_summaries_are_computed_once() -> None:
    """Layers carry mutable condition copies, so hashing fails even without a condition."""
    circuit = QuantumCircuit()
    circuit.append("X", {0})
    analysis = circuit_parallelism(circuit)
    assert analysis.layers[0].condition is None
    with pytest.raises(TypeError, match="unhashable"):
        hash(analysis.layers[0])
    with pytest.raises(TypeError, match="unhashable"):
        hash(analysis)
    assert analysis.all_operations is analysis.all_operations
    assert analysis.unconditional is analysis.unconditional


def _reference_layers(
    circuit: QuantumCircuit,
    symbols: set[str] | None,
    *,
    barriers: bool,
) -> list[tuple[set[tuple[int, ...]], object]]:
    """Scan back from the newest layer until a different condition or shared qubit."""
    layers: list[tuple[set[tuple[int, ...]], object]] = []
    barrier = 0
    for tick in range(len(circuit)):
        ignored = False
        for symbol, locations, metadata in circuit.items(tick):
            if not locations or (symbols is not None and symbol not in symbols):
                ignored = True
                continue
            condition = metadata.get("cond")
            for support in sorted((loc,) if isinstance(loc, int) else tuple(loc) for loc in locations):
                target = len(layers)
                for position in range(len(layers) - 1, barrier - 1, -1):
                    supports, layer_condition = layers[position]
                    if layer_condition != condition or any(q in other for other in supports for q in support):
                        break
                    target = position
                if target == len(layers):
                    layers.append((set(), condition))
                layers[target][0].add(support)
        if ignored and barriers:
            barrier = len(layers)
    return layers


def test_random_packing_matches_backward_scan_with_conditions_and_barriers() -> None:
    """Placement agrees with a direct backward scan across conditions, omissions and barriers."""
    random = Random(7)
    conditions = [None, {"a": "m", "op": "==", "b": 1}, {"a": "m", "op": "==", "b": 0}]
    for _ in range(200):
        circuit = QuantumCircuit()
        for _ in range(15):
            if random.random() < 0.1:
                circuit.append("cop", set(), expr={"t": "m", "op": "=", "a": 1})
                continue
            qubits = random.sample(range(8), random.randint(1, 8))
            tick: dict[str, set] = {}
            while qubits:
                symbol = random.choice(["X", "H", "CX"] if len(qubits) > 1 else ["X", "H"])
                if symbol == "CX":
                    tick.setdefault(symbol, set()).add((qubits.pop(), qubits.pop()))
                else:
                    tick.setdefault(symbol, set()).add(qubits.pop())
            condition = random.choice(conditions)
            if condition is None:
                circuit.append(tick)
            else:
                circuit.append(tick, cond=condition)
        for symbols in [None, {"X", "CX"}]:
            for barriers in [True, False]:
                analysis = circuit_parallelism(
                    circuit,
                    gate_symbols=symbols,
                    ignored_operations_are_barriers=barriers,
                )
                expected = _reference_layers(circuit, symbols, barriers=barriers)
                assert [(layer.locations, layer.condition) for layer in analysis.layers] == expected


def test_condition_order_within_a_tick_does_not_change_packing() -> None:
    """Concurrent batches with different conditions pack the same in any insertion order."""
    first = {"a": "m", "op": "==", "b": 1}
    second = {"a": "m", "op": "==", "b": 0}
    results = []
    for (qubit, condition), (other_qubit, other_condition) in [((0, first), (1, second)), ((1, second), (0, first))]:
        circuit = QuantumCircuit()
        circuit.append("X", {qubit}, cond=condition)
        circuit.update("X", {other_qubit}, tick=0, cond=other_condition)
        circuit.append("X", {2}, cond=first)
        results.append([(layer.locations, layer.condition) for layer in circuit_parallelism(circuit).layers])
    assert results[0] == results[1]
