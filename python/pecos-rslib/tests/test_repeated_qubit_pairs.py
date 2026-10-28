# Copyright 2026 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.

"""Caller errors must be ValueError, never a Rust PanicException."""

import math

import pytest
from pecos_rslib import CoinToss, PauliProp, SparseStab, Stabilizer, StabVec, StateVec


@pytest.mark.parametrize("simulator_class", [SparseStab, Stabilizer, StateVec, StabVec, PauliProp])
@pytest.mark.parametrize(
    "gate",
    [
        "CX",
        "CNOT",
        "CY",
        "CZ",
        "SXX",
        "SXXdg",
        "SYY",
        "SYYdg",
        "SZZ",
        "SZZdg",
        "SWAP",
        "G",
        "RXX",
        "RYY",
        "RZZ",
        "RXYXY2Q",
    ],
)
def test_repeated_pair_is_value_error(simulator_class, gate):
    simulator = simulator_class() if simulator_class is PauliProp else simulator_class(2)
    with pytest.raises(ValueError, match=rf"{gate} requires distinct qubits; repeated qubit 1"):
        simulator.run_2q_gate(gate, (1, 1), None)


@pytest.mark.parametrize("gate", ["cx", "cy", "cz", "swap"])
def test_pauli_prop_direct_gate_rejects_entire_batch(gate):
    simulator = PauliProp()
    simulator.add_paulis({"X": {0}})
    before = simulator.to_dense_string(3)
    with pytest.raises(ValueError, match=rf"{gate.upper()} requires distinct qubits; repeated qubit 2"):
        getattr(simulator, gate)([(0, 1), (2, 2)])
    assert simulator.to_dense_string(3) == before


FIXED_REGISTER_CLASSES = [SparseStab, Stabilizer, StateVec, StabVec]
BATCH_CLASSES = [*FIXED_REGISTER_CLASSES, PauliProp]


def _prepared_simulator(simulator_class):
    if simulator_class is PauliProp:
        simulator = PauliProp()
        simulator.add_paulis({"X": {0}})
    else:
        simulator = simulator_class(3)
        simulator.run_gate("H", {0})
    return simulator


def _snapshot(simulator):
    if isinstance(simulator, PauliProp):
        return simulator.to_dense_string(3)
    if isinstance(simulator, StabVec):
        return simulator.state_vector()
    if isinstance(simulator, StateVec):
        return simulator.__reduce__()[1]  # Register size and exact amplitude bytes.
    return simulator.stab_tableau(), simulator.destab_tableau()


@pytest.mark.parametrize("simulator_class", BATCH_CLASSES)
@pytest.mark.parametrize("gate", ["CX", "CNOT", "SWAP"])
def test_run_gate_repeated_pair_is_value_error(simulator_class, gate):
    simulator = _prepared_simulator(simulator_class)
    with pytest.raises(ValueError, match=rf"{gate} requires distinct qubits; repeated qubit 1"):
        simulator.run_gate(gate, {(1, 1)})


@pytest.mark.parametrize("simulator_class", BATCH_CLASSES)
@pytest.mark.parametrize(
    "params",
    [{}, {"simulate_gate": True}, {"simulate_gate": False}],
    ids=["batch_fast", "per_location", "skipped"],
)
def test_run_gate_repeated_batch_is_atomic(simulator_class, params):
    simulator = _prepared_simulator(simulator_class)
    before = _snapshot(simulator)
    with pytest.raises(ValueError, match="CX requires distinct qubits; repeated qubit 2"):
        simulator.run_gate("CX", {(0, 1), (2, 2)}, **params)
    assert _snapshot(simulator) == before


@pytest.mark.parametrize("simulator_class", FIXED_REGISTER_CLASSES)
@pytest.mark.parametrize(
    "params",
    [{}, {"simulate_gate": True}, {"simulate_gate": False}],
    ids=["batch_fast", "per_location", "skipped"],
)
@pytest.mark.parametrize("invalid_pair", [(2, 3), (3, 2)])
def test_run_gate_out_of_range_batch_is_atomic(simulator_class, params, invalid_pair):
    simulator = _prepared_simulator(simulator_class)
    before = _snapshot(simulator)
    with pytest.raises(ValueError, match="CX: qubit 3 is outside the 3-qubit register"):
        simulator.run_gate("CX", {(0, 1), invalid_pair}, **params)
    assert _snapshot(simulator) == before


@pytest.mark.parametrize("simulator_class", FIXED_REGISTER_CLASSES)
@pytest.mark.parametrize(("gate", "params"), [("CX", None), ("RZZ", {"angle": 0.25})])
@pytest.mark.parametrize("pair", [(0, 9), (9, 0)])
def test_direct_pair_bounds_are_value_error(simulator_class, gate, params, pair):
    simulator = _prepared_simulator(simulator_class)
    before = _snapshot(simulator)
    with pytest.raises(ValueError, match=rf"{gate}: qubit 9 is outside the 3-qubit register"):
        simulator.run_2q_gate(gate, pair, params)
    assert _snapshot(simulator) == before


@pytest.mark.parametrize("simulator_class", FIXED_REGISTER_CLASSES)
@pytest.mark.parametrize("gate", ["X", "H", "MZ"])
def test_direct_single_qubit_bounds_are_value_error(simulator_class, gate):
    simulator = _prepared_simulator(simulator_class)
    before = _snapshot(simulator)
    with pytest.raises(ValueError, match=rf"{gate}: qubit 9 is outside the 3-qubit register"):
        simulator.run_1q_gate(gate, 9, None)
    assert _snapshot(simulator) == before


def test_growable_pauli_prop_accepts_any_single_qubit_index():
    assert PauliProp().run_1q_gate("X", 99, None) is None


class _GateBatchCircuit:
    def __init__(self, gate, locations, params=None):
        self.gate = gate
        self.locations = locations
        self.params = params or {}

    def items(self):
        return [(self.gate, self.locations, self.params)]


@pytest.mark.parametrize(
    ("simulator_class", "entry"),
    [
        (SparseStab, "run_circuit"),
        (SparseStab, "add_faults"),
        (Stabilizer, "run_circuit"),
        (Stabilizer, "add_faults"),
        (StateVec, "run_circuit"),
        (StabVec, "run_circuit"),
        (StabVec, "add_faults"),
    ],
)
@pytest.mark.parametrize(
    ("invalid_pair", "message"),
    [
        ((2, 2), "CX requires distinct qubits; repeated qubit 2"),
        ((2, 3), "CX: qubit 3 is outside the 3-qubit register"),
    ],
)
def test_circuit_gate_batch_is_atomic(simulator_class, entry, invalid_pair, message):
    simulator = _prepared_simulator(simulator_class)
    before = _snapshot(simulator)
    circuit = _GateBatchCircuit("CX", {(0, 1), invalid_pair})
    with pytest.raises(ValueError, match=message):
        getattr(simulator, entry)(circuit)
    assert _snapshot(simulator) == before


def test_pauli_prop_can_grow_beyond_current_support():
    simulator = _prepared_simulator(PauliProp)
    simulator.run_gate("CX", {(0, 9)})
    assert simulator.contains_x(9)
    simulator.run_2q_gate("CX", (9, 12), None)
    assert simulator.contains_x(12)


@pytest.mark.parametrize("gate", ["CX", "SWAP", "RZZ"])
@pytest.mark.parametrize(
    ("pair", "message"),
    [
        ((1, 1), "requires distinct qubits; repeated qubit 1"),
        ((0, 3), "qubit 3 is outside the 3-qubit register"),
        ((3, 0), "qubit 3 is outside the 3-qubit register"),
    ],
)
def test_coin_toss_pair_is_value_error(gate, pair, message):
    simulator = CoinToss(3)
    with pytest.raises(ValueError, match=rf"{gate}.*{message}"):
        simulator.run_gate_2(gate, *pair, None)


def test_coin_toss_valid_pair_remains_noop():
    simulator = CoinToss(3)
    assert simulator.run_gate_2("CX", 0, 1, None) == {}


@pytest.mark.parametrize("simulator_class", BATCH_CLASSES)
def test_run_gate_rotation_repeated_batch_is_atomic(simulator_class):
    simulator = _prepared_simulator(simulator_class)
    before = _snapshot(simulator)
    with pytest.raises(ValueError, match="RZZ requires distinct qubits; repeated qubit 2"):
        simulator.run_gate("RZZ", {(0, 1), (2, 2)}, angle=math.pi / 2)
    assert _snapshot(simulator) == before


class _ChangingIndex:
    """Catch a second extraction of a caller-controlled operand."""

    def __init__(self, first):
        self.first = first
        self.calls = 0

    def __index__(self):
        self.calls += 1
        return self.first if self.calls == 1 else 2


@pytest.mark.parametrize("simulator_class", BATCH_CLASSES)
@pytest.mark.parametrize("path", ["batch", "per_location", "direct"])
def test_pair_operands_are_extracted_once(simulator_class, path):
    simulator = _prepared_simulator(simulator_class)
    expected = _prepared_simulator(simulator_class)
    expected.run_gate("CX", {(0, 1)})
    a, b = _ChangingIndex(0), _ChangingIndex(1)
    if path == "direct":
        simulator.run_2q_gate("CX", (a, b), None)
    else:
        params = {} if path == "batch" else {"simulate_gate": True}
        simulator.run_gate("CX", {(a, b)}, **params)
    assert (a.calls, b.calls) == (1, 1)
    assert _snapshot(simulator) == _snapshot(expected)


@pytest.mark.parametrize("simulator_class", BATCH_CLASSES)
@pytest.mark.parametrize("location", [0, "bad", (0,), (0, 1, 2), (-1, 0), (0, -1), (0, 1.5)])
@pytest.mark.parametrize("path", ["batch", "per_location", "skipped", "direct"])
def test_malformed_pair_has_gate_and_location(simulator_class, location, path):
    simulator = _prepared_simulator(simulator_class)
    before = _snapshot(simulator)
    params = {} if path == "batch" else {"simulate_gate": path != "skipped"}
    operation = (
        (lambda: simulator.run_2q_gate("CX", location, None))
        if path == "direct"
        else (lambda: simulator.run_gate("CX", {(0, 1), location}, **params))
    )
    with pytest.raises(ValueError, match="CX") as error:
        operation()
    assert "CX" in str(error.value)
    assert repr(location) in str(error.value)
    assert "2-tuple of non-negative qubit indices" in str(error.value)
    assert _snapshot(simulator) == before


@pytest.mark.parametrize("method", ["run_gate_1", "run_gate_2", "run_measure"])
def test_coin_toss_all_entry_points_check_bounds(method):
    simulator = CoinToss(3, seed=42)
    control = CoinToss(3, seed=42)
    symbol = "MZ" if method == "run_measure" else "X" if method == "run_gate_1" else "CX"
    args = (3,) if method == "run_measure" else (symbol, 3, None) if method == "run_gate_1" else (symbol, 0, 3, None)
    with pytest.raises(ValueError, match=rf"{symbol}: qubit 3 is outside the 3-qubit register"):
        getattr(simulator, method)(*args)
    assert [simulator.run_measure(0) for _ in range(20)] == [control.run_measure(0) for _ in range(20)]


@pytest.mark.parametrize("simulator_class", BATCH_CLASSES)
@pytest.mark.parametrize("simulate_gate", [False, True])
def test_two_qubit_identity_validates_pairs(simulator_class, simulate_gate):
    simulator = _prepared_simulator(simulator_class)
    before = _snapshot(simulator)
    with pytest.raises(ValueError, match="II requires distinct qubits; repeated qubit 2"):
        simulator.run_gate("II", {(0, 1), (2, 2)}, simulate_gate=simulate_gate)
    if simulator_class is not PauliProp:
        with pytest.raises(ValueError, match="II: qubit 3 is outside the 3-qubit register"):
            simulator.run_gate("II", {(0, 1), (2, 3)}, simulate_gate=simulate_gate)
    assert _snapshot(simulator) == before


@pytest.mark.parametrize(
    ("method", "args", "gate"),
    [
        ("run_gate_1", ("X", -1, None), "X"),
        ("run_gate_2", ("CX", -1, 0, None), "CX"),
        ("run_gate_2", ("CX", 0, -1, None), "CX"),
        ("run_measure", (-1,), "MZ"),
    ],
)
def test_coin_toss_negative_indices_are_gate_named_value_errors(method, args, gate):
    simulator = CoinToss(3)
    with pytest.raises(ValueError, match=rf"{gate}: expected a non-negative qubit index, got -1"):
        getattr(simulator, method)(*args)


@pytest.mark.parametrize("simulator_class", BATCH_CLASSES)
@pytest.mark.parametrize("params", [{}, {"simulate_gate": True}, {"simulate_gate": False}])
@pytest.mark.parametrize("invalid", [(1, 2), -1, 1.5])
def test_single_qubit_invalid_batch_is_atomic(simulator_class, params, invalid):
    simulator = _prepared_simulator(simulator_class)
    if simulator_class is not PauliProp:
        simulator.run_gate("H", {0})  # Undo preparation: X(0) must change this state.
    before = _snapshot(simulator)
    with pytest.raises(ValueError, match="X: expected a non-negative qubit index or 1-tuple"):
        simulator.run_gate("X", {0, invalid}, **params)
    assert _snapshot(simulator) == before


@pytest.mark.parametrize("simulator_class", FIXED_REGISTER_CLASSES)
@pytest.mark.parametrize("params", [{}, {"simulate_gate": True}, {"simulate_gate": False}])
def test_single_qubit_out_of_range_batch_is_atomic(simulator_class, params):
    simulator = simulator_class(3)
    before = _snapshot(simulator)
    with pytest.raises(ValueError, match="X: qubit 3 is outside the 3-qubit register"):
        simulator.run_gate("X", {0, 3}, **params)
    assert _snapshot(simulator) == before


@pytest.mark.parametrize("simulator_class", BATCH_CLASSES)
@pytest.mark.parametrize("params", [{}, {"simulate_gate": True}])
def test_single_qubit_operands_are_extracted_once(simulator_class, params):
    simulator = _prepared_simulator(simulator_class)
    expected = _prepared_simulator(simulator_class)
    expected.run_gate("X", {0}, **params)
    qubit = _ChangingIndex(0)
    simulator.run_gate("X", {qubit}, **params)
    assert qubit.calls == 1
    assert _snapshot(simulator) == _snapshot(expected)
