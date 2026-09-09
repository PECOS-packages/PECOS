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

"""Executable tests for controlled-rotation circuit-builder boundaries."""

import math

import pytest
from pecos.simulators import StateVec
from pecos_rslib.quantum import QubitConflictError, TickCircuit


def _execute(circuit: TickCircuit) -> list[complex]:
    simulator = StateVec(2)
    for tick_index in range(circuit.num_ticks()):
        for gate in circuit.get_tick(tick_index).gate_batches():
            name = gate.gate_type.name
            qubits = list(gate.qubits)
            angles = list(gate.angles)
            if name in {"X", "RZ"}:
                params = {"angle": angles[0]} if name == "RZ" else None
                for qubit in qubits:
                    simulator.backend.run_1q_gate(name, qubit, params)
            elif name == "RZZ":
                for offset in range(0, len(qubits), 2):
                    simulator.backend.run_2q_gate(
                        name,
                        (qubits[offset], qubits[offset + 1]),
                        {"angle": angles[0]},
                    )
            else:
                msg = f"unexpected builder gate {name}"
                raise AssertionError(msg)
    return [complex(value) for value in simulator.backend.vector]


def test_py_tick_handle_crz_preserves_full_matrix() -> None:
    for theta in (-math.pi, math.pi / 3, math.pi, math.tau, 3 * math.pi):
        columns: list[list[complex]] = []
        for basis in range(4):
            circuit = TickCircuit()
            if basis & 1:
                circuit.tick().x([0])
            if basis & 2:
                circuit.tick().x([1])
            circuit.tick().crz(theta, [(1, 0)])
            columns.append(_execute(circuit))

        half = theta / 2
        reference = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, complex(math.cos(half), -math.sin(half)), 0],
            [0, 0, 0, complex(math.cos(half), math.sin(half))],
        ]
        phase = columns[0][0] / reference[0][0]
        assert abs(abs(phase) - 1) < 1e-12
        if theta in {-math.pi, math.pi / 3, math.pi}:
            assert abs(phase - 1) < 1e-12
        else:
            assert min(abs(phase - 1), abs(phase + 1)) < 1e-12
        for column in range(4):
            for row in range(4):
                assert abs(columns[column][row] / phase - reference[row][column]) < 1e-12


def test_py_tick_handle_crz_conflict_is_atomic() -> None:
    circuit = TickCircuit()
    circuit.tick()
    circuit.tick()
    circuit.tick_at(1).h([1])

    with pytest.raises(QubitConflictError):
        circuit.tick_at(0).crz(math.pi / 3, [(0, 1)])

    assert list(circuit.get_tick(0).gate_batches()) == []
    assert [gate.gate_type.name for gate in circuit.get_tick(1).gate_batches()] == ["H"]


def test_py_tick_handle_empty_crz_preserves_metadata_target() -> None:
    circuit = TickCircuit()
    tick = circuit.tick()
    tick.h([0])
    tick.crz(math.pi / 3, [])
    tick.meta("tag", "empty-pairs")

    stored_tick = circuit.get_tick(0)
    assert stored_tick.get_gate_attr(0, "tag") == "empty-pairs"
    assert stored_tick.get_attr("tag") is None


def test_py_tick_handle_crz_metadata_targets_rzz() -> None:
    circuit = TickCircuit()
    circuit.tick().crz(math.pi / 3, [(0, 1)]).meta("tag", "lowered-crz")

    rzz_tick = circuit.get_tick(0)
    rz_tick = circuit.get_tick(1)
    assert [gate.gate_type.name for gate in rzz_tick.gate_batches()] == ["RZZ"]
    assert [gate.gate_type.name for gate in rz_tick.gate_batches()] == ["RZ"]
    assert rzz_tick.get_gate_attr(0, "tag") == "lowered-crz"
    assert rzz_tick.get_attr("tag") is None
    assert rz_tick.get_gate_attr(0, "tag") is None
    assert rz_tick.get_attr("tag") is None
