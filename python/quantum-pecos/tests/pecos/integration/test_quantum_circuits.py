# Copyright 2018 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the terms of Contract
# DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Integration tests for quantum circuit operations."""
from __future__ import annotations

from pecos.circuits import QuantumCircuit


def test_quantum_circuits() -> None:
    """Test quantum circuit operations and validation."""
    # Check the method append with check_overlap == True
    # ---------------------------------------------------
    qc = QuantumCircuit()

    assert len(qc) == 0
    assert qc.active_qudits == []

    qc.append({"int |0>": {0, 1}})

    assert len(qc) == 1
    assert qc.active_qudits == [{0, 1}]

    qc.append({"H": {0}})

    assert len(qc) == 2
    assert qc.active_qudits == [{0, 1}, {0}]

    qc.append({"CNOT": {(0, 1)}})

    assert len(qc) == 3
    assert qc.active_qudits == [{0, 1}, {0}, {0, 1}]

    qc.append({"measure Z": {0}})

    assert len(qc) == 4
    assert qc.active_qudits == [{0, 1}, {0}, {0, 1}, {0}]

    # Check update, add, and discard with check_overlap == True
    # ---------------------------------------------------------

    qc.update({"X": {1}}, tick=1)

    assert len(qc) == 4
    assert qc.active_qudits == [{0, 1}, {0, 1}, {0, 1}, {0}]

    qc.update({"measure Z": {1}})

    assert len(qc) == 4
    assert qc.active_qudits == [{0, 1}, {0, 1}, {0, 1}, {0, 1}]

    qc.discard({1})

    assert len(qc) == 4
    assert qc.active_qudits == [{0, 1}, {0, 1}, {0, 1}, {0}]

    qc.update("X", {1})

    assert len(qc) == 4
    assert qc.active_qudits == [{0, 1}, {0, 1}, {0, 1}, {0, 1}]

    # Check the method append with check_overlap == False
    # ---------------------------------------------------
    qc = QuantumCircuit()

    assert len(qc) == 0
    assert qc.active_qudits == []

    qc.append({"int |0>": {0, 1}})

    assert len(qc) == 1
    assert qc.active_qudits == [{0, 1}]

    qc.append({"H": {0}})

    assert len(qc) == 2
    assert qc.active_qudits == [{0, 1}, {0}]

    # Check update with check_overlap == False
    # -----------------------------------------

    qc.update({"X": {1}}, tick=1)

    assert len(qc) == 2
    assert qc.active_qudits == [{0, 1}, {0, 1}]


def test_tick_view_symbols() -> None:
    """Test TickView.symbols property returns correct gate groupings."""
    qc = QuantumCircuit()
    qc.append("H", {0, 1, 2})
    qc.append({"CX": {(0, 1), (2, 3)}, "H": {4}})
    qc.append("MZ", {0, 1, 2, 3, 4})

    # Tick 0: single gate type
    tick0 = qc[0]
    assert "H" in tick0.symbols
    assert len(tick0.symbols["H"]) == 1
    gate = tick0.symbols["H"][0]
    assert gate.symbol == "H"
    assert gate.locations == {0, 1, 2}
    assert gate.params == {}

    # Tick 1: multiple gate types
    tick1 = qc[1]
    assert set(tick1.symbols.keys()) == {"CX", "H"}
    assert len(tick1.symbols["CX"]) == 1
    assert tick1.symbols["CX"][0].locations == {(0, 1), (2, 3)}
    assert len(tick1.symbols["H"]) == 1
    assert tick1.symbols["H"][0].locations == {4}

    # Tick 2: measurement
    tick2 = qc[2]
    assert "MZ" in tick2.symbols
    assert tick2.symbols["MZ"][0].locations == {0, 1, 2, 3, 4}

    # Non-existent symbol returns KeyError
    assert "X" not in tick0.symbols


def test_tick_view_symbols_with_params() -> None:
    """Test TickView.symbols preserves gate parameters."""
    qc = QuantumCircuit()
    qc.append("RZ", {0, 1}, angles=(0.5,))
    qc.append("RZ", {2}, angles=(0.7,))

    # Same symbol but different params should produce separate Gate entries
    tick0 = qc[0]
    rz_gates = tick0.symbols.get("RZ", [])
    assert len(rz_gates) == 1
    assert rz_gates[0].params["angles"] == (0.5,)

    tick1 = qc[1]
    rz_gates = tick1.symbols.get("RZ", [])
    assert len(rz_gates) == 1
    assert rz_gates[0].params["angles"] == (0.7,)


def test_tick_view_symbols_empty_tick() -> None:
    """Test TickView.symbols on a circuit with an empty tick."""
    qc = QuantumCircuit()
    qc.append("H", {0})
    qc.add_ticks(1)  # empty tick
    qc.append("X", {0})

    tick1 = qc[1]
    assert tick1.symbols == {}


def test_tick_view_symbols_via_iter_ticks() -> None:
    """Test TickView.symbols works when accessed via iter_ticks."""
    qc = QuantumCircuit()
    qc.append("H", {0})
    qc.append("CX", {(0, 1)})

    symbols_per_tick = []
    for tick_view, _tick_idx, _meta in qc.iter_ticks():
        symbols_per_tick.append(list(tick_view.symbols.keys()))

    assert symbols_per_tick == [["H"], ["CX"]]
