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

import copy
import json

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


def test_append_empty_locations_with_params() -> None:
    """Test that params are preserved when appending a gate with empty locations."""
    qc = QuantumCircuit()
    qc.append("cop", set(), a=1, b=2)

    results = list(qc.items())
    assert len(results) == 1
    symbol, locations, params = results[0]
    assert symbol == "cop"
    assert locations == set()
    assert params["a"] == 1
    assert params["b"] == 2


def test_append_empty_locations_no_params() -> None:
    """Test that a gate with empty locations and no params still round-trips."""
    qc = QuantumCircuit()
    qc.append("barrier", set())

    results = list(qc.items())
    assert len(results) == 1
    symbol, locations, params = results[0]
    assert symbol == "barrier"
    assert locations == set()
    assert params == {}


def test_custom_gate_with_arbitrary_params() -> None:
    """Test that arbitrary keyword params are preserved on custom gates with qubits."""
    qc = QuantumCircuit()
    qc.append("my_gate", {0}, foo="bar", count=42)

    results = list(qc.items())
    assert len(results) == 1
    symbol, locations, params = results[0]
    assert symbol == "my_gate"
    assert locations == {0}
    assert params["foo"] == "bar"
    assert params["count"] == 42


def test_known_gate_with_extra_params() -> None:
    """Test that extra keyword params beyond angle are preserved on known gates."""
    qc = QuantumCircuit()
    qc.append("RZ", {0}, angle=0.5, var_output={0: (1, 2)})

    results = list(qc.items())
    assert len(results) == 1
    symbol, _locations, params = results[0]
    assert symbol == "RZ"
    assert params["angle"] == 0.5
    assert params["var_output"] == {0: (1, 2)}


# ---------------------------------------------------------------------------
# Constructor variants
# ---------------------------------------------------------------------------


def test_empty_constructor() -> None:
    """Test default empty constructor."""
    qc = QuantumCircuit()
    assert len(qc) == 0
    assert qc.metadata == {}
    assert qc.qudits == set()
    assert qc.active_qudits == []


def test_constructor_with_num_ticks() -> None:
    """Test constructor with integer creates reserved empty ticks."""
    qc = QuantumCircuit(3)
    assert len(qc) == 3
    # All ticks should be empty
    results = list(qc.items())
    assert results == []


def test_constructor_with_gate_dicts() -> None:
    """Test constructor with list of gate dictionaries."""
    qc = QuantumCircuit([{"H": {0}}, {"CX": {(0, 1)}}, {"measure Z": {0, 1}}])
    assert len(qc) == 3
    results = list(qc.items())
    assert len(results) == 3
    assert results[0][0] == "H"
    assert results[1][0] == "CX"


def test_constructor_with_metadata() -> None:
    """Test constructor with keyword metadata."""
    qc = QuantumCircuit(num_qubits=5, error_free=True)
    assert qc.metadata["num_qubits"] == 5
    assert qc.metadata["error_free"] is True


# ---------------------------------------------------------------------------
# append() -- gate symbol variants
# ---------------------------------------------------------------------------


def test_append_gate_dict() -> None:
    """Test append with gate dictionary (no locations arg)."""
    qc = QuantumCircuit()
    qc.append({"H": {0, 1}, "X": {2}})

    results = list(qc.items())
    symbols = {r[0] for r in results}
    assert symbols == {"H", "X"}


def test_append_string_symbol() -> None:
    """Test append with string symbol and locations."""
    qc = QuantumCircuit()
    qc.append("H", {0, 1, 2})

    results = list(qc.items())
    assert len(results) == 1
    assert results[0][0] == "H"
    assert results[0][1] == {0, 1, 2}


def test_append_two_qubit_gate() -> None:
    """Test append with two-qubit gate locations as tuples."""
    qc = QuantumCircuit()
    qc.append("CX", {(0, 1), (2, 3)})

    results = list(qc.items())
    assert len(results) == 1
    assert results[0][0] == "CX"
    assert results[0][1] == {(0, 1), (2, 3)}


def test_append_prep_and_measure() -> None:
    """Test append with prep and measure gates."""
    qc = QuantumCircuit()
    qc.append("init |0>", {0, 1})
    qc.append("measure Z", {0, 1})

    results = list(qc.items())
    assert len(results) == 2
    assert results[0][0] == "init |0>"
    assert results[1][0] == "measure Z"


def test_append_rotation_gate_with_angle() -> None:
    """Test append with rotation gate using angle param."""
    qc = QuantumCircuit()
    qc.append("RX", {0}, angle=1.57)

    results = list(qc.items())
    assert len(results) == 1
    symbol, locations, params = results[0]
    assert symbol == "RX"
    assert locations == {0}
    assert params["angle"] == 1.57


def test_append_rotation_gate_with_angles_tuple() -> None:
    """Test append with rotation gate using angles tuple param."""
    qc = QuantumCircuit()
    qc.append("RZ", {0, 1}, angles=(0.5,))

    results = list(qc.items())
    assert len(results) == 1
    assert results[0][2]["angles"] == (0.5,)


def test_append_r1xy_gate() -> None:
    """Test R1XY gate with theta and phi angles."""
    qc = QuantumCircuit()
    qc.append("R1XY", {0}, angles=(0.3, 0.7))

    results = list(qc.items())
    assert len(results) == 1
    symbol, _, params = results[0]
    assert symbol == "R1XY"
    assert params["angles"] == (0.3, 0.7)


def test_append_two_qubit_rotation() -> None:
    """Test two-qubit rotation gate with angle."""
    qc = QuantumCircuit()
    qc.append("RZZ", {(0, 1)}, angle=0.25)

    results = list(qc.items())
    assert len(results) == 1
    symbol, locations, params = results[0]
    assert symbol == "RZZ"
    assert (0, 1) in locations
    assert params["angle"] == 0.25


# ---------------------------------------------------------------------------
# append() -- params round-trip
# ---------------------------------------------------------------------------


def test_params_with_qec_metadata() -> None:
    """Test params round-trip with QEC-style metadata (ancilla_ticks, datas, etc.)."""
    qc = QuantumCircuit()
    qc.append(
        "X check",
        set(),
        ancilla_ticks=0,
        data_ticks=[2, 4, 3, 5],
        meas_ticks=7,
        datas=[1, 2, 3, 4],
        ancillas=0,
    )

    results = list(qc.items())
    assert len(results) == 1
    _, _, params = results[0]
    assert params["ancilla_ticks"] == 0
    assert params["data_ticks"] == [2, 4, 3, 5]
    assert params["meas_ticks"] == 7
    assert params["datas"] == [1, 2, 3, 4]
    assert params["ancillas"] == 0


def test_params_with_boolean_values() -> None:
    """Test params with boolean values."""
    qc = QuantumCircuit()
    qc.append("H", {0}, error_free=True, noiseless=False)

    results = list(qc.items())
    _, _, params = results[0]
    assert params["error_free"] is True
    assert params["noiseless"] is False


def test_params_with_string_values() -> None:
    """Test params with string values."""
    qc = QuantumCircuit()
    qc.append("my_gate", {0}, label="ancilla_prep", kind="stabilizer")

    results = list(qc.items())
    _, _, params = results[0]
    assert params["label"] == "ancilla_prep"
    assert params["kind"] == "stabilizer"


def test_params_with_nested_dict() -> None:
    """Test params with nested dictionary values."""
    qc = QuantumCircuit()
    qc.append("H", {0}, config={"depth": 3, "rounds": 10})

    results = list(qc.items())
    _, _, params = results[0]
    assert params["config"] == {"depth": 3, "rounds": 10}


# ---------------------------------------------------------------------------
# update()
# ---------------------------------------------------------------------------


def test_update_at_specific_tick() -> None:
    """Test update adds gates to a specific existing tick."""
    qc = QuantumCircuit()
    qc.append("H", {0})
    qc.append("H", {2})
    qc.update("X", {1}, tick=0)

    results = list(qc.items(tick=0))
    symbols = {r[0] for r in results}
    assert symbols == {"H", "X"}


def test_update_last_tick_default() -> None:
    """Test update defaults to last tick."""
    qc = QuantumCircuit()
    qc.append("H", {0})
    qc.update("X", {1})

    results = list(qc.items(tick=0))
    symbols = {r[0] for r in results}
    assert symbols == {"H", "X"}


def test_update_with_gate_dict() -> None:
    """Test update with gate dictionary form."""
    qc = QuantumCircuit()
    qc.append("H", {0})
    qc.update({"X": {1}, "Z": {2}})

    results = list(qc.items(tick=0))
    symbols = {r[0] for r in results}
    assert symbols == {"H", "X", "Z"}


def test_update_with_params() -> None:
    """Test update passes params through to the gate on a free qubit."""
    qc = QuantumCircuit()
    qc.append("H", {0})
    qc.update("measure Z", {1}, tick=0, forced_outcome=0)

    results = list(qc.items(tick=0))
    measure_results = [r for r in results if r[0] == "measure Z"]
    assert len(measure_results) == 1
    assert measure_results[0][2].get("forced_outcome") == 0


def test_update_emptyappend() -> None:
    """Test update with emptyappend on empty circuit."""
    qc = QuantumCircuit()
    assert len(qc) == 0
    qc.update("H", {0}, emptyappend=True)
    assert len(qc) == 1
    results = list(qc.items())
    assert results[0][0] == "H"


def test_update_negative_tick() -> None:
    """Test update with negative tick index."""
    qc = QuantumCircuit()
    qc.append("H", {0})
    qc.append("X", {1})
    qc.update("Z", {2}, tick=-2)  # Should target tick 0

    results = list(qc.items(tick=0))
    symbols = {r[0] for r in results}
    assert "Z" in symbols


# ---------------------------------------------------------------------------
# discard()
# ---------------------------------------------------------------------------


def test_discard_single_qubit() -> None:
    """Test discard removes a single qubit from a tick."""
    qc = QuantumCircuit()
    qc.append("H", {0, 1, 2})
    qc.discard({1})

    results = list(qc.items(tick=-1))
    assert len(results) == 1
    assert 1 not in results[0][1]


def test_discard_at_specific_tick() -> None:
    """Test discard at a specific tick index."""
    qc = QuantumCircuit()
    qc.append("H", {0, 1})
    qc.append("X", {0, 1})
    qc.discard({0}, tick=0)

    results = list(qc.items(tick=0))
    for _, locations, _ in results:
        assert 0 not in locations

    # Tick 1 should be unaffected
    results = list(qc.items(tick=1))
    all_locs = set()
    for _, locations, _ in results:
        all_locs.update(locations)
    assert 0 in all_locs


# ---------------------------------------------------------------------------
# items() iteration
# ---------------------------------------------------------------------------


def test_items_all_ticks() -> None:
    """Test items() iterates across all ticks."""
    qc = QuantumCircuit()
    qc.append("H", {0})
    qc.append("X", {1})
    qc.append("Z", {2})

    results = list(qc.items())
    assert len(results) == 3
    symbols = [r[0] for r in results]
    assert symbols == ["H", "X", "Z"]


def test_items_specific_tick() -> None:
    """Test items(tick=N) iterates only that tick."""
    qc = QuantumCircuit()
    qc.append("H", {0})
    qc.append("X", {1})

    results = list(qc.items(tick=1))
    assert len(results) == 1
    assert results[0][0] == "X"


def test_items_negative_tick() -> None:
    """Test items(tick=-1) iterates last tick."""
    qc = QuantumCircuit()
    qc.append("H", {0})
    qc.append("X", {1})

    results = list(qc.items(tick=-1))
    assert len(results) == 1
    assert results[0][0] == "X"


def test_items_yields_symbol_locations_params() -> None:
    """Test that items() yields (symbol, locations, params) tuples."""
    qc = QuantumCircuit()
    qc.append("RZ", {0}, angle=0.5)

    for symbol, locations, params in qc.items():
        assert isinstance(symbol, str)
        assert isinstance(locations, set)
        assert isinstance(params, dict)


def test_items_same_symbol_same_params_merged() -> None:
    """Test that gates with same symbol and params in the same tick have locations merged."""
    qc = QuantumCircuit()
    qc.append({"H": {0, 1, 2}})

    results = list(qc.items())
    assert len(results) == 1
    assert results[0][1] == {0, 1, 2}


def test_items_multiple_gate_types_in_tick() -> None:
    """Test items with multiple gate types in same tick via gate dict."""
    qc = QuantumCircuit()
    qc.append({"H": {0}, "X": {1}, "Z": {2}})

    results = list(qc.items())
    assert len(results) == 3
    symbols = {r[0] for r in results}
    assert symbols == {"H", "X", "Z"}


# ---------------------------------------------------------------------------
# iter_ticks()
# ---------------------------------------------------------------------------


def test_iter_ticks_yields_tick_view() -> None:
    """Test iter_ticks yields (TickView, tick_index, metadata)."""
    qc = QuantumCircuit(num_qubits=2)
    qc.append("H", {0})
    qc.append("CX", {(0, 1)})

    ticks = list(qc.iter_ticks())
    assert len(ticks) == 2
    for tick_view, tick_idx, meta in ticks:
        assert isinstance(tick_idx, int)
        assert meta == qc.metadata
        # TickView should support items()
        results = list(tick_view.items())
        assert len(results) >= 1


def test_iter_ticks_metadata_is_circuit_metadata() -> None:
    """Test that iter_ticks yields the circuit-level metadata for every tick."""
    qc = QuantumCircuit(label="test_circuit")
    qc.append("H", {0})
    qc.append("X", {1})

    for _, _, meta in qc.iter_ticks():
        assert meta["label"] == "test_circuit"


# ---------------------------------------------------------------------------
# Indexing: __getitem__, __setitem__, __delitem__
# ---------------------------------------------------------------------------


def test_getitem_returns_tick_view() -> None:
    """Test qc[i] returns a TickView for that tick."""
    qc = QuantumCircuit()
    qc.append("H", {0})
    qc.append("X", {1})

    tick0 = qc[0]
    results = list(tick0.items())
    assert results[0][0] == "H"

    tick1 = qc[1]
    results = list(tick1.items())
    assert results[0][0] == "X"


def test_getitem_negative_index() -> None:
    """Test qc[-1] returns last tick."""
    qc = QuantumCircuit()
    qc.append("H", {0})
    qc.append("X", {1})

    tick = qc[-1]
    results = list(tick.items())
    assert results[0][0] == "X"


def test_delitem_clears_tick() -> None:
    """Test del qc[i] clears the tick."""
    qc = QuantumCircuit()
    qc.append("H", {0})
    qc.append("X", {1})
    del qc[0]

    results = list(qc.items(tick=0))
    assert results == []
    # Length should remain the same (tick is cleared, not removed)
    assert len(qc) == 2


# ---------------------------------------------------------------------------
# __len__, __iter__, __str__
# ---------------------------------------------------------------------------


def test_len() -> None:
    """Test len(qc) returns number of ticks."""
    qc = QuantumCircuit()
    assert len(qc) == 0
    qc.append("H", {0})
    assert len(qc) == 1
    qc.append("X", {1})
    assert len(qc) == 2


def test_iter() -> None:
    """Test iterating over qc yields same as items()."""
    qc = QuantumCircuit()
    qc.append("H", {0})
    qc.append("X", {1})

    from_iter = list(qc)
    from_items = list(qc.items())
    assert len(from_iter) == len(from_items)
    for a, b in zip(from_iter, from_items, strict=False):
        assert a[0] == b[0]


def test_str_representation() -> None:
    """Test string representation includes gate info."""
    qc = QuantumCircuit()
    qc.append("H", {0})
    s = str(qc)
    assert "H" in s
    assert "QuantumCircuit" in s


def test_str_with_metadata() -> None:
    """Test string representation includes metadata when present."""
    qc = QuantumCircuit(label="test")
    qc.append("H", {0})
    s = str(qc)
    assert "label" in s


# ---------------------------------------------------------------------------
# add_ticks()
# ---------------------------------------------------------------------------


def test_add_ticks() -> None:
    """Test add_ticks creates empty ticks."""
    qc = QuantumCircuit()
    qc.append("H", {0})
    qc.add_ticks(2)

    assert len(qc) == 3
    # Empty ticks should yield nothing
    results = list(qc.items(tick=1))
    assert results == []
    results = list(qc.items(tick=2))
    assert results == []


# ---------------------------------------------------------------------------
# qudits and active_qudits
# ---------------------------------------------------------------------------


def test_qudits_tracks_all_used() -> None:
    """Test qudits property returns all qubits ever used."""
    qc = QuantumCircuit()
    qc.append("H", {0, 1})
    qc.append("CX", {(2, 3)})

    assert qc.qudits == {0, 1, 2, 3}


def test_active_qudits_per_tick() -> None:
    """Test active_qudits returns list of sets, one per tick."""
    qc = QuantumCircuit()
    qc.append("H", {0, 1})
    qc.append("CX", {(2, 3)})

    active = qc.active_qudits
    assert len(active) == 2
    assert active[0] == {0, 1}
    assert active[1] == {2, 3}


# ---------------------------------------------------------------------------
# copy()
# ---------------------------------------------------------------------------


def test_copy_preserves_gates() -> None:
    """Test copy preserves all gates and params."""
    qc = QuantumCircuit()
    qc.append("H", {0, 1})
    qc.append("RZ", {0}, angle=0.5)
    qc.append("CX", {(0, 1)})

    qc2 = qc.copy()
    assert len(qc2) == len(qc)

    orig = list(qc.items())
    copied = list(qc2.items())
    assert len(orig) == len(copied)
    for o, c in zip(orig, copied, strict=False):
        assert o[0] == c[0]  # symbol
        assert o[1] == c[1]  # locations
        assert o[2] == c[2]  # params


def test_copy_is_independent() -> None:
    """Test that modifying a copy does not affect the original."""
    qc = QuantumCircuit()
    qc.append("H", {0})

    qc2 = qc.copy()
    qc2.append("X", {1})

    assert len(qc) == 1
    assert len(qc2) == 2


def test_copy_preserves_metadata() -> None:
    """Test copy preserves circuit metadata."""
    qc = QuantumCircuit(label="original")
    qc.append("H", {0})

    qc2 = qc.copy()
    assert qc2.metadata["label"] == "original"


def test_copy_module() -> None:
    """Test copy.copy works on QuantumCircuit."""
    qc = QuantumCircuit()
    qc.append("H", {0})
    qc.append("RZ", {0}, angle=0.5)

    qc2 = copy.copy(qc)
    assert len(qc2) == len(qc)
    assert next(iter(qc.items()))[0] == next(iter(qc2.items()))[0]


# ---------------------------------------------------------------------------
# JSON round-trip
# ---------------------------------------------------------------------------


def test_json_roundtrip_basic() -> None:
    """Test to_json_str / from_json_str round-trip."""
    qc = QuantumCircuit()
    qc.append("H", {0, 1})
    qc.append("CX", {(0, 1)})
    qc.append("measure Z", {0, 1})

    json_str = qc.to_json_str()
    qc2 = QuantumCircuit.from_json_str(json_str)

    assert len(qc2) == len(qc)
    orig = list(qc.items())
    restored = list(qc2.items())
    for o, r in zip(orig, restored, strict=False):
        assert o[0] == r[0]


def test_json_roundtrip_with_params() -> None:
    """Test JSON round-trip preserves gate params."""
    qc = QuantumCircuit()
    qc.append("RZ", {0}, angle=0.5)
    qc.append("my_gate", {1}, custom_param="hello", count=42)

    json_str = qc.to_json_str()
    qc2 = QuantumCircuit.from_json_str(json_str)

    results = list(qc2.items())
    rz_params = results[0][2]
    assert rz_params["angle"] == 0.5

    custom_params = results[1][2]
    assert custom_params["custom_param"] == "hello"
    assert custom_params["count"] == 42


def test_json_roundtrip_with_metadata() -> None:
    """Test JSON round-trip preserves circuit metadata."""
    qc = QuantumCircuit(label="test", num_qubits=5)
    qc.append("H", {0})

    json_str = qc.to_json_str()
    qc2 = QuantumCircuit.from_json_str(json_str)

    assert qc2.metadata["label"] == "test"
    assert qc2.metadata["num_qubits"] == 5


def test_json_roundtrip_var_output() -> None:
    """Test JSON round-trip preserves var_output with int keys and tuple values."""
    qc = QuantumCircuit()
    qc.append("measure Z", {0}, var_output={0: (1, 2)})

    json_str = qc.to_json_str()
    qc2 = QuantumCircuit.from_json_str(json_str)

    results = list(qc2.items())
    assert results[0][2]["var_output"] == {0: (1, 2)}


def test_json_str_is_valid_json() -> None:
    """Test to_json_str produces valid JSON."""
    qc = QuantumCircuit(label="test")
    qc.append("H", {0})

    json_str = qc.to_json_str()
    parsed = json.loads(json_str)
    assert parsed["prog_type"] == "PECOS.QuantumCircuit"
    assert "gates" in parsed


# ---------------------------------------------------------------------------
# TickView API
# ---------------------------------------------------------------------------


def test_tick_view_add() -> None:
    """Test TickView.add() method adds gates."""
    qc = QuantumCircuit()
    qc.append("H", {0})

    tick = qc[0]
    tick.add("X", {1})

    results = list(qc.items(tick=0))
    symbols = {r[0] for r in results}
    assert symbols == {"H", "X"}


def test_tick_view_discard() -> None:
    """Test TickView.discard() method removes locations."""
    qc = QuantumCircuit()
    qc.append("H", {0, 1, 2})

    tick = qc[0]
    tick.discard({1})

    results = list(qc.items(tick=0))
    assert 1 not in results[0][1]


def test_tick_view_active_qudits() -> None:
    """Test TickView.active_qudits property."""
    qc = QuantumCircuit()
    qc.append({"H": {0}, "CX": {(1, 2)}})

    tick = qc[0]
    assert tick.active_qudits == {0, 1, 2} or tick.active_qudits == {0, (1, 2)}


def test_tick_view_metadata() -> None:
    """Test TickView.metadata returns circuit metadata."""
    qc = QuantumCircuit(label="test")
    qc.append("H", {0})

    tick = qc[0]
    assert tick.metadata["label"] == "test"


def test_tick_view_str() -> None:
    """Test TickView string representation."""
    qc = QuantumCircuit()
    qc.append("H", {0})

    tick = qc[0]
    s = str(tick)
    assert "H" in s


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_multiple_appends_increment_ticks() -> None:
    """Test each append creates a new tick."""
    qc = QuantumCircuit()
    for i in range(5):
        qc.append("H", {i})
    assert len(qc) == 5


def test_empty_gate_dict_append() -> None:
    """Test appending an empty gate dict."""
    qc = QuantumCircuit()
    qc.append({})
    # Empty tick should exist but yield nothing
    assert len(qc) == 0 or list(qc.items()) == []


def test_gate_symbol_case_preserved() -> None:
    """Test that the original gate symbol case is preserved in round-trip."""
    qc = QuantumCircuit()
    qc.append("init |0>", {0})
    qc.append("measure Z", {0})

    results = list(qc.items())
    assert results[0][0] == "init |0>"
    assert results[1][0] == "measure Z"


def test_swap_gate() -> None:
    """Test SWAP gate round-trips correctly."""
    qc = QuantumCircuit()
    qc.append("SWAP", {(0, 1)})

    results = list(qc.items())
    assert len(results) == 1
    assert results[0][0] == "SWAP"
    assert (0, 1) in results[0][1]


def test_r2xxyyzz_gate() -> None:
    """Test R2XXYYZZ gate preserves all three angles."""
    qc = QuantumCircuit()
    qc.append("R2XXYYZZ", {(0, 1)}, angles=(0.1, 0.2, 0.3))

    results = list(qc.items())
    assert len(results) == 1
    symbol, _, params = results[0]
    assert symbol == "R2XXYYZZ"
    assert len(params["angles"]) == 3


def test_u_gate() -> None:
    """Test U gate with three angle parameters."""
    qc = QuantumCircuit()
    qc.append("U", {0}, angles=(0.1, 0.2, 0.3))

    results = list(qc.items())
    assert len(results) == 1
    assert results[0][0] == "U"
    assert results[0][2]["angles"] == (0.1, 0.2, 0.3)
