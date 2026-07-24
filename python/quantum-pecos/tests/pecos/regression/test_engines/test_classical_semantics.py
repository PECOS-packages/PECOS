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

"""Tests for configurable legacy-engine classical semantics."""

from __future__ import annotations

import pecos as pc
import pytest
from pecos import BitInt, BitUInt
from pecos.engines.cvm import (
    DefaultClassicalSemantics,
    UnsignedClassicalSemantics,
)
from pecos.simulators import SparseStab


@pytest.mark.parametrize("width", [8, 16, 32, 64])
def test_unsigned_arithmetic_is_fixed_width(width: int) -> None:
    """Unsigned operations use logical shifts, masked shifts, and wrapping arithmetic."""
    semantics = UnsignedClassicalSemantics(width)
    all_ones = (1 << width) - 1

    assert int(semantics.eval_op(">>", all_ones, 1)) == all_ones >> 1
    assert int(semantics.eval_op("<<", 1, width)) == 1
    assert int(semantics.eval_op(">>", 0b100, width + 1)) == 0b10
    assert int(semantics.eval_op("+", all_ones, 1)) == 0
    assert int(semantics.eval_op("-", 0, 1)) == all_ones
    assert int(semantics.eval_op("*", 1 << (width // 2), 1 << (width // 2))) == 0
    assert int(semantics.eval_op("/", 5, 0)) == all_ones
    assert int(semantics.eval_op("%", 5, 0)) == all_ones


def test_unsigned_division_and_bitwise_operations() -> None:
    """High-bit words remain unsigned throughout ALU operations."""
    semantics = UnsignedClassicalSemantics(64)

    assert int(semantics.eval_op("/", -2, 2)) == 0x7FFFFFFFFFFFFFFF
    assert int(semantics.eval_op("~", 0)) == 0xFFFFFFFFFFFFFFFF
    assert int(semantics.eval_op("&", 0b1100, 0b1010)) == 0b1000
    assert int(semantics.eval_op("|", 0b1100, 0b1010)) == 0b1110
    assert int(semantics.eval_op("^", 0b1100, 0b1010)) == 0b0110


def test_unsigned_comparisons_can_select_signedness() -> None:
    """The same word compares differently under signed and unsigned conditions."""
    semantics = UnsignedClassicalSemantics(64)

    assert int(semantics.eval_op("<", -2, 0)) == 0
    assert int(semantics.eval_op("<", -2, 0, signed=True)) == 1
    assert int(semantics.eval_op(">", -2, 0)) == 1
    assert int(semantics.eval_op(">", -2, 0, signed=True)) == 0
    assert int(semantics.eval_op("==", -2, -2)) == 1
    assert int(semantics.eval_op("!=", -2, 3)) == 1


def test_unsigned_instance_width_is_reusable_outside_hybrid_engine() -> None:
    """Each semantics object retains an isolated default width."""
    semantics_8 = UnsignedClassicalSemantics(8)
    semantics_64 = UnsignedClassicalSemantics(64)
    condition = {"a": "x", "op": "<", "b": 256}

    assert int(semantics_8.eval_op("/", 5, 0)) == 0xFF
    assert int(semantics_64.eval_op("/", 5, 0)) == 0xFFFFFFFFFFFFFFFF
    assert semantics_8.eval_condition(condition, {"x": BitInt(8, 1)}) is False
    assert semantics_64.eval_condition(condition, {"x": BitInt(64, 1)}) is True


def test_unsigned_storage_uses_bituint_for_configured_variable_types() -> None:
    """Variable metadata controls zero-extending versus signed storage."""

    class State:
        num_qubits = 1

    circuit = pc.QuantumCircuit(
        cvar_spec={"integer": 64, "narrow": 8, "flag": 1},
        cvar_spec_type={"integer": "cint64", "narrow": "ubit", "flag": "cbool"},
    )
    output = UnsignedClassicalSemantics(
        unsigned_cvar_types={"ubit"},
    ).set_output(State(), circuit, None, None)

    assert isinstance(output["integer"], BitInt)
    assert isinstance(output["narrow"], BitUInt)
    assert isinstance(output["flag"], BitInt)


def test_hybrid_engine_accepts_unsigned_semantics_without_global_patching() -> None:
    """A semantics object controls one engine without changing PECOS defaults."""
    circuit = pc.QuantumCircuit(
        cvar_spec={"x": 64, "shifted": 64, "signed": 64, "unsigned": 64},
        cvar_spec_type={
            "x": "cint64",
            "shifted": "cint64",
            "signed": "cint64",
            "unsigned": "cint64",
        },
        num_qubits=1,
    )
    circuit.append("cop", set(), expr={"t": "x", "op": "=", "a": -2})
    circuit.append("cop", set(), expr={"t": "shifted", "op": ">>", "a": "x", "b": 1})
    circuit.append(
        "cop",
        set(),
        expr={"t": "signed", "op": "=", "a": 1},
        cond={"a": "x", "op": "<", "b": 0, "signed": True},
    )
    circuit.append(
        "cop",
        set(),
        expr={"t": "unsigned", "op": "=", "a": 1},
        cond={"a": "x", "op": "<", "b": 0},
    )

    engine = pc.HybridEngine(
        seed=1,
        regwidth=64,
        classical_semantics=UnsignedClassicalSemantics(64),
    )
    output, _ = engine.run(SparseStab(1), circuit, shot_id=0)

    assert int(output["x"]) == -2
    assert int(output["shifted"]) == 0x7FFFFFFFFFFFFFFF
    assert int(output["signed"]) == 1
    assert int(output["unsigned"]) == 0

    # A normal engine still selects the original signed behavior for the same
    # circuit: the right shift is arithmetic and both comparisons see -2.
    default_engine = pc.HybridEngine(seed=1, regwidth=64)
    default_output, _ = default_engine.run(SparseStab(1), circuit, shot_id=0)
    assert isinstance(default_engine.classical_semantics, DefaultClassicalSemantics)
    assert int(default_output["shifted"]) == -1
    assert int(default_output["signed"]) == 1
    assert int(default_output["unsigned"]) == 1


def test_hybrid_engine_masks_and_zero_extends_narrow_bit_variables() -> None:
    """Narrow unsigned variables truncate on store and zero-extend on reads."""
    circuit = pc.QuantumCircuit(
        cvar_spec={"narrow": 8, "extended": 64},
        cvar_spec_type={"narrow": "cbitvar", "extended": "cint64"},
        num_qubits=1,
    )
    circuit.append("cop", set(), expr={"t": "narrow", "op": "=", "a": 0xFF})
    circuit.append("cop", set(), expr={"t": "extended", "op": "+", "a": "narrow", "b": 0})
    circuit.append("cop", set(), expr={"t": "narrow", "op": "+", "a": "narrow", "b": 1})

    engine = pc.HybridEngine(
        seed=1,
        regwidth=64,
        classical_semantics=UnsignedClassicalSemantics(64),
    )
    output, _ = engine.run(SparseStab(1), circuit, shot_id=0)

    assert int(output["extended"]) == 0xFF
    assert int(output["narrow"]) == 0


@pytest.mark.parametrize("width", [0, 3, 12, 48, -8])
def test_unsigned_semantics_rejects_non_power_of_two_widths(width: int) -> None:
    """Shift masking requires a positive power-of-two word width."""
    with pytest.raises(ValueError, match="positive power of two"):
        UnsignedClassicalSemantics(width)


def test_unsigned_semantics_rejects_mismatched_explicit_width() -> None:
    """One semantics object cannot silently evaluate at multiple widths."""
    semantics = UnsignedClassicalSemantics(64)

    with pytest.raises(ValueError, match="does not match"):
        semantics.eval_op("+", 1, 2, width=32)


def test_hybrid_engine_default_width_matches_unsigned_semantics_default() -> None:
    """The simplest engine-policy construction uses one consistent width."""
    semantics = UnsignedClassicalSemantics()
    engine = pc.HybridEngine(classical_semantics=semantics)

    assert engine.regwidth == semantics.width == 32


def test_hybrid_engine_preserves_falsy_custom_semantics() -> None:
    """Strategy selection distinguishes an explicit object from None."""

    class FalsySemantics(DefaultClassicalSemantics):
        def __bool__(self) -> bool:
            return False

    semantics = FalsySemantics()
    engine = pc.HybridEngine(classical_semantics=semantics)

    assert engine.classical_semantics is semantics


def test_unsigned_set_output_accepts_none_cvar_spec() -> None:
    """Nullable circuit metadata retains the scratch-register behavior."""

    class State:
        num_qubits = 2

    circuit = pc.QuantumCircuit(cvar_spec=None)
    output = UnsignedClassicalSemantics().set_output(
        State(),
        circuit,
        None,
        None,
    )

    assert set(output) == {"__pecos_scratch"}
    assert output["__pecos_scratch"].size == 2
