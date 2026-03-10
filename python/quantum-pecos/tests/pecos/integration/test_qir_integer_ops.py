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

"""Tests translating QIR integer arithmetic programs to QuantumCircuit + HybridEngine.

These tests translate the logic from ArithmeticOps.Targeted.ll and
IntegerSupport.TargetedAlt.ll into QuantumCircuit objects and verify
the expected deterministic results using the old HybridEngine.

Also tests CVM edge cases: zero operands, negative value round-trips
through BitUInt storage, and the signed arithmetic pipeline.
"""

from __future__ import annotations

import pecos as pc
from pecos.engines.cvm.classical import eval_cop, eval_op, get_val
from pecos.simulators import SparseSim


def test_arithmetic_ops() -> None:
    """Translate ArithmeticOps.Targeted.ll and verify expected results.

    The QIR program:
    - 5 qubits, all X'd to |1>
    - Each qubit: measure (gives 1), reset, conditional X (restores to |1>)
    - Classical computation: for each measurement result == 1:
        count += 1, countPos += 5, countNeg -= 2, countMul *= 3
    - Initial values: count=0, countPos=0, countNeg=10, countMul=1
    - All 5 measurements are 1, so:
        count=5, countPos=25, countNeg=0, countMul=243
    """
    qc = pc.QuantumCircuit(
        cvar_spec={"r": 5, "count": 64, "countPos": 64, "countNeg": 64, "countMul": 64},
        num_qubits=5,
    )

    # X all 5 qubits to |1>
    qc.append("X", {0, 1, 2, 3, 4})

    # Measure-reset-conditional-X for each qubit
    for i in range(5):
        qc.append("measure Z", {i}, var=("r", i))
        qc.append("init |0>", {i})
        qc.append("X", {i}, cond={"a": ("r", i), "op": "==", "b": 1})

    # Initialize countNeg=10, countMul=1 (count and countPos default to 0)
    qc.append("cop", set(), expr={"t": "countNeg", "op": "=", "a": 10})
    qc.append("cop", set(), expr={"t": "countMul", "op": "=", "a": 1})

    # Conditional arithmetic based on each measurement result
    for i in range(5):
        cond = {"a": ("r", i), "op": "==", "b": 1}
        qc.append("cop", set(), expr={"t": "count", "op": "+", "a": "count", "b": 1}, cond=cond)
        qc.append("cop", set(), expr={"t": "countPos", "op": "+", "a": "countPos", "b": 5}, cond=cond)
        qc.append("cop", set(), expr={"t": "countNeg", "op": "-", "a": "countNeg", "b": 2}, cond=cond)
        qc.append("cop", set(), expr={"t": "countMul", "op": "*", "a": "countMul", "b": 3}, cond=cond)

    state = SparseSim(5)
    eng = pc.HybridEngine()
    output, _ = eng.run(state, qc, shot_id=0)

    # All measurements should be 1
    for i in range(5):
        assert output["r"][i] == 1, f"Expected r[{i}]=1, got {output['r'][i]}"

    # All results are unsigned (positive), so direct int() works
    assert int(output["count"]) == 5
    assert int(output["countPos"]) == 25
    assert int(output["countNeg"]) == 0
    assert int(output["countMul"]) == 243


def test_integer_support() -> None:
    """Translate IntegerSupport.TargetedAlt.ll and verify expected results.

    The QIR program uses 1 qubit with 5 measurement rounds.
    Each round: Rx(pi) (equiv. to X for Z-basis), measure, reset,
    conditional Rx(pi) (if result==1).

    Due to the alternating X-measure-reset-conditional-X pattern on
    a single qubit, measurements alternate: 1, 0, 1, 0, 1.

    Classical computation:
    - sum: count of 1-results = 3
    - sub: starts at 0, subtract 2 for each 1-result = -6
    - negMul: sub * 3 = -18
    - dblNegMul: sub * -3 = 18
    """
    qc = pc.QuantumCircuit(
        cvar_spec={"r": 5, "sum": 64, "sub": 64, "negMul": 64, "dblNegMul": 64},
        num_qubits=1,
    )

    # 5 rounds of: X, measure, reset, conditional-X
    for i in range(5):
        qc.append("X", {0})
        qc.append("measure Z", {0}, var=("r", i))
        qc.append("init |0>", {0})
        qc.append("X", {0}, cond={"a": ("r", i), "op": "==", "b": 1})

    # Classical: sum += 1 and sub -= 2 for each result that is 1
    for i in range(5):
        cond = {"a": ("r", i), "op": "==", "b": 1}
        qc.append("cop", set(), expr={"t": "sum", "op": "+", "a": "sum", "b": 1}, cond=cond)
        qc.append("cop", set(), expr={"t": "sub", "op": "-", "a": "sub", "b": 2}, cond=cond)

    # Final multiplications (unconditional)
    qc.append("cop", set(), expr={"t": "negMul", "op": "*", "a": "sub", "b": 3})
    qc.append("cop", set(), expr={"t": "dblNegMul", "op": "*", "a": "sub", "b": -3})

    state = SparseSim(1)
    eng = pc.HybridEngine()
    output, _ = eng.run(state, qc, shot_id=0)

    # Measurements alternate: 1, 0, 1, 0, 1
    expected_results = [1, 0, 1, 0, 1]
    for i, expected in enumerate(expected_results):
        assert output["r"][i] == expected, f"Expected r[{i}]={expected}, got {output['r'][i]}"

    # sum = 3 (count of 1-results)
    assert int(output["sum"]) == 3

    # Signed values are directly available from BitInt(64) storage
    assert int(output["sub"]) == -6
    assert int(output["negMul"]) == -18
    assert int(output["dblNegMul"]) == 18


def test_multiply_by_zero() -> None:
    """Verify that multiplying by literal zero produces zero.

    This exercises the fix for the truthiness bug in recur_eval_op
    where `if b:` would skip the operation when b=0.
    """
    qc = pc.QuantumCircuit(
        cvar_spec={"x": 64},
        num_qubits=1,
    )
    # Set x = 42, then x = x * 0
    qc.append("cop", set(), expr={"t": "x", "op": "=", "a": 42})
    qc.append("cop", set(), expr={"t": "x", "op": "*", "a": "x", "b": 0})

    state = SparseSim(1)
    eng = pc.HybridEngine()
    output, _ = eng.run(state, qc, shot_id=0)

    assert int(output["x"]) == 0


def test_add_zero() -> None:
    """Verify that adding literal zero is a no-op.

    Same truthiness bug: `if b:` with b=0 would skip the addition entirely,
    which happens to give the right result for addition but for the wrong reason.
    After the fix, the addition is actually evaluated.
    """
    qc = pc.QuantumCircuit(
        cvar_spec={"x": 64},
        num_qubits=1,
    )
    qc.append("cop", set(), expr={"t": "x", "op": "=", "a": 7})
    qc.append("cop", set(), expr={"t": "x", "op": "+", "a": "x", "b": 0})

    state = SparseSim(1)
    eng = pc.HybridEngine()
    output, _ = eng.run(state, qc, shot_id=0)

    assert int(output["x"]) == 7


def test_bitwise_and_with_zero() -> None:
    """Verify that bitwise AND with zero produces zero."""
    qc = pc.QuantumCircuit(
        cvar_spec={"x": 64},
        num_qubits=1,
    )
    qc.append("cop", set(), expr={"t": "x", "op": "=", "a": 0xFF})
    qc.append("cop", set(), expr={"t": "x", "op": "&", "a": "x", "b": 0})

    state = SparseSim(1)
    eng = pc.HybridEngine()
    output, _ = eng.run(state, qc, shot_id=0)

    assert int(output["x"]) == 0


def test_negative_value_roundtrip_through_bitint_storage() -> None:
    """Verify negative values survive round-trips through BitInt(64) storage.

    The CVM stores classical variables as BitInt(cvar_size) and does
    signed arithmetic via BitInt(regwidth). With regwidth=32 (default)
    and BitInt(64) storage, negative values are stored directly as
    signed 64-bit values and read back without conversion.
    """
    qc = pc.QuantumCircuit(
        cvar_spec={"a": 64, "b": 64, "c": 64},
        num_qubits=1,
    )
    # a = 0 - 1 = -1
    qc.append("cop", set(), expr={"t": "a", "op": "-", "a": 0, "b": 1})
    # b = a * 100 = -100
    qc.append("cop", set(), expr={"t": "b", "op": "*", "a": "a", "b": 100})
    # c = b + 50 = -50
    qc.append("cop", set(), expr={"t": "c", "op": "+", "a": "b", "b": 50})

    state = SparseSim(1)
    eng = pc.HybridEngine()
    output, _ = eng.run(state, qc, shot_id=0)

    assert int(output["a"]) == -1
    assert int(output["b"]) == -100
    assert int(output["c"]) == -50


def test_negative_roundtrip_requires_storage_wider_than_regwidth() -> None:
    """Demonstrate that signed round-trips require storage_width > regwidth.

    With BitUInt(64) storage and regwidth=32, the round-trip works because
    the 64-bit unsigned representation, when masked to 33 bits by BitInt(32),
    preserves the sign bit.

    With BitUInt(32) storage and regwidth=32, the round-trip fails: -1 is
    stored as 0xFFFFFFFF (32 bits), then BitInt(32, 0xFFFFFFFF) masks to
    33 bits giving 0x0FFFFFFFF where bit 32 is 0 (positive), reading as
    4294967295 instead of -1.
    """
    # 64-bit storage with 32-bit regwidth: works
    output_64 = {"x": pc.BitUInt(64)}
    eval_cop({"t": "x", "op": "-", "a": 0, "b": 1}, output_64, width=32, shot_id=0)
    raw_64 = int(output_64["x"])
    recovered_64 = int(pc.BitInt(32, raw_64))
    assert recovered_64 == -1

    # 32-bit storage with 32-bit regwidth: the sign is lost
    output_32 = {"x": pc.BitUInt(32)}
    eval_cop({"t": "x", "op": "-", "a": 0, "b": 1}, output_32, width=32, shot_id=0)
    raw_32 = int(output_32["x"])
    # 0xFFFFFFFF = 4294967295, which in 33-bit BitInt has sign bit 0 (positive)
    recovered_32 = int(pc.BitInt(32, raw_32))
    assert recovered_32 != -1  # sign is lost
    assert recovered_32 == 4294967295


def test_large_unsigned_value_into_bitint() -> None:
    """Verify that large unsigned values from BitUInt(64) flow into BitInt(32).

    When a negative value is stored in BitUInt(64), int() returns a value
    exceeding i64::MAX (e.g. 0xFFFFFFFFFFFFFFFA for -6). The arbitrary-
    precision BitInt constructor must handle this without overflow.
    """
    # Store -6 in BitUInt(64)
    x = pc.BitUInt(64)
    x.set_clip(pc.BitInt(32, -6))

    raw = int(x)
    assert raw == 0xFFFFFFFFFFFFFFFA  # exceeds i64::MAX

    # BitInt(32, raw) must handle this large value and recover -6
    recovered = int(pc.BitInt(32, raw))
    assert recovered == -6


def test_eval_op_multiply_by_zero_directly() -> None:
    """Test eval_op with b=0 at the function level."""
    result = eval_op("*", pc.BitInt(32, 42), 0, width=32)
    assert int(result) == 0


def test_eval_op_modulo_with_zero_dividend() -> None:
    """Test eval_op with a=0."""
    result = eval_op("%", 0, pc.BitInt(32, 7), width=32)
    assert int(result) == 0


def test_chained_negative_arithmetic() -> None:
    """Test a chain of signed operations that stay negative throughout.

    Exercises repeated round-trips through BitInt(64) storage where
    intermediate values are negative.
    """
    qc = pc.QuantumCircuit(
        cvar_spec={"x": 64},
        num_qubits=1,
    )
    # x = -10
    qc.append("cop", set(), expr={"t": "x", "op": "-", "a": 0, "b": 10})
    # x = x - 5 = -15
    qc.append("cop", set(), expr={"t": "x", "op": "-", "a": "x", "b": 5})
    # x = x * 2 = -30
    qc.append("cop", set(), expr={"t": "x", "op": "*", "a": "x", "b": 2})
    # x = x + 10 = -20
    qc.append("cop", set(), expr={"t": "x", "op": "+", "a": "x", "b": 10})

    state = SparseSim(1)
    eng = pc.HybridEngine()
    output, _ = eng.run(state, qc, shot_id=0)

    assert int(output["x"]) == -20


def test_condition_on_zero_valued_variable() -> None:
    """Test that conditions correctly evaluate when a variable holds zero.

    A BitUInt(64) storing 0 should compare equal to integer 0 and not
    equal to 1.
    """
    qc = pc.QuantumCircuit(
        cvar_spec={"flag": 64, "x": 64, "y": 64},
        num_qubits=1,
    )
    # flag = 0 (default), x = 10 (unconditional), y = 20 (unconditional)
    qc.append("cop", set(), expr={"t": "x", "op": "=", "a": 10})
    qc.append("cop", set(), expr={"t": "y", "op": "=", "a": 20})

    # This should NOT execute (flag == 0, not 1)
    qc.append(
        "cop",
        set(),
        expr={"t": "x", "op": "*", "a": "x", "b": 0},
        cond={"a": "flag", "op": "==", "b": 1},
    )
    # This SHOULD execute (flag == 0)
    qc.append(
        "cop",
        set(),
        expr={"t": "y", "op": "*", "a": "y", "b": 0},
        cond={"a": "flag", "op": "==", "b": 0},
    )

    state = SparseSim(1)
    eng = pc.HybridEngine()
    output, _ = eng.run(state, qc, shot_id=0)

    assert int(output["x"]) == 10  # unchanged, condition was false
    assert int(output["y"]) == 0  # multiplied by 0, condition was true


def test_bitwise_not_of_zero() -> None:
    """Test unary NOT of zero via the `c` parameter in recur_eval_op.

    This directly exercises the `elif c:` -> `elif c is not None:` fix.
    With the old code, c=0 would be falsy, causing the unary path to be
    skipped entirely. The code would fall through to `get_val(a, ...)`
    where `a` is None (unary ops don't set `a`), causing a crash.
    """
    output = {"x": pc.BitInt(64)}
    eval_cop({"t": "x", "op": "~", "c": 0}, output, width=32, shot_id=0)
    # ~BitInt(32, 0) = all bits set = -1 in signed 32-bit
    assert int(output["x"]) == -1


def test_comparison_with_zero_operand() -> None:
    """Test that comparison operations with b=0 produce correct results.

    With the old `if b:` check, b=0 would skip the comparison entirely
    and return `a` (the BitInt value) instead of the comparison result.
    So `5 == 0` would return 5 instead of False/0.
    """
    output = {"x": pc.BitInt(64), "eq_result": pc.BitInt(64), "ne_result": pc.BitInt(64)}

    # x = 5
    eval_cop({"t": "x", "op": "=", "a": 5}, output, width=32, shot_id=0)

    # eq_result = (x == 0) should be 0 (False)
    eval_cop({"t": "eq_result", "op": "==", "a": "x", "b": 0}, output, width=32, shot_id=0)
    assert int(output["eq_result"]) == 0  # not 5!

    # ne_result = (x != 0) should be 1 (True)
    eval_cop({"t": "ne_result", "op": "!=", "a": "x", "b": 0}, output, width=32, shot_id=0)
    assert int(output["ne_result"]) == 1


def test_comparison_zero_equals_zero() -> None:
    """Test that 0 == 0 produces True (1), not 0."""
    output = {"x": pc.BitInt(64), "result": pc.BitInt(64)}

    # x stays at default 0
    # result = (x == 0) should be 1 (True)
    eval_cop({"t": "result", "op": "==", "a": "x", "b": 0}, output, width=32, shot_id=0)
    assert int(output["result"]) == 1


def test_shift_by_zero() -> None:
    """Test left and right shift by zero.

    Another b=0 case: x >> 0 and x << 0 should return x unchanged.
    With the old code, the shift would be skipped (correct result by
    accident for shifts, but for the wrong reason).
    """
    output = {"x": pc.BitInt(64), "lsh": pc.BitInt(64), "rsh": pc.BitInt(64)}

    eval_cop({"t": "x", "op": "=", "a": 42}, output, width=32, shot_id=0)
    eval_cop({"t": "lsh", "op": "<<", "a": "x", "b": 0}, output, width=32, shot_id=0)
    eval_cop({"t": "rsh", "op": ">>", "a": "x", "b": 0}, output, width=32, shot_id=0)

    assert int(output["lsh"]) == 42
    assert int(output["rsh"]) == 42


def test_signed_boundary_values() -> None:
    """Test INT32_MIN and INT32_MAX round-trip through BitInt(64) storage."""
    int32_max = 2**31 - 1  # 2147483647
    int32_min = -(2**31)  # -2147483648

    output = {"max_val": pc.BitInt(64), "min_val": pc.BitInt(64)}

    # Store INT32_MAX
    eval_cop({"t": "max_val", "op": "=", "a": int32_max}, output, width=32, shot_id=0)
    assert int(output["max_val"]) == int32_max

    # Store INT32_MIN (0 - 2147483648)
    eval_cop({"t": "min_val", "op": "-", "a": 0, "b": int32_max}, output, width=32, shot_id=0)
    eval_cop({"t": "min_val", "op": "-", "a": "min_val", "b": 1}, output, width=32, shot_id=0)
    assert int(output["min_val"]) == int32_min


def test_signed_overflow_wrapping() -> None:
    """Test overflow wrapping in BitInt(32) arithmetic.

    BitInt(N) uses N+1 internal bits (extra sign bit), so BitInt(32) has
    range -2^32 to 2^32-1, NOT the standard i32 range of -2^31 to 2^31-1.

    Overflow occurs at 2^32: (2^32-1) + 1 wraps to -2^32.
    """
    bitint32_max = 2**32 - 1
    bitint32_min = -(2**32)

    output = {"x": pc.BitInt(64)}

    eval_cop({"t": "x", "op": "=", "a": bitint32_max}, output, width=32, shot_id=0)
    assert int(output["x"]) == bitint32_max

    eval_cop({"t": "x", "op": "+", "a": "x", "b": 1}, output, width=32, shot_id=0)
    assert int(output["x"]) == bitint32_min


def test_subtract_zero() -> None:
    """Test subtraction by zero (b=0 for `-` operator)."""
    output = {"x": pc.BitInt(64)}

    eval_cop({"t": "x", "op": "=", "a": 99}, output, width=32, shot_id=0)
    eval_cop({"t": "x", "op": "-", "a": "x", "b": 0}, output, width=32, shot_id=0)

    assert int(output["x"]) == 99


def test_less_than_zero() -> None:
    """Test less-than comparison with b=0."""
    output = {
        "neg": pc.BitInt(64),
        "pos": pc.BitInt(64),
        "r_neg": pc.BitInt(64),
        "r_pos": pc.BitInt(64),
    }

    # neg = -5
    eval_cop({"t": "neg", "op": "-", "a": 0, "b": 5}, output, width=32, shot_id=0)
    # pos = 5
    eval_cop({"t": "pos", "op": "=", "a": 5}, output, width=32, shot_id=0)

    # (-5 < 0) should be True (1)
    eval_cop({"t": "r_neg", "op": "<", "a": "neg", "b": 0}, output, width=32, shot_id=0)
    # (5 < 0) should be False (0)
    eval_cop({"t": "r_pos", "op": "<", "a": "pos", "b": 0}, output, width=32, shot_id=0)

    assert int(output["r_neg"]) == 1
    assert int(output["r_pos"]) == 0


def test_export_cvar_preserves_negative_sign() -> None:
    """Test that ExportCVar correctly copies negative BitInt values.

    The ExportCVar path creates a copy of the variable for export.
    The old code used BitInt(str(val)) which parsed the binary user bits,
    losing the sign. The fix uses BitInt(val.size, int(val)).
    """
    qc = pc.QuantumCircuit(
        cvar_spec={"x": 64},
        num_qubits=1,
    )
    # x = 0 - 42 = -42
    qc.append("cop", set(), expr={"t": "x", "op": "-", "a": 0, "b": 42})
    # Export x
    qc.append("cop", set(), cop_type="ExportCVar", export="x")

    state = SparseSim(1)
    eng = pc.HybridEngine()
    output, _ = eng.run(state, qc, shot_id=0)

    # When ExportCVar fires, output is replaced by output_export.
    # The exported value should preserve sign.
    assert int(output["x"]) == -42


def test_export_cvar_preserves_positive_value() -> None:
    """Test that ExportCVar also works correctly for positive values."""
    qc = pc.QuantumCircuit(
        cvar_spec={"x": 64},
        num_qubits=1,
    )
    qc.append("cop", set(), expr={"t": "x", "op": "=", "a": 123})
    qc.append("cop", set(), cop_type="ExportCVar", export="x")

    state = SparseSim(1)
    eng = pc.HybridEngine()
    output, _ = eng.run(state, qc, shot_id=0)

    assert int(output["x"]) == 123


def test_division_by_zero_raises() -> None:
    """Test that division by zero raises ZeroDivisionError.

    Before the truthiness fix, b=0 would skip the operation entirely
    (silently returning a instead of raising). Now it correctly raises.
    """
    import pytest

    with pytest.raises(ZeroDivisionError):
        eval_op("/", pc.BitInt(32, 10), 0, width=32)


def test_modulo_by_zero_raises() -> None:
    """Test that modulo by zero raises ZeroDivisionError."""
    import pytest

    with pytest.raises(ZeroDivisionError):
        eval_op("%", pc.BitInt(32, 10), 0, width=32)


def test_bitwise_or_with_zero() -> None:
    """Test that bitwise OR with zero returns the original value."""
    output = {"x": pc.BitInt(64)}
    eval_cop({"t": "x", "op": "=", "a": 0xFF}, output, width=32, shot_id=0)
    eval_cop({"t": "x", "op": "|", "a": "x", "b": 0}, output, width=32, shot_id=0)
    assert int(output["x"]) == 0xFF


def test_bitwise_xor_with_zero() -> None:
    """Test that bitwise XOR with zero returns the original value."""
    output = {"x": pc.BitInt(64)}
    eval_cop({"t": "x", "op": "=", "a": 42}, output, width=32, shot_id=0)
    eval_cop({"t": "x", "op": "^", "a": "x", "b": 0}, output, width=32, shot_id=0)
    assert int(output["x"]) == 42


def test_two_negative_variables_subtraction() -> None:
    """Test subtracting two negative variables: (-10) - (-5) = -5."""
    qc = pc.QuantumCircuit(
        cvar_spec={"x": 64, "y": 64, "result": 64},
        num_qubits=1,
    )
    qc.append("cop", set(), expr={"t": "x", "op": "-", "a": 0, "b": 10})
    qc.append("cop", set(), expr={"t": "y", "op": "-", "a": 0, "b": 5})
    qc.append("cop", set(), expr={"t": "result", "op": "-", "a": "x", "b": "y"})

    state = SparseSim(1)
    eng = pc.HybridEngine()
    output, _ = eng.run(state, qc, shot_id=0)

    assert int(output["x"]) == -10
    assert int(output["y"]) == -5
    assert int(output["result"]) == -5


def test_two_negative_variables_multiplication() -> None:
    """Test multiplying two negative variables: (-4) * (-3) = 12."""
    qc = pc.QuantumCircuit(
        cvar_spec={"x": 64, "y": 64, "result": 64},
        num_qubits=1,
    )
    qc.append("cop", set(), expr={"t": "x", "op": "-", "a": 0, "b": 4})
    qc.append("cop", set(), expr={"t": "y", "op": "-", "a": 0, "b": 3})
    qc.append("cop", set(), expr={"t": "result", "op": "*", "a": "x", "b": "y"})

    state = SparseSim(1)
    eng = pc.HybridEngine()
    output, _ = eng.run(state, qc, shot_id=0)

    assert int(output["x"]) == -4
    assert int(output["y"]) == -3
    assert int(output["result"]) == 12


def test_two_negative_variables_division() -> None:
    """Test dividing two negative variables: (-20) / (-4) = 5."""
    output = {"x": pc.BitInt(64), "y": pc.BitInt(64), "result": pc.BitInt(64)}
    eval_cop({"t": "x", "op": "-", "a": 0, "b": 20}, output, width=32, shot_id=0)
    eval_cop({"t": "y", "op": "-", "a": 0, "b": 4}, output, width=32, shot_id=0)
    eval_cop({"t": "result", "op": "/", "a": "x", "b": "y"}, output, width=32, shot_id=0)

    assert int(output["x"]) == -20
    assert int(output["y"]) == -4
    assert int(output["result"]) == 5


def test_negative_variable_comparison() -> None:
    """Test comparing two negative variables: (-100) < (-10) is True."""
    output = {
        "x": pc.BitInt(64),
        "y": pc.BitInt(64),
        "lt": pc.BitInt(64),
        "gt": pc.BitInt(64),
    }
    eval_cop({"t": "x", "op": "-", "a": 0, "b": 100}, output, width=32, shot_id=0)
    eval_cop({"t": "y", "op": "-", "a": 0, "b": 10}, output, width=32, shot_id=0)

    eval_cop({"t": "lt", "op": "<", "a": "x", "b": "y"}, output, width=32, shot_id=0)
    eval_cop({"t": "gt", "op": ">", "a": "x", "b": "y"}, output, width=32, shot_id=0)

    assert int(output["lt"]) == 1  # -100 < -10
    assert int(output["gt"]) == 0  # -100 > -10 is false
