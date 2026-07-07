"""Test arithmetic and boolean type support in Guppy->Selene pipeline."""

import pytest
from guppylang import guppy
from guppylang.std.quantum import h, measure, qubit, x
from pecos import Guppy, sim
from pecos_rslib import state_vector


def test_integer_arithmetic() -> None:
    """Test integer arithmetic operations."""

    @guppy
    def quantum_add() -> bool:
        q = qubit()
        x = 3
        y = 2
        result = x + y  # result = 5

        if result > 3:  # 5 > 3, so H gate applied
            h(q)

        return measure(q)

    results = sim(Guppy(quantum_add)).qubits(1).quantum(state_vector()).seed(42).run(10).to_dict()

    raw_measurements = results["measurements"]
    # For single bool return, measurements is [[1], [0], ...]
    measurements = [m[-1] if isinstance(m, list) else m for m in raw_measurements]
    assert len(measurements) == 10
    # H gate should give mix of 0s and 1s
    assert 0 in measurements
    assert 1 in measurements


def test_boolean_operations() -> None:
    """Test boolean logic operations."""

    @guppy
    def quantum_bool_logic() -> bool:
        q1 = qubit()
        q2 = qubit()
        h(q1)
        m1 = measure(q1)
        m2 = measure(q2)
        return m1 and not m2

    results = sim(Guppy(quantum_bool_logic)).qubits(2).quantum(state_vector()).seed(42).run(10).to_dict()

    raw_measurements = results["measurements"]
    measurements = [m[-1] if isinstance(m, list) else m for m in raw_measurements]
    assert len(measurements) == 10


def test_integer_comparisons() -> None:
    """Test integer comparison operations."""

    @guppy
    def quantum_compare() -> bool:
        q = qubit()
        threshold = 42
        value = 50

        if value > threshold:
            h(q)

        return measure(q)

    results = sim(Guppy(quantum_compare)).qubits(1).quantum(state_vector()).seed(42).run(10).to_dict()

    raw_measurements = results["measurements"]
    measurements = [m[-1] if isinstance(m, list) else m for m in raw_measurements]
    assert len(measurements) == 10
    assert 0 in measurements
    assert 1 in measurements


def test_arithmetic_in_loop() -> None:
    """Test arithmetic in loop control."""

    @guppy
    def quantum_loop() -> bool:
        q = qubit()
        count = 0
        max_count = 3

        while count < max_count:
            if count == 1:  # Only apply H on second iteration
                h(q)
            count = count + 1

        return measure(q)

    results = sim(Guppy(quantum_loop)).qubits(1).quantum(state_vector()).seed(42).run(10).to_dict()

    raw_measurements = results["measurements"]
    measurements = [m[-1] if isinstance(m, list) else m for m in raw_measurements]
    assert len(measurements) == 10
    assert 0 in measurements
    assert 1 in measurements


def test_chained_comparisons() -> None:
    """Test multiple chained comparisons."""

    @guppy
    def quantum_chain() -> bool:
        q = qubit()
        a = 10
        b = 20
        c = 15

        if a < c and c < b:  # 10 < 15 < 20 is True
            h(q)

        return measure(q)

    results = sim(Guppy(quantum_chain)).qubits(1).quantum(state_vector()).seed(42).run(10).to_dict()

    raw_measurements = results["measurements"]
    measurements = [m[-1] if isinstance(m, list) else m for m in raw_measurements]
    assert len(measurements) == 10
    assert 0 in measurements
    assert 1 in measurements


@pytest.mark.skip(
    reason="Conditional quantum ops based on measurement results cause register count mismatch",
)
def test_arithmetic_with_measurements() -> None:
    """Test using measurement results in arithmetic."""

    @guppy
    def quantum_measure_math() -> bool:
        q1 = qubit()
        q2 = qubit()
        h(q1)
        h(q2)

        m1 = measure(q1)
        m2 = measure(q2)

        # Use measurements in arithmetic (bools as ints)
        q3 = qubit()
        if m1 or m2:  # At least one is True
            h(q3)

        return measure(q3)

    results = sim(Guppy(quantum_measure_math)).qubits(3).quantum(state_vector()).seed(42).run(20).to_dict()

    raw_measurements = results["measurements"]
    measurements = [m[-1] if isinstance(m, list) else m for m in raw_measurements]
    assert len(measurements) == 20
    # Should have mix unless both m1 and m2 are 0 (25% chance)


def test_euclidean_division_semantics() -> None:
    """Negative-operand division follows the HUGR spec (Euclidean).

    idivmod_s is defined as q*m+r=n with 0<=r<m, so -3 % 2 == 1 and
    -3 // 2 == -2 (matching Python). The engine used Rust truncating
    division until this was pinned; the X gate fires only if the engine
    computes the spec'd value, so a wrong result measures 0.
    """

    @guppy
    def euclid_mod() -> bool:
        q = qubit()
        a = -3
        if a % 2 == 1:
            x(q)
        return measure(q)

    @guppy
    def euclid_div() -> bool:
        q = qubit()
        a = -3
        if a // 2 == -2:
            x(q)
        return measure(q)

    for prog in (euclid_mod, euclid_div):
        results = sim(Guppy(prog)).qubits(1).quantum(state_vector()).seed(1).run(3).to_dict()
        raw_measurements = results["measurements"]
        measurements = [m[-1] if isinstance(m, list) else m for m in raw_measurements]
        assert measurements == [1, 1, 1], f"Euclidean semantics violated: {measurements}"


def test_shift_semantics() -> None:
    """Left/right shifts on positive values, X-anchored."""

    @guppy
    def shifts() -> bool:
        q = qubit()
        a = 1
        b = 16
        if (a << 3) == 8 and (b >> 2) == 4:
            x(q)
        return measure(q)

    results = sim(Guppy(shifts)).qubits(1).quantum(state_vector()).seed(1).run(3).to_dict()
    raw_measurements = results["measurements"]
    measurements = [m[-1] if isinstance(m, list) else m for m in raw_measurements]
    assert measurements == [1, 1, 1], f"shift semantics violated: {measurements}"


def test_zero_iteration_loop() -> None:
    """A range(0) loop must run zero iterations and fall through cleanly."""

    @guppy
    def zero_iters() -> bool:
        q = qubit()
        count = 0
        for _i in range(0):
            count = count + 1
        if count == 0:
            x(q)
        return measure(q)

    results = sim(Guppy(zero_iters)).qubits(1).quantum(state_vector()).seed(1).run(3).to_dict()
    raw_measurements = results["measurements"]
    measurements = [m[-1] if isinstance(m, list) else m for m in raw_measurements]
    assert measurements == [1, 1, 1], f"zero-iteration loop misbehaved: {measurements}"


def test_division_by_zero_panics() -> None:
    """Division by zero is a runtime error per the HUGR spec (m=0 panics)."""

    @guppy
    def div_zero() -> bool:
        q = qubit()
        a = 5
        b = 0
        if a // b == 0:
            x(q)
        return measure(q)

    with pytest.raises(RuntimeError, match="division by zero"):
        sim(Guppy(div_zero)).qubits(1).quantum(state_vector()).seed(1).run(1).to_dict()


def test_recursion_rejected_loudly() -> None:
    """Recursive guppy functions must produce a clear engine error, not a
    hang or silent truncation."""

    @guppy
    def recurse(n: int) -> int:
        if n <= 0:
            return 0
        return recurse(n - 1)

    @guppy
    def recursive_main() -> bool:
        q = qubit()
        if recurse(3) == 0:
            x(q)
        return measure(q)

    with pytest.raises(RuntimeError, match="recursion is not supported"):
        sim(Guppy(recursive_main)).qubits(1).quantum(state_vector()).seed(1).run(1).to_dict()
