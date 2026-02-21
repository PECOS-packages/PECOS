"""Isolated tests to debug segmentation fault.

This file contains minimal test cases extracted from test_comprehensive_quantum_operations.py
to identify which specific operation causes the segfault.
"""

import pytest
from guppylang import guppy
from guppylang.std.angles import pi
from guppylang.std.quantum import (
    ch,
    cx,
    cy,
    cz,
    discard,
    h,
    measure,
    qubit,
    reset,
    rx,
    ry,
    rz,
    s,
    sdg,
    t,
    tdg,
    toffoli,
    x,
    y,
    z,
)
from pecos import Guppy, sim
from pecos_rslib import state_vector


def decode_integer_results(results: list[int], n_bits: int) -> list[tuple[bool, ...]]:
    """Decode integer-encoded results back to tuples of booleans."""
    decoded = []
    for val in results:
        bits = [bool(val & (1 << i)) for i in range(n_bits)]
        decoded.append(tuple(bits))
    return decoded


class TestIsolatedOps:
    """Test individual operations in isolation to find segfault source."""

    def test_single_h_gate(self) -> None:
        """Test just H gate."""

        @guppy
        def test() -> bool:
            q = qubit()
            h(q)
            return measure(q)

        results = sim(Guppy(test)).qubits(10).quantum(state_vector()).seed(42).run(10)
        assert len(results.get("measurements", results.get("measurement_0", []))) == 10

    def test_single_x_gate(self) -> None:
        """Test just X gate."""

        @guppy
        def test() -> bool:
            q = qubit()
            x(q)
            return measure(q)

        results = sim(Guppy(test)).qubits(10).quantum(state_vector()).seed(42).run(10)
        assert all(r for r in results.get("measurements", results.get("measurement_0", [])))

    def test_single_y_gate(self) -> None:
        """Test just Y gate."""

        @guppy
        def test() -> bool:
            q = qubit()
            y(q)
            return measure(q)

        results = sim(Guppy(test)).qubits(10).quantum(state_vector()).seed(42).run(10)
        assert all(r for r in results.get("measurements", results.get("measurement_0", [])))

    def test_single_z_gate(self) -> None:
        """Test just Z gate."""

        @guppy
        def test() -> bool:
            q = qubit()
            z(q)
            return measure(q)

        results = sim(Guppy(test)).qubits(10).quantum(state_vector()).seed(42).run(10)
        measurements = results.get("measurements", results.get("measurement_0", []))
        # Z on |0> -> |0>, so all measurements should be 0
        assert all(m[0] == 0 for m in measurements)

    def test_phase_gates_s_sdg(self) -> None:
        """Test S and S-dagger gates."""

        @guppy
        def test() -> bool:
            q = qubit()
            x(q)
            s(q)
            sdg(q)
            return measure(q)

        results = sim(Guppy(test)).qubits(10).quantum(state_vector()).seed(42).run(10)
        measurements = results.get("measurements", results.get("measurement_0", []))
        # X on |0> -> |1>, S-Sdg is identity, so all measurements should be 1
        assert all(m[0] == 1 for m in measurements)

    def test_phase_gates_t_tdg(self) -> None:
        """Test T and T-dagger gates."""

        @guppy
        def test() -> bool:
            q = qubit()
            x(q)
            t(q)
            tdg(q)
            return measure(q)

        results = sim(Guppy(test)).qubits(10).quantum(state_vector()).seed(42).run(10)
        measurements = results.get("measurements", results.get("measurement_0", []))
        # X on |0> -> |1>, T-Tdg is identity, so all measurements should be 1
        assert all(m[0] == 1 for m in measurements)

    def test_rotation_rx(self) -> None:
        """Test Rx rotation."""

        @guppy
        def test() -> bool:
            q = qubit()
            rx(q, pi)
            return measure(q)

        results = sim(Guppy(test)).qubits(10).quantum(state_vector()).seed(42).run(10)
        measurements = results.get("measurements", results.get("measurement_0", []))
        # RX(pi) on |0> -> |1>, so all measurements should be 1
        assert all(m[0] == 1 for m in measurements)

    def test_rotation_ry(self) -> None:
        """Test Ry rotation."""

        @guppy
        def test() -> bool:
            q = qubit()
            ry(q, pi)
            return measure(q)

        results = sim(Guppy(test)).qubits(10).quantum(state_vector()).seed(42).run(10)
        measurements = results.get("measurements", results.get("measurement_0", []))
        # RY(pi) on |0> -> |1>, so all measurements should be 1
        assert all(m[0] == 1 for m in measurements)

    def test_rotation_rz(self) -> None:
        """Test Rz rotation."""

        @guppy
        def test() -> bool:
            q = qubit()
            rz(q, pi)
            return measure(q)

        results = sim(Guppy(test)).qubits(10).quantum(state_vector()).seed(42).run(10)
        measurements = results.get("measurements", results.get("measurement_0", []))
        # RZ on |0> -> |0>, so all measurements should be 0
        assert all(m[0] == 0 for m in measurements)

    def test_two_qubit_cx(self) -> None:
        """Test CX gate."""

        @guppy
        def test() -> tuple[bool, bool]:
            q1 = qubit()
            q2 = qubit()
            x(q1)
            cx(q1, q2)
            return measure(q1), measure(q2)

        results = sim(Guppy(test)).qubits(10).quantum(state_vector()).seed(42).run(10)
        # Should get [1, 1] for both qubits (X on q1, then CX flips q2)
        measurements = results.get("measurements", [])
        assert all(m == [1, 1] for m in measurements)

    def test_two_qubit_cy(self) -> None:
        """Test CY gate."""

        @guppy
        def test() -> tuple[bool, bool]:
            q1 = qubit()
            q2 = qubit()
            x(q1)
            cy(q1, q2)
            return measure(q1), measure(q2)

        results = sim(Guppy(test)).qubits(10).quantum(state_vector()).seed(42).run(10)
        # CY with control=1 should flip target
        measurements = results.get("measurements", [])
        assert all(m == [1, 1] for m in measurements)

    def test_two_qubit_cz(self) -> None:
        """Test CZ gate."""

        @guppy
        def test() -> tuple[bool, bool]:
            q1 = qubit()
            q2 = qubit()
            x(q1)
            x(q2)
            cz(q1, q2)
            return measure(q1), measure(q2)

        results = sim(Guppy(test)).qubits(10).quantum(state_vector()).seed(42).run(10)
        # Both qubits should be |1> (CZ only adds phase, no bit flip)
        measurements = results.get("measurements", [])
        assert all(m == [1, 1] for m in measurements)

    def test_two_qubit_ch(self) -> None:
        """Test CH gate."""

        @guppy
        def test() -> tuple[bool, bool]:
            q1 = qubit()
            q2 = qubit()
            ch(q1, q2)
            return measure(q1), measure(q2)

        results = sim(Guppy(test)).qubits(10).quantum(state_vector()).seed(42).run(10)
        # CH with control=0 does nothing, both stay |0>
        measurements = results.get("measurements", [])
        assert all(m == [0, 0] for m in measurements)

    def test_toffoli(self) -> None:
        """Test Toffoli gate."""

        @guppy
        def test() -> tuple[bool, bool, bool]:
            q1 = qubit()
            q2 = qubit()
            q3 = qubit()
            x(q1)
            x(q2)
            toffoli(q1, q2, q3)
            return measure(q1), measure(q2), measure(q3)

        results = sim(Guppy(test)).qubits(10).quantum(state_vector()).seed(42).run(10)
        # Both controls at |1>, target flips to |1>
        measurements = results.get("measurements", [])
        assert all(m == [1, 1, 1] for m in measurements)

    def test_reset_operation(self) -> None:
        """Test reset operation."""

        @guppy
        def test() -> bool:
            q = qubit()
            x(q)
            reset(q)
            return measure(q)

        results = sim(Guppy(test)).qubits(10).quantum(state_vector()).seed(42).run(10)
        measurements = results.get("measurements", results.get("measurement_0", []))
        # Reset should bring |1> back to |0>
        assert all(m[0] == 0 for m in measurements)

    def test_discard_operation(self) -> None:
        """Test discard operation."""

        @guppy
        def test() -> bool:
            q1 = qubit()
            h(q1)
            discard(q1)
            q2 = qubit()
            x(q2)
            return measure(q2)

        results = sim(Guppy(test)).qubits(10).quantum(state_vector()).seed(42).run(10)
        measurements = results.get("measurements", results.get("measurement_0", []))
        # After discard, X on q2 gives |1>
        assert all(m[0] == 1 for m in measurements)

    def test_complex_sequence(self) -> None:
        """Test a more complex sequence of operations."""

        @guppy
        def test() -> tuple[bool, bool, bool, bool]:
            # Similar to the original test that might be causing issues
            q1 = qubit()
            h(q1)
            x(q1)
            result1 = measure(q1)

            q2 = qubit()
            y(q2)
            result2 = measure(q2)

            q3 = qubit()
            z(q3)
            result3 = measure(q3)

            q4 = qubit()
            x(q4)
            z(q4)
            result4 = measure(q4)

            return result1, result2, result3, result4

        results = sim(Guppy(test)).qubits(10).quantum(state_vector()).seed(42).run(10)
        measurements = results.get("measurements", [])

        for m in measurements:
            # m is now a list like [r1, r2, r3, r4]
            _, r2, r3, r4 = m
            assert r2 == 1  # Y on |0⟩ gives |1⟩
            assert r3 == 0  # Z on |0⟩ doesn't change
            assert r4 == 1  # X on |0⟩ gives |1⟩
