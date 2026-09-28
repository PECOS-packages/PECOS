"""Core quantum operations tests - simplified version."""

import pytest
from guppylang import guppy
from guppylang.std.angles import pi
from guppylang.std.builtins import array, owned
from guppylang.std.builtins import result as record_result
from guppylang.std.quantum import (
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
    t,
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


def get_single_measurements(results: dict) -> list[int]:
    """Read the scalar outcome tag for each shot."""
    return results["outcome"]


def get_measurement_tuples(results: dict, n_bits: int) -> list[tuple[bool, ...]]:
    """Read scalar or array outcome tags with the program's declared width."""
    if n_bits == 1:
        return [(bool(value),) for value in results["outcome"]]
    return [tuple(bool(value) for value in row) for row in results["outcome"]]


class TestSingleQubitGates:
    """Test individual single-qubit gates."""

    def test_x_gate(self) -> None:
        """Test Pauli-X gate."""

        @guppy
        def x_test() -> bool:
            q = qubit()
            x(q)
            output_value = measure(q).read()
            record_result("outcome", output_value)
            return output_value

        results = sim(Guppy(x_test)).qubits(10).quantum(state_vector()).run(10).to_dict()
        measurements = get_single_measurements(results)
        assert all(r == 1 for r in measurements)

    def test_y_gate(self) -> None:
        """Test Pauli-Y gate."""

        @guppy
        def y_test() -> bool:
            q = qubit()
            y(q)
            output_value = measure(q).read()
            record_result("outcome", output_value)
            return output_value

        results = sim(Guppy(y_test)).qubits(10).quantum(state_vector()).run(10).to_dict()
        measurements = get_single_measurements(results)
        assert all(r == 1 for r in measurements)

    def test_z_gate(self) -> None:
        """Test Pauli-Z gate."""

        @guppy
        def z_test() -> bool:
            q = qubit()
            z(q)
            output_value = measure(q).read()
            record_result("outcome", output_value)
            return output_value

        results = sim(Guppy(z_test)).qubits(10).quantum(state_vector()).run(10).to_dict()
        measurements = get_single_measurements(results)
        assert all(r == 0 for r in measurements)

    def test_h_gate(self) -> None:
        """Test Hadamard gate."""

        @guppy
        def h_test() -> bool:
            q = qubit()
            h(q)
            output_value = measure(q).read()
            record_result("outcome", output_value)
            return output_value

        # Use more shots and fixed seed for stability
        results = sim(Guppy(h_test)).qubits(10).quantum(state_vector()).seed(42).run(100).to_dict()
        # Should see both 0 and 1
        measurements = get_single_measurements(results)
        zeros = sum(1 for r in measurements if r == 0)
        ones = sum(1 for r in measurements if r == 1)
        assert zeros > 0
        assert ones > 0

    def test_s_gate(self) -> None:
        """Test S gate."""

        @guppy
        def s_test() -> bool:
            q = qubit()
            x(q)  # |1⟩
            s(q)  # Phase gate
            output_value = measure(q).read()
            record_result("outcome", output_value)
            return output_value

        results = sim(Guppy(s_test)).qubits(10).quantum(state_vector()).run(10).to_dict()
        # S gate doesn't change computational basis
        measurements = get_single_measurements(results)
        assert all(r == 1 for r in measurements)

    def test_t_gate(self) -> None:
        """Test T gate."""

        @guppy
        def t_test() -> bool:
            q = qubit()
            x(q)  # |1⟩
            t(q)  # π/8 gate
            output_value = measure(q).read()
            record_result("outcome", output_value)
            return output_value

        results = sim(Guppy(t_test)).qubits(10).quantum(state_vector()).run(10).to_dict()
        # T gate doesn't change computational basis
        measurements = get_single_measurements(results)
        assert all(r == 1 for r in measurements)


class TestTwoQubitGates:
    """Test two-qubit gates."""

    def test_cx_gate(self) -> None:
        """Test CNOT gate."""

        @guppy
        def cx_test() -> tuple[bool, bool]:
            q1 = qubit()
            q2 = qubit()
            x(q1)  # Control = |1⟩
            cx(q1, q2)  # Target flips
            output_value = measure(q1).read(), measure(q2).read()
            record_result("outcome", array(output_value[0], output_value[1]))
            return output_value

        results = sim(Guppy(cx_test)).qubits(10).quantum(state_vector()).run(10).to_dict()
        # Should get (True, True) for both qubits
        decoded_results = get_measurement_tuples(results, 2)
        assert all(r == (True, True) for r in decoded_results)

    def test_cz_gate(self) -> None:
        """Test CZ gate."""

        @guppy
        def cz_test() -> tuple[bool, bool]:
            q1 = qubit()
            q2 = qubit()
            x(q1)
            x(q2)
            cz(q1, q2)  # Phase when both |1⟩
            output_value = measure(q1).read(), measure(q2).read()
            record_result("outcome", array(output_value[0], output_value[1]))
            return output_value

        results = sim(Guppy(cz_test)).qubits(10).quantum(state_vector()).run(10).to_dict()
        # CZ doesn't change computational basis, both qubits remain |1⟩
        decoded_results = get_measurement_tuples(results, 2)
        assert all(r == (True, True) for r in decoded_results)

    def test_cy_gate(self) -> None:
        """Test CY gate."""

        @guppy
        def cy_test() -> tuple[bool, bool]:
            q1 = qubit()
            q2 = qubit()
            x(q1)  # Control = |1⟩
            cy(q1, q2)  # Apply Y to target
            output_value = measure(q1).read(), measure(q2).read()
            record_result("outcome", array(output_value[0], output_value[1]))
            return output_value

        results = sim(Guppy(cy_test)).qubits(10).quantum(state_vector()).run(10).to_dict()
        # CY with control=1 applies Y to target, Y|0⟩ = i|1⟩, so both measure as |1⟩
        decoded_results = get_measurement_tuples(results, 2)
        assert all(r == (True, True) for r in decoded_results)


class TestQuantumStateManagement:
    """Test state management operations."""

    def test_reset(self) -> None:
        """Test reset operation."""

        @guppy
        def reset_test() -> bool:
            q = qubit()
            x(q)
            reset(q)
            output_value = measure(q).read()
            record_result("outcome", output_value)
            return output_value

        results = sim(Guppy(reset_test)).qubits(10).quantum(state_vector()).run(10).to_dict()
        # Reset should give |0⟩
        measurements = get_single_measurements(results)
        assert all(r == 0 for r in measurements)

    def test_discard(self) -> None:
        """Test discard operation."""

        @guppy
        def discard_test() -> bool:
            q1 = qubit()
            h(q1)
            discard(q1)
            # Allocate new qubit
            q2 = qubit()
            x(q2)
            output_value = measure(q2).read()
            record_result("outcome", output_value)
            return output_value

        results = sim(Guppy(discard_test)).qubits(10).quantum(state_vector()).run(10).to_dict()
        measurements = get_single_measurements(results)
        assert all(r == 1 for r in measurements)


class TestQuantumCircuits:
    """Test quantum circuit patterns."""

    def test_bell_state(self) -> None:
        """Test Bell state creation."""

        @guppy
        def bell_test() -> tuple[bool, bool]:
            q1 = qubit()
            q2 = qubit()
            h(q1)
            cx(q1, q2)
            output_value = measure(q1).read(), measure(q2).read()
            record_result("outcome", array(output_value[0], output_value[1]))
            return output_value

        results = sim(Guppy(bell_test)).qubits(10).quantum(state_vector()).seed(42).run(100).to_dict()
        # Bell state should be correlated
        decoded = get_measurement_tuples(results, 2)
        for a, b in decoded:
            assert a == b  # Bell state is correlated

    def test_ghz_state(self) -> None:
        """Test 3-qubit GHZ state."""

        @guppy
        def ghz_test() -> tuple[bool, bool, bool]:
            q1 = qubit()
            q2 = qubit()
            q3 = qubit()
            h(q1)
            cx(q1, q2)
            cx(q2, q3)
            output_value = measure(q1).read(), measure(q2).read(), measure(q3).read()
            record_result("outcome", array(output_value[0], output_value[1], output_value[2]))
            return output_value

        results = sim(Guppy(ghz_test)).qubits(10).quantum(state_vector()).seed(42).run(100).to_dict()
        # GHZ state should be all-correlated
        decoded = get_measurement_tuples(results, 3)
        for a, b, c in decoded:
            assert a == b == c  # GHZ state is all-correlated


class TestRotationGates:
    """Test rotation gates."""

    def test_rx_gate(self) -> None:
        """Test Rx rotation."""

        @guppy
        def rx_test() -> bool:
            q = qubit()
            rx(q, pi)  # Rx(π) = X up to phase
            output_value = measure(q).read()
            record_result("outcome", output_value)
            return output_value

        results = sim(Guppy(rx_test)).qubits(10).quantum(state_vector()).run(10).to_dict()
        measurements = get_single_measurements(results)
        assert all(r == 1 for r in measurements)

    def test_ry_gate(self) -> None:
        """Test Ry rotation."""

        @guppy
        def ry_test() -> bool:
            q = qubit()
            ry(q, pi)  # Ry(π) flips qubit
            output_value = measure(q).read()
            record_result("outcome", output_value)
            return output_value

        results = sim(Guppy(ry_test)).qubits(10).quantum(state_vector()).run(10).to_dict()
        measurements = get_single_measurements(results)
        assert all(r == 1 for r in measurements)

    def test_rz_gate(self) -> None:
        """Test Rz rotation."""

        @guppy
        def rz_test() -> bool:
            q = qubit()
            rz(q, pi)  # Rz on |0⟩
            output_value = measure(q).read()
            record_result("outcome", output_value)
            return output_value

        results = sim(Guppy(rz_test)).qubits(10).quantum(state_vector()).run(10).to_dict()
        # Rz doesn't change |0⟩ measurement
        measurements = get_single_measurements(results)
        assert all(r == 0 for r in measurements)


class TestControlFlow:
    """Test control flow with quantum operations."""

    def test_conditional_ops(self) -> None:
        """Test conditional quantum operations with boolean constants."""
        # Fixed: Using @owned annotation for qubit parameters

        @guppy
        def apply_conditional_gate(q: qubit @ owned, condition: bool) -> qubit:
            """Apply X gate conditionally based on boolean parameter."""
            if condition:
                x(q)
            # else: do nothing (identity)
            return q

        @guppy
        def test_true_condition() -> bool:
            """Test with condition=True."""
            q = qubit()
            q = apply_conditional_gate(q, True)
            output_value = measure(q).read()
            record_result("outcome", output_value)
            return output_value

        @guppy
        def test_false_condition() -> bool:
            """Test with condition=False."""
            q = qubit()
            q = apply_conditional_gate(q, False)
            output_value = measure(q).read()
            record_result("outcome", output_value)
            return output_value

        # Test with True condition - should apply X gate
        results_true = sim(Guppy(test_true_condition)).qubits(10).quantum(state_vector()).run(10).to_dict()
        measurements_true = get_single_measurements(results_true)
        assert all(r == 1 for r in measurements_true), "True condition should apply X gate"

        # Test with False condition - should not apply X gate
        results_false = sim(Guppy(test_false_condition)).qubits(10).quantum(state_vector()).run(10).to_dict()
        measurements_false = get_single_measurements(results_false)
        assert all(r == 0 for r in measurements_false), "False condition should not apply X gate"

    def test_loop_with_quantum(self) -> None:
        """Test loop with quantum operations."""

        @guppy
        def loop_test() -> int:
            count = 0
            for _i in range(3):
                q = qubit()
                h(q)
                if measure(q).read():
                    count += 1
            output_value = count
            record_result("value", output_value)
            return output_value

        results = sim(Guppy(loop_test)).qubits(10).quantum(state_vector()).seed(42).run(100).to_dict()
        measurements = results["value"]
        assert len(measurements) == 100
        assert all(0 <= count <= 3 for count in measurements)
        # Should see values 0-3
        values = set(measurements)
        assert len(values) >= 2  # At least some variation
