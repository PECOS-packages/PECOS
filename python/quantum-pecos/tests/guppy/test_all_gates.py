"""Gate verification tests for the HUGR interpreter."""

import pytest
from guppylang import guppy
from guppylang.std.quantum import (
    crz,
    cx,
    cy,
    cz,
    discard,
    h,
    measure,
    pi,
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
    v,
    vdg,
    x,
    y,
    z,
)
from pecos import Guppy, sim
from pecos_rslib import state_vector


def run_circuit(
    guppy_func: object,
    num_qubits: int,
    shots: int = 100,
    seed: int = 42,
) -> dict:
    """Run a Guppy function with PECOS direct HUGR interpreter."""
    return sim(Guppy(guppy_func)).qubits(num_qubits).quantum(state_vector()).seed(seed).run(shots).to_dict()


def get_measurements(results: dict) -> list:
    """Extract measurements from results."""
    return results.get("measurements", [])


class TestSingleQubitGates:
    """Test single-qubit gates produce correct state transformations."""

    def test_x_gate(self) -> None:
        """X gate: |0> -> |1>."""

        @guppy
        def circuit() -> bool:
            q = qubit()
            x(q)
            return measure(q)

        measurements = get_measurements(run_circuit(circuit, 1))
        assert all(m == [1] for m in measurements)

    def test_y_gate(self) -> None:
        """Y gate: |0> -> i|1> (measures as |1>)."""

        @guppy
        def circuit() -> bool:
            q = qubit()
            y(q)
            return measure(q)

        measurements = get_measurements(run_circuit(circuit, 1))
        assert all(m == [1] for m in measurements)

    def test_z_gate_on_zero(self) -> None:
        """Z gate: |0> -> |0> (no change in measurement)."""

        @guppy
        def circuit() -> bool:
            q = qubit()
            z(q)
            return measure(q)

        measurements = get_measurements(run_circuit(circuit, 1))
        assert all(m == [0] for m in measurements)

    def test_z_gate_on_one(self) -> None:
        """Z gate: |1> -> -|1> (still measures as |1>)."""

        @guppy
        def circuit() -> bool:
            q = qubit()
            x(q)
            z(q)
            return measure(q)

        measurements = get_measurements(run_circuit(circuit, 1))
        assert all(m == [1] for m in measurements)

    def test_h_gate_superposition(self) -> None:
        """H gate: |0> -> |+> (50/50 distribution)."""

        @guppy
        def circuit() -> bool:
            q = qubit()
            h(q)
            return measure(q)

        measurements = get_measurements(run_circuit(circuit, 1, shots=1000))
        zeros = sum(1 for m in measurements if m == [0])
        ones = sum(1 for m in measurements if m == [1])
        # Expect roughly 50/50 with some tolerance
        assert 400 < zeros < 600, f"Expected ~500 zeros, got {zeros}"
        assert 400 < ones < 600, f"Expected ~500 ones, got {ones}"

    def test_h_h_identity(self) -> None:
        """H-H = I: |0> -> |+> -> |0>."""

        @guppy
        def circuit() -> bool:
            q = qubit()
            h(q)
            h(q)
            return measure(q)

        measurements = get_measurements(run_circuit(circuit, 1))
        assert all(m == [0] for m in measurements)

    def test_s_gate(self) -> None:
        """S gate (phase): |0> -> |0>."""

        @guppy
        def circuit() -> bool:
            q = qubit()
            s(q)
            return measure(q)

        measurements = get_measurements(run_circuit(circuit, 1))
        assert all(m == [0] for m in measurements)

    def test_sdg_gate(self) -> None:
        """Sdg gate (S-dagger): |0> -> |0>."""

        @guppy
        def circuit() -> bool:
            q = qubit()
            sdg(q)
            return measure(q)

        measurements = get_measurements(run_circuit(circuit, 1))
        assert all(m == [0] for m in measurements)

    def test_s_sdg_identity(self) -> None:
        """S-Sdg = I on superposition."""

        @guppy
        def circuit() -> bool:
            q = qubit()
            h(q)
            s(q)
            sdg(q)
            h(q)
            return measure(q)

        measurements = get_measurements(run_circuit(circuit, 1))
        assert all(m == [0] for m in measurements)

    def test_t_gate(self) -> None:
        """T gate: |0> -> |0>."""

        @guppy
        def circuit() -> bool:
            q = qubit()
            t(q)
            return measure(q)

        measurements = get_measurements(run_circuit(circuit, 1))
        assert all(m == [0] for m in measurements)

    def test_tdg_gate(self) -> None:
        """Tdg gate (T-dagger): |0> -> |0>."""

        @guppy
        def circuit() -> bool:
            q = qubit()
            tdg(q)
            return measure(q)

        measurements = get_measurements(run_circuit(circuit, 1))
        assert all(m == [0] for m in measurements)

    def test_t_tdg_identity(self) -> None:
        """T-Tdg = I on superposition."""

        @guppy
        def circuit() -> bool:
            q = qubit()
            h(q)
            t(q)
            tdg(q)
            h(q)
            return measure(q)

        measurements = get_measurements(run_circuit(circuit, 1))
        assert all(m == [0] for m in measurements)

    def test_v_gate_squared(self) -> None:
        """V gate (sqrt(X)): V^2 = X, so |0> -> |1>."""

        @guppy
        def circuit() -> bool:
            q = qubit()
            v(q)
            v(q)
            return measure(q)

        measurements = get_measurements(run_circuit(circuit, 1))
        assert all(m == [1] for m in measurements)

    def test_vdg_gate_squared(self) -> None:
        """Vdg gate: Vdg^2 = X, so |0> -> |1>."""

        @guppy
        def circuit() -> bool:
            q = qubit()
            vdg(q)
            vdg(q)
            return measure(q)

        measurements = get_measurements(run_circuit(circuit, 1))
        assert all(m == [1] for m in measurements)

    def test_v_vdg_identity(self) -> None:
        """V-Vdg = I."""

        @guppy
        def circuit() -> bool:
            q = qubit()
            v(q)
            vdg(q)
            return measure(q)

        measurements = get_measurements(run_circuit(circuit, 1))
        assert all(m == [0] for m in measurements)


class TestRotationGates:
    """Test rotation gates with angle parameters."""

    def test_rx_pi(self) -> None:
        """RX(pi) = X (up to global phase)."""

        @guppy
        def circuit() -> bool:
            q = qubit()
            rx(q, pi)
            return measure(q)

        measurements = get_measurements(run_circuit(circuit, 1))
        assert all(m == [1] for m in measurements)

    def test_ry_pi(self) -> None:
        """RY(pi) = Y (up to global phase)."""

        @guppy
        def circuit() -> bool:
            q = qubit()
            ry(q, pi)
            return measure(q)

        measurements = get_measurements(run_circuit(circuit, 1))
        assert all(m == [1] for m in measurements)

    def test_rz_on_zero(self) -> None:
        """RZ(pi) on |0> = |0>."""

        @guppy
        def circuit() -> bool:
            q = qubit()
            rz(q, pi)
            return measure(q)

        measurements = get_measurements(run_circuit(circuit, 1))
        assert all(m == [0] for m in measurements)

    def test_rz_phase_flip(self) -> None:
        """RZ(pi): |+> -> |-> -> |1> via H-RZ-H."""

        @guppy
        def circuit() -> bool:
            q = qubit()
            h(q)
            rz(q, pi)
            h(q)
            return measure(q)

        measurements = get_measurements(run_circuit(circuit, 1))
        assert all(m == [1] for m in measurements)


class TestTwoQubitGates:
    """Test two-qubit gates."""

    def test_cx_control_zero(self) -> None:
        """CX: control=|0>, no flip."""

        @guppy
        def circuit() -> tuple[bool, bool]:
            q0 = qubit()
            q1 = qubit()
            cx(q0, q1)
            return measure(q0), measure(q1)

        measurements = get_measurements(run_circuit(circuit, 2))
        assert all(m == [0, 0] for m in measurements)

    def test_cx_control_one(self) -> None:
        """CX: control=|1>, flip target."""

        @guppy
        def circuit() -> tuple[bool, bool]:
            q0 = qubit()
            q1 = qubit()
            x(q0)
            cx(q0, q1)
            return measure(q0), measure(q1)

        measurements = get_measurements(run_circuit(circuit, 2))
        assert all(m == [1, 1] for m in measurements)

    def test_cy_control_zero(self) -> None:
        """CY: control=|0>, no flip."""

        @guppy
        def circuit() -> tuple[bool, bool]:
            q0 = qubit()
            q1 = qubit()
            cy(q0, q1)
            return measure(q0), measure(q1)

        measurements = get_measurements(run_circuit(circuit, 2))
        assert all(m == [0, 0] for m in measurements)

    def test_cy_control_one(self) -> None:
        """CY: control=|1>, flip target."""

        @guppy
        def circuit() -> tuple[bool, bool]:
            q0 = qubit()
            q1 = qubit()
            x(q0)
            cy(q0, q1)
            return measure(q0), measure(q1)

        measurements = get_measurements(run_circuit(circuit, 2))
        assert all(m == [1, 1] for m in measurements)

    def test_cz_phase_only(self) -> None:
        """CZ: both |1>, phase only (no measurement change)."""

        @guppy
        def circuit() -> tuple[bool, bool]:
            q0 = qubit()
            q1 = qubit()
            x(q0)
            x(q1)
            cz(q0, q1)
            return measure(q0), measure(q1)

        measurements = get_measurements(run_circuit(circuit, 2))
        assert all(m == [1, 1] for m in measurements)

    def test_crz_control_zero(self) -> None:
        """CRZ: control=|0>, no effect."""

        @guppy
        def circuit() -> tuple[bool, bool]:
            q0 = qubit()
            q1 = qubit()
            crz(q0, q1, pi)
            return measure(q0), measure(q1)

        measurements = get_measurements(run_circuit(circuit, 2))
        assert all(m == [0, 0] for m in measurements)

    def test_crz_control_one(self) -> None:
        """CRZ: control=|1>, applies RZ(pi) to target."""

        @guppy
        def circuit() -> tuple[bool, bool]:
            q0 = qubit()
            q1 = qubit()
            x(q0)  # ctrl = |1>
            h(q1)  # target = |+>
            crz(q0, q1, pi)  # target -> |->
            h(q1)  # target -> |1>
            return measure(q0), measure(q1)

        measurements = get_measurements(run_circuit(circuit, 2))
        assert all(m == [1, 1] for m in measurements)


class TestThreeQubitGates:
    """Test three-qubit gates (Toffoli)."""

    def test_toffoli_00(self) -> None:
        """Toffoli: controls=00, no flip."""

        @guppy
        def circuit() -> tuple[bool, bool, bool]:
            q0 = qubit()
            q1 = qubit()
            q2 = qubit()
            toffoli(q0, q1, q2)
            return measure(q0), measure(q1), measure(q2)

        measurements = get_measurements(run_circuit(circuit, 3))
        assert all(m == [0, 0, 0] for m in measurements)

    def test_toffoli_10(self) -> None:
        """Toffoli: controls=10, no flip."""

        @guppy
        def circuit() -> tuple[bool, bool, bool]:
            q0 = qubit()
            q1 = qubit()
            q2 = qubit()
            x(q0)
            toffoli(q0, q1, q2)
            return measure(q0), measure(q1), measure(q2)

        measurements = get_measurements(run_circuit(circuit, 3))
        assert all(m == [1, 0, 0] for m in measurements)

    def test_toffoli_01(self) -> None:
        """Toffoli: controls=01, no flip."""

        @guppy
        def circuit() -> tuple[bool, bool, bool]:
            q0 = qubit()
            q1 = qubit()
            q2 = qubit()
            x(q1)
            toffoli(q0, q1, q2)
            return measure(q0), measure(q1), measure(q2)

        measurements = get_measurements(run_circuit(circuit, 3))
        assert all(m == [0, 1, 0] for m in measurements)

    def test_toffoli_11(self) -> None:
        """Toffoli: controls=11, flip target."""

        @guppy
        def circuit() -> tuple[bool, bool, bool]:
            q0 = qubit()
            q1 = qubit()
            q2 = qubit()
            x(q0)
            x(q1)
            toffoli(q0, q1, q2)
            return measure(q0), measure(q1), measure(q2)

        measurements = get_measurements(run_circuit(circuit, 3))
        assert all(m == [1, 1, 1] for m in measurements)


class TestResetAndDiscard:
    """Test reset and discard operations."""

    def test_reset_from_one(self) -> None:
        """Reset: |1> -> |0>."""

        @guppy
        def circuit() -> bool:
            q = qubit()
            x(q)
            reset(q)
            return measure(q)

        measurements = get_measurements(run_circuit(circuit, 1))
        assert all(m == [0] for m in measurements)

    def test_reset_from_zero(self) -> None:
        """Reset: |0> -> |0>."""

        @guppy
        def circuit() -> bool:
            q = qubit()
            reset(q)
            return measure(q)

        measurements = get_measurements(run_circuit(circuit, 1))
        assert all(m == [0] for m in measurements)

    def test_discard(self) -> None:
        """Discard doesn't crash and other qubits work."""

        @guppy
        def circuit() -> bool:
            q1 = qubit()
            q2 = qubit()
            x(q1)
            discard(q2)
            return measure(q1)

        measurements = get_measurements(run_circuit(circuit, 2))
        assert all(m == [1] for m in measurements)


class TestEntanglement:
    """Test entangled states."""

    def test_bell_state(self) -> None:
        """Bell state: perfectly correlated 00 or 11."""

        @guppy
        def circuit() -> tuple[bool, bool]:
            q0 = qubit()
            q1 = qubit()
            h(q0)
            cx(q0, q1)
            return measure(q0), measure(q1)

        measurements = get_measurements(run_circuit(circuit, 2, shots=1000))
        # All measurements should be correlated
        assert all(m[0] == m[1] for m in measurements)
        # Should see both 00 and 11
        has_zeros = any(m[0] == 0 for m in measurements)
        has_ones = any(m[0] == 1 for m in measurements)
        assert has_zeros
        assert has_ones

    def test_ghz_3_state(self) -> None:
        """3-qubit GHZ state: perfectly correlated 000 or 111."""

        @guppy
        def circuit() -> tuple[bool, bool, bool]:
            q0 = qubit()
            q1 = qubit()
            q2 = qubit()
            h(q0)
            cx(q0, q1)
            cx(q1, q2)
            return measure(q0), measure(q1), measure(q2)

        measurements = get_measurements(run_circuit(circuit, 3, shots=1000))
        # All measurements should be correlated
        assert all(m[0] == m[1] == m[2] for m in measurements)
        # Should see both 000 and 111
        has_zeros = any(m[0] == 0 for m in measurements)
        has_ones = any(m[0] == 1 for m in measurements)
        assert has_zeros
        assert has_ones


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
