"""Testing the unitaries of the state-vector sim against reference matrix definitions."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

import pecos as pc
from pecos.simulators import StateVec

# Load gate_matrix_def from the same directory (importlib mode doesn't auto-add it to sys.path)
_gate_matrix_def_path = Path(__file__).parent / "gate_matrix_def.py"
_spec = importlib.util.spec_from_file_location("gate_matrix_def", _gate_matrix_def_path)
g = importlib.util.module_from_spec(_spec)
sys.modules["gate_matrix_def"] = g
_spec.loader.exec_module(g)


def amp2prob(a: complex) -> float:
    """Convert amplitude to probability. |amp|^2."""
    return (a * a.conjugate()).real


def U(state: StateVec, qubit: int, angles: list[float]) -> None:
    """Apply U = Rz(phi) Ry(theta) Rz(lambda)."""
    theta, phi, lamb = angles

    state.bindings["RZ"](state, qubit, angles=(lamb,))
    state.bindings["RY"](state, qubit, angles=(theta,))
    state.bindings["RZ"](state, qubit, angles=(phi,))


def Udg(state: StateVec, qubit: int, angles: list[float]) -> None:
    """Apply U-dagger = Rz(-phi) Ry(-theta) Rz(-lambda)."""
    theta, phi, lamb = angles

    state.bindings["RZ"](state, qubit, angles=(-phi,))
    state.bindings["RY"](state, qubit, angles=(-theta,))
    state.bindings["RZ"](state, qubit, angles=(-lamb,))


def compare_gates(
    pecos_sym: str | Callable,
    gate_matrix: pc.Array,
    angles: list[float] | float | None = None,
    test_angles: list[float] | None = None,
    num_qubits: int = 1,
    *,
    verbose: bool = True,
) -> float:
    """Test that a gate applied via PECOS gives the same expectation value as the reference matrix definition."""
    state = StateVec(num_qubits=num_qubits)

    qubits = 0 if num_qubits == 1 else list(range(num_qubits))

    pecos_gate = state.bindings[pecos_sym] if isinstance(pecos_sym, str) else pecos_sym

    # Apply random unitary to each qubit
    if test_angles is not None:
        for i in range(num_qubits):
            inc = 3 * i
            U(state, i, test_angles[0 + inc : 3 + inc])

    # Apply gate under test
    if isinstance(angles, float):
        angles = [angles]

    if angles is not None:
        if len(angles) == 1:
            pecos_gate(state, qubits, angles=(angles[0],))
        else:
            pecos_gate(state, qubits, angles=tuple(angles))
    else:
        pecos_gate(state, qubits)

    # Apply the dagger of the random unitary
    if test_angles is not None:
        for i in reversed(range(num_qubits)):
            inc = 3 * i
            Udg(state, i, test_angles[0 + inc : 3 + inc])

    pecos_prob = state.probability(0)

    # Now compute the expected probability from the reference matrix definition
    zero = pc.array([[1], [0]], dtype=pc.dtypes.complex128)

    ref_state = None
    for _ in range(num_qubits):
        ref_state = zero.copy() if ref_state is None else ref_state & zero

    if test_angles is not None:
        ref_unitary = None
        for i in range(num_qubits):
            inc = 3 * i
            ref_unitary = (
                g.U(*test_angles[0 + inc : 3 + inc])
                if ref_unitary is None
                else ref_unitary & g.U(*test_angles[0 + inc : 3 + inc])
            )

        ref_state = ref_unitary.dot(ref_state)

    if angles is not None:
        gate_matrix = gate_matrix(*angles)

    mtx_result = gate_matrix.dot(ref_state)
    mtx_result = ref_state.conj().T.dot(mtx_result)
    mtx_prob = amp2prob(mtx_result[0][0])

    is_close = pc.isclose(pecos_prob, mtx_prob)

    if not is_close and verbose:
        print(f"{pecos_sym}: {pecos_prob} !~ {mtx_prob}")

    return is_close


def test_sq_gates() -> None:
    """Test single-qubit gates against reference matrix definitions."""
    # Check U seems to be implemented correctly
    for _ in range(10):
        assert compare_gates(U, g.U, angles=pc.random.random(3), test_angles=pc.random.random(3))

    # U1q / R1XY
    for _ in range(10):
        assert compare_gates("R1XY", g.U1q, angles=pc.random.random(2), test_angles=pc.random.random(3))

    # Paulis
    for _ in range(10):
        assert compare_gates("X", g.X, test_angles=pc.random.random(3))

    for _ in range(10):
        assert compare_gates("Y", g.Y, test_angles=pc.random.random(3))

    for _ in range(10):
        assert compare_gates("Z", g.Z, test_angles=pc.random.random(3))

    # Rotations
    for _ in range(10):
        assert compare_gates("RX", g.RX, angles=pc.random.random(1), test_angles=pc.random.random(3))

    for _ in range(10):
        assert compare_gates("RY", g.RY, angles=pc.random.random(1), test_angles=pc.random.random(3))

    for _ in range(10):
        assert compare_gates("RZ", g.RZ, angles=pc.random.random(1), test_angles=pc.random.random(3))

    # Sqrt of Paulis
    for _ in range(10):
        assert compare_gates("SqrtX", g.RX(pc.f64.pi / 2), test_angles=pc.random.random(3))

    for _ in range(10):
        assert compare_gates("SqrtXd", g.RX(-pc.f64.pi / 2), test_angles=pc.random.random(3))

    for _ in range(10):
        assert compare_gates("SqrtY", g.RY(pc.f64.pi / 2), test_angles=pc.random.random(3))

    for _ in range(10):
        assert compare_gates("SqrtYd", g.RY(-pc.f64.pi / 2), test_angles=pc.random.random(3))

    for _ in range(10):
        assert compare_gates("SqrtZ", g.RZ(pc.f64.pi / 2), test_angles=pc.random.random(3))

    for _ in range(10):
        assert compare_gates("SqrtZd", g.RZ(-pc.f64.pi / 2), test_angles=pc.random.random(3))


def test_tq_gates() -> None:
    """Test two-qubit gates against reference matrix definitions."""
    for _ in range(30):
        assert compare_gates(
            "SqrtZZ",
            g.SqrtZZ(),
            test_angles=pc.random.random(2 * 3),
            num_qubits=2,
        )

    # Show the SqrtZZ gate is not CNOT/CX
    not_cnot = [
        compare_gates(
            "SqrtZZ",
            g.CX(),
            test_angles=pc.random.random(2 * 3),
            num_qubits=2,
            verbose=False,
        )
        for _ in range(30)
    ]
    assert any(not i for i in not_cnot)

    for _ in range(30):
        assert compare_gates(
            "RZZ",
            g.RZZ,
            angles=pc.random.random(1),
            test_angles=pc.random.random(2 * 3),
            num_qubits=2,
        )
