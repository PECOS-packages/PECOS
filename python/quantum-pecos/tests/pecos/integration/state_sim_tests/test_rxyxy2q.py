# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at https://www.apache.org/licenses/LICENSE-2.0

"""Phase-exact RXYXY2Q boundary and Clifford simulation regressions."""

from __future__ import annotations

import cmath
import math

import numpy as np
import pytest
from pecos.analysis.find_cliffs import rxyxy2q2cliff, rxyxy2q_matrix
from pecos.circuits import QuantumCircuit
from pecos.simulators.cointoss.state import CoinToss
from pecos.simulators.statevec.state import StateVec as PythonStateVec
from pecos_rslib import ByteMessageBuilder, TickCircuit, lower_clifford_rotation
from pecos_rslib.simulators import PauliProp, SparseStab, Stabilizer, StabVec, StateVec

ANGLES = ((0.73, -0.41), (math.pi, math.pi), (math.pi / 2, math.pi / 2), (-0.73, 0.41))
CLIFFORD_ANGLES = ((math.pi / 2, 0), (math.pi, math.pi / 2), (3 * math.pi / 2, math.pi))
# These are all exported qubit state-vector bindings; no SparseStateVec is exported.
STATEVEC_BACKENDS = (StateVec, StabVec, PythonStateVec)


def matrix(theta: float, phi: float) -> np.ndarray:
    """Independent oracle from docs/user-guide/gates.md, in |00>, |01>, |10>, |11> order."""
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array(
        [
            [c, 0, 0, -1j * cmath.exp(-2j * phi) * s],
            [0, c, -1j * s, 0],
            [0, -1j * s, c, 0],
            [-1j * cmath.exp(2j * phi) * s, 0, 0, c],
        ],
    )


def amplitudes(state: object) -> np.ndarray:
    """Read all backends in the Python wrapper's big-endian basis convention."""
    if isinstance(state, PythonStateVec):
        return np.array(state.vector.tolist())
    if isinstance(state, StabVec):
        little_endian = np.array([complex(real, imag) for real, imag in state.state_vector()])
        return little_endian[[0, 2, 1, 3]]
    return np.array(state.vector_big_endian().tolist())


@pytest.mark.parametrize("backend", STATEVEC_BACKENDS, ids=("rust", "stabvec", "python"))
@pytest.mark.parametrize(("theta", "phi"), ANGLES)
@pytest.mark.parametrize("basis", range(4))
def test_rxyxy2q_matrix_columns(backend: type, theta: float, phi: float, basis: int) -> None:
    """Every input basis column must match complex amplitudes, including global phase."""
    state = backend(2)
    for qubit in range(2):
        if basis & (1 << (1 - qubit)):
            state.run_gate("X", {qubit})
    state.run_gate("RXYXY2Q", {(0, 1)}, angles=(theta, phi))
    np.testing.assert_allclose(amplitudes(state), matrix(theta, phi)[:, basis], rtol=0, atol=1e-12)


@pytest.mark.parametrize("backend", STATEVEC_BACKENDS)
@pytest.mark.parametrize(("theta", "phi"), ANGLES)
def test_rxyxy2q_quantum_circuit_round_trip(backend: type, theta: float, phi: float) -> None:
    """The circuit retains the actual gate and both angles through Rust storage."""
    circuit = QuantumCircuit()
    circuit.append("RXYXY2Q", {(0, 1)}, angles=(theta, phi))
    [(symbol, locations, params)] = list(circuit.items())
    assert symbol == "RXYXY2Q"
    assert locations == {(0, 1)}
    assert params["angles"] == pytest.approx((theta, phi))
    actual, direct = backend(2), backend(2)
    actual.run_circuit(circuit)
    direct.run_gate("RXYXY2Q", {(0, 1)}, angles=(theta, phi))
    np.testing.assert_allclose(amplitudes(actual), amplitudes(direct), rtol=0, atol=1e-12)


def entangle(state: object, theta: float, phi: float, basis: str) -> None:
    """Prepare a Bell state, rotate, then select the joint measurement basis."""
    state.run_gate("H", {0})
    state.run_gate("CX", {(0, 1)})
    state.run_gate("SZ", {0})
    state.run_gate("RXYXY2Q", {(0, 1)}, angles=(theta, phi))
    for qubit, axis in enumerate(basis):
        if axis == "Y":
            state.run_gate("SZdg", {qubit})
        if axis in {"X", "Y"}:
            state.run_gate("H", {qubit})


@pytest.mark.parametrize("backend", [Stabilizer, SparseStab])
@pytest.mark.parametrize(("theta", "phi"), CLIFFORD_ANGLES)
@pytest.mark.parametrize("basis", ["XX", "XY", "YX", "ZZ"])
def test_rxyxy2q_stabilizer_statistics(backend: type, theta: float, phi: float, basis: str) -> None:
    """Joint outcome frequencies agree with state-vector probabilities in X/Y/Z bases."""
    reference = StateVec(2)
    entangle(reference, theta, phi, basis)
    probabilities = abs(amplitudes(reference)) ** 2
    counts = np.zeros(4)
    shots = 1024
    state = backend(2, seed=42)
    for _ in range(shots):
        state.reset()
        entangle(state, theta, phi, basis)
        result = state.run_gate("MZ", {0, 1})
        counts[2 * result.get(0, 0) + result.get(1, 0)] += 1
    np.testing.assert_allclose(counts / shots, probabilities, rtol=0, atol=0.06)


@pytest.mark.parametrize("backend", [Stabilizer, SparseStab, PauliProp])
def test_rxyxy2q_non_clifford_rejected(backend: type) -> None:
    """Unsupported Clifford rotations surface the Rust error with the gate name."""
    with pytest.raises(ValueError, match="RXYXY2Q"):
        backend(2).run_gate("RXYXY2Q", {(0, 1)}, angles=(0.3, 0))


@pytest.mark.parametrize(("theta", "phi"), CLIFFORD_ANGLES)
def test_rxyxy2q_pauli_propagation(theta: float, phi: float) -> None:
    """PauliProp follows the same Clifford conjugation as RXX or RYY."""
    actual, reference = PauliProp(2), PauliProp(2)
    for state in (actual, reference):
        state.track_x([0])
        state.track_z([1])
    actual.run_gate("RXYXY2Q", {(0, 1)}, angles=(theta, phi))
    reference.run_gate("RYY" if phi == math.pi / 2 else "RXX", {(0, 1)}, angle=theta)
    assert actual.get_faults() == reference.get_faults()
    assert actual.get_sign() == reference.get_sign()
    assert actual.get_img() == reference.get_img()


@pytest.mark.parametrize(("theta", "phi"), [*CLIFFORD_ANGLES, (0, 0.37)])
def test_rxyxy2q_clifford_lowering(theta: float, phi: float) -> None:
    """Registry lowering is projectively equivalent to the direct matrix."""
    actual, reference = StateVec(2), StateVec(2)
    actual.run_gate("H", {0})
    reference.run_gate("H", {0})
    for symbol, positions in lower_clifford_rotation("RXYXY2Q", (theta, phi)):
        location = positions[0] if len(positions) == 1 else tuple(positions)
        actual.run_gate(symbol, {location})
    reference.run_gate("RXYXY2Q", {(0, 1)}, angles=(theta, phi))
    a, b = amplitudes(actual), amplitudes(reference)
    np.testing.assert_allclose(np.outer(a, a.conj()), np.outer(b, b.conj()), rtol=0, atol=1e-12)


def test_rxyxy2q_coin_toss_and_message_builder() -> None:
    """No-op simulation and byte messages accept the real gate name."""
    assert CoinToss(2).run_gate("RXYXY2Q", {(0, 1)}, angles=ANGLES[0]) == {}
    builder = ByteMessageBuilder()
    builder.for_quantum_operations()
    builder.rxyxy2q(*ANGLES[0], [(0, 1)])


@pytest.mark.parametrize(("theta", "phi"), ANGLES)
def test_rxyxy2q_analysis_matrix(theta: float, phi: float) -> None:
    """Analysis uses the same documented, phase-exact matrix."""
    np.testing.assert_allclose(rxyxy2q_matrix(theta, phi).tolist(), matrix(theta, phi), rtol=0, atol=1e-12)


@pytest.mark.parametrize(
    ("theta", "phi", "expected"),
    [
        (0, 0.37, "II"),
        (math.pi / 2, 0, "SXX"),
        (math.pi, math.pi / 2, "Y tensor Y"),
        (3 * math.pi / 2, math.pi, "SXXdg"),
        (math.pi, math.pi / 4, "H3 tensor H3"),
        (0.3, 0, False),
    ],
)
def test_rxyxy2q_analysis_cliffords(theta: float, phi: float, expected: str | bool) -> None:
    """Analysis identifies Clifford matrices up to phase, including product Cliffords."""
    assert rxyxy2q2cliff(theta, phi) == expected


@pytest.mark.parametrize("backend", [StateVec, PythonStateVec], ids=["rust", "python"])
@pytest.mark.parametrize("source", ["named", "native"])
@pytest.mark.parametrize("symbol", ["RXYXY2Q", "RXY1Q", "U"])
def test_quantum_circuit_multi_angle_normalization(
    backend: type,
    source: str,
    symbol: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Named inputs and native gates without metadata reach the angles dispatch contract."""
    theta, phi = 0.73, -0.41
    angles = (theta, phi, 0.19) if symbol == "U" else (theta, phi)
    qubits = [0, 1] if symbol == "RXYXY2Q" else [0]
    location = tuple(qubits) if len(qubits) == 2 else qubits[0]
    circuit = QuantumCircuit()
    if source == "named":
        params = {"theta": theta, "phi": phi}
        if symbol == "U":
            params["lambda"] = angles[2]
        circuit.append(symbol, {location}, **params)
    else:
        native = TickCircuit()
        native.tick().add_gate(symbol, qubits, list(angles))
        assert native.get_tick(0).get_gate_attr(0, "_params") is None
        # QuantumCircuit has no public constructor accepting native storage.
        # Attach the real TickCircuit to exercise reconstruction via items().
        monkeypatch.setattr(circuit, "_inner", native)

    actual, direct = backend(2), backend(2)
    for state in (actual, direct):
        state.run_gate("H", {0})
        state.run_gate("CX", {(0, 1)})
    actual.run_circuit(circuit)
    direct.run_gate(symbol, {location}, angles=angles)
    np.testing.assert_allclose(amplitudes(actual), amplitudes(direct), rtol=0, atol=1e-12)
    [(_, _, reconstructed)] = list(circuit.items())
    assert isinstance(reconstructed["angles"], tuple)
