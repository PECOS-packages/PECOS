# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Gate classification must preserve noise and reject unsupported exports."""

import math

import pytest
from pecos.qec.surface.circuit_builder import tick_circuit_to_stim
from pecos_rslib.quantum import TickCircuit


def circuit_with_gate(name: str, qubits: list[int], angles: list[float] | None = None) -> TickCircuit:
    circuit = TickCircuit()
    circuit.tick().add_gate(name, qubits, angles)
    return circuit


@pytest.mark.parametrize(
    ("name", "angles"),
    [
        ("RXYXY2Q", [0.3, 0.1]),
        ("CH", None),
        ("RXXRYYRZZ", [0.1, 0.2, 0.3]),
        ("U2q", [0.0] * 15),
    ],
)
def test_with_noise_structural_two_qubit_gates(name: str, angles: list[float] | None) -> None:
    noisy = circuit_with_gate(name, [0, 1], angles).with_noise(p2=0.5)
    channels = [gate for _, gate in noisy.gate_batches() if gate.is_channel()]
    assert len(channels) == 1
    assert list(channels[0].qubits) == [0, 1]
    terms = channels[0].channel_mixed_pauli_terms()
    assert len(terms) == 16
    assert terms[0] == (0.5, [])
    assert [prob for prob, _ in terms[1:]] == pytest.approx([0.5 / 15] * 15)
    assert {tuple(paulis) for _, paulis in terms} == {
        tuple((pauli, qubit) for qubit, pauli in enumerate((p0, p1)) if pauli != "I") for p0 in "IXYZ" for p1 in "IXYZ"
    }


def test_with_noise_px_preparation() -> None:
    noisy = circuit_with_gate("PX", [0]).with_noise(p_prep=0.5)
    channels = [gate for _, gate in noisy.gate_batches() if gate.is_channel()]
    assert len(channels) == 1
    assert channels[0].channel_mixed_pauli_terms() == [(0.5, []), (0.5, [("Z", 0)])]


def test_with_noise_px_preparation_full_error_channel() -> None:
    circuit = circuit_with_gate("PX", [0])
    circuit.tick().add_gate("MX", [0])
    noisy = circuit.with_noise(p_prep=1.0)
    channels = [gate for _, gate in noisy.gate_batches() if gate.is_channel()]
    assert len(channels) == 1
    assert channels[0].channel_mixed_pauli_terms() == [(0.0, []), (1.0, [("Z", 0)])]
    assert not any(gate.is_channel() for _, gate in circuit.with_noise(p_prep=0.0).gate_batches())


@pytest.mark.parametrize("name", ["PZ", "QAlloc"])
def test_with_noise_z_preparation_retains_bit_flip(name: str) -> None:
    noisy = circuit_with_gate(name, [0]).with_noise(p_prep=0.5)
    channels = [gate for _, gate in noisy.gate_batches() if gate.is_channel()]
    assert len(channels) == 1
    assert channels[0].channel_mixed_pauli_terms() == [(0.5, []), (0.5, [("X", 0)])]


@pytest.mark.parametrize("rate", ["p1", "p2", "p_prep"])
def test_with_noise_ccx_raises(rate: str) -> None:
    circuit = circuit_with_gate("CCX", [0, 1, 2])
    with pytest.raises(ValueError, match=r"CCX.*tick 0"):
        circuit.with_noise(**{rate: 0.1})
    assert circuit.with_noise().gate_count() == 1


@pytest.mark.parametrize("name", ["I", "Idle"])
def test_with_noise_identity_retains_single_qubit_noise(name: str) -> None:
    circuit = TickCircuit()
    if name == "Idle":
        circuit.tick().idle(1, [0])
    else:
        circuit.tick().add_gate(name, [0])
    noisy = circuit.with_noise(p1=0.1)
    assert sum(gate.is_channel() for _, gate in noisy.gate_batches()) == 1


def test_with_noise_rejects_existing_channel_operations() -> None:
    noisy = circuit_with_gate("PX", [0]).with_noise(p_prep=0.5)
    with pytest.raises(ValueError, match="already contains channel operations"):
        noisy.with_noise(p1=0.1)


@pytest.mark.parametrize("phi", [0.0, math.pi, math.pi / 2, 3 * math.pi / 2])
@pytest.mark.parametrize(("theta", "suffix"), [(math.pi / 2, ""), (3 * math.pi / 2, "_DAG"), (math.pi, None)])
def test_tick_circuit_stim_rxyxy2q_clifford(phi: float, theta: float, suffix: str | None) -> None:
    circuit = circuit_with_gate("RXYXY2Q", [0, 1], [theta, phi])
    axis = "X" if phi in (0.0, math.pi) else "Y"
    operation = axis if suffix is None else f"SQRT_{axis}{axis}{suffix}"
    assert tick_circuit_to_stim(circuit, p2=1) == f"{operation} 0 1\nDEPOLARIZE2(1) 0 1"


def test_tick_circuit_stim_rxyxy2q_zero() -> None:
    circuit = circuit_with_gate("RXYXY2Q", [0, 1], [0.0, 0.1])
    assert tick_circuit_to_stim(circuit) == ""


@pytest.mark.parametrize(("theta", "phi"), [(0.3, 0.1), (math.pi / 2, 0.1)])
def test_tick_circuit_stim_rxyxy2q_unsupported_angles(theta: float, phi: float) -> None:
    circuit = circuit_with_gate("RXYXY2Q", [0, 1], [theta, phi])
    with pytest.raises(ValueError, match="RXYXY2Q angles"):
        tick_circuit_to_stim(circuit)


def test_tick_circuit_stim_ccx_raises() -> None:
    with pytest.raises(ValueError, match="CCX"):
        tick_circuit_to_stim(circuit_with_gate("CCX", [0, 1, 2]))


@pytest.mark.parametrize("name", ["I", "QFree", "TrackedPauliMeta"])
def test_tick_circuit_stim_shared_transparent_gates(name: str) -> None:
    assert tick_circuit_to_stim(circuit_with_gate(name, [0])) == ""


@pytest.mark.parametrize("name", ["MX", "MZ", "MPZ", "MeasureFree", "MeasureLeaked"])
def test_with_noise_measurements_have_no_quantum_channel(name: str) -> None:
    noisy = circuit_with_gate(name, [0]).with_noise(p1=0.1, p_prep=0.1)
    assert not any(gate.is_channel() for _, gate in noisy.gate_batches())


def test_with_noise_batched_two_qubit_pairs() -> None:
    noisy = circuit_with_gate("RXYXY2Q", [0, 1, 2, 3], [0.3, 0.1]).with_noise(p2=0.5)
    channels = [gate for _, gate in noisy.gate_batches() if gate.is_channel()]
    assert [list(gate.qubits) for gate in channels] == [[0, 1], [2, 3]]
