# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Independent complex-amplitude oracles for raw injection and T teleportation."""

import itertools

import numpy as np
import pytest
from pecos.qec.surface import LogicalCircuitBuilder, SurfacePatch
from pecos.qec.surface.circuit_builder import (
    DagCircuitRenderer,
    GuppyRenderer,
    OpType,
    StimRenderer,
    TickCircuitRenderer,
)
from pecos.qec.surface.gadgets import default_allocation, fold_sz_round_gadget
from pecos.qec.surface.injection import injection_correction_supports, state_injection


def _mask(support):
    return sum(1 << q for q in support)


def _codespace(patch):
    """Construct codewords directly from the X stabilizer group, not the encoder."""
    orbit = {0}
    for check in patch.geometry.x_stabilizers:
        orbit |= {value ^ _mask(check.data_qubits) for value in orbit}
    zero = np.zeros(1 << patch.geometry.num_data, dtype=complex)
    zero[list(orbit)] = 1 / np.sqrt(len(orbit))
    return zero, zero[np.arange(len(zero)) ^ _mask(patch.geometry.logical_x.data_qubits)]


def _physical_steps(vector, steps, measurements=()):
    """Dense unitary/postselection oracle, independent of PECOS simulators/renderers."""
    outcomes = iter(measurements)
    indices = np.arange(len(vector))
    probability = 1.0
    for step in steps:
        op = step.op_type.name
        if op in {"ALLOC", "COMMENT", "TICK"}:
            continue  # All qubits are initially zero in these tests.
        q = step.qubits[0]
        bit = (indices >> q) & 1
        if op == "H":
            flipped = vector[indices ^ (1 << q)]
            vector = (np.where(bit == 0, vector, -vector) + flipped) / np.sqrt(2)
        elif op == "CX":
            vector = vector[indices ^ (bit << step.qubits[1])]
        elif op == "CZ":
            vector = vector * (-1) ** (bit & ((indices >> step.qubits[1]) & 1))
        elif op == "X":
            vector = vector[indices ^ (1 << q)]
        elif op in {"Z", "SZ", "SZDG", "T", "TDG"}:
            phase = {"Z": -1, "SZ": 1j, "SZDG": -1j, "T": np.exp(1j * np.pi / 4), "TDG": np.exp(-1j * np.pi / 4)}[op]
            vector = vector * np.where(bit, phase, 1)
        elif op == "MEASURE":
            outcome = next(outcomes)
            vector = vector * (bit == outcome)
            norm = float(np.vdot(vector, vector).real)
            assert norm > 1e-12, "Requested an impossible measurement branch"
            probability *= norm
            vector /= np.sqrt(norm)
        else:
            raise AssertionError(op)
    assert next(outcomes, None) is None
    return vector, probability


def _equivalent(actual, expected):
    assert abs(np.vdot(expected, actual)) == pytest.approx(1, abs=1e-10)


@pytest.mark.parametrize(("dx", "dz"), [(2, 2), (3, 3), (3, 5), (5, 3), (5, 5), (7, 7), (13, 13)])
def test_corrections_preserve_logical_amplitudes(dx, dz):
    patch = SurfacePatch.create(dx=dx, dz=dz)
    for index, correction in enumerate(injection_correction_supports(patch)):
        c = set(correction)
        assert len(c & set(patch.geometry.logical_x.data_qubits)) % 2 == 0
        assert [len(c & set(s.data_qubits)) % 2 for s in patch.geometry.x_stabilizers] == [
            int(i == index) for i in range(len(patch.geometry.x_stabilizers))
        ]


@pytest.mark.parametrize(("dx", "dz"), [(3, 3), (2, 3), (3, 2)])
@pytest.mark.parametrize("state", ["Z", "-Z", "X", "-X", "Y", "-Y", "T", "TDG"])
def test_every_projection_branch_encodes_correct_state(state, dx, dz):
    patch = SurfacePatch.create(dx=dx, dz=dz)
    size = 1 << patch.geometry.num_data
    nx = len(patch.geometry.x_stabilizers)
    injection = state_injection(patch, state=state)
    initial = np.zeros(size, dtype=complex)
    initial[0] = 1
    seed, _ = _physical_steps(initial, injection.seed.steps)
    zero, one = _codespace(patch)
    amplitudes = {
        "Z": (1, 0),
        "-Z": (0, 1),
        "X": (1, 1),
        "-X": (1, -1),
        "Y": (1, 1j),
        "-Y": (1, -1j),
        "T": (1, np.exp(1j * np.pi / 4)),
        "TDG": (1, np.exp(-1j * np.pi / 4)),
    }
    alpha, beta = amplitudes[state]
    expected = (alpha * zero + beta * one) / np.sqrt(abs(alpha) ** 2 + abs(beta) ** 2)
    for outcomes in itertools.product((False, True), repeat=nx):
        vector = seed.copy()
        for check, outcome in zip(patch.geometry.x_stabilizers, outcomes, strict=True):
            vector = (vector + (-1) ** outcome * vector[np.arange(size) ^ _mask(check.data_qubits)]) / 2
        assert np.vdot(vector, vector).real == pytest.approx(1 / (1 << nx))
        vector *= np.sqrt(1 << nx)
        vector, _ = _physical_steps(vector, injection.correction_gadget(outcomes).steps)
        _equivalent(vector, expected)


@pytest.mark.parametrize("outcomes", [(False,) * 4, (True,) * 4, (True, False, True, False)])
def test_physical_syndrome_schedule_matches_projection(outcomes):
    patch = SurfacePatch.create(distance=3)
    injection = state_injection(patch)
    vector = np.zeros(1 << 17, dtype=complex)
    vector[0] = 1
    vector, probability = _physical_steps(
        vector,
        injection.seed.steps + injection.projection.steps,
        (*outcomes, 0, 0, 0, 0),
    )
    assert probability == pytest.approx(1 / 16)
    vector, _ = _physical_steps(vector, injection.correction_gadget(outcomes).steps)
    measured_bits = sum(
        int(bit) << q for bit, q in zip(outcomes, injection.seed.allocations[0].x_ancilla_qubits, strict=True)
    )
    zero, one = _codespace(patch)
    _equivalent(vector[np.arange(512) | measured_bits], (zero + np.exp(1j * np.pi / 4) * one) / np.sqrt(2))


@pytest.mark.parametrize("dagger", [False, True])
@pytest.mark.parametrize(
    ("alpha", "beta"),
    [(1, 0), (0, 1), (1 / np.sqrt(2), 1 / np.sqrt(2)), (np.sqrt(0.3), np.sqrt(0.7) * np.exp(0.37j))],
)
def test_all_resource_readouts_and_physical_correction_implement_t(alpha, beta, dagger):
    """Check all physical ancilla readouts, not just one parity or a Z input."""
    patch = SurfacePatch.create(distance=3)
    allocation = default_allocation(patch)
    zero, one = _codespace(patch)
    phase = np.exp((-1 if dagger else 1) * 1j * np.pi / 4)
    data = alpha * zero + beta * one
    resource = (zero + phase * one) / np.sqrt(2)
    vector = np.kron(resource, data)
    indices = np.arange(len(vector))
    # Physical transversal CX, with the data register controlling the resource.
    vector = vector[indices ^ ((indices & 511) << 9)].reshape(512, 512)
    representatives = {}
    total_probability = 0
    branch_probabilities = [0.0, 0.0]
    for bits, row in enumerate(vector):
        probability = np.vdot(row, row).real
        if probability < 1e-12:
            continue
        total_probability += probability
        outcome = (bits & _mask(patch.geometry.logical_z.data_qubits)).bit_count() % 2
        branch_probabilities[outcome] += probability
        expected_branch = alpha * zero + beta * (phase.conjugate() if outcome else phase) * one
        _equivalent(row / np.sqrt(probability), expected_branch)
        representatives[outcome] = row / np.sqrt(probability)
    assert total_probability == pytest.approx(1)
    assert branch_probabilities == pytest.approx([0.5, 0.5])
    assert set(representatives) == {0, 1}
    for outcome, row in representatives.items():
        if outcome:
            full = np.zeros(1 << 17, dtype=complex)
            full[:512] = row
            correction = fold_sz_round_gadget(patch, allocation, round_index=0, dagger=dagger)
            full, probability = _physical_steps(full, correction.steps, (0,) * 8)
            assert probability == pytest.approx(1)
            corrected = full[:512]
        else:
            corrected = row
        _equivalent(corrected, alpha * zero + beta * phase * one)


@pytest.mark.parametrize("state", ["T", "TDG"])
def test_seed_renderers_preserve_non_clifford_gate(state):
    patch = SurfacePatch.create(distance=3)
    injection = state_injection(patch, state=state)
    args = (list(injection.seed.steps), injection.seed.allocations[0], patch, 0, "Z")
    circuit = TickCircuitRenderer(add_detectors=False).render(*args)
    names = [g.gate_type.name for i in range(circuit.num_ticks()) for g in circuit.get_tick(i).gate_batches()]
    assert names.count("T" if state == "T" else "Tdg") == 1
    assert DagCircuitRenderer().render(*args) is not None
    with pytest.raises(ValueError, match="Stim cannot represent"):
        StimRenderer(add_detectors=False).render(*args)
    with pytest.raises(ValueError, match="memory detector annotations"):
        TickCircuitRenderer().render(*args)
    with pytest.raises(ValueError, match="render_gadget_function"):
        GuppyRenderer().render(*args)


def test_static_builder_rejects_real_injection_before_mutation():
    builder = LogicalCircuitBuilder()
    with pytest.raises(NotImplementedError, match="make_surface_t_teleportation"):
        builder.add_t_via_injection("D", "A")
    assert builder._operations == []  # noqa: SLF001 - rejection must be atomic


@pytest.mark.parametrize("kwargs", [{"distance": 1}, {"distance": 3, "rotated": False}])
def test_unsupported_geometry(kwargs):
    with pytest.raises(ValueError, match="rotated patch"):
        state_injection(SurfacePatch.create(**kwargs))


def test_invalid_inputs():
    patch = SurfacePatch.create(distance=3)
    with pytest.raises(ValueError, match="Unsupported injection state"):
        state_injection(patch, state="A")
    with pytest.raises(ValueError, match="boolean outcome"):
        state_injection(patch).correction_gadget((False,))
    allocation = default_allocation(patch)
    allocation.data_qubits[0] = allocation.x_ancilla_qubits[0]
    with pytest.raises(ValueError, match="distinct"):
        state_injection(patch, allocation)
