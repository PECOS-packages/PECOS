# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Exact amplitudes, physical schedules, and injection-site fault conditioning."""

import itertools
from dataclasses import replace

import numpy as np
import pytest
from pecos.guppy_gen.gadget_render import render_gadget_function
from pecos.qec.surface import SurfacePatch, hook_injection
from pecos.qec.surface.circuit_builder import OpType, QubitAllocation, SurfaceCircuitStep, TickCircuitRenderer
from pecos.qec.surface.injection import injection_correction_supports

from qec.surface.test_state_injection import _codespace, _equivalent, _mask, _physical_steps


def _branches(patch, hook, fault=()):
    """Evolve every branch of the actual ancilla circuit without a stabilizer approximation."""
    steps = list(hook.seed.steps) + [s for s in hook.injection.steps if s.op_type != OpType.MEASURE]
    if fault:
        index = next(i for i, s in enumerate(steps) if s.op_type in {OpType.T, OpType.TDG})
        q = steps[index].qubits[0]
        steps[index + 1 : index + 1] = [SurfaceCircuitStep(op, [q]) for op in fault]
    vector = np.zeros(1 << patch.geometry.num_qubits, dtype=complex)
    vector[0] = 1
    vector, _ = _physical_steps(vector, steps)
    allocation = hook.seed.allocations[0]
    nx = len(allocation.x_ancilla_qubits)
    ancillas = allocation.x_ancilla_qubits + allocation.z_ancilla_qubits
    for bits in itertools.product((False, True), repeat=len(ancillas)):
        mask = sum(int(bit) << q for bit, q in zip(bits, ancillas, strict=True))
        row = vector[np.arange(1 << patch.geometry.num_data) | mask]
        probability = float(np.vdot(row, row).real)
        if probability > 1e-12:
            yield (bits[:nx], bits[nx:]), row / np.sqrt(probability), probability


@pytest.mark.parametrize("distance", [2, 3])
@pytest.mark.parametrize("state", ["T", "TDG", "X", "-X", "Y", "-Y"])
def test_every_ideal_branch_has_correct_logical_phase(distance, state):
    patch = SurfacePatch.create(distance=distance)
    hook = hook_injection(patch, state=state)
    phase = {"T": np.exp(1j * np.pi / 4), "TDG": np.exp(-1j * np.pi / 4), "X": 1, "-X": -1, "Y": 1j, "-Y": -1j}[state]
    zero, one = _codespace(patch)
    expected = (zero + phase * one) / np.sqrt(2)
    total = 0
    for syndrome, row, probability in _branches(patch, hook):
        total += probability
        assert hook.accepts((syndrome, syndrome))
        corrected, _ = _physical_steps(row, hook.correction_gadget(syndrome).steps)
        _equivalent(corrected, expected)
    assert total == pytest.approx(1)


@pytest.mark.parametrize("distance", [2, 3])
def test_physical_verification_preserves_every_baseline(distance):
    patch = SurfacePatch.create(distance=distance)
    hook = hook_injection(patch)
    allocation = hook.seed.allocations[0]
    for syndrome, row, _ in _branches(patch, hook):
        full = np.zeros(1 << patch.geometry.num_qubits, dtype=complex)
        full[: len(row)] = row
        bits = syndrome[0] + syndrome[1]
        full, probability = _physical_steps(full, hook.verification.steps, bits)
        assert probability == pytest.approx(1)
        mask = sum(
            int(b) << q for b, q in zip(bits, allocation.x_ancilla_qubits + allocation.z_ancilla_qubits, strict=True)
        )
        _equivalent(full[np.arange(len(row)) | mask], row)


@pytest.mark.parametrize("state", ["T", "TDG"])
@pytest.mark.parametrize("distance", [2, 3])
def test_depolarizing_fault_at_rotation_is_conditioned_not_eliminated(distance, state):
    """An exact noisy T-state calculation, not an X/Y stabilizer proxy benchmark.

    Only the hook rotation is noisy here: I/X/Y/Z probabilities 1-p,p/3,p/3,p/3.
    All measurements and subsequent operations are ideal. X/Y are rejected;
    Z survives and produces the orthogonal magic state. This checks both the
    benefit and the unavoidable O(p) limitation of postselection.
    """
    patch = SurfacePatch.create(distance=distance)
    hook = hook_injection(patch, state=state)
    zero, one = _codespace(patch)
    expected = (zero + np.exp((1 if state == "T" else -1) * 1j * np.pi / 4) * one) / np.sqrt(2)
    indices = np.arange(len(zero))
    accepted = []
    correct = []
    for fault in ((), (OpType.X,), (OpType.X, OpType.Z), (OpType.Z,)):
        accepted_weight = correct_weight = 0
        for syndrome, branch, probability in _branches(patch, hook, fault):
            row = branch.copy()
            if not hook.accepts((syndrome, syndrome)):
                continue
            # Independently project the data onto the baseline check signs.
            # This sums precisely the branches where an ideal verification
            # round produces no detection event.
            for check, bit in zip(patch.geometry.x_stabilizers, syndrome[0], strict=True):
                row = (row + (-1) ** bit * row[indices ^ _mask(check.data_qubits)]) / 2
            for check, bit in zip(patch.geometry.z_stabilizers, syndrome[1], strict=True):
                parity = np.array([(int(i) & _mask(check.data_qubits)).bit_count() % 2 for i in indices])
                row = row * (parity == bit)
            accepted_weight += probability * np.vdot(row, row).real
            row, _ = _physical_steps(row, hook.correction_gadget(syndrome).steps)
            correct_weight += probability * abs(np.vdot(expected, row)) ** 2
        accepted.append(accepted_weight)
        correct.append(correct_weight)
    assert accepted == pytest.approx([1, 0, 0, 1], abs=1e-12)
    assert correct == pytest.approx([1, 0, 0, 0], abs=1e-12)
    p = 0.03
    weights = np.array([1 - p, p / 3, p / 3, p / 3])
    assert weights @ accepted == pytest.approx(1 - 2 * p / 3)
    assert (weights @ correct) / (weights @ accepted) == pytest.approx((1 - p) / (1 - 2 * p / 3))


@pytest.mark.parametrize("distance", [2, 3, 4, 5, 7, 13])
def test_schedule_layers_are_disjoint_local_and_preserve_logical_corrections(distance):
    patch = SurfacePatch.create(distance=distance)
    hook = hook_injection(patch)
    allocation = hook.seed.allocations[0]
    checks = {
        a: set(check.data_qubits)
        for ancillas, family in (
            (allocation.x_ancilla_qubits, patch.geometry.x_stabilizers),
            (allocation.z_ancilla_qubits, patch.geometry.z_stabilizers),
        )
        for a, check in zip(ancillas, family, strict=True)
    }
    for gadget in (hook.injection, hook.verification):
        occupied = set()
        for step in gadget.steps:
            if step.op_type == OpType.TICK:
                occupied.clear()
            elif step.op_type == OpType.CX:
                assert not occupied.intersection(step.qubits)
                occupied.update(step.qubits)
                ancilla = next(q for q in step.qubits if q in checks)
                data = next(q for q in step.qubits if q not in checks)
                assert data in checks[ancilla]
    for basis, family, logical in (
        ("X", patch.geometry.x_stabilizers, patch.geometry.logical_x),
        ("Z", patch.geometry.z_stabilizers, patch.geometry.logical_z),
    ):
        for i, support in enumerate(injection_correction_supports(patch, basis=basis)):
            assert not len(set(support) & set(logical.data_qubits)) % 2
            assert [len(set(support) & set(c.data_qubits)) % 2 for c in family] == [
                int(i == j) for j in range(len(family))
            ]


def test_distance_three_gate_order_matches_reference_schedule():
    """Reference Fig. 2, mapped to PECOS row-major data and check indices."""
    hook = hook_injection(SurfacePatch.create(distance=3))
    expected = [
        [(10, 2), (3, 13), (1, 14)],
        [(10, 1), (11, 3), (12, 7), (6, 13), (0, 14), (4, 15)],
        [(9, 1), (10, 5), (11, 7), (4, 14), (8, 15), (2, 16)],
        [(9, 0), (10, 4), (11, 6), (3, 14), (7, 15), (5, 16)],
    ]
    actual, layer = [], []
    for step in hook.injection.steps:
        if step.op_type == OpType.CX:
            layer.append(tuple(step.qubits))
        elif step.op_type == OpType.TICK and layer:
            actual.append(layer)
            layer = []
    assert actual == expected
    # The second round is reversed NORMAL extraction, not a reversal of the
    # first round's special diagonal order and not the default PECOS schedule.
    verification = [tuple(s.qubits) for s in hook.verification.steps if s.op_type == OpType.CX]
    assert verification[6:12] == [(9, 1), (10, 5), (11, 7), (0, 14), (4, 15), (2, 16)]


@pytest.mark.parametrize("distance", [4, 5, 7, 13])
@pytest.mark.parametrize("state", ["X", "Y"])
def test_large_patches_have_correct_checks_and_signed_logical_observable(distance, state):
    """Check actual Clifford circuits beyond the dense-oracle qubit limit."""
    stim = pytest.importorskip("stim")
    patch = SurfacePatch.create(distance=distance)
    hook = hook_injection(patch, state=state)
    gates = {OpType.ALLOC: "R", OpType.H: "H", OpType.CX: "CX", OpType.SZ: "S", OpType.MEASURE: "M"}
    first = stim.Circuit()
    verify = stim.Circuit()
    for circuit, steps in ((first, hook.seed.steps + hook.injection.steps), (verify, hook.verification.steps)):
        for step in steps:
            if step.op_type in gates:
                circuit.append(gates[step.op_type], step.qubits)
    nx = len(patch.geometry.x_stabilizers)
    count = nx + len(patch.geometry.z_stabilizers)
    for seed in range(4):
        simulator = stim.TableauSimulator(seed=seed)
        simulator.do(first)
        simulator.do(verify)
        measured = simulator.current_measurement_record()
        records = tuple((tuple(measured[k : k + nx]), tuple(measured[k + nx : k + count])) for k in (0, count))
        assert hook.accepts(records)
        for step in hook.correction_gadget(records[0]).steps:
            simulator.do(stim.Circuit(f"{step.op_type.name} {step.qubits[0]}"))
        for basis, checks in (("X", patch.geometry.x_stabilizers), ("Z", patch.geometry.z_stabilizers)):
            for check in checks:
                observable = stim.PauliString(patch.geometry.num_qubits)
                for q in check.data_qubits:
                    observable[q] = basis
                assert simulator.peek_observable_expectation(observable) == 1
        logical_x = stim.PauliString(patch.geometry.num_qubits)
        logical_z = stim.PauliString(patch.geometry.num_qubits)
        for q in patch.geometry.logical_x.data_qubits:
            logical_x[q] = "X"
        for q in patch.geometry.logical_z.data_qubits:
            logical_z[q] = "Z"
        observable = logical_x if state == "X" else 1j * logical_x * logical_z
        assert simulator.peek_observable_expectation(observable) == 1


def test_record_validation_and_random_first_signs():
    hook = hook_injection(SurfacePatch.create(distance=3))
    baseline = ((False, True, False, False), (True, True, True, False))
    assert hook.accepts((baseline,) * 3)
    for family in range(2):
        for i in range(4):
            changed = [list(bits) for bits in baseline]
            changed[family][i] = not changed[family][i]
            assert not hook.accepts((baseline, tuple(tuple(bits) for bits in changed)))
    for records in ((), (baseline,), (((), ()), baseline)):
        with pytest.raises(ValueError, match=r"requires|Expected"):
            hook.accepts(records)
    with pytest.raises(ValueError, match="booleans"):
        hook.correction_gadget(((0,) * 4, (False,) * 4))
    with pytest.raises(ValueError, match="basis"):
        injection_correction_supports(SurfacePatch.create(distance=3), basis="Y")


def test_noncontiguous_allocation_and_tick_rendering():
    patch = SurfacePatch.create(distance=3)
    allocation = QubitAllocation(list(range(20, 38, 2)), [2, 4, 6, 8], [11, 13, 15, 17])
    hook = hook_injection(patch, allocation)
    assert next(s.qubits for s in hook.injection.steps if s.op_type == OpType.T) == [13]
    circuit = TickCircuitRenderer(add_detectors=False).render(
        list(hook.seed.steps + hook.injection.steps + hook.verification.steps),
        allocation,
        patch,
        0,
        "X",
    )
    assert (
        sum(g.gate_type.name == "T" for i in range(circuit.num_ticks()) for g in circuit.get_tick(i).gate_batches())
        == 1
    )
    allocation.z_ancilla_qubits[0] = allocation.data_qubits[0]
    with pytest.raises(ValueError, match="distinct"):
        hook_injection(patch, allocation)


@pytest.mark.parametrize("kwargs", [{"distance": 1}, {"dx": 3, "dz": 5}, {"distance": 3, "rotated": False}])
def test_unsupported_patch(kwargs):
    with pytest.raises(ValueError, match="square rotated"):
        hook_injection(SurfacePatch.create(**kwargs))


def test_unsupported_state():
    with pytest.raises(ValueError, match="Unsupported hook state"):
        hook_injection(SurfacePatch.create(distance=3), state="Z")


def test_acceptance_uses_values_across_sequence_types():
    hook = hook_injection(SurfacePatch.create(distance=3))
    xs = [False] * len(hook.z_corrections)
    zs = [False] * len(hook.x_corrections)
    assert hook.accepts(((tuple(xs), zs), [xs, tuple(zs)], (xs, zs)))
    changed = xs.copy()
    changed[0] = True
    assert not hook.accepts(((tuple(xs), zs), [changed, tuple(zs)]))


def test_hook_seed_is_state_independent_and_flag_drives_docstring():
    patch = SurfacePatch.create(distance=3)
    seed = hook_injection(patch, state="T").seed
    assert seed == hook_injection(patch, state="-Y").seed
    assert seed.basis is None
    assert seed.injection_seed
    rendered = "\n".join(render_gadget_function(replace(seed, name="renamed_hook_seed")))
    assert "state-independent hook injection seed" in rendered
    assert "None" not in rendered
