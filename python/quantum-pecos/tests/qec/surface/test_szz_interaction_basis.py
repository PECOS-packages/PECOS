# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import json

import numpy as np
import pecos as pc
import pytest
import stim
from pecos.qec.surface import NoiseModel, SurfacePatch, TwirlConfig
from pecos.qec.surface.circuit_builder import (
    OpType,
    SurfaceCircuitStep,
    SzzTouchSign,
    _analyze_szz_forward_flow,
    _default_szz_residual_plan,
    _default_szz_sign_vector,
    _propagate_compensated_szz_frame_bits,
    _propagate_sxx_frame_bits,
    _propagate_szz_frame_bits,
    _szz_residual_class,
    _validate_szz_sign_vector,
    build_surface_code_circuit,
    generate_dag_circuit_from_patch,
    generate_stim_from_patch,
    generate_tick_circuit_from_patch,
)
from pecos.qec.surface.decode import build_memory_circuit, generate_circuit_level_dem_from_builder


def _to_numpy_complex(matrix: object) -> np.ndarray:
    arr = np.asarray(matrix)
    if arr.ndim >= 1 and arr.shape[-1:] == (2,) and not np.issubdtype(arr.dtype, np.complexfloating):
        return arr[..., 0].astype(float) + 1j * arr[..., 1].astype(float)
    return np.asarray(matrix, dtype=complex)


def _pauli_matrix(pauli: object) -> np.ndarray:
    return _to_numpy_complex(pauli.to_matrix())


def _kron(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return _to_numpy_complex(pc.kron(pc.array(left), pc.array(right)))


I2 = _pauli_matrix(pc.PauliString.I())
PAULI_X = _pauli_matrix(pc.X(0))
PAULI_Z = _pauli_matrix(pc.Z(0))
H = (PAULI_X + PAULI_Z) / np.sqrt(2.0)
SZ = (I2 + PAULI_Z) / 2 + 1j * (I2 - PAULI_Z) / 2
SZDG = SZ.conj().T
SX = np.cos(np.pi / 4) * I2 - 1j * np.sin(np.pi / 4) * PAULI_X
I4 = _kron(I2, I2)
ZI = _kron(PAULI_Z, I2)
IZ = _kron(I2, PAULI_Z)
ZZ = _pauli_matrix(pc.Z(0) & pc.Z(1))
CZ = (I4 + ZI + IZ - ZZ) / 2
SZZ = np.cos(np.pi / 4) * I4 - 1j * np.sin(np.pi / 4) * ZZ
SZZDG = SZZ.conj().T
CX = _kron(I2, H) @ CZ @ _kron(I2, H)
SXX = _kron(H, H) @ SZZ @ _kron(H, H)


def _equiv_up_to_global_phase(left: np.ndarray, right: np.ndarray, *, atol: float = 1e-10) -> bool:
    flat_right = right.ravel()
    flat_left = left.ravel()
    nonzero = np.flatnonzero(np.abs(flat_right) > atol)
    if nonzero.size == 0:
        return bool(np.allclose(left, right, atol=atol))
    ratio = flat_left[nonzero[0]] / flat_right[nonzero[0]]
    return bool(np.allclose(flat_left, ratio * flat_right, atol=atol))


def _pauli_from_bits(x_bit: bool, z_bit: bool) -> np.ndarray:
    op = I2
    if x_bit:
        op = PAULI_X @ op
    if z_bit:
        op = PAULI_Z @ op
    return op


def _bits_from_pauli(op: np.ndarray) -> tuple[bool, bool]:
    for x_bit in (False, True):
        for z_bit in (False, True):
            if _equiv_up_to_global_phase(op, _pauli_from_bits(x_bit, z_bit)):
                return x_bit, z_bit
    msg = f"not a Pauli up to phase:\n{op}"
    raise AssertionError(msg)


def _matrix_frame_update(
    unitary: np.ndarray,
    x_a: bool,
    z_a: bool,
    x_b: bool,
    z_b: bool,
) -> tuple[bool, bool, bool, bool]:
    before = _kron(_pauli_from_bits(x_a, z_a), _pauli_from_bits(x_b, z_b))
    after = unitary @ before @ unitary.conj().T
    basis = (I2, PAULI_X, PAULI_Z, PAULI_X @ PAULI_Z)
    for left in basis:
        for right in basis:
            candidate = _kron(left, right)
            if _equiv_up_to_global_phase(after, candidate):
                ax, az = _bits_from_pauli(left)
                bx, bz = _bits_from_pauli(right)
                return ax, az, bx, bz
    msg = f"not a tensor-product Pauli up to phase:\n{after}"
    raise AssertionError(msg)


def _record_parity(raw_records: np.ndarray, records: list[int]) -> np.ndarray:
    num_measurements = raw_records.shape[1]
    if not records:
        return np.zeros(raw_records.shape[0], dtype=bool)
    indices = [num_measurements + int(record) for record in records]
    return np.bitwise_xor.reduce(raw_records[:, indices], axis=1)


def _assert_noiseless_record_metadata_is_zero(stim_text: str, tick_circuit: object) -> None:
    circuit = stim.Circuit(stim_text)
    circuit.detector_error_model()

    raw_records = circuit.compile_sampler(seed=20260612).sample(shots=32)
    detectors = json.loads(tick_circuit.get_meta("detectors") or "[]")
    observables = json.loads(tick_circuit.get_meta("observables") or "[]")

    for detector in detectors:
        assert not np.any(_record_parity(raw_records, detector["records"]))
    for observable in observables:
        assert not np.any(_record_parity(raw_records, observable["records"]))

    det_samples, obs_samples = circuit.compile_detector_sampler(seed=20260612).sample(
        shots=32,
        separate_observables=True,
    )
    assert det_samples.shape == (32, circuit.num_detectors)
    assert obs_samples.shape == (32, circuit.num_observables)
    assert not np.any(det_samples)
    assert not np.any(obs_samples)


def test_szz_unitary_identities() -> None:
    assert _equiv_up_to_global_phase(SZZ @ _kron(SZDG, SZDG), CZ)
    assert _equiv_up_to_global_phase(SZZ @ ZZ, SZZDG)
    assert _equiv_up_to_global_phase(
        _kron(I2, H) @ SZZ @ _kron(I2, H),
        CX @ _kron(SZ, SX),
    )


@pytest.mark.parametrize("x_a", [False, True])
@pytest.mark.parametrize("z_a", [False, True])
@pytest.mark.parametrize("x_b", [False, True])
@pytest.mark.parametrize("z_b", [False, True])
def test_szz_frame_rules_match_matrix_oracle(
    x_a: bool,
    z_a: bool,
    x_b: bool,
    z_b: bool,
) -> None:
    assert _propagate_szz_frame_bits(x_a, z_a, x_b, z_b) == _matrix_frame_update(
        SZZ,
        x_a,
        z_a,
        x_b,
        z_b,
    )
    assert _propagate_szz_frame_bits(x_a, z_a, x_b, z_b) == _matrix_frame_update(
        SZZDG,
        x_a,
        z_a,
        x_b,
        z_b,
    )
    assert _propagate_compensated_szz_frame_bits(x_a, z_a, x_b, z_b) == _matrix_frame_update(
        CZ,
        x_a,
        z_a,
        x_b,
        z_b,
    )
    assert _propagate_sxx_frame_bits(x_a, z_a, x_b, z_b) == _matrix_frame_update(
        SXX,
        x_a,
        z_a,
        x_b,
        z_b,
    )


def test_szz_residual_classifier_and_default_plan() -> None:
    patch = SurfacePatch.create(distance=3)
    plan = _default_szz_residual_plan(patch)

    assert _szz_residual_class(0) == "identity"
    assert _szz_residual_class(4) == "identity"
    assert _szz_residual_class(2) == "pauli"
    assert _szz_residual_class(-2) == "pauli"
    assert _szz_residual_class(1) == "odd"
    assert _szz_residual_class(-1) == "odd"

    assert {entry.sign for entry in plan.signs} == {-1, 1}
    assert {entry.gate for entry in plan.boundary_compensations} == {
        "SX",
        "SXDG",
        "SZ",
        "SZDG",
    }
    assert {entry.pauli for entry in plan.class2_residuals} == {"X", "Z"}


def test_szz_bad_sign_vector_rejected_loudly() -> None:
    patch = SurfacePatch.create(distance=3)
    signs = list(_default_szz_sign_vector(patch))
    bad_index = next(i for i, entry in enumerate(signs) if entry.sign == -1)
    bad_entry = signs[bad_index]
    signs[bad_index] = SzzTouchSign(
        bad_entry.stabilizer_type,
        bad_entry.stabilizer_index,
        bad_entry.data_qubit,
        1,
    )

    with pytest.raises(ValueError, match="ancilla residual is pauli"):
        _validate_szz_sign_vector(patch, tuple(signs))


def test_szz_builder_emits_szz_template_and_rejects_stage_later_features() -> None:
    patch = SurfacePatch.create(distance=3)

    cx_ops, _ = build_surface_code_circuit(patch, num_rounds=1, interaction_basis="cx")
    assert any(op.op_type == OpType.CX for op in cx_ops)
    assert not any(op.op_type in {OpType.SZZ, OpType.SZZDG} for op in cx_ops)

    szz_ops, _ = build_surface_code_circuit(patch, num_rounds=1, interaction_basis="szz")
    op_types = {op.op_type for op in szz_ops}
    assert OpType.CX not in op_types
    assert {OpType.SZZ, OpType.SZZDG, OpType.SX, OpType.SXDG, OpType.SZ, OpType.SZDG} <= op_types
    szz_gate_count = sum(op.op_type in {OpType.SZZ, OpType.SZZDG} for op in szz_ops)
    data_compensation_count = sum(
        op.label.startswith("szz_touch_comp:") and op.op_type in {OpType.SX, OpType.SXDG, OpType.SZ, OpType.SZDG}
        for op in szz_ops
    )
    assert data_compensation_count == szz_gate_count

    with pytest.raises(ValueError, match="does not yet support constrained ancilla budgets"):
        build_surface_code_circuit(patch, num_rounds=1, ancilla_budget=1, interaction_basis="szz")

    with pytest.raises(ValueError, match="twirl integration is staged later"):
        build_surface_code_circuit(
            patch,
            num_rounds=1,
            twirl=TwirlConfig(),
            interaction_basis="szz",
        )


def test_szz_tick_circuit_uses_named_szz_gates() -> None:
    patch = SurfacePatch.create(distance=3)
    tick_circuit = generate_tick_circuit_from_patch(
        patch,
        num_rounds=1,
        add_detectors=False,
        interaction_basis="szz",
    )

    gate_names = {
        gate.gate_type.name
        for tick_index in range(tick_circuit.num_ticks())
        for gate in tick_circuit.get_tick(tick_index).gate_batches()
    }
    assert "CX" not in gate_names
    assert {"SZZ", "SZZdg", "SX", "SXdg", "SZ", "SZdg"} <= gate_names


def test_szz_forward_flow_merges_h_sandwich_before_host() -> None:
    ops = [
        SurfaceCircuitStep(OpType.ALLOC, [0]),
        SurfaceCircuitStep(OpType.ALLOC, [1]),
        SurfaceCircuitStep(OpType.H, [0]),
        SurfaceCircuitStep(OpType.H, [0]),
        SurfaceCircuitStep(OpType.SZZ, [0, 1], "g"),
        SurfaceCircuitStep(OpType.MEASURE, [0], "m0"),
        SurfaceCircuitStep(OpType.MEASURE, [1], "m1"),
    ]

    summary = _analyze_szz_forward_flow(ops)

    assert summary.abstract_single_qubit_ops == 2
    assert summary.physical_prefix_pulses == 0
    assert summary.free_standing_single_qubit_ops == 0
    assert summary.pulses == ()


def test_szz_forward_flow_carries_virtual_z_to_measurement() -> None:
    ops = [
        SurfaceCircuitStep(OpType.ALLOC, [0]),
        SurfaceCircuitStep(OpType.ALLOC, [1]),
        SurfaceCircuitStep(OpType.SZ, [0]),
        SurfaceCircuitStep(OpType.SZZ, [0, 1], "g"),
        SurfaceCircuitStep(OpType.MEASURE, [0], "m0"),
        SurfaceCircuitStep(OpType.MEASURE, [1], "m1"),
    ]

    summary = _analyze_szz_forward_flow(ops)

    assert summary.physical_prefix_pulses == 0
    assert summary.virtual_z_two_qubit_carries == 1
    assert summary.virtual_z_measure_discards == 1
    assert [event.kind for event in summary.pulses] == [
        "virtual_z_two_qubit_carry",
        "virtual_z_measure_discard",
    ]


def test_szz_forward_flow_counts_physical_prefixes_at_hosts() -> None:
    ops = [
        SurfaceCircuitStep(OpType.ALLOC, [0]),
        SurfaceCircuitStep(OpType.ALLOC, [1]),
        SurfaceCircuitStep(OpType.H, [0]),
        SurfaceCircuitStep(OpType.SZZ, [0, 1], "g"),
        SurfaceCircuitStep(OpType.H, [1]),
        SurfaceCircuitStep(OpType.MEASURE, [0], "m0"),
        SurfaceCircuitStep(OpType.MEASURE, [1], "m1"),
    ]

    summary = _analyze_szz_forward_flow(ops)

    assert summary.physical_prefix_pulses == 2
    assert summary.two_qubit_prefix_pulses == 1
    assert summary.measurement_prefix_pulses == 1
    assert [event.kind for event in summary.pulses] == [
        "physical_two_qubit_prefix",
        "physical_measurement_prefix",
    ]


@pytest.mark.parametrize(
    ("basis", "expected"),
    [
        (
            "Z",
            {
                "abstract_single_qubit_ops": 108,
                "physical_prefix_pulses": 54,
                "two_qubit_prefix_pulses": 38,
                "measurement_prefix_pulses": 16,
                "virtual_z_two_qubit_carries": 10,
                "virtual_z_measure_discards": 5,
            },
        ),
        (
            "X",
            {
                "abstract_single_qubit_ops": 102,
                "physical_prefix_pulses": 54,
                "two_qubit_prefix_pulses": 37,
                "measurement_prefix_pulses": 17,
                "virtual_z_two_qubit_carries": 10,
                "virtual_z_measure_discards": 3,
            },
        ),
    ],
)
def test_szz_forward_flow_surface_pulse_count_snapshot(basis: str, expected: dict[str, int]) -> None:
    patch = SurfacePatch.create(distance=3)
    ops, _ = build_surface_code_circuit(patch, num_rounds=1, basis=basis, interaction_basis="szz")

    summary = _analyze_szz_forward_flow(ops)

    assert summary.two_qubit_gates == 36
    assert summary.measurements == 21
    assert summary.prep_events == 21
    assert summary.free_standing_single_qubit_ops == 0
    for field, value in expected.items():
        assert getattr(summary, field) == value
    assert summary.physical_prefix_pulses < summary.abstract_single_qubit_ops


def test_szz_direct_renderers_accept_interaction_basis() -> None:
    patch = SurfacePatch.create(distance=3)

    stim_text = generate_stim_from_patch(
        patch,
        num_rounds=1,
        interaction_basis="szz",
        add_detectors=False,
    )
    assert "CX" not in stim_text
    assert "SQRT_X" in stim_text
    assert "S_DAG" in stim_text
    assert "SQRT_ZZ" in stim_text
    assert "SQRT_ZZ_DAG" in stim_text

    dag_circuit = generate_dag_circuit_from_patch(patch, num_rounds=1, interaction_basis="szz")
    gate_names = {dag_circuit.gate(node).gate_type.name for node in dag_circuit.nodes()}
    assert "CX" not in gate_names
    assert {"SZZ", "SZZdg", "SX", "SXdg", "SZ", "SZdg"} <= gate_names


def test_szz_detector_paths_accept_abstract_and_traced_qis_basis() -> None:
    patch = SurfacePatch.create(distance=3)

    stim_text = generate_stim_from_patch(patch, num_rounds=1, interaction_basis="szz")
    assert "DETECTOR" in stim_text

    tick_circuit = generate_tick_circuit_from_patch(patch, num_rounds=1, interaction_basis="szz")
    assert int(tick_circuit.get_meta("num_detectors")) > 0

    memory_circuit = build_memory_circuit(patch=patch, rounds=1, interaction_basis="szz")
    assert int(memory_circuit.get_meta("num_detectors")) == int(tick_circuit.get_meta("num_detectors"))

    traced_memory_circuit = build_memory_circuit(
        patch=patch,
        rounds=1,
        circuit_source="traced_qis",
        interaction_basis="szz",
    )
    assert traced_memory_circuit.get_meta("circuit_source") == "traced_qis"
    assert int(traced_memory_circuit.get_meta("num_detectors")) == int(tick_circuit.get_meta("num_detectors"))


@pytest.mark.parametrize("distance", [3, 5])
@pytest.mark.parametrize("basis", ["Z", "X"])
def test_szz_noiseless_detector_record_equivalence(distance: int, basis: str) -> None:
    patch = SurfacePatch.create(distance=distance)

    cx_text = generate_stim_from_patch(patch, num_rounds=3, basis=basis, interaction_basis="cx")
    szz_text = generate_stim_from_patch(patch, num_rounds=3, basis=basis, interaction_basis="szz")
    cx_tick = generate_tick_circuit_from_patch(patch, num_rounds=3, basis=basis, interaction_basis="cx")
    szz_tick = generate_tick_circuit_from_patch(patch, num_rounds=3, basis=basis, interaction_basis="szz")

    cx_circuit = stim.Circuit(cx_text)
    szz_circuit = stim.Circuit(szz_text)
    assert cx_circuit.num_measurements == szz_circuit.num_measurements
    assert cx_circuit.num_detectors == szz_circuit.num_detectors
    assert cx_circuit.num_observables == szz_circuit.num_observables
    assert cx_tick.get_meta("detectors") == szz_tick.get_meta("detectors")
    assert cx_tick.get_meta("observables") == szz_tick.get_meta("observables")

    _assert_noiseless_record_metadata_is_zero(cx_text, cx_tick)
    _assert_noiseless_record_metadata_is_zero(szz_text, szz_tick)


def test_szz_native_dem_path_uses_interaction_basis() -> None:
    patch = SurfacePatch.create(distance=3)
    noise = NoiseModel(p1=0.001, p2=0.01, p_meas=0.001, p_prep=0.001)

    cx_dem = generate_circuit_level_dem_from_builder(
        patch,
        num_rounds=2,
        noise=noise,
        interaction_basis="cx",
    )
    szz_dem = generate_circuit_level_dem_from_builder(
        patch,
        num_rounds=2,
        noise=noise,
        interaction_basis="szz",
    )

    assert cx_dem != szz_dem
    assert stim.DetectorErrorModel(szz_dem).num_detectors > 0
