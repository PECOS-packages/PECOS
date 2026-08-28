# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import json
import re

import numpy as np
import pecos as pc
import pytest
import stim
from pecos._traced_circuit import normalize_traced_tick_circuit
from pecos.qec.surface import NoiseParameters, SurfacePatch, TwirlConfig
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
    generate_dem_from_tick_circuit,
    generate_dem_from_tick_circuit_via_stim,
    generate_stim_from_patch,
    generate_tick_circuit_from_patch,
)
from pecos.qec.surface.decode import (
    _dem_string_from_cached_surface_topology,
    _surface_native_topology,
    _surface_patch_cache_key,
    build_memory_circuit,
    build_native_sampler,
    generate_circuit_level_dem_from_builder,
)
from pecos.quantum import PHYSICAL_DURATION_META_KEY


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


def _raw_dem_errors(dem_text: str) -> dict[str, float]:
    errors: dict[str, float] = {}
    for line in dem_text.splitlines():
        match = re.match(r"error\(([^)]+)\)\s*(.*)", line.strip())
        if match:
            target = match.group(2).strip()
            errors[target] = errors.get(target, 0.0) + float(match.group(1))
    return errors


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


def _gate_labels_for_tick(tick_circuit: object, tick_index: int) -> list[str | None]:
    return [
        tick_circuit.get_gate_meta(tick_index, gate_index, "label")
        for gate_index, _gate in enumerate(tick_circuit.get_tick(tick_index).gate_batches())
    ]


def _retag_virtual_prefix_duration(tick_circuit: object, duration: float) -> int:
    count = 0
    for tick_index in range(tick_circuit.num_ticks()):
        for gate_index, _gate in enumerate(tick_circuit.get_tick(tick_index).gate_batches()):
            label = tick_circuit.get_gate_meta(tick_index, gate_index, "label")
            if label and label.startswith("szz_virtual_prefix:"):
                tick_circuit.set_gate_meta(tick_index, gate_index, PHYSICAL_DURATION_META_KEY, duration)
                count += 1
    return count


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

    constrained_szz_ops, _ = build_surface_code_circuit(
        patch,
        num_rounds=1,
        ancilla_budget=1,
        interaction_basis="szz",
    )
    constrained_szz_gate_count = sum(op.op_type in {OpType.SZZ, OpType.SZZDG} for op in constrained_szz_ops)
    constrained_compensation_count = sum(
        op.label.startswith("szz_touch_comp:") and op.op_type in {OpType.SX, OpType.SXDG, OpType.SZ, OpType.SZDG}
        for op in constrained_szz_ops
    )
    assert constrained_szz_gate_count == szz_gate_count
    assert constrained_compensation_count == constrained_szz_gate_count

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


@pytest.mark.xfail(
    strict=True,
    raises=ValueError,
    reason="#498: SZZ final provenance reversed",
)
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


@pytest.mark.xfail(
    strict=True,
    raises=ValueError,
    reason="#498: SZZ final provenance reversed",
)
def test_szz_runtime_barriers_allow_strict_traced_hosted_order() -> None:
    patch = SurfacePatch.create(distance=3)

    tick_circuit = build_memory_circuit(
        patch=patch,
        rounds=1,
        circuit_source="traced_qis",
        interaction_basis="szz",
        szz_runtime_barriers="data-prefix",
        require_hosted_operation_order=True,
    )
    assert tick_circuit.get_meta("circuit_source") == "traced_qis"
    assert int(tick_circuit.get_meta("num_detectors")) > 0

    dem = generate_circuit_level_dem_from_builder(
        patch,
        num_rounds=1,
        noise=NoiseParameters(p1=0.0, p2=0.001, p_meas=0.0, p_prep=0.0),
        circuit_source="traced_qis",
        interaction_basis="szz",
        szz_runtime_barriers="data-prefix",
        require_hosted_operation_order=True,
    )
    assert stim.DetectorErrorModel(dem).num_detectors == int(tick_circuit.get_meta("num_detectors"))


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


@pytest.mark.parametrize(
    "check_plan",
    [
        "szz_balanced_data_round_order_1032_v1",
        "szz_balanced_data_round_order_3102_v1",
    ],
)
@pytest.mark.parametrize("basis", ["Z", "X"])
def test_round_order_szz_noiseless_detector_record_equivalence(
    basis: str,
    check_plan: str,
) -> None:
    patch = SurfacePatch.create(distance=3)
    baseline_plan = "szz_balanced_data_v1"

    baseline_text = generate_stim_from_patch(
        patch,
        num_rounds=3,
        basis=basis,
        ancilla_budget=2,
        check_plan=baseline_plan,
    )
    round_order_text = generate_stim_from_patch(
        patch,
        num_rounds=3,
        basis=basis,
        ancilla_budget=2,
        check_plan=check_plan,
    )
    baseline_tick = generate_tick_circuit_from_patch(
        patch,
        num_rounds=3,
        basis=basis,
        ancilla_budget=2,
        check_plan=baseline_plan,
    )
    round_order_tick = generate_tick_circuit_from_patch(
        patch,
        num_rounds=3,
        basis=basis,
        ancilla_budget=2,
        check_plan=check_plan,
    )

    baseline_circuit = stim.Circuit(baseline_text)
    round_order_circuit = stim.Circuit(round_order_text)
    assert round_order_text != baseline_text
    assert round_order_circuit.num_measurements == baseline_circuit.num_measurements
    assert round_order_circuit.num_detectors == baseline_circuit.num_detectors
    assert round_order_circuit.num_observables == baseline_circuit.num_observables
    assert round_order_tick.get_meta("detectors") == baseline_tick.get_meta("detectors")
    assert round_order_tick.get_meta("observables") == baseline_tick.get_meta("observables")

    _assert_noiseless_record_metadata_is_zero(round_order_text, round_order_tick)


def test_szz_native_dem_path_uses_interaction_basis() -> None:
    patch = SurfacePatch.create(distance=3)
    noise = NoiseParameters(p1=0.0, p2=0.01, p2_weights={"ZI": 1.0}, p_meas=0.001, p_prep=0.001)

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


def test_szz_native_dem_respects_gate_specific_p2_overrides() -> None:
    patch = SurfacePatch.create(distance=3)

    inherited_dem = generate_circuit_level_dem_from_builder(
        patch,
        num_rounds=2,
        noise=NoiseParameters(p1=0.0, p2=0.01, p2_weights={"ZI": 1.0}),
        interaction_basis="szz",
    )
    no_szz_dem = generate_circuit_level_dem_from_builder(
        patch,
        num_rounds=2,
        noise=NoiseParameters(p1=0.0, p2=0.01, p2_szz=0.0, p2_weights={"ZI": 1.0}),
        interaction_basis="szz",
    )
    no_szzdg_dem = generate_circuit_level_dem_from_builder(
        patch,
        num_rounds=2,
        noise=NoiseParameters(p1=0.0, p2=0.01, p2_szzdg=0.0, p2_weights={"ZI": 1.0}),
        interaction_basis="szz",
    )
    override_only_dem = generate_circuit_level_dem_from_builder(
        patch,
        num_rounds=2,
        noise=NoiseParameters(
            p1=0.0,
            p2=0.0,
            p2_szz=0.01,
            p2_szzdg=0.01,
            p2_weights={"ZI": 1.0},
        ),
        interaction_basis="szz",
    )

    assert no_szz_dem != inherited_dem
    assert no_szzdg_dem != inherited_dem
    assert "error(" in override_only_dem


def test_szz_native_influence_sampler_respects_override_only_p2() -> None:
    patch = SurfacePatch.create(distance=3)

    zero_sampler = build_native_sampler(
        patch,
        num_rounds=2,
        noise=NoiseParameters(p1=0.0, p2=0.0, p2_szz=0.0, p2_szzdg=0.0, p2_weights={"ZI": 1.0}),
        interaction_basis="szz",
        sampling_model="influence_dem",
    )
    active_sampler = build_native_sampler(
        patch,
        num_rounds=2,
        noise=NoiseParameters(p1=0.0, p2=0.0, p2_szz=0.01, p2_szzdg=0.01, p2_weights={"ZI": 1.0}),
        interaction_basis="szz",
        sampling_model="influence_dem",
    )

    assert "mechanisms=0" in repr(zero_sampler.sampler)
    assert "mechanisms=0" not in repr(active_sampler.sampler)


@pytest.mark.parametrize("basis", ["Z", "X"])
def test_szz_prefix_lowering_preserves_p2_influence_dem(basis: str) -> None:
    patch = SurfacePatch.create(distance=3)
    patch_key = _surface_patch_cache_key(patch)
    noise = NoiseParameters(p1=0.0, p2=0.01, p_meas=0.0, p_prep=0.0)

    plain = _surface_native_topology(
        patch_key,
        1,
        basis,
        None,
        "abstract",
        False,
        interaction_basis="szz",
        szz_physical_prefixes=False,
    )
    lowered = _surface_native_topology(
        patch_key,
        1,
        basis,
        None,
        "abstract",
        False,
        interaction_basis="szz",
        szz_physical_prefixes=True,
    )

    assert _dem_string_from_cached_surface_topology(
        lowered,
        noise,
        decompose_errors=False,
    ) == _dem_string_from_cached_surface_topology(
        plain,
        noise,
        decompose_errors=False,
    )


@pytest.mark.parametrize("basis", ["Z", "X"])
def test_szz_lowered_native_dem_matches_stim_for_prefix_noise(basis: str) -> None:
    patch = SurfacePatch.create(distance=3)
    tick_circuit = generate_tick_circuit_from_patch(
        patch,
        num_rounds=3,
        basis=basis,
        interaction_basis="szz",
        szz_physical_prefixes=True,
    )
    p2 = 0.006
    noise_args = {
        "p1": p2 / 30,
        "p2": p2,
        "p_meas": p2 / 3,
        "p_prep": p2 / 3,
    }

    native_errors = _raw_dem_errors(
        generate_dem_from_tick_circuit(
            tick_circuit,
            decompose_errors=False,
            **noise_args,
        ),
    )
    stim_errors = _raw_dem_errors(
        generate_dem_from_tick_circuit_via_stim(
            tick_circuit,
            decompose_errors=False,
            **noise_args,
        ),
    )

    assert set(native_errors) == set(stim_errors)
    for target, native_probability in native_errors.items():
        stim_probability = stim_errors[target]
        rel_diff = abs(native_probability - stim_probability) / max(
            native_probability,
            stim_probability,
            1e-12,
        )
        assert rel_diff < 0.005, (
            f"{basis} lowered SZZ DEM mismatch for {target}: "
            f"PECOS={native_probability:.8f}, Stim={stim_probability:.8f}"
        )


@pytest.mark.parametrize("basis", ["Z", "X"])
def test_szz_abstract_p2_only_raw_dem_matches_cx(basis: str) -> None:
    patch = SurfacePatch.create(distance=3)
    noise_args = {
        "p1": 0.0,
        "p2": 0.006,
        "p_meas": 0.0,
        "p_prep": 0.0,
    }
    cx_tick_circuit = generate_tick_circuit_from_patch(
        patch,
        num_rounds=3,
        basis=basis,
        interaction_basis="cx",
    )
    szz_tick_circuit = generate_tick_circuit_from_patch(
        patch,
        num_rounds=3,
        basis=basis,
        interaction_basis="szz",
        szz_physical_prefixes=False,
    )

    assert generate_dem_from_tick_circuit(
        cx_tick_circuit,
        decompose_errors=False,
        **noise_args,
    ) == generate_dem_from_tick_circuit(
        szz_tick_circuit,
        decompose_errors=False,
        **noise_args,
    )


@pytest.mark.parametrize("basis", ["Z", "X"])
def test_szz_prefix_lowering_emits_dedicated_prefix_ticks(basis: str) -> None:
    patch = SurfacePatch.create(distance=3)
    tick_circuit = generate_tick_circuit_from_patch(
        patch,
        num_rounds=1,
        basis=basis,
        interaction_basis="szz",
        szz_physical_prefixes=True,
    )

    saw_physical_prefix = False
    saw_virtual_prefix = False
    for tick_index in range(tick_circuit.num_ticks()):
        tick = tick_circuit.get_tick(tick_index)
        labels = _gate_labels_for_tick(tick_circuit, tick_index)
        prefix_labels = [label for label in labels if label and label.startswith("szz_")]
        if not prefix_labels:
            continue

        assert len(prefix_labels) == len(labels)
        if prefix_labels[0].startswith("szz_virtual_prefix:"):
            saw_virtual_prefix = True
            assert all(label.startswith("szz_virtual_prefix:") for label in prefix_labels)
            for gate_index, gate in enumerate(tick.gate_batches()):
                assert gate.gate_type.name == "Z"
                assert tick_circuit.get_gate_meta(tick_index, gate_index, PHYSICAL_DURATION_META_KEY) == 0.0
        else:
            saw_physical_prefix = True
            assert all(label.startswith("szz_physical_prefix:") for label in prefix_labels)
            assert {gate.gate_type.name for gate in tick.gate_batches()} <= {
                "H",
                "F",
                "Fdg",
                "SXdg",
                "SY",
            }
            for gate_index, _gate in enumerate(tick.gate_batches()):
                assert tick_circuit.get_gate_meta(tick_index, gate_index, PHYSICAL_DURATION_META_KEY) is None

    assert saw_physical_prefix
    assert saw_virtual_prefix


@pytest.mark.parametrize("sampling_model", ["dem", "influence_dem"])
def test_szz_native_sampler_accepts_p1_with_physical_prefix_lowering(sampling_model: str) -> None:
    patch = SurfacePatch.create(distance=3)

    sampler = build_native_sampler(
        patch,
        num_rounds=1,
        noise=NoiseParameters(p1=0.001),
        interaction_basis="szz",
        sampling_model=sampling_model,
    )
    det_events, obs_flips = sampler.sample(4, seed=20260612)

    assert det_events.shape == (4, sampler.num_detectors)
    assert obs_flips.shape == (4, sampler.num_observables)
    if sampling_model == "influence_dem":
        assert "mechanisms=0" not in repr(sampler.sampler)


def test_szz_native_dem_rejects_traced_qis_idle_noise() -> None:
    patch = SurfacePatch.create(distance=3)

    with pytest.raises(
        ValueError,
        match=r"dedicated idle noise with circuit_source='traced_qis'.*explicit post-flow idle locations",
    ):
        generate_circuit_level_dem_from_builder(
            patch,
            num_rounds=1,
            noise=NoiseParameters(p_idle=0.001),
            interaction_basis="szz",
            circuit_source="traced_qis",
        )


@pytest.mark.xfail(
    strict=True,
    raises=ValueError,
    reason="#498: SZZ final provenance reversed",
)
@pytest.mark.parametrize("basis", ["Z", "X"])
def test_szz_traced_qis_native_dem_matches_stim_for_p1(basis: str) -> None:
    from pecos.qec.surface.decode import _build_surface_tick_circuit_for_native_model

    patch = SurfacePatch.create(distance=3)
    tick_circuit = _build_surface_tick_circuit_for_native_model(
        patch,
        num_rounds=1,
        basis=basis,
        circuit_source="traced_qis",
        interaction_basis="szz",
    )
    normalize_traced_tick_circuit(tick_circuit, context="SZZ traced-QIS p1 test")
    noise_args = {
        "p1": 0.001,
        "p2": 0.0,
        "p_meas": 0.0,
        "p_prep": 0.0,
    }

    native_errors = _raw_dem_errors(
        generate_dem_from_tick_circuit(
            tick_circuit,
            decompose_errors=False,
            **noise_args,
        ),
    )
    stim_errors = _raw_dem_errors(
        generate_dem_from_tick_circuit_via_stim(
            tick_circuit,
            decompose_errors=False,
            **noise_args,
        ),
    )

    assert set(native_errors) == set(stim_errors)
    for target, native_probability in native_errors.items():
        stim_probability = stim_errors[target]
        rel_diff = abs(native_probability - stim_probability) / max(
            native_probability,
            stim_probability,
            1e-12,
        )
        assert rel_diff < 0.005, (
            f"{basis} traced-QIS SZZ p1 DEM mismatch for {target}: "
            f"PECOS={native_probability:.8f}, Stim={stim_probability:.8f}"
        )


@pytest.mark.parametrize(
    "check_plan",
    [
        "szz_balanced_data_round_order_1032_v1",
        "szz_balanced_data_round_order_3102_v1",
    ],
)
@pytest.mark.parametrize("basis", ["Z", "X"])
@pytest.mark.xfail(
    strict=True,
    raises=ValueError,
    reason="#498: SZZ final provenance reversed",
)
def test_round_order_szz_traced_qis_native_dem_matches_stim_for_p1(
    basis: str,
    check_plan: str,
) -> None:
    from pecos.qec.surface.decode import _build_surface_tick_circuit_for_native_model

    patch = SurfacePatch.create(distance=3)
    tick_circuit = _build_surface_tick_circuit_for_native_model(
        patch,
        num_rounds=1,
        basis=basis,
        ancilla_budget=2,
        circuit_source="traced_qis",
        check_plan=check_plan,
    )
    normalize_traced_tick_circuit(tick_circuit, context="round-order SZZ traced-QIS p1 test")
    noise_args = {
        "p1": 0.001,
        "p2": 0.0,
        "p_meas": 0.0,
        "p_prep": 0.0,
    }

    native_errors = _raw_dem_errors(
        generate_dem_from_tick_circuit(
            tick_circuit,
            decompose_errors=False,
            **noise_args,
        ),
    )
    stim_errors = _raw_dem_errors(
        generate_dem_from_tick_circuit_via_stim(
            tick_circuit,
            decompose_errors=False,
            **noise_args,
        ),
    )

    assert set(native_errors) == set(stim_errors)
    for target, native_probability in native_errors.items():
        stim_probability = stim_errors[target]
        rel_diff = abs(native_probability - stim_probability) / max(
            native_probability,
            stim_probability,
            1e-12,
        )
        assert rel_diff < 0.005, (
            f"{basis} round-order traced-QIS SZZ p1 DEM mismatch for {target}: "
            f"PECOS={native_probability:.8f}, Stim={stim_probability:.8f}"
        )


@pytest.mark.xfail(
    strict=True,
    raises=ValueError,
    reason="#498: SZZ final provenance reversed",
)
def test_szz_public_native_dem_accepts_traced_qis_p1() -> None:
    patch = SurfacePatch.create(distance=3)
    dem = generate_circuit_level_dem_from_builder(
        patch,
        num_rounds=1,
        noise=NoiseParameters(p1=0.001),
        interaction_basis="szz",
        circuit_source="traced_qis",
    )

    assert "error(" in dem
    assert stim.DetectorErrorModel(dem).num_detectors > 0


@pytest.mark.xfail(
    strict=True,
    raises=ValueError,
    reason="#498: SZZ final provenance reversed",
)
@pytest.mark.parametrize("basis", ["Z", "X"])
def test_szz_public_traced_qis_dem_matches_stim_with_z_frame_p1_free(basis: str) -> None:
    from pecos.qec.surface.decode import _build_surface_tick_circuit_for_native_model

    patch = SurfacePatch.create(distance=3)
    tick_circuit = _build_surface_tick_circuit_for_native_model(
        patch,
        num_rounds=1,
        basis=basis,
        circuit_source="traced_qis",
        interaction_basis="szz",
    )
    normalize_traced_tick_circuit(tick_circuit, context="SZZ public traced-QIS p1 test")
    noise = NoiseParameters(p1=0.001)

    native_errors = _raw_dem_errors(
        generate_circuit_level_dem_from_builder(
            patch,
            num_rounds=1,
            noise=noise,
            basis=basis,
            decompose_errors=False,
            circuit_source="traced_qis",
            interaction_basis="szz",
        ),
    )
    stim_errors = _raw_dem_errors(
        generate_dem_from_tick_circuit_via_stim(
            tick_circuit,
            decompose_errors=False,
            p1=noise.p1,
            p2=0.0,
            p_meas=0.0,
            p_prep=0.0,
            p1_gate_rates={"Z": 0.0, "SZ": 0.0, "SZdg": 0.0},
        ),
    )

    assert set(native_errors) == set(stim_errors)
    for target, native_probability in native_errors.items():
        stim_probability = stim_errors[target]
        rel_diff = abs(native_probability - stim_probability) / max(
            native_probability,
            stim_probability,
            1e-12,
        )
        assert rel_diff < 0.005, (
            f"{basis} public traced-QIS SZZ p1 DEM mismatch for {target}: "
            f"PECOS={native_probability:.8f}, Stim={stim_probability:.8f}"
        )


def test_szz_z_frame_gates_are_p1_free_for_native_noise() -> None:
    from types import SimpleNamespace

    from pecos.qec.surface.decode import _szz_z_frame_p1_gate_rates

    assert _szz_z_frame_p1_gate_rates(SimpleNamespace(z_frame_gate_p1_free=True)) == {
        "Z": 0.0,
        "SZ": 0.0,
        "SZdg": 0.0,
    }
    assert _szz_z_frame_p1_gate_rates(SimpleNamespace(z_frame_gate_p1_free=False)) is None


def test_szz_native_dem_accepts_p1_with_physical_prefix_lowering() -> None:
    patch = SurfacePatch.create(distance=3)
    dem = generate_circuit_level_dem_from_builder(
        patch,
        num_rounds=1,
        noise=NoiseParameters(p1=0.001),
        interaction_basis="szz",
    )

    assert "error(" in dem
    assert stim.DetectorErrorModel(dem).num_detectors > 0


@pytest.mark.parametrize("sampling_model", ["dem", "influence_dem"])
def test_szz_native_sampler_accepts_idle_with_physical_prefix_lowering(sampling_model: str) -> None:
    patch = SurfacePatch.create(distance=3)

    sampler = build_native_sampler(
        patch,
        num_rounds=1,
        noise=NoiseParameters(p_idle=0.001),
        interaction_basis="szz",
        sampling_model=sampling_model,
    )
    det_events, obs_flips = sampler.sample(4, seed=20260612)

    assert det_events.shape == (4, sampler.num_detectors)
    assert obs_flips.shape == (4, sampler.num_observables)
    if sampling_model == "influence_dem":
        assert "mechanisms=0" not in repr(sampler.sampler)


def test_szz_native_dem_accepts_idle_with_physical_prefix_lowering() -> None:
    patch = SurfacePatch.create(distance=3)
    dem = generate_circuit_level_dem_from_builder(
        patch,
        num_rounds=1,
        noise=NoiseParameters(p_idle=0.001),
        interaction_basis="szz",
    )

    assert "error(" in dem
    assert stim.DetectorErrorModel(dem).num_detectors > 0


@pytest.mark.parametrize("basis", ["Z", "X"])
def test_szz_idle_dem_uses_lowered_prefix_topology(basis: str) -> None:
    patch = SurfacePatch.create(distance=3)
    patch_key = _surface_patch_cache_key(patch)
    noise = NoiseParameters().with_p_idle_linear(0.01, {"Z": 1.0})

    actual = generate_circuit_level_dem_from_builder(
        patch,
        num_rounds=1,
        basis=basis,
        noise=noise,
        interaction_basis="szz",
        decompose_errors=False,
    )
    lowered = _surface_native_topology(
        patch_key,
        1,
        basis,
        None,
        "abstract",
        True,
        interaction_basis="szz",
        szz_physical_prefixes=True,
    )
    plain = _surface_native_topology(
        patch_key,
        1,
        basis,
        None,
        "abstract",
        True,
        interaction_basis="szz",
        szz_physical_prefixes=False,
    )

    expected = _dem_string_from_cached_surface_topology(
        lowered,
        noise,
        decompose_errors=False,
    )
    assert actual == expected
    assert actual != _dem_string_from_cached_surface_topology(
        plain,
        noise,
        decompose_errors=False,
    )


def test_szz_virtual_prefix_ticks_do_not_contribute_idle_dem() -> None:
    patch = SurfacePatch.create(distance=3)
    noise_kwargs = {
        "p1": 0.0,
        "p2": 0.0,
        "p_meas": 0.0,
        "p_prep": 0.0,
        "p_idle_z_linear_rate": 0.01,
        "decompose_errors": False,
    }

    tagged = generate_tick_circuit_from_patch(
        patch,
        num_rounds=1,
        basis="Z",
        interaction_basis="szz",
        szz_physical_prefixes=True,
    )
    retagged = generate_tick_circuit_from_patch(
        patch,
        num_rounds=1,
        basis="Z",
        interaction_basis="szz",
        szz_physical_prefixes=True,
    )
    assert _retag_virtual_prefix_duration(retagged, 1.0) > 0

    tagged.fill_idle_gates()
    retagged.fill_idle_gates()

    tagged_dem = generate_dem_from_tick_circuit(tagged, **noise_kwargs)
    retagged_dem = generate_dem_from_tick_circuit(retagged, **noise_kwargs)

    assert (
        generate_circuit_level_dem_from_builder(
            patch,
            num_rounds=1,
            basis="Z",
            noise=NoiseParameters().with_p_idle_linear(0.01, {"Z": 1.0}),
            interaction_basis="szz",
            decompose_errors=False,
        )
        == tagged_dem
    )
    assert retagged_dem != tagged_dem
