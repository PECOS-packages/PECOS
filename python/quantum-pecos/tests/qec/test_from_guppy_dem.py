# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Regression tests for the Guppy-to-DEM convenience path."""

import json
from typing import ClassVar

import pytest
from guppylang import guppy
from guppylang.std.builtins import barrier, owned, result
from guppylang.std.quantum import h, measure, qubit, x
from pecos._qis_trace_replay import (
    _reject_partially_lowered_trace,
    _replay_lowered_qis_trace_into_tick_circuit,
    _replay_qis_trace_into_tick_circuit,
    named_result_traces_from_operation_trace,
)
from pecos._traced_circuit import (
    measurement_ids_in_execution_order,
    normalize_traced_tick_circuit,
)
from pecos.guppy import get_num_qubits, make_surface_code
from pecos.qec import DetectorErrorModel
from pecos.qec.surface import RUNTIME_IDLE_TIME_UNITS_PER_SECOND, NoiseModel, SurfacePatch
from pecos.qec.surface.circuit_builder import (
    generate_tick_circuit_from_patch,
)
from pecos.qec.surface.decode import (
    _build_surface_tick_circuit_for_native_model,
    _copy_surface_tick_circuit_metadata,
    _measurement_index_remap_for_orders,
    _remap_surface_record_metadata_json,
    _surface_runtime_measurement_remap_from_result_traces,
    _validate_result_tag_remap_against_traced_measurements,
    capture_guppy_operation_trace,
    generate_circuit_level_dem_from_builder,
    trace_guppy_into_tick_circuit_with_result_traces,
)


@guppy
def _single_measurement() -> None:
    q = qubit()
    b = measure(q)
    result("m", b)


@guppy
def _measurement_feedback() -> None:
    q0 = qubit()
    q1 = qubit()
    h(q0)
    b0 = measure(q0)
    if b0:
        x(q1)
    b1 = measure(q1)
    result("b0", b0)
    result("b1", b1)


@guppy.declare
def pecos_qis_trace_metadata_qubit_hugr(q: qubit @ owned, key: str, value: str) -> qubit: ...


@guppy
def _metadata_before_h_gate() -> None:
    q = qubit()
    q = pecos_qis_trace_metadata_qubit_hugr(q, "source_kind", "szz_data_prefix")
    q = pecos_qis_trace_metadata_qubit_hugr(q, "source_label", "probe:prefix")
    q = pecos_qis_trace_metadata_qubit_hugr(q, "host_id", "probe:host")
    q = pecos_qis_trace_metadata_qubit_hugr(q, "local_role", "basis_prefix")
    h(q)
    _ = measure(q)


@guppy
def _barrier_between_single_qubit_gates() -> None:
    q0 = qubit()
    q1 = qubit()
    h(q0)
    barrier(q0, q1)
    h(q1)
    _ = measure(q0)
    _ = measure(q1)


def test_operation_trace_capture_uses_trace_friendly_quantum_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    import pecos

    def forbidden_stabilizer():
        msg = "trace capture should not validate operations with stabilizer evolution"
        raise AssertionError(msg)

    monkeypatch.setattr(pecos, "stabilizer", forbidden_stabilizer)

    chunks = capture_guppy_operation_trace(_single_measurement, num_qubits=1, seed=0)
    result_names = [trace.get("name") for trace in named_result_traces_from_operation_trace(chunks)]

    assert "m" in result_names


@pytest.mark.xfail(
    reason=(
        "Guppy public barrier(...) is currently optimized away before PECOS "
        "QIS operation collection; hosted SZZ prefix scheduling needs a "
        "barrier-preserving or hosted-operation lowering path."
    ),
    strict=True,
)
def test_guppy_barrier_survives_into_qis_operation_trace() -> None:
    chunks = capture_guppy_operation_trace(
        _barrier_between_single_qubit_gates,
        num_qubits=2,
        seed=0,
    )
    operations = [operation for chunk in chunks for operation in chunk.get("operations", [])]

    assert any(operation == "Barrier" or "Barrier" in operation for operation in operations)


def test_szz_runtime_barrier_survives_into_qis_operation_trace() -> None:
    program = make_surface_code(
        distance=3,
        num_rounds=1,
        basis="Z",
        interaction_basis="szz",
        szz_runtime_barriers="data-prefix",
    )
    chunks = capture_guppy_operation_trace(
        program,
        num_qubits=get_num_qubits(d=3, interaction_basis="szz"),
        seed=0,
    )
    operations = [operation for chunk in chunks for operation in chunk.get("operations", [])]

    assert any(operation == "Barrier" or "Barrier" in str(operation) for operation in operations)


def test_qubit_trace_metadata_stays_ordered_before_gate() -> None:
    chunks = capture_guppy_operation_trace(_metadata_before_h_gate, num_qubits=1, seed=0)
    lowered_ops = [op for chunk in chunks for op in chunk.get("lowered_quantum_ops", [])]

    assert lowered_ops[1]["gate_type"] == "R1XY"
    assert lowered_ops[1]["metadata"] == {
        "host_id": "probe:host",
        "local_role": "basis_prefix",
        "source_kind": "szz_data_prefix",
        "source_label": "probe:prefix",
    }
    assert lowered_ops[-1]["gate_type"] == "MZ"
    assert lowered_ops[-1]["metadata"] == {}


def _dem_text(*, detectors_json: str = "[]", observables_json: str = "[]") -> str:
    dem = DetectorErrorModel.from_guppy(
        _single_measurement,
        num_qubits=1,
        detectors_json=detectors_json,
        observables_json=observables_json,
        p1=0.0,
        p2=0.0,
        p_meas=0.1,
        p_prep=0.0,
        seed=0,
    )
    return dem.to_string()


def _flat_mz_ids(tc) -> list[int]:
    dag = tc.to_dag_circuit()
    ids: list[int] = []
    for node_id in dag.nodes():
        gate = dag.gate(node_id)
        if gate is not None and gate.gate_type.name == "MZ":
            ids.extend(int(mid) for mid in gate.meas_ids)
    return ids


def _flat_idle_gates(tc) -> list[tuple[list[int], float]]:
    dag = tc.to_dag_circuit()
    idles: list[tuple[list[int], float]] = []
    for node_id in dag.nodes():
        gate = dag.gate(node_id)
        if gate is not None and gate.gate_type.name == "Idle":
            idles.append((list(gate.qubits), float(gate.params[0])))
    return idles


def _flat_gate_qubits(tc, gate_type_name: str) -> list[list[int]]:
    dag = tc.to_dag_circuit()
    gate_qubits: list[list[int]] = []
    for node_id in dag.nodes():
        gate = dag.gate(node_id)
        if gate is not None and gate.gate_type.name == gate_type_name:
            gate_qubits.append(list(gate.qubits))
    return gate_qubits


def test_from_guppy_meas_ids_are_normalized_to_records() -> None:
    assert _dem_text(detectors_json='[{"id":0,"meas_ids":[0]}]') == _dem_text(
        detectors_json='[{"id":0,"records":[-1]}]',
    )

    assert _dem_text(observables_json='[{"id":0,"meas_ids":[0]}]') == _dem_text(
        observables_json='[{"id":0,"records":[-1]}]',
    )


@pytest.mark.parametrize(
    "detectors_json",
    [
        "{}",
        '[{"id":0,"records":["-1"]}]',
        '[{"id":0,"records":[-1.2]}]',
        '[{"id":0,"meas_ids":["0"]}]',
    ],
)
def test_from_guppy_rejects_malformed_detector_metadata(detectors_json: str) -> None:
    with pytest.raises(ValueError, match=r"JSON list|integer|record offset|meas_id"):
        _dem_text(detectors_json=detectors_json)


def test_from_guppy_rejects_json_tracked_pauli_observables() -> None:
    with pytest.raises(ValueError, match="tracked_pauli"):
        _dem_text(observables_json='[{"kind":"tracked_pauli","label":"x","pauli":"X0"}]')


def test_from_guppy_rejects_dynamic_control_before_seed_can_select_a_branch() -> None:
    for s in (0, 2, 5):
        with pytest.raises(ValueError, match="branching or looping control flow"):
            DetectorErrorModel.from_guppy(
                _measurement_feedback,
                num_qubits=2,
                detectors_json='[{"id":0,"records":[-2,-1]}]',
                p1=0.0,
                p2=0.0,
                p_meas=0.1,
                p_prep=0.0,
                seed=s,
            )


def test_lowered_replay_uses_measure_result_ids_directly() -> None:
    chunks = [
        {
            "operations": [
                {"AllocateResult": {"id": 42}},
                {"AllocateResult": {"id": 99}},
                {"Quantum": {"Measure": [0, 99]}},
                {"Quantum": {"Measure": [1, 42]}},
            ],
            "lowered_quantum_ops": [
                {"gate_type": "MZ", "qubits": [0], "angles": [], "measurement_result_ids": [42]},
                {"gate_type": "MZ", "qubits": [1], "angles": [], "measurement_result_ids": [99]},
            ],
        },
    ]

    tc = _replay_lowered_qis_trace_into_tick_circuit(chunks)

    assert _flat_mz_ids(tc) == [42, 99]


def test_lowered_replay_fails_on_measurement_count_mismatch() -> None:
    chunks = [
        {
            "operations": [{"Quantum": {"Measure": [0, 7]}}],
            "lowered_quantum_ops": [
                {"gate_type": "MZ", "qubits": [0, 1], "angles": [], "measurement_result_ids": [7]},
            ],
        },
    ]

    with pytest.raises(ValueError, match="carries 1 measurement_result_ids for 2"):
        _replay_lowered_qis_trace_into_tick_circuit(chunks)


def test_lowered_replay_fails_on_missing_measurement_result_ids() -> None:
    chunks = [
        {
            "operations": [{"Quantum": {"Measure": [0, 7]}}],
            "lowered_quantum_ops": [{"gate_type": "MZ", "qubits": [0], "angles": []}],
        },
    ]

    with pytest.raises(ValueError, match="missing measurement_result_ids"):
        _replay_lowered_qis_trace_into_tick_circuit(chunks)


def test_lowered_replay_preserves_runtime_idles() -> None:
    chunks = [
        {
            "operations": [{"Quantum": {"H": 0}}],
            "lowered_quantum_ops": [
                {"gate_type": "Idle", "qubits": [0], "angles": [], "params": [20e-9]},
                {"gate_type": "H", "qubits": [0], "angles": [], "params": []},
            ],
        },
    ]

    tc = _replay_lowered_qis_trace_into_tick_circuit(chunks)

    assert _flat_idle_gates(tc) == [([0], 20.0)]


def test_lowered_replay_converts_runtime_idle_seconds_to_nanosecond_time_units() -> None:
    chunks = [
        {
            "operations": [{"Quantum": {"H": 0}}],
            "lowered_quantum_ops": [
                {"gate_type": "Idle", "qubits": [0], "angles": [], "params": [1.3857e-5]},
                {"gate_type": "H", "qubits": [0], "angles": [], "params": []},
            ],
        },
    ]

    tc = _replay_lowered_qis_trace_into_tick_circuit(chunks)

    assert _flat_idle_gates(tc) == [([0], 13857.0)]


def test_noise_model_converts_runtime_idle_rates_from_seconds_to_dem_time_units() -> None:
    noise = NoiseModel(
        p1=0.001,
        p2=0.002,
        p_meas=0.003,
        p_prep=0.004,
        p_idle=9.0,
        t1=1.5,
        t2=2.5,
        p_idle_z_linear_rate=3.0,
        p_idle_x_quadratic_rate=4.0,
        p_idle_z_quadratic_sine_rate=5.0,
    )

    converted = noise.for_runtime_idle_time_units()

    assert converted.p1 == noise.p1
    assert converted.p2 == noise.p2
    assert converted.p_meas == noise.p_meas
    assert converted.p_prep == noise.p_prep
    assert converted.p_idle == pytest.approx(9.0 / RUNTIME_IDLE_TIME_UNITS_PER_SECOND)
    assert converted.t1 == pytest.approx(1.5 * RUNTIME_IDLE_TIME_UNITS_PER_SECOND)
    assert converted.t2 == pytest.approx(2.5 * RUNTIME_IDLE_TIME_UNITS_PER_SECOND)
    assert converted.p_idle_z_linear_rate == pytest.approx(3.0 / RUNTIME_IDLE_TIME_UNITS_PER_SECOND)
    assert converted.p_idle_x_quadratic_rate == pytest.approx(4.0 / (RUNTIME_IDLE_TIME_UNITS_PER_SECOND**2))
    assert converted.p_idle_z_quadratic_sine_rate == pytest.approx(5.0 / RUNTIME_IDLE_TIME_UNITS_PER_SECOND)


def test_noise_model_rejects_invalid_runtime_idle_time_unit_scale() -> None:
    with pytest.raises(ValueError, match="time_units_per_second"):
        NoiseModel(p_idle_z_linear_rate=1.0).for_runtime_idle_time_units(time_units_per_second=0.0)


def test_lowered_replay_preserves_gate_metadata() -> None:
    chunks = [
        {
            "operations": [{"Quantum": {"H": 0}}],
            "lowered_quantum_ops": [
                {
                    "gate_type": "H",
                    "qubits": [0],
                    "angles": [],
                    "params": [],
                    "metadata": {
                        "source_label": "szz_physical_prefix:H:X0:q0",
                        "source_kind": "szz_prefix",
                    },
                },
            ],
        },
    ]

    tc = _replay_lowered_qis_trace_into_tick_circuit(chunks)

    assert tc.get_gate_meta(0, 0, "source_label") == "szz_physical_prefix:H:X0:q0"
    assert tc.get_gate_meta(0, 0, "source_kind") == "szz_prefix"


def test_lowered_replay_preserves_measurement_crosstalk_payloads() -> None:
    chunks = [
        {
            "operations": [{"Quantum": {"Measure": [0, 0]}}],
            "lowered_quantum_ops": [
                {"gate_type": "PZ", "qubits": [0], "angles": [], "params": []},
                {
                    "gate_type": "MeasCrosstalkLocalPayload",
                    "qubits": [1, 2],
                    "angles": [],
                    "params": [],
                },
                {
                    "gate_type": "MeasCrosstalkGlobalPayload",
                    "qubits": [3, 4],
                    "angles": [],
                    "params": [],
                },
                {
                    "gate_type": "MZ",
                    "qubits": [0],
                    "angles": [],
                    "params": [],
                    "measurement_result_ids": [0],
                },
            ],
        },
    ]

    tc = _replay_lowered_qis_trace_into_tick_circuit(chunks)

    assert _flat_gate_qubits(tc, "MeasCrosstalkLocalPayload") == [[1, 2]]
    assert _flat_gate_qubits(tc, "MeasCrosstalkGlobalPayload") == [[3, 4]]


def test_lowered_replay_can_add_global_crosstalk_payloads_from_measurements() -> None:
    chunks = [
        {
            "operations": [{"Quantum": {"Measure": [0, 7]}}],
            "lowered_quantum_ops": [
                {
                    "gate_type": "MZ",
                    "qubits": [11, 12],
                    "angles": [],
                    "params": [],
                    "measurement_result_ids": [7, 8],
                },
            ],
        },
    ]

    without_payloads = _replay_lowered_qis_trace_into_tick_circuit(chunks)
    with_payloads = _replay_lowered_qis_trace_into_tick_circuit(
        chunks,
        measurement_crosstalk_topology="global_from_measurements",
    )

    assert _flat_gate_qubits(without_payloads, "MeasCrosstalkGlobalPayload") == []
    assert _flat_gate_qubits(with_payloads, "MeasCrosstalkGlobalPayload") == [
        [11, 12],
    ]
    assert _flat_mz_ids(with_payloads) == [7, 8]


def test_raw_replay_can_add_global_crosstalk_payloads_from_measurements() -> None:
    operations = [
        {"AllocateQubit": {"id": 0}},
        {"AllocateResult": {"id": 9}},
        {"Quantum": {"Measure": [0, 9]}},
    ]

    tc = _replay_qis_trace_into_tick_circuit(
        operations,
        measurement_crosstalk_topology="global_from_measurements",
    )

    assert _flat_gate_qubits(tc, "MeasCrosstalkGlobalPayload") == [[0]]
    assert _flat_mz_ids(tc) == [9]


def test_replay_rejects_unknown_measurement_crosstalk_topology() -> None:
    with pytest.raises(ValueError, match="measurement_crosstalk_topology"):
        _replay_lowered_qis_trace_into_tick_circuit(
            [],
            measurement_crosstalk_topology="local_from_vibes",
        )


def test_lowered_runtime_idles_can_drive_memory_noise_dem() -> None:
    from pecos.qec import DetectorErrorModel

    chunks = [
        {
            "operations": [{"Quantum": {"Measure": [0, 0]}}],
            "lowered_quantum_ops": [
                {"gate_type": "PZ", "qubits": [0], "angles": [], "params": []},
                {"gate_type": "H", "qubits": [0], "angles": [], "params": []},
                {"gate_type": "Idle", "qubits": [0], "angles": [], "params": [20e-9]},
                {"gate_type": "H", "qubits": [0], "angles": [], "params": []},
                {"gate_type": "MZ", "qubits": [0], "angles": [], "params": [], "measurement_result_ids": [0]},
            ],
        },
    ]
    tc = _replay_lowered_qis_trace_into_tick_circuit(chunks)
    tc.set_meta("detectors", '[{"id": 0, "records": [-1]}]')
    tc.set_meta("observables", "[]")
    tc.set_meta("num_measurements", "1")

    dem = DetectorErrorModel.from_circuit(
        tc,
        p1=0.0,
        p2=0.0,
        p_meas=0.0,
        p_prep=0.0,
        p_idle_linear_rate=1.0e-3,
    )

    assert dem.num_contributions > 0


def test_lowered_runtime_idles_accept_axis_memory_noise_dem() -> None:
    from pecos.qec import DetectorErrorModel

    chunks = [
        {
            "operations": [{"Quantum": {"Measure": [0, 0]}}],
            "lowered_quantum_ops": [
                {"gate_type": "PZ", "qubits": [0], "angles": [], "params": []},
                {"gate_type": "Idle", "qubits": [0], "angles": [], "params": [10e-9]},
                {"gate_type": "MZ", "qubits": [0], "angles": [], "params": [], "measurement_result_ids": [0]},
            ],
        },
    ]
    tc = _replay_lowered_qis_trace_into_tick_circuit(chunks)
    tc.set_meta("detectors", '[{"id": 0, "records": [-1]}]')
    tc.set_meta("observables", "[]")
    tc.set_meta("num_measurements", "1")

    dem = DetectorErrorModel.from_circuit(
        tc,
        p1=0.0,
        p2=0.0,
        p_meas=0.0,
        p_prep=0.0,
        p_idle_x_linear_rate=1.0e-3,
        p_idle_y_quadratic_rate=1.0e-4,
    )

    assert dem.num_contributions > 0


def test_from_circuit_accepts_biased_p2_weights() -> None:
    from pecos.qec import DetectorErrorModel

    chunks = [
        {
            "operations": [{"Quantum": {"Measure": [1, 0]}}],
            "lowered_quantum_ops": [
                {"gate_type": "PZ", "qubits": [0], "angles": [], "params": []},
                {"gate_type": "PZ", "qubits": [1], "angles": [], "params": []},
                {"gate_type": "CX", "qubits": [0, 1], "angles": [], "params": []},
                {"gate_type": "MZ", "qubits": [1], "angles": [], "params": [], "measurement_result_ids": [0]},
            ],
        },
    ]
    tc = _replay_lowered_qis_trace_into_tick_circuit(chunks)
    tc.set_meta("detectors", '[{"id": 0, "records": [-1]}]')
    tc.set_meta("observables", "[]")
    tc.set_meta("num_measurements", "1")
    pauli_labels = (
        "IX",
        "IY",
        "IZ",
        "XI",
        "XX",
        "XY",
        "XZ",
        "YI",
        "YX",
        "YY",
        "YZ",
        "ZI",
        "ZX",
        "ZY",
        "ZZ",
    )
    weights = dict.fromkeys(pauli_labels, 0.0)
    weights["IX"] = 1.0

    dem = DetectorErrorModel.from_circuit(
        tc,
        p1=0.0,
        p2=0.01,
        p2_weights=weights,
        p_meas=0.0,
        p_prep=0.0,
    )

    assert dem.num_contributions > 0


def test_from_circuit_accepts_biased_p1_weights() -> None:
    from pecos.qec import DetectorErrorModel

    chunks = [
        {
            "operations": [{"Quantum": {"Measure": [0, 0]}}],
            "lowered_quantum_ops": [
                {"gate_type": "PZ", "qubits": [0], "angles": [], "params": []},
                {"gate_type": "H", "qubits": [0], "angles": [], "params": []},
                {"gate_type": "MZ", "qubits": [0], "angles": [], "params": [], "measurement_result_ids": [0]},
            ],
        },
    ]
    tc = _replay_lowered_qis_trace_into_tick_circuit(chunks)
    tc.set_meta("detectors", '[{"id": 0, "records": [-1]}]')
    tc.set_meta("observables", "[]")
    tc.set_meta("num_measurements", "1")

    dem = DetectorErrorModel.from_circuit(
        tc,
        p1=0.01,
        p1_weights={"X": 1.0, "Y": 0.0, "Z": 0.0},
        p2=0.0,
        p_meas=0.0,
        p_prep=0.0,
    )

    assert dem.num_contributions > 0


def test_reject_partially_lowered_trace_passes_on_uniformly_lowered() -> None:
    """A trace where every quantum-carrying chunk is also lowered is accepted
    (this is the real Selene shape; the byte-identical regressions exercise it
    end-to-end). A chunk with only non-quantum ops and no lowered form is fine
    -- there are no gates to drop."""
    chunks = [
        {
            "operations": [{"Quantum": {"Measure": [0, 7]}}],
            "lowered_quantum_ops": [{"gate_type": "MZ", "qubits": [0], "angles": [], "measurement_result_ids": [7]}],
            "lowered_quantum_ops_complete": True,
        },
        {  # allocation/output bookkeeping only; legitimately has no lowered ops
            "operations": [{"AllocateResult": {"id": 7}}, {"RecordOutput": {"id": 7}}],
            "lowered_quantum_ops": [],
        },
    ]
    _reject_partially_lowered_trace(chunks)  # must not raise


def test_reject_partially_lowered_trace_fails_on_mixed_format() -> None:
    """A chunk carrying raw quantum gates but no lowered form, alongside a
    lowered chunk, is rejected fail-loud: the lowered replay would silently
    drop that chunk's (non-measurement) gates, and the meas-count guard would
    not catch it."""
    chunks = [
        {
            "operations": [{"Quantum": {"H": 0}}],
            "lowered_quantum_ops": [{"gate_type": "H", "qubits": [0], "angles": []}],
            "lowered_quantum_ops_complete": True,
        },
        {  # raw quantum gate present, but not lowered -> would be dropped
            "operations": [{"Quantum": {"CX": [0, 1]}}],
            "lowered_quantum_ops": [],
            "lowered_quantum_ops_complete": False,
        },
    ]
    with pytest.raises(ValueError, match=r"does not attest|mixed/partially-lowered|incomplete gate stream"):
        _reject_partially_lowered_trace(chunks)


def test_reject_partially_lowered_trace_fails_on_unlowered_allocation() -> None:
    """``AllocateQubit`` lowers to a prep (PZ), so an unlowered chunk that
    carries only an allocation alongside a lowered chunk would silently drop
    that prep -- it must fail loud too, not just chunks with raw gate ops."""
    chunks = [
        {
            "operations": [{"Quantum": {"H": 0}}],
            "lowered_quantum_ops": [{"gate_type": "H", "qubits": [0], "angles": []}],
            "lowered_quantum_ops_complete": True,
        },
        {  # allocation present (lowers to PZ) but not lowered -> would be dropped
            "operations": [{"AllocateQubit": {"id": 1}}],
            "lowered_quantum_ops": [],
            "lowered_quantum_ops_complete": False,
        },
    ]
    with pytest.raises(ValueError, match=r"does not attest|mixed/partially-lowered|incomplete gate stream"):
        _reject_partially_lowered_trace(chunks)


def test_reject_partially_lowered_trace_fails_within_one_chunk() -> None:
    chunks = [
        {
            "operations": [
                {"AllocateQubit": {"id": 0}},
                {"Quantum": {"H": 0}},
                {"Quantum": {"Measure": [0, 7]}},
            ],
            "lowered_quantum_ops": [
                {"gate_type": "MZ", "qubits": [0], "angles": [], "measurement_result_ids": [7]},
            ],
        },
    ]

    with pytest.raises(ValueError, match="does not attest a complete lowered gate stream"):
        _reject_partially_lowered_trace(chunks)


def test_from_guppy_rejects_entirely_raw_runtime_trace(monkeypatch: pytest.MonkeyPatch) -> None:
    chunks = [
        {
            "operations": [
                {"AllocateQubit": {"id": 0}},
                {"Quantum": {"Measure": [0, 0]}},
            ],
        },
    ]
    monkeypatch.setattr("pecos.tracing.capture_qis_operation_trace", lambda *_args, **_kwargs: chunks)

    with pytest.raises(ValueError, match="does not contain lowered_quantum_ops"):
        DetectorErrorModel.from_guppy(
            _single_measurement,
            num_qubits=1,
            detectors_json='[{"id":0,"records":[-1]}]',
            p1=0.0,
            p2=0.0,
            p_meas=0.1,
            p_prep=0.0,
        )


def test_non_lowered_replay_preserves_non_sequential_result_ids() -> None:
    operations = [
        {"AllocateQubit": {"id": 10}},
        {"AllocateQubit": {"id": 20}},
        {"Quantum": {"Measure": [10, 77]}},
        {"Quantum": {"Measure": [20, 3]}},
    ]

    tc = _replay_qis_trace_into_tick_circuit(operations)

    assert _flat_mz_ids(tc) == [77, 3]


def test_non_lowered_replay_preserves_idle_ops() -> None:
    operations = [
        {"AllocateQubit": {"id": 10}},
        {"Quantum": {"Idle": [20e-9, 10]}},
        {"Quantum": {"H": 10}},
    ]

    tc = _replay_qis_trace_into_tick_circuit(operations)

    assert _flat_idle_gates(tc) == [([0], 20.0)]


def test_from_guppy_surface_code_is_byte_identical_to_reference() -> None:
    """Regression: from_guppy(make_surface_code(...)) must work and match the
    traced_qis reference DEM. A reverted dynamic-control guard had broken this
    exact path (it false-positived on surface's post-measurement gates)."""
    p = {"p1": 0.005, "p2": 0.005, "p_meas": 0.005, "p_prep": 0.005}
    for basis in ("Z", "X"):
        patch = SurfacePatch.create(distance=3)
        ref = _build_surface_tick_circuit_for_native_model(
            patch,
            3,
            basis,
            circuit_source="traced_qis",
        )
        normalize_traced_tick_circuit(ref, context="from_guppy surface reference")
        ref_dem = DetectorErrorModel.from_circuit(ref, **p).to_string()
        got = DetectorErrorModel.from_guppy(
            make_surface_code(distance=3, num_rounds=3, basis=basis),
            num_qubits=get_num_qubits(3),
            detectors_json=ref.get_meta("detectors"),
            observables_json=ref.get_meta("observables"),
            num_measurements=int(ref.get_meta("num_measurements")),
            **p,
        ).to_string()
        assert got == ref_dem, f"surface from_guppy not byte-identical ({basis})"


@pytest.mark.parametrize("distance", [3, 5])
def test_from_guppy_szz_surface_code_is_byte_identical_to_reference(distance: int) -> None:
    """SZZ-basis surface Guppy generation must match the traced-QIS reference DEM."""
    p = {"p1": 0.0, "p2": 0.005, "p_meas": 0.005, "p_prep": 0.005}
    for basis in ("Z", "X"):
        patch = SurfacePatch.create(distance=distance)
        ref = _build_surface_tick_circuit_for_native_model(
            patch,
            3,
            basis,
            circuit_source="traced_qis",
            interaction_basis="szz",
        )
        normalize_traced_tick_circuit(ref, context="from_guppy SZZ surface reference")
        ref_dem = DetectorErrorModel.from_circuit(ref, **p).to_string()
        got = DetectorErrorModel.from_guppy(
            make_surface_code(
                distance=distance,
                num_rounds=3,
                basis=basis,
                interaction_basis="szz",
            ),
            num_qubits=get_num_qubits(distance, interaction_basis="szz"),
            detectors_json=ref.get_meta("detectors"),
            observables_json=ref.get_meta("observables"),
            num_measurements=int(ref.get_meta("num_measurements")),
            **p,
        ).to_string()
        assert got == ref_dem, f"SZZ surface from_guppy not byte-identical (d={distance}, {basis})"


def test_from_guppy_out_of_range_record_fails_loud() -> None:
    with pytest.raises(ValueError, match=r"out of range|record offset"):
        _dem_text(detectors_json='[{"id":0,"records":[-2]}]')  # only 1 measurement


def test_from_guppy_out_of_range_meas_id_fails_loud() -> None:
    with pytest.raises(ValueError, match=r"meas_id|not present"):
        _dem_text(detectors_json='[{"id":0,"meas_ids":[999]}]')


def test_from_guppy_accepts_dem_label_id_forms() -> None:
    """The "D0"/"L0" id convenience form is now normalized in the Rust
    builder (single source of truth), equivalent to the bare integer."""
    assert _dem_text(detectors_json='[{"id":"D0","records":[-1]}]') == _dem_text(
        detectors_json='[{"id":0,"records":[-1]}]',
    )
    assert _dem_text(observables_json='[{"id":"L0","records":[-1]}]') == _dem_text(
        observables_json='[{"id":0,"records":[-1]}]',
    )


def test_from_guppy_rejects_bad_string_id() -> None:
    with pytest.raises(ValueError, match=r"not a valid identifier"):
        _dem_text(detectors_json='[{"id":"X0","records":[-1]}]')


def test_from_guppy_rejects_detector_tracked_pauli() -> None:
    with pytest.raises(ValueError, match="tracked_pauli"):
        _dem_text(detectors_json='[{"kind":"tracked_pauli","label":"x","pauli":"X0"}]')


def test_from_guppy_rejects_entry_without_records_or_meas_ids() -> None:
    with pytest.raises(ValueError, match=r"records|meas_ids|neither"):
        _dem_text(detectors_json='[{"id":0}]')


def test_from_guppy_redundant_records_and_meas_ids_are_accepted() -> None:
    """Co-present records + meas_ids that name the SAME measurement are
    tolerated (the surface logical_circuit path emits both redundantly) and
    produce the same DEM as either form alone. (Non-redundant co-presence is
    rejected fail-loud; that precise semantics is pinned by the deterministic
    Rust unit test ``test_try_build_mixed_records_meas_ids_must_be_redundant``,
    since stamped MeasId values are not predictable from Python here.)"""
    both = _dem_text(detectors_json='[{"id":0,"records":[-1],"meas_ids":[0]}]')
    assert both == _dem_text(detectors_json='[{"id":0,"records":[-1]}]')


# ---------------------------------------------------------------------------
# Constrained-ancilla surface support
# ---------------------------------------------------------------------------


def _constrained_surface_via_guppy(*, d, basis, rounds, budget, noise, check_plan: str | None = None):
    """Build the constrained-surface DEM through `from_guppy`."""
    patch = SurfacePatch.create(distance=d)
    ref = _build_surface_tick_circuit_for_native_model(
        patch,
        num_rounds=rounds,
        basis=basis,
        ancilla_budget=budget,
        circuit_source="traced_qis",
        check_plan=check_plan,
    )
    normalize_traced_tick_circuit(ref, context="from_guppy constrained surface reference")
    ref_dem = DetectorErrorModel.from_circuit(ref, **noise).to_string()

    got = DetectorErrorModel.from_guppy(
        make_surface_code(
            distance=d,
            num_rounds=rounds,
            basis=basis,
            ancilla_budget=budget,
            check_plan=check_plan,
        ),
        num_qubits=get_num_qubits(d, ancilla_budget=budget, check_plan=check_plan),
        detectors_json=ref.get_meta("detectors"),
        observables_json=ref.get_meta("observables"),
        num_measurements=int(ref.get_meta("num_measurements")),
        **noise,
    ).to_string()
    return ref_dem, got, ref


@pytest.mark.parametrize(
    ("d", "basis", "rounds", "budget"),
    [
        (3, "Z", 2, 1),  # small-and-fast, minimum budget (one stabilizer/batch)
        (3, "X", 2, 2),  # asymmetric basis, X/Z paired per batch
        (5, "Z", 3, 5),  # medium constrained case without high-distance DEM cost
    ],
)
def test_from_guppy_constrained_surface_dem_byte_identical(
    d: int,
    basis: str,
    rounds: int,
    budget: int,
) -> None:
    """`from_guppy(make_surface_code(..., ancilla_budget=b))` must produce a
    DEM byte-identical to the reference DEM built through the
    `_build_surface_tick_circuit_for_native_model(circuit_source="traced_qis",
    ancilla_budget=b)` path. Parametrized so a regression isolates to the
    specific (distance, budget, basis) case rather than failing the whole set."""
    noise = {"p1": 0.005, "p2": 0.005, "p_meas": 0.005, "p_prep": 0.005}
    ref_dem, got, _ = _constrained_surface_via_guppy(
        d=d,
        basis=basis,
        rounds=rounds,
        budget=budget,
        noise=noise,
    )
    assert (
        got == ref_dem
    ), f"constrained surface from_guppy not byte-identical for d={d}, budget={budget}, basis={basis}, rounds={rounds}"


@pytest.mark.parametrize(
    "check_plan",
    [
        "szz_balanced_data_round_order_1032_v1",
        "szz_balanced_data_round_order_3102_v1",
    ],
)
def test_from_guppy_round_order_szz_surface_dem_byte_identical(check_plan: str) -> None:
    """Round-order SZZ check plans must stay byte-identical through from_guppy."""
    noise = {"p1": 0.0, "p2": 0.005, "p_meas": 0.005, "p_prep": 0.005}
    ref_dem, got, _ = _constrained_surface_via_guppy(
        d=3,
        basis="X",
        rounds=2,
        budget=2,
        noise=noise,
        check_plan=check_plan,
    )

    assert got == ref_dem


@pytest.mark.parametrize(
    "check_plan",
    [
        None,
        "szz_balanced_data_round_order_1032_v1",
        "szz_balanced_data_round_order_3102_v1",
    ],
)
def test_constrained_surface_traced_metadata_matches_abstract(check_plan: str | None) -> None:
    """Traced surface metadata preserves structure but binds via MeasIds.

    Runtime traces may reorder measurements, so detector/observable metadata
    cannot be copied as positional ``records``. It should preserve the same
    detector/observable IDs and descriptors while replacing abstract records
    with runtime-stable ``meas_ids``.
    """
    patch = SurfacePatch.create(distance=3)
    abstract_tc = _build_surface_tick_circuit_for_native_model(
        patch,
        num_rounds=2,
        basis="Z",
        ancilla_budget=2,
        circuit_source="abstract",
        check_plan=check_plan,
    )
    traced_tc = _build_surface_tick_circuit_for_native_model(
        patch,
        num_rounds=2,
        basis="Z",
        ancilla_budget=2,
        circuit_source="traced_qis",
        check_plan=check_plan,
    )
    for key in (
        "basis",
        "num_measurements",
        "num_detectors",
        "ancilla_budget",
    ):
        a = abstract_tc.get_meta(key)
        b = traced_tc.get_meta(key)
        assert a == b, f"metadata mismatch on key {key!r}: abstract={a!r}, traced={b!r}"

    for key in ("detectors", "observables"):
        abstract_entries = json.loads(abstract_tc.get_meta(key) or "[]")
        traced_entries = json.loads(traced_tc.get_meta(key) or "[]")
        assert len(abstract_entries) == len(traced_entries)
        for abstract_entry, traced_entry in zip(abstract_entries, traced_entries, strict=True):
            assert "records" in abstract_entry
            assert "records" not in traced_entry
            assert "meas_ids" in traced_entry
            assert {k: v for k, v in abstract_entry.items() if k != "records"} == {
                k: v for k, v in traced_entry.items() if k != "meas_ids"
            }
    # ancilla_budget specifically must be the requested budget (stored as a string by set_meta).
    assert traced_tc.get_meta("ancilla_budget") == "2"


@pytest.mark.parametrize("basis", ["X", "Z"])
def test_native_abstract_surface_dem_uses_record_metadata_only_for_r0(basis: str) -> None:
    """Native abstract DEM construction must not mix typed Pauli annotations
    with legacy record metadata.

    Public abstract circuits keep typed annotations by default, but the native
    surface DEM helper consumes JSON record metadata. Mixing both sources makes
    r=0 prep/readout DEMs carry a detectorless logical source that is absent
    from the traced-QIS metadata path.
    """
    patch = SurfacePatch.create(distance=3)
    public_tc = generate_tick_circuit_from_patch(patch, num_rounds=0, basis=basis)
    native_tc = _build_surface_tick_circuit_for_native_model(
        patch,
        num_rounds=0,
        basis=basis,
        circuit_source="abstract",
    )

    assert public_tc.annotations()
    assert native_tc.annotations() == []
    assert json.loads(native_tc.get_meta("detectors") or "[]")
    assert json.loads(native_tc.get_meta("observables") or "[]")

    noise = NoiseModel(p1=0.0, p2=0.001, p_meas=0.0, p_prep=0.0)
    for decompose_errors in (False, True):
        dem_text = generate_circuit_level_dem_from_builder(
            patch,
            num_rounds=0,
            noise=noise,
            basis=basis,
            circuit_source="abstract",
            decompose_errors=decompose_errors,
        )
        detectorless_logical_errors = [
            line for line in dem_text.splitlines() if line.startswith("error") and "L" in line and "D" not in line
        ]
        assert detectorless_logical_errors == []


@pytest.mark.parametrize(
    ("distance", "ancilla_budget"),
    [
        (3, None),
        (3, 2),
        (9, None),
        (9, 17),
    ],
)
@pytest.mark.parametrize("basis", ["X", "Z"])
@pytest.mark.parametrize("rounds", [0, 1, 3])
def test_surface_memory_round_count_contract(
    distance: int,
    ancilla_budget: int | None,
    basis: str,
    rounds: int,
) -> None:
    """Surface memory circuits count only full X/Z syndrome rounds as ``r``.

    Logical SPAM is outside ``r``: the prep phase measures only the random-sign
    stabilizer family (X checks for Z-basis memory, Z checks for X-basis
    memory), and readout measures all data qubits. Restricted and unrestricted
    ancilla schedules must preserve that experiment contract.
    """
    patch = SurfacePatch.create(distance=distance)
    geom = patch.geometry
    num_x_checks = len(geom.x_stabilizers)
    num_z_checks = len(geom.z_stabilizers)
    init_checks = num_z_checks if basis == "X" else num_x_checks
    final_check_detectors = num_x_checks if basis == "X" else num_z_checks
    expected_measurements = init_checks + rounds * (num_x_checks + num_z_checks) + geom.num_data
    expected_detectors = rounds * (num_x_checks + num_z_checks) + final_check_detectors

    tc = generate_tick_circuit_from_patch(
        patch,
        num_rounds=rounds,
        basis=basis,
        ancilla_budget=ancilla_budget,
    )

    assert int(tc.get_meta("num_measurements")) == expected_measurements
    assert len(json.loads(tc.get_meta("detectors") or "[]")) == expected_detectors
    assert json.loads(tc.get_meta("observables") or "[]")


@pytest.mark.parametrize(("d", "budget"), [(3, 1), (3, 2), (5, 3)])
def test_constrained_surface_lowered_qubit_stream_within_budget(d: int, budget: int) -> None:
    """The lowered-trace physical qubit IDs must stay within the budgeted
    pool, and ancilla slots must be empirically reused (more measurements
    than physical ancilla qubits). Pins the load-bearing assumption the
    spike validated, across several (distance, budget) combinations so the
    reuse invariant isn't only checked at one point."""
    import pecos

    program = make_surface_code(distance=d, num_rounds=2, basis="Z", ancilla_budget=budget)
    n_q = get_num_qubits(d, ancilla_budget=budget)
    chunks = list(
        pecos.sim(program)
        .classical(pecos.selene_engine())
        .quantum(pecos.stabilizer())
        .qubits(n_q)
        .seed(0)
        .capture_operation_trace(),
    )

    all_qubits: set[int] = set()
    mz_qubits: list[int] = []
    for chunk in chunks:
        for gate in chunk.get("lowered_quantum_ops") or []:
            qs = [int(q) for q in gate.get("qubits", [])]
            all_qubits.update(qs)
            if str(gate.get("gate_type")) == "MZ":
                mz_qubits.extend(qs)

    max_q = max(all_qubits) if all_qubits else -1
    # Budget enforcement: total physical qubits used must fit in d^2 + budget.
    over_budget_msg = f"max physical qubit id {max_q} exceeds budgeted pool size {n_q}"
    assert max_q < n_q, over_budget_msg
    # Reuse demonstrated: some physical qubit appears in multiple MZ ops.
    reuse = any(mz_qubits.count(q) > 1 for q in set(mz_qubits))
    assert reuse, "no physical qubit appears in more than one MZ op"


def test_constrained_from_guppy_dem_is_consumable_by_pecos_native_decoder() -> None:
    """PECOS-native decoder smoke for the constrained-ancilla DEM: the DEM
    returned by ``from_guppy(...)`` must be consumable by both the PECOS
    sampler (``dem.to_sampler()``) and the PECOS Rust-backed
    ``PyMatchingDecoder.from_dem(...)`` -- the actual downstream surfaces
    callers use, not an external ``pymatching`` install.

    Also asserts ``stim.DetectorErrorModel(dem.to_string_decomposed())``
    parses as a lightweight syntax-compatibility smoke (optional reference,
    not the correctness oracle).
    """
    from pecos_rslib.decoders import PyMatchingDecoder

    p = {"p1": 0.005, "p2": 0.005, "p_meas": 0.005, "p_prep": 0.005}
    patch = SurfacePatch.create(distance=3)
    abstract_tc = _build_surface_tick_circuit_for_native_model(
        patch,
        num_rounds=2,
        basis="Z",
        ancilla_budget=2,
        circuit_source="abstract",
    )
    dem = DetectorErrorModel.from_guppy(
        make_surface_code(distance=3, num_rounds=2, basis="Z", ancilla_budget=2),
        num_qubits=get_num_qubits(3, ancilla_budget=2),
        detectors_json=abstract_tc.get_meta("detectors"),
        observables_json=abstract_tc.get_meta("observables"),
        num_measurements=int(abstract_tc.get_meta("num_measurements")),
        **p,
    )

    # PECOS-native sampler path: the sampler must agree with the DEM it was
    # built from (substantive, not merely ``>= 0``) and actually produce
    # well-shaped samples.
    sampler = dem.to_sampler()
    assert sampler.num_detectors == dem.num_detectors
    assert sampler.num_observables == dem.num_observables
    assert dem.num_observables == 1  # one logical observable for a single patch

    batch = sampler.generate_samples(16, 0)
    assert batch.num_shots == 16
    # Each shot's syndrome covers exactly the DEM's detectors.
    assert len(batch.get_syndrome(0)) == dem.num_detectors
    # The observable mask fits within ``num_observables`` bits (no stray bits).
    assert batch.get_observable_mask(0) >> dem.num_observables == 0

    # PECOS-native Rust-backed matching decoder: DEM is consumable by
    # the actual downstream decoder surface.
    decomp = dem.to_string_decomposed()
    decoder = PyMatchingDecoder.from_dem(decomp)
    assert decoder is not None

    # Lightweight format-compatibility smoke (optional reference coverage,
    # not the correctness oracle). Stim should parse the decomposed DEM.
    import stim

    parsed = stim.DetectorErrorModel(decomp)
    assert parsed.num_detectors >= 0


def test_constrained_from_guppy_fails_loud_on_mismatched_num_measurements() -> None:
    """The constrained-ancilla surface program must flow through the same
    Rust metadata-validation fail-loud path as any other Guppy program.
    No surface-specific bypass: passing a ``num_measurements`` that disagrees
    with the count the traced program actually performs (here, one greater
    than the true count) is rejected by the generic builder, not by anything
    surface-aware in ``from_guppy``. The regex pins the builder's specific
    'declared count disagrees' diagnostic, not just the bare key name, so a
    different ``num_measurements``-mentioning error wouldn't pass spuriously."""
    p = {"p1": 0.005, "p2": 0.005, "p_meas": 0.005, "p_prep": 0.005}
    patch = SurfacePatch.create(distance=3)
    abstract_tc = _build_surface_tick_circuit_for_native_model(
        patch,
        num_rounds=2,
        basis="Z",
        ancilla_budget=2,
        circuit_source="abstract",
    )
    actual = int(abstract_tc.get_meta("num_measurements"))
    wrong = actual + 1

    with pytest.raises(
        ValueError,
        match=r"num_measurements=\d+ disagrees with the \d+ measurement",
    ):
        DetectorErrorModel.from_guppy(
            make_surface_code(distance=3, num_rounds=2, basis="Z", ancilla_budget=2),
            num_qubits=get_num_qubits(3, ancilla_budget=2),
            detectors_json=abstract_tc.get_meta("detectors"),
            observables_json=abstract_tc.get_meta("observables"),
            num_measurements=wrong,
            **p,
        )


@pytest.mark.parametrize("entry", ["get_num_qubits", "make_surface_code"])
def test_constrained_public_api_rejects_invalid_ancilla_budget(entry: str) -> None:
    """Both public entry points that accept ``ancilla_budget`` -- ``get_num_qubits``
    and ``make_surface_code`` -- validate it fail-loud at the boundary (routing
    through ``normalize_ancilla_budget``), so a bad budget never reaches codegen or
    the qubit-count math. ``bool``/``float``/``str`` raise ``TypeError``; ``< 1``
    raises ``ValueError``."""

    def call(budget: object):
        if entry == "get_num_qubits":
            return get_num_qubits(3, ancilla_budget=budget)
        return make_surface_code(distance=3, num_rounds=2, basis="Z", ancilla_budget=budget)

    for bad in (True, 1.5, "2"):
        with pytest.raises(TypeError, match=r"must be int or None"):
            call(bad)
    for bad in (0, -1):
        with pytest.raises(ValueError, match=r"must be >= 1"):
            call(bad)


def test_copy_surface_metadata_propagates_descriptors() -> None:
    """``_copy_surface_tick_circuit_metadata`` must propagate the structured
    detector/observable *descriptor* metadata, not just the raw
    detectors/observables JSON. The constrained build path doesn't populate
    descriptors lazily, so the byte-identical and metadata-match tests above
    never exercise the descriptor branch of the copy helper -- this seeds them
    explicitly on the source and pins that the copy carries them across."""
    from pecos.qec.surface import (
        get_detector_descriptors_from_tick_circuit,
        get_observable_descriptors_from_tick_circuit,
    )
    from pecos.qec.surface.decode import _copy_surface_tick_circuit_metadata
    from pecos_rslib.quantum import TickCircuit

    patch = SurfacePatch.create(distance=3)
    source = _build_surface_tick_circuit_for_native_model(
        patch,
        num_rounds=2,
        basis="Z",
        ancilla_budget=2,
        circuit_source="abstract",
    )
    # Seed the lazily-built descriptor metadata on the source.
    det_desc = get_detector_descriptors_from_tick_circuit(source, patch)
    obs_desc = get_observable_descriptors_from_tick_circuit(source, patch)
    assert source.get_meta("detector_descriptors") is not None
    assert source.get_meta("observable_descriptors") is not None

    target = TickCircuit()
    _copy_surface_tick_circuit_metadata(source, target)

    assert target.get_meta("detector_descriptors") == source.get_meta("detector_descriptors")
    assert target.get_meta("observable_descriptors") == source.get_meta("observable_descriptors")
    # Sanity: the seeded descriptors are non-trivial (real content was copied).
    assert len(det_desc) > 0
    assert len(obs_desc) > 0


def test_surface_metadata_records_bind_to_runtime_meas_ids() -> None:
    remap = _measurement_index_remap_for_orders(
        [0, 1, 0, 2],
        [1, 0, 2, 0],
    )
    assert remap == {0: 1, 1: 0, 2: 3, 3: 2}

    metadata = json.dumps(
        [
            {"id": 0, "records": [-4, -2]},
            {"id": 1, "records": [-3]},
        ],
    )
    remapped = json.loads(
        _remap_surface_record_metadata_json(
            metadata,
            measurement_index_remap=remap,
            num_measurements=4,
        ),
    )
    assert remapped == [
        {"id": 0, "meas_ids": [1, 3]},
        {"id": 1, "meas_ids": [0]},
    ]

    existing_meas_ids = json.dumps([{"id": 2, "meas_ids": [0, 3]}])
    rebound = json.loads(
        _remap_surface_record_metadata_json(
            existing_meas_ids,
            measurement_index_remap=remap,
            num_measurements=4,
        ),
    )
    assert rebound == [{"id": 2, "meas_ids": [1, 2]}]


def test_surface_metadata_records_remap_to_runtime_result_tags() -> None:
    patch = SurfacePatch.create(distance=3)
    abstract_tc = generate_tick_circuit_from_patch(
        patch,
        num_rounds=2,
        basis="Z",
        ancilla_budget=2,
    )
    program = make_surface_code(distance=3, num_rounds=2, basis="Z", ancilla_budget=2)
    _, result_traces = trace_guppy_into_tick_circuit_with_result_traces(
        program,
        get_num_qubits(3, ancilla_budget=2),
        seed=0,
    )

    remap = _surface_runtime_measurement_remap_from_result_traces(
        abstract_tc,
        result_traces,
    )

    assert len(remap) == 29  # 4 prep X stabilizers + 2 rounds * 8 stabilizers + 9 final data measurements
    assert sorted(remap) == list(range(29))
    assert sorted(remap.values()) == list(range(29))


def test_runtime_result_tags_bind_metadata_when_lowered_measurements_reorder() -> None:
    from pecos_rslib.quantum import TickCircuit

    patch = SurfacePatch.create(distance=3)
    abstract_tc = generate_tick_circuit_from_patch(patch, num_rounds=0, basis="Z")
    result_traces = [
        {"name": "sx0:init:meas:0", "values": [False], "result_ids": [0]},
        {"name": "sx1:init:meas:1", "values": [False], "result_ids": [1]},
        {"name": "sx2:init:meas:2", "values": [False], "result_ids": [2]},
        {"name": "sx3:init:meas:3", "values": [False], "result_ids": [3]},
        {
            "name": "final",
            "values": [False] * 9,
            # Semantic data-qubit order, independent of runtime MZ order.
            "result_ids": list(range(4, 13)),
        },
    ]
    remap = _surface_runtime_measurement_remap_from_result_traces(abstract_tc, result_traces)

    traced_tc = TickCircuit()
    traced_tc.tick().mz_with_ids([9, 10, 11, 12], [0, 1, 2, 3])
    traced_tc.tick().mz_with_ids(
        [5, 0, 4, 1, 8, 3, 7, 6, 2],
        [9, 4, 8, 5, 12, 7, 11, 10, 6],
    )

    assert measurement_ids_in_execution_order(traced_tc) != list(range(13))
    _validate_result_tag_remap_against_traced_measurements(
        traced_tc,
        remap,
        expected_measurements=13,
    )

    _copy_surface_tick_circuit_metadata(
        abstract_tc,
        traced_tc,
        measurement_index_remap=remap,
    )
    detectors = json.loads(traced_tc.get_meta("detectors"))
    observables = json.loads(traced_tc.get_meta("observables"))
    assert all("records" not in entry for entry in detectors + observables)
    assert all("meas_ids" in entry for entry in detectors + observables)

    logical_z_qubits = list(patch.geometry.logical_z.data_qubits)
    assert observables == [{"id": 0, "meas_ids": [4 + q for q in logical_z_qubits]}]


def test_result_tag_remap_validation_accepts_exact_traced_meas_ids() -> None:
    from pecos_rslib.quantum import TickCircuit

    tc = TickCircuit()
    tc.tick().mz_with_ids([0, 1], [10, 3])

    remap = {0: 3, 1: 10}

    assert measurement_ids_in_execution_order(tc) == [10, 3]
    _validate_result_tag_remap_against_traced_measurements(
        tc,
        remap,
        expected_measurements=2,
    )


def test_result_tag_remap_validation_rejects_duplicate_traced_meas_ids() -> None:
    # `mz_with_ids` rejects a repeated id, so a duplicate has to be supplied
    # directly to reach this validator.
    class FakeGate:
        gate_type = "MZ"
        qubits: ClassVar[list[int]] = [0, 1]
        meas_ids: ClassVar[list[int]] = [7, 7]

    class FakeTick:
        def gate_batches(self):
            return [FakeGate()]

    class FakeCircuit:
        def num_ticks(self) -> int:
            return 1

        def get_tick(self, tick_idx: int):
            assert tick_idx == 0
            return FakeTick()

    with pytest.raises(ValueError, match="duplicate measured MeasId"):
        _validate_result_tag_remap_against_traced_measurements(
            FakeCircuit(),
            {0: 7, 1: 8},
            expected_measurements=2,
        )


def test_result_tag_remap_validation_rejects_unbound_traced_meas_ids() -> None:
    from pecos_rslib.quantum import TickCircuit

    tc = TickCircuit()
    tc.tick().mz_with_ids([0, 1], [0, 2])

    with pytest.raises(ValueError, match="do not exactly match"):
        _validate_result_tag_remap_against_traced_measurements(
            tc,
            {0: 0, 1: 1},
            expected_measurements=2,
        )


def test_result_tag_remap_validation_rejects_unstamped_measurements() -> None:
    class FakeGate:
        gate_type = "MZ"
        qubits: ClassVar[list[int]] = [0]
        meas_ids: ClassVar[list[int]] = []

    class FakeTick:
        def gate_batches(self):
            return [FakeGate()]

    class FakeCircuit:
        def num_ticks(self) -> int:
            return 1

        def get_tick(self, tick_idx: int):
            assert tick_idx == 0
            return FakeTick()

    with pytest.raises(ValueError, match="carries 0 MeasId"):
        _validate_result_tag_remap_against_traced_measurements(
            FakeCircuit(),
            {0: 0},
            expected_measurements=1,
        )


def test_traced_surface_metadata_uses_runtime_result_tags() -> None:
    patch = SurfacePatch.create(distance=3)
    traced_tc = _build_surface_tick_circuit_for_native_model(
        patch,
        num_rounds=2,
        basis="Z",
        ancilla_budget=2,
        circuit_source="traced_qis",
    )

    assert traced_tc.get_meta("surface_metadata_record_binding") == "runtime_result_tags"
    assert traced_tc.get_meta("circuit_source") == "traced_qis"
    assert int(traced_tc.get_meta("num_measurements")) == 29
    assert len(json.loads(traced_tc.get_meta("detectors"))) > 0
    assert len(json.loads(traced_tc.get_meta("observables"))) == 1


def test_surface_metadata_record_remap_rejects_measurement_drift() -> None:
    with pytest.raises(ValueError, match="measured-qubit multiset"):
        _measurement_index_remap_for_orders([0, 1, 0], [0, 1, 2])


def test_surface_module_cache_collapses_unconstrained_budget_forms() -> None:
    """``get_surface_code_module`` keys its cache on the *effective* budget
    (``normalize_ancilla_budget(d*d-1, budget)``), so ``ancilla_budget=None``
    and any ``budget >= total_ancilla`` resolve to the SAME cached module --
    no redundant codegen for the two ways of saying "unconstrained". A finite
    constrained budget is a distinct entry."""
    from pecos.guppy.surface import get_surface_code_module

    d = 3
    total_ancilla = d * d - 1  # all stabilizer ancillas live simultaneously

    unconstrained_none = get_surface_code_module(d, ancilla_budget=None)
    unconstrained_exact = get_surface_code_module(d, ancilla_budget=total_ancilla)
    unconstrained_large = get_surface_code_module(d, ancilla_budget=10**6)
    # All three "unconstrained" spellings are the identical cached object.
    assert unconstrained_none is unconstrained_exact
    assert unconstrained_none is unconstrained_large
    assert unconstrained_none["ancilla_budget"] == total_ancilla

    constrained = get_surface_code_module(d, ancilla_budget=2)
    # A genuinely-constrained budget is a separate cache entry.
    assert constrained is not unconstrained_none
    assert constrained["ancilla_budget"] == 2
