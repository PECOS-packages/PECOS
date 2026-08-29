# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Tests for the public QIS operation-tracing API."""

import pecos
import pecos_rslib
import pytest
from pecos.quantum import TickCircuit


def _completed_trace() -> list[dict]:
    return [
        {
            "format": "pecos_qis_operation_trace_v1",
            "engine_trace_id": 17,
            "shot_index": 0,
            "chunk_index": 0,
            "stage": "execute",
            "operations": [
                {"AllocateQubit": {"id": 0}},
                {"Quantum": {"H": 0}},
                {"Quantum": {"Idle": [20e-9, 0]}},
                {"Quantum": {"Measure": [0, 7]}},
            ],
            "num_operations": 4,
            "lowered_quantum_ops": [
                {"gate_type": "PZ", "qubits": [0], "angles": [], "params": []},
                {"gate_type": "H", "qubits": [0], "angles": [], "params": []},
                {"gate_type": "Idle", "qubits": [0], "angles": [], "params": [20e-9]},
                {
                    "gate_type": "MZ",
                    "qubits": [0],
                    "angles": [],
                    "params": [],
                    "measurement_result_ids": [7],
                },
            ],
            "lowered_quantum_ops_complete": True,
        },
        {
            "format": "pecos_qis_operation_trace_v1",
            "engine_trace_id": 17,
            "shot_index": 0,
            "chunk_index": 1,
            "stage": "trace_complete",
            "operations": [],
            "num_operations": 0,
            "lowered_quantum_ops": [],
            "named_result_traces": [],
        },
    ]


def _gate_names(circuit: TickCircuit) -> list[str]:
    return [
        batch.gate_type.name
        for tick_index in range(circuit.num_ticks())
        for batch in circuit.get_tick(tick_index).gate_batches()
    ]


def _idle_gates(circuit: TickCircuit) -> list[tuple[list[int], float]]:
    dag = circuit.to_dag_circuit()
    return [
        (list(gate.qubits), float(gate.params[0]))
        for node_id in dag.nodes()
        if (gate := dag.gate(node_id)) is not None and gate.gate_type.name == "Idle"
    ]


def test_tracing_apis_are_exported_at_top_level() -> None:
    assert pecos.capture_qis_operation_trace is pecos.tracing.capture_qis_operation_trace
    assert pecos.qis_operation_trace_to_tick_circuit is pecos.tracing.qis_operation_trace_to_tick_circuit
    assert pecos.trace_program_to_tick_circuit is pecos.tracing.trace_program_to_tick_circuit
    assert {
        "capture_qis_operation_trace",
        "qis_operation_trace_to_tick_circuit",
        "trace_program_to_tick_circuit",
    } <= set(pecos.__all__)


def test_qis_operation_trace_can_be_replayed_or_captured_as_a_tick_circuit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = _completed_trace()
    replayed = pecos.qis_operation_trace_to_tick_circuit(trace)

    def fake_capture(
        program: object,
        num_qubits: int,
        *,
        seed: int = 0,
        runtime: object | None = None,
    ) -> list[dict]:
        del program, num_qubits, seed, runtime
        return trace

    monkeypatch.setattr(pecos.tracing, "capture_qis_operation_trace", fake_capture)
    captured = pecos.trace_program_to_tick_circuit(object(), num_qubits=1, seed=0)

    assert isinstance(replayed, TickCircuit)
    assert _gate_names(replayed) == _gate_names(captured)
    assert "MZ" in _gate_names(replayed)
    assert _idle_gates(replayed) == [([0], 20.0)]
    assert replayed.get_meta("qis_source_measurement_ids") == "[7]"


def test_leakage_measurement_replays_as_accepted_path_mz() -> None:
    trace = _completed_trace()
    trace[0]["operations"][-1] = {"Quantum": {"MeasureLeaked": [0, 7]}}
    trace[0]["lowered_quantum_ops"][-1]["gate_type"] = "MeasureLeaked"

    replayed = pecos.qis_operation_trace_to_tick_circuit(trace)

    assert "MZ" in _gate_names(replayed)
    assert "MeasureLeaked" not in _gate_names(replayed)
    assert replayed.get_meta("qis_source_measurement_ids") == "[7]"


def test_legacy_r1xy_trace_record_replays_as_rxy1q() -> None:
    trace = _completed_trace()
    trace[0]["lowered_quantum_ops"].insert(
        1,
        {
            "gate_type": "R1XY",
            "qubits": [0],
            "angles": [0.5, 0.0],
            "params": [],
        },
    )

    replayed = pecos.qis_operation_trace_to_tick_circuit(trace)

    assert "RXY1Q" in _gate_names(replayed)


def test_qis_operation_trace_conversion_rejects_an_incomplete_trace() -> None:
    with pytest.raises(ValueError, match="terminal trace_complete"):
        pecos.qis_operation_trace_to_tick_circuit(_completed_trace()[:-1])


def test_qis_operation_trace_conversion_rejects_mismatched_measurement_provenance() -> None:
    trace = _completed_trace()
    trace[0]["lowered_quantum_ops"][-1]["measurement_result_ids"] = [8]

    with pytest.raises(ValueError, match="source and runtime-lowered measurement identities do not match"):
        pecos.qis_operation_trace_to_tick_circuit(trace)


def test_qis_operation_trace_conversion_rejects_non_object_chunks() -> None:
    with pytest.raises(TypeError, match="trace chunks must be objects"):
        pecos.qis_operation_trace_to_tick_circuit([None])  # type: ignore[list-item]


def test_qis_operation_trace_conversion_rejects_malformed_lowered_gate_fields() -> None:
    trace = _completed_trace()
    trace[0]["lowered_quantum_ops"][0]["qubits"] = "0"

    with pytest.raises(TypeError, match="field 'qubits' must be a list"):
        pecos.qis_operation_trace_to_tick_circuit(trace)


def test_qis_operation_trace_conversion_rejects_boolean_measurement_ids() -> None:
    trace = _completed_trace()
    trace[0]["lowered_quantum_ops"][-1]["measurement_result_ids"] = [True]

    with pytest.raises(TypeError, match="field 'measurement_result_ids' must contain integers"):
        pecos.qis_operation_trace_to_tick_circuit(trace)


@pytest.mark.parametrize("bad_value", [float("nan"), 10**1_000], ids=["nan", "overflow"])
def test_qis_operation_trace_conversion_rejects_non_finite_gate_values(bad_value: float) -> None:
    trace = _completed_trace()
    trace[0]["lowered_quantum_ops"][1] = {
        "gate_type": "RX",
        "qubits": [0],
        "angles": [bad_value],
        "params": [],
    }

    with pytest.raises(ValueError, match="field 'angles' must contain finite values"):
        pecos.qis_operation_trace_to_tick_circuit(trace)


def test_qis_operation_trace_conversion_does_not_coerce_gate_values() -> None:
    trace = _completed_trace()
    trace[0]["lowered_quantum_ops"][1] = {
        "gate_type": "RX",
        "qubits": [0],
        "angles": ["0.5"],
        "params": [],
    }

    with pytest.raises(TypeError, match="field 'angles' must contain numbers"):
        pecos.qis_operation_trace_to_tick_circuit(trace)


def test_qis_operation_trace_conversion_rejects_boolean_framing_counts() -> None:
    trace = _completed_trace()
    trace[0]["num_operations"] = True

    with pytest.raises(TypeError, match="invalid num_operations"):
        pecos.qis_operation_trace_to_tick_circuit(trace)


@pytest.mark.parametrize(
    ("gate_type", "angles"),
    [
        ("RX", []),
        ("RY", [0.5, 0.25]),
        ("RZ", []),
        ("RXY1Q", [0.5]),
        ("CRZ", []),
        ("RZZ", []),
    ],
)
def test_qis_operation_trace_conversion_rejects_invalid_angle_arity(
    gate_type: str,
    angles: list[float],
) -> None:
    trace = _completed_trace()
    trace[0]["lowered_quantum_ops"][1] = {
        "gate_type": gate_type,
        "qubits": [0, 1] if gate_type in {"CRZ", "RZZ"} else [0],
        "angles": angles,
        "params": [],
    }

    with pytest.raises(ValueError, match=rf"Lowered gate '{gate_type}' expected"):
        pecos.qis_operation_trace_to_tick_circuit(trace)


def test_capture_qis_operation_trace_configures_the_trace_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = _completed_trace()
    calls: list[tuple[str, object]] = []

    class FakeBuilder:
        def classical(self, engine):
            calls.append(("classical", engine))
            return self

        def quantum(self, engine):
            calls.append(("quantum", engine))
            return self

        def qubits(self, count):
            calls.append(("qubits", count))
            return self

        def seed(self, seed):
            calls.append(("seed", seed))
            return self

        def capture_operation_trace(self, shots):
            calls.append(("shots", shots))
            return iter(trace)

    program = object()
    runtime = object()
    monkeypatch.setattr(pecos, "sim", lambda actual: FakeBuilder() if actual is program else None)
    monkeypatch.setattr(pecos, "selene_engine", lambda actual: ("runtime", actual))
    monkeypatch.setattr(pecos_rslib, "coin_toss", lambda: "trace-backend")

    assert pecos.capture_qis_operation_trace(program, 3, seed=11, runtime=runtime) == trace
    assert calls == [
        ("classical", ("runtime", runtime)),
        ("quantum", "trace-backend"),
        ("qubits", 3),
        ("seed", 11),
        ("shots", 1),
    ]


def test_surface_module_legacy_names_remain_compatible() -> None:
    from pecos._traced_circuit import normalize_traced_tick_circuit
    from pecos.qec.surface.circuit_builder import normalize_traced_qis_tick_circuit
    from pecos.qec.surface.decode import (
        capture_guppy_operation_trace,
        trace_guppy_into_tick_circuit,
    )

    assert capture_guppy_operation_trace is pecos.capture_qis_operation_trace
    assert trace_guppy_into_tick_circuit is pecos.trace_program_to_tick_circuit
    assert normalize_traced_qis_tick_circuit is normalize_traced_tick_circuit
