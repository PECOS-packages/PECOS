# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Tests for the public QIS operation-tracing API."""

import json
import math

import pecos
import pecos_rslib
import pytest
from pecos._qis_trace_replay import _replay_qis_trace_into_tick_circuit
from pecos.quantum import TickCircuit
from pecos.simulators import StateVec


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


def _execute_tick_circuit(circuit: TickCircuit) -> list[complex]:
    simulator = StateVec(2)
    for tick_index in range(circuit.num_ticks()):
        for gate in circuit.get_tick(tick_index).gate_batches():
            name = gate.gate_type.name
            qubits = list(gate.qubits)
            angles = list(gate.angles)
            if name in {"PZ", "X", "Z", "RZ"}:
                params = {"angle": angles[0]} if name == "RZ" else None
                for qubit in qubits:
                    simulator.backend.run_1q_gate(name, qubit, params)
            elif name == "RZZ":
                for offset in range(0, len(qubits), 2):
                    simulator.backend.run_2q_gate(
                        name,
                        (qubits[offset], qubits[offset + 1]),
                        {"angle": angles[0]},
                    )
            else:
                msg = f"unexpected replay gate {name}"
                raise AssertionError(msg)
    return [complex(value) for value in simulator.backend.vector]


def test_qis_trace_crz_preserves_full_matrix() -> None:
    for theta in (-math.pi, math.pi / 3, math.pi, math.tau, -math.tau, 3 * math.pi, 3 * math.tau, -3 * math.tau):
        columns: list[list[complex]] = []
        for basis in range(4):
            operations = [
                {"AllocateQubit": {"id": 0}},
                {"AllocateQubit": {"id": 1}},
            ]
            lowered = [
                {"gate_type": "PZ", "qubits": [0], "angles": [], "params": []},
                {"gate_type": "PZ", "qubits": [1], "angles": [], "params": []},
            ]
            if basis & 1:
                operations.append({"Quantum": {"X": 0}})
                lowered.append({"gate_type": "X", "qubits": [0], "angles": [], "params": []})
            if basis & 2:
                operations.append({"Quantum": {"X": 1}})
                lowered.append({"gate_type": "X", "qubits": [1], "angles": [], "params": []})
            operations.append({"Quantum": {"CRZ": [theta, 1, 0]}})
            lowered.append(
                {
                    "gate_type": "CRZ",
                    "qubits": [1, 0],
                    "angles": [theta],
                    "params": [],
                },
            )
            trace = _completed_trace()
            trace[0]["operations"] = operations
            trace[0]["num_operations"] = len(operations)
            trace[0]["lowered_quantum_ops"] = lowered
            circuit = pecos.qis_operation_trace_to_tick_circuit(trace)
            columns.append(_execute_tick_circuit(circuit))

        half = theta / 2
        reference = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, complex(math.cos(half), -math.sin(half)), 0],
            [0, 0, 0, complex(math.cos(half), math.sin(half))],
        ]
        for column in range(4):
            for row in range(4):
                assert abs(columns[column][row] - reference[row][column]) < 1e-12


@pytest.mark.parametrize("lowered", [False, True], ids=["raw", "runtime-lowered"])
def test_qis_trace_rxyxy2q_preserves_angles_and_pair(lowered: bool) -> None:
    operations = [
        {"AllocateQubit": {"id": 7}},
        {"AllocateQubit": {"id": 11}},
        {"AllocateQubit": {"id": 3}},
        {"Quantum": {"RXYXY2Q": [-0.73, 0.41, 3, 7]}},
    ]
    if lowered:
        trace = _completed_trace()
        trace[0]["operations"] = operations
        trace[0]["num_operations"] = len(operations)
        trace[0]["lowered_quantum_ops"] = [
            {
                "gate_type": "RXYXY2Q",
                "qubits": [2, 0],
                "angles": [-0.73, 0.41],
                "params": [],
                "metadata": {"source_label": "xyxy"},
            },
        ]
        circuit = pecos.qis_operation_trace_to_tick_circuit(trace)
        assert circuit.get_gate_meta(0, 0, "source_label") == "xyxy"
    else:
        circuit = _replay_qis_trace_into_tick_circuit(operations)

    gates = [gate for _, gate in circuit.gate_batches() if gate.gate_type.name != "PZ"]
    assert len(gates) == 1
    assert gates[0].gate_type.name == "RXYXY2Q"
    # Gate.angles exposes Angle64's unsigned radians; compare signed representatives.
    assert [math.remainder(angle, math.tau) for angle in gates[0].angles] == pytest.approx([-0.73, 0.41])
    assert gates[0].qubits == [2, 0]


@pytest.mark.parametrize("typed_angles", [False, True], ids=["float", "angle64"])
def test_tick_circuit_rxyxy2q_constructor(typed_angles: bool) -> None:
    theta, phi = -0.73, 0.41
    if typed_angles:
        theta, phi = (pecos_rslib.angle64.from_radians(angle) for angle in (theta, phi))
    circuit = TickCircuit()
    tick = circuit.tick()
    assert tick.rxyxy2q(theta, phi, [(5, 2)]) is tick
    assert circuit.gate_count() == 1
    gates = [circuit.get_tick(0).gate_batches()[0], pecos_rslib.Gate.rxyxy2q(theta, phi, [(5, 2)])]
    for gate in gates:
        assert gate.gate_type.name == "RXYXY2Q"
        assert gate.gate_type == pecos_rslib.GateType.RXYXY2Q
        assert [math.remainder(angle, math.tau) for angle in gate.angles] == pytest.approx([-0.73, 0.41])
        assert gate.qubits == [5, 2]


@pytest.mark.parametrize("lowered", [False, True], ids=["raw", "runtime-lowered"])
def test_qis_trace_unknown_gate_still_fails(lowered: bool) -> None:
    if lowered:
        trace = _completed_trace()
        trace[0]["lowered_quantum_ops"][1]["gate_type"] = "UnknownGate"
        with pytest.raises(ValueError, match=r"Unsupported.*UnknownGate"):
            pecos.qis_operation_trace_to_tick_circuit(trace)
    else:
        with pytest.raises(ValueError, match=r"Unsupported.*UnknownGate"):
            _replay_qis_trace_into_tick_circuit([{"Quantum": {"UnknownGate": 0}}])


def test_qis_trace_rxyxy2q_rejects_incomplete_pair() -> None:
    trace = _completed_trace()
    trace[0]["lowered_quantum_ops"][1] = {
        "gate_type": "RXYXY2Q",
        "qubits": [2],
        "angles": [-0.73, 0.41],
        "params": [],
    }
    with pytest.raises(ValueError, match=r"RXYXY2Q.*expected an even number of qubits"):
        pecos.qis_operation_trace_to_tick_circuit(trace)


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
        batch_adapter_factory: object | None = None,
        clock_model: str | None = None,
    ) -> list[dict]:
        del program, num_qubits, seed, runtime
        assert batch_adapter_factory is None
        assert clock_model is None
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
        ("RXYXY2Q", []),
        ("RXYXY2Q", [0.5]),
        ("RXYXY2Q", [0.5, 0.25, 0.75]),
    ],
)
def test_qis_operation_trace_conversion_rejects_invalid_angle_arity(
    gate_type: str,
    angles: list[float],
) -> None:
    trace = _completed_trace()
    trace[0]["lowered_quantum_ops"][1] = {
        "gate_type": gate_type,
        "qubits": [0, 1] if gate_type in {"CRZ", "RZZ", "RXYXY2Q"} else [0],
        "angles": angles,
        "params": [],
    }

    with pytest.raises(ValueError, match=rf"Lowered gate '{gate_type}' expected"):
        pecos.qis_operation_trace_to_tick_circuit(trace)


@pytest.mark.parametrize("clock_model", [None, "shot_relative"])
def test_capture_qis_operation_trace_configures_the_trace_builder(
    monkeypatch: pytest.MonkeyPatch,
    clock_model: str | None,
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
    monkeypatch.setattr(
        pecos,
        "selene_engine",
        lambda actual, *, batch_adapter_factory, clock_model: ("runtime", actual, batch_adapter_factory, clock_model),
    )
    monkeypatch.setattr(pecos_rslib, "coin_toss", lambda: "trace-backend")

    assert pecos.capture_qis_operation_trace(program, 3, seed=11, runtime=runtime, clock_model=clock_model) == trace
    assert calls == [
        ("classical", ("runtime", runtime, None, clock_model)),
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


@pytest.mark.parametrize("leaked", [False, True])
def test_native_measurement_provenance_uses_source_slot(leaked) -> None:
    trace = _completed_trace()
    trace[0]["operations"][-1] = {
        "RuntimeRequest": {
            "request": {"Native": {"Measure": {"qubit": 0, "result_id": 7, "leaked": leaked}}},
            "result_id": 2**64 - 2,
        },
    }
    replayed = pecos.qis_operation_trace_to_tick_circuit(trace)
    assert replayed.get_meta("qis_source_measurement_ids") == "[7]"


def test_native_request_cannot_silently_lose_lowered_gates() -> None:
    trace = _completed_trace()
    trace[0]["operations"] = [{"RuntimeRequest": {"request": {"Custom": {"tag": 1, "data": []}}, "result_id": None}}]
    trace[0]["num_operations"] = 1
    trace[0]["lowered_quantum_ops"] = []
    with pytest.raises(ValueError, match="does not contain lowered_quantum_ops"):
        pecos.qis_operation_trace_to_tick_circuit(trace)


def _identity_program(*, native_first: bool, reuse: bool, named: bool, native: bool = True) -> str:
    """Use explicit sparse QIR slot 7 and a native future on another qubit."""
    qir = """
        call void @__quantum__qis__reset__body(i64 0)
        call void @__quantum__qis__x__body(i64 0)
        %qm = call i32 @__quantum__qis__m__body(i64 0, i64 7)
    """
    native_ops = (
        """
        %q = call i64 @___qalloc()
        call void @___reset(i64 %q)
        %future = call i64 @___lazy_measure(i64 %q)
    """
        if native
        else ""
    )
    body = native_ops + qir if native_first else qir + native_ops
    body += """
        %old32 = call i32 @__quantum__rt__result_get_one(i64 7)
        %old = icmp ne i32 %old32, 0
    """
    values = ["%old"]
    if reuse:
        body += """
            call void @__quantum__qis__x__body(i64 0)
            %qm2 = call i32 @__quantum__qis__m__body(i64 0, i64 7)
            %new32 = call i32 @__quantum__rt__result_get_one(i64 7)
            %new = icmp ne i32 %new32, 0
        """
        values.append("%new")
    if native:
        body += "%native = call i1 @___read_future_bool(i64 %future)\n"
        values.append("%native")
    if named:
        body += f"%array = alloca [{len(values)} x i8]\n"
        for index, value in enumerate(values):
            body += f"%v{index} = zext i1 {value} to i8\n"
            body += f"%p{index} = getelementptr i8, ptr %array, i64 {index}\n"
            body += f"store i8 %v{index}, ptr %p{index}\n"
        body += f"call void @print_bool_arr_selene(ptr @label, i64 3, ptr %array, i64 {len(values)})\n"
    return f"""
        @label = private constant [3 x i8] c"out"
        declare void @__quantum__qis__reset__body(i64)
        declare void @__quantum__qis__x__body(i64)
        declare i32 @__quantum__qis__m__body(i64, i64)
        declare i32 @__quantum__rt__result_get_one(i64)
        declare i64 @___qalloc()
        declare void @___reset(i64)
        declare i64 @___lazy_measure(i64)
        declare i1 @___read_future_bool(i64)
        declare void @print_bool_arr_selene(ptr, i64, ptr, i64)
        define i64 @qmain(i64 %shot) {{
            {body}
            ret i64 0
        }}
    """


def _identity_sim(program: str):
    return (
        pecos.sim(pecos.Qis(program))
        .classical(pecos.selene_engine("selene_simple_runtime"))
        .quantum(pecos_rslib.state_vector())
        .qubits(2)
        .workers(1)
        .seed(42)
    )


@pytest.mark.parametrize("native_first", [False, True])
@pytest.mark.parametrize("reuse", [False, True])
@pytest.mark.parametrize("slot", [0, 7])
def test_mixed_result_namespaces_preserve_values_and_audited_provenance(*, native_first, reuse, slot) -> None:
    from pecos._qis_trace_replay import named_result_traces_from_operation_trace

    program = _identity_program(native_first=native_first, reuse=reuse, named=True).replace("i64 7", f"i64 {slot}")
    expected_values = [1, 0, 0] if reuse else [1, 0]
    assert _identity_sim(program).run(3).to_dict() == {"out": [expected_values] * 3}
    trace = _identity_sim(program).capture_operation_trace()
    qir_source, native_source = (1, 0) if native_first else (0, 1)
    expected_sources = [qir_source, 2, native_source] if reuse else [qir_source, native_source]
    records = named_result_traces_from_operation_trace(trace)
    assert records == [{"name": "out", "values": [bool(v) for v in expected_values], "result_ids": expected_sources}]
    circuit = pecos.qis_operation_trace_to_tick_circuit(trace)
    assert json.loads(circuit.get_meta("qis_source_measurement_ids")) == list(range(3 if reuse else 2))
    assert trace[-1]["stage"] == "trace_complete"


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("reuse", [False, True])
def test_raw_qir_slots_keep_sparse_names_and_latest_values(*, native, reuse) -> None:
    program = _identity_program(native_first=True, reuse=reuse, named=False, native=native)
    expected = {"measurement_7": [0 if reuse else 1] * 3}
    if native:
        expected["native_measurement_0"] = [0] * 3
    assert _identity_sim(program).run(3).to_dict() == expected
