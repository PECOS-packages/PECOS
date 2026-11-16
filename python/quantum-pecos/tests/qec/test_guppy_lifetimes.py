"""Hand-derived lifetimes with scrambled IDs, emission order, aliases, and reuse."""

from copy import deepcopy

import pytest
from pecos.qec.generic import CircuitOp as Op
from pecos.qec.generic import ScheduledCircuit
from pecos.qec.generic.guppy_lifetimes import bind_runtime_lifetime_windows


@pytest.fixture
def case():
    plan = ScheduledCircuit(
        2,
        (
            (Op("R", (0,)), Op("R", (1,))),
            (Op("RELABEL", (0, 1)),),
            (Op("X", (0,)),),
            (Op("M", (0,)), Op("M", (1,))),
            (Op("R", (0,)),),
            (Op("M", (0,)),),
        ),
        (),
        (),
        role_names=("a", "b"),
    )

    def prep(q):
        return [
            {"AllocateQubit": {"id": q}},
            {
                "TraceMetadata": {
                    "qubit": q,
                    "metadata": {"pecos_qec_allocation_id": str(q), "source_lowering_required": "true"},
                },
            },
            {"Quantum": {"Reset": q}},
        ]

    def measure(q, r):
        return [{"Quantum": {"Measure": [q, r]}}, {"ReleaseQubit": {"id": q}}]

    # Source preparation order differs from role order. Source measurement order
    # differs from runtime emission order. Neither is an identity certificate.
    operations = [
        *prep(22),
        *prep(11),
        "Barrier",
        "Barrier",
        {"Quantum": {"RXY": [3.14, 0.0, 22]}},
        "Barrier",
        *measure(11, 701),
        *measure(22, 702),
        "Barrier",
        *prep(33),
        "Barrier",
        *measure(33, 703),
    ]
    definitions = [
        ("Reset", 1, 0, 5, 11),
        ("Reset", 0, 5, 5, 22),
        ("Rxy", 0, 20, 10, None),
        ("Measure", 0, 50, 10, (9001, 702)),
        ("Measure", 1, 80, 10, (4001, 701)),
        ("Reset", 0, 100, 5, 33),
        ("Measure", 0, 120, 10, (9001, 703)),
    ]
    batches, lowered = [], []
    for index, (kind, q, start, duration, label) in enumerate(definitions):
        op = {kind: {"qubit_id": q}}
        metadata, results = {}, []
        if kind == "Reset":
            metadata = {"pecos_qec_allocation_id": str(label), "source_lowering_required": "true"}
        elif kind == "Measure":
            future, result = label
            op[kind]["result_id"] = future
            results = [result]
        else:
            op[kind].update(theta=3.14, phi=0.0)
        batches.append(
            {
                "shot_id": 100,
                "batch_index": index,
                "num_qubits": 2,
                "start_time_nanos": start,
                "duration_nanos": duration,
                "operations": [op],
                "normalized_operations": deepcopy([op]),
            },
        )
        lowered.append(
            {
                "gate_type": {"Reset": "PZ", "Measure": "MZ", "Rxy": "RXY1Q"}[kind],
                "qubits": [q],
                "metadata": metadata,
                "measurement_result_ids": results,
            },
        )
    header = {
        "format": "pecos_qis_operation_trace_v1",
        "engine_trace_id": 9,
        "shot_index": 7,
        "lowered_quantum_ops_complete": True,
        "runtime_events_complete": True,
    }
    trace = [
        dict(
            header,
            chunk_index=0,
            stage="pending_start",
            operations=operations,
            num_operations=len(operations),
            lowered_quantum_ops=lowered,
            named_result_traces=[],
        ),
        dict(
            header,
            chunk_index=1,
            stage="named_results",
            operations=[],
            num_operations=0,
            lowered_quantum_ops=[],
            named_result_traces=[
                {"name": f"qec:m:{i}", "result_ids": [r], "values": [False]} for i, r in enumerate((702, 701, 703))
            ],
        ),
        dict(
            header,
            chunk_index=2,
            stage="trace_complete",
            operations=[],
            num_operations=0,
            lowered_quantum_ops=[],
            named_result_traces=[],
        ),
    ]
    return plan, {"operation_trace": trace, "runtime_batches": batches}


def test_permuted_sites_scrambled_measurements_relabel_and_future_reuse(case):
    plan, evidence = case
    report = bind_runtime_lifetime_windows(plan, evidence)
    first, second, reused = report["windows"]
    assert [w["program_qubit"] for w in report["windows"]] == [22, 11, 33]
    assert [w["runtime_site"] for w in report["windows"]] == [0, 1, 0]
    assert [w["runtime_future_id"] for w in report["windows"]] == [9001, 4001, 9001]
    assert [w["program_result_id"] for w in report["windows"]] == [702, 701, 703]
    assert first["role_history"] == [
        {"moment": 0, "role": 1, "role_name": "b"},
        {"moment": 1, "role": 0, "role_name": "a"},
    ]
    assert [w["gap_ns"] for w in (first, second, reused)] == [30, 75, 15]
    assert [w["gate_interval_ns"] for w in (first, second, reused)] == [10, 0, 0]
    assert [w["storage_interval_ns"] for w in (first, second, reused)] == [40, 75, 15]
    assert report["normalized_window_gap_ns"] == 120
    assert report["excluded_between_window_gap_ns"] == 40
    assert report["endpoint_binding_verified"]
    assert not report["logical_carrier_residency_verified"]
    assert not report["live_idle_noise_model_verified"]
    assert not report["calibrated_durations_verified"]


def test_swapped_reset_labels_cannot_be_recovered_from_end_measurements(case):
    plan, evidence = case
    lowered = evidence["operation_trace"][0]["lowered_quantum_ops"]
    lowered[0]["metadata"], lowered[1]["metadata"] = lowered[1]["metadata"], lowered[0]["metadata"]
    with pytest.raises(ValueError, match="different program allocations"):
        bind_runtime_lifetime_windows(plan, evidence)


@pytest.mark.parametrize(
    "corruption",
    [
        "missing_reset_label",
        "missing_tag",
        "duplicate_tag",
        "duplicate_result",
        "incomplete",
        "mixed_shot",
        "bad_count",
        "terminal_gate",
        "extra_gate",
    ],
)
def test_incomplete_or_ambiguous_evidence_is_rejected(case, corruption):
    plan, evidence = case
    trace = evidence["operation_trace"]
    if corruption == "missing_reset_label":
        trace[0]["lowered_quantum_ops"][0]["metadata"] = {}
    elif corruption == "missing_tag":
        trace[1]["named_result_traces"].pop()
    elif corruption == "duplicate_tag":
        trace[1]["named_result_traces"].append(trace[1]["named_result_traces"][0])
    elif corruption == "duplicate_result":
        trace[1]["named_result_traces"][1]["result_ids"] = [702]
    elif corruption == "incomplete":
        trace[0]["lowered_quantum_ops_complete"] = False
    elif corruption == "mixed_shot":
        trace[1]["shot_index"] += 1
    elif corruption == "bad_count":
        trace[0]["num_operations"] -= 1
    else:
        target = trace[-1] if corruption == "terminal_gate" else trace[0]
        target["lowered_quantum_ops"].append({"gate_type": "RXY1Q", "qubits": [0]})
    with pytest.raises(ValueError, match=r"Cannot bind|audited runtime"):
        bind_runtime_lifetime_windows(plan, evidence)


def test_normalized_reset_changes_are_not_silently_admitted(case):
    plan, evidence = case
    evidence["runtime_batches"][0]["normalized_operations"][0]["Reset"]["qubit_id"] = 0
    with pytest.raises(ValueError, match="changed preparation/readout"):
        bind_runtime_lifetime_windows(plan, evidence)


def test_shared_boundary_batch_does_not_imply_a_preparation_time(case):
    plan, evidence = case
    extra = {"Rxy": {"qubit_id": 1, "theta": 0.0, "phi": 0.0}}
    evidence["runtime_batches"][0]["operations"].append(deepcopy(extra))
    evidence["runtime_batches"][0]["normalized_operations"].append(extra)
    evidence["operation_trace"][0]["lowered_quantum_ops"].insert(1, {"gate_type": "RXY1Q", "qubits": [1]})
    with pytest.raises(ValueError, match="boundary shares"):
        bind_runtime_lifetime_windows(plan, evidence)


def test_missing_source_provenance_fails(case):
    plan, evidence = case
    evidence["operation_trace"][0]["operations"][1]["TraceMetadata"]["metadata"] = {}
    with pytest.raises(ValueError, match="source reset has no required"):
        bind_runtime_lifetime_windows(plan, evidence)


def test_preparation_in_wrong_source_moment_fails(case):
    plan, evidence = case
    operations = evidence["operation_trace"][0]["operations"]
    operations.insert(3, operations.pop(6))  # Move the first cut between the two preparations.
    with pytest.raises(ValueError, match="source boundary moment mismatch"):
        bind_runtime_lifetime_windows(plan, evidence)


@pytest.mark.parametrize("flag", [True, 1, "yes"])
def test_allocation_provenance_requires_explicit_all_moments(case, flag):
    plan, _ = case
    with pytest.raises(ValueError, match="allocation_provenance"):
        plan.guppy_source(allocation_provenance=flag)


def test_zero_clocks_still_bind_without_claiming_measured_durations(case):
    plan, evidence = case
    for batch in evidence["runtime_batches"]:
        batch["start_time_nanos"] = batch["duration_nanos"] = 0
    report = bind_runtime_lifetime_windows(plan, evidence)
    assert report["normalized_window_gap_ns"] == 0
    assert report["timing_status"] == "zero_durations"
    assert not report["calibrated_durations_verified"]


def test_native_requests_bind_recycled_handles_by_allocation_occurrence(case):
    plan, evidence = case
    chunk = evidence["operation_trace"][0]

    def request(native):
        return {"RuntimeRequest": {"request": {"Native": native}, "result_id": None}}

    rewritten = []
    for op in chunk["operations"]:
        if not isinstance(op, dict):
            rewritten.append(op)
        elif "AllocateQubit" in op:
            rewritten.append(request("Qalloc"))
        elif "TraceMetadata" in op:
            item = op["TraceMetadata"]
            if item["qubit"] == 33:
                item["qubit"] = 22
                item["metadata"]["pecos_qec_allocation_id"] = "22"
            rewritten.append(op)
        elif "ReleaseQubit" in op:
            q = op["ReleaseQubit"]["id"]
            rewritten.append(request({"Qfree": 22 if q == 33 else q}))
        elif "Reset" in op["Quantum"]:
            q = op["Quantum"]["Reset"]
            rewritten.append(request({"Reset": 22 if q == 33 else q}))
        elif "Measure" in op["Quantum"]:
            q, result = op["Quantum"]["Measure"]
            rewritten.append(
                request({"Measure": {"qubit": 22 if q == 33 else q, "result_id": result, "leaked": False}}),
            )
        else:
            rewritten.append(op)
    chunk["operations"] = rewritten
    for low in chunk["lowered_quantum_ops"]:
        if low["metadata"].get("pecos_qec_allocation_id") == "33":
            low["metadata"]["pecos_qec_allocation_id"] = "22"
    report = bind_runtime_lifetime_windows(plan, evidence)
    assert [w["program_qubit"] for w in report["windows"]] == [22, 11, 22]
    assert [w["allocation_occurrence"] for w in report["windows"]] == [1, 1, 2]
    assert all(w["source_qubit_kind"] == "native_handle" for w in report["windows"])
