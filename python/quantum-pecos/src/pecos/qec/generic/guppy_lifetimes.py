# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Bind tagged preparation/readout endpoints to normalized runtime site windows."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pecos.qec.generic.scheduled import ScheduledCircuit


_ALLOCATION_KEY = "pecos_qec_allocation_id"


def _require(condition: bool, message: str) -> None:
    if not condition:
        msg = f"Cannot bind runtime lifetime windows: {message}"
        raise ValueError(msg)


def _source_lifetimes(circuit: ScheduledCircuit) -> list[dict]:
    live, rows = {}, []
    for moment, operations in enumerate(circuit.moments):
        for op in operations:
            if op.name == "R":
                q = op.qubits[0]
                live[q] = {"preparation_moment": moment,
                           "role_history": [{"moment": moment, "role": q}]}
            elif op.name == "RELABEL":
                a, b = op.qubits
                live[a], live[b] = live[b], live[a]
                for q in (a, b):
                    live[q]["role_history"].append({"moment": moment, "role": q})
            elif op.name == "M":
                row = live.pop(op.qubits[0])
                row.update({"measurement_moment": moment, "measurement_index": len(rows),
                            "measurement_tag": f"qec:m:{len(rows)}"})
                for binding in row["role_history"]:
                    if circuit.role_names:
                        binding["role_name"] = circuit.role_names[binding["role"]]
                rows.append(row)
    return rows


def _qis_allocations(circuit: ScheduledCircuit, chunks: list[dict], rows: list[dict]) -> dict:
    from pecos._qis_trace_replay import _validate_audited_trace_stream  # noqa: PLC0415 - optional native backend

    _validate_audited_trace_stream(chunks)
    for chunk in chunks:
        _require(chunk.get("lowered_quantum_ops_complete") is True and chunk.get("runtime_events_complete") is True,
                 "incomplete runtime lowering or event trace")
    tags = {}
    for chunk in chunks:
        for item in chunk["named_result_traces"]:
            name = item["name"]
            _require(name not in tags and len(item["result_ids"]) == 1 and len(item["values"]) == 1,
                     "ambiguous scalar named result")
            _require(type(item["result_ids"][0]) is int and item["result_ids"][0] >= 0,
                     "invalid named program result ID")
            tags[name] = item["result_ids"][0]
    _require(set(tags) == {r["measurement_tag"] for r in rows}, "missing or unexpected measurement tags")
    _require(len(set(tags.values())) == len(tags), "duplicate named program result IDs")
    active, by_result, seen = {}, {}, set()
    native_allocations = []
    occurrences = defaultdict(int)
    moment = 0
    for chunk in chunks:
        for raw_op in chunk["operations"]:
            op = raw_op
            if op == "Barrier":
                moment += 1
                continue
            if not isinstance(op, dict):
                continue
            request = op.get("RuntimeRequest", {}).get("request", {})
            native = request.get("Native")
            if native == "Qalloc":
                native_allocations.append(moment)
                continue
            if isinstance(native, dict):
                if "Reset" in native:
                    op = {"Quantum": {"Reset": native["Reset"]}}
                elif "Measure" in native:
                    measure = native["Measure"]
                    _require(not measure["leaked"], "leakage-aware source measurement is unsupported")
                    op = {"Quantum": {"Measure": [measure["qubit"], measure["result_id"]]}}
                elif "Qfree" in native:
                    op = {"ReleaseQubit": {"id": native["Qfree"]}}
            if "AllocateQubit" in op:
                q = op["AllocateQubit"]["id"]
                _require(type(q) is int and q >= 0 and q not in seen, "reused or invalid program allocation ID")
                seen.add(q)
                active[q] = {"program_qubit": q, "allocation_moment": moment, "allocation_occurrence": 1}
            elif "TraceMetadata" in op:
                item = op["TraceMetadata"]
                metadata = item["metadata"]
                if _ALLOCATION_KEY in metadata:
                    q = item["qubit"]
                    if native_allocations:
                        _require(type(q) is int and q >= 0 and q not in active,
                                 "overlapping or invalid native allocation")
                        occurrences[q] += 1
                        active[q] = {"program_qubit": q, "allocation_moment": native_allocations.pop(0),
                                     "allocation_occurrence": occurrences[q], "source_qubit_kind": "native_handle"}
                    _require(q in active and "reset_moment" not in active[q]
                             and metadata[_ALLOCATION_KEY] == str(q)
                             and metadata.get("source_lowering_required") == "true",
                             "invalid source allocation provenance")
                    active[q]["tagged"] = True
            elif "Quantum" in op:
                quantum = op["Quantum"]
                if "Reset" in quantum:
                    q = quantum["Reset"]
                    _require(q in active and "reset_moment" not in active[q], "reset without a fresh allocation")
                    _require(active[q].get("tagged") is True, "source reset has no required allocation provenance")
                    active[q]["reset_moment"] = moment
                elif "Measure" in quantum:
                    q, result = quantum["Measure"]
                    _require(q in active and "reset_moment" in active[q] and "program_result_id" not in active[q],
                             "measurement without one prepared allocation")
                    _require(type(result) is int and result >= 0 and result not in by_result,
                             "duplicate program result ID")
                    active[q].update({"program_result_id": result, "measurement_moment": moment})
                    by_result[result] = active[q]
                elif "MeasureLeaked" in quantum:
                    msg = "Cannot bind runtime lifetime windows: leakage-aware source measurement is unsupported"
                    raise ValueError(msg)
            elif "ReleaseQubit" in op:
                q = op["ReleaseQubit"]["id"]
                _require(q in active and "program_result_id" in active[q], "release without destructive measurement")
                active.pop(q)
    _require(not active and not native_allocations and len(by_result) == len(rows), "incomplete allocation lifetimes")
    _require(moment == max(0, len(circuit.moments) - 1), "ALL_MOMENTS barrier coverage required")
    bindings = {}
    for row in rows:
        result = tags[row["measurement_tag"]]
        _require(result in by_result, "named result has no source measurement")
        allocation = by_result[result]
        _require(allocation["allocation_moment"] == allocation["reset_moment"] == row["preparation_moment"]
                 and allocation["measurement_moment"] == row["measurement_moment"], "source boundary moment mismatch")
        bindings[result] = {**deepcopy(row), **allocation}
    return bindings


def _boundaries(operations: list[dict]) -> list[dict]:
    return [o for o in operations if any(k in o for k in ("Reset", "Measure", "MeasureLeaked"))]


def bind_runtime_lifetime_windows(circuit: ScheduledCircuit, evidence: dict) -> dict:
    """Bind source endpoints using required reset metadata and measured result IDs.

    Requires a tagged ALL_MOMENTS capture. The output describes normalized runtime
    site intervals. It does not certify logical-state residency, hardware clocks,
    or an idle-noise model. Ambiguous boundary batches and changed reset/readout
    boundaries are rejected. Runtime future IDs are never equated with program IDs.
    """
    from pecos.qec.generic.guppy_schedule import summarize_runtime_batches  # noqa: PLC0415 - avoid module cycle

    chunks, batches = evidence["operation_trace"], evidence["runtime_batches"]
    normalized = [{**b, "operations": b["normalized_operations"]} for b in batches]
    timing = summarize_runtime_batches(normalized)
    bindings = _qis_allocations(circuit, chunks, _source_lifetimes(circuit))
    # The native lowering preserves emission order and maps measurement future
    # IDs to program result IDs. Both sides expose physical simulator site IDs.
    lowered = []
    for chunk in chunks:
        lowered.extend(op for op in chunk["lowered_quantum_ops"] if op["gate_type"] != "Idle")
    cursor, active, windows, allocated = 0, {}, [], set()
    occurrences = defaultdict(int)
    native_handles = {b["program_qubit"] for b in bindings.values() if b.get("source_qubit_kind") == "native_handle"}
    for batch in batches:
        _require(_boundaries(batch["operations"]) == _boundaries(batch["normalized_operations"]),
                 "event adapter changed preparation/readout boundaries")
        grouped = defaultdict(list)
        endpoint_labels = {}
        for operation in batch["normalized_operations"]:
            ((kind, data),) = operation.items()
            if kind in {"Rzz", "Rpp"}:
                sites = (data["qubit_id_1"], data["qubit_id_2"])
            else:
                _require(kind != "Custom", "uninterpreted custom event")
                sites = (data["qubit_id"],)
            for q in sites:
                grouped[q].append(kind)
            _require(cursor < len(lowered), "missing lowered operation")
            low = lowered[cursor]
            cursor += 1
            expected = {"Reset": "PZ", "Measure": "MZ", "MeasureLeaked": "MeasureLeaked",
                        "Rxy": "RXY1Q", "Rz": "RZ", "Rzz": "RZZ", "Rpp": "RXYXY2Q"}[kind]
            _require(low["gate_type"] == expected and low["qubits"] == list(sites),
                     "runtime and lowered operation emission order/site mismatch")
            if kind in {"Reset", "Measure", "MeasureLeaked"}:
                if kind == "Reset":
                    label = low["metadata"].get(_ALLOCATION_KEY)
                    _require(isinstance(label, str) and label.isascii() and label.isdecimal(),
                             "missing required allocation provenance on reset")
                    endpoint_labels[sites[0]] = int(label)
                else:
                    _require(kind == "Measure" and len(low["measurement_result_ids"]) == 1,
                             "unsupported or ambiguous readout kind")
                    endpoint_labels[sites[0]] = (low["measurement_result_ids"][0], data["result_id"])
        start = batch["start_time_nanos"]
        end = start + batch["duration_nanos"]
        for q, kinds in grouped.items():
            is_boundary = any(k in {"Reset", "Measure", "MeasureLeaked"} for k in kinds)
            _require(not is_boundary or len(kinds) == 1,
                     "lifetime boundary shares its site/batch with another operation")
            if kinds == ["Reset"]:
                allocation = endpoint_labels[q]
                _require(q not in active and (allocation not in allocated or allocation in native_handles),
                         "overlapping or repeated tagged preparation")
                occurrences[allocation] += 1
                allocated.add(allocation)
                active[q] = {"runtime_site": q, "program_qubit": allocation,
                             "allocation_occurrence": occurrences[allocation],
                             "preparation_start_ns": start, "preparation_end_ns": end,
                             "preparation_batch": batch["batch_index"], "previous_end": end,
                             "gate_interval_ns": 0, "gap_intervals": []}
                continue
            _require(q in active, "operation outside a reset-to-measurement window")
            window = active[q]
            if start > window["previous_end"]:
                window["gap_intervals"].append({"start_ns": window["previous_end"], "end_ns": start,
                                                 "before_batch": batch["batch_index"]})
            if kinds == ["Measure"]:
                result, future = endpoint_labels[q]
                _require(result in bindings, "readout does not bind to a unique source lifetime")
                source = bindings.pop(result)
                _require(source["program_qubit"] == window["program_qubit"]
                         and source["allocation_occurrence"] == window["allocation_occurrence"],
                         "preparation and measurement belong to different program allocations")
                window.update(source)
                window.update({"runtime_future_id": future, "measurement_start_ns": start,
                               "measurement_end_ns": end, "measurement_batch": batch["batch_index"]})
                window["gap_ns"] = sum(g["end_ns"] - g["start_ns"] for g in window["gap_intervals"])
                window["storage_interval_ns"] = start - window["preparation_end_ns"]
                _require(window["gap_ns"] + window["gate_interval_ns"] == window["storage_interval_ns"],
                         "window intervals do not partition the recorded clock")
                del window["previous_end"]
                windows.append(active.pop(q))
            else:
                window["gate_interval_ns"] += end - start
                window["previous_end"] = end
    _require(cursor == len(lowered) and not active and not bindings, "unconsumed endpoints or unclosed windows")
    windows.sort(key=lambda w: w["measurement_index"])
    total = sum(w["gap_ns"] for w in windows)
    all_gaps = sum(s["between_operation_gap_ns"] for s in timing["sites"])
    return {"status": "source_bound_site_windows", "windows": windows, "normalized_window_gap_ns": total,
            "excluded_between_window_gap_ns": all_gaps - total,
            "timing_status": timing["timing_status"], "execution_scope": "required-reset-provenance tagged execution",
            "scope": "Normalized site gaps after preparation completes and before destructive readout starts",
            "endpoint_binding_verified": True, "logical_carrier_residency_verified": False,
            "calibrated_durations_verified": False, "live_idle_noise_model_verified": False,
            "noise_equivalence_verified": False}
