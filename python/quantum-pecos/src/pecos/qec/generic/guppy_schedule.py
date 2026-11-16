# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Observe runtime batch clocks without interpreting them as calibrated QEC time."""

from __future__ import annotations

import json
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pecos.qec.generic.guppy_ordering import GuppyOrdering

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pecos.qec.generic.scheduled import ScheduledCircuit


def _qis_quantum_name(operation: object) -> str | None:
    """Count quantum work in both logical-QIR and native-handle traces."""
    if not isinstance(operation, dict):
        return None
    if "Quantum" in operation:
        return next(iter(operation["Quantum"]))
    native = operation.get("RuntimeRequest", {}).get("request", {}).get("Native")
    if not isinstance(native, dict):
        return None
    name = next(iter(native))
    if name in {"Rxy", "Rz", "Rzz", "Rpp"}:
        return name.upper()
    if name == "Reset":
        return name
    if name == "Measure":
        return "MeasureLeaked" if native[name]["leaked"] else "Measure"
    # Allocation, release, future reads/refcounts and barriers are not gates.
    return None


def summarize_runtime_batches(batches: Sequence[dict]) -> dict:
    """Summarize batch-assigned occupancy and between-operation gaps by runtime site.

    Gaps exclude time before the first/after the last operation on a site. They
    may cross allocation lifetimes and are NOT live-qubit idle-noise exposure.
    Batch duration is counted once per participating site, including both sites
    of a two-qubit operation. Multiple operations can share one site's batch
    interval. Custom events contribute no inferred site occupancy.
    """
    counts, occupancy, gaps, previous = Counter(), Counter(), Counter(), {}
    first, last, positive = None, 0, False
    shot, capacity = None, None
    for index, batch in enumerate(batches):
        start, duration = batch["start_time_nanos"], batch["duration_nanos"]
        if any(type(v) is not int or v < 0 for v in (start, duration, batch["num_qubits"], batch["shot_id"])):
            msg = "Runtime batch clock, capacity, and shot ID must be nonnegative integers"
            raise ValueError(msg)
        if batch["batch_index"] != index or type(batch["batch_index"]) is not int:
            msg = "Runtime batch indices must be consecutive from zero"
            raise ValueError(msg)
        if index == 0:
            shot, capacity = batch["shot_id"], batch["num_qubits"]
        if (shot, capacity) != (batch["shot_id"], batch["num_qubits"]):
            msg = "Runtime batches must belong to one shot with one site capacity"
            raise ValueError(msg)
        first = start if first is None else min(first, start)
        end = start + duration
        if end > (1 << 64) - 1:
            msg = "Runtime batch end exceeds the native u64 clock"
            raise ValueError(msg)
        last, positive = max(last, end), positive or duration > 0
        touched = set()
        for operation in batch["operations"]:
            if len(operation) != 1:
                msg = "Expected one runtime operation variant"
                raise ValueError(msg)
            ((name, data),) = operation.items()
            if name in {"Rxy", "Rz", "Reset", "Measure", "MeasureLeaked"}:
                sites = (data["qubit_id"],)
            elif name in {"Rzz", "Rpp"}:
                sites = (data["qubit_id_1"], data["qubit_id_2"])
            elif name == "Custom":
                sites = ()
            else:
                msg = f"Unsupported runtime operation variant: {name}"
                raise ValueError(msg)
            counts[name] += 1
            if len(set(sites)) != len(sites) or any(type(q) is not int or not 0 <= q < capacity for q in sites):
                msg = "Invalid runtime site IDs"
                raise ValueError(msg)
            touched.update(sites)
        for q in touched:
            if q in previous:
                if start < previous[q]:
                    msg = f"Runtime batches overlap or go backwards at site {q}"
                    raise ValueError(msg)
                gaps[q] += start - previous[q]
            occupancy[q] += duration
            previous[q] = end
    return {
        "batch_count": len(batches),
        "site_capacity": capacity,
        "timing_status": ("nonzero_reported_durations" if positive else "zero_durations") if batches else "no_batches",
        "first_batch_start_ns": first,
        "last_batch_end_ns": last if batches else None,
        "batch_span_ns": last - first if batches else 0,
        "operation_counts": dict(sorted(counts.items())),
        "sites": [
            {"runtime_site": q, "batch_assigned_operation_ns": occupancy[q], "between_operation_gap_ns": gaps[q]}
            for q in sorted(previous)
        ],
        "gap_scope": "Between emitted operations; excludes initial/final gaps; may span site reuse",
        "live_idle_exposure_verified": False,
    }


@dataclass(frozen=True)
class GuppyScheduleAudit:
    """One completed trace and timestamped runtime batches, retained as JSON snapshots."""

    _report_json: str = field(repr=False)
    _evidence_json: str = field(repr=False)

    def to_dict(self) -> dict:
        """Return an independent summary with explicit evidence limits."""
        return json.loads(self._report_json)

    def evidence(self) -> dict:
        """Return independent raw trace, original batches, and normalized operations."""
        return json.loads(self._evidence_json)

    def describe(self) -> str:
        """Describe observed batch time without claiming hardware calibration."""
        data = self.to_dict()
        timing = data["normalized_runtime"]
        windows = data.get("lifetime_windows")
        lifetimes = (
            ""
            if windows is None
            else (
                f"Tagged execution: {len(windows['windows'])} source-bound site windows, "
                f"{windows['normalized_window_gap_ns']} ns total window gaps; "
                f"{windows['excluded_between_window_gap_ns']} ns between-window gaps excluded\n"
            )
        )
        return (
            f"Runtime schedule: {timing['batch_count']} batches, {timing['batch_span_ns']} ns reported span\n"
            f"Timing status: {timing['timing_status']}\n"
            f"Barrier requests: {data['observed_barrier_requests']}/{data['expected_barrier_requests']}\n"
            f"{lifetimes}"
            "Calibrated durations, source-moment timing, and noise equivalence: not established"
        )


def audit_guppy_schedule(
    circuit: ScheduledCircuit,
    *,
    runtime: object,
    seed: int = 0,
    ordering: GuppyOrdering = GuppyOrdering.ALL_MOMENTS,
    batch_adapter_factory: object | None = None,
    include_lifetimes: bool = False,
) -> GuppyScheduleAudit:
    """Capture one ideal trace and preserve an existing runtime event adapter.

    Supply an explicit plugin object with library_file and get_init_args(), so
    runtime selection and its adapter are known before recording is installed.
    Names, paths, and automatic runtime selection are deliberately not accepted.
    Original and normalized batch operations are saved separately. Unknown custom
    events fail if no adapter interprets them. Capture uses PECOS's coin-toss trace
    execution; outcomes are provenance evidence, not ideal stabilizer statistics.
    This is separate from audit_guppy()'s compiler-bound measurement/DEM analysis.
    include_lifetimes requires ALL_MOMENTS and tags each preparation: metadata can
    introduce local runtime barriers/drains, so these clocks describe the tagged
    execution. Bound endpoints do not certify logical-state residency or noise.
    """
    from pecos import capture_qis_operation_trace  # noqa: PLC0415 - optional backend

    if not hasattr(runtime, "library_file") or not callable(getattr(runtime, "get_init_args", None)):
        msg = "Schedule audits require an explicit runtime plugin object with library_file and get_init_args()"
        raise TypeError(msg)
    if type(include_lifetimes) is not bool or (include_lifetimes and ordering != GuppyOrdering.ALL_MOMENTS):
        msg = "include_lifetimes must be bool and requires ALL_MOMENTS when enabled"
        raise ValueError(msg)
    contract = circuit.guppy_ordering_plan(ordering)
    delegate_factory = batch_adapter_factory
    if delegate_factory is None:
        delegate_factory = getattr(runtime, "batch_adapter_factory", None)
    if delegate_factory is not None and not callable(delegate_factory):
        msg = "batch_adapter_factory must be callable"
        raise TypeError(msg)
    batches = []

    class Recorder:
        def __init__(self) -> None:
            self.delegate = None if delegate_factory is None else delegate_factory()

        def normalize(self, batch: dict) -> list:
            snapshot = deepcopy(batch)
            output = batch["operations"] if self.delegate is None else self.delegate.normalize(batch)
            if not isinstance(output, list):
                msg = "Runtime adapter must return a list of operations"
                raise TypeError(msg)
            if any("Custom" in operation for operation in output):
                msg = "Runtime schedule audit cannot interpret an unhandled custom event"
                raise ValueError(msg)
            snapshot["normalized_operations"] = deepcopy(output)
            batches.append(snapshot)
            return output

    program = circuit.guppy_program(ordering=ordering, allocation_provenance=include_lifetimes)
    trace = capture_qis_operation_trace(
        program,
        circuit.num_qubits,
        seed=seed,
        runtime=runtime,
        batch_adapter_factory=Recorder,
    )
    source_summary = summarize_runtime_batches(batches)
    normalized = summarize_runtime_batches([{**b, "operations": b["normalized_operations"]} for b in batches])
    operations = [op for chunk in trace for op in chunk["operations"]]
    expected = contract.to_dict()["runtime_barrier_calls"]
    observed = operations.count("Barrier")
    if observed != expected:
        msg = f"Observed {observed} source barrier requests; expected {expected}"
        raise ValueError(msg)
    moments = None
    if ordering == GuppyOrdering.ALL_MOMENTS:
        segments = [[]]
        for operation in operations:
            if operation == "Barrier":
                segments.append([])
            else:
                segments[-1].append(operation)
        # A zero-moment experiment has an empty trace rather than one authored moment.
        if not circuit.moments:
            segments = []
        moments = []
        for index, (source, segment) in enumerate(zip(circuit.moments, segments, strict=True)):
            qis_counts = Counter(name for op in segment if (name := _qis_quantum_name(op)) is not None)
            moments.append(
                {
                    "moment": index,
                    "source_gate_counts": dict(Counter(op.name for op in source)),
                    "qis_quantum_counts": dict(qis_counts),
                },
            )
    report = {
        "ordering": contract.to_dict(),
        "seed": seed,
        "source_schedule": circuit.schedule.value,
        "source_citations": list(circuit.citations),
        "runtime_selector": f"{type(runtime).__module__}.{type(runtime).__qualname__}",
        "event_adapter": "identity" if delegate_factory is None else "delegated",
        "source_runtime": source_summary,
        "normalized_runtime": normalized,
        "expected_barrier_requests": expected,
        "observed_barrier_requests": observed,
        "qis_moments": moments,
        "scope": "One completed coin-toss trace; reported batch clock and source QIS barrier segmentation",
        "source_moment_timing_verified": False,
        "calibrated_durations_verified": False,
        "physical_schedule_verified": False,
        "noise_equivalence_verified": False,
        "measurement_binding_verified": False,
        "trace_lowering_complete": bool(trace) and all(c.get("lowered_quantum_ops_complete") is True for c in trace),
        "runtime_events_complete": bool(trace) and all(c.get("runtime_events_complete") is True for c in trace),
        "allocation_provenance": include_lifetimes,
    }
    evidence = {"operation_trace": trace, "runtime_batches": batches}
    if include_lifetimes:
        from pecos.qec.generic.guppy_lifetimes import bind_runtime_lifetime_windows  # noqa: PLC0415

        report["lifetime_windows"] = bind_runtime_lifetime_windows(circuit, evidence)
    return GuppyScheduleAudit(json.dumps(report), json.dumps(evidence))
