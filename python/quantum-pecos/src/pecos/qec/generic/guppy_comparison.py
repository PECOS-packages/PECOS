# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Compare reference and rescheduled Guppy executions under one recorded runtime profile."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

from pecos.qec.generic._schedule_analysis import _rescheduling_contract, _source_listing
from pecos.qec.generic.guppy_ordering import GuppyOrdering
from pecos.qec.generic.scheduled import SchedulePolicy

if TYPE_CHECKING:
    from pecos.qec.generic.guppy_schedule import GuppyScheduleAudit
    from pecos.qec.generic.scheduled import ScheduledCircuit


@dataclass(frozen=True)
class GuppyScheduleComparison:
    """Paired tagged observations; differences are candidate minus reference."""

    reference: GuppyScheduleAudit
    candidate: GuppyScheduleAudit
    _report_json: str = field(repr=False)

    def to_dict(self) -> dict:
        """Return an independent report, aligned by canonical source measurement."""
        return json.loads(self._report_json)

    def describe(self, *, include_windows: bool = False) -> str:
        """Show source depth and runtime time separately, without ranking schedules."""
        report = self.to_dict()
        candidate = report["candidate_policy"]
        lines = [
            "Tagged Guppy schedule comparison (candidate minus reference)",
            f"Metric | Reference | {candidate} | Difference",
        ]
        for name, values in report["metrics"].items():
            lines.append(f"{name} | {values['reference']} | {values['candidate']} | {values['delta']:+d}")
        if include_windows:
            lines.extend(("", f"Measurement | Source roles | Reference gap ns | {candidate} gap ns | Difference"))
            for row in report["measurement_windows"]:
                gap = row["gap_ns"]
                roles = " -> ".join(row["role_names"])
                lines.append(
                    f"{row['measurement_name']} | {roles} | {gap['reference']} | "
                    f"{gap['candidate']} | {gap['delta']:+d}",
                )
        lines.append("Single-trace observations; calibrated timing, logical residency and noise equivalence unverified")
        return "\n".join(lines)

    def explain(self, *, top_windows: int = 5, top_operations: int = 5) -> str:
        """Explain batch-time accounting and the largest source-bound gap changes."""
        if any(type(value) is not int or value < 0 for value in (top_windows, top_operations)):
            msg = "top_windows and top_operations must each be a nonnegative integer"
            raise ValueError(msg)
        report = self.to_dict()
        diagnostics = report["diagnostics"]
        source = diagnostics["source_placements"]
        clock = diagnostics["batch_accounting"]
        delta = clock["deltas"]
        lines = [
            "Schedule diagnostics (candidate minus reference)",
            f"Source operations: {source['listing_order_changed_count']} changed listing position; "
            f"{source['moment_membership_changed_count']} changed moment membership",
            f"Normalized operation-kind counts equal: {clock['operation_kind_counts_equal']}",
            f"Span change: {delta['sum_batch_duration_ns']:+d} ns batch-duration sum "
            f"- ({delta['overlap_excess_ns']:+d} ns overlap excess) "
            f"+ ({delta['uncovered_span_ns']:+d} ns uncovered clock) "
            f"= {delta['reported_span_ns']:+d} ns",
        ]
        placements = [
            row for row in source["changes"] if row["listing_order_changed"] or row["moment_membership_changed"]
        ]
        if top_operations and placements:
            lines.append("First source operations with changed order or moment membership:")
            for row in placements[:top_operations]:
                label = row["source_origin"] or row["source_operation"]
                lines.append(
                    f"  {label}: {row['gate']} {', '.join(row['roles'])}; "
                    f"moment {row['reference_moment']} -> {row['candidate_moment']}, "
                    f"listing {row['reference_listing_index']} -> {row['candidate_listing_index']}; "
                    f"moment peers +{len(row['added_moment_peers'])}/-{len(row['removed_moment_peers'])}",
                )
        for group in sorted(clock["changes"], key=lambda g: -abs(g["delta_sum_duration_ns"])):
            kinds = ", ".join(f"{kind} x{count}" for kind, count in group["operation_counts"].items()) or "empty"
            lines.append(
                f"  {kinds}, {group['duration_ns']} ns/batch: "
                f"{group['reference_batches']} -> {group['candidate_batches']} batches "
                f"({group['delta_sum_duration_ns']:+d} ns summed duration)",
            )
        changed = [row for row in report["measurement_windows"] if row["gap_ns"]["delta"]]
        if top_windows and changed:
            lines.append("Largest absolute window-gap changes:")
            ranked = sorted(changed, key=lambda r: (-abs(r["gap_ns"]["delta"]), r["measurement_index"]))
            for row in ranked[:top_windows]:
                roles = " -> ".join(row["role_names"])
                lines.append(f"  {row['measurement_name']} ({roles}): {row['gap_ns']['delta']:+d} ns")
        lines.append("Accounting only; specific source-gate causes and logical-noise improvements are not established")
        return "\n".join(lines)


def _paired_report(
    reference: ScheduledCircuit,
    candidate: ScheduledCircuit,
    first: GuppyScheduleAudit,
    second: GuppyScheduleAudit,
) -> dict:
    a, b = first.to_dict(), second.to_dict()
    metric_pairs = {
        "source_moments": (len(reference.moments), len(candidate.moments)),
        "runtime_batches": (a["normalized_runtime"]["batch_count"], b["normalized_runtime"]["batch_count"]),
        "reported_span_ns": (a["normalized_runtime"]["batch_span_ns"], b["normalized_runtime"]["batch_span_ns"]),
        "sum_window_gaps_ns": (
            a["lifetime_windows"]["normalized_window_gap_ns"],
            b["lifetime_windows"]["normalized_window_gap_ns"],
        ),
    }
    first_windows = {w["measurement_index"]: w for w in a["lifetime_windows"]["windows"]}
    second_windows = {w["measurement_index"]: w for w in b["lifetime_windows"]["windows"]}
    if set(first_windows) != set(second_windows) or len(first_windows) != reference.num_measurements:
        msg = "Schedule comparison requires complete canonical measurement windows"
        raise ValueError(msg)
    rows = []
    for index in sorted(first_windows):
        old, new = first_windows[index], second_windows[index]
        roles = [r["role"] for r in old["role_history"]]
        if roles != [r["role"] for r in new["role_history"]]:
            msg = "Schedule comparison changed the source lifetime role history"
            raise ValueError(msg)
        row = {
            "measurement_index": index,
            "measurement_tag": f"qec:m:{index}",
            "measurement_name": (
                reference.measurement_names[index] if reference.measurement_names else f"measurement[{index}]"
            ),
            "role_history": roles,
            "role_names": [reference.role_names[q] if reference.role_names else f"q[{q}]" for q in roles],
        }
        for name in ("gap_ns", "gate_interval_ns", "storage_interval_ns"):
            row[name] = {"reference": old[name], "candidate": new[name], "delta": new[name] - old[name]}
        row["runtime_sites"] = {"reference": old["runtime_site"], "candidate": new["runtime_site"]}
        rows.append(row)
    return {
        "reference_policy": reference.schedule.value,
        "candidate_policy": candidate.schedule.value,
        "source_listing_order_preserved": _source_listing(reference) == _source_listing(candidate),
        "metrics": {k: {"reference": x, "candidate": y, "delta": y - x} for k, (x, y) in metric_pairs.items()},
        "measurement_windows": rows,
        "timing_status": {
            "reference": a["normalized_runtime"]["timing_status"],
            "candidate": b["normalized_runtime"]["timing_status"],
        },
        "source_citations": list(reference.citations),
        "scope": "Two tagged coin-toss traces; source-equivalent unitary-region repacking",
        "source_rescheduling_contract_verified": True,
        "physical_schedule_verified": False,
        "calibrated_durations_verified": False,
        "logical_carrier_residency_verified": False,
        "noise_equivalence_verified": False,
        "adapter_state_verified": False,
        "gap_scope": "Sum over source-bound site windows; not wall-clock duration or calibrated idle exposure",
    }


def compare_guppy_schedules(
    circuit: ScheduledCircuit,
    *,
    runtime: object,
    seed: int = 0,
    candidate_policy: SchedulePolicy = SchedulePolicy.ASAP_UNITARIES,
) -> GuppyScheduleComparison:
    """Capture reference and candidate schedules with the same plugin bytes and arguments.

    Only a REFERENCE plan is accepted. Its selected candidate must preserve the
    per-wire unitary sequence in every region, all boundary moments, and all
    named measurement/parity definitions. Both captures use ALL_MOMENTS with
    required preparation labels. These labels may change runtime scheduling.
    Arguments are snapshotted once; a fresh adapter is constructed for each run.
    Plugin-internal nondeterminism and adapter state are not certified equivalent.
    """
    from pecos._engine_builders import _runtime_native_gates  # noqa: PLC0415 - optional native backend

    if circuit.schedule != SchedulePolicy.REFERENCE:
        msg = "Schedule comparison requires the original REFERENCE plan"
        raise ValueError(msg)
    if candidate_policy not in (SchedulePolicy.ASAP_UNITARIES, SchedulePolicy.COMPACT_UNITARIES):
        msg = "candidate_policy must be ASAP_UNITARIES or COMPACT_UNITARIES"
        raise ValueError(msg)
    candidate = circuit.rescheduled(candidate_policy)
    if _rescheduling_contract(circuit) != _rescheduling_contract(candidate):
        msg = "Candidate violates the source rescheduling contract"
        raise ValueError(msg)
    if candidate_policy == SchedulePolicy.COMPACT_UNITARIES and _source_listing(circuit) != _source_listing(candidate):
        msg = "Compact candidate changed the complete source listing order"
        raise ValueError(msg)
    if not hasattr(runtime, "library_file") or not callable(getattr(runtime, "get_init_args", None)):
        msg = "Schedule comparisons require an explicit runtime plugin object"
        raise TypeError(msg)
    library = Path(runtime.library_file).resolve(strict=True)
    digest = hashlib.sha256(library.read_bytes()).hexdigest()
    args = tuple(runtime.get_init_args())
    if any(not isinstance(arg, str) for arg in args):
        msg = "Runtime init arguments must be strings"
        raise TypeError(msg)
    factory = getattr(runtime, "batch_adapter_factory", None)
    gates = _runtime_native_gates(runtime)
    gates = None if gates is None else tuple(gates)
    search_dirs = tuple(str(Path(p).resolve()) for p in getattr(runtime, "library_search_dirs", []))
    clock_model = getattr(runtime, "clock_model", "continuous")
    frozen_runtime = SimpleNamespace(
        library_file=library,
        get_init_args=lambda: list(args),
        batch_adapter_factory=factory,
        native_gates=gates,
        library_search_dirs=search_dirs,
        clock_model=clock_model,
    )
    audits = []
    for plan in (circuit, candidate):
        if hashlib.sha256(library.read_bytes()).hexdigest() != digest:
            msg = "Runtime plugin library changed during the comparison"
            raise ValueError(msg)
        audits.append(
            plan.audit_guppy_schedule(
                runtime=frozen_runtime,
                seed=seed,
                ordering=GuppyOrdering.ALL_MOMENTS,
                include_lifetimes=True,
            ),
        )
    if hashlib.sha256(library.read_bytes()).hexdigest() != digest:
        msg = "Runtime plugin library changed during the comparison"
        raise ValueError(msg)
    report = _paired_report(circuit, candidate, *audits)
    from pecos.qec.generic.guppy_diagnostics import diagnose_schedule_pair  # noqa: PLC0415

    report["diagnostics"] = diagnose_schedule_pair(circuit, candidate, audits[0].evidence(), audits[1].evidence())
    report["runtime_profile"] = {
        "selector": f"{type(runtime).__module__}.{type(runtime).__qualname__}",
        "library_file": str(library),
        "library_sha256": digest,
        "init_args": list(args),
        "seed": seed,
        "native_gates": gates,
        "library_search_dirs": search_dirs,
        "native_future_protocol": True,
        "clock_model": clock_model,
        "event_adapter": "identity" if factory is None else "delegated",
    }
    return GuppyScheduleComparison(*audits, json.dumps(report))
