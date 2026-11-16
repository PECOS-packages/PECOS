# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Exact batch-clock accounting and source-placement differences, without causal claims."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

from pecos.qec.generic._schedule_analysis import _rescheduling_contract, _source_placements
from pecos.qec.generic.guppy_schedule import summarize_runtime_batches

if TYPE_CHECKING:
    from pecos.qec.generic.scheduled import ScheduledCircuit


def _batch_accounting(evidence: dict) -> dict:
    batches = [{**batch, "operations": batch["normalized_operations"]} for batch in evidence["runtime_batches"]]
    timing = summarize_runtime_batches(batches)
    groups = Counter()
    intervals = []
    for batch in batches:
        kinds = tuple(sorted(Counter(next(iter(op)) for op in batch["operations"]).items()))
        duration = batch["duration_nanos"]
        groups[kinds, duration] += 1
        if duration:
            intervals.append((batch["start_time_nanos"], batch["start_time_nanos"] + duration))
    covered, previous_end = 0, None
    for start, end in sorted(intervals):
        if previous_end is None:
            covered += end - start
        elif end > previous_end:
            covered += end - max(start, previous_end)
        previous_end = end if previous_end is None else max(previous_end, end)
    total = sum(batch["duration_nanos"] for batch in batches)
    span = timing["batch_span_ns"]
    return {"reported_span_ns": span, "sum_batch_duration_ns": total,
            "covered_clock_ns": covered, "overlap_excess_ns": total - covered,
            "uncovered_span_ns": span - covered, "operation_counts": timing["operation_counts"],
            "groups": [{"operation_counts": dict(kinds), "duration_ns": duration,
                        "batch_count": count, "sum_duration_ns": duration * count}
                       for (kinds, duration), count in sorted(groups.items())]}


def _placement_changes(reference: ScheduledCircuit, candidate: ScheduledCircuit) -> dict:
    old, old_groups = _source_placements(reference)
    new, new_groups = _source_placements(candidate)
    if old.keys() != new.keys() or any((old[k]["gate"], old[k]["roles"]) != (new[k]["gate"], new[k]["roles"])
                                     for k in old):
        msg = "Cannot align source operations across schedules"
        raise ValueError(msg)
    changes = []
    for identity, row in old.items():
        other = new[identity]
        moved = row["moment"] != other["moment"]
        reordered = row["listing_index"] != other["listing_index"]
        regrouped = old_groups[row["moment"]] != new_groups[other["moment"]]
        if moved or reordered or regrouped:
            changes.append({"source_operation": identity, "gate": row["gate"], "roles": row["role_names"],
                            "source_origin": row["source_origin"],
                            "reference_moment": row["moment"], "candidate_moment": other["moment"],
                            "reference_listing_index": row["listing_index"],
                            "candidate_listing_index": other["listing_index"],
                            "added_moment_peers": sorted(new_groups[other["moment"]] - old_groups[row["moment"]]),
                            "removed_moment_peers": sorted(old_groups[row["moment"]] - new_groups[other["moment"]]),
                            "moment_index_changed": moved, "listing_order_changed": reordered,
                            "moment_membership_changed": regrouped})
    return {"matched_operation_count": len(old), "changes": changes,
            "moment_index_changed_count": sum(r["moment_index_changed"] for r in changes),
            "listing_order_changed_count": sum(r["listing_order_changed"] for r in changes),
            "moment_membership_changed_count": sum(r["moment_membership_changed"] for r in changes),
            "scope": "Authored source placements; no source-operation-to-runtime-batch binding"}


def diagnose_schedule_pair(reference: ScheduledCircuit, candidate: ScheduledCircuit,
                           reference_evidence: dict, candidate_evidence: dict) -> dict:
    """Account for observed differences between source-equivalent schedules.

    Batch-duration sums are additive even when global clock intervals overlap.
    Span = summed batch duration - overlap excess + uncovered span. Empty and
    zero-duration batches retain their reported timestamps. Grouping by operation
    kinds and duration does not certify equal angles, sites, or physical pulses.
    The caller supplies captures; this does not revalidate their source/runtime
    provenance or establish a source-operation-to-runtime-batch binding.
    """
    if _rescheduling_contract(reference) != _rescheduling_contract(candidate):
        msg = "Diagnostics require the same source rescheduling contract"
        raise ValueError(msg)
    old, new = _batch_accounting(reference_evidence), _batch_accounting(candidate_evidence)

    def groups(summary: dict) -> dict:
        return {(tuple(row["operation_counts"].items()), row["duration_ns"]): row for row in summary["groups"]}

    a, b = groups(old), groups(new)
    changes = []
    for kinds, duration in sorted(a.keys() | b.keys()):
        key = (kinds, duration)
        x, y = a.get(key, {}).get("batch_count", 0), b.get(key, {}).get("batch_count", 0)
        if x != y:
            changes.append({"operation_counts": dict(kinds), "duration_ns": duration,
                            "reference_batches": x, "candidate_batches": y,
                            "delta_batches": y - x, "delta_sum_duration_ns": (y - x) * duration})
    deltas = {name: new[name] - old[name] for name in
              ("reported_span_ns", "sum_batch_duration_ns", "overlap_excess_ns", "uncovered_span_ns")}
    if deltas["reported_span_ns"] != (deltas["sum_batch_duration_ns"] - deltas["overlap_excess_ns"]
                                      + deltas["uncovered_span_ns"]):
        msg = "Batch clock accounting did not reconcile"
        raise ValueError(msg)
    return {"source_placements": _placement_changes(reference, candidate),
            "batch_accounting": {"reference": old, "candidate": new, "changes": changes, "deltas": deltas,
                                 "operation_kind_counts_equal": old["operation_counts"] == new["operation_counts"],
                                 "span_identity": "sum_batch_duration_ns - overlap_excess_ns + uncovered_span_ns",
                                 "scope": "Whole reported batch intervals; grouping uses normalized operation kinds"},
            "runtime_causality_verified": False,
            "scope": "Source-placement differences and exact clock accounting, not gate-level causal attribution"}
