"""Hand-calculated clock unions, batching and source identities for diagnostics."""

import json
from dataclasses import replace

import pytest
from pecos.qec.generic import CircuitOp as Op
from pecos.qec.generic import GuppyScheduleComparison, ScheduledCircuit, SchedulePolicy
from pecos.qec.generic.guppy_diagnostics import _batch_accounting, _placement_changes, diagnose_schedule_pair


def _evidence(*rows):
    return {"runtime_batches": [{"shot_id": 0, "batch_index": i, "num_qubits": 2,
                                 "start_time_nanos": start, "duration_nanos": duration,
                                 "normalized_operations": [{"Rxy": {"qubit_id": q}} for q in sites]}
                                for i, (start, duration, sites) in enumerate(rows)]}


def _plan():
    return ScheduledCircuit(2, ((Op("R", (0,)), Op("R", (1,))), (Op("X", (0,)),), (Op("X", (1,)),),
                                (Op("X", (0,)),), (Op("M", (0,)), Op("M", (1,)))), (), ())


def test_span_decomposition_with_overlap_and_global_gaps():
    reference = _evidence((0, 10, (0,)), (5, 10, (1,)), (20, 5, (0,)))
    candidate = _evidence((0, 10, (0,)), (0, 10, (1,)), (10, 5, (0,)))
    plan = _plan()
    diagnostics = diagnose_schedule_pair(plan, plan.rescheduled(SchedulePolicy.ASAP_UNITARIES), reference, candidate)
    old, new = (diagnostics["batch_accounting"][key] for key in ("reference", "candidate"))
    assert old["sum_batch_duration_ns"] == new["sum_batch_duration_ns"] == 25
    assert (old["covered_clock_ns"], old["overlap_excess_ns"], old["uncovered_span_ns"]) == (20, 5, 5)
    assert (new["covered_clock_ns"], new["overlap_excess_ns"], new["uncovered_span_ns"]) == (15, 10, 0)
    assert diagnostics["batch_accounting"]["deltas"] == {
        "reported_span_ns": -10, "sum_batch_duration_ns": 0, "overlap_excess_ns": 5, "uncovered_span_ns": -5,
    }
    assert not diagnostics["runtime_causality_verified"]


def test_batch_grouping_counts_duration_once_and_records_changed_packing():
    old = _evidence((0, 10, (0,)), (10, 10, (1,)), (20, 10, (0,)))
    new = _evidence((0, 10, (0, 1)), (10, 10, (0,)))
    plan = _plan()
    accounting = diagnose_schedule_pair(plan, plan, old, new)["batch_accounting"]
    assert accounting["operation_kind_counts_equal"]
    assert accounting["changes"] == [
        {"operation_counts": {"Rxy": 1}, "duration_ns": 10, "reference_batches": 3,
         "candidate_batches": 1, "delta_batches": -2, "delta_sum_duration_ns": -20},
        {"operation_counts": {"Rxy": 2}, "duration_ns": 10, "reference_batches": 0,
         "candidate_batches": 1, "delta_batches": 1, "delta_sum_duration_ns": 10},
    ]
    assert accounting["deltas"]["reported_span_ns"] == -10


def test_nested_intervals_and_disjoint_out_of_order_emission():
    report = _batch_accounting(_evidence((5, 10, (1,)), (20, 5, (1,)), (0, 30, (0,))))
    assert report["reported_span_ns"] == report["covered_clock_ns"] == 30
    assert report["sum_batch_duration_ns"] == 45
    assert report["overlap_excess_ns"] == 15
    assert report["uncovered_span_ns"] == 0
    with pytest.raises(ValueError, match="overlap"):
        _batch_accounting(_evidence((0, 30, (0,)), (5, 10, (0,))))


def test_empty_normalized_batches_and_zero_timestamps_retain_clock_scope():
    empty = _batch_accounting(_evidence())
    assert empty["reported_span_ns"] == empty["covered_clock_ns"] == empty["uncovered_span_ns"] == 0
    zero = _batch_accounting(_evidence((5, 0, (0,)), (15, 0, (1,))))
    assert zero["reported_span_ns"] == zero["uncovered_span_ns"] == 10
    assert zero["sum_batch_duration_ns"] == zero["overlap_excess_ns"] == 0
    custom = _batch_accounting(_evidence((5, 10, ()), (30, 0, ())))
    assert custom["reported_span_ns"] == 25
    assert custom["covered_clock_ns"] == 10
    assert custom["uncovered_span_ns"] == 15
    assert custom["groups"][1]["operation_counts"] == {}


def test_deleting_empty_moments_is_not_a_change_of_batch_membership_or_order():
    plan = _plan()
    reference = replace(plan, moments=((), *plan.moments, ()))
    report = _placement_changes(reference, plan)
    assert report["matched_operation_count"] == 7
    assert report["moment_index_changed_count"] == 7
    assert report["listing_order_changed_count"] == report["moment_membership_changed_count"] == 0


def test_repeated_gate_identities_and_relabel_boundaries_survive_repacking():
    base = _plan()
    plan = replace(base, moments=(*base.moments[:-1], (Op("RELABEL", (0, 1)),), base.moments[-1]))
    candidate = plan.rescheduled(SchedulePolicy.ASAP_UNITARIES)
    changes = _placement_changes(plan, candidate)
    assert changes["matched_operation_count"] == 8
    assert changes["moment_membership_changed_count"] == 2  # only the first X on each role joins a new group
    first_x = next(r for r in changes["changes"] if r["source_operation"] == "q[0]/op[1]")
    assert first_x["added_moment_peers"] == ["q[1]/op[1]"]
    assert first_x["removed_moment_peers"] == []
    assert len({r["source_operation"] for r in changes["changes"]}) == len(changes["changes"])
    assert next(r for r in changes["changes"] if r["gate"] == "RELABEL")["moment_membership_changed"] is False


def test_diagnostics_reject_changed_order_on_shared_nonanchor_wire():
    plan = ScheduledCircuit(3, (tuple(Op("R", (q,)) for q in range(3)),
                               (Op("CX", (0, 2)),), (Op("CZ", (1, 2)),),
                               tuple(Op("M", (q,)) for q in range(3))), (), ())
    changed = replace(plan, moments=(plan.moments[0], plan.moments[2], plan.moments[1], plan.moments[3]))
    with pytest.raises(ValueError, match="same source rescheduling contract"):
        diagnose_schedule_pair(plan, changed, _evidence(), _evidence())


def test_explanation_ranks_absolute_gaps_and_keeps_signs():
    plan = _plan()
    diagnostics = diagnose_schedule_pair(plan, plan.rescheduled(SchedulePolicy.ASAP_UNITARIES),
                                         _evidence((0, 10, (0,))), _evidence((0, 20, (0,))))
    report = {"diagnostics": diagnostics, "measurement_windows": [
        {"measurement_index": 0, "measurement_name": "small", "role_names": ["data"], "gap_ns": {"delta": 3}},
        {"measurement_index": 1, "measurement_name": "large", "role_names": ["data", "ancilla"],
         "gap_ns": {"delta": -9}},
    ]}
    comparison = GuppyScheduleComparison(None, None, json.dumps(report))
    text = comparison.explain(top_windows=1)
    assert "large (data -> ancilla): -9 ns" in text
    assert "small" not in text
    assert "+10 ns batch-duration sum" in text
    assert "causes" in text
    assert "q[0]/op[1]: X q[0]" in text
    assert "First source" not in comparison.explain(top_operations=0)
    assert "Largest" not in comparison.explain(top_windows=0)
    for invalid in (-1, True, 1.5):
        with pytest.raises(ValueError, match="nonnegative integer"):
            comparison.explain(top_windows=invalid)
        with pytest.raises(ValueError, match="nonnegative integer"):
            comparison.explain(top_operations=invalid)
