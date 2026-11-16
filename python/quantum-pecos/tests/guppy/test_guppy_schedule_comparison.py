"""Actual compilation can reduce source depth without reducing a serial runtime span."""

import sys

import pytest
from pecos import sim, stabilizer
from pecos.qec.generic import CircuitOp as Op
from pecos.qec.generic import GuppyOrdering, ScheduledCircuit, SchedulePolicy
from selene_simple_runtime_plugin import SimpleRuntimePlugin

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="Native aggregate helper ABI adapter not implemented")


@pytest.mark.parametrize("policy", [SchedulePolicy.ASAP_UNITARIES, SchedulePolicy.COMPACT_UNITARIES])
def test_reference_and_candidate_share_profile_and_function_but_not_source_depth(policy):
    plan = ScheduledCircuit(2, ((Op("R", (0,)), Op("R", (1,))), (Op("X", (0,)),), (Op("X", (1,)),),
                                (Op("X", (0,)),), (Op("M", (0,)), Op("M", (1,)))), (), (),
                            role_names=("data", "ancilla"), measurement_names=("data.final", "ancilla.final"))
    comparison = plan.compare_guppy_schedules(runtime=SimpleRuntimePlugin(
        duration_ns_reset=5, duration_ns_rxy=10, duration_ns_measure=20,
    ), seed=17, candidate_policy=policy)
    report = comparison.to_dict()
    assert report["metrics"]["source_moments"] == {"reference": 5, "candidate": 4, "delta": -1}
    assert report["metrics"]["reported_span_ns"] == {"reference": 80, "candidate": 80, "delta": 0}
    assert [r["measurement_name"] for r in report["measurement_windows"]] == ["data.final", "ancilla.final"]
    assert report["runtime_profile"]["seed"] == 17
    assert len(report["runtime_profile"]["library_sha256"]) == 64
    diagnostics = report["diagnostics"]
    assert diagnostics["batch_accounting"]["deltas"]["reported_span_ns"] == 0
    assert diagnostics["batch_accounting"]["operation_kind_counts_equal"]
    assert diagnostics["source_placements"]["moment_membership_changed_count"] == 2
    assert "= +0 ns" in comparison.explain()
    assert report["candidate_policy"] == policy.value
    assert policy.value in comparison.describe()
    for variant in (plan, plan.rescheduled(policy)):
        program = variant.guppy_program(ordering=GuppyOrdering.ALL_MOMENTS, allocation_provenance=True)
        result = sim(program).qubits(2).quantum(stabilizer()).seed(17).run(4).to_dict()
        assert result == {"qec:m:0": [0] * 4, "qec:m:1": [1] * 4}
    assert comparison.reference.to_dict()["allocation_provenance"]
    assert comparison.candidate.to_dict()["allocation_provenance"]
    report["measurement_windows"].clear()
    assert len(comparison.to_dict()["measurement_windows"]) == 2
    assert "not wall-clock" in comparison.to_dict()["gap_scope"]


def test_empty_reference_has_no_windows_and_no_invented_speedup():
    report = ScheduledCircuit(1, ((), (), ()), (), ()).compare_guppy_schedules(
        runtime=SimpleRuntimePlugin(),
    ).to_dict()
    assert report["metrics"]["source_moments"]["delta"] == -3
    assert report["metrics"]["reported_span_ns"] == {"reference": 0, "candidate": 0, "delta": 0}
    assert report["measurement_windows"] == []
    assert report["timing_status"] == {"reference": "no_batches", "candidate": "no_batches"}
    assert not report["calibrated_durations_verified"]
    assert report["diagnostics"]["source_placements"]["matched_operation_count"] == 0
    assert report["diagnostics"]["batch_accounting"]["changes"] == []
