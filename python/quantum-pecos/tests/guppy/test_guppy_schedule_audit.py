"""Actual runtime clocks remain separate from authored moments and noise exposure."""

import sys

import pytest
from pecos import capture_qis_operation_trace
from pecos.qec.color import ColorExtraction, color_code_implementation, color_code_spec, color_memory
from pecos.qec.generic import CircuitOp as Op
from pecos.qec.generic import GuppyOrdering, ScheduledCircuit
from selene_simple_runtime_plugin import SimpleRuntimePlugin

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="Native aggregate helper ABI adapter not implemented")


def _plan():
    return ScheduledCircuit(2, ((Op("R", (0,)), Op("R", (1,))), (Op("X", (0,)),), (Op("X", (1,)),),
                                (Op("M", (0,)), Op("M", (1,)))), (), ())


def test_native_clock_arithmetic_and_source_segmentation_with_identity_recording():
    plan = _plan()
    runtime = SimpleRuntimePlugin(duration_ns_reset=5, duration_ns_rxy=10, duration_ns_measure=20)
    audit = plan.audit_guppy_schedule(runtime=runtime, seed=17)
    report = audit.to_dict()
    assert report["normalized_runtime"]["batch_span_ns"] == 70  # 2*5 + 2*10 + 2*20, serial runtime
    assert report["normalized_runtime"]["sites"] == [
        {"runtime_site": 0, "batch_assigned_operation_ns": 35, "between_operation_gap_ns": 15},
        {"runtime_site": 1, "batch_assigned_operation_ns": 35, "between_operation_gap_ns": 30},
    ]
    assert report["source_runtime"] == report["normalized_runtime"]
    assert report["observed_barrier_requests"] == 3
    assert [m["qis_quantum_counts"] for m in report["qis_moments"]] == [
        {"Reset": 2}, {"RXY": 1}, {"RXY": 1}, {"Measure": 2},
    ]
    assert report["trace_lowering_complete"]
    assert report["runtime_events_complete"]
    for key in ["physical_schedule_verified", "source_moment_timing_verified", "calibrated_durations_verified",
                "noise_equivalence_verified", "measurement_binding_verified"]:
        assert not report[key]
    # Recorder installation must preserve ordinary source/lowered execution.
    baseline = capture_qis_operation_trace(plan.guppy_program(ordering=GuppyOrdering.ALL_MOMENTS), 2,
                                           runtime=runtime, seed=17)
    captured = audit.evidence()["operation_trace"]
    for key in ["operations", "lowered_quantum_ops", "named_result_traces", "measurement_results"]:
        assert [c[key] for c in captured] == [c[key] for c in baseline]


@pytest.mark.parametrize("ordering", list(GuppyOrdering))
def test_runtime_with_zero_durations_does_not_imply_calibration(ordering):
    audit = _plan().audit_guppy_schedule(runtime=SimpleRuntimePlugin(), ordering=ordering)
    report = audit.to_dict()
    assert report["normalized_runtime"]["timing_status"] == "zero_durations"
    assert report["normalized_runtime"]["batch_span_ns"] == 0
    assert (report["qis_moments"] is not None) == (ordering == GuppyOrdering.ALL_MOMENTS)
    assert "not established" in audit.describe()


def test_empty_program_has_no_fabricated_clock_evidence():
    audit = ScheduledCircuit(1, ((), (), ()), (), ()).audit_guppy_schedule(runtime=SimpleRuntimePlugin())
    assert audit.to_dict()["normalized_runtime"]["timing_status"] == "no_batches"
    assert audit.to_dict()["observed_barrier_requests"] == 2


def test_paper_memory_preserves_all_boundaries_and_runtime_timing_records():
    plan = color_memory(color_code_implementation(color_code_spec("488", 3), ColorExtraction.BRICKWORK_CZ), 2, "Z")
    audit = plan.audit_guppy_schedule(runtime=SimpleRuntimePlugin(
        duration_ns_reset=5, duration_ns_rxy=10, duration_ns_rzz=25, duration_ns_measure=40,
    ))
    report = audit.to_dict()
    assert report["observed_barrier_requests"] == len(plan.moments) - 1
    assert len(report["qis_moments"]) == len(plan.moments)
    assert report["normalized_runtime"]["operation_counts"]["Measure"] == plan.num_measurements
    assert report["normalized_runtime"]["timing_status"] == "nonzero_reported_durations"
    assert report["source_runtime"] == report["normalized_runtime"]
    assert not report["noise_equivalence_verified"]
