"""Compiled reset provenance binds source endpoints without claiming idle noise."""

import sys

import pytest
from pecos import sim, stabilizer
from pecos.qec.color import (
    ColorExtraction,
    color_code_implementation,
    color_code_spec,
    color_memory,
    lacroix_memory,
    periodic_488_code,
    periodic_488_implementation,
    periodic_488_memory,
)
from pecos.qec.dem import build_dem_from_guppy
from pecos.qec.generic import CircuitOp as Op
from pecos.qec.generic import GuppyOrdering, MeasurementParity, ScheduledCircuit, bind_runtime_lifetime_windows
from selene_simple_runtime_plugin import SimpleRuntimePlugin

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="Native aggregate helper ABI adapter not implemented")


def test_compiled_site_reuse_excludes_gap_between_lifetimes():
    plan = ScheduledCircuit(
        2,
        (
            (Op("R", (0,)), Op("R", (1,))),
            (Op("X", (0,)),),
            (Op("M", (0,)),),
            (Op("X", (1,)),),
            (Op("R", (0,)),),
            (Op("X", (0,)),),
            (Op("M", (0,)), Op("M", (1,))),
        ),
        (),
        (),
    )
    audit = plan.audit_guppy_schedule(
        runtime=SimpleRuntimePlugin(
            duration_ns_reset=5,
            duration_ns_rxy=10,
            duration_ns_measure=20,
        ),
        include_lifetimes=True,
    )
    report = audit.to_dict()
    windows = report["lifetime_windows"]
    assert windows == bind_runtime_lifetime_windows(plan, audit.evidence())
    assert len(windows["windows"]) == 3
    assert windows["normalized_window_gap_ns"] == 70
    assert windows["excluded_between_window_gap_ns"] == 10
    assert report["normalized_runtime"]["operation_counts"] == {"Reset": 3, "Rxy": 3, "Measure": 3}
    assert windows["endpoint_binding_verified"]
    assert not windows["logical_carrier_residency_verified"]
    assert not windows["live_idle_noise_model_verified"]
    assert "Tagged execution: 3" in audit.describe()


def test_tagged_compilation_keeps_random_bell_parity_and_fresh_preparation():
    plan = ScheduledCircuit(
        2,
        (
            (Op("R", (0,)), Op("R", (1,))),
            (Op("H", (0,)),),
            (Op("CX", (0, 1)),),
            (Op("RELABEL", (0, 1)),),
            (Op("M", (0,)),),
            (Op("M", (1,)),),
            (Op("R", (0,)),),
            (Op("M", (0,)),),
        ),
        (MeasurementParity((0, 1), "bell"), MeasurementParity((2,), "fresh")),
        (),
    )
    program = plan.guppy_program(ordering=GuppyOrdering.ALL_MOMENTS, allocation_provenance=True)
    columns = sim(program).qubits(2).quantum(stabilizer()).seed(15).run(64).to_dict()
    detectors, observables = plan.dem_spec()
    build = build_dem_from_guppy(
        program,
        num_qubits=2,
        detectors=detectors,
        observables=observables,
        p1=0.0,
        p2=0.0,
        p_meas=0.0,
        p_prep=0.0,
    )
    assert build.named_result_binding == "compiler_direct_scalar_complete"
    assert build.evaluate_result_columns(columns) == [([0, 0], 0)] * 64
    assert set(columns["qec:m:0"]) == {0, 1}
    for shot in range(64):
        assert plan.evaluate_named({tag: values[shot] for tag, values in columns.items()}) == ((False, False), ())
    windows = plan.audit_guppy_schedule(runtime=SimpleRuntimePlugin(), include_lifetimes=True).to_dict()[
        "lifetime_windows"
    ]
    assert [[r["role"] for r in w["role_history"]] for w in windows["windows"]] == [[1, 0], [0, 1], [0]]
    assert windows["timing_status"] == "zero_durations"
    assert all(w["storage_interval_ns"] == 0 for w in windows["windows"])
    assert not windows["calibrated_durations_verified"]


@pytest.mark.parametrize("family", ["488", "swap666", "lacroix", "periodic"])
def test_paper_memory_tagged_endpoints_cover_every_measurement(family):
    if family == "488":
        plan = color_memory(color_code_implementation(color_code_spec("488", 3), ColorExtraction.BRICKWORK_CZ), 2, "Z")
    elif family == "swap666":
        impl = color_code_implementation(color_code_spec("666", 3), ColorExtraction.SWAP_SUPERDENSE)
        plan = color_memory(impl, 3, "Z")
    elif family == "lacroix":
        plan = lacroix_memory("d3_west", 2, "Z")
    else:
        plan = periodic_488_memory(periodic_488_implementation(periodic_488_code(4), gate_set="native-cz"), 2, "Z")
    report = plan.audit_guppy_schedule(
        runtime=SimpleRuntimePlugin(
            duration_ns_reset=5,
            duration_ns_rxy=10,
            duration_ns_rzz=25,
            duration_ns_measure=40,
        ),
        include_lifetimes=True,
    ).to_dict()
    windows = report["lifetime_windows"]
    assert len(windows["windows"]) == plan.num_measurements
    assert [w["measurement_index"] for w in windows["windows"]] == list(range(plan.num_measurements))
    assert all(w["gap_ns"] + w["gate_interval_ns"] == w["storage_interval_ns"] for w in windows["windows"])
    assert windows["normalized_window_gap_ns"] > 0
    assert not windows["noise_equivalence_verified"]
    assert report["source_citations"] == list(plan.citations)


@pytest.mark.parametrize("num_moments", [0, 1, 3])
def test_empty_tagged_program_has_no_windows(num_moments):
    report = (
        ScheduledCircuit(1, ((),) * num_moments, (), ())
        .audit_guppy_schedule(
            runtime=SimpleRuntimePlugin(),
            include_lifetimes=True,
        )
        .to_dict()["lifetime_windows"]
    )
    assert report["windows"] == []
    assert report["timing_status"] == "no_batches"
    assert report["normalized_window_gap_ns"] == report["excluded_between_window_gap_ns"] == 0


@pytest.mark.parametrize(("ordering", "flag"), [(GuppyOrdering.FUNCTIONAL, True), (GuppyOrdering.ALL_MOMENTS, 1)])
def test_invalid_lifetime_options_fail_before_compilation(monkeypatch, ordering, flag):
    def unexpected_compile(*_args, **_kwargs):
        pytest.fail("Invalid options reached compilation")

    monkeypatch.setattr(ScheduledCircuit, "guppy_program", unexpected_compile)
    with pytest.raises(ValueError, match="include_lifetimes"):
        ScheduledCircuit(1, (), (), ()).audit_guppy_schedule(
            runtime=SimpleRuntimePlugin(),
            ordering=ordering,
            include_lifetimes=flag,
        )
