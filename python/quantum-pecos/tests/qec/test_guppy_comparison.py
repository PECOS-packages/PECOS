"""Independent contract and arithmetic checks for paired schedule observations."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
from pecos.qec.generic import CircuitOp as Op
from pecos.qec.generic import MeasurementParity, ScheduledCircuit, SchedulePolicy
from pecos.qec.generic.guppy_comparison import _paired_report, _rescheduling_contract


def _plan():
    return ScheduledCircuit(
        2,
        (
            (Op("R", (0,)), Op("R", (1,))),
            (Op("X", (0,)),),
            (Op("Z", (1,)),),
            (Op("H", (0,)),),
            (Op("M", (0,)), Op("M", (1,))),
        ),
        (),
        (),
        role_names=("data", "check"),
        measurement_names=("data.final", "check.final"),
    )


def test_region_contract_accepts_only_disjoint_repacking():
    plan = _plan()
    packed = plan.rescheduled(SchedulePolicy.ASAP_UNITARIES)
    assert len(packed.moments) == 4
    assert _rescheduling_contract(plan) == _rescheduling_contract(packed)
    # Same gates and supports, wrong on-wire order: H and X do not commute.
    wrong = replace(
        packed,
        moments=(packed.moments[0], (Op("H", (0,)), Op("Z", (1,))), (Op("X", (0,)),), packed.moments[-1]),
    )
    assert _rescheduling_contract(plan) != _rescheduling_contract(wrong)
    # Measurement ordinals are part of the contract even on disjoint wires.
    reversed_readout = replace(packed, moments=(*packed.moments[:-1], tuple(reversed(packed.moments[-1]))))
    assert _rescheduling_contract(plan) != _rescheduling_contract(reversed_readout)
    for changed in [
        replace(packed, detectors=(MeasurementParity((0,), "new"),)),
        replace(packed, role_names=("check", "data")),
        replace(packed, measurement_names=("check.final", "data.final")),
    ]:
        assert _rescheduling_contract(plan) != _rescheduling_contract(changed)


def test_contract_keeps_relabel_boundaries_and_mixed_boundary_moments():
    plan = ScheduledCircuit(
        2,
        ((Op("R", (0,)), Op("R", (1,))), (Op("RELABEL", (0, 1)),), (Op("M", (0,)), Op("X", (1,))), (Op("M", (1,)),)),
        (),
        (),
    )
    assert _rescheduling_contract(plan) == _rescheduling_contract(plan.rescheduled(SchedulePolicy.ASAP_UNITARIES))
    wrong = replace(
        plan,
        moments=(plan.moments[0], (Op("X", (0,)),), plan.moments[1], (Op("M", (0,)),), plan.moments[-1]),
    )
    assert _rescheduling_contract(plan) != _rescheduling_contract(wrong)


def _audit(windows, span):
    return SimpleNamespace(
        to_dict=lambda: {
            "normalized_runtime": {
                "batch_count": 7,
                "batch_span_ns": span,
                "timing_status": "nonzero_reported_durations",
            },
            "lifetime_windows": {"windows": windows, "normalized_window_gap_ns": sum(w["gap_ns"] for w in windows)},
        },
    )


def test_measurement_identity_aligns_windows_not_sites_or_emission_order():
    plan = _plan()
    old = [
        {
            "measurement_index": 0,
            "role_history": [{"role": 0}],
            "gap_ns": 30,
            "gate_interval_ns": 10,
            "storage_interval_ns": 40,
            "runtime_site": 1,
        },
        {
            "measurement_index": 1,
            "role_history": [{"role": 1}],
            "gap_ns": 5,
            "gate_interval_ns": 20,
            "storage_interval_ns": 25,
            "runtime_site": 0,
        },
    ]
    new = [
        {**old[1], "runtime_site": 1, "gap_ns": 15, "storage_interval_ns": 35},
        {**old[0], "runtime_site": 0, "gap_ns": 10, "storage_interval_ns": 20},
    ]
    report = _paired_report(plan, plan.rescheduled(SchedulePolicy.ASAP_UNITARIES), _audit(old, 80), _audit(new, 75))
    assert report["metrics"]["reported_span_ns"] == {"reference": 80, "candidate": 75, "delta": -5}
    assert report["metrics"]["sum_window_gaps_ns"] == {"reference": 35, "candidate": 25, "delta": -10}
    assert [r["gap_ns"]["delta"] for r in report["measurement_windows"]] == [-20, 10]
    assert [r["measurement_name"] for r in report["measurement_windows"]] == ["data.final", "check.final"]
    assert [r["role_names"] for r in report["measurement_windows"]] == [["data"], ["check"]]
    assert not report["noise_equivalence_verified"]


def test_nonreference_or_inequivalent_candidate_fails_before_execution(monkeypatch):
    with pytest.raises(ValueError, match="original REFERENCE"):
        _plan().rescheduled(SchedulePolicy.ASAP_UNITARIES).compare_guppy_schedules(runtime=None)
    monkeypatch.setattr(
        ScheduledCircuit,
        "rescheduled",
        lambda self, _policy: replace(self, detectors=(MeasurementParity((0,), "changed"),)),
    )
    with pytest.raises(ValueError, match="source rescheduling contract"):
        _plan().compare_guppy_schedules(runtime=None)


def test_profile_snapshot_and_library_mutation_are_checked(tmp_path, monkeypatch):
    library = tmp_path / "runtime.so"
    library.write_bytes(b"first")
    called = []
    arguments = ["--time", "10"]
    plugin = SimpleNamespace(library_file=library, get_init_args=lambda: arguments)
    plugin.native_gates = ["rxy", "rz", "rzz"]
    plugin.library_search_dirs = [tmp_path]
    plugin.clock_model = "shot_relative"

    def capture(_self, *, runtime, **_kwargs):
        from pecos._engine_builders import _configure_selene_runtime

        builder = SimpleNamespace(selene_runtime_plugin=lambda *args, **kwargs: (args, kwargs))
        if not called:
            direct = _configure_selene_runtime(builder, plugin)
            frozen = _configure_selene_runtime(builder, runtime)
            # Tuples and lists are both accepted gate sequences by the native builder.
            assert direct[0] == frozen[0]
            assert list(direct[1].pop("native_gates")) == list(frozen[1].pop("native_gates"))
            assert direct[1] == frozen[1]
        called.append(runtime.get_init_args())
        assert runtime.native_gates == ("rxy", "rz", "rzz")
        assert runtime.library_search_dirs == (str(tmp_path),)
        assert runtime.clock_model == "shot_relative"
        arguments[-1] = "999"  # Changes to the original object must not change subsequent runs.
        plugin.native_gates.append("rpp")
        runtime.get_init_args().append("ignored")
        if len(called) == 2:
            library.write_bytes(b"changed")

    monkeypatch.setattr(ScheduledCircuit, "audit_guppy_schedule", capture)
    with pytest.raises(ValueError, match="library changed"):
        _plan().compare_guppy_schedules(runtime=plugin)
    assert called == [["--time", "10"], ["--time", "10"]]


def test_known_plugin_gate_defaults_are_retained():
    from pecos._engine_builders import _runtime_native_gates

    class KnownPlugin:
        pass

    KnownPlugin.__module__ = "selene_soft_rz_runtime_plugin.plugin"
    assert _runtime_native_gates(KnownPlugin()) == ["rxy", "rz", "rzz"]


def test_invalid_runtime_arguments_fail_before_capture(tmp_path):
    library = tmp_path / "runtime.so"
    library.write_bytes(b"test")
    with pytest.raises(TypeError, match="init arguments"):
        _plan().compare_guppy_schedules(runtime=SimpleNamespace(library_file=library, get_init_args=lambda: [2]))
