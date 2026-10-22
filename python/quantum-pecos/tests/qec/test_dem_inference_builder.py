# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Integration of inferred annotations with the shared DEM builder."""

from __future__ import annotations

import pecos
import pytest
from guppylang import guppy
from guppylang.std.builtins import array, output
from guppylang.std.quantum import cx, measure, qubit
from pecos import NoiseParameters
from pecos.qec import DemInferenceSettings, Detector, DetectorErrorModel, Observable, rec


@guppy
def _inferred_program() -> None:
    q0, q1, q2 = qubit(), qubit(), qubit()
    cx(q0, q1)
    m0 = measure(q0).read()
    m1 = measure(q1).read()
    m2 = measure(q2).read()
    output("DETECTOR", m2 ^ m0)
    output("raw measurements", array(m2, m0, m1))
    output("obs", m1)


def test_settings_are_reusable_immutable_values_with_rust_defaults() -> None:
    defaults = DemInferenceSettings()
    changed = defaults.with_probe_shots(80).with_observable_tags(["logical"])
    assert defaults.probe_shots is None
    assert defaults.observable_tags == ["obs"]
    assert changed.probe_shots == 80
    assert changed.observable_tags == ["logical"]
    assert defaults.resolved_probe_shots(10) == 75
    with pytest.raises(AttributeError):
        defaults.raw_tag = "other"
    with pytest.raises(ValueError, match="distinct"):
        DetectorErrorModel.builder().with_inference(defaults.with_raw_tag("obs"))
    with pytest.raises(ValueError, match="at least 1"):
        DetectorErrorModel.builder().with_inference(defaults.with_validation_rows(0))


@pytest.mark.parametrize(
    "setter",
    ["with_num_measurements"],
)
@pytest.mark.parametrize("inference_first", [False, True])
def test_inference_and_explicit_metadata_conflict_in_both_orders(setter: str, *, inference_first: bool) -> None:
    builder = DetectorErrorModel.builder()
    argument = 2 if setter == "with_num_measurements" else ("[]" if setter.endswith("json") else [])
    if inference_first:
        builder.with_inference(DemInferenceSettings())
        with pytest.raises(ValueError, match="cannot be combined"):
            getattr(builder, setter)(argument)
    else:
        getattr(builder, setter)(argument)
        with pytest.raises(ValueError, match="cannot be combined"):
            builder.with_inference(DemInferenceSettings())


def test_inference_setter_rejects_wrong_types_and_repeated_calls() -> None:
    builder = DetectorErrorModel.builder()
    with pytest.raises(TypeError, match="DemInferenceSettings"):
        builder.with_inference(True)
    builder.with_inference(DemInferenceSettings())
    with pytest.raises(ValueError, match="only be called once"):
        builder.with_inference(DemInferenceSettings())


def test_inferred_build_shares_noise_idle_and_evaluator() -> None:
    noise = NoiseParameters().with_p1(0.001).with_p2(0.01).with_p_meas(0.02).with_p_prep(0.001)
    noise = noise.with_p_idle_linear(0.03, {"Z": 1.0})
    build = (
        DetectorErrorModel.builder()
        .with_program(_inferred_program)
        .with_qubits(3)
        .with_inference(DemInferenceSettings().with_validation_rows(8))
        .with_seed(7)
        .with_noise(noise)
        .with_idle_after_2q(1.0)
        .build()
    )
    explicit = (
        DetectorErrorModel.builder()
        .with_program(_inferred_program)
        .with_qubits(3)
        .with_detectors_json(build.detectors_json)
        .with_observables_json(build.observables_json)
        .with_noise(noise)
        .with_idle_after_2q(1.0)
        .build()
    )
    assert build.dem.to_string() == explicit.dem.to_string()
    assert build.circuit.gate_counts_by_type()["Idle"] == 2
    assert build.inference is not None
    assert build.inference.raw_measurement_ids == (2, 0, 1)
    assert build.audit["inference"]["schedule_validation"] == "empirical_trace_comparison"
    assert build.audit["soundness_scope"] == "empirical_static_schedule"
    columns = {"raw measurements": [[0, 1, 1], [1, 1, 0], [1, 0, 1]]}
    assert build.evaluate_result_columns(columns) == [([1], 1), ([0], 0), ([1], 1)]
    assert build.evaluate_runtime_record([1, 0, 0]) == ([1], 0)
    with pytest.raises(ValueError, match="expects"):
        build.evaluate_result_columns({"raw measurements": [[1]]})
    assert explicit.inference is None
    results = (
        pecos.sim(_inferred_program)
        .classical(pecos.selene_engine())
        .quantum(pecos.stabilizer())
        .qubits(3)
        .seed(5)
        .run(8)
    )
    actual_columns = results.to_shot_map().to_dict()
    assert build.evaluate_result_columns(actual_columns) == [([0], 0)] * 8


@guppy
def _selected_program() -> None:
    q0, q1, q2 = qubit(), qubit(), qubit()
    cx(q0, q1)
    m0 = measure(q0).read()
    m1 = measure(q1).read()
    m2 = measure(q2).read()
    output("parities", array(m0 ^ m2, m0 & m1))
    output("raw measurements", array(m2, m0, m1))
    output("logical", m1)
    output("m1", m1)


def test_mixed_definitions_select_only_requested_elements(monkeypatch: pytest.MonkeyPatch) -> None:
    from pecos.qec import dem as dem_module

    def unexpected_analysis(*_args: object, **_kwargs: object) -> None:
        pytest.fail("record-only mixed definitions must not require compiler tag analysis")

    monkeypatch.setattr(dem_module, "_preflight_guppy_static_schedule", unexpected_analysis)
    build = (
        DetectorErrorModel.builder()
        .with_program(_selected_program)
        .with_qubits(3)
        .with_detectors([Detector(rec[-2]), Detector.inferred("parities", element=0)])
        .with_observables([Observable.inferred("logical", id=70), Observable(rec[-3], id=2)])
        .build()
    )
    assert build.inference.detector_labels == (("parities", 0),)
    assert build.inference.observable_labels == (("logical", 0),)
    assert build.inference.detectors_json == build.detectors_json
    assert build.inference.observables_json == build.observables_json
    assert build.inference.build_dem().to_string() == build.dem.to_string()
    assert build.dem is build.dem
    for circuit in (build.circuit, build.inference.circuit):
        assert circuit.get_meta("num_measurements") == "3"
        assert circuit.get_meta("guppy_dem_annotation_method") == "coin_toss_affine_inference_v2"
    assert build.evaluate_runtime_record([1, 1, 0]) == ([1, 1], (1 << 70) | (1 << 2))
    assert [item["source"] for item in build.audit["definition_origins"]] == [
        "explicit",
        "inferred",
        "explicit",
        "inferred",
    ]
    assert build.audit["definition_origins"][-1]["tag"] == "logical"
    with pytest.raises(ValueError, match="not affine"):
        (
            DetectorErrorModel.builder()
            .with_program(_selected_program)
            .with_qubits(3)
            .with_detectors([Detector.inferred("parities", element=1)])
            .with_observables([])
            .build()
        )


@pytest.mark.parametrize("settings_first", [False, True])
def test_explicit_definitions_skip_inference_even_with_settings(*, settings_first: bool) -> None:
    builder = DetectorErrorModel.builder().with_program(_selected_program).with_qubits(3)
    if settings_first:
        builder.with_inference(DemInferenceSettings().with_raw_tag("missing raw tag"))
    builder.with_detectors([Detector(rec[-2])]).with_observables([])
    if not settings_first:
        builder.with_inference(DemInferenceSettings().with_raw_tag("missing raw tag"))
    build = builder.build()
    assert build.inference is None
    assert build.evaluate_runtime_record([0, 1, 0]) == ([1], 0)


@pytest.mark.parametrize("explicit_role", ["detectors", "observables"])
def test_role_mixing_does_not_require_unused_tags(explicit_role: str) -> None:
    settings = DemInferenceSettings().with_detector_tag("parities").with_observable_tags(["logical"])
    builder = DetectorErrorModel.builder().with_program(_selected_program).with_qubits(3)
    if explicit_role == "detectors":
        # The unused detector tag does not exist. Only logical is inferred.
        settings = settings.with_detector_tag("absent detector")
        builder.with_detectors([Detector(rec[-3], rec[-1])])
    else:
        # Select just the affine element; there is no default observable tag.
        settings = settings.with_observable_tags(["absent observable"])
        builder.with_detectors([Detector.inferred("parities")]).with_observables([])
    build = builder.with_inference(settings).build()
    assert build.inference is not None
    assert build.evaluate_runtime_record([1, 1, 0]) == ([1], 1 if explicit_role == "detectors" else 0)


def test_mixed_named_explicit_binding_keeps_compiler_provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    from unittest.mock import Mock

    from pecos.qec import dem as dem_module
    from pecos_rslib import hugr_lowering

    compiler = Mock(wraps=hugr_lowering.compile_hugr_to_qis)
    monkeypatch.setattr(hugr_lowering, "compile_hugr_to_qis", compiler)
    build = (
        DetectorErrorModel.builder()
        .with_program(_selected_program)
        .with_qubits(3)
        .with_detectors([Detector.inferred("parities")])
        .with_observables([Observable("m1")])
        .build()
    )
    assert compiler.call_count == 1
    assert build.inference.detector_labels == (("parities", 0),)
    assert build.evaluate_runtime_record([1, 1, 0]) == ([1], 1)
    assert build.evaluate_result_columns({"raw measurements": [[0, 1, 1]], "m1": [[1]]}) == [([1], 1)]
    with pytest.raises(ValueError, match="direct scalar"):
        (
            DetectorErrorModel.builder()
            .with_program(_selected_program)
            .with_qubits(3)
            .with_detectors([Detector.inferred("logical")])
            .with_observables([Observable("parities")])
            .build()
        )


def test_invalid_selectors_and_duplicate_ids_fail_before_execution() -> None:
    from pecos_rslib import Qis

    with pytest.raises(ValueError, match="non-negative"):
        Detector.inferred("tag", element=-1)
    with pytest.raises(ValueError, match="non-empty"):
        Observable.inferred("")
    with pytest.raises(ValueError, match="unique"):
        (
            DetectorErrorModel.builder()
            .with_program(Qis("not executable"))
            .with_qubits(3)
            .with_detectors([Detector.inferred("parities", id=0), Detector(rec[-1], id=0)])
            .build()
        )


def test_selected_element_out_of_range() -> None:
    with pytest.raises(ValueError, match="out of range"):
        (
            DetectorErrorModel.builder()
            .with_program(_selected_program)
            .with_qubits(3)
            .with_detectors([Detector.inferred("parities", element=9)])
            .with_observables([])
            .build()
        )


def test_observable_only_inference_does_not_require_detector_definitions() -> None:
    build = (
        DetectorErrorModel.builder()
        .with_program(_selected_program)
        .with_qubits(3)
        .with_observables([Observable.inferred("logical", id=2)])
        .build()
    )
    assert build.dem.num_detectors == 0
    assert build.inference.detector_supports == ()
    assert build.evaluate_runtime_record([0, 1, 0]) == ([], 4)
    assert build.audit["definition_origins"] == [
        {"role": "observable", "id": 2, "source": "inferred", "tag": "logical", "element": 0},
    ]


@pytest.mark.parametrize("key", ["coords", "inferred_from_result_tag", "result_element"])
@pytest.mark.parametrize("role", ["detector", "observable", "inferred_detector"])
def test_reserved_metadata_rejected_before_runtime(key: str, role: str) -> None:
    from pecos_rslib import Qis

    factory = {
        "detector": lambda: Detector(rec[-1], metadata={key: 1}),
        "observable": lambda: Observable(rec[-1], metadata={key: 1}),
        "inferred_detector": lambda: Detector.inferred("checks", metadata={key: 1}),
    }[role]
    spec = factory()
    builder = (
        DetectorErrorModel.builder()
        .with_program(Qis("not executable"))
        .with_qubits(3)
        .with_inference(DemInferenceSettings())
    )
    if role == "observable":
        builder.with_observables([spec])
    else:
        builder.with_detectors([spec])
    with pytest.raises(ValueError, match=key):
        builder.build()


def test_public_inference_method() -> None:
    rows = [[i & 1, (i >> 1) & 1] for i in range(8)]
    settings = DemInferenceSettings().with_validation_rows(2)
    assert settings.infer([3, 9], rows, [[a ^ b] for a, b in rows], [[b] for _, b in rows]) == ([[3, 9]], [[9]])


@guppy
def _raw_and_nonraw_arrays() -> None:
    q0, q1, q2 = qubit(), qubit(), qubit()
    cx(q0, q1)
    m0 = measure(q0).read()
    m1 = measure(q1).read()
    m2 = measure(q2).read()
    output("checks", m0 ^ m2)
    output("raw measurements", array(m2, m0, m1))
    output("other", array(m2, m0, m1))


def test_mixed_build_does_not_bind_nonraw_arrays() -> None:
    build = (
        DetectorErrorModel.builder()
        .with_program(_raw_and_nonraw_arrays)
        .with_qubits(3)
        .with_detectors([Detector(rec[-1]), Detector.inferred("checks")])
        .with_observables([])
        .build()
    )
    assert build.inference is not None
    assert build.evaluate_results({"raw measurements": [1, 0, 0]}) == ([1, 1], 0)
    with pytest.raises(ValueError, match="missing referenced"):
        build.evaluate_results({"other": [1, 0, 0]})


def test_explicit_probe_count_checks_solver_minimum() -> None:
    settings = DemInferenceSettings().with_validation_rows(2)
    with pytest.raises(ValueError, match="explicit probe_shots 5 for raw width 3 requires at least 6 rows"):
        settings.with_probe_shots(5).resolved_probe_shots(3)
    assert settings.with_probe_shots(6).resolved_probe_shots(3) == 6
    assert settings.resolved_probe_shots(3) == 38
