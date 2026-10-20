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
from pecos_rslib.qec import _correlate_measurement_ids, _infer_affine_columns


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
    assert build.audit["inference"]["schedule_validation"] == "probe_compared"
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


def test_rust_solver_rejects_nonlinearity_and_correlates_permuted_ids() -> None:
    rows = [[i & 1, (i >> 1) & 1] for i in range(8)]
    assert _infer_affine_columns(rows, [[a ^ b] for a, b in rows], 2) == [(0, [0, 1])]
    with pytest.raises(ValueError, match="not affine"):
        _infer_affine_columns(rows, [[a & b] for a, b in rows], 2)
    assert _correlate_measurement_ids([[b, a] for a, b in rows], rows, [3, 9]) == [9, 3]


def test_direct_ids_do_not_skip_schedule_comparison(monkeypatch: pytest.MonkeyPatch) -> None:
    from pecos import tracing
    from pecos.qec.guppy_output_dem import infer_guppy_dem_annotations

    original = tracing._capture_qis_operation_traces  # noqa: SLF001

    def changed_schedule(*args: object, **kwargs: object) -> list[dict]:
        trace = original(*args, **kwargs)
        if kwargs["shots"] == 1:
            for chunk in trace:
                for result in chunk.get("named_result_traces", []):
                    if result["name"] == "raw measurements":
                        result["result_ids"] = [0, 1, 2]
        if kwargs["shots"] > 1:
            for chunk in trace:
                if chunk["lowered_quantum_ops"]:
                    chunk["lowered_quantum_ops"][0]["qubits"] = [99]
                    break
        return trace

    monkeypatch.setattr(tracing, "_capture_qis_operation_traces", changed_schedule)
    with pytest.raises(ValueError, match="schedule changed"):
        infer_guppy_dem_annotations(_inferred_program, num_qubits=3)


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
    from pecos.qec import guppy_output_dem

    original = guppy_output_dem.infer_guppy_dem_annotations
    calls = []

    def infer(*args: object, **kwargs: object) -> object:
        calls.append(kwargs)
        return original(*args, **kwargs)

    def unexpected_analysis(*_args: object, **_kwargs: object) -> None:
        pytest.fail("record-only mixed definitions must not require compiler tag analysis")

    monkeypatch.setattr(guppy_output_dem, "infer_guppy_dem_annotations", infer)
    monkeypatch.setattr(dem_module, "_preflight_guppy_static_schedule", unexpected_analysis)
    build = (
        DetectorErrorModel.builder()
        .with_program(_selected_program)
        .with_qubits(3)
        .with_detectors([Detector(rec[-2]), Detector.inferred("parities", element=0)])
        .with_observables([Observable.inferred("logical", id=70), Observable(rec[-3], id=2)])
        .build()
    )
    assert len(calls) == 1
    assert calls[0]["detector_outputs"] == [("parities", 0)]
    assert calls[0]["observable_outputs"] == [("logical", 0)]
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
def test_explicit_definitions_skip_inference_even_with_settings(
    monkeypatch: pytest.MonkeyPatch,
    *,
    settings_first: bool,
) -> None:
    from pecos.qec import guppy_output_dem

    def unexpected_inference(*_args: object, **_kwargs: object) -> None:
        pytest.fail("explicit definitions must not invoke inference")

    monkeypatch.setattr(guppy_output_dem, "infer_guppy_dem_annotations", unexpected_inference)
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


def test_mixed_named_explicit_binding_keeps_compiler_provenance() -> None:
    build = (
        DetectorErrorModel.builder()
        .with_program(_selected_program)
        .with_qubits(3)
        .with_detectors([Detector.inferred("parities")])
        .with_observables([Observable("m1")])
        .build()
    )
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


def test_invalid_selectors_and_duplicate_ids_fail_before_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    from pecos.qec import guppy_output_dem

    def unexpected_inference(*_args: object, **_kwargs: object) -> None:
        pytest.fail("invalid definitions must fail before inference")

    monkeypatch.setattr(guppy_output_dem, "infer_guppy_dem_annotations", unexpected_inference)
    with pytest.raises(ValueError, match="non-negative"):
        Detector.inferred("tag", element=-1)
    with pytest.raises(ValueError, match="non-empty"):
        Observable.inferred("")
    with pytest.raises(ValueError, match="unique"):
        (
            DetectorErrorModel.builder()
            .with_program(_selected_program)
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


@pytest.mark.parametrize(
    ("corruption", "message"),
    [
        ("schedule", "schedule changed"),
        ("raw_binding", "binding changed"),
        ("output_width", "inconsistent shot widths"),
        ("terminal", "terminal trace_complete"),
        ("partial_lowering", "complete lowered gate stream"),
    ],
)
def test_every_fitted_shot_has_schedule_and_binding_checks(
    monkeypatch: pytest.MonkeyPatch,
    corruption: str,
    message: str,
) -> None:
    from pecos import tracing
    from pecos.qec.guppy_output_dem import infer_guppy_dem_annotations

    original = tracing._capture_qis_operation_traces  # noqa: SLF001

    def corrupt_fitting_shot(*args: object, **kwargs: object) -> list[dict]:
        trace = original(*args, **kwargs)
        # Leave the baseline and correlation shots untouched. The old path
        # fitted a separate untraced run, so it never checked these samples.
        if kwargs["shots"] <= 32:
            return trace
        last_id = max(chunk["shot_index"] for chunk in trace)
        last_shot = [chunk for chunk in trace if chunk["shot_index"] == last_id]
        if corruption == "terminal":
            return [chunk for chunk in trace if chunk is not last_shot[-1]]
        if corruption in {"schedule", "partial_lowering"}:
            chunk = next(chunk for chunk in last_shot if chunk["lowered_quantum_ops"])
            if corruption == "schedule":
                chunk["lowered_quantum_ops"][0]["qubits"] = [99]
            else:
                chunk["lowered_quantum_ops_complete"] = False
        else:
            tag = "raw measurements" if corruption == "raw_binding" else "DETECTOR"
            result = next(
                result for chunk in last_shot for result in chunk["named_result_traces"] if result["name"] == tag
            )
            if corruption == "raw_binding":
                result["values"][0] = not result["values"][0]
            else:
                result["values"].append(False)
        return trace

    monkeypatch.setattr(tracing, "_capture_qis_operation_traces", corrupt_fitting_shot)
    with pytest.raises(ValueError, match=message):
        infer_guppy_dem_annotations(_inferred_program, num_qubits=3, probe_shots=64, provenance_shots=32)
