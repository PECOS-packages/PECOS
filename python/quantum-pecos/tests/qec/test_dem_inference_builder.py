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
from pecos.qec import DemInferenceSettings, DetectorErrorModel
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
    ["with_detectors", "with_observables", "with_detectors_json", "with_observables_json", "with_num_measurements"],
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
