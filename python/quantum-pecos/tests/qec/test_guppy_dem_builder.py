# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Contract tests for the unified Guppy detector-error-model builder."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pecos
import pytest
from guppylang import guppy
from guppylang.std.builtins import result
from guppylang.std.quantum import cx, measure, qubit
from pecos.qec import (
    Detector,
    DetectorErrorModel,
    GuppyDemBuild,
    GuppyDemBuilder,
    Observable,
    build_dem_from_guppy,
    rec,
)
from pecos.qec.surface import NoiseParameters
from pecos_rslib.quantum import TickCircuit

if TYPE_CHECKING:
    from collections.abc import Callable


@guppy
def _tagged_two_qubit_program() -> None:
    q0 = qubit()
    q1 = qubit()
    cx(q0, q1)
    result("m0", measure(q0))
    result("m1", measure(q1))


_DETECTORS_JSON = '[{"id":0,"records":[-2]}]'
_OBSERVABLES_JSON = '[{"id":0,"records":[-1]}]'


def test_builder_matches_both_wrappers_with_noise_and_inserted_idles() -> None:
    noise = NoiseParameters(p1=0.0, p2=0.01, p_meas=0.02, p_prep=0.0, p_idle_z_linear_rate=0.03)
    via_json_builder = (
        DetectorErrorModel.builder()
        .with_program(_tagged_two_qubit_program)
        .with_qubits(2)
        .with_detectors_json(_DETECTORS_JSON)
        .with_observables_json(_OBSERVABLES_JSON)
        .with_noise(noise)
        .with_idle_after_2q(1.0)
        .build()
    )
    via_from_guppy = DetectorErrorModel.from_guppy(
        _tagged_two_qubit_program,
        num_qubits=2,
        detectors_json=_DETECTORS_JSON,
        observables_json=_OBSERVABLES_JSON,
        noise=noise,
        idle_after_2q_duration=1.0,
    )
    via_typed_builder = (
        DetectorErrorModel.builder()
        .with_program(_tagged_two_qubit_program)
        .with_qubits(2)
        .with_detectors([Detector(rec[-2])])
        .with_observables([Observable(rec[-1])])
        .with_noise(noise)
        .with_idle_after_2q(1.0)
        .build()
    )
    via_typed_wrapper = build_dem_from_guppy(
        _tagged_two_qubit_program,
        num_qubits=2,
        detectors=[Detector(rec[-2])],
        observables=[Observable(rec[-1])],
        noise=noise,
        idle_after_2q_duration=1.0,
    )

    expected = via_from_guppy.to_string()
    assert isinstance(via_json_builder, GuppyDemBuild)
    assert isinstance(DetectorErrorModel.builder(), GuppyDemBuilder)
    assert via_json_builder.dem.to_string() == expected
    assert via_typed_builder.dem.to_string() == expected
    assert via_typed_wrapper.dem.to_string() == expected


def test_builder_matches_both_wrappers_with_result_tags() -> None:
    detectors_json = '[{"id":0,"result_tags":["m0"]}]'
    observables_json = '[{"id":0,"result_tags":["m1"]}]'
    noise = NoiseParameters(p1=0.0, p2=0.0, p_meas=0.1, p_prep=0.0)
    via_json_builder = (
        DetectorErrorModel.builder()
        .with_program(_tagged_two_qubit_program)
        .with_qubits(2)
        .with_detectors_json(detectors_json)
        .with_observables_json(observables_json)
        .with_noise(noise)
        .build()
    )
    via_from_guppy = DetectorErrorModel.from_guppy(
        _tagged_two_qubit_program,
        num_qubits=2,
        detectors_json=detectors_json,
        observables_json=observables_json,
        noise=noise,
    )
    via_typed_builder = (
        DetectorErrorModel.builder()
        .with_program(_tagged_two_qubit_program)
        .with_qubits(2)
        .with_detectors([Detector("m0")])
        .with_observables([Observable("m1")])
        .with_noise(noise)
        .build()
    )
    via_typed_wrapper = build_dem_from_guppy(
        _tagged_two_qubit_program,
        num_qubits=2,
        detectors=[Detector("m0")],
        observables=[Observable("m1")],
        noise=noise,
    )

    expected = via_from_guppy.to_string()
    assert via_json_builder.dem.to_string() == expected
    assert via_typed_builder.dem.to_string() == expected
    assert via_typed_wrapper.dem.to_string() == expected


@pytest.mark.parametrize(
    ("configure", "missing"),
    [
        (lambda builder: builder.with_qubits(2).with_detectors_json(_DETECTORS_JSON), "with_program"),
        (
            lambda builder: builder.with_program(_tagged_two_qubit_program).with_detectors_json(_DETECTORS_JSON),
            "with_qubits",
        ),
        (lambda builder: builder.with_program(_tagged_two_qubit_program).with_qubits(2), "with_detectors"),
    ],
)
def test_builder_reports_missing_required_setters(
    configure: Callable[[GuppyDemBuilder], GuppyDemBuilder],
    missing: str,
) -> None:
    with pytest.raises(ValueError, match=missing):
        configure(DetectorErrorModel.builder()).build()


@pytest.mark.parametrize(
    ("setter", "value"),
    [
        ("with_program", _tagged_two_qubit_program),
        ("with_qubits", 2),
        ("with_detectors", [Detector(rec[-1])]),
        ("with_observables", [Observable(rec[-1])]),
        ("with_detectors_json", _DETECTORS_JSON),
        ("with_observables_json", _OBSERVABLES_JSON),
        ("with_num_measurements", 2),
        ("with_noise", NoiseParameters()),
        ("with_idle_after_2q", 1.0),
        ("with_strip_traced_idles", True),
        ("with_runtime", None),
        ("with_seed", 7),
        ("with_require_hosted_operation_order", True),
        ("with_max_hosted_tick_separation", 3),
    ],
)
def test_every_setter_rejects_a_second_call(setter: str, value: Any) -> None:
    builder = DetectorErrorModel.builder()
    getattr(builder, setter)(value)

    with pytest.raises(ValueError, match=setter):
        getattr(builder, setter)(value)


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("with_detectors", "with_detectors_json"),
        ("with_detectors_json", "with_detectors"),
        ("with_observables", "with_observables_json"),
        ("with_observables_json", "with_observables"),
    ],
)
def test_typed_and_json_spellings_for_one_role_conflict(first: str, second: str) -> None:
    values = {
        "with_detectors": [Detector(rec[-1])],
        "with_detectors_json": _DETECTORS_JSON,
        "with_observables": [Observable(rec[-1])],
        "with_observables_json": _OBSERVABLES_JSON,
    }
    builder = DetectorErrorModel.builder()
    getattr(builder, first)(values[first])

    with pytest.raises(ValueError, match="cannot be combined"):
        getattr(builder, second)(values[second])


@pytest.mark.parametrize("typed_setter", ["with_detectors", "with_observables"])
@pytest.mark.parametrize("typed_first", [False, True])
def test_num_measurements_conflicts_with_typed_specs(typed_setter: str, typed_first: bool) -> None:
    specs = [Detector(rec[-1])] if typed_setter == "with_detectors" else [Observable(rec[-1])]
    builder = DetectorErrorModel.builder()

    def combine_typed_specs_and_measurement_count() -> None:
        if typed_first:
            getattr(builder, typed_setter)(specs).with_num_measurements(1)
        else:
            getattr(builder.with_num_measurements(1), typed_setter)(specs)

    with pytest.raises(ValueError, match="with_num_measurements"):
        combine_typed_specs_and_measurement_count()


def test_builder_result_evaluates_simulation_result_columns() -> None:
    build = (
        DetectorErrorModel.builder()
        .with_program(_tagged_two_qubit_program)
        .with_qubits(2)
        .with_detectors([Detector("m0")])
        .with_observables([Observable("m1")])
        .with_noise(NoiseParameters(p_meas=0.1))
        .build()
    )
    columns = (
        pecos.sim(_tagged_two_qubit_program)
        .classical(pecos.selene_engine())
        .quantum(pecos.stabilizer())
        .qubits(2)
        .seed(7)
        .run(3)
        .to_shot_map()
        .to_dict()
    )

    assert build.evaluate_result_columns(columns) == [([0], 0)] * 3


def test_json_builder_audit_accepts_legacy_id_aliases() -> None:
    build = (
        DetectorErrorModel.builder()
        .with_program(_tagged_two_qubit_program)
        .with_qubits(2)
        .with_detectors_json('[{"detector_id":"D0","records":[-2]}]')
        .with_observables_json('[{"observable_id":"L0","records":[-1]}]')
        .with_noise(NoiseParameters(p_meas=0.1))
        .build()
    )

    assert build.evaluate_measurements({0: 1, 1: 1}) == ([1], 1)


def test_builder_setter_order_does_not_change_the_dem() -> None:
    noise = NoiseParameters(p1=0.01, p2=0.02, p_meas=0.03, p_prep=0.04)
    first = (
        DetectorErrorModel.builder()
        .with_program(_tagged_two_qubit_program)
        .with_qubits(2)
        .with_detectors([Detector(rec[-2])])
        .with_observables([Observable(rec[-1])])
        .with_noise(noise)
        .with_seed(11)
        .build()
    )
    second = (
        DetectorErrorModel.builder()
        .with_seed(11)
        .with_observables([Observable(rec[-1])])
        .with_noise(noise)
        .with_detectors([Detector(rec[-2])])
        .with_qubits(2)
        .with_program(_tagged_two_qubit_program)
        .build()
    )

    assert first.dem.to_string() == second.dem.to_string()


def test_builder_rejects_circuit_inputs_with_from_circuit_guidance() -> None:
    with pytest.raises(ValueError, match="from_circuit"):
        (
            DetectorErrorModel.builder()
            .with_program(TickCircuit())
            .with_qubits(1)
            .with_detectors_json(_DETECTORS_JSON)
            .build()
        )
