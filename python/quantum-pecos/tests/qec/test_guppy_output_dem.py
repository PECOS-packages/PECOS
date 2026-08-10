# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Tests for parity annotations learned from Guppy result outputs."""

from __future__ import annotations

import json

import pytest
from guppylang import guppy
from guppylang.std.builtins import array, result
from guppylang.std.quantum import measure, qubit
from pecos.qec import infer_guppy_dem_annotations


@guppy
def _measure_three_into_array() -> array[bool, 3]:
    q0 = qubit()
    q1 = qubit()
    q2 = qubit()
    return array(measure(q0), measure(q1), measure(q2))


@guppy
def _computed_parity_outputs() -> None:
    measurements = _measure_three_into_array()
    m0 = measurements[0]
    m1 = measurements[1]
    m2 = measurements[2]
    result("DETECTOR", m0 ^ m1)
    result("DETECTOR", m1 ^ m2)
    result("raw measurements", measurements)
    result("obs", m0 ^ m2)


@guppy
def _raw_results_incomplete() -> None:
    q0 = qubit()
    q1 = qubit()
    m0 = measure(q0)
    result("raw measurements", m0)
    m1 = measure(q1)
    result("DETECTOR", m0 ^ m1)
    result("obs", m0)


@guppy
def _reordered_raw_array() -> None:
    m0 = measure(qubit())
    m1 = measure(qubit())
    m2 = measure(qubit())
    result("DETECTOR", m2 ^ m0)
    result("raw measurements", array(m2, m0, m1))
    result("obs", m1)


def test_infers_computed_detector_and_observable_parities_and_builds_dem() -> None:
    inferred = infer_guppy_dem_annotations(
        _computed_parity_outputs,
        num_qubits=3,
        probe_shots=64,
        validation_rows=16,
        seed=7,
        require_raw_provenance=False,
    )

    assert inferred.raw_measurement_ids == (0, 1, 2)
    assert inferred.detector_supports == ((0, 1), (1, 2))
    assert inferred.observable_supports == ((0, 2),)
    assert inferred.observable_labels == (("obs", 0),)
    assert inferred.raw_binding == "assumed_canonical_result_order"
    assert json.loads(inferred.detectors_json) == [
        {"id": 0, "meas_ids": [0, 1], "inferred_from_result_tag": "DETECTOR"},
        {"id": 1, "meas_ids": [1, 2], "inferred_from_result_tag": "DETECTOR"},
    ]

    dem = inferred.build_dem(p1=0.0, p2=0.0, p_meas=0.1, p_prep=0.0)
    assert dem.num_detectors == 2
    assert dem.num_observables == 1
    assert "D0" in dem.to_string()


def test_computed_array_provenance_is_correlated_to_qis_result_ids() -> None:
    inferred = infer_guppy_dem_annotations(
        _computed_parity_outputs,
        num_qubits=3,
        probe_shots=32,
        provenance_shots=16,
        validation_rows=8,
        seed=7,
    )

    assert inferred.raw_measurement_ids == (0, 1, 2)
    assert inferred.raw_binding == "probe_correlated_result_ids"


def test_correlated_provenance_preserves_reordered_raw_identity() -> None:
    inferred = infer_guppy_dem_annotations(
        _reordered_raw_array,
        num_qubits=3,
        probe_shots=32,
        provenance_shots=16,
        validation_rows=8,
        seed=13,
    )

    assert inferred.raw_measurement_ids == (2, 0, 1)
    assert inferred.detector_supports == ((2, 0),)
    assert inferred.observable_supports == ((1,),)
    assert inferred.raw_binding == "probe_correlated_result_ids"


def test_raw_tag_must_cover_canonical_qis_measurement_order() -> None:
    with pytest.raises(ValueError, match="emits 1 values during provenance probing"):
        infer_guppy_dem_annotations(
            _raw_results_incomplete,
            num_qubits=2,
            probe_shots=32,
            validation_rows=8,
            seed=3,
        )
