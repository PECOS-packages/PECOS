# Copyright 2026 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0

from collections.abc import Callable

import pytest
from pecos.quantum import (
    DistanceResult,
    LogicalOperatorInfo,
    PauliString,
    StabilizerCode,
    StabilizerCodeSpec,
)


def _five_qubit_spec() -> StabilizerCodeSpec:
    return StabilizerCodeSpec(
        5,
        [
            PauliString.from_dense_str("XZZXI"),
            PauliString.from_dense_str("IXZZX"),
            PauliString.from_dense_str("XIXZZ"),
            PauliString.from_dense_str("ZXIXZ"),
        ],
        [PauliString.from_dense_str("ZZZZZ")],
        [PauliString.from_dense_str("XXXXX")],
    )


def _repetition_spec() -> StabilizerCodeSpec:
    return StabilizerCodeSpec(
        3,
        [
            PauliString.from_dense_str("ZZI"),
            PauliString.from_dense_str("IZZ"),
        ],
        [PauliString.from_dense_str("ZZZ")],
        [PauliString.from_dense_str("XXX")],
    )


def test_five_qubit_hand_built_spec_finds_genuine_weight_three_logical() -> None:
    spec = _five_qubit_spec()
    spec.verify()

    assert spec.num_qubits == 5
    assert spec.num_logical_qubits == 1
    assert len(spec.stabilizers) == 4
    assert spec.logical_zs == [PauliString.from_dense_str("ZZZZZ")]
    assert spec.logical_xs == [PauliString.from_dense_str("XXXXX")]

    result = spec.distance()

    assert isinstance(result, DistanceResult)
    assert result.distance == 3
    assert result.min_weight_operator.weight() == 3
    assert StabilizerCode.five_qubit().syndrome(result.min_weight_operator) == [False] * 4


def test_steane_css_and_general_searches_both_find_distance_three() -> None:
    spec = StabilizerCodeSpec.from_stabilizer_code(StabilizerCode.steane())

    general = spec.distance()
    css = spec.distance(css=True)

    assert general is not None
    assert css is not None
    assert general.distance == css.distance == 3


def test_repetition_min_weight_logicals_expose_equivalence_information() -> None:
    spec = _repetition_spec()
    result = spec.distance()
    logicals = spec.min_weight_logicals()

    assert result is not None
    assert result.distance == 1
    assert logicals
    assert all(isinstance(info, LogicalOperatorInfo) for info in logicals)
    assert all(info.weight == info.operator.weight() == 1 for info in logicals)
    assert all(info.equivalent_logicals == [("Z", 0)] for info in logicals)
    assert {info.equivalence_string() for info in logicals} <= {"X0", "Z0"}


def test_from_stabilizer_code_steane_round_trip_finds_distance_three() -> None:
    code = StabilizerCode.steane()
    spec = StabilizerCodeSpec.from_stabilizer_code(code)

    spec.verify()
    result = spec.distance()

    assert spec.num_qubits == code.num_qubits()
    assert spec.num_logical_qubits == code.num_logical_qubits()
    assert result is not None
    assert result.distance == 3


def test_max_weight_below_true_distance_returns_no_results() -> None:
    spec = _five_qubit_spec()

    assert spec.distance(max_weight=2) is None
    assert spec.min_weight_logicals(max_weight=2) == []


@pytest.mark.parametrize(
    "constructor",
    [StabilizerCode.steane, StabilizerCode.five_qubit, StabilizerCode.shor],
)
def test_spec_distance_matches_stabilizer_code_oracle(
    constructor: Callable[[], StabilizerCode],
) -> None:
    code = constructor()
    result = StabilizerCodeSpec.from_stabilizer_code(code).distance()

    assert result is not None
    assert result.distance == code.distance()


def test_noncommuting_stabilizers_raise_python_exception_on_verify() -> None:
    spec = StabilizerCodeSpec(
        1,
        [PauliString.from_dense_str("X"), PauliString.from_dense_str("Z")],
        [],
        [],
    )

    with pytest.raises(ValueError, match="Stabilizer generators 0 and 1 anticommute"):
        spec.verify()


def test_constructor_errors_are_python_exceptions() -> None:
    with pytest.raises(ValueError, match="Number of logical X and Z operators must match"):
        StabilizerCodeSpec(
            1,
            [],
            [PauliString.from_dense_str("Z")],
            [],
        )


def test_quantum_namespace_exports_distance_search_types() -> None:
    import pecos.quantum as quantum

    assert quantum.StabilizerCodeSpec is StabilizerCodeSpec
    assert quantum.DistanceResult is DistanceResult
    assert quantum.LogicalOperatorInfo is LogicalOperatorInfo
    assert {"StabilizerCodeSpec", "DistanceResult", "LogicalOperatorInfo"} <= set(quantum.__all__)
