# Copyright 2026 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0

import re

import pytest
from pecos.quantum import (
    PauliString,
    StabilizerCodeSpec,
    StabilizerCodeSpecBuilder,
    X,
    Xs,
    Y,
    Ys,
    Z,
    Zs,
    pauli_string,
)


def _original_checks() -> list[PauliString]:
    return [
        Xs([3, 4, 7, 8]),
        Xs([5, 6, 7, 9]),
        Zs([2, 4, 5, 7]),
        Zs([7, 8, 9]),
        Zs([0, 1]) * Y(2),
        pauli_string("X0 X2 Z3 Y4"),
        pauli_string("X1 X2 Z6 Y5"),
    ]


def _fixed_checks() -> list[PauliString]:
    checks = _original_checks()
    checks[4] = Zs([0, 1, 2])
    return checks


def _final_checks() -> list[PauliString]:
    return [
        Zs([2, 4, 5, 7]),
        Xs([3, 4, 7, 8]),
        Xs([5, 6, 7, 9]),
        pauli_string("X0 X2 Z3 Y4"),
        pauli_string("X1 X2 Z6 Y5"),
        Zs([0, 1, 2]),
        Xs([0, 1]),
        Zs([3, 8]),
        Zs([6, 9]),
    ]


def _builder_with_checks(checks: list[PauliString]) -> StabilizerCodeSpecBuilder:
    builder = StabilizerCodeSpec.builder(10)
    for check in checks:
        builder.check(check)
    return builder


def test_original_doc_checks_report_a_real_anticommuting_pair() -> None:
    checks = _original_checks()

    with pytest.raises(
        ValueError,
        match=r"Stabilizer generators \d+ and \d+ anticommute",
    ) as exc_info:
        _builder_with_checks(checks).build_verified()

    pair = re.search(r"generators (\d+) and (\d+) anticommute", str(exc_info.value))
    assert pair is not None
    first, second = (int(index) for index in pair.groups())
    assert checks[first].anticommutes_with(checks[second])

    with pytest.raises(ValueError, match="Stabilizers do not all commute with each other"):
        _builder_with_checks(checks).build_with_discovered_logicals()


def test_fixed_doc_checks_build_a_distance_two_code() -> None:
    spec = _builder_with_checks(_fixed_checks()).build_with_discovered_logicals()
    result = spec.distance()

    assert spec.num_logical_qubits == 3
    assert result is not None
    assert result.distance == 2
    assert result.min_weight_operator.weight() == 2


def test_final_doc_checks_build_a_distance_three_code() -> None:
    spec = _builder_with_checks(_final_checks()).build_with_discovered_logicals()
    result = spec.distance()

    assert spec.num_logical_qubits == 1
    assert len(spec.destabilizers) == 9
    assert result is not None
    assert result.distance == 3


def test_multi_qubit_pauli_helpers_match_single_qubit_composition() -> None:
    assert Xs([0, 2, 5]) == X(0) & X(2) & X(5)
    assert Ys((1, 3)) == Y(1) & Y(3)
    assert Zs([]) == PauliString.I()
    assert Zs(range(3)) == Z(0) & Z(1) & Z(2)


def test_builder_is_consumed_and_validates_logical_counts() -> None:
    builder = StabilizerCodeSpec.builder(1)
    spec = builder.build()

    assert spec.num_logical_qubits == 1
    with pytest.raises(RuntimeError, match="already been consumed"):
        builder.check(Z(0))

    builder = StabilizerCodeSpec.builder(1)
    builder.logical_z(Z(0))
    builder.logical_x(X(0))
    spec = builder.build_verified()

    assert spec.num_logical_qubits == 1
    assert spec.logical_zs == [Z(0)]
    assert spec.logical_xs == [X(0)]

    mismatched = StabilizerCodeSpec.builder(1)
    mismatched.logical_x(X(0))
    with pytest.raises(ValueError, match="Number of logical X and Z operators must match"):
        mismatched.build()


def test_string_summary_lists_final_code_generators() -> None:
    spec = _builder_with_checks(_final_checks()).build_with_discovered_logicals()
    summary = str(spec)

    assert "[[10, 1]]" in summary
    assert "Stabilizer generators:" in summary
    assert "Destabilizer generators:" in summary
    assert "Z1:" in summary
    assert "X1:" in summary
    assert repr(spec) == "StabilizerCodeSpec([[10, 1]])"
    assert all(operator.to_dense_str(10) in summary for operator in spec.stabilizers)
