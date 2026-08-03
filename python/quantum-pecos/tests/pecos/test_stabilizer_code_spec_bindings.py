# Copyright 2026 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0

from collections.abc import Callable

import numpy as np
import pytest
from pecos.quantum import (
    DistanceResult,
    LogicalOperatorInfo,
    ParityCheckMatrix,
    PauliString,
    StabilizerCode,
    StabilizerCodeSpec,
    SymplecticMatrix,
    Zs,
)

_HAMMING_H = [
    [1, 0, 1, 0, 1, 0, 1],
    [0, 1, 1, 0, 0, 1, 1],
    [0, 0, 0, 1, 1, 1, 1],
]


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
    assert "distance=" in repr(result)
    assert str(result.min_weight_operator) in repr(result)
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
    assert all(str(info.operator) in repr(info) for info in logicals)
    assert all(info.equivalence_string() in repr(info) for info in logicals)


def test_verbose_distance_searches_write_progress_to_stderr(
    capfd: pytest.CaptureFixture[str],
) -> None:
    spec = _repetition_spec()

    assert spec.distance(verbose=True) is not None
    assert "Checking weight" in capfd.readouterr().err

    assert spec.min_weight_logicals(verbose=True)
    assert "Checking weight" in capfd.readouterr().err


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
    assert spec.shortest_logicals(delta=1, max_weight=2) == []


def test_five_qubit_shortest_logicals_include_requested_weight_range() -> None:
    spec = _five_qubit_spec()
    minimum = spec.min_weight_logicals()
    delta_zero = spec.shortest_logicals()
    delta_one = spec.shortest_logicals(delta=1)
    delta_two = spec.shortest_logicals(delta=2)

    minimum_operators = [info.operator for info in minimum]
    delta_zero_operators = [info.operator for info in delta_zero]
    delta_one_operators = [info.operator for info in delta_one]
    delta_two_operators = [info.operator for info in delta_two]

    assert len(minimum_operators) == 30
    assert delta_zero_operators == minimum_operators
    assert delta_one_operators == minimum_operators
    assert delta_two_operators[: len(minimum_operators)] == minimum_operators
    assert sum(info.weight == 5 for info in delta_two) == 18
    assert {info.weight for info in delta_two} == {3, 5}
    assert all(StabilizerCode.five_qubit().syndrome(info.operator) == [False] * 4 for info in delta_two)


def test_repetition_shortest_logicals_exclude_non_logicals_in_range() -> None:
    logicals = _repetition_spec().shortest_logicals(delta=1)

    assert logicals
    assert {info.weight for info in logicals} == {1}


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


@pytest.mark.parametrize(
    "rows",
    [
        _HAMMING_H,
        np.asarray(_HAMMING_H, dtype=np.int64),
        np.asarray(_HAMMING_H, dtype=np.uint8),
    ],
    ids=["lists", "numpy-int64", "numpy-uint8"],
)
def test_parity_check_matrix_builds_steane_code_from_dense_inputs(rows: object) -> None:
    matrix = ParityCheckMatrix(rows)
    builder = StabilizerCodeSpec.builder(7)
    builder.checks_from_css(matrix, matrix)
    spec = builder.build_with_discovered_logicals()
    result = spec.distance()

    assert matrix.num_checks() == matrix.rank() == 3
    assert matrix.num_qubits() == 7
    assert matrix.rows() == _HAMMING_H
    assert len(matrix.to_x_stabilizers()) == len(matrix.to_z_stabilizers()) == 3
    assert repr(matrix) == "ParityCheckMatrix(shape=(3, 7))"
    assert result is not None
    assert result.distance == StabilizerCode.steane().distance() == 3


def test_symplectic_matrix_builds_five_qubit_code() -> None:
    rows = [
        [1, 0, 0, 1, 0, 0, 1, 1, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 1, 1, 0],
        [1, 0, 1, 0, 0, 0, 0, 0, 1, 1],
        [0, 1, 0, 1, 0, 1, 0, 0, 0, 1],
    ]
    matrix = SymplecticMatrix.from_dense(rows)
    builder = StabilizerCodeSpec.builder(5)
    builder.checks_from_symplectic(matrix)
    spec = builder.build_with_discovered_logicals()
    result = spec.distance()

    assert matrix.num_rows() == matrix.rank() == 4
    assert matrix.num_qubits() == 5
    assert matrix.rows() == rows
    assert matrix.x_block() == [row[:5] for row in rows]
    assert matrix.z_block() == [row[5:] for row in rows]
    assert matrix.to_positive_paulis() == spec.stabilizers
    assert repr(matrix) == "SymplecticMatrix(shape=(4, 5))"
    assert result is not None
    assert result.distance == 3


def test_code_matrix_entry_and_shape_errors_are_value_errors() -> None:
    with pytest.raises(ValueError, match=r"row 0.*value 2"):
        ParityCheckMatrix([[0, 2]])
    with pytest.raises(ValueError, match=r"row 0.*value -1"):
        SymplecticMatrix([[0, -1]])
    with pytest.raises(ValueError, match=r"row 1.*columns"):
        ParityCheckMatrix([[1, 0], [1]])
    with pytest.raises(ValueError, match="even column count"):
        SymplecticMatrix.from_dense([[1, 0, 1]])


def test_code_matrix_builder_validation_errors_preserve_diagnostics() -> None:
    width_mismatch = StabilizerCodeSpec.builder(2)
    with pytest.raises(ValueError, match=r"3 qubits, expected 2"):
        width_mismatch.checks_from_css(
            ParityCheckMatrix([[1, 0, 0]]),
            ParityCheckMatrix.zeros(0, 2),
        )

    nonorthogonal = StabilizerCodeSpec.builder(2)
    with pytest.raises(ValueError, match=r"X row 0 and Z row 0"):
        nonorthogonal.checks_from_css(
            ParityCheckMatrix([[1, 0]]),
            ParityCheckMatrix([[1, 0]]),
        )

    symplectic_width_mismatch = StabilizerCodeSpec.builder(2)
    with pytest.raises(ValueError, match=r"3 qubits, expected 2"):
        symplectic_width_mismatch.checks_from_symplectic(SymplecticMatrix.zeros(0, 3))


def test_spec_constructor_rejects_dependent_stabilizers() -> None:
    with pytest.raises(ValueError, match=r"rank 2, count 3"):
        StabilizerCodeSpec(
            3,
            [Zs([0, 1]), Zs([1, 2]), Zs([0, 2])],
            [],
            [],
        )


def test_zero_row_parity_check_matrix_preserves_width_for_x_only_code() -> None:
    x_checks = ParityCheckMatrix([[1, 1]])
    z_checks = ParityCheckMatrix.zeros(0, 2)
    builder = StabilizerCodeSpec.builder(2)
    builder.checks_from_css(x_checks, z_checks)
    spec = builder.build_with_discovered_logicals()

    assert z_checks.num_checks() == 0
    assert z_checks.num_qubits() == 2
    assert z_checks.rows() == []
    assert spec.num_logical_qubits == 1
    assert spec.stabilizers == x_checks.to_x_stabilizers()


def test_quantum_namespace_exports_distance_search_types() -> None:
    import pecos.quantum as quantum

    assert quantum.StabilizerCodeSpec is StabilizerCodeSpec
    assert quantum.DistanceResult is DistanceResult
    assert quantum.LogicalOperatorInfo is LogicalOperatorInfo
    assert quantum.ParityCheckMatrix is ParityCheckMatrix
    assert quantum.SymplecticMatrix is SymplecticMatrix
    assert {
        "StabilizerCodeSpec",
        "DistanceResult",
        "LogicalOperatorInfo",
        "ParityCheckMatrix",
        "SymplecticMatrix",
    } <= set(quantum.__all__)
