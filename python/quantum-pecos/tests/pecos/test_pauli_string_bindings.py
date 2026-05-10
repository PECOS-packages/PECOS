# Copyright 2026 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0

import pytest
from pecos_rslib import Pauli, PauliString, X, Z


def test_pauli_string_from_str_accepts_dense_and_sparse_formats() -> None:
    expected = X(0) & X(1) & Z(3)

    assert PauliString.from_str("XXIZ") == expected
    assert PauliString.from_str("X0 X1 Z3") == expected
    assert PauliString.from_str("X 0 X 1 Z 3") == expected


def test_pauli_string_explicit_from_dense_and_sparse_formats() -> None:
    expected = X(0) & Z(3)

    assert PauliString.from_dense_str("XIIZ") == expected
    assert PauliString.from_sparse_str("X0 Z3") == expected


def test_pauli_string_from_str_sparse_keeps_phase_and_high_qubits() -> None:
    pauli = PauliString.from_str("-i X2 Z10000")

    assert pauli.get_phase() == 3
    assert pauli.get_paulis() == [(Pauli.X, 2), (Pauli.Z, 10000)]
    assert pauli.weight() == 2


def test_pauli_string_dense_and_sparse_round_trips() -> None:
    pauli = PauliString.from_sparse_str("-i X2 Z4")

    assert pauli.to_sparse_str() == "-iX2 Z4"
    assert pauli.to_dense_str() == "-iIIXIZ"
    assert pauli.to_dense_str(num_qubits=7) == "-iIIXIZII"
    assert PauliString.from_sparse_str(pauli.to_sparse_str()) == pauli
    assert PauliString.from_dense_str(pauli.to_dense_str()) == pauli


def test_pauli_string_tuple_constructor_canonicalizes_for_hashing() -> None:
    sorted_pauli = PauliString([(Pauli.X, 0), (Pauli.Y, 3)])
    unsorted_pauli = PauliString([(Pauli.Y, 3), (Pauli.X, 0)])
    constructed = X(0) & PauliString.Y(3)

    assert sorted_pauli == unsorted_pauli == constructed
    assert hash(sorted_pauli) == hash(unsorted_pauli) == hash(constructed)
    assert {sorted_pauli: "first", unsorted_pauli: "second"} == {constructed: "second"}


def test_pauli_string_tuple_constructor_rejects_duplicate_qubits() -> None:
    with pytest.raises(ValueError, match="multiple non-identity"):
        PauliString([(Pauli.X, 0), (Pauli.Z, 0)])

    assert PauliString([(Pauli.I, 0), (Pauli.X, 0)]) == X(0)


def test_quantum_namespace_exports_pauli_constructors() -> None:
    import pecos.quantum as quantum
    from pecos.quantum import pauli_string

    expected = X(0) & Z(3)

    assert quantum.X(0) & quantum.Z(3) == expected
    assert pauli_string("X0 Z3") == expected
    assert pauli_string("XIIZ") == expected
    assert pauli_string(((quantum.Pauli.X, 0), (quantum.Pauli.Z, 3))) == expected
    assert pauli_string({0: quantum.Pauli.X, 3: quantum.Pauli.Z}) == expected
