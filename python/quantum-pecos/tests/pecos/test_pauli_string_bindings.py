# Copyright 2026 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0

import pytest
from pecos_rslib import H, Pauli, PauliString, X, Y, Z


def test_pauli_string_from_str_accepts_dense_and_sparse_formats() -> None:
    expected = X(0) & X(1) & Z(3)

    assert PauliString.from_str("XXIZ") == expected
    assert PauliString.from_str("X0 X1 Z3") == expected
    assert PauliString.from_str("X 0 X 1 Z 3") == expected


def test_pauli_string_common_representations_are_interchangeable_and_hash_equal() -> None:
    from_constructors = X(0) & Y(2) & Z(5)
    from_class_constructors = PauliString.X(0) & PauliString.Y(2) & PauliString.Z(5)
    from_sparse = PauliString.from_str("X0 Y2 Z5")
    from_explicit_sparse = PauliString.from_sparse_str("X0 Y2 Z5")
    from_dense = PauliString.from_str("XIYIIZ")
    from_explicit_dense = PauliString.from_dense_str("XIYIIZ")
    from_tuples = PauliString([(Pauli.Z, 5), (Pauli.X, 0), (Pauli.Y, 2)])

    forms = [
        from_constructors,
        from_class_constructors,
        from_sparse,
        from_explicit_sparse,
        from_dense,
        from_explicit_dense,
        from_tuples,
    ]
    assert all(form == from_constructors for form in forms)
    assert len({hash(form) for form in forms}) == 1
    assert {form: idx for idx, form in enumerate(forms)} == {from_constructors: len(forms) - 1}


def test_pauli_string_tensor_result_is_pauli_string() -> None:
    tensor = X(0) & Y(3)

    assert isinstance(tensor, PauliString)
    assert tensor.get_paulis() == [(Pauli.X, 0), (Pauli.Y, 3)]


def test_pauli_string_tensor_equality_hash_and_text_forms_match() -> None:
    tensor = X(0) & Y(3)
    same_sparse = PauliString.from_sparse_str("X0 Y3")
    same_dense = PauliString.from_dense_str("XIIY")
    same_from_tuples = PauliString([(Pauli.Y, 3), (Pauli.X, 0)])

    assert tensor == same_sparse == same_dense == same_from_tuples
    assert len({tensor, same_sparse, same_dense, same_from_tuples}) == 1
    assert tensor.to_sparse_str() == "+X0 Y3"
    assert tensor.to_dense_str() == "+XIIY"


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


def test_pauli_string_tensor_rejects_overlapping_qubits() -> None:
    with pytest.raises(ValueError, match=r"overlapping qubits: \[0\]"):
        _ = X(0) & Y(0)

    with pytest.raises(ValueError, match="tensor product requires disjoint Pauli support"):
        _ = X(0) & Z(0)

    with pytest.raises(ValueError, match=r"overlapping qubits: \[2\]"):
        _ = (X(0) & Y(2)) & Z(2)


def test_pauli_string_tensor_rejects_non_pauli_operands_explicitly() -> None:
    with pytest.raises(TypeError):
        _ = X(0) & H(1)

    with pytest.raises(TypeError):
        _ = H(0) & H(1)


def test_pauli_string_composition_allows_same_qubit() -> None:
    composed = X(0) * Z(0)

    assert composed.get_paulis() == [(Pauli.Y, 0)]
    assert composed.get_phase() == 3


def test_pauli_string_tensor_result_is_hashable() -> None:
    tensor = X(0) & Y(3)

    assert {tensor: "xy"}[PauliString.from_sparse_str("X0 Y3")] == "xy"


def test_pauli_string_tensor_preserves_phase_through_string_roundtrip() -> None:
    tensor = -X(0) & Z(1)

    assert tensor.get_phase() == 2
    assert tensor.to_sparse_str() == "-X0 Z1"
    assert tensor.to_dense_str() == "-XZ"
    assert PauliString.from_sparse_str(tensor.to_sparse_str()) == tensor
    assert PauliString.from_dense_str(tensor.to_dense_str()) == tensor


def test_quantum_namespace_exports_pauli_constructors() -> None:
    import pecos.quantum as quantum
    from pecos.quantum import pauli_string

    expected = X(0) & Z(3)

    assert quantum.X(0) & quantum.Z(3) == expected
    assert pauli_string("X0 Z3") == expected
    assert pauli_string("XIIZ") == expected
    assert pauli_string(((quantum.Pauli.X, 0), (quantum.Pauli.Z, 3))) == expected
    assert pauli_string({0: quantum.Pauli.X, 3: quantum.Pauli.Z}) == expected
