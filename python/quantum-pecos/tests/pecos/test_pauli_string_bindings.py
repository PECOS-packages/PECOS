# Copyright 2026 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0

import numpy as np
import pytest
from pecos_rslib import H, Pauli, PauliSequence, PauliStabilizerGroup, PauliString, X, Xs, Y, Ys, Z, Zs


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
    with pytest.raises(ValueError, match="duplicate Pauli entry for qubit 0"):
        PauliString([(Pauli.X, 0), (Pauli.Z, 0)])

    with pytest.raises(ValueError, match="duplicate Pauli entry for qubit 0"):
        PauliString([(Pauli.I, 0), (Pauli.X, 0)])


def test_pauli_string_tensor_rejects_overlapping_qubits() -> None:
    with pytest.raises(ValueError, match=r"qubit 0 appears in both operands"):
        _ = X(0) & Y(0)

    with pytest.raises(ValueError, match="use disjoint Pauli support for the tensor product"):
        _ = X(0) & Z(0)

    with pytest.raises(ValueError, match=r"qubit 2 appears in both operands"):
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


@pytest.mark.parametrize("position", [0, 1])
@pytest.mark.parametrize(
    ("index", "error"),
    [
        (True, TypeError),
        (False, TypeError),
        (None, TypeError),
        ("1", TypeError),
        (1.0, TypeError),
        (-1, ValueError),
        (2**128, ValueError),
    ],
)
def test_constructor_index_errors(index: object, error: type[Exception], position: int) -> None:
    """The same invalid index has the same error at every tuple position."""
    items = [(Pauli.X, 0), (Pauli.Z, 1)]
    items[position] = (Pauli.X, index)
    with pytest.raises(error, match=f"paulis item {position} qubit"):
        PauliString(items)


@pytest.mark.parametrize("name", ["X", "Y", "Z"])
@pytest.mark.parametrize(
    ("index", "error"),
    [
        (True, TypeError),
        (False, TypeError),
        (None, TypeError),
        ("1", TypeError),
        (1.0, TypeError),
        (-1, ValueError),
        (2**128, ValueError),
    ],
)
def test_helper_index_errors(name: str, index: object, error: type[Exception]) -> None:
    """All Pauli-building helpers share qubit validation."""
    single = {"X": X, "Y": Y, "Z": Z}[name]
    multiple = {"X": Xs, "Y": Ys, "Z": Zs}[name]
    with pytest.raises(error, match="qubit"):
        single(index)
    with pytest.raises(error, match="qubit"):
        getattr(PauliString, name)(index)
    for position, indices in enumerate([[index, 2], [2, index]]):
        with pytest.raises(error, match=f"qubits item {position}"):
            multiple(indices)


@pytest.mark.parametrize("integer_type", [np.int64, np.uint32])
def test_constructor_helpers_accept_numpy_indices(integer_type: type) -> None:
    """Every explicit-index constructor retains NumPy integer interoperability."""
    assert PauliString([(Pauli.X, integer_type(0)), (Pauli.Y, integer_type(1))]) == X(0) * Y(1)
    for name, single, multiple in [("X", X, Xs), ("Y", Y, Ys), ("Z", Z, Zs)]:
        assert single(integer_type(2)) == single(2)
        assert getattr(PauliString, name)(integer_type(2)) == single(2)
        assert multiple([integer_type(0), integer_type(2)]) == single(0) * single(2)


def test_constructor_implicit_paulis() -> None:
    """Shape selection retains the implicit-index form."""
    assert PauliString([Pauli.X, Pauli.I, Pauli.Z]) == X(0) * Z(2)


@pytest.mark.parametrize("width", [0, 3, 5])
def test_dense_str_rejects_truncation(width: int) -> None:
    pauli = PauliString.from_sparse_str("X0 Z5")
    with pytest.raises(ValueError, match=rf"qubit \d+ is outside num_qubits {width}"):
        pauli.to_dense_str(width)
    assert pauli.to_dense_str(6) == "+XIIIIZ"


def test_signed_sequence_dense_roundtrip() -> None:
    original = PauliSequence.from_str("-X\n+iIZ\n-iY\n+I")
    dense = "-XI\n+iIZ\n-iYI\n+II"
    assert original.to_dense_str() == dense
    assert str(original) == dense
    assert PauliSequence.from_str(dense).paulis() == original.paulis()


def test_signed_stabilizer_dense_roundtrip() -> None:
    original = PauliStabilizerGroup([-(X(0) & X(1)), Z(0) & Z(1)])
    dense = "-XX\n+ZZ"
    assert original.to_dense_str() == dense
    assert str(original) == dense
    assert PauliStabilizerGroup.from_str(dense).stabilizers() == original.stabilizers()


@pytest.mark.parametrize("text", ["X0 Z0", "X0 X0", "Z1 Z1 X0"])
def test_sparse_parser_rejects_duplicate_qubits(text: str) -> None:
    with pytest.raises(ValueError, match="duplicate Pauli entry for qubit"):
        PauliString.from_sparse_str(text)
    with pytest.raises(ValueError, match="duplicate Pauli entry for qubit"):
        PauliSequence.from_str("Z2\n" + text)


def test_multi_qubit_constructors_reject_duplicates() -> None:
    from pecos_rslib import Xs, Ys, Zs

    for constructor in [Xs, Ys, Zs]:
        with pytest.raises(ValueError, match="duplicate Pauli entry for qubit 1"):
            constructor([1, 1])


def test_pauli_string_order_and_identities_do_not_affect_keys() -> None:
    sorted_pauli = Xs([0, 2])
    unsorted_pauli = Xs([2, 0])
    with_identity = PauliString([(Pauli.X, 2), (Pauli.I, 5), (Pauli.X, 0)])
    for pauli in (unsorted_pauli, with_identity):
        assert pauli == sorted_pauli
        assert hash(pauli) == hash(sorted_pauli)
        assert pauli in {sorted_pauli}
        assert {sorted_pauli: "found"}[pauli] == "found"
        assert pauli.weight() == 2
        assert pauli.get_paulis() == [(Pauli.X, 0), (Pauli.X, 2)]
    with pytest.raises(ValueError, match="duplicate Pauli entry for qubit 0"):
        PauliString([(Pauli.I, 0), (Pauli.X, 0)])


def test_constructor_and_width_errors_are_value_errors():
    """Caller errors must stay ordinary Python exceptions, never PanicException."""
    import pytest
    from pecos_rslib import Pauli, PauliString

    with pytest.raises(ValueError, match="duplicate Pauli entry for qubit 7"):
        PauliString([(Pauli.X, 7), (Pauli.Z, 7)])
    with pytest.raises(ValueError, match="qubit 7 is outside num_qubits 3"):
        PauliString([(Pauli.X, 7)]).to_dense_str(num_qubits=3)
