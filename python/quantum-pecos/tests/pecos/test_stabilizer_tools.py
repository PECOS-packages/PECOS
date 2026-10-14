"""Independent algebra and grouping checks for stabilizer analysis."""

from itertools import combinations, product

import numpy as np
import pytest
from pecos import QuantumCircuit
from pecos.analysis.stabilizer_tools import (
    commutes,
    compatible_stabilizer_groups,
    iter_stabilizer_products,
    measurement_compatible,
    pauli_to_circuit,
    pauli_to_dict,
    pauli_to_string,
)

MATRICES = {
    "I": np.eye(2),
    "X": np.array([[0, 1], [1, 0]]),
    "Y": np.array([[0, -1j], [1j, 0]]),
    "Z": np.diag([1, -1]),
}
PHASES = {"+": 1, "-": -1, "+i": 1j, "-i": -1j}
PAULIS = ["".join(p) for p in product("IXYZ", repeat=2)]


def matrix(body: str) -> np.ndarray:
    """Build an independent dense matrix oracle."""
    result = np.eye(1)
    for symbol in body:
        result = np.kron(result, MATRICES[symbol])
    return result


def test_conversions_and_explicit_identity_width() -> None:
    """Round trips preserve identity padding, including empty support."""
    for body in ["", "III", "IXYZI"]:
        mapping = pauli_to_dict(body)
        assert pauli_to_string(mapping, len(body)) == body
        assert pauli_to_string(pauli_to_circuit(body), len(body)) == body
    assert pauli_to_dict("IXYX") == {"X": {1, 3}, "Y": {2}}
    assert pauli_to_string({"I": {2}, "X": {0}}, 4) == "XIII"


def test_commutation_against_matrix_oracle() -> None:
    """All two-qubit pairs agree with matrix commutation in all input forms."""
    for first, second in product(PAULIS, repeat=2):
        a, b = matrix(first), matrix(second)
        expected = np.array_equal(a @ b, b @ a)
        assert commutes(first, second) == expected
        assert commutes(pauli_to_dict(first), pauli_to_circuit(second)) == expected
    assert commutes("X", "XI")
    assert not commutes({"X": {1_000_000}}, {"Z": {1_000_000}})


def test_ordered_products_against_matrix_oracle() -> None:
    """All two-qubit generator pairs preserve product order and complex phases."""
    initial = matrix("XY")
    for first, second in product(PAULIS, repeat=2):
        a, b = matrix(first), matrix(second)
        expected = [initial, initial @ b, initial @ a, initial @ a @ b]
        actual = list(iter_stabilizer_products([first, second], "XY"))
        for (sign, body), reference in zip(actual, expected, strict=True):
            np.testing.assert_array_equal(PHASES[sign] * matrix(body), reference)


def test_empty_and_dependent_generators() -> None:
    """Empty and redundant inputs have explicit, deterministic behavior."""
    assert list(iter_stabilizer_products([])) == [("+", "")]
    assert list(iter_stabilizer_products([], "II")) == [("+", "II")]
    assert list(iter_stabilizer_products(["X", "X"])) == [("+", "I"), ("+", "X"), ("+", "X"), ("+", "I")]
    assert compatible_stabilizer_groups([], "X") == {}


def test_measurement_compatibility_is_stronger_than_commutation() -> None:
    """Commuting XX and ZZ still need different local measurement bases."""
    assert commutes("XX", "ZZ")
    assert not measurement_compatible("XX", "ZZ")
    assert measurement_compatible("XI", "IX")
    assert measurement_compatible("", "")


def test_groups_are_maximal_jointly_compatible_and_deterministic() -> None:
    """A shared compatible neighbor must not merge incompatible XX/ZZ groups."""
    generators = ["XXII", "ZZII", "IIXX"]
    result = compatible_stabilizer_groups(generators, "IIII")
    assert result == compatible_stabilizer_groups(generators, "IIII")
    assert result["+IIII"] == [
        {"+IIXX", "+XXII", "+XXXX"},
        {"+IIXX", "+ZZII", "+ZZXX"},
        {"+IIXX", "-YYII", "-YYXX"},
    ]
    stabs = {s + p for s, p in iter_stabilizer_products(generators) if p != "IIII"}
    for variant, groups in result.items():
        candidates = {s for s in stabs if measurement_compatible(variant[1:], s[1:])}
        assert set.union(*groups) == candidates
        for group in groups:
            assert all(measurement_compatible(a[1:], b[1:]) for a, b in combinations(group, 2))
            assert all(any(not measurement_compatible(s[1:], g[1:]) for g in group) for s in candidates - group)


@pytest.mark.parametrize("body", ["A", "x", "X0", "+X", " X", "iY"])
def test_invalid_dense_inputs(body: str) -> None:
    """Dense APIs reject ambiguous notation rather than dropping a phase."""
    with pytest.raises(ValueError, match="phase-free"):
        pauli_to_dict(body)
    with pytest.raises(ValueError, match="phase-free"):
        list(iter_stabilizer_products([body]))


def test_validation() -> None:
    """Reject malformed supports, truncation, mismatched widths and conditions."""
    for mapping in [{"X": {-1}}, {"X": {True}}, {"H": {0}}, {"X": {0}, "Z": {0}}]:
        with pytest.raises(ValueError, match="Pauli"):
            pauli_to_string(mapping, 2)
    with pytest.raises(ValueError, match="exceeds"):
        pauli_to_string({"X": {2}}, 2)
    with pytest.raises(ValueError, match="exceeds"):
        pauli_to_string("III", 2)
    with pytest.raises(ValueError, match="equal lengths"):
        measurement_compatible("X", "XX")
    with pytest.raises(ValueError, match="equal lengths"):
        list(iter_stabilizer_products(["X", "XX"]))
    with pytest.raises(ValueError, match="equal lengths"):
        list(iter_stabilizer_products(["X"], "XX"))
    with pytest.raises(ValueError, match="must commute"):
        compatible_stabilizer_groups(["X", "Z"], "I")
    with pytest.raises(ValueError, match="one tick"):
        commutes(QuantumCircuit([{"X": {0}}, {"Z": {0}}]), "X")
    conditioned = QuantumCircuit()
    conditioned.append("X", {0}, cond={"a": "c", "op": "==", "b": 1})
    with pytest.raises(ValueError, match="metadata"):
        commutes(conditioned, "X")
