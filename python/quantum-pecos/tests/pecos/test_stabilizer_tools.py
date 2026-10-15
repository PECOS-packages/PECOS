"""Independent algebra and grouping checks for stabilizer analysis."""

from itertools import combinations, product

import numpy as np
import pytest
from pecos import QuantumCircuit
from pecos.analysis.stabilizer_tools import (
    compatible_stabilizer_groups,
    pauli_to_circuit,
    pauli_to_dict,
    to_pauli_string,
)
from pecos_rslib import PauliStabilizerGroup, PauliString

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


def test_conversions() -> None:
    """All edge forms preserve support, with width supplied at rendering time."""
    for body in ["", "III", "IXYZI"]:
        pauli = to_pauli_string(body)
        assert to_pauli_string(pauli) is pauli
        assert to_pauli_string(pauli_to_dict(pauli)) == pauli
        circuit = pauli_to_circuit(pauli)
        assert len(circuit) == 1
        assert to_pauli_string(circuit) == pauli
        assert pauli.to_dense_str(len(body)) == "+" + body
    assert pauli_to_dict(to_pauli_string("IXYX")) == {"X": {1, 3}, "Y": {2}}
    assert to_pauli_string("-XZ") == -to_pauli_string("X0 Z1")
    assert to_pauli_string({"I": {2}, "X": {0}}).to_dense_str(4) == "+XIII"
    assert to_pauli_string({"X": {np.int64(1)}, "Z": {np.uint32(3)}}) == to_pauli_string("X1 Z3")
    assert to_pauli_string(QuantumCircuit([{"X": {np.int64(1)}}])) == to_pauli_string("X1")


def test_products_against_matrix_oracle() -> None:
    """All two-qubit pairs preserve multiplication order and complex phases."""
    initial = to_pauli_string("XY")
    for first, second in product(PAULIS, repeat=2):
        a, b = matrix(first), matrix(second)
        left, right = to_pauli_string(first), to_pauli_string(second)
        assert left.commutes_with(right) == np.array_equal(a @ b, b @ a)
        actual = initial * left * right
        sign = ("+", "+i", "-", "-i")[actual.get_phase()]
        body = actual.to_dense_str(2)[len(sign) :]
        np.testing.assert_array_equal(PHASES[sign] * matrix(body), matrix("XY") @ a @ b)


@pytest.mark.parametrize("generators", [["XX", "ZZ"], ["-XX", "ZZ"], ["XX", "ZZ", "-YY"]])
def test_group_products_against_matrix_oracle(generators: list[str]) -> None:
    """Generator masks and distinct elements retain absolute product phases."""
    group = PauliStabilizerGroup([to_pauli_string(g) for g in generators])
    products = []
    for mask in range(1 << len(generators)):
        expected = np.eye(4)
        for index, generator in enumerate(generators):
            if mask & (1 << index):
                expected = expected @ ((-1 if generator.startswith("-") else 1) * matrix(generator.lstrip("-")))
        actual = group.element(mask)
        sign = ("+", "+i", "-", "-i")[actual.get_phase()]
        np.testing.assert_array_equal(PHASES[sign] * matrix(actual.to_dense_str(2)[len(sign) :]), expected)
        products.append(actual)
    assert set(group.elements()) == set(products)
    assert len(group.elements()) == 4


@pytest.mark.parametrize("first", ["XX", "-XX"])
def test_bell_groups(first: str) -> None:
    """Signed generators produce absolute phases in both keys and groups."""
    group = PauliStabilizerGroup.from_str(first + "\nZZ")
    identity = to_pauli_string("II")
    xx, zz = to_pauli_string(first), to_pauli_string("ZZ")
    yy = xx * zz
    result = compatible_stabilizer_groups(group, identity)
    if first == "XX":
        assert result[identity] == [{yy}, {xx}, {zz}]
        assert list(result) == [yy, identity, xx, zz]
    else:
        assert result[identity] == [{xx}, {yy}, {zz}]
        assert list(result) == [xx, identity, yy, zz]
    assert result[xx] == [{xx}]
    assert result[yy] == [{yy}]
    assert result[zz] == [{zz}]
    assert compatible_stabilizer_groups(group, xx) == result
    negative = compatible_stabilizer_groups(group, -identity)
    assert negative == {-variant: cliques for variant, cliques in result.items()}


def test_dependent_and_invalid_groups() -> None:
    """Redundant generators are valid, but negative identity is not."""
    independent = PauliStabilizerGroup.from_str("XX\nZZ")
    dependent = PauliStabilizerGroup.from_str("XX\nZZ\n-YY")
    assert compatible_stabilizer_groups(dependent, PauliString()) == compatible_stabilizer_groups(
        independent,
        PauliString(),
    )
    with pytest.raises(ValueError, match="-I"):
        PauliStabilizerGroup.from_str("XX\nZZ\nYY")
    assert compatible_stabilizer_groups(PauliStabilizerGroup([]), to_pauli_string("X")) == {}
    # XX and its variant -YY commute with ZZ but share no local basis with it.
    assert compatible_stabilizer_groups(PauliStabilizerGroup.from_str("Z0 Z1"), to_pauli_string("X0 X1")) == {}


def test_groups_are_maximal_jointly_compatible_and_deterministic() -> None:
    """A shared compatible neighbor must not merge incompatible XX/ZZ groups."""
    group = PauliStabilizerGroup.from_str("XXII\nZZII\nIIXX")
    result = compatible_stabilizer_groups(group, PauliString())
    assert result == compatible_stabilizer_groups(group, PauliString())
    stabilizers = {s for s in group.elements() if s.weight()}
    for variant, groups in result.items():
        candidates = {s for s in stabilizers if variant.qubitwise_commutes_with(s)}
        assert set.union(*groups) == candidates
        assert groups == sorted(groups, key=lambda clique: tuple(sorted(map(str, clique))))
        for clique in groups:
            assert all(a.qubitwise_commutes_with(b) for a, b in combinations(clique, 2))
            assert all(any(not s.qubitwise_commutes_with(g) for g in clique) for s in candidates - clique)


@pytest.mark.parametrize("operator", ["iII", "-iXX", "XI"])
def test_invalid_operator(operator: str) -> None:
    """Operators must be real and centralize the stabilizer group."""
    with pytest.raises(ValueError, match=r"real phase|commute"):
        compatible_stabilizer_groups(PauliStabilizerGroup.from_str("XX\nZZ"), to_pauli_string(operator))


@pytest.mark.parametrize("phase", ["-", "i", "-i"])
def test_lossy_bridges_rejected(phase: str) -> None:
    """Legacy types cannot represent phases, even on identity."""
    for body in ["X", "I"]:
        pauli = to_pauli_string(phase + body)
        with pytest.raises(ValueError, match="phase"):
            pauli_to_dict(pauli)
        with pytest.raises(ValueError, match="phase"):
            pauli_to_circuit(pauli)


def test_invalid_edge_inputs() -> None:
    """Validate symbols and circuits while delegating qubit checks to bindings."""
    with pytest.raises(ValueError, match="multiple"):
        to_pauli_string({"X": {np.int64(0)}, "Z": {0}})
    with pytest.raises(ValueError, match="symbol"):
        to_pauli_string({"H": {0}})
    with pytest.raises(ValueError, match="symbol"):
        to_pauli_string(QuantumCircuit([{"H": {0}}]))
    with pytest.raises(TypeError, match="Expected"):
        to_pauli_string(123)
    for circuit in [QuantumCircuit(), QuantumCircuit([{"X": {0}}, {"Z": {0}}])]:
        with pytest.raises(ValueError, match="one tick"):
            to_pauli_string(circuit)
    conditioned = QuantumCircuit()
    conditioned.append("X", {0}, cond={"a": "c", "op": "==", "b": 1})
    for circuit in [conditioned, QuantumCircuit([{"X": {0}}], label="test")]:
        with pytest.raises(ValueError, match="metadata"):
            to_pauli_string(circuit)


@pytest.mark.parametrize("index", [True, False, None, "1", 1.0])
def test_mapping_rejects_noninteger_qubits(index: object) -> None:
    """Mappings use the binding's strict integer protocol."""
    with pytest.raises(TypeError, match="item 0"):
        to_pauli_string({"X": [index]})


@pytest.mark.parametrize("index", [-1, 2**128])
def test_mapping_rejects_out_of_range_qubits(index: int) -> None:
    """Invalid integer indices get a value error identifying the item."""
    with pytest.raises(ValueError, match="item 0"):
        to_pauli_string({"X": [index]})
