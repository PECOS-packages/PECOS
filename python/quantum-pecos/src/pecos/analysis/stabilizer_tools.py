"""Pauli conversion, ordered generator products, and local measurement groups.

Dense inputs use one I/X/Y/Z character per qubit, starting at qubit zero.
Input phases are not accepted; product phases are returned separately. Products
and maximal compatible groups can grow exponentially with the generator count.
"""

from __future__ import annotations

from collections.abc import Mapping
from itertools import chain, combinations, product
from typing import TYPE_CHECKING

import networkx as nx
from pecos_rslib import PauliString

from pecos.circuits import QuantumCircuit

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from collections.abc import Set as AbstractSet

PauliInput = str | Mapping[str, "AbstractSet[int]"] | QuantumCircuit


def _validate_dense(operator: str) -> None:
    if not isinstance(operator, str):
        msg = "Expected a dense I/X/Y/Z string"
        raise TypeError(msg)
    if set(operator) - set("IXYZ"):
        msg = "Expected a phase-free dense I/X/Y/Z string"
        raise ValueError(msg)


def pauli_to_dict(operator: str) -> dict[str, set[int]]:
    """Convert a dense Pauli to disjoint nonidentity qubit sets."""
    _validate_dense(operator)
    result: dict[str, set[int]] = {}
    for qubit, symbol in enumerate(operator):
        if symbol != "I":
            result.setdefault(symbol, set()).add(qubit)
    return result


def _locations(operator: PauliInput) -> dict[int, str]:
    if isinstance(operator, str):
        operator = pauli_to_dict(operator)
    if isinstance(operator, QuantumCircuit):
        if len(operator) != 1:
            msg = "A Pauli circuit must contain exactly one tick"
            raise ValueError(msg)
        entries = []
        for symbol, locations, metadata in operator.items():
            if metadata:
                msg = "A Pauli circuit must have no operation metadata"
                raise ValueError(msg)
            entries.append((symbol, locations))
    elif isinstance(operator, Mapping):
        entries = list(operator.items())
    else:
        msg = "Expected a Pauli string, mapping, or single-tick QuantumCircuit"
        raise TypeError(msg)
    result: dict[int, str] = {}
    for symbol, locations in entries:
        if symbol not in {"I", "X", "Y", "Z"}:
            msg = f"Invalid Pauli symbol: {symbol!r}"
            raise ValueError(msg)
        for qubit in locations:
            if isinstance(qubit, bool) or not isinstance(qubit, int) or qubit < 0:
                msg = "Pauli locations must be nonnegative integer qubit indices"
                raise ValueError(msg)
            if qubit in result:
                msg = f"Multiple Pauli operators specified for qubit {qubit}"
                raise ValueError(msg)
            result[qubit] = symbol
    return result


def pauli_to_string(operator: PauliInput, num_qubits: int) -> str:
    """Convert to dense form, preserving explicit width and rejecting truncation."""
    if isinstance(num_qubits, bool) or not isinstance(num_qubits, int) or num_qubits < 0:
        msg = "num_qubits must be a nonnegative integer"
        raise ValueError(msg)
    locations = _locations(operator)
    if (isinstance(operator, str) and len(operator) > num_qubits) or any(q >= num_qubits for q in locations):
        msg = "Pauli operator exceeds num_qubits"
        raise ValueError(msg)
    return "".join(locations.get(q, "I") for q in range(num_qubits))


def pauli_to_circuit(operator: str) -> QuantumCircuit:
    """Convert a dense Pauli into one tick, omitting identity operations."""
    return QuantumCircuit([pauli_to_dict(operator)])


def commutes(first: PauliInput, second: PauliInput) -> bool:
    """Check Pauli commutation; absent qubits are identities in every input form."""
    first_locs, second_locs = _locations(first), _locations(second)
    left = PauliString.from_sparse_str(" ".join(f"{p}{q}" for q, p in first_locs.items() if p != "I"))
    right = PauliString.from_sparse_str(" ".join(f"{p}{q}" for q, p in second_locs.items() if p != "I"))
    return left.commutes_with(right)


def measurement_compatible(first: str, second: str) -> bool:
    """Whether equal-width Paulis share a product measurement basis.

    This is qubit-wise compatibility, stronger than commutation: XX and ZZ
    commute, but cannot be measured in the same individual-qubit bases.
    """
    _validate_dense(first)
    _validate_dense(second)
    if len(first) != len(second):
        msg = "Pauli strings must have equal lengths"
        raise ValueError(msg)
    return all(a == "I" or b == "I" or a == b for a, b in zip(first, second, strict=True))


def iter_stabilizer_products(
    generators: Sequence[str],
    operator: str | None = None,
) -> Iterator[tuple[str, str]]:
    """Yield an operator times each ordered subset of Pauli generators.

    Subsets follow binary lexicographic order (last generator varies fastest).
    Multiplication is operator * generator[0] * generator[1] * ...; phases are
    '+', '+i', '-', or '-i'. Noncommuting generators are allowed, so imaginary
    phases are retained. Dependent generators retain duplicate products.
    Empty generators yield the supplied operator, or the zero-qubit identity.
    Work is exponential, but results are yielded without materializing 2**n items.
    Validation happens when iteration starts.
    """
    generators = tuple(generators)
    for generator in generators:
        _validate_dense(generator)
    if operator is not None:
        _validate_dense(operator)
    width = len(generators[0]) if generators else len(operator or "")
    if any(len(g) != width for g in generators) or (operator is not None and len(operator) != width):
        msg = "Generators and operator must have equal lengths"
        raise ValueError(msg)
    initial = PauliString.from_str(operator if operator is not None else "I" * width)
    parsed = [PauliString.from_str(g) for g in generators]
    for selected in product((False, True), repeat=len(parsed)):
        value = initial
        for include, generator in zip(selected, parsed, strict=True):
            if include:
                value = value * generator
        phase = ("+", "+i", "-", "-i")[value.get_phase()]
        yield phase, value.to_dense_str(width)[len(phase) :]


def compatible_stabilizer_groups(generators: Sequence[str], operator: str) -> dict[str, list[set[str]]]:
    """Find all maximal jointly measurable stabilizer groups for each variant.

    Each group is pairwise qubit-wise compatible, including with the operator
    variant. Identity stabilizers are excluded. Groups may overlap; this is not
    a minimum measurement-setting partition. Generators must commute. Outputs
    have deterministic variant and group order, and include product signs.
    Both product enumeration and maximal-clique enumeration are exponential.
    """
    generators = tuple(generators)
    variants = iter_stabilizer_products(generators, operator)
    first_variant = next(variants)  # Validate inputs before graph construction.
    if any(not commutes(a, b) for a, b in combinations(generators, 2)):
        msg = "Stabilizer generators must commute"
        raise ValueError(msg)
    stabilizers = sorted({(sign, body) for sign, body in iter_stabilizer_products(generators) if set(body) - {"I"}})
    graph = nx.Graph()
    graph.add_nodes_from(sign + body for sign, body in stabilizers)
    for (sign_a, a), (sign_b, b) in combinations(stabilizers, 2):
        if measurement_compatible(a, b):
            graph.add_edge(sign_a + a, sign_b + b)
    result: dict[str, list[set[str]]] = {}
    for sign, body in chain((first_variant,), variants):
        candidates = [s + p for s, p in stabilizers if measurement_compatible(body, p)]
        if candidates:
            cliques = sorted(tuple(sorted(c)) for c in nx.find_cliques(graph.subgraph(candidates)))
            result[sign + body] = [set(c) for c in cliques]
    return result
