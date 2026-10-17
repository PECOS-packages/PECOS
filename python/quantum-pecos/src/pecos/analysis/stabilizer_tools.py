"""Bridges for Rust Paulis and maximal local measurement groups.

Group elements are materialized in exponential space in the rank. Enumerating
all maximal compatible groups can also take exponential time and space.
"""

from __future__ import annotations

from collections.abc import Mapping
from itertools import combinations
from typing import TYPE_CHECKING

import networkx as nx
from pecos_rslib import Pauli, PauliString

from pecos.circuits import QuantumCircuit

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import SupportsIndex

    from pecos_rslib import PauliStabilizerGroup


def to_pauli_string(
    value: PauliString | str | Mapping[str, Iterable[SupportsIndex]] | QuantumCircuit,
) -> PauliString:
    """Convert a Pauli, signed dense/sparse string, mapping, or one-tick circuit.

    Circuits must have no circuit or operation metadata. Qubit extraction and
    duplicate nonidentity-qubit validation use the Rust binding constructor.
    """
    if isinstance(value, PauliString):
        return value
    if isinstance(value, str):
        return PauliString.from_str(value)
    if isinstance(value, QuantumCircuit):
        if len(value) != 1:
            msg = f"A Pauli circuit must contain exactly one tick, got {len(value)}; supply a single-tick circuit"
            raise ValueError(msg)
        if value.metadata:
            msg = f"A Pauli circuit must have no circuit metadata, got {value.metadata!r}; remove the metadata"
            raise ValueError(msg)
        entries = []
        for symbol, locations, metadata in value.items():
            if metadata:
                msg = f"Pauli operation {symbol!r} has metadata {metadata!r}; remove the operation metadata"
                raise ValueError(msg)
            entries.append((symbol, locations))
    elif isinstance(value, Mapping):
        entries = list(value.items())
    else:
        msg = (
            f"value has type {type(value).__name__} and value {value!r}; "
            "expected a PauliString, string, mapping, or single-tick QuantumCircuit"
        )
        raise TypeError(msg)
    paulis = []
    for symbol, locations in entries:
        if symbol not in {"I", "X", "Y", "Z"}:
            msg = f"Invalid Pauli symbol: {symbol!r}; use I, X, Y, or Z"
            raise ValueError(msg)
        pauli = Pauli.from_str(symbol)
        paulis.extend((pauli, qubit) for qubit in locations)
    return PauliString(paulis)


def pauli_to_dict(pauli: PauliString) -> dict[str, set[int]]:
    """Convert to nonidentity qubit sets; reject phases other than +1."""
    if pauli.get_phase() != 0:
        msg = f"Pauli {pauli} has phase {pauli.get_phase()}; a mapping requires phase +1 (0 quarter turns)"
        raise ValueError(msg)
    result: dict[str, set[int]] = {}
    for symbol, qubit in pauli.get_paulis():
        result.setdefault(str(symbol), set()).add(qubit)
    return result


def pauli_to_circuit(pauli: PauliString) -> QuantumCircuit:
    """Convert to one tick, omitting identity; reject phases other than +1."""
    return QuantumCircuit([pauli_to_dict(pauli)])


def compatible_stabilizer_groups(
    group: PauliStabilizerGroup,
    operator: PauliString,
) -> dict[PauliString, list[set[PauliString]]]:
    """Find maximal jointly measurable stabilizer groups for each operator variant.

    The operator must have real phase and commute with every generator.
    Identity and group members are valid operators. Variants are operator * s
    for each distinct group element s, retaining absolute phases. Candidates
    are nonidentity group elements qubitwise compatible with the variant.
    Variants with no candidates are omitted. Groups are maximal cliques and
    may overlap; this is not a minimum measurement-setting partition.

    Variant keys are ordered lexicographically by str(pauli). Each clique is
    sorted by that same string and the group list is ordered lexicographically
    by the resulting tuples of strings. Group members themselves form sets.
    Element and maximal-clique enumeration have exponential cost.
    """
    if operator.get_phase() not in (0, 2):
        msg = f"Operator {operator} has phase {operator.get_phase()}; supply a real phase (+1 or -1)"
        raise ValueError(msg)
    for index, generator in enumerate(group.stabilizers()):
        if not operator.commutes_with(generator):
            msg = (
                f"Operator {operator} anticommutes with generator {index} ({generator}); "
                "supply an operator commuting with every stabilizer generator"
            )
            raise ValueError(msg)
    elements = group.elements()
    stabilizers = sorted((s for s in elements if s.weight()), key=str)
    graph = nx.Graph()
    graph.add_nodes_from(stabilizers)
    for first, second in combinations(stabilizers, 2):
        if first.qubitwise_commutes_with(second):
            graph.add_edge(first, second)
    result: dict[PauliString, list[set[PauliString]]] = {}
    for variant in sorted((operator * s for s in elements), key=str):
        candidates = [s for s in stabilizers if variant.qubitwise_commutes_with(s)]
        if candidates:
            cliques = sorted(
                nx.find_cliques(graph.subgraph(candidates)),
                key=lambda clique: tuple(sorted(map(str, clique))),
            )
            result[variant] = [set(clique) for clique in cliques]
    return result
