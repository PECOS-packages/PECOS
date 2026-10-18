# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Generic GF(2) Clifford word compilation and signed Pauli corrections.

Matrix is a hashable tuple of row tuples of integer bits. Images are columns;
appending a generator in time order multiplies its matrix on the left.
The basis is (x0, ..., x{k-1}, z0, ..., z{k-1}).
"""

import heapq
import re
from collections.abc import Mapping
from itertools import product

Matrix = tuple[tuple[int, ...], ...]


def _multiply(left: Matrix, right: Matrix) -> Matrix:
    return tuple(
        tuple(sum(a * b for a, b in zip(row, col, strict=True)) % 2 for col in zip(*right, strict=True)) for row in left
    )


def _identity(size: int) -> Matrix:
    return tuple(tuple(int(i == j) for j in range(size)) for i in range(size))


def min_cost_words(generators: Mapping[str, Matrix], costs: Mapping[str, int]) -> dict[Matrix, tuple[str, ...]]:
    """Dijkstra enumeration with nonnegative integer costs.

    Ties prefer fewer operations, then lexicographically smaller time-order
    words. This well-founded ordering also handles zero-cost cycles. An empty
    generator mapping is rejected because its matrix dimension is unknown.
    """
    if not generators:
        msg = "generators must be nonempty to determine the matrix dimension"
        raise ValueError(msg)
    first_name, first_matrix = next(iter(generators.items()))
    if not isinstance(first_matrix, tuple):
        msg = f"Generator {first_name!r}: expected a tuple matrix, got {first_matrix!r}"
        raise TypeError(msg)
    size = len(first_matrix)
    if not size or size % 2:
        msg = f"Matrix dimension must be positive and even, got {size}"
        raise ValueError(msg)
    symplectic = tuple(tuple(int((i + size // 2) % size == j) for j in range(size)) for i in range(size))
    for name, matrix in generators.items():
        if not isinstance(name, str):
            msg = f"Generator names must be strings, got {name!r}"
            raise TypeError(msg)
        if not name:
            msg = "Generator names must be nonempty"
            raise ValueError(msg)
        if not isinstance(matrix, tuple) or any(
            not isinstance(row, tuple) or any(type(bit) is not int for bit in row) for row in matrix
        ):
            msg = f"Generator {name!r}: expected a tuple matrix of int bits, got {matrix!r}"
            raise TypeError(msg)
        if len(matrix) != size or any(len(row) != size or set(row) - {0, 1} for row in matrix):
            msg = f"Generator {name!r}: expected {size}x{size} matrix of bits, got {matrix!r}"
            raise ValueError(msg)
        transpose = tuple(zip(*matrix, strict=True))
        if _multiply(_multiply(transpose, symplectic), matrix) != symplectic:
            msg = f"Generator {name!r} is not symplectic: {matrix!r}"
            raise ValueError(msg)
        if name not in costs:
            msg = f"Generator {name!r} has no cost"
            raise ValueError(msg)
        if type(costs[name]) is not int:
            msg = f"Generator {name!r}: expected an int cost, got {costs[name]!r}"
            raise TypeError(msg)
        if costs[name] < 0:
            msg = f"Generator {name!r}: expected a nonnegative cost, got {costs[name]!r}"
            raise ValueError(msg)
    if extra := sorted(set(costs) - set(generators), key=repr):
        msg = f"costs name unknown generators {extra!r}"
        raise ValueError(msg)
    identity = _identity(size)
    best = {identity: (0, 0, ())}
    queue = [(0, 0, (), identity)]
    while queue:
        cost, length, word, element = heapq.heappop(queue)
        if best[element] != (cost, length, word):
            continue
        for name in sorted(generators):
            neighbor = _multiply(generators[name], element)
            candidate = (cost + costs[name], length + 1, (*word, name))
            if neighbor not in best or candidate < best[neighbor]:
                best[neighbor] = candidate
                heapq.heappush(queue, (*candidate, neighbor))
    return {matrix: record[2] for matrix, record in best.items()}


def parse_signed_logical_pauli(image: str, size: int) -> tuple[int, tuple[int, ...]]:
    """Return (sign bit, symplectic bits) for a canonical signed logical Pauli on size qubits."""
    if not isinstance(image, str):
        msg = f"Expected a signed logical Pauli string, got {image!r}"
        raise TypeError(msg)
    if not image or image[0] not in "+-":
        msg = f"Expected signed logical Pauli, got {image!r}"
        raise ValueError(msg)
    factors = re.findall(r"([XYZ])(0|[1-9][0-9]*)", image[1:])
    if image[1:] != "I" and (not factors or "".join(p + q for p, q in factors) != image[1:]):
        msg = f"Invalid logical Pauli {image!r}"
        raise ValueError(msg)
    indices = [int(q) for _, q in factors]
    if indices != sorted(set(indices)) or any(q >= size for q in indices):
        msg = f"Noncanonical or out-of-range logical Pauli {image!r} for k={size}"
        raise ValueError(msg)
    letters = {int(q): p for p, q in factors}
    return int(image[0] == "-"), tuple(int(letters.get(j, "I") in axes) for axes in ("XY", "ZY") for j in range(size))


def logical_pauli_fixup(actual: Mapping[str, str], target: Mapping[str, str]) -> str:
    """Return an unsigned canonical logical Pauli applied AFTER actual.

    The two actions must have identical symplectic parts. Factors are ordered
    by logical index, with Y for iXZ and I for the identity. The search tries
    all 4^k logical Paulis, which is meant for small k.
    """
    size = len(actual) // 2
    labels = tuple(f"{axis}{j}" for axis in "XZ" for j in range(size))
    if not size or set(actual) != set(labels) or set(target) != set(labels):
        msg = f"Actions must have matching complete X/Z basis keys, got {tuple(actual)!r}, {tuple(target)!r}"
        raise ValueError(msg)
    equations = []
    for label in labels:
        sign, bits = parse_signed_logical_pauli(actual[label], size)
        target_sign, target_bits = parse_signed_logical_pauli(target[label], size)
        if bits != target_bits:
            msg = f"Symplectic parts differ at {label}: {actual[label]!r} versus {target[label]!r}"
            raise ValueError(msg)
        equations.append((bits, sign ^ target_sign))
    for letters in product("IXYZ", repeat=size):
        x = tuple(int(p in "XY") for p in letters)
        z = tuple(int(p in "ZY") for p in letters)
        if all(
            sum(x[j] * bits[size + j] + z[j] * bits[j] for j in range(size)) % 2 == sign for bits, sign in equations
        ):
            return "".join(f"{p}{j}" for j, p in enumerate(letters) if p != "I") or "I"
    msg = f"Signed actions admit no Pauli correction: {actual!r}, {target!r}"
    raise ValueError(msg)
