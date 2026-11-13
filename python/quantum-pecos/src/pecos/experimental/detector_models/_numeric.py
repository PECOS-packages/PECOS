# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0
"""Small dense reference helpers using PECOS arrays and standard-library scalars."""

from __future__ import annotations

import math
from typing import Never

import pecos as pc

TOLERANCE = 1e-10


def fail(message: str) -> Never:
    """Raise a validation failure with the caller's diagnostic."""
    raise ValueError(message)


def array(values: object) -> pc.Array:
    """Copy finite complex data into a PECOS array."""
    data = _nested_lists(values)
    result = pc.array(data, dtype=pc.complex128)
    pending = [result.tolist()]
    while pending:
        value = pending.pop()
        if isinstance(value, list):
            pending.extend(value)
        elif not (math.isfinite(complex(value).real) and math.isfinite(complex(value).imag)):
            fail("operator entries must be finite")
    return result


def _nested_lists(value: object) -> object:
    if hasattr(value, "tolist"):
        return _nested_lists(value.tolist())
    if isinstance(value, (tuple, list)):
        return [_nested_lists(item) for item in value]
    return value


def operators(values: object) -> pc.Array:
    """Normalize a matrix or nonempty stack of Kraus matrices."""
    result = array(values)
    if result.ndim == 2:
        result = array([result.tolist()])
    if result.ndim != 3 or not all(result.shape):
        fail("Kraus operators must have nonempty (rank, output, input) shape")
    return result


def matrices(values: pc.Array) -> tuple[pc.Array, ...]:
    """Split a validated Kraus stack into two-dimensional PECOS arrays."""
    return tuple(array(value) for value in values.tolist())


def zeros(rows: int, columns: int) -> pc.Array:
    """Create a complex zero matrix."""
    return array([[0j] * columns for _ in range(rows)])


def identity(dimension: int) -> pc.Array:
    """Create a complex identity matrix."""
    return array([[complex(row == column) for column in range(dimension)] for row in range(dimension)])


def effect(values: pc.Array) -> pc.Array:
    """Compute sum K-adjoint K without an einsum dependency."""
    dimension = values.shape[2]
    return sum((operator.conj().T * operator for operator in matrices(values)), zeros(dimension, dimension))


def close(left: pc.Array, right: pc.Array, tolerance: float = TOLERANCE) -> bool:
    """Compare matrix shapes and entries with absolute tolerance."""
    return left.shape == right.shape and bool(pc.allclose(left, right, atol=tolerance, rtol=0))


def trace_preserving(values: pc.Array) -> None:
    """Validate completeness of all channel or instrument outcomes."""
    if not close(effect(values), identity(values.shape[2])):
        fail("all channel/instrument outcomes must together preserve trace")


def positive_semidefinite(value: pc.Array) -> bool:
    """Check Hermitian positive semidefiniteness by pivoted Schur elimination.

    Pivoting on the largest remaining diagonal handles exact rank deficiencies.
    The residual is allowed a dimension-scaled roundoff budget, not a physical
    negative eigenvalue. This is a small reference validation, not a solver API.
    """
    data = value.tolist()
    count = len(data)
    if any(
        abs(data[row][column] - data[column][row].conjugate()) > TOLERANCE
        for row in range(count)
        for column in range(count)
    ):
        return False
    tolerance = TOLERANCE / max(1, count)
    for start in range(count):
        pivot = max(range(start, count), key=lambda index: data[index][index].real)
        if pivot != start:
            data[start], data[pivot] = data[pivot], data[start]
            for row in data:
                row[start], row[pivot] = row[pivot], row[start]
        diagonal = data[start][start].real
        if diagonal < -tolerance:
            return False
        if diagonal <= tolerance:
            return all(
                abs(data[row][column]) <= tolerance for row in range(start, count) for column in range(start, count)
            )
        for row in range(start + 1, count):
            for column in range(start + 1, count):
                data[row][column] -= data[row][start] * data[start][column] / diagonal
    return True


def encode_array(values: object) -> dict[str, object]:
    """Encode a finite complex array as shape plus explicit real/imaginary pairs."""
    result = array(values)
    pending = [result.tolist()]
    flattened = []
    while pending:
        value = pending.pop()
        if isinstance(value, list):
            pending.extend(reversed(value))
        else:
            number = complex(value)
            flattened.append([number.real, number.imag])
    return {"shape": list(result.shape), "values": flattened}


def decode_array(payload: dict[str, object]) -> pc.Array:
    """Decode only the declared nonempty numeric complex-pair representation."""
    shape, values = payload["shape"], payload["values"]
    if not isinstance(shape, list) or not shape or any(type(size) is not int or size < 1 for size in shape):
        fail("serialized array shape must contain positive integer dimensions")
    if not isinstance(values, list) or len(values) != math.prod(shape):
        fail("serialized complex array size disagrees with its shape")
    if any(
        not isinstance(pair, list) or len(pair) != 2 or any(type(component) not in (int, float) for component in pair)
        for pair in values
    ):
        fail("serialized complex values must be real/imaginary numeric pairs")
    result = array([complex(*pair) for pair in values])
    return result.reshape(tuple(shape))
