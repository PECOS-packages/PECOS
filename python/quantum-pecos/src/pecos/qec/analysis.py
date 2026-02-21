# Copyright 2024 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""QEC result analysis and post-processing utilities.

This module provides utilities for analyzing quantum error correction
results, including logical operator extraction, fidelity calculation,
and syndrome processing.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


def logical_x_from_data(d: int, data: Sequence[int]) -> int:
    """Extract logical X value from measurement data.

    For a surface code, the logical X operator is supported on the
    left column of data qubits (indices 0, d, 2d, ...).

    Args:
        d: Code distance.
        data: Measurement outcomes for d^2 data qubits.

    Returns:
        Logical X value (0 or 1) computed as XOR of left column.
    """
    if len(data) != d * d:
        msg = f"Expected {d*d} data qubits, got {len(data)}"
        raise ValueError(msg)

    result = 0
    for i in range(d):
        result ^= data[i * d]
    return result


def logical_z_from_data(d: int, data: Sequence[int]) -> int:
    """Extract logical Z value from measurement data.

    For a surface code, the logical Z operator is supported on the
    top row of data qubits (indices 0, 1, 2, ..., d-1).

    Args:
        d: Code distance.
        data: Measurement outcomes for d^2 data qubits.

    Returns:
        Logical Z value (0 or 1) computed as XOR of top row.
    """
    if len(data) != d * d:
        msg = f"Expected {d*d} data qubits, got {len(data)}"
        raise ValueError(msg)

    result = 0
    for i in range(d):
        result ^= data[i]
    return result


def logical_from_data(d: int, data: Sequence[int]) -> tuple[int, int]:
    """Extract both logical X and Z values from measurement data.

    Args:
        d: Code distance.
        data: Measurement outcomes for d^2 data qubits.

    Returns:
        Tuple of (logical_x, logical_z) values.
    """
    return logical_x_from_data(d, data), logical_z_from_data(d, data)


def logical_fidelity(
    outcomes: Sequence[Sequence[int]],
    d: int,
    basis: int,
    expected: int = 0,
) -> tuple[float, float]:
    """Compute logical fidelity from multiple measurement outcomes.

    Calculates the fraction of shots where the logical measurement
    matches the expected value, with binomial error bars.

    Args:
        outcomes: List of measurement outcomes, each with d^2 data qubits.
        d: Code distance.
        basis: Measurement basis (0=X, 1=Z).
        expected: Expected logical value (default 0).

    Returns:
        Tuple of (fidelity, error) where error is the standard deviation
        assuming binomial statistics: sqrt(f * (1-f) / num_shots).
    """
    if not outcomes:
        msg = "No outcomes provided"
        raise ValueError(msg)

    num_shots = len(outcomes)
    successes = 0

    for data in outcomes:
        logical = logical_x_from_data(d, data) if basis == 0 else logical_z_from_data(d, data)

        if logical == expected:
            successes += 1

    fidelity = successes / num_shots

    # Binomial error bar
    error = 0.0 if fidelity == 0.0 or fidelity == 1.0 else math.sqrt(fidelity * (1 - fidelity) / num_shots)

    return fidelity, error


def syndrome_difference(
    syndromes: Sequence[Sequence[int]],
) -> list[list[int]]:
    """Compute syndrome differences between consecutive rounds.

    For MWPM decoding, we need the syndrome changes (differences)
    between rounds rather than the raw syndromes. The first round
    is compared against all-zeros (initialization).

    Args:
        syndromes: List of syndrome measurements, one per round.
            Each syndrome is a sequence of stabilizer measurement results.

    Returns:
        List of syndrome differences. The i-th difference is
        syndromes[i] XOR syndromes[i-1] (with syndromes[-1] = 0).
    """
    if not syndromes:
        return []

    num_stab = len(syndromes[0])
    result = []

    # First round: difference from all-zeros
    result.append(list(syndromes[0]))

    # Subsequent rounds: difference from previous
    for i in range(1, len(syndromes)):
        diff = [syndromes[i][j] ^ syndromes[i - 1][j] for j in range(num_stab)]
        result.append(diff)

    return result


def syndrome_to_detection_events(
    syndromes: Sequence[Sequence[int]],
) -> list[tuple[int, int]]:
    """Convert syndrome history to detection event coordinates.

    Detection events are (stabilizer_index, round) pairs where
    the syndrome changed (difference is 1).

    Args:
        syndromes: List of syndrome measurements, one per round.

    Returns:
        List of (stabilizer_index, round) tuples for each detection event.
    """
    differences = syndrome_difference(syndromes)
    events = []

    for t, diff in enumerate(differences):
        for s, val in enumerate(diff):
            if val == 1:
                events.append((s, t))

    return events


def logical_error_rate(
    outcomes: Sequence[Sequence[int]],
    d: int,
    basis: int,
    expected: int = 0,
) -> tuple[float, float]:
    """Compute logical error rate from measurement outcomes.

    This is simply 1 - fidelity, with propagated error bars.

    Args:
        outcomes: List of measurement outcomes, each with d^2 data qubits.
        d: Code distance.
        basis: Measurement basis (0=X, 1=Z).
        expected: Expected logical value (default 0).

    Returns:
        Tuple of (error_rate, error_bar).
    """
    fidelity, error = logical_fidelity(outcomes, d, basis, expected)
    return 1 - fidelity, error


def lower_bound_fidelity(f1: float, f2: float) -> float:
    """Compute lower bound on true fidelity from two measurements.

    Uses the formula: bound = (4/5) * (f1 + f2) - (3/5)

    This provides a statistical lower bound accounting for
    measurement correlations when measuring in two different bases.

    Args:
        f1: Fidelity from first measurement basis.
        f2: Fidelity from second measurement basis.

    Returns:
        Lower bound on true fidelity.
    """
    return (4 / 5) * (f1 + f2) - (3 / 5)
