# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""CNOT schedule for parallel surface code syndrome extraction.

Computes a 4-round windmill (N/Z) CNOT schedule for the rotated
surface code. Each round contains a set of CX gates that can be
executed simultaneously without conflicts (no data qubit is touched
twice in the same round).

Bulk stabilizer data_qubits ordering: (TL, TR, BL, BR)

Windmill pattern:
    Round | X stab touches | Z stab touches
    1     | TR (idx 1)     | TR (idx 1)
    2     | TL (idx 0)     | BR (idx 3)
    3     | BR (idx 3)     | TL (idx 0)
    4     | BL (idx 2)     | BL (idx 2)

Boundary stabilizers are scheduled in pairs across two consecutive
rounds depending on which boundary they sit on:
    X boundaries: right first, then left
    Z boundaries: top first, then bottom
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pecos.qec.surface.patch import SurfacePatch


def _classify_boundary(stab_type: str, data_qubits: tuple[int, ...], d: int) -> str:
    """Classify which boundary a weight-2 stabilizer sits on.

    Returns one of: 'top', 'bottom', 'left', 'right'.
    """
    rows = [q // d for q in data_qubits]
    cols = [q % d for q in data_qubits]

    if stab_type == "X":
        # X boundaries are top and bottom
        if all(r == 0 for r in rows):
            return "top"
        if all(r == d - 1 for r in rows):
            return "bottom"
    else:
        # Z boundaries are left and right
        if all(c == d - 1 for c in cols):
            return "right"
        if all(c == 0 for c in cols):
            return "left"

    msg = f"Cannot classify boundary for {stab_type} stab with qubits {data_qubits} (d={d})"
    raise ValueError(msg)


def get_stab_schedule(
    stab_type: str,
    data_qubits: tuple[int, ...],
    is_boundary: bool,
    d: int,
) -> list[tuple[int, int]]:
    """Compute the per-stabilizer CNOT schedule.

    Args:
        stab_type: 'X' or 'Z'
        data_qubits: Tuple of data qubit IDs. For bulk (weight 4):
            (TL, TR, BL, BR). For boundary (weight 2): two qubits.
        is_boundary: Whether this is a boundary stabilizer.
        d: Code distance.

    Returns:
        List of (round_0based, data_qubit) pairs, sorted by round.
    """
    if not is_boundary:
        # Bulk weight-4 stabilizer
        tl, tr, bl, br = data_qubits
        if stab_type == "X":
            return [(0, tr), (1, tl), (2, br), (3, bl)]
        return [(0, tr), (1, br), (2, tl), (3, bl)]

    # Boundary weight-2 stabilizer
    boundary = _classify_boundary(stab_type, data_qubits, d)

    if boundary == "bottom":
        # Bottom X: rounds 0,1 -- right first then left
        q_left, q_right = data_qubits
        return [(0, q_right), (1, q_left)]
    if boundary == "top":
        # Top X: rounds 2,3 -- right first then left
        q_left, q_right = data_qubits
        return [(2, q_right), (3, q_left)]
    if boundary == "left":
        # Left Z: rounds 0,1 -- top first then bottom
        q_top, q_bottom = data_qubits
        return [(0, q_top), (1, q_bottom)]
    if boundary == "right":
        # Right Z: rounds 2,3 -- top first then bottom
        q_top, q_bottom = data_qubits
        return [(2, q_top), (3, q_bottom)]

    msg = f"Unknown boundary: {boundary}"
    raise ValueError(msg)


def compute_cnot_schedule(patch: SurfacePatch) -> list[list[tuple[str, int, int]]]:
    """Compute the 4-round parallel CNOT schedule for a surface code patch.

    Args:
        patch: A SurfacePatch instance.

    Returns:
        List of 4 rounds, each a list of (stab_type, stab_index, data_qubit)
        tuples representing CX gates to execute in parallel.
    """
    d = patch.distance
    rounds: list[list[tuple[str, int, int]]] = [[] for _ in range(4)]

    for stab in patch.x_stabilizers:
        schedule = get_stab_schedule("X", stab.data_qubits, stab.is_boundary, d)
        for rnd, data_q in schedule:
            rounds[rnd].append(("X", stab.index, data_q))

    for stab in patch.z_stabilizers:
        schedule = get_stab_schedule("Z", stab.data_qubits, stab.is_boundary, d)
        for rnd, data_q in schedule:
            rounds[rnd].append(("Z", stab.index, data_q))

    # Sort each round by (stab_index, X before Z) so gates are interleaved
    # by stabilizer index with X gates preceding Z gates at the same index.
    for rnd in rounds:
        rnd.sort(key=lambda g: (g[1], 0 if g[0] == "X" else 1))

    return rounds
