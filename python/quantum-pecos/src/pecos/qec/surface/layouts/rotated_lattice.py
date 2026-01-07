# Copyright 2024 The PECOS Developers
# Licensed under the Apache License, Version 2.0

r"""Rotated surface code lattice geometry.

The rotated surface code arranges data qubits on a diagonal lattice,
requiring fewer physical qubits for the same code distance compared
to the standard (non-rotated) layout.

Qubit layout for d=3 rotated code:
        0
       / \\
      1   2
     / \\ / \\
    3   4   5
     \\ / \\ /
      6   7
       \\ /
        8
"""

from dataclasses import dataclass

from pecos.qec.surface.layouts.square_lattice import StabilizerSupport


@dataclass(frozen=True)
class RotatedPosition:
    """Position in the rotated lattice coordinate system."""

    x: int
    y: int


def compute_rotated_x_stabilizers(d: int) -> list[StabilizerSupport]:
    """Compute X stabilizer supports for rotated surface code.

    Args:
        d: Code distance (must be odd >= 3)

    Returns:
        List of StabilizerSupport for X stabilizers
    """
    if d < 3 or d % 2 == 0:
        msg = f"Distance must be odd >= 3, got {d}"
        raise ValueError(msg)

    supports = []
    stab_idx = 0

    # Bulk X stabilizers (weight 4)
    for row in range(d - 1):
        for col in range(d - 1):
            if (row + col) % 2 == 0:
                q_tl = row * d + col
                q_tr = row * d + col + 1
                q_bl = (row + 1) * d + col
                q_br = (row + 1) * d + col + 1

                supports.append(
                    StabilizerSupport(
                        index=stab_idx,
                        data_qubits=(q_tl, q_tr, q_bl, q_br),
                        is_boundary=False,
                    ),
                )
                stab_idx += 1

    # Boundary X stabilizers (weight 2) - top and bottom edges
    for col in range(0, d - 1, 2):
        q1 = col
        q2 = col + 1
        supports.append(
            StabilizerSupport(
                index=stab_idx,
                data_qubits=(q1, q2),
                is_boundary=True,
            ),
        )
        stab_idx += 1

    for col in range((d - 1) % 2, d - 1, 2):
        q1 = (d - 1) * d + col
        q2 = (d - 1) * d + col + 1
        supports.append(
            StabilizerSupport(
                index=stab_idx,
                data_qubits=(q1, q2),
                is_boundary=True,
            ),
        )
        stab_idx += 1

    supports.sort(key=lambda s: s.index)
    return supports


def compute_rotated_z_stabilizers(d: int) -> list[StabilizerSupport]:
    """Compute Z stabilizer supports for rotated surface code.

    Args:
        d: Code distance (must be odd >= 3)

    Returns:
        List of StabilizerSupport for Z stabilizers
    """
    if d < 3 or d % 2 == 0:
        msg = f"Distance must be odd >= 3, got {d}"
        raise ValueError(msg)

    supports = []
    stab_idx = 0

    # Bulk Z stabilizers (weight 4)
    for row in range(d - 1):
        for col in range(d - 1):
            if (row + col) % 2 == 1:
                q_tl = row * d + col
                q_tr = row * d + col + 1
                q_bl = (row + 1) * d + col
                q_br = (row + 1) * d + col + 1

                supports.append(
                    StabilizerSupport(
                        index=stab_idx,
                        data_qubits=(q_tl, q_tr, q_bl, q_br),
                        is_boundary=False,
                    ),
                )
                stab_idx += 1

    # Boundary Z stabilizers (weight 2) - left and right edges
    for row in range(0, d - 1, 2):
        q1 = row * d
        q2 = (row + 1) * d
        supports.append(
            StabilizerSupport(
                index=stab_idx,
                data_qubits=(q1, q2),
                is_boundary=True,
            ),
        )
        stab_idx += 1

    for row in range((d - 1) % 2, d - 1, 2):
        q1 = row * d + (d - 1)
        q2 = (row + 1) * d + (d - 1)
        supports.append(
            StabilizerSupport(
                index=stab_idx,
                data_qubits=(q1, q2),
                is_boundary=True,
            ),
        )
        stab_idx += 1

    supports.sort(key=lambda s: s.index)
    return supports


def get_rotated_logical_x(d: int) -> tuple[int, ...]:
    """Get logical X operator qubits (left edge)."""
    return tuple(i * d for i in range(d))


def get_rotated_logical_z(d: int) -> tuple[int, ...]:
    """Get logical Z operator qubits (top edge)."""
    return tuple(range(d))


def rotated_id_to_position(qubit_id: int, d: int) -> tuple[int, int]:
    """Convert qubit ID to (x, y) position in rotated coordinates."""
    row = qubit_id // d
    col = qubit_id % d
    x = col * 2 + 1
    y = (d - row) * 2 - 1
    return (x, y)


def rotated_position_to_id(x: int, y: int, d: int) -> int:
    """Convert rotated position to qubit ID."""
    col = (x - 1) // 2
    row = d - (y + 1) // 2
    return row * d + col


def generate_surface_layout(
    width: int,
    height: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Generate rotated surface code layout positions.

    This is the most common surface code variant, using d^2 data qubits for
    distance d. The non-rotated variant uses more qubits; for that, use
    generate_nonrotated_surface_layout() instead.

    The rotated surface code places data qubits at odd-odd positions
    and ancilla qubits at even-even positions in the interior, with
    boundary ancillas on the edges.

    Args:
        width: Width of the patch (code distance in Z direction).
        height: Height of the patch (code distance in X direction).

    Returns:
        A tuple containing:
        - data_positions: List of (x, y) coordinates for data qubits.
        - ancilla_positions: List of (x, y) coordinates for ancilla qubits.
    """
    lattice_height = 2 * height
    lattice_width = 2 * width

    data_positions: list[tuple[int, int]] = []
    ancilla_positions: list[tuple[int, int]] = []

    for y in range(lattice_height + 1):
        for x in range(lattice_width + 1):
            if 0 < x < lattice_width and 0 < y < lattice_height:
                # Interior (no boundary stabilizers)
                if x % 2 == 1 and y % 2 == 1:
                    # Data qubit at odd-odd positions
                    data_positions.append((x, y))
                elif x % 2 == 0 and y % 2 == 0:
                    # Ancilla at even-even positions
                    ancilla_positions.append((x, y))

            elif 0 < x < lattice_width or 0 < y < lattice_height:
                # Boundary ancillas (not corners)
                if y == 0:
                    # Top edge: X stabilizers
                    if x != 0 and x % 4 == 0:
                        ancilla_positions.append((x, y))

                elif x == 0 and (y - 2) % 4 == 0:
                    # Left edge
                    ancilla_positions.append((x, y))

                if y == lattice_height:
                    # Bottom edge
                    if height % 2 == 0:
                        if x != 0 and x % 4 == 0:
                            ancilla_positions.append((x, y))
                    elif (x - 2) % 4 == 0:
                        ancilla_positions.append((x, y))

                elif x == lattice_width:
                    # Right edge
                    if width % 2 == 1:
                        if y != 0 and y % 4 == 0:
                            ancilla_positions.append((x, y))
                    elif (y - 2) % 4 == 0:
                        ancilla_positions.append((x, y))

    return data_positions, ancilla_positions
