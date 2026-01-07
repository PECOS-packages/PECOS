# Copyright 2024 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""4.8.8 Triangular Color Code geometry and layout.

The 4.8.8 color code is a topological quantum error correction code
where qubits are arranged on a triangular lattice with stabilizers
that are colored red, green, and blue.

Key properties:
- Three colors: red, green, blue
- Red stabilizers are weight-4 (squares)
- Green/Blue stabilizers are weight-4 or weight-8 (octagons and boundaries)
- Transversal H, S, and CNOT gates
- Distance d requires O(d^2) qubits
"""

from typing import Any


def generate_488_layout(
    distance: int,
) -> tuple[dict[int, tuple[int, int]], list[list[Any]]]:
    """Generate the 4.8.8 color code layout.

    Creates a triangular lattice with the specified code distance.

    Args:
        distance: Code distance (must be odd >= 3)

    Returns:
        Tuple of (qubit positions dict, polygon list)
        Each polygon is [qubit_id, ..., color_string]
    """
    if distance < 3 or distance % 2 == 0:
        msg = f"Distance must be odd >= 3, got {distance}"
        raise ValueError(msg)

    lattice_height = 4 * distance - 4
    lattice_width = 2 * distance - 2
    pos_qubits = []
    pos_checks = []

    for y in range(lattice_width + 1):
        for x in range(lattice_height + 1):
            # Skip positions outside the triangular region
            if ((x, y) == (x, x + 2) and x % 2 == 1 and y % 8 == 3) or (
                (x, y) == (4 * distance - y, y) and x % 2 == 1 and y % 8 == 7
            ):
                pass
            elif (x, y) > (x, x) or (x, y) > (4 * distance - y - 2, y):
                continue

            # Place data qubits
            if x % 2 == 0 and y % 2 == 0:
                if (y / 2) % 4 == 1 or (y / 2) % 4 == 2:
                    if (x / 2) % 4 == 2 or (x / 2) % 4 == 3:
                        pos_qubits.append((x, y))
                elif (x / 2) % 4 == 0 or (x / 2) % 4 == 1:
                    pos_qubits.append((x, y))

            # Place check positions
            if x % 4 == 1 and y % 4 == 3:
                pos_checks.append((x, y))

            if y == 0 and x % 8 == 5:
                pos_checks.append((x, y))

    # Sort positions for consistent indexing
    pos_qubits = sorted(pos_qubits, key=lambda point: (-point[1], point[0]))
    pos_checks = sorted(pos_checks, key=lambda point: (-point[1], point[0]))

    # Create position mappings
    nodeid2pos = {i: pos_qubits[i] for i in range(len(pos_qubits))}
    pos2nodeid = {v: k for k, v in nodeid2pos.items()}

    # Find polygons (stabilizers)
    polygons = []
    for x, y in pos_checks:
        if square := _find_square(x, y, pos2nodeid):
            polygons.append(square)
        elif y == 0 and (gon := _find_bottom_polygon(x, y, pos2nodeid)):
            polygons.append(gon)
        elif octo := _find_octagon(x, y, pos2nodeid):
            polygons.append(octo)

    return nodeid2pos, polygons


def _find_square(
    x: int,
    y: int,
    pos2nodeid: dict[tuple[int, int], int],
) -> list[Any] | None:
    """Find a square (weight-4) stabilizer at position."""
    square_coords = [(x - 1, y + 1), (x - 1, y - 1), (x + 1, y - 1), (x + 1, y + 1)]
    square_ids = []

    for coord in square_coords:
        nid = pos2nodeid.get(coord)
        if nid is None:
            return None
        square_ids.append(nid)

    square_ids.append("red")
    return square_ids


def _find_octagon(
    x: int,
    y: int,
    pos2nodeid: dict[tuple[int, int], int],
) -> list[Any] | None:
    """Find an octagon (weight-8) stabilizer at position."""
    octagon_coords = [
        (x - 1, y + 3),
        (x - 3, y + 1),
        (x - 3, y - 1),
        (x - 1, y - 3),
        (x + 1, y - 3),
        (x + 3, y - 1),
        (x + 3, y + 1),
        (x + 1, y + 3),
    ]
    octo_ids = []

    for coord in octagon_coords:
        nid = pos2nodeid.get(coord)
        if nid is not None:
            octo_ids.append(nid)

    if not octo_ids:
        return None

    # Determine color based on position
    if (x - 1) // 4 % 2:
        octo_ids.append("green")
    else:
        octo_ids.append("blue")

    return octo_ids


def _find_bottom_polygon(
    x: int,
    y: int,
    pos2nodeid: dict[tuple[int, int], int],
) -> list[Any] | None:
    """Find a bottom boundary stabilizer."""
    coords = [
        (x - 1, y + 2),
        (x - 3, y),
        (x + 3, y),
        (x + 1, y + 2),
    ]
    found_ids = []

    for coord in coords:
        nid = pos2nodeid.get(coord)
        if nid is None:
            return None
        found_ids.append(nid)

    if (x - 1) // 4 % 2:
        found_ids.append("green")
    else:
        found_ids.append("blue")

    return found_ids
