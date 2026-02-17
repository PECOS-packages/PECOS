# Copyright 2024 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Standard square lattice geometry for surface codes.

Qubit layout for distance d (example d=3):
    0  1  2
    3  4  5
    6  7  8

X stabilizers measure in checkerboard pattern with H-CNOT-H.
Z stabilizers measure in checkerboard pattern with CNOT.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class StabilizerSupport:
    """Definition of a single stabilizer.

    Attributes:
        index: Stabilizer index in syndrome array
        data_qubits: Data qubit indices this stabilizer acts on
        is_boundary: True if weight-2 boundary stabilizer
    """

    index: int
    data_qubits: tuple[int, ...]
    is_boundary: bool

    @property
    def weight(self) -> int:
        """Number of data qubits in this stabilizer."""
        return len(self.data_qubits)


def compute_x_stabilizer_supports(d: int) -> list[StabilizerSupport]:
    """Compute data qubit indices for each X stabilizer.

    X stabilizers use the H-CNOT-H pattern where the ancilla controls
    CNOTs to data qubits.

    Args:
        d: Code distance (must be odd >= 3)

    Returns:
        List of StabilizerSupport objects, ordered by stabilizer index.
    """
    num_stab = (d**2 - 1) // 2
    num_bound = d - 1
    start_bulk = num_bound // 2
    end_bulk = num_stab - start_bulk

    supports: list[StabilizerSupport] = []

    # Bulk stabilizers (weight 4)
    j = 1
    for i in range(start_bulk, end_bulk):
        if j + d + 1 > d**2 - 1:
            break
        supports.append(
            StabilizerSupport(
                index=i,
                data_qubits=(j, j + 1, j + d, j + d + 1),
                is_boundary=False,
            ),
        )
        if i % (d - 1) == num_bound // 2 - 1:
            j += 4
        else:
            j += 2

    # Top boundary stabilizers (weight 2)
    j = 0
    for i in range(num_bound // 2):
        supports.append(
            StabilizerSupport(
                index=i,
                data_qubits=(j, j + 1),
                is_boundary=True,
            ),
        )
        j += 2

    # Bottom boundary stabilizers (weight 2)
    j = (d - 1) * d + 1
    for i in range(num_stab - num_bound // 2, num_stab):
        supports.append(
            StabilizerSupport(
                index=i,
                data_qubits=(j, j + 1),
                is_boundary=True,
            ),
        )
        j += 2

    supports.sort(key=lambda s: s.index)
    return supports


def compute_z_stabilizer_supports(d: int) -> list[StabilizerSupport]:
    """Compute data qubit indices for each Z stabilizer.

    Z stabilizers use direct CNOTs from data qubits to ancilla.

    Args:
        d: Code distance (must be odd >= 3)

    Returns:
        List of StabilizerSupport objects, ordered by stabilizer index.
    """
    num_stab = (d**2 - 1) // 2
    num_bound = d - 1
    start_bulk = num_bound // 2
    end_bulk = num_stab - start_bulk

    supports: list[StabilizerSupport] = []

    # Bulk stabilizers (weight 4)
    j = 2 * d - 2
    for i in range(start_bulk, end_bulk):
        supports.append(
            StabilizerSupport(
                index=i,
                data_qubits=(j, j + d, j + 1, j + d + 1),
                is_boundary=False,
            ),
        )
        if i % (d - 1) == num_bound // 2 - 1:
            j += 2 * d
            j = j % d - 1 + d
        else:
            j += 2 * d
        if j >= d**2:
            j = (j % d) - 1

    # Right boundary stabilizers (weight 2)
    j = 2 * d - 1
    for i in range(num_bound // 2):
        k = j - d
        supports.append(
            StabilizerSupport(
                index=i,
                data_qubits=(k, j),
                is_boundary=True,
            ),
        )
        j += 2 * d

    # Left boundary stabilizers (weight 2)
    j = d
    for i in range(num_stab - num_bound // 2, num_stab):
        k = j + d
        supports.append(
            StabilizerSupport(
                index=i,
                data_qubits=(j, k),
                is_boundary=True,
            ),
        )
        j += 2 * d

    supports.sort(key=lambda s: s.index)
    return supports


def get_stabilizer_counts(d: int) -> tuple[int, int, int]:
    """Get stabilizer count breakdown for a surface code.

    Args:
        d: Code distance (must be odd >= 3).

    Returns:
        Tuple of (total_per_basis, num_bulk, num_boundary) where:
        - total_per_basis: Total stabilizers per basis (X or Z)
        - num_bulk: Number of weight-4 bulk stabilizers
        - num_boundary: Number of weight-2 boundary stabilizers
    """
    total = (d**2 - 1) // 2
    num_boundary = d - 1
    num_bulk = total - num_boundary
    return total, num_bulk, num_boundary


def get_bulk_stabilizer_indices(d: int) -> list[int]:
    """Get indices of bulk (weight-4) stabilizers.

    Bulk stabilizers are in the interior of the lattice and have
    weight 4 (act on 4 data qubits).

    Args:
        d: Code distance (must be odd >= 3).

    Returns:
        List of stabilizer indices for bulk stabilizers.
    """
    num_boundary = d - 1
    start_bulk = num_boundary // 2
    total = (d**2 - 1) // 2
    end_bulk = total - start_bulk
    return list(range(start_bulk, end_bulk))


def get_boundary_stabilizer_indices(d: int) -> list[int]:
    """Get indices of boundary (weight-2) stabilizers.

    Boundary stabilizers are on the edges of the lattice and have
    weight 2 (act on 2 data qubits).

    Args:
        d: Code distance (must be odd >= 3).

    Returns:
        List of stabilizer indices for boundary stabilizers.
    """
    num_boundary = d - 1
    start_bulk = num_boundary // 2
    total = (d**2 - 1) // 2
    end_bulk = total - start_bulk

    # Boundary indices are before start_bulk and after end_bulk
    return list(range(start_bulk)) + list(range(end_bulk, total))


def get_bulk_stabilizers(d: int, stab_type: str = "X") -> list[StabilizerSupport]:
    """Get bulk stabilizers for the specified type.

    Args:
        d: Code distance (must be odd >= 3).
        stab_type: "X" or "Z".

    Returns:
        List of bulk StabilizerSupport objects.
    """
    supports = compute_x_stabilizer_supports(d) if stab_type == "X" else compute_z_stabilizer_supports(d)
    return [s for s in supports if not s.is_boundary]


def get_boundary_stabilizers(d: int, stab_type: str = "X") -> list[StabilizerSupport]:
    """Get boundary stabilizers for the specified type.

    Args:
        d: Code distance (must be odd >= 3).
        stab_type: "X" or "Z".

    Returns:
        List of boundary StabilizerSupport objects.
    """
    supports = compute_x_stabilizer_supports(d) if stab_type == "X" else compute_z_stabilizer_supports(d)
    return [s for s in supports if s.is_boundary]


def generate_nonrotated_surface_layout(
    width: int,
    height: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[list[tuple[int, int]]]]:
    """Generate non-rotated surface code layout.

    The non-rotated surface code has data qubits on a checkerboard pattern
    with more physical qubits than the rotated variant for the same distance.
    For the common rotated layout, use generate_surface_layout() instead.

    Args:
        width: Width of the patch (code distance in Z direction).
        height: Height of the patch (code distance in X direction).

    Returns:
        A tuple containing:
        - data_positions: List of (x, y) coordinates for data qubits.
        - ancilla_positions: List of (x, y) coordinates for ancilla qubits.
        - polygons: List of polygons representing stabilizer checks.
    """
    lattice_height = 2 * (height - 1)
    lattice_width = 2 * (width - 1)

    data_positions: list[tuple[int, int]] = []
    ancilla_positions: list[tuple[int, int]] = []
    polygons: list[list[tuple[int, int]]] = []

    for y in range(lattice_height + 1):
        for x in range(lattice_width + 1):
            if (x % 2 == 0 and y % 2 == 0) or (x % 2 == 1 and y % 2 == 1):
                # Data qubit
                data_positions.append((x, y))

            elif x % 2 == 1 and y % 2 == 0:
                # X ancilla
                ancilla_positions.append((x, y))

                poly: list[tuple[int, int]] = []
                if y != lattice_height:
                    poly.append((x, y + 1))
                if x != 0:
                    poly.append((x - 1, y))
                if y != 0:
                    poly.append((x, y - 1))
                if x != lattice_width:
                    poly.append((x + 1, y))
                polygons.append(poly)

            elif x % 2 == 0 and y % 2 == 1:
                # Z ancilla
                ancilla_positions.append((x, y))

                poly = []
                if y != lattice_height:
                    poly.append((x, y + 1))
                if x != 0:
                    poly.append((x - 1, y))
                if y != 0:
                    poly.append((x, y - 1))
                if x != lattice_width:
                    poly.append((x + 1, y))
                polygons.append(poly)

    return data_positions, ancilla_positions, polygons
