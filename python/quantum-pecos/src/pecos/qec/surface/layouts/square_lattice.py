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
    n_stab = (d**2 - 1) // 2
    n_bound = d - 1
    start_bulk = n_bound // 2
    end_bulk = n_stab - start_bulk

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
            )
        )
        if i % (d - 1) == n_bound // 2 - 1:
            j += 4
        else:
            j += 2

    # Top boundary stabilizers (weight 2)
    j = 0
    for i in range(n_bound // 2):
        supports.append(
            StabilizerSupport(
                index=i,
                data_qubits=(j, j + 1),
                is_boundary=True,
            )
        )
        j += 2

    # Bottom boundary stabilizers (weight 2)
    j = (d - 1) * d + 1
    for i in range(n_stab - n_bound // 2, n_stab):
        supports.append(
            StabilizerSupport(
                index=i,
                data_qubits=(j, j + 1),
                is_boundary=True,
            )
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
    n_stab = (d**2 - 1) // 2
    n_bound = d - 1
    start_bulk = n_bound // 2
    end_bulk = n_stab - start_bulk

    supports: list[StabilizerSupport] = []

    # Bulk stabilizers (weight 4)
    j = 2 * d - 2
    for i in range(start_bulk, end_bulk):
        supports.append(
            StabilizerSupport(
                index=i,
                data_qubits=(j, j + d, j + 1, j + d + 1),
                is_boundary=False,
            )
        )
        if i % (d - 1) == n_bound // 2 - 1:
            j += 2 * d
            j = j % d - 1 + d
        else:
            j += 2 * d
        if j >= d**2:
            j = (j % d) - 1

    # Right boundary stabilizers (weight 2)
    j = 2 * d - 1
    for i in range(n_bound // 2):
        k = j - d
        supports.append(
            StabilizerSupport(
                index=i,
                data_qubits=(k, j),
                is_boundary=True,
            )
        )
        j += 2 * d

    # Left boundary stabilizers (weight 2)
    j = d
    for i in range(n_stab - n_bound // 2, n_stab):
        k = j + d
        supports.append(
            StabilizerSupport(
                index=i,
                data_qubits=(j, k),
                is_boundary=True,
            )
        )
        j += 2 * d

    supports.sort(key=lambda s: s.index)
    return supports
