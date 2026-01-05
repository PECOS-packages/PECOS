# Copyright 2024 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Magic State Distillation protocol geometry.

This module provides geometry structures for magic state distillation (MSD)
protocols using surface codes. MSD uses a two-level approach:

1. Inner code (distance-2): Fast error detection with 4 data qubits
2. Outer code (distance-3): Full surface code protection with 9 data qubits

The inner code acts as a filter - if errors are detected, the state is
discarded before expensive outer code expansion.

Qubit layout (3x3 grid):
    0  1  2
    3  4  5
    6  7  8

Inner code uses qubits {0, 1, 3, 4} (top-left 2x2).
Outer code adds qubits {2, 5, 6, 7, 8} (right column and bottom row).
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class InnerCodeGeometry:
    """Distance-2 inner code geometry for MSD.

    The inner code uses 4 data qubits in a 2x2 arrangement from the
    top-left corner of the full 3x3 grid.

    Attributes:
        data_qubits: Indices of data qubits (0, 1, 3, 4).
        z_stabilizer: Data qubit indices for the Z stabilizer.
        x_stabilizers: Data qubit indices for X stabilizers (top, bottom).
    """

    data_qubits: tuple[int, ...] = (0, 1, 3, 4)

    # Z stabilizer: measures Z on all 4 qubits (column-major order)
    z_stabilizer: tuple[int, ...] = (0, 3, 1, 4)

    # X stabilizers: top row [0,1] and bottom row [3,4]
    x_stabilizers: tuple[tuple[int, ...], ...] = ((0, 1), (3, 4))

    @property
    def num_data(self) -> int:
        """Number of data qubits."""
        return len(self.data_qubits)

    @property
    def num_x_stabilizers(self) -> int:
        """Number of X stabilizers."""
        return len(self.x_stabilizers)

    @property
    def num_z_stabilizers(self) -> int:
        """Number of Z stabilizers."""
        return 1

    @property
    def num_syndromes(self) -> int:
        """Total syndrome bits per round."""
        return self.num_x_stabilizers + self.num_z_stabilizers


@dataclass(frozen=True)
class OuterCodeGeometry:
    """Distance-3 outer code geometry for MSD.

    The outer code is a full d=3 surface code with 9 data qubits.

    Attributes:
        data_qubits: Indices of all 9 data qubits.
        inner_qubits: Indices that come from inner code (0, 1, 3, 4).
        expansion_qubits: New qubits added for outer code (2, 5, 6, 7, 8).
        x_stabilizers: X stabilizer supports (4 total: 2 bulk, 2 boundary).
        z_stabilizers: Z stabilizer supports (4 total: 2 bulk, 2 boundary).
    """

    data_qubits: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7, 8)
    inner_qubits: tuple[int, ...] = (0, 1, 3, 4)
    expansion_qubits: tuple[int, ...] = (2, 5, 6, 7, 8)

    # X stabilizers (4 total)
    # Boundary: [0,1] (top), [7,8] (bottom)
    # Bulk: [1,2,4,5], [3,4,6,7]
    x_stabilizers: tuple[tuple[int, ...], ...] = (
        (0, 1),  # boundary top
        (1, 2, 4, 5),  # bulk
        (3, 4, 6, 7),  # bulk
        (7, 8),  # boundary bottom
    )

    # Z stabilizers (4 total)
    # Boundary: [2,5] (right), [3,6] (left)
    # Bulk: [0,1,3,4] (same as inner Z), [4,5,7,8]
    z_stabilizers: tuple[tuple[int, ...], ...] = (
        (2, 5),  # boundary right
        (0, 1, 3, 4),  # bulk (inner code stabilizer)
        (4, 5, 7, 8),  # bulk
        (3, 6),  # boundary left
    )

    @property
    def num_data(self) -> int:
        """Number of data qubits."""
        return len(self.data_qubits)

    @property
    def num_expansion(self) -> int:
        """Number of qubits added during expansion."""
        return len(self.expansion_qubits)

    @property
    def num_x_stabilizers(self) -> int:
        """Number of X stabilizers."""
        return len(self.x_stabilizers)

    @property
    def num_z_stabilizers(self) -> int:
        """Number of Z stabilizers."""
        return len(self.z_stabilizers)

    @property
    def num_syndromes(self) -> int:
        """Total syndrome bits per round."""
        return self.num_x_stabilizers + self.num_z_stabilizers

    def get_bulk_x_stabilizers(self) -> tuple[tuple[int, ...], ...]:
        """Get bulk (weight-4) X stabilizers."""
        return tuple(s for s in self.x_stabilizers if len(s) == 4)

    def get_boundary_x_stabilizers(self) -> tuple[tuple[int, ...], ...]:
        """Get boundary (weight-2) X stabilizers."""
        return tuple(s for s in self.x_stabilizers if len(s) == 2)

    def get_bulk_z_stabilizers(self) -> tuple[tuple[int, ...], ...]:
        """Get bulk (weight-4) Z stabilizers."""
        return tuple(s for s in self.z_stabilizers if len(s) == 4)

    def get_boundary_z_stabilizers(self) -> tuple[tuple[int, ...], ...]:
        """Get boundary (weight-2) Z stabilizers."""
        return tuple(s for s in self.z_stabilizers if len(s) == 2)


@dataclass(frozen=True)
class MSDProtocol:
    """Magic State Distillation protocol geometry.

    Combines inner and outer code geometry with protocol parameters.

    Attributes:
        inner: Inner code geometry (distance-2).
        outer: Outer code geometry (distance-3).
        inner_rounds: Number of syndrome extraction rounds for inner code.
        outer_rounds: Number of syndrome extraction rounds for outer code.
    """

    inner: InnerCodeGeometry = field(default_factory=InnerCodeGeometry)
    outer: OuterCodeGeometry = field(default_factory=OuterCodeGeometry)
    inner_rounds: int = 2
    outer_rounds: int = 1

    @property
    def total_data_qubits(self) -> int:
        """Total data qubits needed (same as outer code)."""
        return self.outer.num_data

    @property
    def inner_syndrome_bits(self) -> int:
        """Syndrome bits per inner code round."""
        return self.inner.num_syndromes

    @property
    def outer_syndrome_bits(self) -> int:
        """Syndrome bits per outer code round."""
        return self.outer.num_syndromes

    @property
    def total_inner_syndromes(self) -> int:
        """Total syndrome bits from inner code (all rounds)."""
        return self.inner.num_syndromes * self.inner_rounds

    @property
    def total_outer_syndromes(self) -> int:
        """Total syndrome bits from outer code (all rounds)."""
        return self.outer.num_syndromes * self.outer_rounds

    def get_expansion_prep_states(self) -> dict[int, str]:
        """Get preparation states for expansion qubits.

        When expanding from inner to outer code:
        - Bottom row qubits (6, 7, 8) are prepared in |+>
        - Right column qubits (2, 5) are prepared in |0>

        Returns:
            Dict mapping qubit index to preparation state ('+' or '0').
        """
        return {
            2: "0",  # Right column: |0>
            5: "0",  # Right column: |0>
            6: "+",  # Bottom row: |+>
            7: "+",  # Bottom row: |+>
            8: "+",  # Bottom row: |+>
        }

    def get_inner_init_states(self) -> dict[int, str]:
        """Get initialization states for inner code qubits.

        For T state distillation:
        - Qubit 0: T|+> (T gate applied to |+>)
        - Qubit 1: |0>
        - Qubits 3, 4: |+>

        Returns:
            Dict mapping qubit index to preparation state.
        """
        return {
            0: "T+",  # T|+> magic state
            1: "0",  # |0>
            3: "+",  # |+>
            4: "+",  # |+>
        }


def create_msd_protocol(
    inner_rounds: int = 2,
    outer_rounds: int = 1,
) -> MSDProtocol:
    """Create an MSD protocol with specified parameters.

    Args:
        inner_rounds: Number of syndrome extraction rounds for inner code.
            Default is 2 (two rounds, check for consistency).
        outer_rounds: Number of syndrome extraction rounds for outer code.
            Default is 1.

    Returns:
        Configured MSDProtocol instance.
    """
    return MSDProtocol(
        inner=InnerCodeGeometry(),
        outer=OuterCodeGeometry(),
        inner_rounds=inner_rounds,
        outer_rounds=outer_rounds,
    )
