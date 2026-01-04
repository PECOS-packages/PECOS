# Copyright 2024 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Surface code geometry and parity matrices.

This module provides pure geometry computations for surface codes,
including stabilizer support calculation and parity check matrices.

Layouts:
    square_lattice: Standard square lattice (d^2 data qubits)
    rotated_lattice: Rotated lattice (more efficient)

Functions:
    compute_x_stabilizer_supports: Get X stabilizer qubit indices
    compute_z_stabilizer_supports: Get Z stabilizer qubit indices
    parity_matrix_x: Generate X parity check matrix
    parity_matrix_z: Generate Z parity check matrix
"""

from pecos.qec.surface.layouts import (
    StabilizerSupport,
    compute_rotated_x_stabilizers,
    compute_rotated_z_stabilizers,
    compute_x_stabilizer_supports,
    compute_z_stabilizer_supports,
    get_rotated_logical_x,
    get_rotated_logical_z,
)
from pecos.qec.surface.parity import (
    parity_matrix_x,
    parity_matrix_z,
)
from pecos.qec.surface.patch import (
    LogicalOperator,
    PatchGeometry,
    PatchOrientation,
    Stabilizer,
    SurfacePatch,
    SurfacePatchBuilder,
)

__all__ = [
    # Data structures
    "StabilizerSupport",
    # Square lattice
    "compute_x_stabilizer_supports",
    "compute_z_stabilizer_supports",
    # Rotated lattice
    "compute_rotated_x_stabilizers",
    "compute_rotated_z_stabilizers",
    "get_rotated_logical_x",
    "get_rotated_logical_z",
    # Parity matrices
    "parity_matrix_x",
    "parity_matrix_z",
    # Patch classes
    "SurfacePatch",
    "SurfacePatchBuilder",
    "PatchGeometry",
    "PatchOrientation",
    "Stabilizer",
    "LogicalOperator",
]
