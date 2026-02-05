# Copyright 2024 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Surface code lattice layouts."""

from pecos.qec.surface.layouts.rotated_lattice import (
    RotatedPosition,
    compute_rotated_x_stabilizers,
    compute_rotated_z_stabilizers,
    generate_surface_layout,
    get_rotated_logical_x,
    get_rotated_logical_z,
    rotated_id_to_position,
    rotated_position_to_id,
)
from pecos.qec.surface.layouts.square_lattice import (
    StabilizerSupport,
    compute_x_stabilizer_supports,
    compute_z_stabilizer_supports,
    generate_nonrotated_surface_layout,
    get_boundary_stabilizer_indices,
    get_boundary_stabilizers,
    get_bulk_stabilizer_indices,
    get_bulk_stabilizers,
    get_stabilizer_counts,
)

__all__ = [
    # Rotated lattice (most common, default)
    "RotatedPosition",
    "compute_rotated_x_stabilizers",
    "compute_rotated_z_stabilizers",
    "generate_surface_layout",
    "get_rotated_logical_x",
    "get_rotated_logical_z",
    "rotated_id_to_position",
    "rotated_position_to_id",
    # Non-rotated lattice
    "StabilizerSupport",
    "compute_x_stabilizer_supports",
    "compute_z_stabilizer_supports",
    "generate_nonrotated_surface_layout",
    # Stabilizer categories
    "get_bulk_stabilizer_indices",
    "get_bulk_stabilizers",
    "get_boundary_stabilizer_indices",
    "get_boundary_stabilizers",
    "get_stabilizer_counts",
]
