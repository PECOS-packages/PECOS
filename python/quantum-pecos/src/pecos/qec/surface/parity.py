# Copyright 2024 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Parity check matrix generation for surface codes."""

import pecos
from pecos.qec.surface.layouts import (
    compute_x_stabilizer_supports,
    compute_z_stabilizer_supports,
)


def parity_matrix_x(d: int) -> pecos.Array:
    """Generate X stabilizer parity check matrix.

    Args:
        d: Code distance

    Returns:
        Binary matrix of shape (num_stab, num_data) where entry (i,j)=1
        if stabilizer i acts on qubit j
    """
    num_data = d * d
    stabs = compute_x_stabilizer_supports(d)
    num_stab = len(stabs)

    matrix = pecos.zeros((num_stab, num_data), dtype="int64")
    for stab in stabs:
        for q in stab.data_qubits:
            matrix[stab.index, q] = 1

    return matrix


def parity_matrix_z(d: int) -> pecos.Array:
    """Generate Z stabilizer parity check matrix.

    Args:
        d: Code distance

    Returns:
        Binary matrix of shape (num_stab, num_data)
    """
    num_data = d * d
    stabs = compute_z_stabilizer_supports(d)
    num_stab = len(stabs)

    matrix = pecos.zeros((num_stab, num_data), dtype="int64")
    for stab in stabs:
        for q in stab.data_qubits:
            matrix[stab.index, q] = 1

    return matrix
