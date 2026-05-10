"""Quantum-information channel representations and measures.

This module re-exports the Rust-backed implementations from
``pecos_rslib.quantum_info``. Computation and validation happen in Rust; this
file only provides the public Python import location.
"""

from __future__ import annotations

from pecos_rslib.quantum_info import (
    ChiMatrix,
    ChoiMatrix,
    KrausOps,
    PauliChannel,
    ProcessTomographyDesign,
    Ptm,
    Stinespring,
    SuperOp,
    average_gate_fidelity,
    entropy,
    gate_error,
    hellinger_distance,
    hellinger_fidelity,
    logarithmic_negativity,
    matrix_unit_basis,
    negativity,
    partial_trace_qubits,
    partial_trace_subsystems,
    pauli_channel_diamond_distance,
    pauli_channel_diamond_norm,
    process_fidelity,
    purity,
    random_density_matrix,
    random_quantum_channel,
    schmidt_decomposition,
    shannon_entropy,
    state_fidelity,
    state_fidelity_with_density_matrix,
)

__all__ = [
    "ChiMatrix",
    "ChoiMatrix",
    "KrausOps",
    "PauliChannel",
    "ProcessTomographyDesign",
    "Ptm",
    "Stinespring",
    "SuperOp",
    "average_gate_fidelity",
    "entropy",
    "gate_error",
    "hellinger_distance",
    "hellinger_fidelity",
    "logarithmic_negativity",
    "matrix_unit_basis",
    "negativity",
    "partial_trace_qubits",
    "partial_trace_subsystems",
    "pauli_channel_diamond_distance",
    "pauli_channel_diamond_norm",
    "process_fidelity",
    "purity",
    "random_density_matrix",
    "random_quantum_channel",
    "schmidt_decomposition",
    "shannon_entropy",
    "state_fidelity",
    "state_fidelity_with_density_matrix",
]
