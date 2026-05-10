"""Quantum-information channel representations and measures.

This module re-exports the Rust-backed implementations from
``pecos_rslib.quantum_info``. Computation and validation happen in Rust; this
file only provides the public Python import location.
"""

from __future__ import annotations

from pecos_rslib.quantum_info import (
    ChoiMatrix,
    KrausOps,
    PauliChannel,
    Ptm,
    average_gate_fidelity,
    gate_error,
    pauli_channel_diamond_distance,
    pauli_channel_diamond_norm,
    process_fidelity,
    purity,
    random_density_matrix,
    random_quantum_channel,
    state_fidelity,
    state_fidelity_with_density_matrix,
)

__all__ = [
    "ChoiMatrix",
    "KrausOps",
    "PauliChannel",
    "Ptm",
    "average_gate_fidelity",
    "gate_error",
    "pauli_channel_diamond_distance",
    "pauli_channel_diamond_norm",
    "process_fidelity",
    "purity",
    "random_density_matrix",
    "random_quantum_channel",
    "state_fidelity",
    "state_fidelity_with_density_matrix",
]
