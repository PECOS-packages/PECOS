# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Helix algebra and C4-Helix flagged memory circuits (arXiv:2609.03194)."""

from pecos.qec.helix.code import HelixCode, c4_helix_code, helix_code
from pecos.qec.helix.compile import Matrix, logical_pauli_fixup, min_cost_words, parse_signed_logical_pauli
from pecos.qec.helix.gadgets import (
    HelixAllocation,
    c4_round_steps,
    chosen_weight8_supports,
    memory_steps,
    weight8_check_steps,
)
from pecos.qec.helix.logical import LogicalGate, c4_helix_gates, logical_action, symplectic_action
from pecos.qec.helix.lowering import LoweredHelixCircuit, lower_steps
from pecos.qec.helix.memory import memory_circuit, memory_conjugates

__all__ = [
    "HelixAllocation",
    "HelixCode",
    "LogicalGate",
    "LoweredHelixCircuit",
    "Matrix",
    "c4_helix_code",
    "c4_helix_gates",
    "c4_round_steps",
    "chosen_weight8_supports",
    "helix_code",
    "logical_action",
    "logical_pauli_fixup",
    "lower_steps",
    "memory_circuit",
    "memory_conjugates",
    "memory_steps",
    "min_cost_words",
    "parse_signed_logical_pauli",
    "symplectic_action",
    "weight8_check_steps",
]
