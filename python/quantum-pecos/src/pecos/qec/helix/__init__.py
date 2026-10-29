# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Helix algebra and C4-Helix flagged memory and logic circuits (arXiv:2609.03194)."""

from pecos.qec.helix.code import HelixCode, c4_helix_code, helix_code
from pecos.qec.helix.compile import Matrix, logical_pauli_fixup, min_cost_words, parse_signed_logical_pauli
from pecos.qec.helix.gadgets import (
    HelixAllocation,
    c4_round_steps,
    chosen_weight8_supports,
    logical_gate_steps,
    memory_steps,
    syndrome_round_gadgets,
    weight8_check_steps,
)
from pecos.qec.helix.logic import logic_circuit, logic_conjugates, logic_gadgets
from pecos.qec.helix.logical import (
    LogicalGate,
    c4_helix_gates,
    compile_logical_clifford,
    logical_action,
    logical_pauli_ops,
    logical_pauli_string,
    logical_token_ops,
    symplectic_action,
)
from pecos.qec.helix.lowering import LoweredHelixCircuit, helix_result_classifier, helix_result_refs, lower_steps
from pecos.qec.helix.memory import memory_circuit, memory_conjugates, memory_definitions, set_discovered_metadata

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
    "compile_logical_clifford",
    "helix_code",
    "helix_result_classifier",
    "helix_result_refs",
    "logic_circuit",
    "logic_conjugates",
    "logic_gadgets",
    "logical_action",
    "logical_gate_steps",
    "logical_pauli_fixup",
    "logical_pauli_ops",
    "logical_pauli_string",
    "logical_token_ops",
    "lower_steps",
    "memory_circuit",
    "memory_conjugates",
    "memory_definitions",
    "memory_steps",
    "min_cost_words",
    "parse_signed_logical_pauli",
    "set_discovered_metadata",
    "symplectic_action",
    "syndrome_round_gadgets",
    "weight8_check_steps",
]
