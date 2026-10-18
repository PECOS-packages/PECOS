# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""The step vocabulary shared by code-family gadget builders."""

from dataclasses import dataclass, field
from enum import Enum, auto


class OpType(Enum):
    """Circuit operation types."""

    # Qubit management
    ALLOC = auto()  # Allocate qubit
    PREP = auto()  # Prepare qubit in |0>

    # Single-qubit gates
    H = auto()  # Hadamard
    F = auto()  # face Clifford
    FDG = auto()  # face Clifford dagger
    SX = auto()  # sqrt X
    SXDG = auto()  # sqrt X dagger
    SY = auto()  # sqrt Y
    SYDG = auto()  # sqrt Y dagger
    SZ = auto()  # sqrt Z / phase
    SZDG = auto()  # sqrt Z dagger
    X = auto()  # Pauli X
    Z = auto()  # Pauli Z
    T = auto()  # diag(1, exp(i*pi/4)); raw magic-state injection
    TDG = auto()  # T dagger

    # Two-qubit gates
    CX = auto()  # CNOT
    CZ = auto()  # Controlled Z
    SZZ = auto()  # sqrt ZZ
    SZZDG = auto()  # sqrt ZZ dagger

    # Measurement
    MEASURE = auto()  # Destructive measurement

    # Structural
    TICK = auto()  # Layer separator
    COMMENT = auto()  # Comment/annotation

    # Annotation: declares a candidate tracked Pauli at this circuit position.
    # The propagator records its forward propagation to detectors and
    # observables; per-shot "did this Pauli fire?" is consumed at sampling
    # time. ``qubits`` is the single data qubit the Pauli acts on; ``label``
    # carries the Pauli kind and site as ``"{X|Y|Z}@s<site_idx>"``.
    TRACKED_PAULI = auto()

    # qubits == [a, b] exchanges the roles of positions a and b for all later
    # steps. A virtual swap with no physical operation and no noise location;
    # requires exactly two distinct qubits.
    RELABEL = auto()


@dataclass
class CircuitStep:
    """A code-family gadget builder step."""

    op_type: OpType
    qubits: list[int] = field(default_factory=list)
    label: str = ""  # For comments, variable names, etc.
