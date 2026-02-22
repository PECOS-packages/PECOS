# Copyright 2025 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Quantum operator types, circuit representation, and utilities.

This module provides fundamental quantum types for PECOS:

Circuit Representation:
- DagCircuit: DAG-based quantum circuit (nodes=gates, edges=qubit wires)
- Gate: Quantum gate with type, parameters, and qubit targets
- GateType: Enum of supported gate types (H, X, CX, RZ, Measure, etc.)
- QubitId: Qubit identifier

Pauli Operators:
- Pauli operators (I, X, Y, Z)
- Pauli strings (multi-qubit Pauli operators)
- Array support for quantum operators (via pecos.array)

All functionality is provided by pecos_rslib - this module just re-exports
with clean documentation for quantum computing use cases.

Examples:
    >>> from pecos.quantum import DagCircuit, Gate, QubitId

    >>> # Build a Bell state circuit
    >>> circuit = DagCircuit()
    >>> h = circuit.add_gate(Gate.h([0]))
    >>> cx = circuit.add_gate(Gate.cx([(0, 1)]))
    >>> circuit.connect(h, cx, 0)

    >>> # Query circuit properties
    >>> print(circuit.gate_count())  # 2
    >>> print(circuit.depth())  # 1

    >>> from pecos.quantum import Pauli
    >>> x = Pauli.X
    >>> z = Pauli.Z
    >>> print(x)  # "X"

    >>> # Create error collections for noise models
    >>> SINGLE_QUBIT_ERRORS = [Pauli.X, Pauli.Y, Pauli.Z]

    >>> # Create Pauli arrays (Rust-backed, dtype=pauli)
    >>> from pecos import array
    >>> errors = array([Pauli.X, Pauli.Y, Pauli.Z])

    >>> # Create Pauli strings with convenient syntax
    >>> from pecos.quantum import pauli_string
    >>> ps = pauli_string("XYZ", phase=-1)  # -X_0 Y_1 Z_2
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pecos.typing import INTEGER_TYPES

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pecos.typing import Integer

# Import types from pecos_rslib
try:
    from pecos_rslib import Pauli, PauliString

    # Circuit representation types
    # HUGR conversion utilities
    from pecos_rslib.quantum import (
        DagCircuit,
        DagCircuitWouldCycleError,
        Gate,
        GateType,
        HugrConversionError,
        QubitConflictError,
        QubitId,
        Tick,
        TickCircuit,
        TickHandle,
        TickMeasureHandle,
        TickPrepHandle,
        gate_type_to_hugr_op,
        hugr_op_to_gate_type,
        hugr_to_dag_circuit,
        is_quantum_operation,
    )
except ImportError as e:
    # Provide helpful error message if Rust bindings not built
    msg = (
        f"Failed to import quantum types from pecos_rslib: {e}\n"
        "Make sure pecos_rslib is properly installed with: uv sync"
    )
    raise ImportError(msg) from e


def pauli_string(
    operators: str | Sequence[tuple[Pauli, int]] | dict[int, Pauli],
    phase: complex = 1,
) -> PauliString:
    """Create a PauliString from a convenient specification.

    This function provides a user-friendly way to create PauliString objects
    with support for multiple input formats and intuitive phase specification.

    Args:
        operators: One of the following:
            - String like "XYZ" or "IXZI" (sequential qubits starting at 0)
            - List of (Pauli, qubit_index) tuples
            - Dict mapping qubit_index -> Pauli
        phase: Phase factor, one of:
            - 1 or +1: Plus one (default)
            - -1: Minus one
            - 1j or +1j: Plus i
            - -1j: Minus i

    Returns:
        PauliString object

    Examples:
        >>> from pecos.quantum import Pauli, pauli_string

        >>> # From string (sequential qubits)
        >>> ps = pauli_string("XYZ")
        >>> print(ps)  # X_0 Y_1 Z_2

        >>> # From list of (Pauli, qubit) tuples
        >>> ps = pauli_string([(Pauli.X, 0), (Pauli.Z, 2)])
        >>> print(ps)  # X_0 Z_2

        >>> # From dict
        >>> ps = pauli_string({0: Pauli.X, 2: Pauli.Z})
        >>> print(ps)  # X_0 Z_2

        >>> # With phase
        >>> ps = pauli_string("XYZ", phase=-1)
        >>> print(ps)  # -X_0 Y_1 Z_2

        >>> ps = pauli_string([(Pauli.Y, 1)], phase=1j)
        >>> print(ps)  # +i*Y_1

        >>> ps = pauli_string("Z", phase=-1j)
        >>> print(ps)  # -i*Z_0
    """
    # Convert phase to integer code
    if isinstance(phase, (int, *INTEGER_TYPES)):
        if phase == 1:
            phase_code = 0  # +1
        elif phase == -1:
            phase_code = 2  # -1
        else:
            msg = f"Invalid integer phase: {phase}. Must be +1 or -1"
            raise ValueError(msg)
    elif isinstance(phase, complex):
        if phase == 1j:
            phase_code = 1  # +i
        elif phase == -1j:
            phase_code = 3  # -i
        else:
            msg = f"Invalid complex phase: {phase}. Must be +1j or -1j"
            raise ValueError(msg)
    else:
        msg = f"Invalid phase type: {type(phase)}. Must be int or complex"
        raise TypeError(msg)

    # Convert operators to list of (Pauli, qubit) tuples
    if isinstance(operators, str):
        # String format - use from_str then update phase
        ps = PauliString.from_str(operators)
        if phase_code != 0:
            # Need to recreate with correct phase
            paulis = ps.get_paulis()
            return PauliString(paulis, phase=phase_code)
        return ps
    if isinstance(operators, dict):
        # Dict format - convert to list
        paulis = [(pauli, qubit) for qubit, pauli in sorted(operators.items())]
        return PauliString(paulis, phase=phase_code)
    if isinstance(operators, list):
        # Already in list format
        return PauliString(operators, phase=phase_code)
    msg = f"Invalid operators type: {type(operators)}. Must be str, dict, or list"
    raise TypeError(msg)


__all__ = [
    # Circuit representation
    "DagCircuit",
    "DagCircuitWouldCycleError",
    "Gate",
    "GateType",
    # HUGR conversion utilities
    "HugrConversionError",
    # Pauli operators
    "Pauli",
    "PauliString",
    "QubitConflictError",
    "QubitId",
    "Tick",
    "TickCircuit",
    "TickHandle",
    "TickMeasureHandle",
    "TickPrepHandle",
    "gate_type_to_hugr_op",
    "hugr_op_to_gate_type",
    "hugr_to_dag_circuit",
    "is_quantum_operation",
    "pauli_string",
]
