# Copyright 2025 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Convert PECOS QuantumCircuit to SLR format."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pecos.qeclib import qubit
from pecos.slr import Barrier, Comment, CReg, Main, Parallel, QReg

if TYPE_CHECKING:
    from pecos.circuits.quantum_circuit import QuantumCircuit


def quantum_circuit_to_slr(qc: QuantumCircuit) -> Main:
    """Convert a PECOS QuantumCircuit to SLR format.

    Args:
        qc: A PECOS QuantumCircuit object

    Returns:
        An SLR Main block representing the circuit

    Note:
        - QuantumCircuit's parallel gate structure is preserved
        - Assumes standard gate names from PECOS
    """
    # Determine number of qubits from the circuit
    max_qubit = -1
    for tick in qc:
        if hasattr(tick, "items"):
            # Dictionary-like format
            for _gate_symbol, locations, _params in tick.items():
                for loc in locations:
                    max_qubit = (
                        max(max_qubit, *loc)
                        if isinstance(loc, tuple)
                        else max(max_qubit, loc)
                    )
        else:
            # Tuple format
            gate_symbol, locations, _params = tick
            for loc in locations:
                max_qubit = (
                    max(max_qubit, *loc)
                    if isinstance(loc, tuple)
                    else max(max_qubit, loc)
                )

    num_qubits = max_qubit + 1 if max_qubit >= 0 else 0

    if num_qubits == 0:
        # Empty circuit
        return Main()

    # Create quantum register
    ops = []
    q = QReg("q", num_qubits)
    ops.append(q)

    # Track if we need classical registers for measurements
    has_measurements = False
    measurement_count = 0

    # First pass: check for measurements
    for tick_idx in range(len(qc)):
        tick = qc[tick_idx]
        if hasattr(tick, "items"):
            # Dictionary-like format
            for gate_symbol, locations, _params in tick.items():
                # Handle various measurement formats in PECOS
                if gate_symbol.upper() in [
                    "M",
                    "MZ",
                    "MX",
                    "MY",
                    "MEASURE",
                ] or gate_symbol in [
                    "measure Z",
                    "Measure",
                    "Measure +Z",
                    "Measure Z",
                    "measure",
                ]:
                    has_measurements = True
                    measurement_count += len(locations)
        else:
            # Tuple format
            gate_symbol, locations, _params = tick
            if gate_symbol.upper() in ["M", "MZ", "MX", "MY", "MEASURE"]:
                has_measurements = True
                measurement_count += len(locations)

    # Create classical register if needed
    if has_measurements:
        c = CReg("c", measurement_count)
        ops.append(c)
        current_measurement = 0
    else:
        c = None
        current_measurement = 0

    # Process each tick (time slice)
    for tick_idx in range(len(qc)):
        tick = qc[tick_idx]
        # Check if tick is empty
        if not tick or (hasattr(tick, "__len__") and len(tick) == 0):
            # Empty tick - add barrier
            ops.append(Barrier())
            continue

        # Check if we have multiple gates in parallel
        parallel_ops = []

        # Handle different tick formats
        if hasattr(tick, "items"):
            # Dictionary-like format
            for gate_symbol, locations, _params in tick.items():
                gate_ops = _convert_gate_set(
                    gate_symbol,
                    locations,
                    q,
                    c,
                    current_measurement,
                )
                parallel_ops.extend(gate_ops)

                # Update measurement counter
                if gate_symbol.upper() in ["M", "MZ", "MX", "MY", "MEASURE"]:
                    current_measurement += len(locations)
        else:
            # Tuple format (symbol, locations, params)
            gate_symbol, locations, _params = tick
            gate_ops = _convert_gate_set(
                gate_symbol,
                locations,
                q,
                c,
                current_measurement,
            )
            parallel_ops.extend(gate_ops)

            # Update measurement counter
            if gate_symbol.upper() in ["M", "MZ", "MX", "MY", "MEASURE"]:
                current_measurement += len(locations)

        # Add operations for this tick
        if len(parallel_ops) > 1:
            # Multiple operations in parallel
            ops.append(Parallel(*parallel_ops))
        elif len(parallel_ops) == 1:
            # Single operation
            ops.append(parallel_ops[0])

        # Add tick boundary if not the last tick
        if tick_idx < len(qc) - 1:
            ops.append(Barrier())

    return Main(*ops)


def _convert_gate_set(gate_symbol, locations, q, c, measurement_offset):
    """Convert a set of gates with the same symbol to SLR operations.

    Args:
        gate_symbol: The gate symbol/name
        locations: Set of qubit locations where the gate is applied
        q: Quantum register
        c: Classical register (may be None)
        measurement_offset: Current offset for measurements

    Returns:
        List of SLR operations
    """
    ops = []
    gate_upper = gate_symbol.upper()

    # Map gate symbols to operations
    if gate_upper == "H":
        for loc in locations:
            if isinstance(loc, int):
                ops.append(qubit.H(q[loc]))
            elif isinstance(loc, tuple) and len(loc) == 1:
                ops.append(qubit.H(q[loc[0]]))
    elif gate_upper == "X":
        for loc in locations:
            if isinstance(loc, int):
                ops.append(qubit.X(q[loc]))
            elif isinstance(loc, tuple) and len(loc) == 1:
                ops.append(qubit.X(q[loc[0]]))
    elif gate_upper == "Y":
        for loc in locations:
            if isinstance(loc, int):
                ops.append(qubit.Y(q[loc]))
            elif isinstance(loc, tuple) and len(loc) == 1:
                ops.append(qubit.Y(q[loc[0]]))
    elif gate_upper == "Z":
        for loc in locations:
            if isinstance(loc, int):
                ops.append(qubit.Z(q[loc]))
            elif isinstance(loc, tuple) and len(loc) == 1:
                ops.append(qubit.Z(q[loc[0]]))
    elif gate_upper in ["S", "SZ"]:
        for loc in locations:
            if isinstance(loc, int):
                ops.append(qubit.SZ(q[loc]))
            elif isinstance(loc, tuple) and len(loc) == 1:
                ops.append(qubit.SZ(q[loc[0]]))
    elif gate_upper in ["SDG", "S_DAG", "SZDG", "SZ_DAG"]:
        for loc in locations:
            if isinstance(loc, int):
                ops.append(qubit.SZdg(q[loc]))
            elif isinstance(loc, tuple) and len(loc) == 1:
                ops.append(qubit.SZdg(q[loc[0]]))
    elif gate_upper == "T":
        for loc in locations:
            if isinstance(loc, int):
                ops.append(qubit.T(q[loc]))
            elif isinstance(loc, tuple) and len(loc) == 1:
                ops.append(qubit.T(q[loc[0]]))
    elif gate_upper in ["TDG", "T_DAG"]:
        for loc in locations:
            if isinstance(loc, int):
                ops.append(qubit.Tdg(q[loc]))
            elif isinstance(loc, tuple) and len(loc) == 1:
                ops.append(qubit.Tdg(q[loc[0]]))
    elif gate_upper in ["CX", "CNOT"]:
        ops.extend(
            qubit.CX(q[loc[0]], q[loc[1]])
            for loc in locations
            if isinstance(loc, tuple) and len(loc) == 2
        )
    elif gate_upper == "CY":
        ops.extend(
            qubit.CY(q[loc[0]], q[loc[1]])
            for loc in locations
            if isinstance(loc, tuple) and len(loc) == 2
        )
    elif gate_upper == "CZ":
        ops.extend(
            qubit.CZ(q[loc[0]], q[loc[1]])
            for loc in locations
            if isinstance(loc, tuple) and len(loc) == 2
        )
    elif gate_upper == "SWAP":
        for loc in locations:
            if isinstance(loc, tuple) and len(loc) == 2:
                # Decompose SWAP into 3 CNOTs
                ops.append(qubit.CX(q[loc[0]], q[loc[1]]))
                ops.append(qubit.CX(q[loc[1]], q[loc[0]]))
                ops.append(qubit.CX(q[loc[0]], q[loc[1]]))
    elif gate_upper in ["M", "MZ", "MEASURE"] or gate_symbol in [
        "measure Z",
        "Measure",
        "Measure +Z",
        "Measure Z",
        "measure",
    ]:
        # Handle various PECOS measurement formats
        if c is not None:
            idx = measurement_offset
            for loc in locations:
                if isinstance(loc, int):
                    if idx < len(c):
                        ops.append(qubit.Measure(q[loc]) > c[idx])
                        idx += 1
                elif isinstance(loc, tuple) and len(loc) == 1 and idx < len(c):
                    ops.append(qubit.Measure(q[loc[0]]) > c[idx])
                    idx += 1
    elif gate_upper == "MX":
        if c is not None:
            idx = measurement_offset
            for loc in locations:
                if isinstance(loc, int) and idx < len(c):
                    ops.append(qubit.H(q[loc]))
                    ops.append(qubit.Measure(q[loc]) > c[idx])
                    idx += 1
    elif gate_upper == "MY":
        if c is not None:
            idx = measurement_offset
            for loc in locations:
                if isinstance(loc, int) and idx < len(c):
                    ops.append(qubit.SZdg(q[loc]))
                    ops.append(qubit.H(q[loc]))
                    ops.append(qubit.Measure(q[loc]) > c[idx])
                    idx += 1
    elif gate_upper in ["R", "RZ", "RESET"]:
        for loc in locations:
            if isinstance(loc, int):
                ops.append(qubit.Prep(q[loc]))
            elif isinstance(loc, tuple) and len(loc) == 1:
                ops.append(qubit.Prep(q[loc[0]]))
    elif gate_upper == "RX":
        for loc in locations:
            if isinstance(loc, int):
                ops.append(qubit.Prep(q[loc]))
                ops.append(qubit.H(q[loc]))
    elif gate_upper == "RY":
        for loc in locations:
            if isinstance(loc, int):
                ops.append(qubit.Prep(q[loc]))
                ops.append(qubit.H(q[loc]))
                ops.append(qubit.SZ(q[loc]))
    else:
        # Unknown gate - add as comment
        ops.append(Comment(f"Unknown gate: {gate_symbol} on {locations}"))

    return ops
