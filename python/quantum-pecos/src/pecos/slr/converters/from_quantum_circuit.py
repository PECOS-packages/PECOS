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

from pecos.slr import Barrier, CReg, Main, Parallel, QReg
from pecos.slr.qeclib import qubit

if TYPE_CHECKING:
    from pecos.circuits.quantum_circuit import QuantumCircuit


# Upper-cased PECOS measurement spellings and the basis each one measures.
_MEASUREMENT_BASES = {
    "M": "Z",
    "MZ": "Z",
    "MX": "X",
    "MY": "Y",
    "MEASURE": "Z",
    "MEASURE Z": "Z",
    "MEASURE +Z": "Z",
    "MEASURE X": "X",
    "MEASURE +X": "X",
    "MEASURE Y": "Y",
    "MEASURE +Y": "Y",
}


_ONE_QUBIT_GATES = {
    "H": qubit.H,
    "X": qubit.X,
    "Y": qubit.Y,
    "Z": qubit.Z,
    "S": qubit.SZ,
    "SZ": qubit.SZ,
    "SDG": qubit.SZdg,
    "S_DAG": qubit.SZdg,
    "SZDG": qubit.SZdg,
    "SZ_DAG": qubit.SZdg,
    "T": qubit.T,
    "TDG": qubit.Tdg,
    "T_DAG": qubit.Tdg,
    "R": qubit.PZ,
    "RESET": qubit.PZ,
}
_TWO_QUBIT_GATES = {"CX": qubit.CX, "CNOT": qubit.CX, "CY": qubit.CY, "CZ": qubit.CZ}


def _measurement_basis(symbol: str) -> str | None:
    """Return the basis for supported PECOS measurement spellings."""
    return _MEASUREMENT_BASES.get(symbol.upper())


def _one_qubit_location(symbol, location):
    """Validate a single-qubit location."""
    if isinstance(location, int):
        return location
    if isinstance(location, tuple) and len(location) == 1 and isinstance(location[0], int):
        return location[0]
    msg = f"Unsupported one-qubit location for {symbol}: {location!r}; expected an int or a 1-tuple of int"
    raise ValueError(msg)


def _two_qubit_location(symbol, location):
    """Validate a two-qubit location."""
    if isinstance(location, tuple) and len(location) == 2 and all(isinstance(idx, int) for idx in location):
        return location
    msg = f"Unsupported two-qubit location for {symbol}: {location!r}; expected a 2-tuple of int"
    raise ValueError(msg)


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
    # Size both registers from the same validated source operations.
    max_qubit = -1
    measurement_count = 0
    for gate_symbol, locations, params in qc:
        if params:
            msg = f"Unsupported parameters for QuantumCircuit gate {gate_symbol}: {list(params)}"
            raise ValueError(msg)
        basis = _measurement_basis(gate_symbol)
        gate_upper = gate_symbol.upper()
        if basis is not None:
            measurement_count += len(locations)
        if basis is not None or gate_upper in _ONE_QUBIT_GATES:
            for loc in locations:
                max_qubit = max(max_qubit, _one_qubit_location(gate_symbol, loc))
        elif gate_upper in _TWO_QUBIT_GATES or gate_upper == "SWAP":
            for loc in locations:
                max_qubit = max(max_qubit, *_two_qubit_location(gate_symbol, loc))
        else:
            msg = f"Unsupported QuantumCircuit gate {gate_symbol} on {locations}"
            raise NotImplementedError(msg)

    num_qubits = max_qubit + 1 if max_qubit >= 0 else 0

    if num_qubits == 0:
        # Empty circuit
        return Main()

    # Create quantum register
    ops = []
    q = QReg("q", num_qubits)
    ops.append(q)

    # Create classical register if needed
    if measurement_count:
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
                if _measurement_basis(gate_symbol) is not None:
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
            if _measurement_basis(gate_symbol) is not None:
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
    basis = _measurement_basis(gate_symbol)

    # Map gate symbols to operations
    if gate_upper in _ONE_QUBIT_GATES:
        gate = _ONE_QUBIT_GATES[gate_upper]
        ops.extend(gate(q[_one_qubit_location(gate_symbol, loc)]) for loc in locations)
    elif gate_upper in _TWO_QUBIT_GATES:
        gate = _TWO_QUBIT_GATES[gate_upper]
        for loc in locations:
            first, second = _two_qubit_location(gate_symbol, loc)
            ops.append(gate(q[first], q[second]))
    elif gate_upper == "SWAP":
        for loc in locations:
            first, second = _two_qubit_location(gate_symbol, loc)
            # Decompose SWAP into 3 CNOTs
            ops.append(qubit.CX(q[first], q[second]))
            ops.append(qubit.CX(q[second], q[first]))
            ops.append(qubit.CX(q[first], q[second]))
    elif basis is not None:
        for i, loc in enumerate(locations):
            idx = _one_qubit_location(gate_symbol, loc)
            if basis == "Y":
                ops.append(qubit.SZdg(q[idx]))
            if basis in ["X", "Y"]:
                ops.append(qubit.H(q[idx]))
            ops.append(qubit.Measure(q[idx]) > c[measurement_offset + i])
            if basis in ["X", "Y"]:
                ops.append(qubit.H(q[idx]))
            if basis == "Y":
                ops.append(qubit.SZ(q[idx]))
    else:
        msg = f"Unsupported QuantumCircuit gate {gate_symbol} on {locations}"
        raise NotImplementedError(msg)

    return ops
