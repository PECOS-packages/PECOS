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

from collections.abc import Mapping
from typing import TYPE_CHECKING

from pecos.slr import Barrier, CReg, Main, Parallel, QReg
from pecos.slr.qeclib import qubit
from pecos.slr.vars import IDENTIFIER_PATTERN

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
    "RESET": qubit.PZ,
}
_TWO_QUBIT_GATES = {"CX": qubit.CX, "CNOT": qubit.CX, "CY": qubit.CY, "CZ": qubit.CZ}


def _measurement_basis(symbol: str) -> str | None:
    """Return the basis for supported PECOS measurement spellings."""
    return _MEASUREMENT_BASES.get(symbol.upper())


def _is_qubit_index(idx):
    """Return whether idx is a valid qubit index; a negative one would wrap."""
    return isinstance(idx, int) and idx >= 0


def _one_qubit_location(symbol, location):
    """Validate a single-qubit location."""
    if _is_qubit_index(location):
        return location
    if isinstance(location, tuple) and len(location) == 1 and _is_qubit_index(location[0]):
        return location[0]
    msg = f"Unsupported one-qubit location for {symbol}: {location!r}; expected a non-negative int or a 1-tuple of one"
    raise ValueError(msg)


def _two_qubit_location(symbol, location):
    """Validate a two-qubit location."""
    if (
        isinstance(location, tuple)
        and len(location) == 2
        and all(_is_qubit_index(idx) for idx in location)
        and location[0] != location[1]
    ):
        return location
    msg = f"Unsupported two-qubit location for {symbol}: {location!r}; expected a 2-tuple of distinct non-negative ints"
    raise ValueError(msg)


def _validate_classical_register_name(name, context, quantum_sizes):
    """Require an SLR identifier distinct from every quantum register name."""
    if quantum_sizes is None and name == "q":
        msg = (
            f"Classical register {name!r} in {context} conflicts with the converter's quantum register; "
            "rename it to a different identifier, such as 'c'"
        )
        raise ValueError(msg)
    if not isinstance(name, str) or not IDENTIFIER_PATTERN.fullmatch(name):
        msg = (
            f"Invalid classical register name {name!r} in {context}; "
            "rename it to match [A-Za-z_][A-Za-z0-9_]*, such as 'c'"
        )
        raise ValueError(msg)
    if quantum_sizes is not None and name in quantum_sizes:
        msg = (
            f"Classical register {name!r} in {context} conflicts with quantum register {name!r}; "
            "rename one register to a distinct SLR identifier"
        )
        raise ValueError(msg)


def _validate_destination(symbol, key, location, value, declared_sizes, register_sizes, quantum_sizes):
    """Validate a destination and size its register before any SLR emission."""
    pair_type = (tuple, list) if key == "var" else tuple
    valid = isinstance(value, pair_type) and len(value) == 2 and isinstance(value[0], str) and _is_qubit_index(value[1])
    if not valid:
        msg = f"Invalid {key} for QuantumCircuit gate {symbol} at location {location!r}: {value!r}"
        raise ValueError(msg)
    name, index = value
    _validate_classical_register_name(
        name,
        f"{key} for QuantumCircuit gate {symbol} at location {location!r}: {value!r}",
        quantum_sizes,
    )
    if name in declared_sizes and index >= declared_sizes[name]:
        msg = (
            f"Invalid {key} for QuantumCircuit gate {symbol} at location {location!r}: "
            f"{(name, index)!r} exceeds cvar_spec width {declared_sizes[name]}"
        )
        raise ValueError(msg)
    register_sizes[name] = max(register_sizes.get(name, 0), index + 1)
    return name, index


def _measurement_destinations(symbol, locations, params, declared_sizes, register_sizes, quantum_sizes):
    """Validate both destination fields; var takes precedence as in qc2phir."""
    destinations = {}
    if "var_output" in params:
        output = params["var_output"]
        if not isinstance(output, Mapping):
            msg = f"Invalid var_output for QuantumCircuit gate {symbol} at locations {locations!r}: {output!r}"
            raise ValueError(msg)
        for loc, value in output.items():
            if loc not in locations:
                msg = f"Extra var_output location for QuantumCircuit gate {symbol} at location {loc!r}: {value!r}"
                raise ValueError(msg)
            destinations[loc] = _validate_destination(
                symbol,
                "var_output",
                loc,
                value,
                declared_sizes,
                register_sizes,
                quantum_sizes,
            )
        for loc in locations:
            if loc not in output:
                msg = f"Missing var_output for QuantumCircuit gate {symbol} at location {loc!r}: None"
                raise ValueError(msg)
    if "var" in params:
        value = params["var"]
        if len(locations) != 1:
            msg = (
                f"Invalid var for QuantumCircuit gate {symbol} at locations {locations!r}: "
                f"{value!r}; expected one location"
            )
            raise ValueError(msg)
        loc = next(iter(locations))
        destinations[loc] = _validate_destination(
            symbol,
            "var",
            loc,
            value,
            declared_sizes,
            register_sizes,
            quantum_sizes,
        )
    return destinations


def _classical_register_sizes(metadata, quantum_sizes):
    """Read declared widths before emission, retaining zero widths for bounds checks."""
    sizes = metadata.get("cvar_spec", {})
    if not isinstance(sizes, Mapping):
        msg = f"Invalid QuantumCircuit cvar_spec: {sizes!r}; expected a register-to-size mapping"
        raise TypeError(msg)
    for name, size in sizes.items():
        _validate_classical_register_name(name, "cvar_spec", quantum_sizes)
        if not _is_qubit_index(size):
            msg = f"Invalid QuantumCircuit cvar_spec entry {name!r}: {size!r}; expected a non-negative integer width"
            raise ValueError(msg)
    return dict(sizes)


def _quantum_register_metadata(metadata):
    """Validate native or JSON quantum metadata without emitting SLR objects."""
    for key, other in (("qvars", "qvar_spec"), ("qvar_spec", "qvars")):
        if key in metadata and other not in metadata:
            msg = f"Missing QuantumCircuit {other}; supply both qvars and qvar_spec or omit both"
            raise ValueError(msg)
    if "qvars" not in metadata:
        return None, None
    sizes = metadata["qvar_spec"]
    refs = metadata["qvars"]
    for key, value in (("qvar_spec", sizes), ("qvars", refs)):
        if not isinstance(value, Mapping):
            msg = f"Invalid QuantumCircuit {key}: {value!r}; expected a mapping"
            raise TypeError(msg)
    for name, size in sizes.items():
        if not isinstance(name, str) or not IDENTIFIER_PATTERN.fullmatch(name):
            msg = (
                f"Invalid quantum register name {name!r} in qvar_spec; "
                "rename it to match [A-Za-z_][A-Za-z0-9_]*, such as 'data'"
            )
            raise ValueError(msg)
        if not _is_qubit_index(size) or isinstance(size, bool):
            msg = f"Invalid QuantumCircuit qvar_spec entry {name!r}: {size!r}; expected a non-negative integer width"
            raise ValueError(msg)
    normalized = {}
    slots = {}
    for key, value in refs.items():
        idx = int(key) if isinstance(key, str) and key.isascii() and key.isdecimal() else key
        if not _is_qubit_index(idx) or isinstance(idx, bool):
            msg = f"Invalid qvars qubit id {key!r}; use a non-negative int or a decimal integer string"
            raise ValueError(msg)
        if idx in normalized:
            msg = f"Duplicate qvars qubit id {key!r}; use only one key for qubit id {idx}"
            raise ValueError(msg)
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            msg = f"Invalid qvars entry {key!r}: {value!r}; use a (register, index) pair"
            raise ValueError(msg)
        name, index = value
        if not isinstance(name, str) or name not in sizes:
            msg = f"Invalid qvars entry {key!r}: {value!r}; declare its register in qvar_spec"
            raise ValueError(msg)
        if not _is_qubit_index(index) or isinstance(index, bool) or index >= sizes[name]:
            msg = (
                f"Invalid qvars entry {key!r}: {value!r}; "
                f"use a non-negative integer index below qvar_spec width {sizes[name]}"
            )
            raise ValueError(msg)
        slot = (name, index)
        if slot in slots:
            msg = (
                f"Duplicate qvars slot {slot!r} for qubit ids {slots[slot]!r} and {key!r}; "
                "map each qubit id to a distinct (register, index)"
            )
            raise ValueError(msg)
        normalized[idx] = slot
        slots[slot] = key
    return dict(sizes), normalized


def quantum_circuit_to_slr(qc: QuantumCircuit) -> Main:
    """Convert a PECOS QuantumCircuit to SLR format.

    Args:
        qc: A PECOS QuantumCircuit object

    Returns:
        An SLR Main block representing the circuit

    Note:
        - QuantumCircuit's parallel gate structure is preserved
        - Assumes standard gate names from PECOS
        - Destinations or cvar_spec select explicit classical registers; otherwise
          measurement results are allocated sequentially in a single c register
        - Gate groups within a tick are unordered, so bare measurements can move
          relative to destination measurements in Stim records without changing register contents.
    """
    # Size both registers from the same validated source operations.
    max_qubit = -1
    measurement_count = 0
    metadata = getattr(qc, "metadata", {})
    destination_mode = "cvar_spec" in metadata
    quantum_sizes, quantum_refs = _quantum_register_metadata(metadata)
    declared_sizes = _classical_register_sizes(metadata, quantum_sizes)
    register_sizes = dict(declared_sizes)
    gate_destinations = []
    used_qubits = set()
    for gate_symbol, locations, params in qc:
        basis = _measurement_basis(gate_symbol)
        unsupported = [key for key in params if basis is None or key not in {"var_output", "var"}]
        if unsupported:
            msg = f"Unsupported parameters for QuantumCircuit gate {gate_symbol}: {unsupported}"
            raise ValueError(msg)
        gate_upper = gate_symbol.upper()
        destinations = _measurement_destinations(
            gate_symbol,
            locations,
            params,
            declared_sizes,
            register_sizes,
            quantum_sizes,
        )
        gate_destinations.append(destinations)
        if basis is not None:
            measurement_count += len(locations)
            destination_mode |= "var_output" in params or "var" in params
        if basis is not None or gate_upper in _ONE_QUBIT_GATES:
            for loc in locations:
                idx = _one_qubit_location(gate_symbol, loc)
                used_qubits.add(idx)
                max_qubit = max(max_qubit, idx)
        elif gate_upper in _TWO_QUBIT_GATES or gate_upper == "SWAP":
            for loc in locations:
                indices = _two_qubit_location(gate_symbol, loc)
                used_qubits.update(indices)
                max_qubit = max(max_qubit, *indices)
        else:
            msg = f"Unsupported QuantumCircuit gate {gate_symbol} on {locations}"
            raise NotImplementedError(msg)

    if quantum_refs is not None:
        for idx in sorted(used_qubits):
            if idx not in quantum_refs:
                msg = f"Missing qvars entry for gate qubit id {idx!r}; add its (register, index) mapping"
                raise ValueError(msg)

    num_qubits = max_qubit + 1 if max_qubit >= 0 else 0

    if not destination_mode and measurement_count:
        _validate_classical_register_name("c", "automatic measurement register", quantum_sizes)
        register_sizes["c"] = measurement_count
    cregs = {name: CReg(name, size) for name, size in register_sizes.items() if size}

    if quantum_sizes is None:
        if num_qubits == 0:
            return Main(*cregs.values())
        q = QReg("q", num_qubits)
        ops = [q]
    else:
        qregs = {name: QReg(name, size) for name, size in quantum_sizes.items() if size}
        q = {idx: qregs[name][index] for idx, (name, index) in quantum_refs.items()}
        ops = list(qregs.values())

    ops.extend(cregs.values())
    current_measurement = 0
    validated_destinations = iter(gate_destinations)

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
        gate_sets = tick.items() if hasattr(tick, "items") else (tick,)
        for gate_symbol, locations, _params in gate_sets:
            destinations = next(validated_destinations)
            if _measurement_basis(gate_symbol) is not None:
                if not destination_mode:
                    destinations = {loc: ("c", current_measurement + i) for i, loc in enumerate(locations)}
                current_measurement += len(locations)
            parallel_ops.extend(_convert_gate_set(gate_symbol, locations, q, cregs, destinations))

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


def _convert_gate_set(gate_symbol, locations, q, cregs, destinations):
    """Convert a set of gates with the same symbol to SLR operations.

    Args:
        gate_symbol: The gate symbol/name
        locations: Set of qubit locations where the gate is applied
        q: Quantum register or validated qubit-id-to-register-element mapping
        cregs: Classical registers by name
        destinations: Validated measurement destinations by location (empty for bare measurements)

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
        for loc in locations:
            idx = _one_qubit_location(gate_symbol, loc)
            if basis == "Y":
                ops.append(qubit.SZdg(q[idx]))
            if basis in ["X", "Y"]:
                ops.append(qubit.H(q[idx]))
            measurement = qubit.Measure(q[idx])
            if loc in destinations:
                name, index = destinations[loc]
                measurement = measurement > cregs[name][index]
            ops.append(measurement)
            if basis in ["X", "Y"]:
                ops.append(qubit.H(q[idx]))
            if basis == "Y":
                ops.append(qubit.SZ(q[idx]))
    else:
        msg = f"Unsupported QuantumCircuit gate {gate_symbol} on {locations}"
        raise NotImplementedError(msg)

    return ops
