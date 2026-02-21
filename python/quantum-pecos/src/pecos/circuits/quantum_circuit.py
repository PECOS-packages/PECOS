# Copyright 2018 The PECOS Developers
# Copyright 2018 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the terms of Contract
# DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Contains the class ``QuantumCircuit``, which is used to represent quantum circuits.

This implementation uses TickCircuit from the Rust backend as internal storage.
"""

from __future__ import annotations

import copy
import json
from collections import defaultdict
from collections.abc import MutableSequence
from typing import TYPE_CHECKING

import pecos as pc
from pecos.circuits import qc2phir

try:
    from pecos_rslib import QubitConflictError, TickCircuit
except ImportError:
    TickCircuit = None  # type: ignore[misc, assignment]
    QubitConflictError = None  # type: ignore[misc, assignment]

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from pecos.typing import JSONDict, JSONValue

# Type aliases
Location = int | tuple[int, ...]
LocationSet = set[Location] | list[Location] | tuple[Location, ...]
GateDict = dict[str, LocationSet]
CircuitSetup = int | list[GateDict] | None

# Symbol to TickHandle method mapping for single-qubit gates
_SINGLE_QUBIT_GATES = {
    "I": "i",
    "H": "h",
    "F": "f",
    "FDG": "fdg",
    "X": "x",
    "Y": "y",
    "Z": "z",
    # sqrt gates
    "SX": "sx",
    "SXDG": "sxdg",
    "SY": "sy",
    "SYDG": "sydg",
    "SZ": "sz",
    "SZDG": "szdg",
    # Aliases
    "Q": "sx",
    "QD": "sxdg",
    "R": "sy",
    "RD": "sydg",
    "S": "sz",
    "SD": "szdg",
    "SDG": "szdg",  # Also accept SDG as alias
    "T": "t",
    "TDG": "tdg",
}

# Symbol to TickHandle method mapping for rotation gates (take angle parameter)
_ROTATION_GATES = {
    "RX": "rx",
    "RY": "ry",
    "RZ": "rz",
}

# Symbol to TickHandle method mapping for two-qubit gates
_TWO_QUBIT_GATES = {
    "CX": "cx",
    "CNOT": "cx",
    "CY": "cy",
    "CZ": "cz",
    "SXX": "sxx",
    "SXXDG": "sxxdg",
    "SYY": "syy",
    "SYYDG": "syydg",
    "SZZ": "szz",
    "SZZDG": "szzdg",
}

# Symbol to TickHandle method mapping for two-qubit rotation gates
_TWO_QUBIT_ROTATION_GATES = {
    "RXX": "rxx",
    "RYY": "ryy",
    "RZZ": "rzz",
}

# Symbol to TickHandle method mapping for R1XY gate (takes theta, phi angles)
_R1XY_GATES = {
    "R1XY": "r1xy",
}

# Symbol to TickHandle method mapping for U gate (takes theta, phi, lambda angles)
_U_GATES = {
    "U": "u",
}

# Symbol to TickHandle method mapping for R2XXYYZZ gate (takes 3 angles: zz, yy, xx)
# This gate is decomposed into RZZ + RYY + RXX
_R2XXYYZZ_GATES = {
    "R2XXYYZZ": "r2xxyyzz",
    "RZZRYYRXX": "r2xxyyzz",  # Alternative name
    "RXXYYZZ": "r2xxyyzz",  # Alternative name
}

# SWAP gate - decomposed into CX gates: SWAP(a,b) = CX(a,b) CX(b,a) CX(a,b)
_SWAP_GATES = {"SWAP"}

# Prep/measure gates
_PREP_GATES = {
    "PREP",
    "init",
    "Init",
    "init |0>",
    "Init |0>",
    "RESET",
    "Reset",
    "reset",
}
_MEASURE_GATES = {"MEASURE", "MZ", "measure", "Measure", "measure Z"}

# GateType string to symbol mapping (for iteration)
_GATETYPE_TO_SYMBOL = {
    "I": "I",
    "H": "H",
    "F": "F",
    "Fdg": "FDG",
    "X": "X",
    "Y": "Y",
    "Z": "Z",
    "SX": "SX",
    "SXdg": "SXDG",
    "SY": "SY",
    "SYdg": "SYDG",
    "SZ": "SZ",
    "SZdg": "SZDG",
    "T": "T",
    "Tdg": "TDG",
    "RX": "RX",
    "RY": "RY",
    "RZ": "RZ",
    "R1XY": "R1XY",
    "U": "U",
    "CX": "CX",
    "CY": "CY",
    "CZ": "CZ",
    "SXX": "SXX",
    "SXXdg": "SXXDG",
    "SYY": "SYY",
    "SYYdg": "SYYDG",
    "SZZ": "SZZ",
    "SZZdg": "SZZDG",
    "RXX": "RXX",
    "RYY": "RYY",
    "RZZ": "RZZ",
    "R2XXYYZZ": "R2XXYYZZ",
    "Prep": "init |0>",
    "Measure": "measure",
    "MeasureFree": "measure",
    "QAlloc": "QAlloc",
    "QFree": "QFree",
    "Idle": "Idle",
}


class QuantumCircuit(MutableSequence):
    """A representation of a quantum circuit.

    Similar to [{gate_symbol: set of qudits, ...}, ...] where each element is a time step in which gates act in
    parallel.

    This implementation uses TickCircuit from the Rust backend as internal storage.
    """

    def __init__(
        self,
        circuit_setup: CircuitSetup = None,
        **metadata: JSONValue,
    ) -> None:
        """Initialize a QuantumCircuit.

        Args:
            circuit_setup (None, int, list of dict): Initial circuit configuration. Can be None (empty circuit),
                int (number of initial ticks), or list of dicts (pre-configured ticks).
            **metadata: Additional metadata to associate with the circuit as keyword arguments.
        """
        if TickCircuit is None:
            msg = "TickCircuit not available. Please install pecos_rslib."
            raise ImportError(msg)

        self._inner = TickCircuit()
        self.metadata = metadata
        self._qudits: set[int] = set()
        # Track logically reserved ticks (for backwards compatibility with empty tick creation)
        self._reserved_ticks = 0

        if "tracked_qudits" in metadata:
            msg = "tracked_qudits is not a valid metadata key"
            raise ValueError(msg)

        if circuit_setup is not None:
            self._circuit_setup(circuit_setup)

    @property
    def qudits(self) -> set[int]:
        """Returns all qudits used in the circuit."""
        return set(self._inner.all_qubits())

    @qudits.setter
    def qudits(self, value: set[int]) -> None:
        """Setter for backwards compatibility."""
        self._qudits = value

    @property
    def active_qudits(self) -> list[set[int]]:
        """Returns the active_qudits of all the ticks."""
        result = []
        for tick_idx in range(len(self)):
            tick = self._inner.get_tick(tick_idx)
            if tick is not None:
                # Get individual qubits from all gates in the tick
                active: set[int] = set()
                for gate in tick.gates():
                    for q in gate.qubits:
                        active.add(q)
                result.append(active)
            else:
                result.append(set())
        return result

    def _add_gate_to_tick(
        self,
        tick_handle: object,
        symbol: str | object,
        locations: LocationSet,
        **params: JSONValue,
    ) -> None:
        """Add a gate to a tick handle based on symbol."""
        # Handle logical gate objects that have a .symbol attribute
        if not isinstance(symbol, str):
            symbol = symbol.symbol if hasattr(symbol, "symbol") else str(symbol)
        symbol_upper = symbol.upper()

        # Convert locations to list, filtering out None values (placeholders for logical gates)
        loc_list = [loc for loc in locations if loc is not None]
        if not loc_list:
            return

        # Serialize params for storage (handle tuples -> lists)
        def make_serializable(obj: object) -> object:
            if isinstance(obj, tuple):
                return list(obj)
            if isinstance(obj, frozenset):
                return list(obj)
            if isinstance(obj, set):
                return list(obj)
            return obj

        params_json = json.dumps({k: make_serializable(v) for k, v in params.items()}) if params else ""

        # Helper to store original symbol and params in metadata (idempotent - skips if qubit already used)
        def add_with_symbol(
            method: Callable[..., object],
            *args: float,
        ) -> object | None:
            try:
                result = method(*args)
            except QubitConflictError:
                # Qubit already in use in this tick - skip (idempotent behavior)
                return None
            else:
                # Store original symbol and params for round-trip preservation
                if hasattr(result, "meta"):
                    result.meta("_symbol", symbol)
                    if params_json:
                        result.meta("_params", params_json)
                return result

        # Handle single-qubit gates
        if symbol_upper in _SINGLE_QUBIT_GATES:
            method_name = _SINGLE_QUBIT_GATES[symbol_upper]
            if hasattr(tick_handle, method_name):
                method = getattr(tick_handle, method_name)
                for loc in loc_list:
                    if isinstance(loc, tuple):
                        for q in loc:
                            add_with_symbol(method, q)
                    else:
                        add_with_symbol(method, loc)
                return
            # Fall through to custom gate handler if method doesn't exist

        # Handle rotation gates
        if symbol_upper in _ROTATION_GATES:
            method_name = _ROTATION_GATES[symbol_upper]
            if hasattr(tick_handle, method_name):
                method = getattr(tick_handle, method_name)
                angle = params.get("angle", params.get("theta", 0.0))
                for loc in loc_list:
                    if isinstance(loc, tuple):
                        for q in loc:
                            add_with_symbol(method, angle, q)
                    else:
                        add_with_symbol(method, angle, loc)
                return
            # Fall through to custom gate handler if method doesn't exist

        # Handle two-qubit gates
        if symbol_upper in _TWO_QUBIT_GATES:
            method_name = _TWO_QUBIT_GATES[symbol_upper]
            if hasattr(tick_handle, method_name):
                method = getattr(tick_handle, method_name)
                for loc in loc_list:
                    if isinstance(loc, tuple) and len(loc) == 2:
                        add_with_symbol(method, loc[0], loc[1])
                return
            # Fall through to custom gate handler if method doesn't exist

        # Handle two-qubit rotation gates
        if symbol_upper in _TWO_QUBIT_ROTATION_GATES:
            method_name = _TWO_QUBIT_ROTATION_GATES[symbol_upper]
            if hasattr(tick_handle, method_name):
                method = getattr(tick_handle, method_name)
                angle = params.get("angle", params.get("theta", 0.0))
                for loc in loc_list:
                    if isinstance(loc, tuple) and len(loc) == 2:
                        add_with_symbol(method, angle, loc[0], loc[1])
                return
            # Fall through to custom gate handler if method doesn't exist

        # Handle R1XY gate (takes theta, phi angles)
        if symbol_upper in _R1XY_GATES:
            method_name = _R1XY_GATES[symbol_upper]
            if hasattr(tick_handle, method_name):
                method = getattr(tick_handle, method_name)
                # Handle angles tuple or individual theta/phi params
                angles = params.get("angles")
                if angles is not None and len(angles) >= 2:
                    theta = angles[0]
                    phi = angles[1]
                else:
                    theta = params.get("theta", params.get("angle", 0.0))
                    phi = params.get("phi", 0.0)
                for loc in loc_list:
                    if isinstance(loc, tuple):
                        for q in loc:
                            add_with_symbol(method, theta, phi, q)
                    else:
                        add_with_symbol(method, theta, phi, loc)
                return
            # Fall through to custom gate handler if method doesn't exist

        # Handle U gate (takes theta, phi, lambda angles)
        if symbol_upper in _U_GATES:
            method_name = _U_GATES[symbol_upper]
            if hasattr(tick_handle, method_name):
                method = getattr(tick_handle, method_name)
                # Handle angles tuple or individual theta/phi/lambda params
                angles = params.get("angles")
                if angles is not None and len(angles) >= 3:
                    theta = angles[0]
                    phi = angles[1]
                    lambda_ = angles[2]
                else:
                    theta = params.get("theta", 0.0)
                    phi = params.get("phi", 0.0)
                    lambda_ = params.get("lambda", params.get("lambda_", 0.0))
                for loc in loc_list:
                    if isinstance(loc, tuple):
                        for q in loc:
                            add_with_symbol(method, theta, phi, lambda_, q)
                    else:
                        add_with_symbol(method, theta, phi, lambda_, loc)
                return
            # Fall through to custom gate handler if method doesn't exist

        # Handle R2XXYYZZ gate (takes 3 angles: zz, yy, xx)
        # R2XXYYZZ is not a native GateType. We store it as RZZ with metadata
        # containing all three angles and the original symbol. When iterating,
        # _iter_tick reconstructs the R2XXYYZZ gate from this metadata.
        if symbol_upper in _R2XXYYZZ_GATES:
            # Handle angles tuple or individual parameters
            angles = params.get("angles")
            if angles is not None and len(angles) >= 3:
                zz_angle = angles[0]
                yy_angle = angles[1]
                xx_angle = angles[2]
            else:
                zz_angle = params.get("zz", 0.0)
                yy_angle = params.get("yy", 0.0)
                xx_angle = params.get("xx", 0.0)

            for loc in loc_list:
                if isinstance(loc, tuple) and len(loc) == 2:
                    # Store as RZZ with R2XXYYZZ metadata
                    result = tick_handle.rzz(zz_angle, loc[0], loc[1])
                    if hasattr(result, "meta"):
                        result.meta("_symbol", symbol)
                        # Store all three angles as comma-separated string
                        result.meta(
                            "_r2xxyyzz_angles",
                            f"{zz_angle},{yy_angle},{xx_angle}",
                        )
                        if params_json:
                            result.meta("_params", params_json)
            return

        # Handle SWAP gate - stored as CX with metadata
        # SWAP is not a native GateType. We store it as CX with metadata
        # indicating it's a SWAP. The simulator bindings handle SWAP directly.
        if symbol_upper in _SWAP_GATES:
            for loc in loc_list:
                if isinstance(loc, tuple) and len(loc) == 2:
                    # Store as CX with SWAP metadata
                    result = tick_handle.cx(loc[0], loc[1])
                    if hasattr(result, "meta"):
                        result.meta("_symbol", symbol)
                        if params_json:
                            result.meta("_params", params_json)
            return

        # Handle prep gates - idempotent (skip if qubit already used in tick)
        if symbol in _PREP_GATES or symbol_upper == "PREP":
            for loc in loc_list:
                if isinstance(loc, tuple):
                    for q in loc:
                        try:
                            result = tick_handle.pz(q)
                            result.meta("_symbol", symbol)
                            if params_json:
                                result.meta("_params", params_json)
                        except QubitConflictError:
                            pass  # Qubit already initialized in this tick
                else:
                    try:
                        result = tick_handle.pz(loc)
                        result.meta("_symbol", symbol)
                        if params_json:
                            result.meta("_params", params_json)
                    except QubitConflictError:
                        pass  # Qubit already initialized in this tick
            return

        # Handle measure gates - idempotent (skip if qubit already used in tick)
        if symbol in _MEASURE_GATES or symbol_upper == "MEASURE":
            for loc in loc_list:
                if isinstance(loc, tuple):
                    for q in loc:
                        try:
                            result = tick_handle.mz(q)
                            result.meta("_symbol", symbol)
                            if params_json:
                                result.meta("_params", params_json)
                        except QubitConflictError:
                            pass  # Qubit already measured in this tick
                else:
                    try:
                        result = tick_handle.mz(loc)
                        result.meta("_symbol", symbol)
                        if params_json:
                            result.meta("_params", params_json)
                    except QubitConflictError:
                        pass  # Qubit already measured in this tick
            return

        # Fallback: try to use the symbol directly as a method name
        method_name = symbol.lower()
        if hasattr(tick_handle, method_name):
            method = getattr(tick_handle, method_name)
            for loc in loc_list:
                if isinstance(loc, tuple):
                    if len(loc) == 2:
                        add_with_symbol(method, loc[0], loc[1])
                    else:
                        for q in loc:
                            add_with_symbol(method, q)
                else:
                    add_with_symbol(method, loc)
        else:
            # Store unrecognized gates using a no-op gate with metadata
            # This allows round-trip preservation for simulator-specific gates
            # Use I gate (identity) as carrier for unknown single-qubit gates
            for loc in loc_list:
                if isinstance(loc, tuple):
                    if len(loc) == 2:
                        # Two-qubit gate - use CX as carrier
                        result = tick_handle.cx(loc[0], loc[1])
                        if hasattr(result, "meta"):
                            result.meta("_symbol", symbol)
                            result.meta("_custom_gate", "true")
                            if params_json:
                                result.meta("_params", params_json)
                    else:
                        # Multi-qubit locations as individual qubits
                        for q in loc:
                            result = tick_handle.i(q)
                            if hasattr(result, "meta"):
                                result.meta("_symbol", symbol)
                                result.meta("_custom_gate", "true")
                                if params_json:
                                    result.meta("_params", params_json)
                else:
                    # Single-qubit gate - use I (identity) as carrier
                    result = tick_handle.i(loc)
                    if hasattr(result, "meta"):
                        result.meta("_symbol", symbol)
                        result.meta("_custom_gate", "true")
                        if params_json:
                            result.meta("_params", params_json)

    def append(
        self,
        symbol: str | GateDict,
        locations: LocationSet | None = None,
        **params: JSONValue,
    ) -> None:
        """Adds a new gate=>gate_locations (set) pair to the end of the circuit.

        Args:
            symbol(str or dict): A gate dictionary of gate symbol => set of qudit ids or tuples of qudit ids
            locations: Set of qudit ids or tuples of qudit ids where the gate is applied. If None, symbol must
                be a gate dict.
            **params: Additional parameters for the gate (e.g., angle values for rotation gates)
        """
        # If locations is None then assume symbol is a gate_dict
        gate_dict: GateDict = symbol if locations is None else {symbol: locations}  # type: ignore[assignment]

        tick_handle = self._inner.tick()

        for gate_symbol, gate_locations in gate_dict.items():
            if gate_locations:
                self._add_gate_to_tick(
                    tick_handle,
                    gate_symbol,
                    gate_locations,
                    **params,
                )

    def update(
        self,
        symbol: str | GateDict,
        locations: LocationSet | None = None,
        tick: int = -1,
        *,
        emptyappend: bool = False,
        **params: JSONValue,
    ) -> None:
        """Updates a group of parallel gates to include the gate acting on the set of qudits.

        Args:
            symbol(str or dict): A gate dictionary of gate symbol => set of qudit ids or tuples of qudit ids
            locations(set or None): Set of qudit ids or tuples of qudit ids where the gate is applied. If None,
                symbol must be a gate dict.
            tick(int): The time (tick) when the update should occur.
            emptyappend(bool): Whether it is allowed to add an empty tick if the QuantumCircuit is empty.
            **params: Additional parameters for the gate (e.g., angle values for rotation gates)
        """
        # If locations is None then assume symbol is a gate_dict
        gate_dict: GateDict = symbol if locations is None else {symbol: locations}  # type: ignore[assignment]

        # Get logical and physical tick counts
        logical_ticks = len(self)  # includes reserved ticks
        # Use next_tick_index() to get actual tick count including empty ticks
        # (num_ticks() excludes trailing empty ticks which breaks reserved ticks)
        physical_ticks = self._inner.next_tick_index()

        # Handle empty circuit case with negative tick index
        if logical_ticks == 0 and tick < 0:
            if emptyappend:
                # Create a new tick
                tick_handle = self._inner.tick()
            else:
                # Cannot update empty circuit with negative tick without emptyappend
                return
        else:
            # Handle negative indices (use logical_ticks for the calculation)
            actual_tick = tick if tick >= 0 else logical_ticks + tick

            # If we're trying to access a tick that doesn't exist physically yet, create it
            tick_handle = self._inner.tick() if actual_tick >= physical_ticks else self._inner.tick_at(actual_tick)

        for gate_symbol, gate_locations in gate_dict.items():
            if gate_locations:
                self._add_gate_to_tick(
                    tick_handle,
                    gate_symbol,
                    gate_locations,
                    **params,
                )

    def discard(self, locations: LocationSet, tick: int = -1) -> None:
        """Discards ``locations`` for tick ``tick``.

        Args:
            locations: Set of qudit ids or tuples of qudit ids to discard from the tick
            tick: The time (tick) index from which to discard the locations. Defaults to -1 (last tick).
        """
        # Handle negative indices
        actual_tick = tick if tick >= 0 else len(self) + tick

        # Convert locations to list of qubits
        qubits = []
        for loc in locations:
            if isinstance(loc, tuple):
                qubits.extend(loc)
            else:
                qubits.append(loc)

        self._inner.discard(qubits, actual_tick)

    def add_ticks(self, num_ticks: int) -> None:
        """Appends empty ticks to the circuit.

        Args:
            num_ticks: The number of empty ticks to append to the circuit
        """
        self._reserved_ticks += num_ticks
        self._inner.reserve_ticks(num_ticks)

    def items(
        self,
        tick: int | None = None,
    ) -> Iterator[tuple[str, set[Location], JSONDict]]:
        """An iterator through all gates/qudits in the quantum circuit.

        If ``tick`` is not None then it will iterate over only the qudits/qudits in the corresponding tick.

        Args:
            tick: The time (tick) index to iterate over. If None, iterates over all ticks.
        """
        if tick is None:
            for tick_idx in range(len(self)):
                yield from self._iter_tick(tick_idx)
        else:
            actual_tick = tick if tick >= 0 else len(self) + tick
            yield from self._iter_tick(actual_tick)

    def _iter_tick(
        self,
        tick_idx: int,
    ) -> Iterator[tuple[str, set[Location], JSONDict]]:
        """Iterate over gates in a specific tick.

        Gates with the same symbol and params are grouped together with their
        locations merged into a single set, matching the original input format.
        """
        tick_obj = self._inner.get_tick(tick_idx)
        if tick_obj is None:
            return

        # Collect gates and group by (symbol, params_json) to merge locations
        # Use a dict to preserve insertion order and group gates
        grouped: dict[tuple[str, str], tuple[set[Location], JSONDict]] = {}

        for gate_idx, gate in enumerate(tick_obj.gates()):
            # Check for stored original symbol in metadata
            stored_symbol = tick_obj.get_gate_attr(gate_idx, "_symbol")

            if stored_symbol is not None:
                symbol = stored_symbol
            else:
                gate_type_str = str(gate.gate_type)
                # Extract gate type name from "GateType.H" format
                if "." in gate_type_str:
                    gate_type_str = gate_type_str.rsplit(".", maxsplit=1)[-1]
                symbol = _GATETYPE_TO_SYMBOL.get(gate_type_str, gate_type_str)

            qubits = list(gate.qubits)
            if len(qubits) == 1:
                location: Location = qubits[0]
            else:
                location = tuple(qubits)

            # Extract params from gate (angles, etc.)
            params: JSONDict = {}

            # Check for stored params (general case)
            stored_params_json = tick_obj.get_gate_attr(gate_idx, "_params")
            if stored_params_json is not None:
                try:
                    stored_params = json.loads(stored_params_json)
                    # Convert lists back to tuples for "angles"
                    if "angles" in stored_params and isinstance(
                        stored_params["angles"],
                        list,
                    ):
                        stored_params["angles"] = tuple(stored_params["angles"])
                    # Fix JSON type issues (e.g., var_output keys become strings)
                    stored_params = self._fix_json_meta(stored_params)
                    params.update(stored_params)
                except json.JSONDecodeError:
                    pass

            # Check for custom gate params (stored as JSON in metadata) - legacy
            custom_params_json = tick_obj.get_gate_attr(gate_idx, "_custom_params")
            if custom_params_json is not None:
                try:
                    custom_params = json.loads(custom_params_json)
                    # Convert lists back to tuples for "angles"
                    if "angles" in custom_params and isinstance(
                        custom_params["angles"],
                        list,
                    ):
                        custom_params["angles"] = tuple(custom_params["angles"])
                    # Fix JSON type issues (e.g., var_output keys become strings)
                    custom_params = self._fix_json_meta(custom_params)
                    params.update(custom_params)
                except json.JSONDecodeError:
                    pass

            # Check for R2XXYYZZ special case (stored as RZZ with metadata)
            if (
                r2xxyyzz_angles := tick_obj.get_gate_attr(gate_idx, "_r2xxyyzz_angles")
            ) is not None and stored_symbol in _R2XXYYZZ_GATES:
                # Reconstruct R2XXYYZZ angles from metadata
                angle_parts = r2xxyyzz_angles.split(",")
                if len(angle_parts) >= 3:
                    params["angles"] = [float(a) for a in angle_parts[:3]]
            elif hasattr(gate, "angles"):
                angles = gate.angles
                if angles:
                    if len(angles) == 1:
                        # Single angle gates (RX, RY, RZ, RXX, RYY, RZZ)
                        params["angle"] = angles[0]
                    elif len(angles) == 2:
                        # Two angle gates (R1XY)
                        params["theta"] = angles[0]
                        params["phi"] = angles[1]
                    elif len(angles) == 3:
                        # Three angle gates (U)
                        params["theta"] = angles[0]
                        params["phi"] = angles[1]
                        params["lambda"] = angles[2]

            # Create a hashable key from symbol and params
            # Sort params keys for consistent hashing
            params_key = json.dumps(params, sort_keys=True) if params else ""
            key = (symbol, params_key)

            if key in grouped:
                # Add location to existing group
                grouped[key][0].add(location)
            else:
                # Create new group
                grouped[key] = ({location}, params)

        # Yield grouped results
        for (symbol, _), (locations, params) in grouped.items():
            yield symbol, locations, params

    def iter_ticks(self) -> Iterator[tuple[TickView, int, JSONDict]]:
        """Iterate over circuit time ticks.

        Yields:
            Tuples containing gate collection view, tick number, and metadata.
        """
        for tick_idx in range(len(self)):
            yield TickView(self, tick_idx), tick_idx, self.metadata

    def insert(
        self,
        tick: int,
        item: GateDict | tuple[GateDict, JSONDict],
    ) -> None:
        """Inserts ``gate_dict`` into ``ticks`` at index ``tick``.

        Args:
            tick: The time (tick) index where the item should be inserted
            item: Either a gate dictionary or a tuple of (gate_dict, params) to insert at the specified tick
        """
        if isinstance(item, tuple):
            gate_dict, params = item
        else:
            gate_dict, params = item, {}

        tick_handle = self._inner.insert_tick(tick)

        for gate_symbol, gate_locations in gate_dict.items():
            if gate_locations:
                self._add_gate_to_tick(
                    tick_handle,
                    gate_symbol,
                    gate_locations,
                    **params,
                )

    def _circuit_setup(self, circuit_setup: CircuitSetup) -> None:
        if isinstance(circuit_setup, int):
            # Reserve empty ticks (logically, not physically in the Rust backend)
            self._reserved_ticks = circuit_setup
            self._inner.reserve_ticks(circuit_setup)
        else:
            # Build circuit from other description (a shallow copy).
            for other_tick in circuit_setup:
                self.append(other_tick)

    def to_json_str(self) -> str:
        """Creates a json str representation of the QuantumCircuit listing all the gates.

        It does not preserve ticks or parallel gating of different gate types.
        """
        metadata = self.metadata

        gates = []
        for sym, qubits, meta in self.items():
            gate = {
                "sym": sym,
                "qubits": list(qubits),
                "metadata": meta,
            }
            gates.append(gate)

        prog = {
            "prog_type": "PECOS.QuantumCircuit",
            "PECOS_version": str(pc.__version__),
            "prog_metadata": metadata,
            "gates": gates,
        }

        return json.dumps(prog)

    @staticmethod
    def _fix_json_meta(meta: JSONDict) -> JSONDict:
        """Fix some of the type issues for converting json rep back to a QuantumCircuit."""
        if "var_output" in meta:
            meta["var_output"] = {int(k): tuple(v) for k, v in meta["var_output"].items()}
        return meta

    @classmethod
    def from_json_str(cls, qc_json: str) -> QuantumCircuit:
        """Converts a json str that represents a QuantumCircuit back into a QuantumCircuit object."""
        qc_dict = json.loads(qc_json)

        qc_meta = qc_dict["prog_metadata"]
        qc = QuantumCircuit(**qc_meta)

        for gate_dict in qc_dict["gates"]:
            sym = gate_dict["sym"]

            qubits = gate_dict["qubits"]
            qubits = set(qubits) if qubits and isinstance(qubits[0], int) else {tuple(q) for q in qubits}

            meta = gate_dict["metadata"]
            meta = cls._fix_json_meta(meta)

            qc.append(sym, qubits, **meta)

        return qc

    def to_phir_dict(self) -> dict:
        """Converts this QuantumCircuit into the PHIR format as a dict."""
        return qc2phir.to_phir_dict(self)

    def to_phir_json(self) -> str:
        """Converts this QuantumCircuit into the PHIR/JSON format."""
        return qc2phir.to_phir_json(self)

    def __getitem__(self, tick: int) -> TickView:
        """Returns tick when instance[index] is used.

        Args:
            tick(int): Tick index of the circuit.
        """
        actual_tick = tick if tick >= 0 else len(self) + tick
        return TickView(self, actual_tick)

    def __setitem__(self, tick: int, item: tuple[GateDict, JSONDict]) -> None:
        """Set gate collection at specified tick."""
        actual_tick = tick if tick >= 0 else len(self) + tick
        gate_dict, params = item

        # Get qubits to discard first
        tick_obj = self._inner.get_tick(actual_tick)
        if tick_obj is not None:
            qubits_to_discard = list(tick_obj.active_qubits())
            if qubits_to_discard:
                self._inner.discard(qubits_to_discard, actual_tick)

        # Add new gates
        tick_handle = self._inner.tick_at(actual_tick)
        for gate_symbol, gate_locations in gate_dict.items():
            if gate_locations:
                self._add_gate_to_tick(
                    tick_handle,
                    gate_symbol,
                    gate_locations,
                    **params,
                )

    def __len__(self) -> int:
        """Used to return number of ticks when len() is used on an instance of this class."""
        # Return max of actual ticks and reserved ticks (for backwards compatibility)
        return max(self._inner.num_ticks(), self._reserved_ticks)

    def __delitem__(self, tick: int) -> None:
        """Used to delete a tick. For example: del instance[key].

        Args:
            tick: The time (tick) index to delete (replace with an empty tick)
        """
        actual_tick = tick if tick >= 0 else len(self) + tick
        tick_obj = self._inner.get_tick(actual_tick)
        if tick_obj is not None:
            qubits_to_discard = list(tick_obj.active_qubits())
            if qubits_to_discard:
                self._inner.discard(qubits_to_discard, actual_tick)

    def __str__(self) -> str:
        """String returned when a string representation is requested. This occurs during printing."""
        str_list = []
        for tick_idx in range(len(self)):
            tick_list = []
            for symbol, locations, params in self._iter_tick(tick_idx):
                if len(params) == 0:
                    tick_list.append(f"'{symbol}': {locations}")
                else:
                    tick_list.append(f"'{symbol}': loc: {locations} - params={params}")
            tick_list_str = ", ".join(tick_list)
            str_list.append(f"{{{tick_list_str}}}")

        if self.metadata:
            return "QuantumCircuit(params={}, ticks=[{}])".format(
                str(self.metadata),
                ", ".join(str_list),
            )
        return "QuantumCircuit([{}])".format(", ".join(str_list))

    def __repr__(self) -> str:
        """Return a string representation."""
        return self.__str__()

    def __copy__(self) -> QuantumCircuit:
        """Create a shallow copy."""
        newone = QuantumCircuit()
        newone.metadata = dict(self.metadata)
        # Copy gates tick by tick
        for i in range(len(self)):
            for symbol, locations, params in self._iter_tick(i):
                newone.update(symbol, locations, tick=i, **params)
        return newone

    def copy(self) -> QuantumCircuit:
        """Create a shallow copy of the circuit."""
        return copy.copy(self)

    def __iter__(self) -> Iterator[tuple[str, LocationSet, JSONDict]]:
        """Iterate over all gates in the circuit."""
        return self.items()


class TickView:
    """A view into a specific tick of the circuit.

    Provides the same interface as the old ParamGateCollection for backwards compatibility.
    """

    class Gate:
        """Gate representation with symbol, parameters, and locations."""

        __slots__ = ("locations", "params", "symbol")

        def __init__(self, symbol: str, params: JSONDict, locations: set[Location]) -> None:
            """Initialize a Gate with its symbol, parameters, and locations."""
            self.symbol = symbol
            self.params = params
            self.locations = locations

        def __repr__(self) -> str:
            """Return a string representation of the Gate."""
            return f"Gate(symbol={self.symbol!r}, params={self.params!r}, locations={self.locations!r})"

    def __init__(self, circuit: QuantumCircuit, tick_idx: int) -> None:
        """Initialize a TickView.

        Args:
            circuit: The parent QuantumCircuit.
            tick_idx: The tick index this view represents.
        """
        self._circuit = circuit
        self._tick_idx = tick_idx

    @property
    def circuit(self) -> QuantumCircuit:
        """Returns the parent circuit (for backwards compatibility)."""
        return self._circuit

    @property
    def active_qudits(self) -> set[Location]:
        """Returns the active qudits for this tick."""
        tick = self._circuit._inner.get_tick(self._tick_idx)
        if tick is None:
            return set()

        active: set[Location] = set()
        for gate in tick.gates():
            qubits = list(gate.qubits)
            if len(qubits) == 1:
                active.add(qubits[0])
            else:
                active.add(tuple(qubits))
        return active

    @property
    def metadata(self) -> JSONDict:
        """Returns the circuit metadata."""
        return self._circuit.metadata

    @property
    def symbols(self) -> dict[str, list[Gate]]:
        """Returns a dictionary mapping gate symbols to lists of Gate objects.

        Each Gate has .symbol, .params, and .locations attributes.

        Example:
            >>> tick = circuit[0]
            >>> for gate in tick.symbols["CX"]:
            ...     print(gate.locations)
            ...
        """
        result: dict[str, list[TickView.Gate]] = defaultdict(list)
        for symbol, locations, params in self.items():
            result[symbol].append(self.Gate(symbol, params, locations))
        return dict(result)

    def add(
        self,
        symbol: str | GateDict | None,
        locations: LocationSet | None = None,
        **params: JSONValue,
    ) -> TickView:
        """Add a gate to this tick.

        Args:
            symbol: Gate symbol or gate dictionary.
            locations: Set of qudit locations where the gate is applied.
            **params: Additional parameters for the gate.
        """
        gate_dict: GateDict = symbol if locations is None else {symbol: locations}  # type: ignore[assignment]

        if gate_dict:
            tick_handle = self._circuit._inner.tick_at(self._tick_idx)
            for gate_symbol, gate_locations in gate_dict.items():
                if gate_locations:
                    self._circuit._add_gate_to_tick(
                        tick_handle,
                        gate_symbol,
                        gate_locations,
                        **params,
                    )

        return self

    def discard(self, locations: LocationSet) -> TickView:
        """Remove gate locations from this tick.

        Args:
            locations: Set of qudit ids or tuples of qudit ids to remove.
        """
        qubits = []
        for loc in locations:
            if isinstance(loc, tuple):
                qubits.extend(loc)
            else:
                qubits.append(loc)

        self._circuit._inner.discard(qubits, self._tick_idx)
        return self

    def items(
        self,
        _tick: None = None,
    ) -> Iterator[tuple[str, set[Location], JSONDict]]:
        """Generator to return a dictionary-like iter."""
        yield from self._circuit._iter_tick(self._tick_idx)

    def __str__(self) -> str:
        """Return string representation of the tick."""
        tick_list = []
        for symbol, locations, params in self.items():
            if len(params) == 0:
                tick_list.append(f"'{symbol}': {locations}")
            else:
                tick_list.append(f"'{symbol}': loc: {locations} - params={params}")
        tick_list_str = ", ".join(tick_list)

        return f"Tick({{{tick_list_str}}})"

    def __repr__(self) -> str:
        """Return detailed string representation of the tick."""
        return self.__str__()


# Keep ParamGateCollection as an alias for backwards compatibility
ParamGateCollection = TickView
