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

"""C++-based sparse stabilizer simulator for PECOS.

This module provides a Python interface to the high-performance C++ implementation of sparse stabilizer simulation,
enabling efficient quantum circuit simulation for stabilizer circuits with reduced memory overhead and improved
performance compared to dense state vector representations.
"""

from __future__ import annotations

# ruff: noqa: SLF001

from typing import TYPE_CHECKING, NoReturn

from pecos_rslib._pecos_rslib import CppSparseSim as CppRustSparseSim

if TYPE_CHECKING:
    from pecos.typing import SimulatorGateParams


class CppSparseSimRs:
    """C++-based sparse stabilizer simulator wrapped via Rust.

    A high-performance sparse stabilizer simulator implemented in C++, exposed through Rust bindings,
    providing efficient simulation of quantum circuits that can be represented using the stabilizer
    formalism with reduced memory requirements.
    """

    def __init__(self, num_qubits: int, seed: int | None = None):
        """Initialize the C++-based sparse simulator.

        Args:
            num_qubits: Number of qubits to simulate.
            seed: Optional seed for the RNG. If None, uses hardware random.
        """
        if seed is not None:
            self._sim = CppRustSparseSim(num_qubits, seed)
        else:
            self._sim = CppRustSparseSim(num_qubits)
        self.num_qubits = num_qubits
        self.bindings = dict(gate_dict)

    def reset(self) -> CppSparseSimRs:
        """Reset the simulator to its initial state.

        Returns:
            Self for method chaining.
        """
        self._sim.reset()
        return self

    def set_seed(self, seed: int) -> None:
        """Set the RNG seed for this simulator instance.

        Args:
            seed: The seed value for the random number generator.
        """
        self._sim.set_seed(seed)

    def run_gate(
        self,
        symbol: str,
        locations: set[int] | set[tuple[int, ...]],
        **params: SimulatorGateParams,
    ) -> dict[int, int]:
        """Execute a quantum gate on specified locations.

        Args:
            symbol: Gate symbol/name to execute.
            locations: Set of qubit locations to apply the gate to.
            **params: Additional gate parameters.

        Returns:
            Dictionary mapping locations to measurement results.
        """
        output = {}

        if params.get("simulate_gate", True) and locations:
            for location in locations:
                if params.get("angles") and len(params["angles"]) == 1:
                    params.update({"angle": params["angles"][0]})
                elif "angle" in params and "angles" not in params:
                    params["angles"] = (params["angle"],)

                if symbol in self.bindings:
                    results = self.bindings[symbol](self, location, **params)
                else:
                    msg = f"Gate {symbol} is not supported in this simulator."
                    raise Exception(msg)

                if results is not None:
                    output[location] = results

        return output

    def run_circuit(
        self,
        circuit,
        removed_locations: set[int] | None = None,
    ) -> dict[int, int]:
        """Execute a quantum circuit.

        Args:
            circuit: Quantum circuit to execute.
            removed_locations: Optional set of locations to exclude.

        Returns:
            Dictionary mapping locations to measurement results.
        """
        if removed_locations is None:
            removed_locations = set()

        results = {}
        for symbol, locations, params in circuit.items():
            gate_results = self.run_gate(
                symbol,
                locations - removed_locations,
                **params,
            )
            results.update(gate_results)

        return results

    def add_faults(self, circuit, removed_locations: set[int] | None = None) -> None:
        """Add faults to the simulator by running a circuit.

        Args:
            circuit: Circuit containing fault operations.
            removed_locations: Optional set of locations to exclude.
        """
        self.run_circuit(circuit, removed_locations)

    @property
    def stabs(self) -> TableauWrapper:
        """Get stabilizers tableau wrapper.

        Returns:
            Wrapper for accessing stabilizer tableau.
        """
        return TableauWrapper(self._sim, is_stab=True)

    @property
    def destabs(self) -> TableauWrapper:
        """Get destabilizers tableau wrapper.

        Returns:
            Wrapper for accessing destabilizer tableau.
        """
        return TableauWrapper(self._sim, is_stab=False)

    def print_stabs(
        self,
        *,
        verbose: bool = True,
        print_y: bool = True,
        print_destabs: bool = False,
    ) -> str | tuple[str, str]:
        """Print stabilizer tableau(s).

        Args:
            verbose: Whether to print to stdout.
            print_y: Whether to print Y operators as Y (True) or W (False).
            print_destabs: Whether to also print destabilizers.

        Returns:
            String representation of stabilizers, or tuple if destabs included.
        """
        stabs_raw = self._sim.stab_tableau()
        stabs_lines = stabs_raw.strip().split("\n")
        stabs_formatted = [
            adjust_tableau_string(line, is_stab=True, print_y=print_y)
            for line in stabs_lines
        ]

        if print_destabs:
            destabs_raw = self._sim.destab_tableau()
            destabs_lines = destabs_raw.strip().split("\n")
            destabs_formatted = [
                adjust_tableau_string(line, is_stab=False, print_y=print_y)
                for line in destabs_lines
            ]

            if verbose:
                print("Stabilizers:")
                for line in stabs_formatted:
                    print(line)
                print("Destabilizers:")
                for line in destabs_formatted:
                    print(line)
            return stabs_formatted, destabs_formatted
        else:
            if verbose:
                print("Stabilizers:")
                for line in stabs_formatted:
                    print(line)
            return stabs_formatted

    def logical_sign(self, logical_op) -> NoReturn:  # noqa: ARG002
        """Calculate logical sign (not implemented).

        Args:
            logical_op: Logical operator to analyze.

        Raises:
            NotImplementedError: This method is not yet implemented.
        """
        msg = "logical_sign method not implemented yet"
        raise NotImplementedError(msg)

    def refactor(
        self, xs, zs, choose=None, prefer=None, protected=None
    ) -> NoReturn:  # noqa: ARG002
        """Refactor stabilizer tableau (not implemented).

        Args:
            xs: X component.
            zs: Z component.
            choose: Choice parameter.
            prefer: Preference parameter.
            protected: Protection parameter.

        Raises:
            NotImplementedError: This method is not yet implemented.
        """
        msg = "refactor method not implemented yet"
        raise NotImplementedError(msg)

    def find_stab(self, xs, zs) -> NoReturn:  # noqa: ARG002
        """Find stabilizer (not implemented).

        Args:
            xs: X component.
            zs: Z component.

        Raises:
            NotImplementedError: This method is not yet implemented.
        """
        msg = "find_stab method not implemented yet"
        raise NotImplementedError(msg)

    def copy(self) -> NoReturn:
        """Create a copy of the simulator (not implemented).

        Raises:
            NotImplementedError: This method is not yet implemented.
        """
        msg = "copy method not implemented yet"
        raise NotImplementedError(msg)


class TableauWrapper:
    def __init__(self, sim, *, is_stab: bool):
        self._sim = sim
        self._is_stab = is_stab

    def print_tableau(
        self, *, verbose: bool = False, print_y: bool = False
    ) -> list[str]:
        if self._is_stab:
            tableau = self._sim.stab_tableau()
        else:
            tableau = self._sim.destab_tableau()

        lines = tableau.strip().split("\n")
        adjusted_lines = [
            adjust_tableau_string(line, is_stab=self._is_stab, print_y=print_y)
            for line in lines
        ]

        if verbose:
            for line in adjusted_lines:
                print(line)

        return adjusted_lines


def _measure_z_forced(sim, qubit: int, params: dict) -> int | None:
    """Perform forced Z measurement, returning None (omitted) when result is 0."""
    params.get("forced_outcome", 0)
    # Debug output
    # print(f"[Python] _measure_z_forced: qubit={qubit}, forced_outcome={forced}")
    result = sim.run_1q_gate("MZForced", qubit, params)
    # print(f"[Python] _measure_z_forced: result={result}")
    # For compatibility with Python simulators, return None when measurement is 0
    # This causes the result to be omitted from the output dict
    if result == 0:
        return None
    return result


def _init_to_zero(sim, qubit: int, forced_outcome: int = -1) -> None:
    """Initialize qubit to |0> by measuring and correcting.

    Args:
        sim: The simulator instance
        qubit: The qubit to initialize
        forced_outcome: The forced measurement outcome (-1 for random, 0 or 1 for forced)
    """
    # Measure the qubit with optional forcing
    if forced_outcome == -1:
        result = sim.mz(qubit)
    else:
        # Use forced measurement - this matches Python's behavior
        result = sim.run_1q_gate("MZForced", qubit, {"forced_outcome": forced_outcome})
        result = result if result is not None else 0
    # If it's |1>, flip it to |0>
    if result:
        sim.x(qubit)
    return None


def _init_to_one(sim, qubit: int, forced_outcome: int = -1) -> None:
    """Initialize qubit to |1> by measuring and correcting.

    Args:
        sim: The simulator instance
        qubit: The qubit to initialize
        forced_outcome: The forced measurement outcome (-1 for random, 0 or 1 for forced)
    """
    # Measure the qubit with optional forcing
    if forced_outcome == -1:
        result = sim.mz(qubit)
    else:
        # Use forced measurement
        result = sim.run_1q_gate("MZForced", qubit, {"forced_outcome": forced_outcome})
        result = result if result is not None else 0
    # If it's |0>, flip it to |1>
    if not result:
        sim.x(qubit)
    return None


def _init_to_plus(sim, qubit: int) -> None:
    """Initialize qubit to |+>."""
    # First ensure |0> (no forcing since we want deterministic init)
    _init_to_zero(sim, qubit, forced_outcome=-1)
    # Apply H to get |+>
    sim.h(qubit)
    return None


def _init_to_minus(sim, qubit: int) -> None:
    """Initialize qubit to |->."""
    # First ensure |1>
    _init_to_one(sim, qubit)
    # Apply H to get |->
    sim.h(qubit)
    return None


def _init_to_plus_i(sim, qubit: int) -> None:
    """Initialize qubit to |+i> using H5 gate."""
    # C++ H5 on |0> produces iY which is iW (what we need for |+i>)
    _init_to_zero(sim, qubit, forced_outcome=-1)
    sim.run_1q_gate("H5", qubit, {})
    return None


def _init_to_minus_i(sim, qubit: int) -> None:
    """Initialize qubit to |-i> using H6 gate."""
    # C++ H6 on |0> produces -iY which is -iW (what we need for |-i>)
    _init_to_zero(sim, qubit, forced_outcome=-1)
    sim.run_1q_gate("H6", qubit, {})
    return None


def adjust_tableau_string(line: str, *, is_stab: bool, print_y: bool = True) -> str:
    """
    Adjust the tableau string to ensure the sign part always takes up two spaces
    and handle Y vs W display based on print_y parameter.

    Args:
        line (str): A single line from the tableau string.
        is_stab (bool): True if this is a stabilizer, False if destabilizer.
        print_y (bool): If True, show Y operators as Y. If False, show as W with proper phase.

    Returns:
        str: The adjusted line with proper spacing and Y/W formatting.
    """
    # First handle the sign formatting
    if is_stab:
        if line.startswith("+i"):
            adjusted = " i" + line[2:]
        elif line.startswith("-i"):
            adjusted = "-i" + line[2:]
        elif line.startswith("i"):
            adjusted = " i" + line[1:]  # Handle bare imaginary (no + or -)
        elif line.startswith("+"):
            adjusted = "  " + line[1:]
        elif line.startswith("-"):
            adjusted = " -" + line[1:]
        else:
            adjusted = "  " + line  # Default case, shouldn't happen with correct input
    else:
        # For destabilizers, strip all signs (no phases shown)
        # Remove any sign prefix (+, -, +i, -i, i) and add two spaces
        if line.startswith("+i") or line.startswith("-i"):
            adjusted = "  " + line[2:]  # Strip 2 chars for imaginary signs
        elif line.startswith("i"):
            adjusted = "  " + line[1:]  # Strip 1 char for bare imaginary
        elif line.startswith("+") or line.startswith("-"):
            adjusted = "  " + line[1:]  # Strip 1 char for real signs
        else:
            adjusted = "  " + line  # No sign to strip

    # Handle Y vs W conversion based on print_y parameter
    if not print_y:
        # Simply replace Y with W - the phase is already correct from C++
        adjusted = adjusted.replace("Y", "W")

    return adjusted


# Define the gate dictionary - reuse the same mappings as SparseSim
gate_dict = {
    "I": lambda sim, q, **params: None,  # noqa: ARG005
    "X": lambda sim, q, **params: sim._sim.run_1q_gate("X", q, params),
    "Y": lambda sim, q, **params: sim._sim.run_1q_gate("Y", q, params),
    "Z": lambda sim, q, **params: sim._sim.run_1q_gate("Z", q, params),
    "SX": lambda sim, q, **params: sim._sim.run_1q_gate("SX", q, params),
    "SXdg": lambda sim, q, **params: sim._sim.run_1q_gate("SXdg", q, params),
    "SY": lambda sim, q, **params: sim._sim.run_1q_gate("SY", q, params),
    "SYdg": lambda sim, q, **params: sim._sim.run_1q_gate("SYdg", q, params),
    "SZ": lambda sim, q, **params: sim._sim.run_1q_gate("SZ", q, params),
    "SZdg": lambda sim, q, **params: sim._sim.run_1q_gate("SZdg", q, params),
    # Alternative names for square root gates
    "Q": lambda sim, q, **params: sim._sim.run_1q_gate(
        "SX", q, params
    ),  # Q = sqrt(X) = SX
    "Qd": lambda sim, q, **params: sim._sim.run_1q_gate("SXdg", q, params),  # Q† = SXdg
    "R": lambda sim, q, **params: sim._sim.run_1q_gate(
        "SY", q, params
    ),  # R = sqrt(Y) = SY
    "Rd": lambda sim, q, **params: sim._sim.run_1q_gate("SYdg", q, params),  # R† = SYdg
    "S": lambda sim, q, **params: sim._sim.run_1q_gate("SZ", q, params),  # S gate is SZ
    "Sd": lambda sim, q, **params: sim._sim.run_1q_gate("SZdg", q, params),  # S dagger
    "H": lambda sim, q, **params: sim._sim.run_1q_gate("H", q, params),
    "H2": lambda sim, q, **params: sim._sim.run_1q_gate("H2", q, params),
    "H3": lambda sim, q, **params: sim._sim.run_1q_gate("H3", q, params),
    "H4": lambda sim, q, **params: sim._sim.run_1q_gate("H4", q, params),
    "H5": lambda sim, q, **params: sim._sim.run_1q_gate("H5", q, params),
    "H6": lambda sim, q, **params: sim._sim.run_1q_gate("H6", q, params),
    "F": lambda sim, q, **params: sim._sim.run_1q_gate("F", q, params),
    "Fdg": lambda sim, q, **params: sim._sim.run_1q_gate("Fdg", q, params),
    "F1": lambda sim, q, **params: sim._sim.run_1q_gate(
        "F", q, params
    ),  # Alternative name for F
    "F1d": lambda sim, q, **params: sim._sim.run_1q_gate(
        "Fdg", q, params
    ),  # Alternative name for Fdg
    "F2": lambda sim, q, **params: sim._sim.run_1q_gate("F2", q, params),
    "F2dg": lambda sim, q, **params: sim._sim.run_1q_gate("F2dg", q, params),
    "F2d": lambda sim, q, **params: sim._sim.run_1q_gate(
        "F2dg", q, params
    ),  # Alternative name for F2dg
    "F3": lambda sim, q, **params: sim._sim.run_1q_gate("F3", q, params),
    "F3dg": lambda sim, q, **params: sim._sim.run_1q_gate("F3dg", q, params),
    "F3d": lambda sim, q, **params: sim._sim.run_1q_gate(
        "F3dg", q, params
    ),  # Alternative name for F3dg
    "F4": lambda sim, q, **params: sim._sim.run_1q_gate("F4", q, params),
    "F4dg": lambda sim, q, **params: sim._sim.run_1q_gate("F4dg", q, params),
    "F4d": lambda sim, q, **params: sim._sim.run_1q_gate(
        "F4dg", q, params
    ),  # Alternative name for F4dg
    "II": lambda sim, qs, **params: None,  # noqa: ARG005
    "CX": lambda sim, qs, **params: sim._sim.run_2q_gate("CX", qs, params),
    "CNOT": lambda sim, qs, **params: sim._sim.run_2q_gate("CX", qs, params),
    "CY": lambda sim, qs, **params: sim._sim.run_2q_gate("CY", qs, params),
    "CZ": lambda sim, qs, **params: sim._sim.run_2q_gate("CZ", qs, params),
    "SWAP": lambda sim, qs, **params: sim._sim.run_2q_gate("SWAP", qs, params),
    "G": lambda sim, qs, **params: sim._sim.run_2q_gate(
        "G2", qs, params
    ),  # G is an alias for G2
    "G2": lambda sim, qs, **params: sim._sim.run_2q_gate("G2", qs, params),
    "SXX": lambda sim, qs, **params: sim._sim.run_2q_gate("SXX", qs, params),
    "SXXdg": lambda sim, qs, **params: sim._sim.run_2q_gate("SXXdg", qs, params),
    "SYY": lambda sim, qs, **params: sim._sim.run_2q_gate("SYY", qs, params),
    "SYYdg": lambda sim, qs, **params: sim._sim.run_2q_gate("SYYdg", qs, params),
    "SZZ": lambda sim, qs, **params: sim._sim.run_2q_gate("SZZ", qs, params),
    "SZZdg": lambda sim, qs, **params: sim._sim.run_2q_gate("SZZdg", qs, params),
    "SqrtXX": lambda sim, qs, **params: sim._sim.run_2q_gate(
        "SXX", qs, params
    ),  # SqrtXX is an alias for SXX
    "MZ": lambda sim, q, **params: sim._sim.run_1q_gate("MZ", q, params),
    "MX": lambda sim, q, **params: sim._sim.run_1q_gate("MX", q, params),
    "MY": lambda sim, q, **params: sim._sim.run_1q_gate("MY", q, params),
    "Measure +X": lambda sim, q, **params: sim._sim.run_1q_gate("MX", q, params),
    "Measure +Y": lambda sim, q, **params: sim._sim.run_1q_gate("MY", q, params),
    "Measure +Z": lambda sim, q, **params: sim._sim.run_1q_gate("MZ", q, params),
    "Measure": lambda sim, q, **params: sim._sim.run_1q_gate("MZ", q, params),
    "measure Z": lambda sim, q, **params: _measure_z_forced(sim._sim, q, params),
    "MZForced": lambda sim, q, **params: _measure_z_forced(sim._sim, q, params),
    # PZForced - for the forced projection gate, we still support forced_outcome
    "PZForced": lambda sim, q, **params: (
        _init_to_zero(sim._sim, q, forced_outcome=params.get("forced_outcome", 0))
        if params.get("forced_outcome", 0) == 0
        else _init_to_one(sim._sim, q, forced_outcome=params.get("forced_outcome", 1))
    ),
    # Init gates - always initialize to the specified state, ignore forced_outcome
    # CppSparseStab doesn't have PZ/PX/PY projection gates, so we measure and correct
    "Init": lambda sim, q, **params: _init_to_zero(sim._sim, q),  # Init to |0>
    "init |0>": lambda sim, q, **params: _init_to_zero(
        sim._sim, q, forced_outcome=params.get("forced_outcome", -1)
    ),
    "init |1>": lambda sim, q, **params: _init_to_one(sim._sim, q),
    "init |+>": lambda sim, q, **params: _init_to_plus(sim._sim, q),
    "init |->": lambda sim, q, **params: _init_to_minus(sim._sim, q),
    "init |+i>": lambda sim, q, **params: _init_to_plus_i(sim._sim, q),
    "init |-i>": lambda sim, q, **params: _init_to_minus_i(sim._sim, q),
}

__all__ = ["CppSparseSimRs", "gate_dict"]
