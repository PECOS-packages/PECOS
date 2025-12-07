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

"""State vector simulator implementation.

This module provides the StateVec class, a quantum state vector simulator that uses a high-performance Rust backend
for efficient quantum circuit simulation with full quantum state representation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pecos_rslib.simulators import StateVec as StateVecRs

from pecos.simulators.statevec.bindings import get_bindings

if TYPE_CHECKING:
    from pecos.circuits import QuantumCircuit
    from pecos.circuits.quantum_circuit import ParamGateCollection
    from pecos.typing import SimulatorGateParams


class StateVec:
    """Quantum state vector simulator.

    A quantum state vector simulator that uses a high-performance Rust backend (StateVecRs) for efficient
    simulation of arbitrary quantum circuits with full quantum state representation.
    """

    def __init__(self, num_qubits: int, seed: int | None = None) -> None:
        """Initializes the state vector simulator.

        Args:
            num_qubits (int): The number of qubits in the quantum system.
            seed (int | None): Optional seed for the random number generator.
        """
        self.backend = StateVecRs(num_qubits, seed)
        self.num_qubits = num_qubits
        self.bindings = get_bindings(self)

    @property
    def vector(self) -> Array:  # noqa: F821 - Array is a forward reference
        """Get the state vector as an Array of complex numbers.

        Returns:
            Array of complex amplitudes representing the quantum state.
        """
        return self.backend.vector_big_endian()

    def reset(self) -> StateVec:
        """Resets the quantum state to the all-zero state."""
        self.backend.reset()
        return self

    def run_gate(
        self,
        symbol: str,
        locations: set[int] | set[tuple[int, ...]],
        **params: SimulatorGateParams,
    ) -> dict[int, int]:
        """Applies a gate to the quantum state.

        Args:
            symbol (str): The gate symbol (e.g., "X", "H", "CX").
            locations (set): The qubit(s) to which the gate is applied.
            params (dict, optional): Parameters for the gate (e.g., rotation angles).

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

                # Convert list to tuple if needed (for Rust bindings compatibility)
                loc_to_use = location
                if isinstance(location, list):
                    loc_to_use = tuple(location)

                if symbol in self.bindings:
                    results = self.bindings[symbol](self, loc_to_use, **params)
                else:
                    msg = f"Gate {symbol} is not supported in this simulator."
                    raise Exception(msg)

                if results:
                    output[location] = results

        return output

    def run_circuit(
        self,
        circuit: QuantumCircuit | ParamGateCollection,
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
