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

"""QuEST density matrix simulator implementation.

This module provides the QuestDensityMatrix class, a quantum density matrix simulator that uses the QuEST
(Quantum Exact Simulation Toolkit) library as its backend for simulating mixed quantum states and noisy circuits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pecos_rslib.simulators import QuestDensityMatrix as RustQuestDensityMatrix

from pecos.simulators.quest_densitymatrix.bindings import get_bindings

if TYPE_CHECKING:
    from pecos.circuits import QuantumCircuit
    from pecos.circuits.quantum_circuit import ParamGateCollection
    from pecos.typing import SimulatorGateParams


class QuestDensityMatrix:
    """QuEST density matrix simulator.

    A quantum density matrix simulator that uses the QuEST library backend for efficient
    simulation of mixed quantum states and noisy quantum circuits.
    """

    def __init__(self, num_qubits: int, seed: int | None = None) -> None:
        """Initializes the QuEST density matrix simulator.

        Args:
            num_qubits (int): The number of qubits in the quantum system.
            seed (int | None): Optional seed for the random number generator.
        """
        self.backend = RustQuestDensityMatrix(num_qubits, seed)
        self.num_qubits = num_qubits
        self.bindings = get_bindings(self)

    @property
    def matrix(self) -> list[list[complex]]:
        """Get the density matrix as a 2D list of complex numbers.

        Returns:
            2D list of complex amplitudes representing the density matrix.
        """
        # QuEST stores density matrix internally - we need to extract it
        # For now, we'll construct it from probabilities (simplified)
        # A full implementation would extract the full density matrix
        size = 2**self.num_qubits
        matrix = [[complex(0, 0) for _ in range(size)] for _ in range(size)]

        # This is a simplified version - full implementation would extract
        # the actual density matrix elements from QuEST
        for i in range(size):
            prob = self.backend.probability(i)
            if prob > 0:
                # Diagonal elements only for now
                matrix[i][i] = complex(prob, 0)

        return matrix

    def reset(self) -> QuestDensityMatrix:
        """Resets the quantum state to the all-zero density matrix."""
        self.backend.reset()
        return self

    def run_gate(
        self,
        symbol: str,
        locations: set[int] | set[tuple[int, ...]],
        **params: SimulatorGateParams,
    ) -> dict[int, int]:
        """Applies a gate to the quantum density matrix.

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
                    msg = f"Gate {symbol} is not supported in the QuEST density matrix simulator."
                    raise Exception(msg)

                if results is not None:
                    output[location] = results

        return output

    def run_circuit(
        self,
        circuit: QuantumCircuit | ParamGateCollection,
        removed_locations: set[int] | None = None,
    ) -> dict[int, int]:
        """Runs a quantum circuit on the simulator.

        Args:
            circuit: The quantum circuit to run.
            removed_locations: Optional set of locations to exclude.

        Returns:
            Dictionary mapping measurement locations to results.
        """
        if removed_locations is None:
            removed_locations = set()

        output = {}
        for symbol, locations, params in circuit.items():
            results = self.run_gate(
                symbol,
                locations - removed_locations,
                **params,
            )
            if results:
                output.update(results)

        return output

    def __repr__(self) -> str:
        """String representation of the simulator."""
        return f"QuestDensityMatrix(num_qubits={self.num_qubits})"

    def get_probability(self, index: int) -> float:
        """Get the probability of a computational basis state.

        Args:
            index: The basis state index.

        Returns:
            The probability of the given basis state.
        """
        return self.backend.probability(index)

    def prepare_computational_basis(self, index: int) -> None:
        """Prepare a computational basis state.

        Args:
            index: The basis state index to prepare.
        """
        self.backend.prepare_computational_basis(index)
