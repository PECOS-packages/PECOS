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

"""QuEST state vector simulator implementation.

This module provides the QuestStateVec class, a quantum state vector simulator that uses the QuEST
(Quantum Exact Simulation Toolkit) library as its backend for efficient quantum circuit simulation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pecos_rslib.simulators import QuestStateVec as RustQuestStateVec

import pecos as pc
from pecos.simulators.quest_statevec.bindings import get_bindings

if TYPE_CHECKING:
    from pecos.circuits import QuantumCircuit
    from pecos.circuits.quantum_circuit import ParamGateCollection
    from pecos.typing import SimulatorGateParams


class QuestStateVec:
    """QuEST state vector simulator.

    A quantum state vector simulator that uses the QuEST library backend for efficient
    simulation of arbitrary quantum circuits with full quantum state representation.
    """

    def __init__(self, num_qubits: int, seed: int | None = None) -> None:
        """Initializes the QuEST state vector simulator.

        Args:
            num_qubits (int): The number of qubits in the quantum system.
            seed (int | None): Optional seed for the random number generator.
        """
        self.backend = RustQuestStateVec(num_qubits, seed)
        self.num_qubits = num_qubits
        self.bindings = get_bindings(self)

    @property
    def vector(self) -> Array:  # noqa: F821 - Array is a forward reference
        """Get the state vector as an Array of complex numbers.

        Returns:
            Array of complex amplitudes representing the quantum state.
        """
        # QuEST stores amplitudes internally - we need to extract them
        amplitudes = []
        for i in range(2**self.num_qubits):
            re, im = self.backend.get_amplitude(i)
            amplitudes.append(complex(re, im))
        return pc.array(amplitudes, dtype=pc.dtypes.complex128)

    def reset(self) -> QuestStateVec:
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
                    msg = f"Gate {symbol} is not supported in the QuEST simulator."
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
        return f"QuestStateVec(num_qubits={self.num_qubits})"

    def get_probability(self, index: int) -> float:
        """Get the probability of a computational basis state.

        Args:
            index: The basis state index.

        Returns:
            The probability of the given basis state.
        """
        return self.backend.probability(index)

    def get_amplitude(self, index: int) -> complex:
        """Get the amplitude of a computational basis state.

        Args:
            index: The basis state index.

        Returns:
            The complex amplitude of the given basis state.
        """
        re, im = self.backend.get_amplitude(index)
        return complex(re, im)
