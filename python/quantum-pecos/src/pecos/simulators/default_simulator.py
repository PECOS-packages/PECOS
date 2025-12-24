# Copyright 2018 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Default simulator implementation for PECOS quantum error correction.

This module provides a generic simulator interface that can dynamically select
appropriate simulator backends based on quantum circuit requirements and available
system resources.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pecos.circuits import QuantumCircuit

JSONType = dict[str, Any] | list[Any] | str | int | float | bool | None


class DefaultSimulator:
    """A class providing default method implementations for simulators.

    This class provides default implementations of the SimulatorProtocol interface.
    Simulator implementations can inherit from this class to get the standard
    behavior, or implement the SimulatorProtocol directly for custom behavior.
    """

    def __init__(self) -> None:
        """Initialize the DefaultSimulator.

        Creates an empty bindings dictionary to store gate operation mappings.
        """
        self.bindings = {}

    def run_gate(
        self,
        symbol: str,
        locations: set[int] | set[tuple[int, ...]],
        **params: JSONType,
    ) -> dict[int | tuple[int, ...], JSONType]:
        """Run a gate operation on the simulator.

        Args:
        ----
            symbol: The gate symbol/name to execute.
            locations: Set of qubit indices or tuples of indices where the gate should be applied.
            **params: Additional parameters for the gate operation.

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

                # TODO: get params var value ... -> result = {'sym':, 'index':, 'result': result, 'qubit': location}

                if results:
                    output[location] = results

        return output

    def run_circuit(
        self,
        circuit: QuantumCircuit,
        removed_locations: set | None = None,
    ) -> dict[int | tuple[int, ...], JSONType]:
        """Run a quantum circuit on the simulator.

        Args:
            circuit (QuantumCircuit): A circuit instance or object with an appropriate items() generator.
            removed_locations (set | None): Optional set of locations to skip when running the circuit.

        Returns:
            dict[int | tuple[int, ...], JSONType]: Circuit output. Note that this output format may differ
            from what a ``circuit_runner`` will return for the same method named ``run_circuit``.

        """
        output = {}

        for symbol, locations, params in circuit.items():
            gate_locations = locations
            if removed_locations is not None:
                gate_locations = set(locations) - removed_locations
                # TODO: need to handle multi-qubit ops that are partially removed

            gate_output = self.run_gate(symbol, gate_locations, **params)

            if gate_output:
                output.update(gate_output)

        return output
