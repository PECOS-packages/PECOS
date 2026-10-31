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

from math import isfinite, tau
from typing import TYPE_CHECKING, Any

from pecos.simulators.batch_validation import is_empty_batch, replayable_locations, validate_batch

if TYPE_CHECKING:
    from pecos_rslib import GateRegistry

    from pecos.circuits import QuantumCircuit

JSONType = dict[str, Any] | list[Any] | str | int | float | bool | None


class DefaultSimulator:
    """A class providing default method implementations for simulators.

    This class provides default implementations of the SimulatorProtocol interface.
    Simulator implementations can inherit from this class to get the standard
    behavior, or implement the SimulatorProtocol directly for custom behavior.
    """

    gate_capacity: int | None = None
    # Generic user bindings accept sized, indexable locations except strings and
    # mappings. Concrete wrappers restrict this policy to match their dispatch.
    _allow_singletons = True
    _singleton_symbols = frozenset()
    _allow_sequence_locations = True

    def __init__(self) -> None:
        """Initialize the DefaultSimulator.

        Creates an empty bindings dictionary to store gate operation mappings.
        """
        self.bindings = {}

    def validate_gate_batch(self, symbol: str, locations: object, **_params: object) -> None:
        """Validate a batch using this state's capacity and binding location policy."""
        validate_batch(
            symbol,
            locations,
            self.gate_capacity,
            allow_singletons=self._allow_singletons or symbol in self._singleton_symbols,
            allow_sequences=self._allow_sequence_locations,
        )

    def validate_gate_params(self, symbol: str, **params: object) -> None:
        """Check an active registered step's parameters; states without a contract accept all."""

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

        if is_empty_batch(locations):
            return output
        locations = replayable_locations(locations)
        self.validate_gate_batch(symbol, locations, **params)

        if params.get("simulate_gate", True):
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
        gate_registry: GateRegistry | None = None,
    ) -> dict[int | tuple[int, ...], JSONType]:
        """Run a quantum circuit on the simulator.

        If a gate_registry is provided and a gate symbol is registered in it,
        the gate is decomposed into base gates before being passed to run_gate.

        Args:
            circuit (QuantumCircuit): A circuit instance or object with an appropriate items() generator.
            removed_locations (set | None): Optional set of locations to skip when running the circuit.
            gate_registry: Optional GateRegistry for custom gate decomposition at simulation time.

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

            if gate_registry and symbol not in self.bindings and gate_registry.contains(symbol):
                gate_output = self._run_decomposed_gate(gate_registry, symbol, gate_locations, **params)
            else:
                gate_output = self.run_gate(symbol, gate_locations, **params)

            if gate_output:
                output.update(gate_output)

        return output

    def _run_decomposed_gate(
        self,
        gate_registry: GateRegistry,
        symbol: str,
        locations: set[int] | set[tuple[int, ...]],
        **params: JSONType,
    ) -> dict[int | tuple[int, ...], JSONType]:
        """Validate every location's decomposition before running any step."""
        locations = replayable_locations(locations)
        validate_batch(
            f"Gate {symbol}",
            locations,
            self.gate_capacity,
            arity=gate_registry.quantum_arity(symbol),
        )
        # Read angles as run_gate does: `angles`, or `angle` for a one-angle gate.
        input_angles = params.get("angles", ())
        if "angle" in params and "angles" not in params:
            input_angles = (params["angle"],)
        # Simulator bindings use radians; the registry API uses turns.
        turns = []
        for angle in input_angles:
            radians = float(angle)
            if not isfinite(radians):
                msg = f"Gate {symbol}: angle must be finite, got {radians} rad"
                raise ValueError(msg)
            turns.append(radians / tau)
        steps = []
        for location in locations:
            qubits = list(location) if isinstance(location, tuple) else [location]
            steps.extend(gate_registry.decompose(symbol, qubits, turns) or ())

        validated_steps = []
        for step_symbol, step_qubits, step_turns, step_meta in steps:
            # Registry steps are built-in gate types; nested registered gates
            # are not supported by the builder or run_gate dispatch.
            if step_symbol not in self.bindings:
                msg = f"Gate {symbol} step {step_symbol} is not supported in this simulator."
                raise ValueError(msg)
            step_loc = {step_qubits[0] if len(step_qubits) == 1 else tuple(step_qubits)}
            step_params = dict(params)
            step_angles = [angle * tau for angle in step_turns]
            if step_angles:
                step_params["angles"] = tuple(step_angles)
                if len(step_angles) == 1:
                    step_params["angle"] = step_angles[0]
            else:
                step_params.pop("angles", None)
                step_params.pop("angle", None)
            step_params.update(step_meta)
            if not params.get("simulate_gate", True):
                step_params["simulate_gate"] = False
            try:
                self.validate_gate_batch(step_symbol, step_loc, **step_params)
                if step_params.get("simulate_gate", True):
                    self.validate_gate_params(step_symbol, **step_params)
            except ValueError as error:
                msg = f"Gate {symbol} step {step_symbol}: {error}"
                raise ValueError(msg) from error
            validated_steps.append((step_symbol, step_loc, step_params))

        output = {}
        for step_symbol, step_loc, step_params in validated_steps:
            step_output = self.run_gate(step_symbol, step_loc, **step_params)
            if step_output:
                output.update(step_output)
        return output
