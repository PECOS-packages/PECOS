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

"""Coin toss simulator for PECOS.

This module provides a Python interface to the high-performance Rust implementation of a coin toss quantum simulator.
The simulator ignores all quantum gates and returns random measurement results based on a configurable probability,
making it useful for debugging classical logic paths and testing error correction protocols with random noise.
"""

from __future__ import annotations

from operator import index
from typing import TYPE_CHECKING

from pecos_rslib.simulators import CoinToss as RustCoinToss

from pecos.simulators.batch_validation import GATE_ARITIES, replayable_locations, validate_batch

if TYPE_CHECKING:
    from collections.abc import Iterable

    from pecos.circuits import QuantumCircuit
    from pecos.typing import SimulatorGateParams


class CoinToss:
    """Rust-based coin toss quantum simulator.

    A high-performance coin toss simulator implemented in Rust that ignores all quantum operations and uses
    coin tosses for measurements. This is useful for debugging classical branches in quantum algorithms
    and testing error correction protocols with random noise.
    """

    def __init__(
        self,
        num_qubits: int,
        prob: float = 0.5,
        seed: int | None = None,
    ) -> None:
        """Initializes the Rust-backed coin toss simulator.

        Args:
            num_qubits (int): The number of qubits in the quantum system.
            prob (float): Probability of measuring |1> (default: 0.5).
            seed (int | None): Optional seed for the random number generator.
        """
        self._sim = RustCoinToss(num_qubits, prob, seed)
        self.gate_capacity = num_qubits
        self.num_qubits = num_qubits
        self.bindings = dict(gate_dict)

    @property
    def prob(self) -> float:
        """Get the current measurement probability."""
        return self._sim.prob

    @prob.setter
    def prob(self, value: float) -> None:
        """Set the measurement probability."""
        self._sim.prob = value

    def reset(self) -> CoinToss:
        """Reset the simulator (no-op for coin toss, but maintains interface compatibility).

        Returns:
            CoinToss: Returns self for method chaining.
        """
        self._sim.reset()
        return self

    def set_seed(self, seed: int) -> None:
        """Set the seed for reproducible randomness.

        Args:
            seed (int): Seed value for the random number generator.
        """
        self._sim.set_seed(seed)

    def mz(self, qubit: int) -> int:
        """Perform a coin toss measurement on the given qubit.

        Args:
            qubit (int): The qubit index to measure.

        Returns:
            int: The measurement result (0 or 1).
        """
        result = self._sim.run_measure(qubit)
        return next(iter(result.values())) if result else 0

    def is_measurement(self, symbol: str) -> bool:
        """Recognize built-in measuring bindings and unbound "measure" prefixes.

        Custom callables may also measure; dispatch always calls the binding.
        HybridEngine validates unbound prefixed spellings before rejecting them
        as unsupported when enabled.
        """
        binding = self.bindings.get(symbol)
        if binding is not None:
            return binding in (_measure_gate, _measure_z_gate, _measure_forced_gate)
        return symbol.lower().startswith("measure")

    def validate_gate_batch(self, symbol: str, locations: int | Iterable[int], **params: object) -> None:
        """Validate known arities before dispatch, preserving built-in no-op inputs.

        A custom binding's arity is unknown unless its spelling supplies one.
        Built-in measuring bindings always take one scalar, even when aliased.
        """
        binding = self.bindings.get(symbol)
        if binding is _noop_gate:
            return
        arity = GATE_ARITIES.get(symbol) if binding is not None else None
        if self.is_measurement(symbol):
            arity = 1
        if arity is None and symbol.lower().startswith("measure"):
            arity = 1
        if arity is not None:
            validate_batch(symbol, _locations_batch(locations), self.gate_capacity, allow_singletons=False, arity=arity)
        if binding in (_measure_z_gate, _measure_forced_gate):
            _forced_outcome(symbol, params, required=binding is _measure_forced_gate)

    def run_gate(
        self,
        symbol: str,
        locations: int | Iterable[int],
        **params: SimulatorGateParams,
    ) -> dict[int, int]:
        """Dispatch bindings in iteration order, recording nonzero results.

        Args:
            symbol: Bound spellings call their binding; unbound "measure" prefixes measure.
            locations: A qubit index or an iterable of qubit indices. Duplicates
                draw another coin; one-shot iterators are buffered for validation.
            **params: ``simulate_gate=False`` validates but draws no coins.
                Other parameters are forwarded to the binding.

        Returns:
            dict[int, int]: Nonzero binding results, keyed by the original location.
        """
        binding = self.bindings.get(symbol)
        if binding is None:
            if not self.is_measurement(symbol):
                return {}
            binding = _measure_gate
        locations = _locations_batch(locations)
        self.validate_gate_batch(symbol, locations, **params)
        measurement_results = {}
        if params.get("simulate_gate", True):
            for location in locations:
                result = binding(self, location, **params)
                if result:
                    measurement_results[location] = result
        return measurement_results

    def run_circuit(self, circuit: QuantumCircuit) -> dict[int, int]:
        """Dispatch each circuit item through its bindings and collect nonzero results.

        Args:
            circuit: The quantum circuit to execute. Each measurement batch is
                validated before drawing coins, even with ``simulate_gate=False``.

        Returns:
            dict[int, int]: Dictionary mapping qubit indices to measurement results (1 for |1>).
        """
        measurement_results = {}

        # Each item validates and dispatches through the same binding path.
        for gate_data in circuit.items():
            # Gate data is a tuple: (gate_symbol, locations, params)
            if isinstance(gate_data, tuple) and len(gate_data) >= 2:
                gate_symbol, locations = gate_data[0], gate_data[1]

                params = gate_data[2] if len(gate_data) >= 3 else {}
                measurement_results.update(self.run_gate(gate_symbol, locations, **params))

        return measurement_results


def _locations_batch(locations: int | Iterable[int]) -> Iterable[int]:
    """Keep integer-like scalars and their original result keys intact."""
    try:
        index(locations)
    except TypeError:
        return replayable_locations(locations)
    return (locations,)


def _forced_outcome(symbol: str, params: dict, *, required: bool = False) -> int | None:
    """Match native forced-outcome values: -1 draws a coin, 0/1 do not."""
    if required and "forced_outcome" not in params:
        msg = f"{symbol} requires a 'forced_outcome' parameter"
        raise ValueError(msg)
    msg = f"{symbol} requires 'forced_outcome' to be -1 (random), 0, or 1"
    try:
        outcome = index(params.get("forced_outcome", -1))
    except TypeError as exc:
        raise ValueError(msg) from exc
    if outcome not in (-1, 0, 1):
        raise ValueError(msg)
    return None if outcome == -1 else outcome


def _measure_z_gate(state: CoinToss, qubit: int, **params: SimulatorGateParams) -> int:
    """Measure Z, honouring an optional native forced outcome."""
    outcome = _forced_outcome("MZ", params)
    return state.mz(qubit) if outcome is None else outcome


def _measure_forced_gate(state: CoinToss, qubit: int, **params: SimulatorGateParams) -> int:
    """Require a native forced outcome, with -1 selecting a random coin."""
    outcome = _forced_outcome("MZForced", params, required=True)
    return state.mz(qubit) if outcome is None else outcome


# Gate dictionary mapping gate symbols to no-op functions
# This maintains compatibility with the expected gate bindings interface
def _noop_gate(*args: object, **kwargs: object) -> None:
    """No-operation function for all gates."""


def _measure_gate(state: CoinToss, qubit: int, **_params: SimulatorGateParams) -> int:
    """Return |1> with probability state.prob or |0> otherwise."""
    return state.mz(qubit)


gate_dict = {
    # All gates are no-ops except measurements
    # Single-qubit gates
    "I": _noop_gate,
    "X": _noop_gate,
    "Y": _noop_gate,
    "Z": _noop_gate,
    "H": _noop_gate,
    "S": _noop_gate,
    "Sd": _noop_gate,
    "SX": _noop_gate,
    "SY": _noop_gate,
    "SZ": _noop_gate,
    "SXdg": _noop_gate,
    "SYdg": _noop_gate,
    "SZdg": _noop_gate,
    "T": _noop_gate,
    "Tdg": _noop_gate,
    # Face gates
    "F": _noop_gate,
    "Fdg": _noop_gate,
    "F1": _noop_gate,
    "F1d": _noop_gate,
    "F2": _noop_gate,
    "F2d": _noop_gate,
    "F3": _noop_gate,
    "F3d": _noop_gate,
    "F4": _noop_gate,
    "F4d": _noop_gate,
    # Hadamard variants
    "H1": _noop_gate,
    "H2": _noop_gate,
    "H3": _noop_gate,
    "H4": _noop_gate,
    "H5": _noop_gate,
    "H6": _noop_gate,
    "H+z+x": _noop_gate,
    "H-z-x": _noop_gate,
    "H+y-z": _noop_gate,
    "H-y-z": _noop_gate,
    "H-x+y": _noop_gate,
    "H-x-y": _noop_gate,
    # Rotation gates
    "RX": _noop_gate,
    "RY": _noop_gate,
    "RZ": _noop_gate,
    "RXY1Q": _noop_gate,
    "R1XY": _noop_gate,
    "U1q": _noop_gate,
    # Other gates
    "Q": _noop_gate,
    "Qd": _noop_gate,
    "R": _noop_gate,
    "Rd": _noop_gate,
    # Two-qubit gates
    "CX": _noop_gate,
    "CY": _noop_gate,
    "CZ": _noop_gate,
    "SWAP": _noop_gate,
    "CNOT": _noop_gate,
    # Entangling gates
    "SXX": _noop_gate,
    "SYY": _noop_gate,
    "SZZ": _noop_gate,
    "SXXdg": _noop_gate,
    "SYYdg": _noop_gate,
    "SZZdg": _noop_gate,
    "SqrtZZ": _noop_gate,
    # Rotation gates
    "RXX": _noop_gate,
    "RYY": _noop_gate,
    "RZZ": _noop_gate,
    "RXYXY2Q": _noop_gate,
    "RXXRYYRZZ": _noop_gate,
    "R2XXYYZZ": _noop_gate,
    # Other gates
    "G": _noop_gate,
    "II": _noop_gate,
    # Initialization gates
    "Init": _noop_gate,
    "Init +Z": _noop_gate,
    "Init -Z": _noop_gate,
    "Init +X": _noop_gate,
    "Init -X": _noop_gate,
    "Init +Y": _noop_gate,
    "Init -Y": _noop_gate,
    "init |0>": _noop_gate,
    "init |1>": _noop_gate,
    # Leak gates
    "leak": _noop_gate,
    "leak |0>": _noop_gate,
    "leak |1>": _noop_gate,
    "unleak |0>": _noop_gate,
    "unleak |1>": _noop_gate,
    # Measurements - these actually return random results
    "Measure": _measure_z_gate,
    "measure Z": _measure_z_gate,
    "Measure +Z": _measure_z_gate,
    "Measure -Z": _measure_gate,
    "Measure +X": _measure_gate,
    "Measure -X": _measure_gate,
    "Measure +Y": _measure_gate,
    "Measure -Y": _measure_gate,
    "MZForced": _measure_forced_gate,
    "measure X": _measure_gate,
    "measure Y": _measure_gate,
    "MX": _measure_gate,
    "MY": _measure_gate,
    "MZ": _measure_z_gate,
}
