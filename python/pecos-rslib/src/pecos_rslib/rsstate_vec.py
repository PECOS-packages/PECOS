# Copyright 2024 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Rust-based state vector simulator for PECOS.

This module provides a Python interface to the high-performance Rust implementation of quantum state vector simulation,
enabling efficient quantum circuit simulation with full quantum state representation and support for arbitrary quantum
gates and measurements.
"""

# Gate bindings require consistent interfaces even if not all parameters are used.

from __future__ import annotations

from typing import TYPE_CHECKING

from pecos_rslib._pecos_rslib import RsStateVec

if TYPE_CHECKING:
    from pecos.circuits import QuantumCircuit
    from pecos.typing import SimulatorGateParams


class StateVecRs:
    """Rust-based quantum state vector simulator.

    A high-performance quantum state vector simulator implemented in Rust, providing efficient simulation of arbitrary
    quantum circuits with full quantum state representation and support for complex quantum operations.
    """

    def __init__(self, num_qubits: int, seed: int | None = None) -> None:
        """Initializes the Rust-backed state vector simulator.

        Args:
            num_qubits (int): The number of qubits in the quantum system.
            seed (int | None): Optional seed for the random number generator.
        """
        self._sim = RsStateVec(num_qubits, seed)
        self.num_qubits = num_qubits
        self.bindings = dict(gate_dict)

    @property
    def vector(self):
        """Get the state vector as an Array of complex numbers.

        Returns:
            Array of complex amplitudes representing the quantum state.
        """
        # Use Rust's optimized big-endian conversion
        # This is ~10-100x faster than the previous Python implementation
        return self._sim.vector_big_endian()

    def reset(self) -> StateVecRs:
        """Resets the quantum state to the all-zero state."""
        self._sim.reset()
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
            locations (set[int] | set[tuple[int, ...]]): The qubit(s) to which the gate is applied.
            params (dict, optional): Parameters for the gate (e.g., rotation angles).

        Returns:
            dict[int, int]: Measurement results if applicable, empty dict otherwise.
        """
        # self._sim.run_gate(symbol, location, params)
        output = {}

        if params.get("simulate_gate", True) and locations:
            # Normalize parameters once before loop (not per location)
            if params.get("angles") and len(params["angles"]) == 1:
                params.update({"angle": params["angles"][0]})
            elif "angle" in params and "angles" not in params:
                params["angles"] = (params["angle"],)

            for location in locations:
                # Convert list to tuple if needed (for Rust bindings compatibility)
                loc_to_use = location
                if isinstance(location, list):
                    loc_to_use = tuple(
                        location,
                    )  # Necessary conversion for Rust bindings

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
        circuit: "QuantumCircuit",
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


# Define the gate dictionary
gate_dict = {
    "I": lambda _sim, _q, **_params: None,
    "X": lambda sim, q, **_params: sim._sim.run_1q_gate("X", q, _params),
    "Y": lambda sim, q, **_params: sim._sim.run_1q_gate("Y", q, _params),
    "Z": lambda sim, q, **_params: sim._sim.run_1q_gate("Z", q, _params),
    "SX": lambda sim, q, **_params: sim._sim.run_1q_gate("SX", q, _params),
    "SXdg": lambda sim, q, **_params: sim._sim.run_1q_gate("SXdg", q, _params),
    "SY": lambda sim, q, **_params: sim._sim.run_1q_gate("SY", q, _params),
    "SYdg": lambda sim, q, **_params: sim._sim.run_1q_gate("SYdg", q, _params),
    "SZ": lambda sim, q, **_params: sim._sim.run_1q_gate("SZ", q, _params),
    "SZdg": lambda sim, q, **_params: sim._sim.run_1q_gate("SZdg", q, _params),
    "H": lambda sim, q, **_params: sim._sim.run_1q_gate("H", q, _params),
    "H1": lambda sim, q, **_params: sim._sim.run_1q_gate("H", q, _params),
    "H2": lambda sim, q, **_params: sim._sim.run_1q_gate("H2", q, _params),
    "H3": lambda sim, q, **_params: sim._sim.run_1q_gate("H3", q, _params),
    "H4": lambda sim, q, **_params: sim._sim.run_1q_gate("H4", q, _params),
    "H5": lambda sim, q, **_params: sim._sim.run_1q_gate("H5", q, _params),
    "H6": lambda sim, q, **_params: sim._sim.run_1q_gate("H6", q, _params),
    "H+z+x": lambda sim, q, **_params: sim._sim.run_1q_gate("H", q, _params),
    "H-z-x": lambda sim, q, **_params: sim._sim.run_1q_gate("H2", q, _params),
    "H+y-z": lambda sim, q, **_params: sim._sim.run_1q_gate("H3", q, _params),
    "H-y-z": lambda sim, q, **_params: sim._sim.run_1q_gate("H4", q, _params),
    "H-x+y": lambda sim, q, **_params: sim._sim.run_1q_gate("H5", q, _params),
    "H-x-y": lambda sim, q, **_params: sim._sim.run_1q_gate("H6", q, _params),
    "F": lambda sim, q, **_params: sim._sim.run_1q_gate("F", q, _params),
    "Fdg": lambda sim, q, **_params: sim._sim.run_1q_gate("Fdg", q, _params),
    "F2": lambda sim, q, **_params: sim._sim.run_1q_gate("F2", q, _params),
    "F2dg": lambda sim, q, **_params: sim._sim.run_1q_gate("F2dg", q, _params),
    "F3": lambda sim, q, **_params: sim._sim.run_1q_gate("F3", q, _params),
    "F3dg": lambda sim, q, **_params: sim._sim.run_1q_gate("F3dg", q, _params),
    "F4": lambda sim, q, **_params: sim._sim.run_1q_gate("F4", q, _params),
    "F4dg": lambda sim, q, **_params: sim._sim.run_1q_gate("F4dg", q, _params),
    "II": lambda _sim, _qs, **_params: None,
    "CX": lambda sim, qs, **_params: sim._sim.run_2q_gate(
        "CX",
        tuple(qs) if isinstance(qs, list) else qs,
        _params,
    ),
    "CNOT": lambda sim, qs, **_params: sim._sim.run_2q_gate(
        "CX",
        tuple(qs) if isinstance(qs, list) else qs,
        _params,
    ),
    "CY": lambda sim, qs, **_params: sim._sim.run_2q_gate(
        "CY",
        tuple(qs) if isinstance(qs, list) else qs,
        _params,
    ),
    "CZ": lambda sim, qs, **_params: sim._sim.run_2q_gate(
        "CZ",
        tuple(qs) if isinstance(qs, list) else qs,
        _params,
    ),
    "SXX": lambda sim, qs, **_params: sim._sim.run_2q_gate(
        "SXX",
        tuple(qs) if isinstance(qs, list) else qs,
        _params,
    ),
    "SXXdg": lambda sim, qs, **_params: sim._sim.run_2q_gate(
        "SXXdg",
        tuple(qs) if isinstance(qs, list) else qs,
        _params,
    ),
    "SYY": lambda sim, qs, **_params: sim._sim.run_2q_gate(
        "SYY",
        tuple(qs) if isinstance(qs, list) else qs,
        _params,
    ),
    "SYYdg": lambda sim, qs, **_params: sim._sim.run_2q_gate(
        "SYYdg",
        tuple(qs) if isinstance(qs, list) else qs,
        _params,
    ),
    "SZZ": lambda sim, qs, **_params: sim._sim.run_2q_gate(
        "SZZ",
        tuple(qs) if isinstance(qs, list) else qs,
        _params,
    ),
    "SZZdg": lambda sim, qs, **_params: sim._sim.run_2q_gate(
        "SZZdg",
        tuple(qs) if isinstance(qs, list) else qs,
        _params,
    ),
    "SWAP": lambda sim, qs, **_params: sim._sim.run_2q_gate(
        "SWAP",
        tuple(qs) if isinstance(qs, list) else qs,
        _params,
    ),
    "G": lambda sim, qs, **_params: sim._sim.run_2q_gate(
        "G2",
        tuple(qs) if isinstance(qs, list) else qs,
        _params,
    ),
    "G2": lambda sim, qs, **_params: sim._sim.run_2q_gate(
        "G2",
        tuple(qs) if isinstance(qs, list) else qs,
        _params,
    ),
    "MZ": lambda sim, q, **_params: sim._sim.run_1q_gate("MZ", q, _params),
    "MX": lambda sim, q, **_params: sim._sim.run_1q_gate("MX", q, _params),
    "MY": lambda sim, q, **_params: sim._sim.run_1q_gate("MY", q, _params),
    "PZ": lambda sim, q, **_params: sim._sim.run_1q_gate("PZ", q, _params),
    "PX": lambda sim, q, **_params: sim._sim.run_1q_gate("PX", q, _params),
    "PY": lambda sim, q, **_params: sim._sim.run_1q_gate("PY", q, _params),
    "PnZ": lambda sim, q, **_params: sim._sim.run_1q_gate("PnZ", q, _params),
    "Init": lambda sim, q, **_params: sim._sim.run_1q_gate("PZ", q, _params),
    "Init +Z": lambda sim, q, **_params: sim._sim.run_1q_gate("PZ", q, _params),
    "Init -Z": lambda sim, q, **_params: sim._sim.run_1q_gate("PnZ", q, _params),
    "Init +X": lambda sim, q, **_params: sim._sim.run_1q_gate("PX", q, _params),
    "Init -X": lambda sim, q, **_params: sim._sim.run_1q_gate("PnX", q, _params),
    "Init +Y": lambda sim, q, **_params: sim._sim.run_1q_gate("PY", q, _params),
    "Init -Y": lambda sim, q, **_params: sim._sim.run_1q_gate("PnY", q, _params),
    "init |0>": lambda sim, q, **_params: sim._sim.run_1q_gate("PZ", q, _params),
    "init |1>": lambda sim, q, **_params: sim._sim.run_1q_gate("PnZ", q, _params),
    "init |+>": lambda sim, q, **_params: sim._sim.run_1q_gate("PX", q, _params),
    "init |->": lambda sim, q, **_params: sim._sim.run_1q_gate("PnX", q, _params),
    "init |+i>": lambda sim, q, **_params: sim._sim.run_1q_gate("PY", q, _params),
    "init |-i>": lambda sim, q, **_params: sim._sim.run_1q_gate("PnY", q, _params),
    "leak": lambda sim, q, **_params: sim._sim.run_1q_gate("PZ", q, _params),
    "leak |0>": lambda sim, q, **_params: sim._sim.run_1q_gate("PZ", q, _params),
    "leak |1>": lambda sim, q, **_params: sim._sim.run_1q_gate("PnZ", q, _params),
    "unleak |0>": lambda sim, q, **_params: sim._sim.run_1q_gate("PZ", q, _params),
    "unleak |1>": lambda sim, q, **_params: sim._sim.run_1q_gate("PnZ", q, _params),
    "Measure +X": lambda sim, q, **_params: sim._sim.run_1q_gate("MX", q, _params),
    "Measure +Y": lambda sim, q, **_params: sim._sim.run_1q_gate("MY", q, _params),
    "Measure +Z": lambda sim, q, **_params: sim._sim.run_1q_gate("MZ", q, _params),
    "Q": lambda sim, q, **_params: sim._sim.run_1q_gate("SX", q, _params),
    "Qd": lambda sim, q, **_params: sim._sim.run_1q_gate("SXdg", q, _params),
    "R": lambda sim, q, **_params: sim._sim.run_1q_gate("SY", q, _params),
    "Rd": lambda sim, q, **_params: sim._sim.run_1q_gate("SYdg", q, _params),
    "S": lambda sim, q, **_params: sim._sim.run_1q_gate("SZ", q, _params),
    "Sd": lambda sim, q, **_params: sim._sim.run_1q_gate("SZdg", q, _params),
    "F1": lambda sim, q, **_params: sim._sim.run_1q_gate("F", q, _params),
    "F1d": lambda sim, q, **_params: sim._sim.run_1q_gate("Fdg", q, _params),
    "F2d": lambda sim, q, **_params: sim._sim.run_1q_gate("F2dg", q, _params),
    "F3d": lambda sim, q, **_params: sim._sim.run_1q_gate("F3dg", q, _params),
    "F4d": lambda sim, q, **_params: sim._sim.run_1q_gate("F4dg", q, _params),
    "SqrtXX": lambda sim, qs, **_params: sim._sim.run_2q_gate(
        "SXX",
        tuple(qs) if isinstance(qs, list) else qs,
        _params,
    ),
    "SqrtYY": lambda sim, qs, **_params: sim._sim.run_2q_gate(
        "SYY",
        tuple(qs) if isinstance(qs, list) else qs,
        _params,
    ),
    "SqrtZZ": lambda sim, qs, **_params: sim._sim.run_2q_gate(
        "SZZ",
        tuple(qs) if isinstance(qs, list) else qs,
        _params,
    ),
    "Measure": lambda sim, q, **_params: sim._sim.run_1q_gate("MZ", q, _params),
    "measure Z": lambda sim, q, **_params: sim._sim.run_1q_gate("MZ", q, _params),
    # "MZForced": lambda sim, q, **_params: sim._sim.run_1q_gate("MZForced", q, _params),
    # "PZForced": lambda sim, q, **_params: sim._sim.run_1q_gate("PZForced", q, _params),
    "SqrtXXd": lambda sim, qs, **_params: sim._sim.run_2q_gate(
        "SXXdg",
        tuple(qs) if isinstance(qs, list) else qs,
        _params,
    ),
    "SqrtYYd": lambda sim, qs, **_params: sim._sim.run_2q_gate(
        "SYYdg",
        tuple(qs) if isinstance(qs, list) else qs,
        _params,
    ),
    "SqrtZZd": lambda sim, qs, **_params: sim._sim.run_2q_gate(
        "SZZdg",
        tuple(qs) if isinstance(qs, list) else qs,
        _params,
    ),
    "SqrtX": lambda sim, q, **_params: sim._sim.run_1q_gate("SX", q, _params),
    "SqrtXd": lambda sim, q, **_params: sim._sim.run_1q_gate("SXdg", q, _params),
    "SqrtY": lambda sim, q, **_params: sim._sim.run_1q_gate("SY", q, _params),
    "SqrtYd": lambda sim, q, **_params: sim._sim.run_1q_gate("SYdg", q, _params),
    "SqrtZ": lambda sim, q, **_params: sim._sim.run_1q_gate("SZ", q, _params),
    "SqrtZd": lambda sim, q, **_params: sim._sim.run_1q_gate("SZdg", q, _params),
    "RX": lambda sim, q, **_params: sim._sim.run_1q_gate(
        "RX",
        q,
        {"angle": _params["angles"][0]} if "angles" in _params else {"angle": 0},
    ),
    "RY": lambda sim, q, **_params: sim._sim.run_1q_gate(
        "RY",
        q,
        {"angle": _params["angles"][0]} if "angles" in _params else {"angle": 0},
    ),
    "RZ": lambda sim, q, **_params: sim._sim.run_1q_gate(
        "RZ",
        q,
        {"angle": _params["angles"][0]} if "angles" in _params else {"angle": 0},
    ),
    "R1XY": lambda sim, q, **_params: sim._sim.run_1q_gate(
        "R1XY",
        q,
        {"angles": _params["angles"]},  # Changed from "angle" to "angles"
    ),
    "T": lambda sim, q, **_params: sim._sim.run_1q_gate("T", q, _params),
    "Tdg": lambda sim, q, **_params: sim._sim.run_1q_gate("Tdg", q, _params),
    "RXX": lambda sim, qs, **_params: sim._sim.run_2q_gate(
        "RXX",
        tuple(qs) if isinstance(qs, list) else qs,
        {"angle": _params["angles"][0]} if "angles" in _params else {"angle": 0},
    ),
    "RYY": lambda sim, qs, **_params: sim._sim.run_2q_gate(
        "RYY",
        tuple(qs) if isinstance(qs, list) else qs,
        {"angle": _params["angles"][0]} if "angles" in _params else {"angle": 0},
    ),
    "RZZ": lambda sim, qs, **_params: sim._sim.run_2q_gate(
        "RZZ",
        tuple(qs) if isinstance(qs, list) else qs,
        {"angle": _params["angles"][0]} if "angles" in _params else {"angle": 0},
    ),
    "RZZRYYRXX": lambda sim, qs, **_params: sim._sim.run_2q_gate(
        "RZZRYYRXX",
        tuple(qs) if isinstance(qs, list) else qs,
        {"angles": _params["angles"]} if "angles" in _params else {"angles": [0, 0, 0]},
    ),
    "R2XXYYZZ": lambda sim, qs, **_params: sim._sim.run_2q_gate(
        "RZZRYYRXX",
        tuple(qs) if isinstance(qs, list) else qs,
        {"angles": _params["angles"]} if "angles" in _params else {"angles": [0, 0, 0]},
    ),
}

# "force output": qmeas.force_output,

__all__ = ["StateVecRs", "gate_dict"]
