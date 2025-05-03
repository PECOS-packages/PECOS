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

# ruff: noqa: SLF001

from __future__ import annotations

from typing import Any, List

from pecos_rslib._pecos_rslib import RsStateVec as RustStateVec


class StateVecRs:
    def __init__(self, num_qubits: int):
        """
        Initializes the Rust-backed state vector simulator.

        Args:
            num_qubits (int): The number of qubits in the quantum system.
        """
        self._sim = RustStateVec(num_qubits)
        self.num_qubits = num_qubits
        self.bindings = dict(gate_dict)

    @property
    def vector(self) -> List[complex]:
        raw_vector = self._sim.vector
        # Convert to list of complex numbers
        if isinstance(raw_vector[0], (list, tuple)):
            vector = [complex(r, i) for r, i in raw_vector]
        else:
            vector = list(raw_vector)

        # Convert vector from little-endian to big-endian ordering to match BasicSV
        num_qubits = self.num_qubits

        # Create indices mapping using pure Python
        indices = list(range(len(vector)))
        # Convert indices to binary strings with proper length
        binary_indices = [format(idx, f"0{num_qubits}b") for idx in indices]
        # Reverse bits to change endianness
        reordered_indices = [int(bits[::-1], 2) for bits in binary_indices]

        # Reorder the vector using pure Python
        final_vector = [vector[idx] for idx in reordered_indices]

        return final_vector

    def reset(self):
        """Resets the quantum state to the all-zero state."""
        self._sim.reset()
        return self

    def run_gate(
        self,
        symbol: str,
        locations: set[int] | set[tuple[int, ...]],
        **params: Any,
    ) -> dict[int, int]:
        """
        Applies a gate to the quantum state.

        Args:
            symbol (str): The gate symbol (e.g., "X", "H", "CX").
            location (tuple[int, ...]): The qubit(s) to which the gate is applied.
            params (dict, optional): Parameters for the gate (e.g., rotation angles).

        Returns:
            None
        """
        # self._sim.run_gate(symbol, location, params)
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

                if results:
                    output[location] = results

        return output

    def run_circuit(
        self,
        circuit,
        removed_locations: set[int] | None = None,
    ) -> dict[int, int]:
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
    "I": lambda sim, q, **params: None,
    "X": lambda sim, q, **params: sim._sim.run_1q_gate("X", q, params),
    "Y": lambda sim, q, **params: sim._sim.run_1q_gate("Y", q, params),
    "Z": lambda sim, q, **params: sim._sim.run_1q_gate("Z", q, params),
    "SX": lambda sim, q, **params: sim._sim.run_1q_gate("SX", q, params),
    "SXdg": lambda sim, q, **params: sim._sim.run_1q_gate("SXdg", q, params),
    "SY": lambda sim, q, **params: sim._sim.run_1q_gate("SY", q, params),
    "SYdg": lambda sim, q, **params: sim._sim.run_1q_gate("SYdg", q, params),
    "SZ": lambda sim, q, **params: sim._sim.run_1q_gate("SZ", q, params),
    "SZdg": lambda sim, q, **params: sim._sim.run_1q_gate("SZdg", q, params),
    "H": lambda sim, q, **params: sim._sim.run_1q_gate("H", q, params),
    "H1": lambda sim, q, **params: sim._sim.run_1q_gate("H", q, params),
    "H2": lambda sim, q, **params: sim._sim.run_1q_gate("H2", q, params),
    "H3": lambda sim, q, **params: sim._sim.run_1q_gate("H3", q, params),
    "H4": lambda sim, q, **params: sim._sim.run_1q_gate("H4", q, params),
    "H5": lambda sim, q, **params: sim._sim.run_1q_gate("H5", q, params),
    "H6": lambda sim, q, **params: sim._sim.run_1q_gate("H6", q, params),
    "H+z+x": lambda sim, q, **params: sim._sim.run_1q_gate("H", q, params),
    "H-z-x": lambda sim, q, **params: sim._sim.run_1q_gate("H2", q, params),
    "H+y-z": lambda sim, q, **params: sim._sim.run_1q_gate("H3", q, params),
    "H-y-z": lambda sim, q, **params: sim._sim.run_1q_gate("H4", q, params),
    "H-x+y": lambda sim, q, **params: sim._sim.run_1q_gate("H5", q, params),
    "H-x-y": lambda sim, q, **params: sim._sim.run_1q_gate("H6", q, params),
    "F": lambda sim, q, **params: sim._sim.run_1q_gate("F", q, params),
    "Fdg": lambda sim, q, **params: sim._sim.run_1q_gate("Fdg", q, params),
    "F2": lambda sim, q, **params: sim._sim.run_1q_gate("F2", q, params),
    "F2dg": lambda sim, q, **params: sim._sim.run_1q_gate("F2dg", q, params),
    "F3": lambda sim, q, **params: sim._sim.run_1q_gate("F3", q, params),
    "F3dg": lambda sim, q, **params: sim._sim.run_1q_gate("F3dg", q, params),
    "F4": lambda sim, q, **params: sim._sim.run_1q_gate("F4", q, params),
    "F4dg": lambda sim, q, **params: sim._sim.run_1q_gate("F4dg", q, params),
    "II": lambda sim, qs, **params: None,
    "CX": lambda sim, qs, **params: sim._sim.run_2q_gate("CX", qs, params),
    "CNOT": lambda sim, qs, **params: sim._sim.run_2q_gate("CX", qs, params),
    "CY": lambda sim, qs, **params: sim._sim.run_2q_gate("CY", qs, params),
    "CZ": lambda sim, qs, **params: sim._sim.run_2q_gate("CZ", qs, params),
    "SXX": lambda sim, qs, **params: sim._sim.run_2q_gate("SXX", qs, params),
    "SXXdg": lambda sim, qs, **params: sim._sim.run_2q_gate("SXXdg", qs, params),
    "SYY": lambda sim, qs, **params: sim._sim.run_2q_gate("SYY", qs, params),
    "SYYdg": lambda sim, qs, **params: sim._sim.run_2q_gate("SYYdg", qs, params),
    "SZZ": lambda sim, qs, **params: sim._sim.run_2q_gate("SZZ", qs, params),
    "SZZdg": lambda sim, qs, **params: sim._sim.run_2q_gate("SZZdg", qs, params),
    "SWAP": lambda sim, qs, **params: sim._sim.run_2q_gate("SWAP", qs, params),
    "G": lambda sim, qs, **params: sim._sim.run_2q_gate("G2", qs, params),
    "G2": lambda sim, qs, **params: sim._sim.run_2q_gate("G2", qs, params),
    "MZ": lambda sim, q, **params: sim._sim.run_1q_gate("MZ", q, params),
    "MX": lambda sim, q, **params: sim._sim.run_1q_gate("MX", q, params),
    "MY": lambda sim, q, **params: sim._sim.run_1q_gate("MY", q, params),
    "PZ": lambda sim, q, **params: sim._sim.run_1q_gate("PZ", q, params),
    "PX": lambda sim, q, **params: sim._sim.run_1q_gate("PX", q, params),
    "PY": lambda sim, q, **params: sim._sim.run_1q_gate("PY", q, params),
    "PnZ": lambda sim, q, **params: sim._sim.run_1q_gate("PnZ", q, params),
    "Init": lambda sim, q, **params: sim._sim.run_1q_gate("PZ", q, params),
    "Init +Z": lambda sim, q, **params: sim._sim.run_1q_gate("PZ", q, params),
    "Init -Z": lambda sim, q, **params: sim._sim.run_1q_gate("PnZ", q, params),
    "Init +X": lambda sim, q, **params: sim._sim.run_1q_gate("PX", q, params),
    "Init -X": lambda sim, q, **params: sim._sim.run_1q_gate("PnX", q, params),
    "Init +Y": lambda sim, q, **params: sim._sim.run_1q_gate("PY", q, params),
    "Init -Y": lambda sim, q, **params: sim._sim.run_1q_gate("PnY", q, params),
    "init |0>": lambda sim, q, **params: sim._sim.run_1q_gate("PZ", q, params),
    "init |1>": lambda sim, q, **params: sim._sim.run_1q_gate("PnZ", q, params),
    "init |+>": lambda sim, q, **params: sim._sim.run_1q_gate("PX", q, params),
    "init |->": lambda sim, q, **params: sim._sim.run_1q_gate("PnX", q, params),
    "init |+i>": lambda sim, q, **params: sim._sim.run_1q_gate("PY", q, params),
    "init |-i>": lambda sim, q, **params: sim._sim.run_1q_gate("PnY", q, params),
    "leak": lambda sim, q, **params: sim._sim.run_1q_gate("PZ", q, params),
    "leak |0>": lambda sim, q, **params: sim._sim.run_1q_gate("PZ", q, params),
    "leak |1>": lambda sim, q, **params: sim._sim.run_1q_gate("PnZ", q, params),
    "unleak |0>": lambda sim, q, **params: sim._sim.run_1q_gate("PZ", q, params),
    "unleak |1>": lambda sim, q, **params: sim._sim.run_1q_gate("PnZ", q, params),
    "Measure +X": lambda sim, q, **params: sim._sim.run_1q_gate("MX", q, params),
    "Measure +Y": lambda sim, q, **params: sim._sim.run_1q_gate("MY", q, params),
    "Measure +Z": lambda sim, q, **params: sim._sim.run_1q_gate("MZ", q, params),
    "Q": lambda sim, q, **params: sim._sim.run_1q_gate("SX", q, params),
    "Qd": lambda sim, q, **params: sim._sim.run_1q_gate("SXdg", q, params),
    "R": lambda sim, q, **params: sim._sim.run_1q_gate("SY", q, params),
    "Rd": lambda sim, q, **params: sim._sim.run_1q_gate("SYdg", q, params),
    "S": lambda sim, q, **params: sim._sim.run_1q_gate("SZ", q, params),
    "Sd": lambda sim, q, **params: sim._sim.run_1q_gate("SZdg", q, params),
    "F1": lambda sim, q, **params: sim._sim.run_1q_gate("F", q, params),
    "F1d": lambda sim, q, **params: sim._sim.run_1q_gate("Fdg", q, params),
    "F2d": lambda sim, q, **params: sim._sim.run_1q_gate("F2dg", q, params),
    "F3d": lambda sim, q, **params: sim._sim.run_1q_gate("F3dg", q, params),
    "F4d": lambda sim, q, **params: sim._sim.run_1q_gate("F4dg", q, params),
    "SqrtXX": lambda sim, qs, **params: sim._sim.run_2q_gate("SXX", qs, params),
    "SqrtYY": lambda sim, qs, **params: sim._sim.run_2q_gate("SYY", qs, params),
    "SqrtZZ": lambda sim, qs, **params: sim._sim.run_2q_gate("SZZ", qs, params),
    "Measure": lambda sim, q, **params: sim._sim.run_1q_gate("MZ", q, params),
    "measure Z": lambda sim, q, **params: sim._sim.run_1q_gate("MZ", q, params),
    # "MZForced": lambda sim, q, **params: sim._sim.run_1q_gate("MZForced", q, params),
    # "PZForced": lambda sim, q, **params: sim._sim.run_1q_gate("PZForced", q, params),
    "SqrtXXd": lambda sim, qs, **params: sim._sim.run_2q_gate("SXXdg", qs, params),
    "SqrtYYd": lambda sim, qs, **params: sim._sim.run_2q_gate("SYYdg", qs, params),
    "SqrtZZd": lambda sim, qs, **params: sim._sim.run_2q_gate("SZZdg", qs, params),
    "SqrtX": lambda sim, q, **params: sim._sim.run_1q_gate("SX", q, params),
    "SqrtXd": lambda sim, q, **params: sim._sim.run_1q_gate("SXdg", q, params),
    "SqrtY": lambda sim, q, **params: sim._sim.run_1q_gate("SY", q, params),
    "SqrtYd": lambda sim, q, **params: sim._sim.run_1q_gate("SYdg", q, params),
    "SqrtZ": lambda sim, q, **params: sim._sim.run_1q_gate("SZ", q, params),
    "SqrtZd": lambda sim, q, **params: sim._sim.run_1q_gate("SZdg", q, params),
    "RX": lambda sim, q, **params: sim._sim.run_1q_gate(
        "RX",
        q,
        {"angle": params["angles"][0]} if "angles" in params else {"angle": 0},
    ),
    "RY": lambda sim, q, **params: sim._sim.run_1q_gate(
        "RY",
        q,
        {"angle": params["angles"][0]} if "angles" in params else {"angle": 0},
    ),
    "RZ": lambda sim, q, **params: sim._sim.run_1q_gate(
        "RZ",
        q,
        {"angle": params["angles"][0]} if "angles" in params else {"angle": 0},
    ),
    "R1XY": lambda sim, q, **params: sim._sim.run_1q_gate(
        "R1XY",
        q,
        {"angles": params["angles"]},  # Changed from "angle" to "angles"
    ),
    "T": lambda sim, q, **params: sim._sim.run_1q_gate("T", q, params),
    "Tdg": lambda sim, q, **params: sim._sim.run_1q_gate("Tdg", q, params),
    "RXX": lambda sim, qs, **params: sim._sim.run_2q_gate(
        "RXX",
        qs,
        {"angle": params["angles"][0]} if "angles" in params else {"angle": 0},
    ),
    "RYY": lambda sim, qs, **params: sim._sim.run_2q_gate(
        "RYY",
        qs,
        {"angle": params["angles"][0]} if "angles" in params else {"angle": 0},
    ),
    "RZZ": lambda sim, qs, **params: sim._sim.run_2q_gate(
        "RZZ",
        qs,
        {"angle": params["angles"][0]} if "angles" in params else {"angle": 0},
    ),
    "RZZRYYRXX": lambda sim, qs, **params: sim._sim.run_2q_gate(
        "RZZRYYRXX",
        qs,
        {"angles": params["angles"]} if "angles" in params else {"angles": [0, 0, 0]},
    ),
    "R2XXYYZZ": lambda sim, qs, **params: sim._sim.run_2q_gate(
        "RZZRYYRXX",
        qs,
        {"angles": params["angles"]} if "angles" in params else {"angles": [0, 0, 0]},
    ),
}

# "force output": qmeas.force_output,

__all__ = ["StateVecRs", "gate_dict"]
