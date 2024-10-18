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

from typing import Any

from pecos_rslib._pecos_rslib import SparseSim as RustSparseSim


class SparseSimRs:
    def __init__(self, num_qubits: int):
        self._sim = RustSparseSim(num_qubits)
        self.num_qubits = num_qubits
        self.bindings = dict(gate_dict)

    def reset(self):
        self._sim.reset()
        return self

    def run_gate(
        self,
        symbol: str,
        locations: set[int] | set[tuple[int, ...]],
        **params: Any,
    ) -> dict[int, int]:
        output = {}

        if params.get("simulate_gate", True) and locations:
            for location in locations:
                if isinstance(location, int):
                    location = (location,)

                if params.get("angles") and len(params["angles"]) == 1:
                    params.update({"angle": params["angles"][0]})
                elif "angle" in params and "angles" not in params:
                    params["angles"] = (params["angle"],)

                if symbol in self.bindings:
                    results = self.bindings[symbol](self, location, **params)
                else:
                    results = self._sim.run_gate(symbol, location, params)

                if results:
                    output.update(results)

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

    def add_faults(self, circuit, removed_locations: set[int] | None = None) -> None:
        self.run_circuit(circuit, removed_locations)

    # def print_stabs(self, *, verbose: bool = True, print_y: bool = True, print_destabs: bool = False) -> list[str]:
    #     return self._sim.print_stabs(verbose, print_y, print_destabs)

    @property
    def stabs(self):
        return TableauWrapper(self._sim, is_stab=True)

    @property
    def destabs(self):
        return TableauWrapper(self._sim, is_stab=False)

    def print_stabs(
        self,
        *,
        verbose: bool = True,
        print_y: bool = True,
        print_destabs: bool = False,
    ):
        stabs = self._sim.stab_tableau()
        if print_destabs:
            destabs = self._sim.destab_tableau()
            if verbose:
                print("Stabilizers:")
                print(stabs)
                print("Destabilizers:")
                print(destabs)
            return stabs, destabs
        else:
            if verbose:
                print("Stabilizers:")
                print(stabs)
            return stabs

    def logical_sign(self, logical_op):
        # This method needs to be implemented based on the Python version
        # It might require additional Rust functions to be exposed
        msg = "logical_sign method not implemented yet"
        raise NotImplementedError(msg)

    def refactor(self, xs, zs, choose=None, prefer=None, protected=None):
        # This method needs to be implemented based on the Python version
        # It might require additional Rust functions to be exposed
        msg = "refactor method not implemented yet"
        raise NotImplementedError(msg)

    def find_stab(self, xs, zs):
        # This method needs to be implemented based on the Python version
        # It might require additional Rust functions to be exposed
        msg = "find_stab method not implemented yet"
        raise NotImplementedError(msg)

    def copy(self):
        # This method needs to be implemented
        # It might require an additional Rust function to be exposed
        msg = "copy method not implemented yet"
        raise NotImplementedError(msg)


class TableauWrapper:
    def __init__(self, sim, *, is_stab: bool):
        self._sim = sim
        self._is_stab = is_stab

    def print_tableau(self, *, verbose: bool = False) -> list[str]:
        if self._is_stab:
            tableau = self._sim.stab_tableau()
        else:
            tableau = self._sim.destab_tableau()

        lines = tableau.strip().split("\n")
        adjusted_lines = [
            adjust_tableau_string(line, is_stab=self._is_stab) for line in lines
        ]

        if verbose:
            for line in adjusted_lines:
                print(line)

        return adjusted_lines


def adjust_tableau_string(line: str, *, is_stab: bool) -> str:
    """
    Adjust the tableau string to ensure the sign part always takes up two spaces
    and convert 'Y' to 'W'. For destabilizers, always use two spaces for the sign.

    Args:
        line (str): A single line from the tableau string.
        is_stab (bool): True if this is a stabilizer, False if destabilizer.

    Returns:
        str: The adjusted line with proper spacing for signs and 'W' instead of 'Y'.
    """
    if is_stab:
        if line.startswith("+i"):
            adjusted = " i" + line[2:]
        elif line.startswith("-i"):
            adjusted = "-i" + line[2:]
        elif line.startswith("+"):
            adjusted = "  " + line[1:]
        elif line.startswith("-"):
            adjusted = " -" + line[1:]
        else:
            adjusted = "  " + line  # Default case, shouldn't happen with correct input
    else:
        # For destabilizers, always use two spaces for the sign
        adjusted = "  " + line[1:]

    return adjusted.replace("Y", "W")


# Define the gate dictionary
gate_dict = {
    "I": lambda sim, q, **params: None,
    "X": lambda sim, q, **params: sim._sim.run_gate("X", q, params),
    "Y": lambda sim, q, **params: sim._sim.run_gate("Y", q, params),
    "Z": lambda sim, q, **params: sim._sim.run_gate("Z", q, params),
    "SX": lambda sim, q, **params: sim._sim.run_gate("SX", q, params),
    "SXdg": lambda sim, q, **params: sim._sim.run_gate("SXdg", q, params),
    "SY": lambda sim, q, **params: sim._sim.run_gate("SY", q, params),
    "SYdg": lambda sim, q, **params: sim._sim.run_gate("SYdg", q, params),
    "SZ": lambda sim, q, **params: sim._sim.run_gate("SZ", q, params),
    "SZdg": lambda sim, q, **params: sim._sim.run_gate("SZdg", q, params),
    "H": lambda sim, q, **params: sim._sim.run_gate("H", q, params),
    "H2": lambda sim, q, **params: sim._sim.run_gate("H2", q, params),
    "H3": lambda sim, q, **params: sim._sim.run_gate("H3", q, params),
    "H4": lambda sim, q, **params: sim._sim.run_gate("H4", q, params),
    "H5": lambda sim, q, **params: sim._sim.run_gate("H5", q, params),
    "H6": lambda sim, q, **params: sim._sim.run_gate("H6", q, params),
    "F": lambda sim, q, **params: sim._sim.run_gate("F", q, params),
    "Fdg": lambda sim, q, **params: sim._sim.run_gate("Fdg", q, params),
    "F2": lambda sim, q, **params: sim._sim.run_gate("F2", q, params),
    "F2dg": lambda sim, q, **params: sim._sim.run_gate("F2dg", q, params),
    "F3": lambda sim, q, **params: sim._sim.run_gate("F3", q, params),
    "F3dg": lambda sim, q, **params: sim._sim.run_gate("F3dg", q, params),
    "F4": lambda sim, q, **params: sim._sim.run_gate("F4", q, params),
    "F4dg": lambda sim, q, **params: sim._sim.run_gate("F4dg", q, params),
    "II": lambda sim, qs, **params: None,
    "CX": lambda sim, qs, **params: sim._sim.run_gate("CX", qs, params),
    "CNOT": lambda sim, qs, **params: sim._sim.run_gate("CX", qs, params),
    "CY": lambda sim, qs, **params: sim._sim.run_gate("CY", qs, params),
    "CZ": lambda sim, qs, **params: sim._sim.run_gate("CZ", qs, params),
    "SXX": lambda sim, qs, **params: sim._sim.run_gate("SXX", qs, params),
    "SXXdg": lambda sim, qs, **params: sim._sim.run_gate("SXXdg", qs, params),
    "SYY": lambda sim, qs, **params: sim._sim.run_gate("SYY", qs, params),
    "SYYdg": lambda sim, qs, **params: sim._sim.run_gate("SYYdg", qs, params),
    "SZZ": lambda sim, qs, **params: sim._sim.run_gate("SZZ", qs, params),
    "SZZdg": lambda sim, qs, **params: sim._sim.run_gate("SZZdg", qs, params),
    "SWAP": lambda sim, qs, **params: sim._sim.run_gate("SWAP", qs, params),
    "G": lambda sim, qs, **params: sim._sim.run_gate("G2", qs, params),
    "G2": lambda sim, qs, **params: sim._sim.run_gate("G2", qs, params),
    "MZ": lambda sim, q, **params: sim._sim.run_gate("MZ", q, params),
    "MX": lambda sim, q, **params: sim._sim.run_gate("MX", q, params),
    "MY": lambda sim, q, **params: sim._sim.run_gate("MY", q, params),
    "PZ": lambda sim, q, **params: sim._sim.run_gate("PZ", q, params),
    "PX": lambda sim, q, **params: sim._sim.run_gate("PX", q, params),
    "PY": lambda sim, q, **params: sim._sim.run_gate("PY", q, params),
    "PnZ": lambda sim, q, **params: sim._sim.run_gate("PnZ", q, params),
    "Init +Z": lambda sim, q, **params: sim._sim.run_gate("PZ", q, params),
    "Init -Z": lambda sim, q, **params: sim._sim.run_gate("PnZ", q, params),
    "Init +X": lambda sim, q, **params: sim._sim.run_gate("PX", q, params),
    "Init -X": lambda sim, q, **params: sim._sim.run_gate("PnX", q, params),
    "Init +Y": lambda sim, q, **params: sim._sim.run_gate("PY", q, params),
    "Init -Y": lambda sim, q, **params: sim._sim.run_gate("PnY", q, params),
    "init |0>": lambda sim, q, **params: sim._sim.run_gate("PZ", q, params),
    "init |1>": lambda sim, q, **params: sim._sim.run_gate("PnZ", q, params),
    "init |+>": lambda sim, q, **params: sim._sim.run_gate("PX", q, params),
    "init |->": lambda sim, q, **params: sim._sim.run_gate("PnX", q, params),
    "init |+i>": lambda sim, q, **params: sim._sim.run_gate("PY", q, params),
    "init |-i>": lambda sim, q, **params: sim._sim.run_gate("PnY", q, params),
    "leak": lambda sim, q, **params: sim._sim.run_gate("PZ", q, params),
    "leak |0>": lambda sim, q, **params: sim._sim.run_gate("PZ", q, params),
    "leak |1>": lambda sim, q, **params: sim._sim.run_gate("PnZ", q, params),
    "unleak |0>": lambda sim, q, **params: sim._sim.run_gate("PZ", q, params),
    "unleak |1>": lambda sim, q, **params: sim._sim.run_gate("PnZ", q, params),
    "Measure +X": lambda sim, q, **params: sim._sim.run_gate("MX", q, params),
    "Measure +Y": lambda sim, q, **params: sim._sim.run_gate("MY", q, params),
    "Measure +Z": lambda sim, q, **params: sim._sim.run_gate("MZ", q, params),
    "Q": lambda sim, q, **params: sim._sim.run_gate("SX", q, params),
    "Qd": lambda sim, q, **params: sim._sim.run_gate("SXdg", q, params),
    "R": lambda sim, q, **params: sim._sim.run_gate("SY", q, params),
    "Rd": lambda sim, q, **params: sim._sim.run_gate("SYdg", q, params),
    "S": lambda sim, q, **params: sim._sim.run_gate("SZ", q, params),
    "Sd": lambda sim, q, **params: sim._sim.run_gate("SZdg", q, params),
    "H1": lambda sim, q, **params: sim._sim.run_gate("H1", q, params),
    "F1": lambda sim, q, **params: sim._sim.run_gate("F", q, params),
    "F1d": lambda sim, q, **params: sim._sim.run_gate("Fdg", q, params),
    "F2d": lambda sim, q, **params: sim._sim.run_gate("F2dg", q, params),
    "F3d": lambda sim, q, **params: sim._sim.run_gate("F3dg", q, params),
    "F4d": lambda sim, q, **params: sim._sim.run_gate("F4dg", q, params),
    "SqrtXX": lambda sim, qs, **params: sim._sim.run_gate("SXX", qs, params),
    "SqrtYY": lambda sim, qs, **params: sim._sim.run_gate("SYY", qs, params),
    "SqrtZZ": lambda sim, qs, **params: sim._sim.run_gate("SZZ", qs, params),
    "measure Z": lambda sim, q, **params: sim._sim.run_gate("MZ", q, params),
    "MZForced": lambda sim, q, **params: sim._sim.run_gate("MZForced", q, params),
    "PZForced": lambda sim, q, **params: sim._sim.run_gate("PZForced", q, params),
}

# "force output": qmeas.force_output,

__all__ = ["SparseSimRs", "gate_dict"]
