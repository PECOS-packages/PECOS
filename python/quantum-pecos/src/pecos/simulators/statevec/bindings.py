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

"""Gate bindings for the state vector simulator.

This module provides the gate bindings that map gate symbols to their corresponding implementations
in the Rust backend for the state vector simulator.
"""

# ruff: noqa: ARG005

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pecos.simulators.statevec.state import StateVec


def get_bindings(state: StateVec) -> dict:
    """Get gate bindings for the state vector simulator.

    Args:
        state: The StateVec instance to bind gates to.

    Returns:
        Dictionary mapping gate symbols to their implementations.
    """
    # Get reference to backend's internal simulator for efficiency
    sim = state.backend._sim  # noqa: SLF001

    return {
        # Single-qubit gates
        "I": lambda s, q, **p: None,
        "X": lambda s, q, **p: sim.run_1q_gate("X", q, p),
        "Y": lambda s, q, **p: sim.run_1q_gate("Y", q, p),
        "Z": lambda s, q, **p: sim.run_1q_gate("Z", q, p),
        "SX": lambda s, q, **p: sim.run_1q_gate("SX", q, p),
        "SXdg": lambda s, q, **p: sim.run_1q_gate("SXdg", q, p),
        "SY": lambda s, q, **p: sim.run_1q_gate("SY", q, p),
        "SYdg": lambda s, q, **p: sim.run_1q_gate("SYdg", q, p),
        "SZ": lambda s, q, **p: sim.run_1q_gate("SZ", q, p),
        "SZdg": lambda s, q, **p: sim.run_1q_gate("SZdg", q, p),
        "H": lambda s, q, **p: sim.run_1q_gate("H", q, p),
        "H1": lambda s, q, **p: sim.run_1q_gate("H", q, p),
        "H2": lambda s, q, **p: sim.run_1q_gate("H2", q, p),
        "H3": lambda s, q, **p: sim.run_1q_gate("H3", q, p),
        "H4": lambda s, q, **p: sim.run_1q_gate("H4", q, p),
        "H5": lambda s, q, **p: sim.run_1q_gate("H5", q, p),
        "H6": lambda s, q, **p: sim.run_1q_gate("H6", q, p),
        "H+z+x": lambda s, q, **p: sim.run_1q_gate("H", q, p),
        "H-z-x": lambda s, q, **p: sim.run_1q_gate("H2", q, p),
        "H+y-z": lambda s, q, **p: sim.run_1q_gate("H3", q, p),
        "H-y-z": lambda s, q, **p: sim.run_1q_gate("H4", q, p),
        "H-x+y": lambda s, q, **p: sim.run_1q_gate("H5", q, p),
        "H-x-y": lambda s, q, **p: sim.run_1q_gate("H6", q, p),
        "F": lambda s, q, **p: sim.run_1q_gate("F", q, p),
        "Fdg": lambda s, q, **p: sim.run_1q_gate("Fdg", q, p),
        "F2": lambda s, q, **p: sim.run_1q_gate("F2", q, p),
        "F2dg": lambda s, q, **p: sim.run_1q_gate("F2dg", q, p),
        "F3": lambda s, q, **p: sim.run_1q_gate("F3", q, p),
        "F3dg": lambda s, q, **p: sim.run_1q_gate("F3dg", q, p),
        "F4": lambda s, q, **p: sim.run_1q_gate("F4", q, p),
        "F4dg": lambda s, q, **p: sim.run_1q_gate("F4dg", q, p),
        "T": lambda s, q, **p: sim.run_1q_gate("T", q, p),
        "Tdg": lambda s, q, **p: sim.run_1q_gate("Tdg", q, p),
        # Two-qubit gates
        "II": lambda s, qs, **p: None,
        "CX": lambda s, qs, **p: sim.run_2q_gate(
            "CX",
            tuple(qs) if isinstance(qs, list) else qs,
            p,
        ),
        "CNOT": lambda s, qs, **p: sim.run_2q_gate(
            "CX",
            tuple(qs) if isinstance(qs, list) else qs,
            p,
        ),
        "CY": lambda s, qs, **p: sim.run_2q_gate(
            "CY",
            tuple(qs) if isinstance(qs, list) else qs,
            p,
        ),
        "CZ": lambda s, qs, **p: sim.run_2q_gate(
            "CZ",
            tuple(qs) if isinstance(qs, list) else qs,
            p,
        ),
        "SXX": lambda s, qs, **p: sim.run_2q_gate(
            "SXX",
            tuple(qs) if isinstance(qs, list) else qs,
            p,
        ),
        "SXXdg": lambda s, qs, **p: sim.run_2q_gate(
            "SXXdg",
            tuple(qs) if isinstance(qs, list) else qs,
            p,
        ),
        "SYY": lambda s, qs, **p: sim.run_2q_gate(
            "SYY",
            tuple(qs) if isinstance(qs, list) else qs,
            p,
        ),
        "SYYdg": lambda s, qs, **p: sim.run_2q_gate(
            "SYYdg",
            tuple(qs) if isinstance(qs, list) else qs,
            p,
        ),
        "SZZ": lambda s, qs, **p: sim.run_2q_gate(
            "SZZ",
            tuple(qs) if isinstance(qs, list) else qs,
            p,
        ),
        "SZZdg": lambda s, qs, **p: sim.run_2q_gate(
            "SZZdg",
            tuple(qs) if isinstance(qs, list) else qs,
            p,
        ),
        "SWAP": lambda s, qs, **p: sim.run_2q_gate(
            "SWAP",
            tuple(qs) if isinstance(qs, list) else qs,
            p,
        ),
        "G": lambda s, qs, **p: sim.run_2q_gate(
            "G2",
            tuple(qs) if isinstance(qs, list) else qs,
            p,
        ),
        "G2": lambda s, qs, **p: sim.run_2q_gate(
            "G2",
            tuple(qs) if isinstance(qs, list) else qs,
            p,
        ),
        # Measurements
        "MZ": lambda s, q, **p: sim.run_1q_gate("MZ", q, p),
        "MX": lambda s, q, **p: sim.run_1q_gate("MX", q, p),
        "MY": lambda s, q, **p: sim.run_1q_gate("MY", q, p),
        "Measure +X": lambda s, q, **p: sim.run_1q_gate("MX", q, p),
        "Measure +Y": lambda s, q, **p: sim.run_1q_gate("MY", q, p),
        "Measure +Z": lambda s, q, **p: sim.run_1q_gate("MZ", q, p),
        "Measure": lambda s, q, **p: sim.run_1q_gate("MZ", q, p),
        "measure Z": lambda s, q, **p: sim.run_1q_gate("MZ", q, p),
        # Projections/Initializations
        "PZ": lambda s, q, **p: sim.run_1q_gate("PZ", q, p),
        "PX": lambda s, q, **p: sim.run_1q_gate("PX", q, p),
        "PY": lambda s, q, **p: sim.run_1q_gate("PY", q, p),
        "PnZ": lambda s, q, **p: sim.run_1q_gate("PnZ", q, p),
        "Init": lambda s, q, **p: sim.run_1q_gate("PZ", q, p),
        "Init +Z": lambda s, q, **p: sim.run_1q_gate("PZ", q, p),
        "Init -Z": lambda s, q, **p: sim.run_1q_gate("PnZ", q, p),
        "Init +X": lambda s, q, **p: sim.run_1q_gate("PX", q, p),
        "Init -X": lambda s, q, **p: sim.run_1q_gate("PnX", q, p),
        "Init +Y": lambda s, q, **p: sim.run_1q_gate("PY", q, p),
        "Init -Y": lambda s, q, **p: sim.run_1q_gate("PnY", q, p),
        "init |0>": lambda s, q, **p: sim.run_1q_gate("PZ", q, p),
        "init |1>": lambda s, q, **p: sim.run_1q_gate("PnZ", q, p),
        "init |+>": lambda s, q, **p: sim.run_1q_gate("PX", q, p),
        "init |->": lambda s, q, **p: sim.run_1q_gate("PnX", q, p),
        "init |+i>": lambda s, q, **p: sim.run_1q_gate("PY", q, p),
        "init |-i>": lambda s, q, **p: sim.run_1q_gate("PnY", q, p),
        "leak": lambda s, q, **p: sim.run_1q_gate("PZ", q, p),
        "leak |0>": lambda s, q, **p: sim.run_1q_gate("PZ", q, p),
        "leak |1>": lambda s, q, **p: sim.run_1q_gate("PnZ", q, p),
        "unleak |0>": lambda s, q, **p: sim.run_1q_gate("PZ", q, p),
        "unleak |1>": lambda s, q, **p: sim.run_1q_gate("PnZ", q, p),
        # Aliases
        "Q": lambda s, q, **p: sim.run_1q_gate("SX", q, p),
        "Qd": lambda s, q, **p: sim.run_1q_gate("SXdg", q, p),
        "R": lambda s, q, **p: sim.run_1q_gate("SY", q, p),
        "Rd": lambda s, q, **p: sim.run_1q_gate("SYdg", q, p),
        "S": lambda s, q, **p: sim.run_1q_gate("SZ", q, p),
        "Sd": lambda s, q, **p: sim.run_1q_gate("SZdg", q, p),
        "F1": lambda s, q, **p: sim.run_1q_gate("F", q, p),
        "F1d": lambda s, q, **p: sim.run_1q_gate("Fdg", q, p),
        "F2d": lambda s, q, **p: sim.run_1q_gate("F2dg", q, p),
        "F3d": lambda s, q, **p: sim.run_1q_gate("F3dg", q, p),
        "F4d": lambda s, q, **p: sim.run_1q_gate("F4dg", q, p),
        "SqrtX": lambda s, q, **p: sim.run_1q_gate("SX", q, p),
        "SqrtXd": lambda s, q, **p: sim.run_1q_gate("SXdg", q, p),
        "SqrtY": lambda s, q, **p: sim.run_1q_gate("SY", q, p),
        "SqrtYd": lambda s, q, **p: sim.run_1q_gate("SYdg", q, p),
        "SqrtZ": lambda s, q, **p: sim.run_1q_gate("SZ", q, p),
        "SqrtZd": lambda s, q, **p: sim.run_1q_gate("SZdg", q, p),
        "SqrtXX": lambda s, qs, **p: sim.run_2q_gate(
            "SXX",
            tuple(qs) if isinstance(qs, list) else qs,
            p,
        ),
        "SqrtYY": lambda s, qs, **p: sim.run_2q_gate(
            "SYY",
            tuple(qs) if isinstance(qs, list) else qs,
            p,
        ),
        "SqrtZZ": lambda s, qs, **p: sim.run_2q_gate(
            "SZZ",
            tuple(qs) if isinstance(qs, list) else qs,
            p,
        ),
        "SqrtXXd": lambda s, qs, **p: sim.run_2q_gate(
            "SXXdg",
            tuple(qs) if isinstance(qs, list) else qs,
            p,
        ),
        "SqrtYYd": lambda s, qs, **p: sim.run_2q_gate(
            "SYYdg",
            tuple(qs) if isinstance(qs, list) else qs,
            p,
        ),
        "SqrtZZd": lambda s, qs, **p: sim.run_2q_gate(
            "SZZdg",
            tuple(qs) if isinstance(qs, list) else qs,
            p,
        ),
        # Rotation gates
        "RX": lambda s, q, **p: sim.run_1q_gate(
            "RX",
            q,
            {"angle": p["angles"][0]} if "angles" in p else {"angle": 0},
        ),
        "RY": lambda s, q, **p: sim.run_1q_gate(
            "RY",
            q,
            {"angle": p["angles"][0]} if "angles" in p else {"angle": 0},
        ),
        "RZ": lambda s, q, **p: sim.run_1q_gate(
            "RZ",
            q,
            {"angle": p["angles"][0]} if "angles" in p else {"angle": 0},
        ),
        "R1XY": lambda s, q, **p: sim.run_1q_gate("R1XY", q, {"angles": p["angles"]}),
        "RXX": lambda s, qs, **p: sim.run_2q_gate(
            "RXX",
            tuple(qs) if isinstance(qs, list) else qs,
            {"angle": p["angles"][0]} if "angles" in p else {"angle": 0},
        ),
        "RYY": lambda s, qs, **p: sim.run_2q_gate(
            "RYY",
            tuple(qs) if isinstance(qs, list) else qs,
            {"angle": p["angles"][0]} if "angles" in p else {"angle": 0},
        ),
        "RZZ": lambda s, qs, **p: sim.run_2q_gate(
            "RZZ",
            tuple(qs) if isinstance(qs, list) else qs,
            {"angle": p["angles"][0]} if "angles" in p else {"angle": 0},
        ),
        "RZZRYYRXX": lambda s, qs, **p: sim.run_2q_gate(
            "RZZRYYRXX",
            tuple(qs) if isinstance(qs, list) else qs,
            {"angles": p["angles"]} if "angles" in p else {"angles": [0, 0, 0]},
        ),
        "R2XXYYZZ": lambda s, qs, **p: sim.run_2q_gate(
            "RZZRYYRXX",
            tuple(qs) if isinstance(qs, list) else qs,
            {"angles": p["angles"]} if "angles" in p else {"angles": [0, 0, 0]},
        ),
    }
