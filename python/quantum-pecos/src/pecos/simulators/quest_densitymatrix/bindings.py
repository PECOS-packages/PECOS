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

"""Gate bindings for the QuEST density matrix simulator.

This module provides the gate bindings that map gate symbols to their corresponding implementations
in the QuEST backend for the density matrix simulator.
"""

# Gate bindings require consistent interfaces even if not all parameters are used.
# This is a design pattern where all gates must have the same signature for polymorphic dispatch.

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pecos.protocols import QuantumBackend
    from pecos.simulators.quest_densitymatrix.state import QuestDensityMatrix


def _init_one(sim: QuestDensityMatrix, q: int, _p: dict[str, Any]) -> None:
    """Initialize qubit to |1⟩ state."""
    # Measure the qubit
    result_dict = sim.run_gate("MZ", {q})
    result = result_dict.get(q, 0) if result_dict else 0
    # If it's 0, flip it to 1
    if result == 0:
        sim.run_gate("X", {q})


def _init_plus(sim: QuestDensityMatrix, q: int, _p: dict[str, Any]) -> None:
    """Initialize qubit to |+⟩ state."""
    sim.reset()  # First reset to |0⟩
    sim.run_gate("H", {q})  # Then apply H to get |+⟩


def _init_minus(sim: QuestDensityMatrix, q: int, _p: dict[str, Any]) -> None:
    """Initialize qubit to |-⟩ state."""
    sim.reset()  # First reset to |0⟩
    sim.run_gate("X", {q})  # Apply X to get |1⟩
    sim.run_gate("H", {q})  # Then apply H to get |-⟩


def _init_plusi(sim: QuestDensityMatrix, q: int, _p: dict[str, Any]) -> None:
    """Initialize qubit to |+i⟩ state."""
    sim.reset()  # First reset to |0⟩
    sim.run_gate("H", {q})  # Apply H to get |+⟩
    sim.run_gate("Sdg", {q})  # Apply S† to get |+i⟩


def _init_minusi(sim: QuestDensityMatrix, q: int, _p: dict[str, Any]) -> None:
    """Initialize qubit to |-i⟩ state."""
    sim.reset()  # First reset to |0⟩
    sim.run_gate("H", {q})  # Apply H to get |+⟩
    sim.run_gate("S", {q})  # Apply S to get |-i⟩


def _rxx_decomposition(
    backend: QuantumBackend,
    qs: int | list[int] | tuple[int, ...],
    p: dict[str, Any],
) -> None:
    """RXX(theta) a, b = SY a; CZ a, b; RX(-theta) b; CZ a, b; SYdg a."""
    q1, q2 = (qs[0], qs[1]) if isinstance(qs, list | tuple) else (qs, qs)
    theta = p["angles"][0] if "angles" in p else p.get("angle", 0)

    # SY a
    backend.sy_gate(q1)
    # CZ a, b
    backend.run_2q_gate("CZ", (q1, q2), None)
    # RX(-theta) b
    backend.run_1q_gate("RX", q2, {"angle": -theta})
    # CZ a, b
    backend.run_2q_gate("CZ", (q1, q2), None)
    # SYdg a
    backend.sydg_gate(q1)


def _ryy_decomposition(
    backend: QuantumBackend,
    qs: int | list[int] | tuple[int, ...],
    p: dict[str, Any],
) -> None:
    """RYY(theta) a, b = SX a; SX b; RZZ(theta) a, b; SXdg a; SXdg b."""
    q1, q2 = (qs[0], qs[1]) if isinstance(qs, list | tuple) else (qs, qs)
    theta = p["angles"][0] if "angles" in p else p.get("angle", 0)

    # SX a; SX b
    backend.sx_gate(q1)
    backend.sx_gate(q2)
    # RZZ(theta) a, b
    _rzz_decomposition(backend, (q1, q2), {"angle": theta})
    # SXdg a; SXdg b
    backend.sxdg_gate(q1)
    backend.sxdg_gate(q2)


def _rzz_decomposition(
    backend: QuantumBackend,
    qs: int | list[int] | tuple[int, ...],
    p: dict[str, Any],
) -> None:
    """RZZ(theta) a, b = H a; H b; RXX(theta) a, b; H a; H b."""
    q1, q2 = (qs[0], qs[1]) if isinstance(qs, list | tuple) else (qs, qs)
    theta = p["angles"][0] if "angles" in p else p.get("angle", 0)

    # H a; H b
    backend.run_1q_gate("H", q1, None)
    backend.run_1q_gate("H", q2, None)
    # RXX(theta) a, b
    _rxx_decomposition(backend, (q1, q2), {"angle": theta})
    # H a; H b
    backend.run_1q_gate("H", q1, None)
    backend.run_1q_gate("H", q2, None)


def _cy_decomposition(
    backend: QuantumBackend,
    qs: int | list[int] | tuple[int, ...],
) -> None:
    """CY = SZdg(q2); CX(q1,q2); SZ(q2) - Note: reversed from trait due to sign convention."""
    q1, q2 = (qs[0], qs[1]) if isinstance(qs, list | tuple) else (qs, qs)

    # SZdg q2
    backend.szdg_gate(q2)
    # CX q1, q2
    backend.run_2q_gate("CX", (q1, q2), None)
    # SZ q2
    backend.sz_gate(q2)


def get_bindings(state: QuestDensityMatrix) -> dict:
    """Get gate bindings for the QuEST density matrix simulator.

    Args:
        state: The QuestDensityMatrix instance to bind gates to.

    Returns:
        Dictionary mapping gate symbols to their implementations.
    """
    # Get reference to backend for efficiency
    backend = state.backend

    return {
        # Single-qubit gates
        "I": lambda _s, _q, **_p: None,
        "X": lambda _s, q, **_p: backend.run_1q_gate("X", q, None),
        "Y": lambda _s, q, **_p: backend.run_1q_gate("Y", q, None),
        "Z": lambda _s, q, **_p: backend.run_1q_gate("Z", q, None),
        "H": lambda _s, q, **_p: backend.run_1q_gate("H", q, None),
        "H1": lambda _s, q, **_p: backend.run_1q_gate("H", q, None),
        "H2": lambda _s, q, **_p: backend.h2_gate(q),
        "H3": lambda _s, q, **_p: backend.h3_gate(q),
        "H4": lambda _s, q, **_p: backend.h4_gate(q),
        "H5": lambda _s, q, **_p: backend.h5_gate(q),
        "H6": lambda _s, q, **_p: backend.h6_gate(q),
        "H+z+x": lambda _s, q, **_p: backend.run_1q_gate("H", q, None),
        "H-z-x": lambda _s, q, **_p: backend.h2_gate(q),
        "H+y-z": lambda _s, q, **_p: backend.h3_gate(q),
        "H-y-z": lambda _s, q, **_p: backend.h4_gate(q),
        "H-x+y": lambda _s, q, **_p: backend.h5_gate(q),
        "H-x-y": lambda _s, q, **_p: backend.h6_gate(q),
        # Square root gates (available from traits)
        "SX": lambda _s, q, **_p: backend.sx_gate(q),
        "SXdg": lambda _s, q, **_p: backend.sxdg_gate(q),
        "SY": lambda _s, q, **_p: backend.sy_gate(q),
        "SYdg": lambda _s, q, **_p: backend.sydg_gate(q),
        "SZ": lambda _s, q, **_p: backend.sz_gate(q),
        "SZdg": lambda _s, q, **_p: backend.szdg_gate(q),
        # Face gates (F gates) - decompositions from traits
        "F": lambda _s, q, **_p: (backend.sx_gate(q), backend.sz_gate(q))[-1] or None,
        "Fdg": lambda _s, q, **_p: (backend.szdg_gate(q), backend.sxdg_gate(q))[-1]
        or None,
        "F2": lambda _s, q, **_p: (backend.sxdg_gate(q), backend.sy_gate(q))[-1]
        or None,
        "F2dg": lambda _s, q, **_p: (backend.sydg_gate(q), backend.sx_gate(q))[-1]
        or None,
        "F3": lambda _s, q, **_p: (backend.sxdg_gate(q), backend.sz_gate(q))[-1]
        or None,
        "F3dg": lambda _s, q, **_p: (backend.szdg_gate(q), backend.sx_gate(q))[-1]
        or None,
        "F4": lambda _s, q, **_p: (backend.sz_gate(q), backend.sx_gate(q))[-1] or None,
        "F4dg": lambda _s, q, **_p: (backend.sxdg_gate(q), backend.szdg_gate(q))[-1]
        or None,
        # Two-qubit gates
        "II": lambda _s, _qs, **_p: None,
        "CX": lambda _s, qs, **_p: backend.run_2q_gate(
            "CX",
            tuple(qs) if isinstance(qs, list) else qs,
            None,
        ),
        "CNOT": lambda _s, qs, **_p: backend.run_2q_gate(
            "CX",
            tuple(qs) if isinstance(qs, list) else qs,
            None,
        ),
        "CY": lambda _s, qs, **_p: _cy_decomposition(backend, qs),
        "CZ": lambda _s, qs, **_p: backend.run_2q_gate(
            "CZ",
            tuple(qs) if isinstance(qs, list) else qs,
            None,
        ),
        # Measurements
        "MZ": lambda _s, q, **_p: backend.run_1q_gate("MZ", q, None),
        "MX": lambda _s, q, **_p: backend.mx_gate(q),
        "MY": lambda _s, q, **_p: backend.my_gate(q),
        "Measure": lambda _s, q, **_p: backend.run_1q_gate("MZ", q, None),
        "measure Z": lambda _s, q, **_p: backend.run_1q_gate("MZ", q, None),
        "Measure +Z": lambda _s, q, **_p: backend.run_1q_gate("MZ", q, None),
        # Projections/Initializations (map to reset for now)
        "PZ": lambda _s, _q, **_p: backend.reset() or None,
        "Init": lambda _s, _q, **_p: backend.reset() or None,
        "Init +Z": lambda _s, _q, **_p: backend.reset() or None,
        "init |0>": lambda _s, _q, **_p: backend.reset() or None,
        # Rotation gates
        "RX": lambda _s, q, **p: backend.run_1q_gate(
            "RX",
            q,
            (
                {"angle": p["angles"][0]}
                if "angles" in p
                else {"angle": p.get("angle", 0)}
            ),
        ),
        "RY": lambda _s, q, **p: backend.run_1q_gate(
            "RY",
            q,
            (
                {"angle": p["angles"][0]}
                if "angles" in p
                else {"angle": p.get("angle", 0)}
            ),
        ),
        "RZ": lambda _s, q, **p: backend.run_1q_gate(
            "RZ",
            q,
            (
                {"angle": p["angles"][0]}
                if "angles" in p
                else {"angle": p.get("angle", 0)}
            ),
        ),
        "R1XY": lambda _s, q, **p: backend.r1xy_gate(
            p["angles"][0] if "angles" in p else p.get("theta", 0),
            (
                p["angles"][1]
                if "angles" in p and len(p["angles"]) > 1
                else p.get("phi", 0)
            ),
            q,
        ),
        "RXX": lambda _s, qs, **p: _rxx_decomposition(backend, qs, p),
        "RYY": lambda _s, qs, **p: _ryy_decomposition(backend, qs, p),
        "RZZ": lambda _s, qs, **p: _rzz_decomposition(backend, qs, p),
        "R2XXYYZZ": lambda _s, qs, **p: backend.rzzryyrxx_gate(
            p["angles"][0] if "angles" in p else 0,
            p["angles"][1] if "angles" in p and len(p["angles"]) > 1 else 0,
            p["angles"][2] if "angles" in p and len(p["angles"]) > 2 else 0,
            qs[0] if isinstance(qs, list | tuple) else qs,
            qs[1] if isinstance(qs, list | tuple) else qs,
        ),
        "RZZRYYRXX": lambda _s, qs, **p: backend.rzzryyrxx_gate(
            p["angles"][0] if "angles" in p else 0,
            p["angles"][1] if "angles" in p and len(p["angles"]) > 1 else 0,
            p["angles"][2] if "angles" in p and len(p["angles"]) > 2 else 0,
            qs[0] if isinstance(qs, list | tuple) else qs,
            qs[1] if isinstance(qs, list | tuple) else qs,
        ),
        # T gates - use RZ implementation instead of trait methods
        "T": lambda _s, q, **_p: backend.run_1q_gate(
            "RZ",
            q,
            {"angle": 0.7853981633974483},
        ),  # π/4
        "TDG": lambda _s, q, **_p: backend.run_1q_gate(
            "RZ",
            q,
            {"angle": -0.7853981633974483},
        ),  # -π/4
        "Tdg": lambda _s, q, **_p: backend.run_1q_gate(
            "RZ",
            q,
            {"angle": -0.7853981633974483},
        ),  # StateVec compatibility
        "TDAGGER": lambda _s, q, **_p: backend.run_1q_gate(
            "RZ",
            q,
            {"angle": -0.7853981633974483},
        ),
        # Two-qubit Clifford gates from traits
        "SXX": lambda _s, qs, **_p: backend.sxx_gate(
            qs[0] if isinstance(qs, list | tuple) else qs,
            qs[1] if isinstance(qs, list | tuple) else qs,
        ),
        "SXXdg": lambda _s, qs, **_p: (
            backend.x(qs[0] if isinstance(qs, list | tuple) else qs),
            backend.x(qs[1] if isinstance(qs, list | tuple) else qs),
            backend.sxx_gate(
                qs[0] if isinstance(qs, list | tuple) else qs,
                qs[1] if isinstance(qs, list | tuple) else qs,
            ),
        )[-1]
        or None,
        "SYY": lambda _s, qs, **_p: backend.syy_gate(
            qs[0] if isinstance(qs, list | tuple) else qs,
            qs[1] if isinstance(qs, list | tuple) else qs,
        ),
        "SYYdg": lambda _s, qs, **_p: (
            backend.y(qs[0] if isinstance(qs, list | tuple) else qs),
            backend.y(qs[1] if isinstance(qs, list | tuple) else qs),
            backend.syy_gate(
                qs[0] if isinstance(qs, list | tuple) else qs,
                qs[1] if isinstance(qs, list | tuple) else qs,
            ),
        )[-1]
        or None,
        "SZZ": lambda _s, qs, **_p: backend.szz_gate(
            qs[0] if isinstance(qs, list | tuple) else qs,
            qs[1] if isinstance(qs, list | tuple) else qs,
        ),
        "SZZdg": lambda _s, qs, **_p: (
            backend.z(qs[0] if isinstance(qs, list | tuple) else qs),
            backend.z(qs[1] if isinstance(qs, list | tuple) else qs),
            backend.szz_gate(
                qs[0] if isinstance(qs, list | tuple) else qs,
                qs[1] if isinstance(qs, list | tuple) else qs,
            ),
        )[-1]
        or None,
        "SWAP": lambda _s, qs, **_p: backend.swap_gate(
            qs[0] if isinstance(qs, list | tuple) else qs,
            qs[1] if isinstance(qs, list | tuple) else qs,
        ),
        "G": lambda _s, qs, **_p: (
            backend.run_2q_gate("CZ", tuple(qs) if isinstance(qs, list) else qs, None),
            backend.run_1q_gate(
                "H",
                qs[0] if isinstance(qs, list | tuple) else qs,
                None,
            ),
            backend.run_1q_gate(
                "H",
                qs[1] if isinstance(qs, list | tuple) else qs,
                None,
            ),
            backend.run_2q_gate("CZ", tuple(qs) if isinstance(qs, list) else qs, None),
        )[-1]
        or None,
        "G2": lambda _s, qs, **_p: (
            backend.run_2q_gate("CZ", tuple(qs) if isinstance(qs, list) else qs, None),
            backend.run_1q_gate(
                "H",
                qs[0] if isinstance(qs, list | tuple) else qs,
                None,
            ),
            backend.run_1q_gate(
                "H",
                qs[1] if isinstance(qs, list | tuple) else qs,
                None,
            ),
            backend.run_2q_gate("CZ", tuple(qs) if isinstance(qs, list) else qs, None),
        )[-1]
        or None,
        # S and S-dagger gates
        "S": lambda _s, q, **_p: backend.s(q),
        "Sdg": lambda _s, q, **_p: backend.sdg(q),
        "SDAG": lambda _s, q, **_p: backend.sdg(q),
        "SDG": lambda _s, q, **_p: backend.sdg(q),
        # Initialization gates for error states
        "Init -Z": lambda s, q, **p: _init_one(s, q, p),
        "Init +X": lambda s, q, **p: _init_plus(s, q, p),
        "Init -X": lambda s, q, **p: _init_minus(s, q, p),
        "Init +Y": lambda s, q, **p: _init_plusi(s, q, p),
        "Init -Y": lambda s, q, **p: _init_minusi(s, q, p),
    }
