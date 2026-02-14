# Copyright 2026 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Gate properties for optimization passes.

This module defines metadata about quantum gates used by optimization passes
to determine which gates can be cancelled, merged, or simplified.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pecos.slr.ast.nodes import GateKind

if TYPE_CHECKING:
    from pecos.slr.ast.nodes import GateOp

# Gates that are their own inverse (G * G = I)
SELF_INVERSE_GATES: frozenset[GateKind] = frozenset(
    {
        GateKind.X,
        GateKind.Y,
        GateKind.Z,
        GateKind.H,
        GateKind.CX,
        GateKind.CY,
        GateKind.CZ,
        GateKind.CH,
    },
)

# Mapping from gate to its inverse
INVERSE_PAIRS: dict[GateKind, GateKind] = {
    GateKind.S: GateKind.Sdg,
    GateKind.Sdg: GateKind.S,
    GateKind.T: GateKind.Tdg,
    GateKind.Tdg: GateKind.T,
    GateKind.SX: GateKind.SXdg,
    GateKind.SXdg: GateKind.SX,
    GateKind.SY: GateKind.SYdg,
    GateKind.SYdg: GateKind.SY,
    GateKind.SZ: GateKind.SZdg,
    GateKind.SZdg: GateKind.SZ,
    GateKind.SXX: GateKind.SXXdg,
    GateKind.SXXdg: GateKind.SXX,
    GateKind.SYY: GateKind.SYYdg,
    GateKind.SYYdg: GateKind.SYY,
    GateKind.SZZ: GateKind.SZZdg,
    GateKind.SZZdg: GateKind.SZZ,
    GateKind.F: GateKind.Fdg,
    GateKind.Fdg: GateKind.F,
    GateKind.F4: GateKind.F4dg,
    GateKind.F4dg: GateKind.F4,
}

# Rotation gates that can be merged (angle parameters are additive)
ROTATION_GATES: frozenset[GateKind] = frozenset(
    {
        GateKind.RX,
        GateKind.RY,
        GateKind.RZ,
        GateKind.RZZ,
    },
)


def is_self_inverse(gate: GateKind) -> bool:
    """Check if a gate is its own inverse (G * G = I)."""
    return gate in SELF_INVERSE_GATES


def get_inverse(gate: GateKind) -> GateKind | None:
    """Get the inverse of a gate, or None if no explicit inverse is defined."""
    return INVERSE_PAIRS.get(gate)


def is_rotation_gate(gate: GateKind) -> bool:
    """Check if a gate is a parameterized rotation gate."""
    return gate in ROTATION_GATES


def targets_match(gate1: GateOp, gate2: GateOp) -> bool:
    """Check if two gates act on the same qubits in the same order.

    For two-qubit gates, order matters (CX(a,b) != CX(b,a)).
    """
    if len(gate1.targets) != len(gate2.targets):
        return False
    return all(
        t1.allocator == t2.allocator and t1.index == t2.index
        for t1, t2 in zip(gate1.targets, gate2.targets, strict=True)
    )
