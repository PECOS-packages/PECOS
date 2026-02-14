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

"""Inverse gate cancellation optimization pass.

Removes consecutive inverse gate pairs acting on the same qubits.
For example: S-Sdg, T-Tdg, SX-SXdg all cancel to identity.
"""

from __future__ import annotations

from pecos.slr.ast.nodes import GateOp
from pecos.slr.ast.optimizations.base import StatementListOptimizer
from pecos.slr.ast.optimizations.gate_properties import get_inverse, targets_match


class InverseCancellationPass(StatementListOptimizer):
    """Remove consecutive inverse gate pairs acting on the same qubits.

    Gates with explicit inverses (like S and Sdg) cancel when consecutive:
    G * G_dagger = I

    Supported cancellations:
    - S-Sdg, Sdg-S (phase gates)
    - T-Tdg, Tdg-T (T gates)
    - SX-SXdg, SY-SYdg, SZ-SZdg (square root gates)
    - SXX-SXXdg, SYY-SYYdg, SZZ-SZZdg (two-qubit square root gates)
    - F-Fdg, F4-F4dg (face rotation gates)

    Example:
        # Before optimization
        S(q[0]), Sdg(q[0])  # These cancel

        # After optimization
        (empty - both gates removed)
    """

    @property
    def name(self) -> str:
        return "inverse_cancellation"

    def _should_cancel(self, gate1: GateOp, gate2: GateOp) -> bool:
        """Check if gate2 is the inverse of gate1 on the same qubits.

        Gates cancel if:
        1. gate2 is the inverse of gate1
        2. They act on the same qubits in the same order
        """
        inverse = get_inverse(gate1.gate)
        return inverse is not None and gate2.gate == inverse and targets_match(gate1, gate2)
