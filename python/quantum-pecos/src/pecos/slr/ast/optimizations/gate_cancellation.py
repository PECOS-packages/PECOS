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

"""Gate cancellation optimization pass.

Removes consecutive self-inverse gates acting on the same qubits.
For example: X-X, H-H, CX-CX (on same control/target) all cancel to identity.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pecos.slr.ast.optimizations.base import StatementListOptimizer
from pecos.slr.ast.optimizations.gate_properties import is_self_inverse, targets_match

if TYPE_CHECKING:
    from pecos.slr.ast.nodes import GateOp


class GateCancellationPass(StatementListOptimizer):
    """Remove consecutive self-inverse gates acting on the same qubits.

    Self-inverse gates satisfy G * G = I, so consecutive identical gates
    on the same qubits can be removed entirely.

    Supported cancellations:
    - X-X, Y-Y, Z-Z (Pauli gates)
    - H-H (Hadamard)
    - CX-CX, CY-CY, CZ-CZ, CH-CH (two-qubit Clifford gates on same qubits)

    Example:
        # Before optimization
        X(q[0]), X(q[0])  # These cancel

        # After optimization
        (empty - both gates removed)
    """

    @property
    def name(self) -> str:
        return "gate_cancellation"

    def _should_cancel(self, gate1: GateOp, gate2: GateOp) -> bool:
        """Check if two gates cancel (same self-inverse gate on same qubits).

        Gates cancel if:
        1. They are the same gate type
        2. The gate type is self-inverse
        3. They act on the same qubits in the same order
        4. Neither has parameters (parameterized gates handled by rotation_merging)
        """
        return (
            gate1.gate == gate2.gate
            and is_self_inverse(gate1.gate)
            and targets_match(gate1, gate2)
            and not gate1.params
            and not gate2.params
        )
