"""Bitflip noise with leakage for measurement operations.

This module provides noise models for quantum measurement operations that
include both bitflip errors and leakage effects, providing comprehensive
error modeling for measurement processes.
"""

# Copyright 2023 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

from __future__ import annotations

from typing import TYPE_CHECKING

import pecos as pc
from pecos.reps.pyphir.op_types import QOp

if TYPE_CHECKING:
    from pecos.protocols import MachineProtocol


def noise_meas_bitflip_leakage(
    op: QOp,
    p: float,
    machine: MachineProtocol,
) -> list[QOp] | None:
    """Bit-flip noise model for measurements.

    Args:
    ----
        op: Ideal quantum operation.
        p: measurement error rate.
        machine: Machine protocol instance containing leakage state information.
    """
    # Bit flip noise
    # --------------
    # Use fused operation to check and get error indices in one pass
    error_indices = pc.random.compare_indices(len(op.args), p)

    noise = []

    leakded = machine.leaked_qubits & set(op.args)
    if leakded:
        noisy_ops = machine.meas_leaked(leakded)
        noise.extend(noisy_ops)

    if error_indices:
        bitflips = [op.args[idx] for idx in error_indices]

        noisy_op = QOp(
            name="Measure",
            args=list(op.args),
            returns=list(op.returns),
            metadata=dict(op.metadata),
        )
        noisy_op.metadata["bitflips"] = bitflips
        noise.append(noisy_op)

        return noise

    if noise:
        return noise
    return None
