"""Bitflip noise implementation for measurement operations.

This module provides noise models for quantum measurement operations,
applying bitflip errors to measurement results to simulate measurement
errors in quantum error correction protocols.
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

import pecos as pc
from pecos.reps.pyphir.op_types import QOp


def noise_meas_bitflip(op: QOp, p: float) -> list[QOp] | None:
    """Bit-flip noise model for measurements.

    Args:
    ----
        op: Ideal quantum operation.
        p: measurement error rate.
    """
    # Bit flip noise
    # --------------
    # Use fused operation to check and get error indices in one pass
    error_indices = pc.random.compare_indices(len(op.args), p)

    if error_indices:
        bitflips = [op.args[idx] for idx in error_indices]

        # The name stays hardcoded here, unlike the leakage sibling which preserves it.
        # Preserving it emits "Measure +Z", which four simulators reject or silently skip
        # because only sparsestab and pauliprop expand the shared alias table -- see the
        # alias-normalisation issue. Metadata is copied so the bitflips write below does
        # not mutate the input, testing None explicitly rather than falsiness.
        noisy_op = QOp(
            name="Measure",
            args=list(op.args),
            returns=list(op.returns),
            metadata={} if op.metadata is None else dict(op.metadata),
        )
        noisy_op.metadata["bitflips"] = bitflips
        return [noisy_op]

    return None
