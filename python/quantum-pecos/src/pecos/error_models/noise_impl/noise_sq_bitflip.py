"""Single-qubit bitflip noise implementation.

This module provides bitflip noise models for single-qubit operations,
applying X (bitflip) errors to individual qubits with specified
probabilities during quantum computations.
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


def noise_sq_bitflip(op: QOp, p: float) -> list[QOp] | None:
    """The noise model for qubit (re)initialization.

    Args:
    ----
        op: Ideal quantum operation.
        p: Probability of bitflip.
    """
    # Use fused operation to check and get error indices in one pass
    error_indices = pc.random.compare_indices(len(op.args), p)

    if error_indices:
        flip_locs = [op.args[idx] for idx in error_indices]
        return [QOp(name="X", args=flip_locs, metadata={})]

    return None
