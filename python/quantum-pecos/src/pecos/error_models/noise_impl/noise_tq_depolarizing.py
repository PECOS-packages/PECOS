"""Two-qubit depolarizing noise implementation.

This module provides depolarizing noise models for two-qubit operations,
applying random two-qubit Pauli errors to qubit pairs during
two-qubit gate operations like CNOT and CZ gates.
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


def noise_tq_depolarizing(op: QOp, p: float, noise_dict: dict) -> list[QOp] | None:
    """Apply two-qubit depolarizing noise to quantum operation.

    Applies depolarizing noise to two-qubit operations by randomly selecting
    fault combinations from the noise dictionary and applying them to the
    individual qubits of each two-qubit pair.

    Args:
        op: Quantum operation to apply noise to.
        p: Probability of noise occurring on each qubit pair.
        noise_dict: Dictionary mapping fault pair tuples to their probabilities.

    Returns:
        List of single-qubit noise operations to apply,
        or None if no noise is applied.

    Raises:
        NotImplementedError: If leakage faults are encountered.
    """
    # Use fused operation to check and get error indices in one pass
    error_indices = pc.random.compare_indices(len(op.args), p)

    if error_indices:
        noise = {}
        for idx in error_indices:
            loc = op.args[idx]
            rand = pc.random.random(1)[0]
            p_tot = 0.0
            for (fault1, fault2), prob in noise_dict.items():
                p_tot += prob

                if p_tot >= rand:
                    loc1, loc2 = loc
                    if fault1 != "I":
                        noise.setdefault(fault1, []).append(loc1)
                    if fault2 != "I":
                        noise.setdefault(fault2, []).append(loc2)
                    break

        if noise:
            if "L" in noise:
                msg = "Not implemented yet!"
                raise NotImplementedError(msg)

            buffered_ops = []
            for sym, args in noise.items():
                buffered_ops.append(QOp(name=sym, args=args, metadata={}))
            return buffered_ops

    return None
