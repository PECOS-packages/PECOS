"""Two-qubit depolarizing noise with leakage.

This module provides depolarizing noise models for two-qubit operations
that include leakage to states outside the computational subspace,
providing comprehensive error modeling for two-qubit quantum gates.
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


def noise_tq_depolarizing_leakage(
    op: QOp,
    p: float,
    noise_dict: dict,
    machine: MachineProtocol,
) -> list[QOp] | None:
    """Two-qubit gate depolarizing noise plus leakage."""
    # TODO: precompute, in PyPHIR, a flattened version of args
    args = set()
    for a in op.args:
        for q in a:
            args.add(q)

    leaked = machine.leaked_qubits & args

    # Don't apply a gate if an input qubit has already leaked
    if leaked:
        not_leaked = args - leaked

        # TODO: precompute, in PyPHIR, a flattened version of args
        new_args = []
        for a, b in op.args:
            if a not in not_leaked and b not in leaked:
                new_args.append([a, b])
        op = QOp(name=op.name, args=new_args, metadata=dict(op.metadata))

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
            buffered_ops = []
            for sym, args in noise.items():
                if sym != "L":
                    buffered_ops.append(QOp(name=sym, args=args, metadata={}))
                else:
                    noisy_ops = machine.leak(set(args))
                    buffered_ops.extend(noisy_ops)
            return buffered_ops

    return None
