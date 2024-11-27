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

from pecos.error_models.noise_impl.noise_initz_bitflip import noise_initz_bitflip
from pecos.reps.pypmir.op_types import QOp


def noise_initz_bitflip_leakage(op: QOp, p: float, machine):
    """The noise model for qubit (re)initialization, including leakage support.

    Args:
    ----
        op: Ideal quantum operation.
        p: Probability of bitflip.
    """
    args = set(op.args)
    leaked = machine.leaked_qubits & args

    noise = []
    noisy_ops = machine.unleak(leaked)
    if noise:
        noise.extend(noisy_ops)

    not_leaked = args - leaked
    if not_leaked:
        remaining_inits = QOp(
            name=op.name,
            args=list(not_leaked),
            metadata=dict(op.metadata),
        )
        noisy_ops = noise_initz_bitflip(remaining_inits, p)
        if noisy_ops:
            noise.extend(noisy_ops)

    return noise
