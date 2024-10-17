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

import numpy as np

from pecos.reps.pypmir.op_types import QOp


def noise_meas_bitflip(op: QOp, p: float):
    """Bit-flip noise model for measurements.

    Args:
    ----
        op: Ideal quantum operation.
        p: measurement error rate.
    """
    # Bit flip noise
    # --------------
    rand_nums = np.random.random(len(op.args)) <= p

    noise = []

    if np.any(rand_nums):
        bitflips = []
        for r, loc in zip(rand_nums, op.args):
            if r:
                bitflips.append(loc)

        noisy_op = QOp(
            name="Measure",
            args=list(op.args),
            returns=list(op.returns),
            metadata=dict(op.metadata),
        )
        noisy_op.metadata["bitflips"] = bitflips
        noise.append(noisy_op)
        return noise

    else:
        return None
