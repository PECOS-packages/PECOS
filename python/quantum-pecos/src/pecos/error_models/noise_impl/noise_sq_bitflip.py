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


def noise_sq_bitflip(op: QOp, p: float):
    """The noise model for qubit (re)initialization.

    Args:
    ----
        op: Ideal quantum operation.
        p: Probability of bitflip.
    """
    rand_nums = np.random.random(len(op.args)) <= p

    if np.any(rand_nums):
        flip_locs = []
        for r, loc in zip(rand_nums, op.args):
            if r:
                flip_locs.append(loc)

            return [QOp(name="X", args=flip_locs, metadata={})]
        return None

    else:
        return None
