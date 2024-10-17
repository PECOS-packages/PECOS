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


def noise_tq_depolarizing(op: QOp, p: float, noise_dict: dict):
    rand_nums = np.random.random(len(op.args)) <= p

    if np.any(rand_nums):
        noise = {}
        for r, loc in zip(rand_nums, op.args):
            if r:
                rand = np.random.random()
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
