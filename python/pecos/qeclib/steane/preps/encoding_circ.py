# Copyright 2024 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

from pecos.qeclib import qubit
from pecos.slr import Block, Comment, QReg


class EncodingCircuit(Block):
    def __init__(self, q: QReg):
        self.q = q
        super().__init__()
        self.extend(
            Comment("\nEncoding circuit"),
            Comment("---------------"),
            qubit.Prep(
                q[0],
                q[1],
                q[2],
                q[3],
                q[4],
                q[5],
            ),
            Comment("\nq[6] is the input qubit\n"),
            qubit.CX(q[6], q[5]),
            Comment(""),
            qubit.H(q[1]),
            qubit.CX(q[1], q[0]),
            Comment(""),
            qubit.H(q[2]),
            qubit.CX(q[2], q[4]),
            Comment("\n---------------"),
            qubit.H(q[3]),
            qubit.CX(
                (q[3], q[5]),
                (q[2], q[0]),
                (q[6], q[4]),
            ),
            Comment("\n---------------"),
            qubit.CX(
                (q[2], q[6]),
                (q[3], q[4]),
                (q[1], q[5]),
            ),
            Comment("\n---------------"),
            qubit.CX(
                (q[1], q[6]),
                (q[3], q[0]),
            ),
            Comment(""),
        )
