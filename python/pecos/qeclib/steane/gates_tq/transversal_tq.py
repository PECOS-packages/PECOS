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
from pecos.slr import Barrier, Block, Comment, QReg


class CX(Block):
    def __init__(self, q1: QReg, q2: QReg, *, barrier=True):
        if len(q1.elems) != 7:
            msg = f"Size of register {len(q1.elems)} != 7"
            raise Exception(msg)

        if len(q2.elems) != 7:
            msg = f"Size of register {len(q2.elems)} != 7"
            raise Exception(msg)

        super().__init__()
        self.extend(
            Comment("Transversal Logical CX"),
        )
        if barrier:
            self.extend(
                Barrier(q1, q2),
            )
        self.extend(
            qubit.CX(
                (q1[0], q2[0]),
                (q1[1], q2[1]),
                (q1[2], q2[2]),
                (q1[3], q2[3]),
                (q1[4], q2[4]),
                (q1[5], q2[5]),
                (q1[6], q2[6]),
            ),
        )
        if barrier:
            self.extend(
                Barrier(q1, q2),
            )


class CY(Block):
    def __init__(self, q1: QReg, q2: QReg):
        if len(q1.elems) != 7:
            msg = f"Size of register {len(q1.elems)} != 7"
            raise Exception(msg)

        if len(q2.elems) != 7:
            msg = f"Size of register {len(q2.elems)} != 7"
            raise Exception(msg)

        super().__init__(
            Comment("Transversal Logical CY"),
            Barrier(q1, q2),
            qubit.CY(
                (q1[0], q2[0]),
                (q1[1], q2[1]),
                (q1[2], q2[2]),
                (q1[3], q2[3]),
                (q1[4], q2[4]),
                (q1[5], q2[5]),
                (q1[6], q2[6]),
            ),
            Barrier(q1, q2),
        )


class CZ(Block):
    def __init__(self, q1: QReg, q2: QReg):
        if len(q1.elems) != 7:
            msg = f"Size of register {len(q1.elems)} != 7"
            raise Exception(msg)

        if len(q2.elems) != 7:
            msg = f"Size of register {len(q2.elems)} != 7"
            raise Exception(msg)

        super().__init__(
            Comment("Transversal Logical CZ"),
            Barrier(q1, q2),
            qubit.CZ(
                (q1[0], q2[0]),
                (q1[1], q2[1]),
                (q1[2], q2[2]),
                (q1[3], q2[3]),
                (q1[4], q2[4]),
                (q1[5], q2[5]),
                (q1[6], q2[6]),
            ),
            Barrier(q1, q2),
        )


class SZZ(Block):
    def __init__(self, q1: QReg, q2: QReg):
        if len(q1.elems) != 7:
            msg = f"Size of register {len(q1.elems)} != 7"
            raise Exception(msg)

        if len(q2.elems) != 7:
            msg = f"Size of register {len(q2.elems)} != 7"
            raise Exception(msg)

        super().__init__(
            Comment("Transversal Logical SZZ"),
            Barrier(q1, q2),
            qubit.SZZ(
                (q1[0], q2[0]),
                (q1[1], q2[1]),
                (q1[2], q2[2]),
                (q1[3], q2[3]),
                (q1[4], q2[4]),
                (q1[5], q2[5]),
                (q1[6], q2[6]),
            ),
            Barrier(q1, q2),
        )
