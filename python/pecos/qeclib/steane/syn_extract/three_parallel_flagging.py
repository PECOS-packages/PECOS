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

from pecos.qeclib import qubit as gq
from pecos.slr import Barrier, Block, Comment, CReg, QReg


class ThreeParallelFlaggingXZZ(Block):
    """
    Args:
        data (QReg[7]):
        ancillas (QReg[3]):
        flag_x (CReg[3]):
        flag_z (CReg[3]):
        flags (CReg[3]):
        last_raw_syn_x (CReg[3]):
        last_raw_syn_z (CReg[3]):
    """

    def __init__(
        self,
        data: QReg,
        ancillas: QReg,
        flag_x: CReg,
        flag_z: CReg,
        flags: CReg,
        last_raw_syn_x: CReg,
        last_raw_syn_z: CReg,
    ):
        super().__init__()
        d = data
        a = ancillas
        q = [a[0], d[0], d[1], d[2], d[3], d[4], d[5], d[6], a[1], a[2]]

        self.extend(
            Comment(),
            flag_x.set(0),
            flag_z.set(0),
            Comment(),
            Comment("X check 1, Z check 2, Z check 3"),
            Comment("==============================="),
            Comment(),
            gq.Prep(
                q[0],
                q[8],
                q[9],
            ),
            Comment(),
            gq.H(
                q[0],
                q[8],
                q[9],
            ),
            Comment(),
            gq.CX(q[0], q[4]),
            Comment("5 -> 4", newline=False),
            gq.CZ(q[8], q[6]),
            Comment("6 -> 6", newline=False),
            gq.CZ(q[9], q[3]),
            Comment("7 -> 3", newline=False),
            Comment(),
            Barrier(q[0], q[8]),
            gq.CZ(q[0], q[8]),
            Barrier(q[0], q[8]),
            Comment(),
            gq.CX(q[0], q[1]),
            Comment("1 -> 1", newline=False),
            gq.CZ(q[8], q[5]),
            Comment("2 -> 5", newline=False),
            gq.CZ(q[9], q[4]),
            Comment("5 -> 4", newline=False),
            Comment(),
            gq.CX(q[0], q[2]),
            Comment("3 -> 2", newline=False),
            gq.CZ(q[8], q[3]),
            Comment("7 -> 3", newline=False),
            gq.CZ(q[9], q[7]),
            Comment("4 -> 7", newline=False),
            Comment(),
            Barrier(q[0], q[9]),
            gq.CZ(q[0], q[9]),
            Barrier(q[0], q[9]),
            Comment(),
            gq.CX(q[0], q[3]),
            Comment("7 -> 3", newline=False),
            gq.CZ(q[8], q[2]),
            Comment("3 -> 2", newline=False),
            gq.CZ(q[9], q[6]),
            Comment("6 -> 6", newline=False),
            Comment(),
            gq.H(
                q[0],
                q[8],
                q[9],
            ),
            Comment(),
            gq.Measure(q[0]) > flag_x[0],
            gq.Measure(q[8]) > flag_z[1],
            gq.Measure(q[9]) > flag_z[2],
            Comment(),
            flag_x[0].set(flag_x[0] ^ last_raw_syn_x[0]),
            flag_z[1].set(flag_z[1] ^ last_raw_syn_z[1]),
            flag_z[2].set(flag_z[2] ^ last_raw_syn_z[2]),
            Comment(),
            flags.set(flag_x | flag_z),
            Comment(),
        )


class ThreeParallelFlaggingZXX(Block):
    def __init__(
        self,
        data: QReg,
        ancillas: QReg,
        flag_x: CReg,
        flag_z: CReg,
        flags: CReg,
        last_raw_syn_x: CReg,
        last_raw_syn_z: CReg,
    ):
        super().__init__()
        d = data
        a = ancillas
        q = [a[0], d[0], d[1], d[2], d[3], d[4], d[5], d[6], a[1], a[2]]

        self.extend(
            Comment(),
            Comment("Z check 1, X check 2, X check 3"),
            Comment("==============================="),
            Comment(),
            gq.Prep(
                q[0],
                q[8],
                q[9],
            ),
            Comment(),
            gq.H(
                q[0],
                q[8],
                q[9],
            ),
            Comment(),
            Barrier(q[0], q[4]),
            gq.CZ(q[0], q[4]),
            Comment("5 -> 4", newline=False),
            Barrier(q[0], q[4]),
            Comment(),
            Barrier(q[8], q[6]),
            gq.CX(q[8], q[6]),
            Comment("6 -> 6", newline=False),
            Barrier(q[8], q[6]),
            Comment(),
            Barrier(q[9], q[3]),
            gq.CX(q[9], q[3]),
            Comment("7 -> 3", newline=False),
            Barrier(q[9], q[3]),
            Comment(),
            Barrier(a[0], d[0], d[1], d[2], d[3], d[4], d[5], d[6], a[1], a[2]),
            gq.CZ(q[8], q[0]),
            Barrier(a[0], d[0], d[1], d[2], d[3], d[4], d[5], d[6], a[1], a[2]),
            Comment(),
            Barrier(q[0], q[1]),
            gq.CZ(q[0], q[1]),
            Comment("1 -> 1", newline=False),
            Barrier(q[0], q[1]),
            Comment(),
            Barrier(q[8], q[5]),
            gq.CX(q[8], q[5]),
            Comment("2 -> 5", newline=False),
            Barrier(q[8], q[5]),
            Comment(),
            Barrier(q[9], q[4]),
            gq.CX(q[9], q[4]),
            Comment("5 -> 4", newline=False),
            Barrier(q[9], q[4]),
            Comment(),
            Barrier(q[0], q[2]),
            gq.CZ(q[0], q[2]),
            Comment("3 -> 2", newline=False),
            Barrier(q[0], q[2]),
            Comment(),
            Barrier(q[8], q[3]),
            gq.CX(q[8], q[3]),
            Comment("7 -> 3", newline=False),
            Barrier(q[8], q[3]),
            Comment(),
            Barrier(q[9], q[7]),
            gq.CX(q[9], q[7]),
            Comment("4 -> 7", newline=False),
            Barrier(q[9], q[7]),
            Comment(),
            Barrier(a[0], d[0], d[1], d[2], d[3], d[4], d[5], d[6], a[1], a[2]),
            gq.CZ(q[9], q[0]),
            Barrier(a[0], d[0], d[1], d[2], d[3], d[4], d[5], d[6], a[1], a[2]),
            Comment(),
            Barrier(q[0], q[3]),
            gq.CZ(q[0], q[3]),
            Comment("7 -> 3", newline=False),
            Barrier(q[0], q[3]),
            Comment(),
            Barrier(q[8], q[2]),
            gq.CX(q[8], q[2]),
            Comment("3 -> 2", newline=False),
            Barrier(q[8], q[2]),
            Comment(),
            Barrier(q[9], q[6]),
            gq.CX(q[9], q[6]),
            Comment("6 -> 6", newline=False),
            Barrier(q[9], q[6]),
            Comment(),
            gq.H(
                q[0],
                q[8],
                q[9],
            ),
            Comment(),
            gq.Measure(q[0]) > flag_z[0],
            gq.Measure(q[8]) > flag_x[1],
            gq.Measure(q[9]) > flag_x[2],
            Comment(),
            Comment("XOR flags/syndromes"),
            flag_z[0].set(flag_z[0] ^ last_raw_syn_z[0]),
            flag_x[1].set(flag_x[1] ^ last_raw_syn_x[1]),
            flag_x[2].set(flag_x[2] ^ last_raw_syn_x[2]),
            Comment(),
            flags.set(flag_x | flag_z),
            Comment(),
        )
