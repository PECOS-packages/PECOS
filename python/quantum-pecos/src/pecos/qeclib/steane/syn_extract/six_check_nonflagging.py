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
from pecos.slr import Block, Comment, CReg, QReg


class SixUnflaggedSyn(Block):
    def __init__(self, data: QReg, ancillas: QReg, syn_x: CReg, syn_z: CReg):
        super().__init__()

        d = data
        a = ancillas

        q = [a[0], d[0], d[1], d[2], d[3], d[4], d[5], d[6], a[1], a[2]]

        self.extend(
            Comment(),
            Comment("Run the 6 non-flagged checks (if non-trivial flags)"),
            Comment("==================================================="),
            Comment("// X check 1, Z check 2, Z check 3"),
            Comment(),
            syn_x.set(0),
            syn_z.set(0),
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
            gq.CX(q[0], q[3]),
            Comment("X1", newline=False),
            gq.CZ(q[8], q[6]),
            Comment("Z2", newline=False),
            gq.CZ(q[9], q[7]),
            Comment("Z3", newline=False),
            Comment(),
            gq.CX(q[0], q[2]),
            Comment("X1", newline=False),
            gq.CZ(q[8], q[3]),
            Comment("Z2", newline=False),
            gq.CZ(q[9], q[6]),
            Comment("Z3", newline=False),
            Comment(),
            gq.CX(q[0], q[4]),
            Comment("X1", newline=False),
            gq.CZ(q[8], q[2]),
            Comment("Z2", newline=False),
            gq.CZ(q[9], q[3]),
            Comment("Z3", newline=False),
            Comment(),
            gq.CX(q[0], q[1]),
            Comment("X1", newline=False),
            gq.CZ(q[8], q[5]),
            Comment("Z2", newline=False),
            gq.CZ(q[9], q[4]),
            Comment("Z3", newline=False),
            Comment(),
            gq.H(
                q[0],
                q[8],
                q[9],
            ),
            Comment(),
            gq.Measure(q[0]) > syn_x[0],
            gq.Measure(q[8]) > syn_z[1],
            gq.Measure(q[9]) > syn_z[2],
            Comment(),
            Comment("// Z check 1, X check 2, X check 3"),
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
            gq.CZ(q[0], q[3]),
            Comment("Z1 0,3", newline=False),
            gq.CX(q[8], q[6]),
            Comment("X2 6,8", newline=False),
            gq.CX(q[9], q[7]),
            Comment("X3 7,9", newline=False),
            Comment(),
            gq.CZ(q[0], q[2]),
            Comment("Z1 0,2", newline=False),
            gq.CX(q[8], q[3]),
            Comment("X2 3,8", newline=False),
            gq.CX(q[9], q[6]),
            Comment("X3 6,9", newline=False),
            Comment(),
            gq.CZ(q[0], q[4]),
            Comment("Z1 0,4", newline=False),
            gq.CX(q[8], q[2]),
            Comment("X2 2,8", newline=False),
            gq.CX(q[9], q[3]),
            Comment("X3 3,9", newline=False),
            Comment(),
            gq.CZ(q[0], q[1]),
            Comment("Z1 0,1", newline=False),
            gq.CX(q[8], q[5]),
            Comment("X2 5,8", newline=False),
            gq.CX(q[9], q[4]),
            Comment("X3 4,9", newline=False),
            Comment(),
            gq.H(
                q[0],
                q[8],
                q[9],
            ),
            Comment(),
            gq.Measure(q[0]) > syn_z[0],
            gq.Measure(q[8]) > syn_x[1],
            gq.Measure(q[9]) > syn_x[2],
            Comment(),
        )
