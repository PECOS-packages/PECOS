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
from pecos.slr import Block, Comment, QReg, util


class EncodingCircuit:
    def __init__(self, q: QReg):
        self.q = q

    def qasm(self):
        qasm = f"""
        // Encoding circuit
        // ---------------
        reset {self.q[0]};
        reset {self.q[1]};
        reset {self.q[2]};
        reset {self.q[3]};
        reset {self.q[4]};
        reset {self.q[5]};

        // q[6] is the input qubit

        cx {self.q[6]},{self.q[5]};

        h {self.q[1]};
        cx {self.q[1]}, {self.q[0]};

        h {self.q[2]};
        cx {self.q[2]}, {self.q[4]};

        // ---------------
        h {self.q[3]};
        cx {self.q[3]}, {self.q[5]};
        cx {self.q[2]}, {self.q[0]};
        cx {self.q[6]}, {self.q[4]};

        // ---------------
        cx {self.q[2]}, {self.q[6]};
        cx {self.q[3]}, {self.q[4]};
        cx {self.q[1]}, {self.q[5]};

        // ---------------
        cx {self.q[1]}, {self.q[6]};
        cx {self.q[3]}, {self.q[0]};
        """

        return util.rm_white_space(qasm)


class EncodingCircuit2(Block):
    def __init__(self, q: QReg):
        if len(q.elems) != 7:
            msg = f"Size of register {len(q.elems)} != 7"
            raise Exception(msg)

        super().__init__(
            Comment("Encoding circuit"),
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
        )
