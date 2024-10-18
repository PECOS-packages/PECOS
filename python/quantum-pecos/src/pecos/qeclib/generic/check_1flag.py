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

from pecos.qeclib.qubit import CH, CX, CY, CZ, H, Measure, Prep
from pecos.slr import Barrier, Bit, Block, Comment, Qubit


class Check1Flag(Block):
    def __init__(
        self,
        d: list[Qubit],
        ops: str,
        a: Qubit,
        flag: Qubit,
        out: Bit,
        out_flag: Bit,
    ):
        super().__init__()

        n: int = len(d)

        if n <= 2:
            msg = "Check must be of weight 3 or more"
            raise Exception(msg)

        if len(ops) == 1:
            ops *= n
        elif n != len(ops):
            msg = "Must use a single operator or have the same number of operators as data qubits."
            raise Exception(msg)

        for o in ops:
            if o not in ["X", "Y", "Z", "H"]:
                msg = 'Only "X", "Y", "Z", and "H" are accepted.'
                raise Exception(msg)

        self.extend(
            Comment(f"Measure check {ops}"),
            Prep(a, flag),
            H(a),
            Barrier(a, d[0]),
            self.cu(ops[0], a, d[0]),
            Barrier(a, flag),
            CX(a, flag),
            Barrier(a, flag),
        )

        for i in range(1, n - 1):
            self.extend(
                self.cu(ops[i], a, d[i]),
                Barrier(a, d[i]),  # To preserve order
            )

        self.extend(
            Barrier(a, flag),
            CX(a, flag),
            Barrier(a, flag),
            self.cu(ops[-1], a, d[-1]),
            Barrier(a, d[-1]),
            H(a),
            Measure(a) > out,
            Measure(flag) > out_flag,
        )

    @staticmethod
    def cu(u, a, d):
        if u == "X":
            return CX(a, d)
        elif u == "Y":
            return CY(a, d)
        elif u == "Z":
            return CZ(a, d)
        elif u == "H":
            return CH(a, d)
        else:
            msg = f"Symbol '{u}' not supported!"
            raise Exception(msg)
