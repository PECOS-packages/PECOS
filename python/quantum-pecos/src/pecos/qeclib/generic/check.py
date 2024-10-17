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

from pecos.qeclib.qubit import CX, CY, CZ, H, Measure, Prep
from pecos.slr import Barrier, Bit, Block, Comment, Qubit


class Check(Block):
    def __init__(
        self,
        d: list[Qubit],
        paulis: str,
        a: Qubit,
        out: Bit,
    ):
        super().__init__()

        n: int = len(d)

        if n <= 1:
            msg = "Check must be of weight 2 or more"
            raise Exception(msg)

        if len(paulis) == 1:
            paulis *= n
        elif n != len(paulis):
            msg = "Must use a single Pauli or have the same number of Paulis as data qubits."
            raise Exception(msg)

        for p in paulis:
            if p not in ["X", "Y", "Z"]:
                msg = 'Only "X", "Y" and "Z" are accepted.'
                raise Exception(msg)

        ps = paulis

        self.extend(
            Comment(f"Measure check {ps}"),
            Prep(a),
            H(a),
        )

        for i in range(n):
            self.extend(
                Barrier(a, d[i]),  # to preserve order
                self.cp(ps[i], a, d[i]),
                Barrier(a, d[i]),  # to preserve order
            )

        self.extend(
            H(a),
            Measure(a) > out,
        )

    @staticmethod
    def cp(p, a, d):
        if p == "X":
            return CX(a, d)
        elif p == "Y":
            return CY(a, d)
        elif p == "Z":
            return CZ(a, d)
        else:
            msg = f"Symbol '{p}' not supported!"
            raise Exception(msg)
