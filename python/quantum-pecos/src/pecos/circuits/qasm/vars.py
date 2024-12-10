# Copyright 2021 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.


from pecos.circuits.qasm import expr


class Var:
    def __eq__(self, other):
        return expr.Equiv(self, other)

    def __ne__(self, other):
        return expr.NE(self, other)

    def __ge__(self, other):
        return expr.GE(self, other)

    def __gt__(self, other):
        return expr.GT(self, other)

    def __le__(self, other):
        return expr.LE(self, other)

    def __lt__(self, other):
        return expr.LT(self, other)

    def __xor__(self, other):
        return expr.XOR(self, other)


class Reg(Var):
    def __init__(self, qasm, sym: str, size: int, var_type: str) -> None:
        self.qasm = qasm
        self.sym = sym
        self.size = size
        self.var_type = var_type

    def __str__(self) -> str:
        return f"{self.sym}"

    def reg_str(self):
        return f"{self.var_type} {self.sym}[{self.size}];"

    def __getitem__(self, idx):
        if idx > self.size - 1:
            msg = "list index out of range"
            raise IndexError(msg)

        return SubBit(self, idx)


class SubBit(Var):
    def __init__(self, var, idx: int) -> None:
        self.var = var
        self.idx = idx

    def __str__(self) -> str:
        return f"{self.var.sym}[{self.idx}]"


class CReg(Reg):
    def __init__(self, qasm, sym: str, size: int, declare=True) -> None:
        super().__init__(qasm, sym, size, var_type="creg")

        if declare:
            self.qasm.cregs.append(self)


class QReg(Reg):
    def __init__(self, qasm, sym: str, size: int, declare=True) -> None:
        super().__init__(qasm, sym, size, var_type="qreg")

        if declare:
            self.qasm.qregs.append(self)
