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

from pathlib import Path

from pecos.circuits.qasm import std_gates
from pecos.circuits.qasm.barrier import Barrier
from pecos.circuits.qasm.block import Block
from pecos.circuits.qasm.conditionals import CIf
from pecos.circuits.qasm.expr import Assign
from pecos.circuits.qasm.func import Func
from pecos.circuits.qasm.gates import ArgGate, Gate
from pecos.circuits.qasm.std_gates import Measure, Reset
from pecos.circuits.qasm.vars import CReg, QReg


class QASM:
    """Represents an OpenQASM 2.0 circuit."""

    def __init__(self, language="OPENQASM 2.0", header=True) -> None:
        self.language = f"{language};"
        self.includes = []
        self.definitions = []
        self.qregs = []
        self.cregs = []
        self.body = Block()
        self.g = std_gates
        self.header = header

    def qreg(self, sym, size, declare=True):
        return QReg(self, sym, size, declare=declare)

    def creg(self, sym, size, declare=True):
        return CReg(self, sym, size, declare=declare)

    def define(self, name, params, qargs, body):
        param_size = 0
        if isinstance(params, str):
            param_size = 1
        elif params:
            param_size = len(params)
            params = ",".join(params)
        else:
            params = ""

        size = 0
        if isinstance(qargs, str):
            size = 1
        else:
            size = len(qargs)
            qargs = ",".join(qargs)

        df_str = [f"gate {name}({params}) {qargs}", "{"]
        df_str.extend([f"   {g};" for g in body])
        df_str.append("}")
        df_str = "\n".join(df_str)

        self.definitions.append(df_str)

        if params:
            return ArgGate(name, size=size, num_args=param_size, qasm_def=df_str)
        else:
            return Gate(name, size=size, qasm_def=df_str)

    def custom(self, text: str):
        self.body.append(text)

    def br(self, n: int = 1):
        for _ in range(n):
            self.body.append("")

    def comment(self, text: str, br=True):
        if br:
            return f"\n// {text}"
        else:
            return f"// {text}"

    def include(self, *args):
        for a in args:
            self.includes.append(f'include "{a}";')

    def block(self, *args):
        arg_list = []
        for a in args:
            if "\n" in str(a):
                arg_list.extend(str(a).split("\n"))
            else:
                arg_list.append(str(a))

        self.body.extend(*arg_list)

    def block_cif(self, cond, *args):
        self.cif_block(cond, *args)

    def cif_block(self, cond, *args):
        cond = str(cond)

        arg_list = []
        for a in args:
            if "\n" in str(a):
                for ai in str(a).split("\n"):
                    ai = str(ai)

                    if not ai.strip().startswith("//") and ai.strip() != "":
                        arg_list.append(f"if({cond}) {str(ai)}")
                    else:
                        arg_list.append(f"{str(ai)}")
            else:
                a = str(a)

                if not a.strip().startswith("//") and a.strip() != "":
                    arg_list.append(f"if({cond}) {str(a)}")
                else:
                    arg_list.append(f"{str(a)}")

        self.body.extend(*arg_list)

    def gate(self, sym, *args, params=None):
        if params:
            return ArgGate(sym)(*args, params=params)
        else:
            return Gate(sym)(*args)

    def measure(self, qreg, creg):
        return Measure(qreg, creg)

    def reset(self, *qargs):
        return Reset(*qargs)

    def barrier(self, *qargs):
        return Barrier(*qargs)

    def cif(self, cond, expr, else_expr=None):
        return CIf(cond, expr, else_expr)

    def assign(self, left, right):
        return Assign(left, right)

    def func(self, name):
        return Func(name=name)

    def __str__(self) -> str:
        if self.header:
            lines = [self.language]

            if not self.includes:
                lines.append('include "qelib1.inc";')
            else:
                lines.extend(self.includes)

            lines.extend([f"\n{d}\n" for d in self.definitions])

            lines.extend([q.reg_str() for q in self.qregs])
            lines.extend([c.reg_str() for c in self.cregs])
        else:
            lines = []

        for line in self.body:
            line = str(line).strip()
            if line and not line.startswith("//"):
                if not line.endswith(";"):
                    lines.append(f"{line};")
                else:
                    lines.append(f"{line}")
            else:
                lines.append(line)

        return "\n".join(lines)

    def save(self, file: str) -> str:
        """Writes qasm text to file."""
        with Path.open(file, "w") as f:
            f.write(str(self))

        return file
