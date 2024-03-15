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
from __future__ import annotations

from typing import TYPE_CHECKING

from pecos.slr.util import rm_white_space
from pecos.slr.vars import Reg

if TYPE_CHECKING:
    from pecos.slr.vars import Elem, QReg


class Barrier:
    def __init__(self, *qregs: QReg | tuple[QReg]):
        self.qregs = qregs

    def qasm(self):
        if isinstance(self.qregs, list | tuple | set):
            qubits = []
            for q in self.qregs:
                qubits.append(str(q))
            qubits = ", ".join(qubits)
        else:
            qubits = self.qregs

        qasm = f"barrier {qubits};"

        return qasm


class Comment:
    """A comment for human readability of output qasm."""

    def __init__(self, *txt):
        self.txt = "\n".join(txt)

    def qasm(self):
        txt = self.txt.split("\n")
        txt = [f"// {t}" if t.strip() != "" else t for t in txt]
        return "\n".join(txt)


class QASM:
    """A comment for human readability of output qasm."""

    def __init__(self, *txt):
        self.txt = "\n".join(txt)

    def qasm(self):
        return rm_white_space(self.txt)


class Permute:
    """Permutes the indices that the elements of the register so that Reg[i] now refers to Reg[j]."""

    def __init__(self, elems_i: list[Elem] | Reg, elems_f: list[Elem] | Reg, *, comment: bool = True):
        self.elems_i = elems_i
        self.elems_f = elems_f
        self.comment = comment

    def execute(self):
        if isinstance(self.elems_i, Reg) and isinstance(self.elems_f, Reg):
            if len(self.elems_i.elems) != len(self.elems_f.elems):
                msg = "Number of input and output elements are not the same."
                raise Exception(msg)

            for ei, ej in zip(self.elems_i.elems, self.elems_f.elems, strict=True):
                ei.reg, ej.reg = ej.reg, ei.reg
                ei.index, ej.index = ej.index, ei.index

        else:
            if set(self.elems_i) != set(self.elems_f):
                msg = "The set of input elements are not the same as the set of output elements"
                raise Exception(msg)
            if not (len(self.elems_i) == len(set(self.elems_i)) == len(self.elems_f) == len(set(self.elems_f))):
                msg = "The number of input and output elements are not the same."
                raise Exception(msg)

            temp = []
            for ei in self.elems_i:
                temp.append((ei.reg, ei.index))

            for ti, ef in zip(temp, self.elems_f, strict=True):
                ef.reg = ti[0]
                ef.index = ti[1]

    def qasm(self):
        self.execute()

        if self.comment:
            qstr = []
            for ei, ej in zip(self.elems_i, self.elems_f, strict=True):
                qstr.append(f"{ei} -> {ej}")
            return "// Permuting: " + ", ".join(qstr)
        else:
            return
