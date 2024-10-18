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

from __future__ import annotations

from pecos import __version__


class QASMGenerator:
    def __init__(self, includes: list[str] | None = None, add_versions=True):
        self.output = []
        self.current_scope = None
        self.includes = includes
        self.cond = None
        self.add_versions = add_versions

    def write(self, line):
        self.output.append(line)

    def enter_block(self, block):
        previous_scope = self.current_scope
        self.current_scope = block

        block_name = type(block).__name__

        # self.output.append("# Entering new block")
        if block_name == "Main":
            self.write("OPENQASM 2.0;")
            if self.includes:
                for inc in self.includes:
                    self.write(f'include "{str(inc)}";')
            else:
                # TODO: dump definitions in for things that are used instead of using includes
                self.write('include "hqslib1.inc";')
            if self.add_versions:
                self.write(f"// Generated using: PECOS version {__version__}")
            for var in block.vars:
                var_def = self.process_var_def(var)
                self.write(var_def)

            for op in block.ops:
                op_name = type(op).__name__
                if op_name == "Vars":
                    for var in op.vars:
                        var_def = self.process_var_def(var)
                        self.write(var_def)
        return previous_scope

    def process_var_def(self, var):
        var_type = type(var).__name__
        return f"{var_type.lower()} {var.sym}[{var.size}];"

    def exit_block(self, block):
        # self.output.append("# Exiting block")
        pass

    def generate_block(self, block):
        previous_scope = self.enter_block(block)

        block_name = type(block).__name__

        if block_name == "If":
            # self.cond = block.cond.qasm()
            self.cond = self.generate_op(block.cond)
            self.block_op_loop(block)
            self.cond = None

        elif block_name == "Repeat":
            for _ in range(block.cond):
                self.block_op_loop(block)
        else:
            self.block_op_loop(block)

        self.exit_block(block)
        self.current_scope = previous_scope

    def block_op_loop(self, block):
        if len(block.ops) == 0:
            self.write("")
        else:
            for op in block.ops:
                # TODO: figure out how to identify Block types without using isinstance
                if hasattr(op, "ops"):
                    self.generate_block(op)
                else:
                    self.write(self.generate_op(op))

    def generate_op(self, op):
        op_name = type(op).__name__

        stat = False

        if op_name == "Barrier":
            stat = True
            if isinstance(op.qregs, list | tuple | set):
                qubits = []
                for q in op.qregs:
                    qubits.append(str(q))
                qubits = ", ".join(qubits)
            else:
                qubits = op.qregs

            op_str = f"barrier {qubits};"
        elif op_name == "Comment":
            txt = op.txt.split("\n")
            if op.space:
                txt = [f" {t}" if t.strip() != "" else t for t in txt]
            if not op.newline:
                txt = [f"<same_line>{t}" if t.strip() != "" else t for t in txt]

            txt = [f"//{t}" if t.strip() != "" else t for t in txt]
            op_str = "\n".join(txt)

        elif op_name == "Permute":
            op_str = process_permute(op)

        elif op_name == "SET":
            stat = True
            op_str = self.process_set(op)

        elif op_name in [
            "EQUIV",
            "NEQUIV",
            "LT",
            "GT",
            "LE",
            "GE",
            "MUL",
            "DIV",
            "XOR",
            "AND",
            "OR",
            "PLUS",
            "MINUS",
            "RSHIFT",
            "LSHIFT",
        ]:
            op_str = self.process_general_binary_op(op)

        elif op_name in ["NEG", "NOT"]:
            op_str = self.process_general_unary_op(op)

        elif op_name == "Vars":
            op_str = None

        elif op_name in ["CReg", "QReg"]:
            op_str = str(op.sym)

        elif op_name in ["Bit", "Qubit"]:
            op_str = f"{op.reg.sym}[{op.index}]"

        elif isinstance(op, int):
            op_str = str(op)

        elif hasattr(op, "is_qgate") and op.is_qgate:
            stat = True
            op_str = self.process_qgate(op)

        elif hasattr(op, "gen"):
            op_str = op.gen(self)

        elif hasattr(op, "qasm"):
            stat = True
            op_str = op.qasm()

        else:
            msg = f"Operation '{op}' not handled!"
            raise NotImplementedError(msg)

        if self.cond and stat and op_str:
            cond = self.cond
            if cond.startswith("(") and cond.endswith(")"):
                cond = cond[1:-1]
            op_list = op_str.split("\n")
            op_new = []
            for o in op_list:
                o = o.strip()
                if o != "" and not o.startswith("//"):
                    for qi in o.split(";"):
                        qi = qi.strip()
                        if qi != "" and not qi.startswith("//"):
                            op_new.append(f"if({cond}) {qi};")
                else:
                    op_new.append(o)

            op_str = "\n".join(op_new)

        return op_str

    def process_qgate(self, op):
        sym = op.sym
        if op.qsize == 2:
            match sym:
                # TODO: Fix this... These are not qasm gates
                case "SXX":
                    op_str = self.qgate_tq_qasm(op, "SXX")
                case "SYY":
                    op_str = self.qgate_tq_qasm(op, "SYY")
                case "SZZ":
                    op_str = self.qgate_tq_qasm(op, "ZZ")
                case "SXXdg":
                    op_str = self.qgate_tq_qasm(op, "SXXdg")
                case "SYYdg":
                    op_str = self.qgate_tq_qasm(op, "SYYdg")
                case "SZZdg":
                    op_str = self.qgate_tq_qasm(op, "SZZdg")
                case _:
                    op_str = self.qgate_tq_qasm(op)

        else:
            match sym:
                case "Measure":
                    op_str = " ".join(
                        [
                            f"measure {str(q)} -> {c};"
                            for q, c in zip(op.qargs, op.cout, strict=True)
                        ],
                    )

                case "F":
                    op_str = " ".join(
                        [f"rx(pi/2) {str(q)};\nrz(pi/2) {str(q)};" for q in op.qargs],
                    )

                case "Fdg":
                    op_str = " ".join(
                        [f"ry(-pi/2) {str(q)};\nrz(-pi/2) {str(q)};" for q in op.qargs],
                    )

                case "F4":
                    op_str = " ".join(
                        [f"ry(-pi/2) {str(q)};\nrz(pi/2) {str(q)};" for q in op.qargs],
                    )

                case "F4dg":
                    op_str = " ".join(
                        [f"rx(-pi/2) {str(q)};\nrz(-pi/2) {str(q)};" for q in op.qargs],
                    )

                case "Prep":
                    op_str = self.qgate_qasm(op, "reset")

                case "T":
                    op_str = self.qgate_qasm(op, "rz(pi/4)")

                case "Tdg":
                    op_str = self.qgate_qasm(op, "rz(-pi/4)")

                case "SX":
                    op_str = self.qgate_qasm(op, "rx(pi/2)")

                case "SY":
                    op_str = self.qgate_qasm(op, "ry(pi/2)")

                case "SZ":
                    op_str = self.qgate_qasm(op, "rz(pi/2)")

                case "SXdg":
                    op_str = self.qgate_qasm(op, "rx(-pi/2)")

                case "SYdg":
                    op_str = self.qgate_qasm(op, "ry(-pi/2)")

                case "SZdg":
                    op_str = self.qgate_qasm(op, "rz(-pi/2)")

                case _:
                    op_str = self.qgate_qasm(op)

        return op_str

    def qgate_qasm(self, op, repr_str: str | None = None):
        if repr_str is None:
            repr_str = op.sym.lower()

        if op.params:
            str_cargs = ", ".join([str(p) for p in op.params])
            repr_str = f"{repr_str}({str_cargs})"

        str_list = []

        # TODO: Make sure this all works for different sized multi-qubit gates!!!!!!!!!!!!!!! <<<<<<<<<<

        if op.qsize > 1:
            if not isinstance(op.qargs[0], tuple) and len(op.qargs) > 1:
                op.qargs = (op.qargs,)

        for q in op.qargs:
            if isinstance(q, tuple):
                if len(q) != op.qsize:
                    msg = f"Expected size {op.qsize} got size {len(q)}"
                    raise Exception(msg)
                qs = ",".join([str(qi) for qi in q])
                str_list.append(f"{repr_str} {qs};")

            str_list.append(f"{repr_str} {str(q)};")

        return "\n".join(str_list)

    def qgate_tq_qasm(self, op, repr_str: str | None = None):
        if repr_str is None:
            repr_str = op.sym.lower()

        if op.params:
            str_cargs = ",".join([str(p) for p in op.params])
            repr_str = f"{repr_str}({str_cargs})"

        str_list = []

        if not isinstance(op.qargs[0], tuple) and len(op.qargs) == 2:
            op.qargs = (op.qargs,)

        for q in op.qargs:
            if isinstance(q, tuple):
                q1, q2 = q
                str_list.append(f"{repr_str} {str(q1)}, {str(q2)};")
            else:
                msg = f"For TQ gate, expected args to be a collection of size two tuples! Got: {op.qargs}"
                raise TypeError(msg)

        return "\n".join(str_list)

    def process_set(self, op):
        right_qasm = (
            op.right.qasm() if hasattr(op.right, "qasm") else self.generate_op(op.right)
        )
        if right_qasm.startswith("(") and right_qasm.endswith(")"):
            right_qasm = right_qasm[1:-1]
        return f"{op.left} = {right_qasm};"

    def process_general_binary_op(self, op):
        left_qasm = (
            op.left.qasm() if hasattr(op.left, "qasm") else self.generate_op(op.left)
        )
        right_qasm = (
            op.right.qasm() if hasattr(op.right, "qasm") else self.generate_op(op.right)
        )
        return f"({left_qasm} {op.symbol} {right_qasm})"

    def process_general_unary_op(self, op):
        right_qasm = (
            op.value.qasm() if hasattr(op.value, "qasm") else self.generate_op(op.vale)
        )
        return f"({op.symbol}{right_qasm})"

    def get_output(self):
        qasm = "\n".join(self.output)
        return qasm.replace("\n//<same_line>", "  //")


def process_permute(op):
    # TODO: Make permuting safer...
    if hasattr(op.elems_i, "elems") and hasattr(op.elems_f, "elems"):
        if len(op.elems_i.elems) != len(op.elems_f.elems):
            msg = "Number of input and output elements are not the same."
            raise Exception(msg)

        for ei, ej in zip(op.elems_i.elems, op.elems_f.elems, strict=True):
            ei.reg, ej.reg = ej.reg, ei.reg
            ei.index, ej.index = ej.index, ei.index

    else:
        if set(op.elems_i) != set(op.elems_f):
            msg = "The set of input elements are not the same as the set of output elements"
            raise Exception(msg)
        if not (
            len(op.elems_i)
            == len(set(op.elems_i))
            == len(op.elems_f)
            == len(set(op.elems_f))
        ):
            msg = "The number of input and output elements are not the same."
            raise Exception(msg)

        temp = []
        for ei in op.elems_i:
            temp.append((ei.reg, ei.index))

        for ti, ef in zip(temp, op.elems_f, strict=True):
            ef.reg = ti[0]
            ef.index = ti[1]

    if op.comment:
        qstr = []
        for ei, ej in zip(op.elems_i, op.elems_f, strict=True):
            qstr.append(f"{ei} -> {ej}")
        return "// Permuting: " + ", ".join(qstr)
    else:
        return ""
