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
from pecos.slr.gen_codes.generator import Generator
from pecos.slr.vars import Reg


class QASMGenerator(Generator):
    def __init__(
        self,
        includes: list[str] | None = None,
        skip_headers=False,
        add_versions=True,
    ):
        self.output = []
        self.current_scope = None
        self.includes = includes
        self.cond = None
        self.skip_headers = skip_headers
        self.add_versions = add_versions
        self.permutation_map = {}  # Maps (reg_name, index) to (new_reg_name, new_index)

    def write(self, line):
        self.output.append(line)

    def enter_block(self, block):
        previous_scope = self.current_scope
        self.current_scope = block

        block_name = type(block).__name__

        # self.output.append("# Entering new block")
        if block_name == "Main" and not self.skip_headers:
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
            # Generate the condition with permutations applied
            self.cond = self.generate_op(block.cond)
            
            # Process the operations inside the If block
            # We need to create a new instance of the block_op_loop method
            # to ensure that permutations are applied to the operations inside the If block
            if len(block.ops) == 0:
                self.write("")
            else:
                for op in block.ops:
                    # TODO: figure out how to identify Block types without using isinstance
                    if hasattr(op, "ops"):
                        self.generate_block(op)
                    else:
                        self.write(self.generate_op(op))
            
            # Reset the condition
            self.cond = None
            
            # Process the else block if it exists
            if block.else_block:
                # TODO: Handle else blocks
                pass

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
            # Save the current permutation map
            saved_permutation_map = self.permutation_map.copy()
            
            for op in block.ops:
                # TODO: figure out how to identify Block types without using isinstance
                if hasattr(op, "ops"):
                    self.generate_block(op)
                else:
                    self.write(self.generate_op(op))
            
            # Restore the permutation map
            self.permutation_map = saved_permutation_map

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
            # For Permute operations, we need to update the permutation_map
            # to track the permutation for subsequent operations
            
            # Get the input and output elements
            elems_i = op.elems_i
            elems_f = op.elems_f
            
            # Check if we're permuting whole registers or individual elements
            if isinstance(elems_i, Reg) and isinstance(elems_f, Reg):
                # Whole register permutation
                reg_i = elems_i
                reg_f = elems_f
                
                # Check if registers have the same size
                if reg_i.size != reg_f.size:
                    msg = f"Cannot permute registers of different sizes: {reg_i.sym}[{reg_i.size}] and {reg_f.sym}[{reg_f.size}]"
                    raise ValueError(msg)
                
                # Create a permutation map for each element in the registers
                new_perm_map = {}
                for i in range(reg_i.size):
                    new_perm_map[(reg_i.sym, i)] = (reg_f.sym, i)
                    new_perm_map[(reg_f.sym, i)] = (reg_i.sym, i)
                
                # Create a comment string to describe the permutation
                comment = ""
                if op.comment:
                    comment = f"// Permutation: {reg_i.sym} <-> {reg_f.sym}"
            else:
                # Element-wise permutation
                if hasattr(elems_i, "elems") and hasattr(elems_f, "elems"):
                    elems_i = elems_i.elems
                    elems_f = elems_f.elems
                
                # Validate that the permutation is valid
                if len(elems_i) != len(elems_f):
                    msg = "Number of input and output elements are not the same."
                    raise Exception(msg)
                
                if set(str(e) for e in elems_i) != set(str(e) for e in elems_f):
                    msg = "The set of input elements are not the same as the set of output elements"
                    raise Exception(msg)
                
                # Create a new permutation map for this permutation
                new_perm_map = {}
                for ei, ef in zip(elems_i, elems_f, strict=True):
                    if hasattr(ei.reg, 'sym') and hasattr(ef.reg, 'sym'):
                        # Create a key from the input element's register sym and index
                        key = (ei.reg.sym, ei.index)
                        # Map it to the output element's register sym and index
                        new_perm_map[key] = (ef.reg.sym, ef.index)
                
                # Create a comment string to describe the permutation
                comment = ""
                if op.comment:
                    qstr = []
                    for ei, ej in zip(elems_i, elems_f, strict=True):
                        qstr.append(f"{ei} -> {ej}")
                    comment = "// Permutation: " + ", ".join(qstr)
            
            # Compose the new permutation with the existing one
            updated_perm_map = {}
            
            # For each source element in the existing permutation map
            for src, intermediate in self.permutation_map.items():
                # If the intermediate element is in the new permutation map,
                # update the mapping to point to the new destination
                if intermediate in new_perm_map:
                    updated_perm_map[src] = new_perm_map[intermediate]
                else:
                    # Otherwise, keep the existing mapping
                    updated_perm_map[src] = intermediate
            
            # Add new mappings from the new permutation map
            for src, dst in new_perm_map.items():
                if src not in self.permutation_map:
                    updated_perm_map[src] = dst
            
            # Update the permutation map
            self.permutation_map = updated_perm_map
            
            op_str = comment

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

        elif op_name == "Measure":
            # Check if this is a register-wide measurement (QReg > CReg)
            if len(op.qargs) == 1 and len(op.cout) == 1 and hasattr(op.qargs[0], 'elems') and hasattr(op.cout[0], 'elems'):
                # This is a register-wide measurement, unroll it into individual measurements
                qreg = op.qargs[0]
                creg = op.cout[0]
                
                # Generate individual measurements for each qubit in the register
                measurements = []
                for i in range(qreg.size):
                    qubit = qreg[i]
                    cbit = creg[i]
                    measurements.append(f"measure {self.apply_permutation(qubit)} -> {self.apply_permutation(cbit)};")
                
                op_str = " ".join(measurements)
            else:
                # This is an individual measurement, handle it as before
                op_str = " ".join(
                    [
                        f"measure {self.apply_permutation(q)} -> {self.apply_permutation(c)};"
                        for q, c in zip(op.qargs, op.cout, strict=True)
                    ],
                )

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
                            f"measure {self.apply_permutation(q)} -> {self.apply_permutation(c)};"
                            for q, c in zip(op.qargs, op.cout, strict=True)
                        ],
                    )

                case "F":
                    op_str = "\n".join(
                        [
                            self.qgate_sq_qasm(op, "rx(pi/2)"),
                            self.qgate_sq_qasm(op, "rz(pi/2)"),
                        ],
                    )

                case "Fdg":
                    op_str = "\n".join(
                        [
                            self.qgate_sq_qasm(op, "ry(-pi/2)"),
                            self.qgate_sq_qasm(op, "rz(-pi/2)"),
                        ],
                    )

                case "F4":
                    op_str = "\n".join(
                        [
                            self.qgate_sq_qasm(op, "ry(-pi/2)"),
                            self.qgate_sq_qasm(op, "rz(pi/2)"),
                        ],
                    )

                case "F4dg":
                    op_str = "\n".join(
                        [
                            self.qgate_sq_qasm(op, "rx(-pi/2)"),
                            self.qgate_sq_qasm(op, "rz(-pi/2)"),
                        ],
                    )

                case "Prep":
                    op_str = self.qgate_sq_qasm(op, "reset")

                case "T":
                    op_str = self.qgate_sq_qasm(op, "rz(pi/4)")

                case "Tdg":
                    op_str = self.qgate_sq_qasm(op, "rz(-pi/4)")

                case "SX":
                    op_str = self.qgate_sq_qasm(op, "rx(pi/2)")

                case "SY":
                    op_str = self.qgate_sq_qasm(op, "ry(pi/2)")

                case "SZ":
                    op_str = self.qgate_sq_qasm(op, "rz(pi/2)")

                case "SXdg":
                    op_str = self.qgate_sq_qasm(op, "rx(-pi/2)")

                case "SYdg":
                    op_str = self.qgate_sq_qasm(op, "ry(-pi/2)")

                case "SZdg":
                    op_str = self.qgate_sq_qasm(op, "rz(-pi/2)")

                case _:
                    op_str = self.qgate_sq_qasm(op)

        return op_str

    def qgate_sq_qasm(self, op, repr_str: str | None = None):
        if op.qsize != 1:
            msg = "qgate_qasm only supports single qubit gates"
            raise Exception(msg)

        if repr_str is None:
            repr_str = op.sym.lower()

        if op.params:
            str_cargs = ", ".join([str(p) for p in op.params])
            repr_str = f"{repr_str}({str_cargs})"

        str_list = []

        for q in op.qargs:
            if type(q).__name__ == "QReg":
                lines = [f"{repr_str} {qubit};" for qubit in q]
                str_list.extend(lines)

            elif isinstance(q, tuple):
                if len(q) != op.qsize:
                    msg = f"Expected size {op.qsize} got size {len(q)}"
                    raise Exception(msg)
                qs = ",".join([str(qi) for qi in q])
                str_list.append(f"{repr_str} {qs};")

            else:
                # Apply permutation to the qubit
                q_str = self.apply_permutation(q)
                str_list.append(f"{repr_str} {q_str};")

        return "\n".join(str_list)

    def qgate_tq_qasm(self, op, repr_str: str | None = None):
        if op.qsize != 2:
            msg = "qgate_tq_qasm only supports single qubit gates"
            raise Exception(msg)

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
                
                # Apply permutation to the qubits
                q1_str = self.apply_permutation(q1)
                q2_str = self.apply_permutation(q2)
                
                str_list.append(f"{repr_str} {q1_str}, {q2_str};")
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
        
        # Apply permutation to the left-hand side
        left_str = self.apply_permutation(op.left)
        
        return f"{left_str} = {right_qasm};"

    def process_general_binary_op(self, op):
        # Apply permutation to the left operand if it's a register element
        if hasattr(op.left, 'reg') and hasattr(op.left, 'index') and hasattr(op.left.reg, 'sym'):
            left_qasm = self.apply_permutation(op.left)
        else:
            left_qasm = op.left.qasm() if hasattr(op.left, "qasm") else self.generate_op(op.left)
        
        # Apply permutation to the right operand if it's a register element
        if hasattr(op.right, 'reg') and hasattr(op.right, 'index') and hasattr(op.right.reg, 'sym'):
            right_qasm = self.apply_permutation(op.right)
        else:
            right_qasm = op.right.qasm() if hasattr(op.right, "qasm") else self.generate_op(op.right)
        
        return f"({left_qasm} {op.symbol} {right_qasm})"

    def process_general_unary_op(self, op):
        right_qasm = (
            op.value.qasm() if hasattr(op.value, "qasm") else self.generate_op(op.vale)
        )
        return f"({op.symbol}{right_qasm})"

    def get_output(self):
        qasm = "\n".join(self.output)
        return qasm.replace("\n//<same_line>", "  //")

    def apply_permutation(self, elem):
        """Apply the permutation mapping to an element and return the permuted element as a string."""
        if hasattr(elem, 'reg') and hasattr(elem, 'index') and hasattr(elem.reg, 'sym'):
            key = (elem.reg.sym, elem.index)
            if key in self.permutation_map:
                new_reg_sym, new_index = self.permutation_map[key]
                return f"{new_reg_sym}[{new_index}]"
        return str(elem)
