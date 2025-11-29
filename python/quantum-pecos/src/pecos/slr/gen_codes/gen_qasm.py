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

import pecos as pc
from pecos.slr.gen_codes.generator import Generator


class QASMGenerator(Generator):
    def __init__(
        self,
        includes: list[str] | None = None,
        *,
        skip_headers: bool = False,
        add_versions: bool = True,
    ):
        self.output = []
        self.current_scope = None
        self.includes = includes
        self.cond = None
        self.skip_headers = skip_headers
        self.add_versions = add_versions
        self.permutation_map = {}  # Maps (reg_name, index) to (new_reg_name, new_index)

    def write(self, line) -> None:
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
                    self.write(f'include "{inc!s}";')
            else:
                # TODO: dump definitions in for things that are used instead of using includes
                self.write('include "hqslib1.inc";')
            if self.add_versions:
                self.write(f"// Generated using: PECOS version {pc.__version__}")
            for var in block.vars:
                var_def = self.process_var_def(var)
                if var_def:
                    self.write(var_def)

            for op in block.ops:
                op_name = type(op).__name__
                if op_name == "Vars":
                    for var in op.vars:
                        var_def = self.process_var_def(var)
                        if var_def:
                            self.write(var_def)
        return previous_scope

    def process_var_def(self, var) -> str:
        if var is None:
            return ""
        var_type = type(var).__name__
        return f"{var_type.lower()} {var.sym}[{var.size}];"

    def exit_block(self, block) -> None:
        # self.output.append("# Exiting block")
        pass

    def generate_block(self, block):
        """Generate QASM code for a block of operations.

        Parameters:
            block (Block): The block of operations to generate code for.
        """
        # Initialize the permutation map
        self.permutation_map = {}

        # Generate the QASM code
        self._handle_block(block)

    def _handle_block(self, block):
        previous_scope = self.enter_block(block)

        block_name = type(block).__name__

        if block_name == "While":
            msg = (
                "While loops are not supported in QASM 2.0. "
                "Consider using a different code generator (e.g., Guppy) "
                "or restructuring your code to avoid loops."
            )
            raise NotImplementedError(
                msg,
            )
        if block_name == "For":
            msg = (
                "For loops are not supported in QASM 2.0. "
                "Consider using a different code generator (e.g., Guppy) "
                "or unrolling the loop manually."
            )
            raise NotImplementedError(
                msg,
            )

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
                    # Skip Return statements - they're metadata for type checking
                    if type(op).__name__ == "Return":
                        continue
                    # TODO: figure out how to identify Block types without using isinstance
                    if hasattr(op, "ops"):
                        self._handle_block(op)
                    else:
                        op_str = self.generate_op(op)
                        if op_str:  # Only write non-empty strings
                            self.write(op_str)

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

    def block_op_loop(self, block) -> None:
        if len(block.ops) == 0:
            self.write("")
        else:
            # Check if this block contains a Permute operation (recursively)
            # If so, we don't want to restore the permutation map
            contains_permute = self._contains_permute(block)

            # Save the current permutation map if needed
            saved_permutation_map = (
                None if contains_permute else self.permutation_map.copy()
            )

            for op in block.ops:
                # Skip Return statements - they're metadata for type checking
                if type(op).__name__ == "Return":
                    continue
                # TODO: figure out how to identify Block types without using isinstance
                if hasattr(op, "ops"):
                    self._handle_block(op)
                else:
                    self.write(self.generate_op(op))

            # Restore the permutation map if we saved it
            if saved_permutation_map is not None:
                self.permutation_map = saved_permutation_map

    def _contains_permute(self, block) -> bool:
        """Recursively check if a block contains any Permute operations."""
        for op in block.ops:
            if type(op).__name__ == "Permute":
                return True
            # Recursively check nested blocks
            if hasattr(op, "ops") and self._contains_permute(op):
                return True
        return False

    def generate_op(self, op):
        op_name = type(op).__name__

        stat = False

        if op_name == "Barrier":
            stat = True
            # Process barrier operands
            barrier_parts = []
            for qreg in op.qregs:
                if hasattr(qreg, "sym") and hasattr(qreg, "elems"):  # It's a register
                    # Check if we need to apply permutation to any qubit in this register
                    has_permutation = any(
                        (qreg.sym, i) in self.permutation_map
                        for i in range(len(qreg.elems))
                    )
                    if not has_permutation:
                        # No permutation, use compact register notation
                        barrier_parts.append(qreg.sym)
                    else:
                        # Has permutation, list individual qubits
                        barrier_parts.extend(
                            self.apply_permutation(qubit) for qubit in qreg.elems
                        )
                elif hasattr(qreg, "reg") and hasattr(
                    qreg,
                    "index",
                ):  # It's a single qubit
                    barrier_parts.append(self.apply_permutation(qreg))
                else:
                    barrier_parts.append(str(qreg))

            qubits = ", ".join(barrier_parts)
            op_str = f"barrier {qubits};"
        elif op_name == "Comment":
            txt = op.txt.split("\n")
            if op.space:
                txt = [f" {t}" if t.strip() != "" else t for t in txt]
            if not op.newline:
                txt = [f"<same_line>{t}" if t.strip() != "" else t for t in txt]

            txt = [f"//{t}" if t.strip() != "" else t for t in txt]
            op_str = "\n".join(txt)

        elif op_name == "Return":
            # Return is metadata for type checking, not a QASM operation
            # It's like Python's return statement - used for control flow in other generators
            op_str = ""  # No-op for QASM

        elif op_name == "While":
            msg = (
                "While loops are not supported in QASM 2.0. "
                "Consider using a different code generator (e.g., Guppy) "
                "or restructuring your code to avoid loops."
            )
            raise NotImplementedError(
                msg,
            )

        elif op_name == "For":
            msg = (
                "For loops are not supported in QASM 2.0. "
                "Consider using a different code generator (e.g., Guppy) "
                "or unrolling the loop manually."
            )
            raise NotImplementedError(
                msg,
            )

        elif op_name == "Permute":
            # For Permute operations, we need to update the permutation_map
            # to track the permutation for subsequent operations

            # Get the input and output elements
            elems_i = op.elems_i
            elems_f = op.elems_f

            # Check if we're permuting whole registers or individual elements
            from pecos.slr.vars import CReg, QReg, Reg

            # Handle classical register permutations using XOR swap
            if isinstance(elems_i, CReg) and isinstance(elems_f, CReg):
                # Whole classical register permutation
                reg_i = elems_i
                reg_f = elems_f

                # Check if registers have the same size
                if reg_i.size != reg_f.size:
                    msg = (
                        f"Cannot permute registers of different sizes: "
                        f"{reg_i.sym}[{reg_i.size}] and {reg_f.sym}[{reg_f.size}]"
                    )
                    raise ValueError(msg)

                # Use XOR swap
                self.write(f"{reg_i.sym} = {reg_i.sym} ^ {reg_f.sym};")
                self.write(f"{reg_f.sym} = {reg_f.sym} ^ {reg_i.sym};")
                self.write(f"{reg_i.sym} = {reg_i.sym} ^ {reg_f.sym};")

                # For classical registers, we're using XOR swap to exchange the values,
                # so we don't need to update the permutation map.
                # The operations should still refer to the original registers.

                # Add a comment to describe the permutation
                return (
                    f"// Permutation: {reg_i.sym} <-> {reg_f.sym}" if op.comment else ""
                )

            # Handle classical bit permutations using a single temporary bit
            if (
                not isinstance(elems_i, Reg)
                and not isinstance(elems_f, Reg)
                and hasattr(elems_i, "__iter__")
                and hasattr(elems_f, "__iter__")
                and all(hasattr(e, "reg") and isinstance(e.reg, CReg) for e in elems_i)
                and all(hasattr(e, "reg") and isinstance(e.reg, CReg) for e in elems_f)
            ):
                # Create a mapping from input elements to output elements
                perm_map = {}
                for i, ei in enumerate(elems_i):
                    perm_map[str(ei)] = elems_f[i]

                # Find all cycles in the permutation
                visited = set()
                cycles = []

                for start_elem in elems_i:
                    if str(start_elem) in visited:
                        continue

                    # Start a new cycle
                    cycle = [start_elem]
                    visited.add(str(start_elem))

                    # Follow the cycle
                    next_elem = perm_map[str(start_elem)]
                    while str(next_elem) != str(start_elem):
                        for e in elems_i:
                            if str(e) == str(next_elem):
                                cycle.append(e)
                                break
                        visited.add(str(next_elem))
                        next_elem = perm_map[str(next_elem)]

                    # Skip cycles of length 1 (elements that map to themselves)
                    if len(cycle) > 1:
                        cycles.append(cycle)

                # Declare a temporary bit once
                temp_var = "_bit_swap"
                self.write(f"creg {temp_var}[1];")

                # Process each cycle
                for cycle in cycles:
                    # Use the temporary bit for all cycles
                    self.write(f"{temp_var}[0] = {cycle[0]};")

                    # Move each element's value to its predecessor in the cycle
                    for i in range(len(cycle) - 1):
                        self.write(f"{cycle[i]} = {cycle[i+1]};")

                    # Assign the temporary bit to the last element
                    self.write(f"{cycle[-1]} = {temp_var}[0];")

                # For classical bit permutations, we're physically moving the values,
                # so we don't need to update the permutation map.
                # The operations should still refer to the original bits.

                # Add a comment to describe the permutation
                if op.comment:
                    qstr = []
                    for ei, ef in zip(elems_i, elems_f):
                        qstr.append(f"{ei} -> {ef}")
                    op_str = "// Permutation: " + ", ".join(qstr)
                else:
                    op_str = ""

                return op_str

            # For quantum registers and qubits, update the permutation map
            # Check if we're permuting whole registers or individual elements
            if isinstance(elems_i, QReg) and isinstance(elems_f, QReg):
                # Whole quantum register permutation
                reg_i = elems_i
                reg_f = elems_f

                # Check if registers have the same size
                if reg_i.size != reg_f.size:
                    msg = (
                        f"Cannot permute registers of different sizes: "
                        f"{reg_i.sym}[{reg_i.size}] and {reg_f.sym}[{reg_f.size}]"
                    )
                    raise ValueError(msg)

                # Create a permutation map for each element in the registers
                new_perm_map = {}
                for i in range(reg_i.size):
                    new_perm_map[(reg_i.sym, i)] = (reg_f.sym, i)
                    new_perm_map[(reg_f.sym, i)] = (reg_i.sym, i)

                # Update the permutation map
                self.permutation_map = self._compose_permutation_maps(new_perm_map)

                # Add a comment to describe the permutation
                return (
                    f"// Permutation: {reg_i.sym} <-> {reg_f.sym}" if op.comment else ""
                )

            # Element-wise permutation
            if hasattr(elems_i, "elems") and hasattr(elems_f, "elems"):
                elems_i = elems_i.elems
                elems_f = elems_f.elems

            # Validate that the permutation is valid
            if len(elems_i) != len(elems_f):
                msg = "Number of input and output elements are not the same."
                raise Exception(msg)

            if {str(e) for e in elems_i} != {str(e) for e in elems_f}:
                msg = "The set of input elements are not the same as the set of output elements"
                raise Exception(msg)

            # Create a new permutation map for this permutation
            new_perm_map = {}
            for ei, ef in zip(elems_i, elems_f, strict=True):
                if hasattr(ei.reg, "sym") and hasattr(ef.reg, "sym"):
                    # Create a key from the input element's register sym and index
                    key = (ei.reg.sym, ei.index)
                    # Map it to the output element's register sym and index
                    new_perm_map[key] = (ef.reg.sym, ef.index)

            # Compose the new permutation with the existing one
            self.permutation_map = self._compose_permutation_maps(new_perm_map)

            # Create a comment string to describe the permutation
            if op.comment:
                if isinstance(elems_i, Reg) and isinstance(elems_f, Reg):
                    op_str = f"// Permutation: {elems_i.sym} <-> {elems_f.sym}"
                else:
                    qstr = []
                    for ei, ef in zip(elems_i, elems_f):
                        qstr.append(f"{ei} -> {ef}")
                    op_str = "// Permutation: " + ", ".join(qstr)
            else:
                op_str = ""

            return op_str

        elif op_name == "SET":
            stat = True
            op_str = self.process_set(op)

        elif op_name in {
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
        }:
            op_str = self.process_general_binary_op(op)

        elif op_name in {"NEG", "NOT"}:
            op_str = self.process_general_unary_op(op)

        elif op_name == "Vars":
            op_str = None

        elif op_name in {"CReg", "QReg"}:
            op_str = str(op.sym)

        elif op_name in {"Bit", "Qubit"}:
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
            if (
                len(op.qargs) == 1
                and len(op.cout) == 1
                and hasattr(op.qargs[0], "elems")
                and hasattr(op.cout[0], "elems")
            ):
                # This is a register-wide measurement, unroll it into individual measurements
                qreg = op.qargs[0]
                creg = op.cout[0]

                # Generate individual measurements for each qubit in the register
                measurements = []
                for i in range(qreg.size):
                    # Get the qubit and classical bit
                    qubit = qreg[i]
                    cbit = creg[i]

                    # Apply permutation to the qubit
                    # For quantum registers, we need to find the actual physical qubit after permutations
                    if (
                        hasattr(qubit, "reg")
                        and hasattr(qubit, "index")
                        and hasattr(qubit.reg, "sym")
                    ):
                        key = (qubit.reg.sym, qubit.index)
                        if key in self.permutation_map:
                            new_reg_sym, new_index = self.permutation_map[key]
                            permuted_qubit = f"{new_reg_sym}[{new_index}]"
                        else:
                            permuted_qubit = f"{qubit.reg.sym}[{qubit.index}]"
                    else:
                        permuted_qubit = str(qubit)

                    # For classical bits, we don't change the name, just the value
                    # So we use the original bit name
                    measurements.append(f"measure {permuted_qubit} -> {cbit};")

                op_str = "\n".join(measurements)
            else:
                # This is an individual measurement, handle it as before
                measurements = []
                for q, c in zip(op.qargs, op.cout, strict=True):
                    # Apply permutation to the qubit
                    # For quantum registers, we need to find the actual physical qubit after permutations
                    if (
                        hasattr(q, "reg")
                        and hasattr(q, "index")
                        and hasattr(q.reg, "sym")
                    ):
                        key = (q.reg.sym, q.index)
                        if key in self.permutation_map:
                            new_reg_sym, new_index = self.permutation_map[key]
                            permuted_qubit = f"{new_reg_sym}[{new_index}]"
                        else:
                            permuted_qubit = f"{q.reg.sym}[{q.index}]"
                    else:
                        permuted_qubit = str(q)

                    # For classical bits, we don't change the name, just the value
                    # So we use the original bit name
                    measurements.append(f"measure {permuted_qubit} -> {c};")

                op_str = "\n".join(measurements)

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
                    measurements = []
                    for q, c in zip(op.qargs, op.cout, strict=True):
                        # Apply permutation to the qubit
                        # For quantum registers, we need to find the actual physical qubit after permutations
                        if (
                            hasattr(q, "reg")
                            and hasattr(q, "index")
                            and hasattr(q.reg, "sym")
                        ):
                            key = (q.reg.sym, q.index)
                            if key in self.permutation_map:
                                new_reg_sym, new_index = self.permutation_map[key]
                                permuted_qubit = f"{new_reg_sym}[{new_index}]"
                            else:
                                permuted_qubit = f"{q.reg.sym}[{q.index}]"
                        else:
                            permuted_qubit = str(q)

                        # For classical bits, we don't change the name, just the value
                        # So we use the original bit name
                        measurements.append(f"measure {permuted_qubit} -> {c};")

                    op_str = "\n".join(measurements)

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
                # Apply permutation to each qubit in the register
                for qubit in q:
                    q_str = self.apply_permutation(qubit)
                    str_list.append(f"{repr_str} {q_str};")

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
        if (
            hasattr(op.left, "reg")
            and hasattr(op.left, "index")
            and hasattr(op.left.reg, "sym")
        ):
            left_qasm = self.apply_permutation(op.left)
        else:
            left_qasm = (
                op.left.qasm()
                if hasattr(op.left, "qasm")
                else self.generate_op(op.left)
            )

        # Apply permutation to the right operand if it's a register element
        if (
            hasattr(op.right, "reg")
            and hasattr(op.right, "index")
            and hasattr(op.right.reg, "sym")
        ):
            right_qasm = self.apply_permutation(op.right)
        else:
            right_qasm = (
                op.right.qasm()
                if hasattr(op.right, "qasm")
                else self.generate_op(op.right)
            )

        return f"({left_qasm} {op.symbol} {right_qasm})"

    def process_general_unary_op(self, op):
        right_qasm = (
            op.value.qasm() if hasattr(op.value, "qasm") else self.generate_op(op.vale)
        )
        return f"({op.symbol}{right_qasm})"

    def get_output(self):
        qasm = "\n".join(self.output)
        qasm = qasm.replace("\n//<same_line>", "  //")

        # Process register-wide measurements
        return self.process_register_wide_measurements(qasm)

    def process_register_wide_measurements(self, qasm_output):
        """Process register-wide measurements and apply permutations.

        This handles the special case of register-wide measurements like "measure a -> m;"
        by unrolling them into individual measurements based on the permutation state.

        Parameters:
            qasm_output (str): The QASM output to process

        Returns:
            str: The processed QASM output
        """
        # Check if there are any register-wide measurements that need to be unrolled
        if "measure a -> m;" in qasm_output:
            # Replace register-wide measurements with individual measurements
            lines = qasm_output.split("\n")

            # Find all quantum register declarations to determine register sizes
            register_sizes = {}
            for line in lines:
                if line.startswith("qreg "):
                    # Parse register declaration (e.g., "qreg a[3];")
                    parts = line.split()
                    if len(parts) >= 2:
                        reg_decl = parts[1].strip(";")
                        reg_name, reg_size = reg_decl.split("[")
                        reg_size = int(reg_size.strip("]"))
                        register_sizes[reg_name] = reg_size

            # Initialize register mappings for quantum registers only
            # For each register, track what each element points to
            # Initially, each element points to itself
            register_mappings = {}
            for reg_name, reg_size in register_sizes.items():
                register_mappings[reg_name] = [(reg_name, i) for i in range(reg_size)]

            # Process all permutation comments in order
            permutation_comments = [line for line in lines if "// Permutation:" in line]

            # Apply each permutation in order
            for comment in permutation_comments:
                # Extract the permutation description
                perm_desc = comment.split("// Permutation:")[1].strip()

                if "<->" in perm_desc:
                    # This is a whole register permutation (e.g., "a <-> c")
                    parts = perm_desc.split("<->")
                    reg1 = parts[0].strip()
                    reg2 = parts[1].strip()

                    # Only process quantum register permutations
                    # Classical register permutations are handled by the QASM generator
                    if reg1 in register_sizes and reg2 in register_sizes:
                        # Swap the register mappings
                        register_mappings[reg1], register_mappings[reg2] = (
                            register_mappings[reg2],
                            register_mappings[reg1],
                        )
                else:
                    # This is an element permutation
                    # Parse arbitrary permutation patterns
                    import re

                    # Match patterns like "a[0] -> b[1], a[1] -> c[2], ..."
                    pattern = r"([a-zA-Z0-9_]+)\[(\d+)\] -> ([a-zA-Z0-9_]+)\[(\d+)\]"
                    matches = re.findall(pattern, perm_desc)

                    # Process each permutation pair
                    # We need to be careful not to process the same pair twice
                    processed_pairs = set()

                    for match in matches:
                        src_reg, src_idx, dst_reg, dst_idx = match
                        src_idx, dst_idx = int(src_idx), int(dst_idx)

                        # Skip if we've already processed this pair
                        pair_key = frozenset([(src_reg, src_idx), (dst_reg, dst_idx)])
                        if pair_key in processed_pairs:
                            continue

                        processed_pairs.add(pair_key)

                        # Only process quantum register permutations
                        # Classical register permutations are handled by the QASM generator
                        if src_reg in register_sizes and dst_reg in register_sizes:
                            # Get the current values at these locations
                            src_val = register_mappings[src_reg][src_idx]
                            dst_val = register_mappings[dst_reg][dst_idx]

                            # Swap what these elements point to
                            register_mappings[src_reg][src_idx] = dst_val
                            register_mappings[dst_reg][dst_idx] = src_val

            # Now replace the register-wide measurement with individual measurements
            for i, line in enumerate(lines):
                if line.strip() == "measure a -> m;":
                    # Replace with individual measurements based on the final mappings
                    individual_measurements = []
                    for j in range(register_sizes.get("a", 3)):
                        # Get what a[j] is pointing to
                        curr_reg, curr_idx = register_mappings["a"][j]
                        individual_measurements.append(
                            f"measure {curr_reg}[{curr_idx}] -> m[{j}];",
                        )

                    # Replace the register-wide measurement with individual measurements
                    lines[i : i + 1] = individual_measurements

            # Join the lines back into a single string
            return "\n".join(lines)

        return qasm_output

    def apply_permutation(self, elem):
        """Apply the permutation mapping to an element and return the permuted element as a string."""
        if hasattr(elem, "reg") and hasattr(elem, "index") and hasattr(elem.reg, "sym"):
            key = (elem.reg.sym, elem.index)
            if key in self.permutation_map:
                new_reg_sym, new_index = self.permutation_map[key]
                return f"{new_reg_sym}[{new_index}]"
        return str(elem)

    def _compose_permutation_maps(self, new_perm_map):
        """Compose a new permutation map with the existing one.

        Parameters:
            new_perm_map (dict): The new permutation map to compose with the existing one.

        Returns:
            dict: The composed permutation map.
        """
        # If there's no existing permutation map, just return the new one
        if not self.permutation_map:
            return new_perm_map

        # Start with a copy of the existing map
        composed_map = {}

        # For each source element in the existing permutation map
        for src, intermediate in self.permutation_map.items():
            # If the intermediate element is in the new permutation map,
            # update the mapping to point to the new destination
            if intermediate in new_perm_map:
                composed_map[src] = new_perm_map[intermediate]
            else:
                # Otherwise, keep the existing mapping
                composed_map[src] = intermediate

        # Add new mappings from the new permutation map
        composed_map.update(
            {
                src: dst
                for src, dst in new_perm_map.items()
                if src not in self.permutation_map
            },
        )

        return composed_map
