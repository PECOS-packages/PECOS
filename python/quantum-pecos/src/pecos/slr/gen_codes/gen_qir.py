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

import re
from collections import OrderedDict
from typing import TYPE_CHECKING

from pecos_rslib.llvm import binding, ir

import pecos as pc
from pecos.qeclib.qubit import qgate_base
from pecos.slr import Block, If, Repeat
from pecos.slr.cops import (
    NEG,
    NOT,
    SET,
    BinOp,
    UnaryOp,
)
from pecos.slr.gen_codes.generator import Generator
from pecos.slr.gen_codes.qir_gate_mapping import QIRGateMetadata
from pecos.slr.misc import Barrier, Comment, Permute
from pecos.slr.vars import Bit, CReg, QReg, Qubit, Reg, Vars

if TYPE_CHECKING:
    from llvmlite.ir import DoubleType, IntType, PointerType, Type, VoidType

    from pecos.slr import Main
    from pecos.slr.cops import (
        CompOp,
    )


class QIRTypes:
    """Class to hold the types used in QIR compilation"""

    def __init__(self, module: ir.Module):
        """Parameters:

        module (llvmlite.ir.Module): an LLVM module to write to.
        """

        # Create some useful types to use in compilation later
        qubit_ty = module.context.get_identified_type("Qubit")
        result_ty = module.context.get_identified_type("Result")
        self.void_type: VoidType = ir.VoidType()
        self.bool_type: IntType = ir.IntType(1)
        self.int_type: IntType = ir.IntType(64)
        self.double_type: DoubleType = ir.DoubleType()
        self.qubit_ptr_type: PointerType = qubit_ty.as_pointer()
        self.result_ptr_type: PointerType = result_ty.as_pointer()
        self.tag_type: PointerType = ir.IntType(8).as_pointer()


class QIRFunc:
    """Represents a callable function in a QIR program"""

    def __init__(self, module: ir.Module, ret_ty: Type, arg_tys: list[Type], name: str):
        """Parameters:

        module (llvmlite.ir.Module): an LLVM module to write to.
        ret_ty (llvmlite.ir.Type): the LLVM return type for the QIR function
        arg_tys (list[llvmlite.ir.Type]): a list of types for parameters of the QIR function
        name (str): the name of the QIR function
        """
        self.binding = ir.Function(
            module,
            ir.FunctionType(ret_ty, arg_tys),
            name=name,
        )

    def create_call(
        self,
        builder: ir.IRBuilder,
        args: list[binding.ValueRef],
        name: str,
    ) -> binding.ValueRef:
        """A helper method to call a QIR Gate.

        Parameters:
        builder (llvmlite.ir.IRBuilder): a builder for generating instructions in the
        current LLVM basic block."""

        return builder.call(self.binding, args, name)

    def __repr__(self) -> str:
        return self.name


class QIRGate(QIRFunc):
    """Represents a quantum gate in QIR"""

    def __init__(self, module: ir.Module, arg_tys: list[Type], name: str):
        """Parameters:

        module (llvmlite.ir.Module): and LLVM module to write to.
        arg_tys (QIRTypes): a collection of LLVM types for the QIR to use.
        name (str): the name of the quantum gate without QIR mangling."""

        self._arg_tys = arg_tys
        suffix = "__body" if "adj" not in name else ""  # Handle __adj gates
        self._mangled_name: str = f"__quantum__qis__{name}{suffix}"
        self._name: str = name
        super().__init__(module, ir.VoidType(), arg_tys, self._mangled_name)

    @property
    def mangled_name(self) -> str:
        """Returns the full mangled QIR name for a gate."""

        return self._mangled_name

    @property
    def name(self) -> str:
        """Returns the core name of the quantum gate in QIR naming convention."""

        return self._name

    @property
    def llvm_type_str(self) -> str:
        """Returns the llvm type as a string."""

        return f'void @{self.self_mangle_name}({", ".join(map(str, self._arg_tys))})'


class CRegFuncs:
    """A collection of QIR Functions that aren't gates"""

    def __init__(self, module: ir.Module, types: QIRTypes):
        """Parameters:

        module (llvmlite.ir.Module): and LLVM module to write to.
        types (QIRTypes): a collection of LLVM types for the QIR to use."""

        self.create_creg_func = QIRFunc(
            module,
            types.bool_type.as_pointer(),
            [types.int_type],
            "create_creg",
        )

        self.creg_to_int_func = QIRFunc(
            module,
            types.int_type,
            [types.bool_type.as_pointer()],
            "get_int_from_creg",
        )

        self.get_creg_bit_func = QIRFunc(
            module,
            types.bool_type,
            [types.bool_type.as_pointer(), types.int_type],
            "get_creg_bit",
        )

        self.set_creg_bit_func = QIRFunc(
            module,
            types.void_type,
            [types.bool_type.as_pointer(), types.int_type, types.bool_type],
            "set_creg_bit",
        )

        self.set_creg_func = QIRFunc(
            module,
            types.void_type,
            [types.bool_type.as_pointer(), types.int_type],
            "set_creg_to_int",
        )

        self.int_result_func = QIRFunc(
            module,
            types.void_type,
            [types.int_type, types.tag_type],
            "__quantum__rt__int_record_output",
        )

    # TODO: add functions to set and read bits in a creg


class MzToBit(QIRFunc):
    """Represents a QIR measure call in the Z basis that writes to a bit in a creg."""

    def __init__(self, module: ir.Module, types: QIRTypes):
        """Parameters:

        module (llvmlite.ir.Module): an LLVM module to write to.
        types (QIRTypes): a collection of LLVM types for the QIR to use.
        """

        super().__init__(
            module,
            types.void_type,
            [types.qubit_ptr_type, types.bool_type.as_pointer(), types.int_type],
            "mz_to_creg_bit",
        )


class QIRGenerator(Generator):
    """Class to generate QIR from SLR. This should enable better compilation of conditional programs."""

    def __init__(self):
        # NOTE: Include files don't exist in QIR
        self.current_block: Block = None
        self.setup_module()
        # Create a field qreg_list
        self._qreg_dict: dict[str, tuple[int, int]] = OrderedDict()
        self._qubit_count: int = 0
        self._measure_count: int = 0
        self._creg_dict: dict[str, tuple[binding.ValueRef, bool]] = {}
        self._result_cregs: set[str] = set()
        self._gate_declaration_cache: dict[str, QIRGate] = {}
        self._barrier_cache: dict[int, QIRFunc] = {}

        # Initialize the permutation map
        self.permutation_map = {}

    def setup_module(self):
        """Helper function to help setup various types and functions needed
        in the QIR production."""
        self._module = ir.Module(name=__file__)

        # store them in a read-only object
        self._types = QIRTypes(self._module)

        # setup the measurement function to be used
        self._mz_to_bit = MzToBit(self._module, self._types)

        # setup functions to manipulate cregs
        self._creg_funcs = CRegFuncs(self._module, self._types)

        # declare the main function
        main_fnty = ir.FunctionType(self._types.void_type, [])
        self._main_func = ir.Function(self._module, main_fnty, name="main")

        # Now implement the function
        self.entry_block = self._main_func.append_basic_block(name="entry")
        self.current_block = self.entry_block
        self._builder = ir.IRBuilder(self.entry_block)
        self._builder.comment(f"// Generated using: PECOS version {pc.__version__}")

        def icmp_signed_closure(op: str):
            return lambda left, right: self._builder.icmp_signed(op, left, right)

        self._op_map: dict = {
            "==": icmp_signed_closure("=="),
            "!=": icmp_signed_closure("!="),
            "<": icmp_signed_closure("<"),
            ">": icmp_signed_closure(">"),
            "<=": icmp_signed_closure("<="),
            ">=": icmp_signed_closure(">="),
            "*": self._builder.mul,
            "/": self._builder.udiv,
            "^": self._builder.xor,
            "&": self._builder.and_,
            "|": self._builder.or_,
            "+": self._builder.add,
            "-": self._builder.sub,
            ">>": self._builder.lshr,
            "<<": self._builder.shl,
        }

    def create_creg(self, creg: CReg):
        """Add a call to create_creg in the current block.

        Parameters:

        creg (slr.vars.CReg): An SLR classical register that should transform into a
        classical register in the QIR.
        """

        if creg.size >= 64:
            msg = f"Classical registers are limited to storing 64 bits (requested: {creg.size})"
            raise ValueError(
                msg,
            )

        self._creg_dict[creg.sym] = (
            self._creg_funcs.create_creg_func.create_call(
                self._builder,
                [ir.Constant(ir.IntType(64), creg.size)],
                f"{creg.sym}",
            ),
            creg.result,
        )

    def create_qreg(self, qreg: QReg):
        """Uses an OrderedDict to globally flatten quantum registers into a single global register.
        Parameters:

        qreg (slr.vars.QReg): An SLR quantum register.
        Its qubits will map to unique numbered qubits in the QIR.
        """

        self._qreg_dict[qreg.sym] = (
            self._qubit_count,
            self._qubit_count + qreg.size - 1,
        )
        self._qubit_count += qreg.size

    def _generate_results(self) -> None:
        """Generates the proper results calls at the end of the SLR program,
        according to all the classical registers that were defined."""
        for reg_name, (reg_inst, result) in self._creg_dict.items():
            if not result:  # ignore non-result cregs
                continue
            # add global tag for each CReg
            reg_name_bytes = bytearray(reg_name.encode("utf-8"))
            tag_type = ir.ArrayType(ir.IntType(8), len(reg_name))
            reg_tag = ir.GlobalVariable(self._module, tag_type, reg_name)
            reg_tag.initializer = ir.Constant(tag_type, reg_name_bytes)
            reg_tag.global_constant = True
            reg_tag.linkage = "private"

            # convert creg to an integer and return that as a result
            c_int = self._creg_funcs.creg_to_int_func.create_call(
                self._builder,
                [reg_inst],
                "",
            )
            reg_tag_gep = reg_tag.gep(
                (ir.Constant(ir.IntType(32), 0), ir.Constant(ir.IntType(32), 0)),
            )
            self._creg_funcs.int_result_func.create_call(
                self._builder,
                [c_int, reg_tag_gep],
                "",
            )

    def generate_block(self, block: Main) -> None:
        """Primary entry point for generation of QIR.
        Parameters:

        block (slr.block.Main): An SLR entry-point block."""

        self._handle_main_block(block)
        self._handle_block(block)
        self._generate_results()
        self._builder.ret_void()

    def _handle_var(self, reg: Reg) -> None:
        match reg:
            case QReg():
                self.create_qreg(reg)
            case CReg():
                self.create_creg(reg)

    def _handle_main_block(self, block: Main) -> None:
        """Process the main block of the SLR program for conversion into a QIR program.

        Parameters:

        block (Main): the SLR entry-point block"""

        for var in block.vars:
            self._handle_var(var)

        for op in block.ops:
            op_name = type(op).__name__
            if op_name == "Vars":
                for var in op.vars:
                    self._handle_var(var)

    def _handle_block(self, block: Block) -> None:
        """Process a block of operations.

        Parameters:

        block (Block): the current SLR block to convert into a QIR block."""

        self._current_block = block
        repeat_times = block.cond if isinstance(block, Repeat) else 1

        for _ in range(repeat_times):
            for block_or_op in block.ops:
                match block_or_op:
                    case If():
                        pred = self._convert_cond_to_pred(block_or_op.cond)
                        if block_or_op.else_block:
                            with self._builder.if_else(pred) as (then, otherwise):
                                with then:
                                    self._handle_block(Block(*block_or_op.ops))
                                with otherwise:
                                    self._handle_block(block_or_op.else_block)
                        else:
                            with self._builder.if_then(pred):
                                self._handle_block(Block(*block_or_op.ops))
                    case Block():
                        self._handle_block(block_or_op)
                    case _:  # non-Block operation
                        self._handle_op(block_or_op)

    def _convert_cond_to_pred(self, cond: CompOp):
        """Converts an SLR expression into a QIR condition."""

        if not isinstance(cond.left, Reg | Bit):
            msg = "Left side of condition must be a register"
            raise TypeError(msg)
        if isinstance(cond.left, Reg):
            # Apply permutation to the register
            reg_sym, _ = self.apply_permutation(cond.left)

            # Get the register pointer
            reg_fetch = self._creg_dict[reg_sym][0]

            lhs = self._creg_funcs.creg_to_int_func.create_call(
                self._builder,
                [reg_fetch],
                "",
            )
        elif isinstance(cond.left, Bit):
            # Apply permutation to the bit
            reg_sym, idx = self.apply_permutation(cond.left)

            # Get the register pointer
            reg_fetch = self._creg_dict[reg_sym][0]

            # Get the bit value
            index = ir.Constant(self._types.int_type, idx)
            lhs = self._creg_funcs.get_creg_bit_func.create_call(
                self._builder,
                [reg_fetch, index],
                "",
            )
        if isinstance(cond.right, int):
            rhs = ir.Constant(self._types.int_type, cond.right)
        else:
            # Apply permutation to the right side if it's a register or bit
            if isinstance(cond.right, Reg | Bit):
                reg_sym, _ = self.apply_permutation(cond.right)
                rhs_reg_fetch = self._creg_dict[reg_sym][0]
            else:
                rhs_reg_fetch = self._creg_dict[cond.right.sym][0]

            rhs = self._creg_funcs.creg_to_int_func.create_call(
                self._builder,
                [rhs_reg_fetch],
                "",
            )
        return self._builder.icmp_signed(cond.symbol, lhs, rhs)

    def _convert_set_op(self, op):
        """Converts an SLR assignment operation to a QIR one"""

        if isinstance(op.left, Bit):
            # Apply permutation to the bit
            reg_sym, idx = self.apply_permutation(op.left)

            # Get the register pointer
            reg_ptr = self._creg_dict[reg_sym][0]

            if isinstance(op.right, int):
                if op.right in (0, 1):
                    rhs = ir.Constant(self._types.bool_type, op.right)
                else:
                    msg = (
                        f"SET operation for bit must have rhs of 0 or 1, got {op.right}"
                    )
                    raise ValueError(msg)
            elif isinstance(op.right, BinOp):
                rhs = self._convert_binary_op(op.right)
            elif isinstance(op.right, UnaryOp):
                rhs = self._convert_unary_op(op.right)
            elif isinstance(op.right, Bit):
                # Apply permutation to the right side bit
                right_reg_sym, right_idx = self.apply_permutation(op.right)

                # Get the register pointer
                rhs_reg_fetch = self._creg_dict[right_reg_sym][0]

                r_index = ir.Constant(self._types.int_type, right_idx)
                rhs = self._creg_funcs.get_creg_bit_func.create_call(
                    self._builder,
                    [rhs_reg_fetch, r_index],
                    "",
                )
            else:
                rhs_reg_fetch = self._creg_dict[op.right.sym][0]
                rhs = self._creg_funcs.creg_to_int_func.create_call(
                    self._builder,
                    [rhs_reg_fetch],
                    "",
                )

            # Set the bit value
            l_index = ir.Constant(self._types.int_type, idx)
            return self._creg_funcs.set_creg_bit_func.create_call(
                self._builder,
                [reg_ptr, l_index, rhs],
                "",
            )
        if isinstance(op.left, CReg):
            # Apply permutation to the register
            reg_sym, _ = self.apply_permutation(op.left)

            # Get the register pointer
            reg_ptr = self._creg_dict[reg_sym][0]

            if isinstance(op.right, int):
                rhs = ir.Constant(self._types.int_type, op.right)
            elif isinstance(op.right, BinOp):
                rhs = self._convert_binary_op(op.right)
            elif isinstance(op.right, UnaryOp):
                rhs = self._convert_unary_op(op.right)
            elif isinstance(op.right, Bit):
                # Apply permutation to the right side bit
                right_reg_sym, right_idx = self.apply_permutation(op.right)

                # Get the register pointer
                rhs_reg_fetch = self._creg_dict[right_reg_sym][0]

                r_index = ir.Constant(self._types.int_type, right_idx)
                rhs = self._creg_funcs.get_creg_bit_func.create_call(
                    self._builder,
                    [rhs_reg_fetch, r_index],
                    "",
                )
            else:
                # Apply permutation to the right side register
                right_reg_sym, _ = self.apply_permutation(op.right)

                # Get the register pointer
                rhs_reg_fetch = self._creg_dict[right_reg_sym][0]

                rhs = self._creg_funcs.creg_to_int_func.create_call(
                    self._builder,
                    [rhs_reg_fetch],
                    "",
                )

            # Set the register value
            return self._creg_funcs.set_creg_func.create_call(
                self._builder,
                [reg_ptr, rhs],
                "",
            )
        msg = f"SET operation not implemented for {op.left} (type: {type(op.left)})"
        raise NotImplementedError(msg)

    def _convert_binary_op(self, op):
        """Converts an SLR binary operation to a QIR arithmetic instruction"""

        lhs, rhs = None, None
        if isinstance(op.left, int):
            # hack
            if isinstance(op.right, Bit):
                lhs = ir.Constant(self._types.bool_type, op.left)
            else:
                lhs = ir.Constant(self._types.int_type, op.left)
        elif isinstance(op.left, BinOp):
            lhs = self._convert_binary_op(op.left)
        elif isinstance(op.left, UnaryOp):
            lhs = self._convert_unary_op(op.left)
        elif isinstance(op.left, Bit):
            reg_fetch = self._creg_dict[op.left.reg.sym][0]
            l_index = ir.Constant(self._types.int_type, op.left.index)
            lhs = self._creg_funcs.get_creg_bit_func.create_call(
                self._builder,
                [reg_fetch, l_index],
                "",
            )
        else:
            reg_fetch = self._creg_dict[op.left.sym][0]
            lhs = self._creg_funcs.creg_to_int_func.create_call(
                self._builder,
                [reg_fetch],
                "",
            )
        if isinstance(op.right, int):
            # hack
            if isinstance(op.left, Bit):
                rhs = ir.Constant(self._types.bool_type, op.right)
            else:
                rhs = ir.Constant(self._types.int_type, op.right)
        elif isinstance(op.right, BinOp):
            rhs = self._convert_binary_op(op.right)
        elif isinstance(op.right, UnaryOp):
            rhs = self._convert_unary_op(op.right)
        elif isinstance(op.right, Bit):
            rhs_reg_fetch = self._creg_dict[op.right.reg.sym][0]
            r_index = ir.Constant(self._types.int_type, op.right.index)
            rhs = self._creg_funcs.get_creg_bit_func.create_call(
                self._builder,
                [rhs_reg_fetch, r_index],
                "",
            )
        else:
            rhs_reg_fetch = self._creg_dict[op.right.sym][0]
            rhs = self._creg_funcs.creg_to_int_func.create_call(
                self._builder,
                [rhs_reg_fetch],
                "",
            )
        return self._op_map[op.symbol](lhs, rhs)

    def _convert_unary_op(self, op):
        """Converts a unary negation operation to QIR binary instructions via llvmlite helper"""
        if isinstance(op.value, int):
            match op:
                case NEG():
                    return ir.Constant(self._types.int_type, -op.value)
                case NOT():
                    return ir.Constant(self._types.int_type, ~op.value)
        elif isinstance(op.value, Bit):
            reg_fetch = self._creg_dict[op.value.reg.sym][0]
            index = ir.Constant(self._types.int_type, op.value.index)
            reg_val = self._creg_funcs.get_creg_bit_func.create_call(
                self._builder,
                [reg_fetch, index],
                "",
            )
            match op:
                case NEG():
                    return self._builder.neg(reg_val)
                case NOT():
                    return self._builder.not_(reg_val)
        else:
            reg_fetch = self._creg_dict[op.value.sym][0]
            reg_val = self._creg_funcs.creg_to_int_func.create_call(
                self._builder,
                [reg_fetch],
                "",
            )
            match op:
                case NEG():
                    return self._builder.neg(reg_val)
                case NOT():
                    return self._builder.not_(reg_val)

    def _handle_op(self, op) -> None:
        """Process a single operation.

        op (Any): An op must be an SLR construct and not an arbitrary python type."""

        match op:
            case Barrier():
                self._handle_barrier(op)
            case Comment():
                new_comment = op.txt.replace("\n", "")
                self._builder.comment(
                    new_comment,
                )  # TODO: Handle 'space', 'newline' params
            case Permute():
                self._handle_permute(op)
            case SET():
                self._convert_set_op(op)
            case BinOp():
                self._convert_binary_op(op)
            case UnaryOp():
                self._convert_unary_op(op)
            case Vars():
                for var in op.vars:
                    self._handle_var(var)
            case qgate_base.QGate():
                self._handle_quantum_gate(op)
            case _:
                msg = f"Unsupported operation: {type(op).__name__}"
                raise NotImplementedError(msg)

    def _handle_barrier(self, barrier: Barrier) -> None:
        """Process a barrier operation."""
        length = 0
        qubits: list[Qubit] = []

        for item in barrier.qregs:
            match item:
                case Qubit():
                    length += 1
                    qubits.append(item)

                case QReg():
                    length += item.size
                    qubits.extend(item.elems)
                case _:  # assume tuple[QReg]
                    for qreg in item:
                        length += qreg.size
                        qubits.extend(qreg.elems)
                # TODO: tuple[QReg]

        if length not in self._barrier_cache:
            self._barrier_cache[length] = QIRFunc(
                self._module,
                self._types.void_type,
                [self._types.qubit_ptr_type] * length,
                f"__quantum__qis__barrier{length}__body",
            )
        barrier_func = self._barrier_cache[length]
        barrier_func.create_call(
            self._builder,
            [self._qarg_to_qubit_ptr(index) for index in qubits],
            name="",
        )

    def _handle_quantum_gate(self, gate: qgate_base.QGate) -> None:
        """Process a quantum gate.

        gate (slr.qubit.qgate_base.QGate): An SLR quantum gate or measurement operation
        to transform into a QIR Gate"""

        match type(gate).__name__:
            case "Measure":
                creg_or_bit = gate.cout[0]
                if isinstance(creg_or_bit, CReg):
                    ll_creg = self._creg_dict[creg_or_bit.sym][0]
                    for i, q in enumerate(gate.qargs[0]):
                        self._measure_count += 1
                        qubit_ptr = self._qarg_to_qubit_ptr(q)
                        self._mz_to_bit.create_call(
                            self._builder,
                            [qubit_ptr, ll_creg, ir.Constant(self._types.int_type, i)],
                            name="",
                        )
                elif isinstance(creg_or_bit, Bit):
                    ll_creg = self._creg_dict[creg_or_bit.reg.sym][0]
                    self._measure_count += 1
                    qubit_ptr = self._qarg_to_qubit_ptr(gate.qargs[0])
                    self._mz_to_bit.create_call(
                        self._builder,
                        [
                            qubit_ptr,
                            ll_creg,
                            ir.Constant(self._types.int_type, creg_or_bit.index),
                        ],
                        name="",
                    )
            case _:
                self._create_qgate_call(gate)

    def _create_qgate_call(self, gate: qgate_base.QGate) -> None:
        """A helper method to generate QIR for quantum gate operation.

        gate (QGate): a quantum gate to generate as QIR."""

        qgate_meta = QIRGateMetadata[gate.sym]
        # If theres a decomposition lambda, invoke that with the gate to generate
        # the decomposed gates needed in the circuit. The lambda defines any
        # necessary mappings of parameters and qargs to the decomposed gates from
        # the 'source' gate
        if qgate_meta.decomposer:
            self._builder.comment(f"Decomposing gate: {gate.sym}")
            decomposed_gates = qgate_meta.decomposer(gate)
            for decomposed_gate in decomposed_gates:
                self._create_qgate_call(decomposed_gate)
            return

        if isinstance(gate.qargs[0], QReg):
            for qubit in gate.qargs[0].elems:
                new_gate = gate.copy()
                new_gate.qargs = [qubit]
                self._create_qgate_call(new_gate)
            return
        if (
            isinstance(gate.qargs, tuple)
            and len(gate.qargs) != gate.qsize
            and all(isinstance(q, Qubit) for q in gate.qargs)
        ):
            for qubit in gate.qargs:
                new_gate = gate.copy()
                new_gate.qargs = [qubit]
                self._create_qgate_call(new_gate)
            return
        if isinstance(gate.qargs, tuple) and all(
            isinstance(e, tuple) for e in gate.qargs
        ):
            for pair in gate.qargs:
                new_gate = gate.copy()
                new_gate.qargs = pair
                self._create_qgate_call(new_gate)
            return
        qargs = gate.qargs
        if len(qargs) != gate.qsize:
            msg = f"Gate {gate.sym} expects {gate.qsize} qubits, but {len(qargs)} were provided."
            raise ValueError(
                msg,
            )

        if gate.sym not in self._gate_declaration_cache:
            declare_args = []
            if gate.has_parameters:
                declare_args = [self._types.double_type] * len(gate.params)
            declare_args.extend([self._types.qubit_ptr_type] * gate.qsize)

            gate_declaration = QIRGate(
                self._module,
                declare_args,
                name=qgate_meta.qir_name,
            )
            self._gate_declaration_cache[gate.sym] = gate_declaration

        gate_declaration = self._gate_declaration_cache[gate.sym]
        gate_args = []
        if gate.has_parameters:
            gate_args = [
                ir.Constant(self._types.double_type, param) for param in gate.params
            ]
        gate_args.extend([self._qarg_to_qubit_ptr(qarg) for qarg in qargs])

        # Create the actual invocation on the builder using the args passed in
        gate_declaration.create_call(self._builder, gate_args, name="")

    def _qarg_to_qubit_ptr(self, qarg: Qubit) -> ir.Constant:
        """Return a pointer to a qubit in the 'global quantum register', based on the register
        and index passed in the `qarg` param.

        Parameters:

        qarg (slr.qubit.vars.Qubit): a qubit in an SLR quantum register (QReg)"""

        # Apply permutation to the qubit
        reg_sym, index = self.apply_permutation(qarg)

        # Get the qubit index in the global array
        qubit_index = self._qreg_dict[reg_sym][0] + index

        return ir.Constant(self._types.int_type, qubit_index).inttoptr(
            self._types.qubit_ptr_type,
        )

    def _handle_permute(self, op: Permute) -> None:
        """Handle a permutation operation.

        Parameters:
            op (Permute): The permutation operation to handle.
        """
        # Get the input and output elements
        elems_i = op.elems_i
        elems_f = op.elems_f

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

            # Add a comment to describe the permutation
            self._builder.comment(f"; Permutation: {reg_i.sym} <-> {reg_f.sym}")

            # Compose the new permutation with the existing one
            updated_perm_map = self._compose_permutation_maps(new_perm_map)

            # Update the permutation map
            self.permutation_map = updated_perm_map

        elif isinstance(elems_i, CReg) and isinstance(elems_f, CReg):
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

            # Get the register pointers
            reg_i_ptr = self._creg_dict[reg_i.sym][0]
            reg_f_ptr = self._creg_dict[reg_f.sym][0]

            # Use XOR operations to swap the registers
            # a = a ^ b
            a_xor_b = self._builder.xor(
                self._creg_funcs.creg_to_int_func.create_call(
                    self._builder,
                    [reg_i_ptr],
                    "",
                ),
                self._creg_funcs.creg_to_int_func.create_call(
                    self._builder,
                    [reg_f_ptr],
                    "",
                ),
            )
            self._creg_funcs.set_creg_func.create_call(
                self._builder,
                [reg_i_ptr, a_xor_b],
                "",
            )

            # b = b ^ a
            b_xor_a = self._builder.xor(
                self._creg_funcs.creg_to_int_func.create_call(
                    self._builder,
                    [reg_f_ptr],
                    "",
                ),
                a_xor_b,
            )
            self._creg_funcs.set_creg_func.create_call(
                self._builder,
                [reg_f_ptr, b_xor_a],
                "",
            )

            # a = a ^ b
            a_xor_b_xor_a = self._builder.xor(
                a_xor_b,
                b_xor_a,
            )
            self._creg_funcs.set_creg_func.create_call(
                self._builder,
                [reg_i_ptr, a_xor_b_xor_a],
                "",
            )

            # Add a comment to describe the permutation
            self._builder.comment(f"; Permutation: {reg_i.sym} <-> {reg_f.sym}")

        elif (
            not isinstance(elems_i, Reg)
            and not isinstance(elems_f, Reg)
            and hasattr(elems_i, "__iter__")
            and hasattr(elems_f, "__iter__")
        ):
            # Element-wise permutation
            if len(elems_i) != len(elems_f):
                msg = f"Cannot permute different numbers of elements: {len(elems_i)} and {len(elems_f)}"
                raise ValueError(msg)

            # Check if we're dealing with quantum bits
            if any(hasattr(e, "reg") and isinstance(e.reg, QReg) for e in elems_i):
                # Element-wise permutation for quantum registers
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
                    if (
                        hasattr(ei.reg, "sym")
                        and hasattr(ef.reg, "sym")
                        and isinstance(ei.reg, QReg)
                    ):
                        # Create a key from the input element's register sym and index
                        key = (ei.reg.sym, ei.index)
                        # Map it to the output element's register sym and index
                        new_perm_map[key] = (ef.reg.sym, ef.index)

                # Add a comment to describe the permutation
                perm_str = ", ".join(
                    [f"{ei} -> {ef}" for ei, ef in zip(elems_i, elems_f)],
                )
                self._builder.comment(f"; Permutation: {perm_str}")

                # Compose the new permutation with the existing one
                updated_perm_map = self._compose_permutation_maps(new_perm_map)

                # Update the permutation map
                self.permutation_map = updated_perm_map

            # Check if we're dealing with classical bits
            elif any(hasattr(e, "reg") and isinstance(e.reg, CReg) for e in elems_i):
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

                # Create a temporary bit if needed
                if cycles:
                    temp_var = "_bit_swap"
                    if temp_var not in self._creg_dict:
                        temp_ptr = self._creg_funcs.create_creg_func.create_call(
                            self._builder,
                            [ir.Constant(self._types.int_type, 1)],
                            temp_var,
                        )
                        self._creg_dict[temp_var] = (temp_ptr, False)
                    else:
                        temp_ptr = self._creg_dict[temp_var][0]

                # Process each cycle
                for cycle in cycles:
                    # Use the temporary bit for all cycles
                    first = cycle[0]
                    first_ptr = self._creg_dict[first.reg.sym][0]

                    # Save the first element's value to the temporary bit
                    first_val = self._creg_funcs.get_creg_bit_func.create_call(
                        self._builder,
                        [first_ptr, ir.Constant(self._types.int_type, first.index)],
                        "",
                    )
                    self._creg_funcs.set_creg_bit_func.create_call(
                        self._builder,
                        [temp_ptr, ir.Constant(self._types.int_type, 0), first_val],
                        "",
                    )

                    # Move each element's value to its predecessor in the cycle
                    for i in range(len(cycle) - 1):
                        curr = cycle[i]
                        next_elem = cycle[i + 1]
                        curr_ptr = self._creg_dict[curr.reg.sym][0]
                        next_ptr = self._creg_dict[next_elem.reg.sym][0]

                        next_val = self._creg_funcs.get_creg_bit_func.create_call(
                            self._builder,
                            [
                                next_ptr,
                                ir.Constant(self._types.int_type, next_elem.index),
                            ],
                            "",
                        )
                        self._creg_funcs.set_creg_bit_func.create_call(
                            self._builder,
                            [
                                curr_ptr,
                                ir.Constant(self._types.int_type, curr.index),
                                next_val,
                            ],
                            "",
                        )

                    # Assign the temporary bit to the last element
                    last = cycle[-1]
                    last_ptr = self._creg_dict[last.reg.sym][0]
                    temp_val = self._creg_funcs.get_creg_bit_func.create_call(
                        self._builder,
                        [temp_ptr, ir.Constant(self._types.int_type, 0)],
                        "",
                    )
                    self._creg_funcs.set_creg_bit_func.create_call(
                        self._builder,
                        [
                            last_ptr,
                            ir.Constant(self._types.int_type, last.index),
                            temp_val,
                        ],
                        "",
                    )

                # Add a comment to describe the permutation
                perm_str = ", ".join(
                    [f"{ei} -> {ef}" for ei, ef in zip(elems_i, elems_f)],
                )
                self._builder.comment(f"; Permutation: {perm_str}")

                # For classical bit permutations, we're physically moving the values,
                # so we don't need to update the permutation map.
                # The operations should still refer to the original bits.

    def _compose_permutation_maps(self, new_perm_map):
        """Compose a new permutation map with the existing one.

        This method composes two permutation maps by applying the new map to the results of the existing map.
        For example, if the existing map maps A to B, and the new map maps B to C, the composed map will map A to C.

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

    def apply_permutation(self, qarg: Qubit | Bit | QReg | CReg) -> tuple[str, int]:
        """Apply the permutation mapping to a qubit or bit and return the permuted register symbol and index.

        Parameters:
            qarg (Qubit | Bit | QReg | CReg): The qubit, bit, or register to apply the permutation to.

        Returns:
            tuple[str, int]: The permuted register symbol and index.
        """
        # Handle Qubit/Bit objects which have a reg attribute
        if hasattr(qarg, "reg") and hasattr(qarg, "index") and hasattr(qarg.reg, "sym"):
            key = (qarg.reg.sym, qarg.index)
            if key in self.permutation_map:
                return self.permutation_map[key]
            return (qarg.reg.sym, qarg.index)
        # Handle QReg/CReg objects which are registers themselves
        if hasattr(qarg, "sym"):
            # For a register, we return the symbol and index 0 (whole register)
            return (qarg.sym, 0)
        # Fallback for other types
        return (qarg.reg.sym, qarg.index)

    def _ll_with_attributes(self) -> str:
        """Patches attributes into the .ll for the program:

        Example attributes:
        attributes #0 = { "entry_point" "output_labeling_schema"
        "qir_profiles"="custom" "required_num_qubits"="22" "required_num_results"="22" }
        """
        ll_text: str = _fix_internal_consts(str(self._module))
        mod_w_attr = ll_text.replace("@main()", "@main() #0")

        # to get around line length limitations
        mod_w_attr += '\nattributes #0 = { "entry_point"'
        mod_w_attr += ' "qir_profiles"="custom"'
        mod_w_attr += f' "required_num_qubits"="{self._qubit_count}"'
        mod_w_attr += f' "required_num_results"="{self._measure_count}" }}'
        return mod_w_attr

    def get_output(self) -> str:
        """Stringify the module as .ll text"""
        binding.shutdown()
        return self._ll_with_attributes()

    def get_bc(self) -> bytes:
        """Return LLVM bitcode for the text"""
        bc = binding.parse_assembly(self.get_output()).as_bitcode()
        binding.shutdown()
        return bc


def _fix_internal_consts(llvm_ir: str) -> str:
    """Converts all global variable tag declarations to remove quotation marks
    from the numbers. Ex. @"1" = --- becomes @1 = ---

    Parameters
    ----------
    llvm_ir : str
    The llvm string we are trying to modify

    Returns
    -------
    tuple(str, dict)
    Returns a tuple that contains the updated llvm ir string, and a dictionary that contains
    the variable and its corresponding string constant
    """

    # substitute all instances of variable num with quotes, with just number (@"0" -> @0)

    return re.sub('([@%])"([^"]+)"', r"\1\2", llvm_ir)
