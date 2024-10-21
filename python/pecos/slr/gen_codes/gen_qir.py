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

from collections import OrderedDict
from dataclasses import dataclass
from functools import cached_property

from llvmlite import ir
from llvmlite.ir import IntType, PointerType, VoidType

from pecos import __version__
from pecos.qeclib.qubit import qgate_base
from pecos.qeclib.qubit.sq_hadamards import H
from pecos.qeclib.qubit.tq_cliffords import CX
from pecos.slr import Block, Main
from pecos.slr.gen_codes.generator import Generator
from pecos.slr.misc import Barrier, Comment, Permute
from pecos.slr.vars import CReg, QReg, Reg, Vars


@dataclass
class QIRTypes:
    """Dataclass to hold the types used in QIR compilation"""

    boolType: IntType
    intType: IntType
    qubitPtrType: PointerType
    resultPtrType: PointerType


class QIRGenerator(Generator):
    """Class to generate QIR from SLR. This should enable better compilation of conditional programs."""

    def __init__(self, includes: list[str] | None = None):
        # self.output = []
        self.current_block: Block = None
        # self.includes = includes
        # self.cond = None
        self.setup_module()
        # Create a field qreg_list
        self.qreg_dict: dict[str, tuple[int, int]] = OrderedDict()
        self._qubit_count: int = 0
        self._creg: dict[str, object] = {}  # replace 'object' with the correct type

    def setup_module(self):
        self._module = ir.Module(name=__file__)

        # Create some useful types to use in compilation later
        boolTy = ir.IntType(1)
        intTy = ir.IntType(64)
        qubitTy = self._module.context.get_identified_type("Qubit")
        resultTy = self._module.context.get_identified_type("Result")
        qubitPtrTy = qubitTy.as_pointer()
        resultPtrTy = resultTy.as_pointer()

        self._types = QIRTypes(boolTy, intTy, qubitPtrTy, resultPtrTy)
        fnty = ir.FunctionType(VoidType, [])

        self._main_func = ir.Function(self._module, fnty, name="main")

        self.create_creg_func = ir.Function(
            self._module,
            ir.FunctionType(boolTy.as_pointer(), [intTy]),
            name="create_creg",
        )

        self.mz_to_creg_bit_func = ir.Function(
            self._module,
            ir.FunctionType(VoidType, qubitPtrType, [boolTy.as_pointer(), intTy]),
            name="mz_to_creg_bit"
        )

        # Now implement the function
        self.entry_block = self._main_func.append_basic_block(name="entry")
        self.current_block = self.entry_block
        self.builder = ir.IRBuilder(self.entry_block)
        self.builder.comment(f"// Generated using: PECOS version {__version__}")

    def create_creg(self, creg: CReg):
        """Add a call to create_creg in the current block."""
        self._creg[creg.sym] = self.builder.call(
            self.create_creg_func,
            [ir.Constant(ir.IntType(64), creg.size)],
            name=f"{creg.sym}",
        )

    def create_qreg(self, qreg: QReg):
        """Uses an OrderedDict to globally flatten quantum registers into a single global register."""
        self.qreg_dict[qreg.sym] = (self._qubit_count, self._qubit_count + qreg.size - 1)
        self._qubit_count += qreg.size
        # return self.builder.call(self.create_creg_func, [ir.Constant(ir.IntType(64), creg.size)], name = f"{creg.sym}")

    def generate_block(self, block: Main) -> None:
        """Primary entry point for generation. Processes an slr Main block."""

        self._handle_main_block(block)
        self._handle_block(block)

        self.builder.ret_void()

    def _handle_var(self, reg: Reg) -> None:
        match reg:
            case QReg():
                self.create_qreg(reg)
            case CReg():
                self.create_creg(reg)

    def _handle_main_block(self, block) -> None:
        """Process the main block of the program."""
        for var in block.vars:
            self._handle_var(var)

        for op in block.ops:
            op_name = type(op).__name__
            if op_name == "Vars":
                for var in op.vars:
                    self._handle_var(var)

    def _handle_block(self, block: Block) -> None:
        """Process a block of operations."""

        self._current_block = block
        for block_or_op in block.ops:
            match block_or_op:
                case Block():
                    self._handle_block(block_or_op)
                case _:  # non-block operation
                    self._handle_op(block_or_op)

    def _handle_op(self, op) -> None:
        """Process a single operation."""

        match op:
            case Barrier(qregs):
                pass
            case Comment(txt, space, newline):
                pass
            case Permute(elems_i, elems_f, comment):
                pass
            case Vars(vars):
                pass
            case CReg(sym, size):
                pass
            case qgate_base.QGate(_, _):
                self._handle_quantum_gate(op)

    def _handle_quantum_gate(self, gate: qgate_base.QGate) -> None:
        """Process a quantum gate."""
        match gate:
            case H(sym, qargs, params):
                self.builder.call(self._h_func, [self._qarg_to_qubit_ptr(qargs[0])], name="")
            case CX(_, qargs, _):
                self.builder.call(
                    self._cx_func,
                    [
                        self._qarg_to_qubit_ptr(qargs[0]),
                        self._qarg_to_qubit_ptr(qargs[1]),
                    ],
                    name="",
                )
            case Measure():
                self.mz_to_creg_bit

    @cached_property
    def _h_func(self) -> ir.Function:
        return ir.Function(
            self._module,
            ir.FunctionType(VoidType, [self._types.qubitPtrType]),
            name="__quantum__qis__h__body",
        )

    @cached_property
    def _cx_func(self) -> ir.Function:
        return ir.Function(
            self._module,
            ir.FunctionType(VoidType, [self._types.qubitPtrType, self._types.qubitPtrType]),
            name="__quantum__qis__cnot__body",
        )

    def _qarg_to_qubit_ptr(self, qarg):
        """Return a pointer to a qubit in the global register, based on the register and index
        passed in the `qarg` param."""
        index = qarg.index
        qubit_index = self.qreg_dict[qarg.sym][0] + index
        return self.builder.inttoptr(ir.Constant(ir.IntType(64), qubit_index), self._types.qubitPtrType)

    def get_output(self) -> str:
        # This will stringify the module as ll via __repr__
        return str(self._module)
