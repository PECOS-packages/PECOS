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

from llvmlite import ir
from llvmlite.ir import IntType, PointerType, VoidType

from pecos import __version__

# from pecos.qeclib.qubit.qgate_base import QGate
from pecos.qeclib.qubit import qgate_base
from pecos.qeclib.qubit.sq_hadamards import H
from pecos.qeclib.qubit.tq_cliffords import CX

# from pecos.slr.block import Block
from pecos.slr import block
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
        self.current_block: block.Block = None
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

        #         declare i1 @get_creg_bit(i1*, i64)

        # declare void @set_creg_bit(i1*, i64, i1)

        # declare void @set_creg_to_int(i1*, i64)

        # declare i1 @__quantum__qis__read_result__body(%Result*)

        # declare i1* @create_creg(i64)

        # declare i64 @get_int_from_creg(i1*)

        # declare void @mz_to_creg_bit(%Qubit*, i1*, i64)

        # declare void @__quantum__rt__int_record_output(i64, i8*)
        self.create_creg_func = ir.Function(
            self._module,
            ir.FunctionType(boolTy.as_pointer(), [intTy]),
            name="create_creg",
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

    def generate_block(self, block):
        """Process an slr block."""

        # IMPORTANT NOTE: NEED TO UPDATE VALUE OF SELF.CURRENT_BLOCK

        block_name = type(block).__name__

        # TODO: see if we can refactor to remove casing on string name
        # of types
        match block_name:
            case "Main":
                self._handle_main_block(block)

            case "If":  # NOT IMPLEMENTED
                self.cond = self.generate_op(block.cond)
                self.block_op_loop(block)
                self.cond = None

            case "Repeat":  # NOT IMPLEMENTED
                for _ in range(block.cond):
                    self.block_op_loop(block)
            case _:
                self._handle_block(block)

    def _handle_var(self, reg: Reg):
        match reg:
            case QReg(_, _):
                self.create_qreg(reg)
            case CReg(_, _):
                self.create_creg(reg)

    def _handle_main_block(self, block):
        """Process the main block of the program."""
        for var in block.vars:
            self._handle_var(var)

        for op in block.ops:
            op_name = type(op).__name__
            if op_name == "Vars":
                for var in op.vars:
                    self._handle_var(var)

    def _handle_block(self, block):
        """Process a block of operations."""
        self._current_block = block
        for op in block.ops:
            if hasattr(op, "ops"):  # TODO: figure out how to identify Block types without using isinstance
                self._handle_block(op)
            else:
                self._handle_op(op)

    def _handle_op(self, op):
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

        # From gen_qasm.py:
        # op_name = type(op).__name__

        # stat = False

        # if op_name == "Barrier":
        #     stat = True
        #     if isinstance(op.qregs, list | tuple | set):
        #         qubits = []
        #         for q in op.qregs:
        #             qubits.append(str(q))
        #         qubits = ", ".join(qubits)
        #     else:
        #         qubits = op.qregs

        #     op_str = f"barrier {qubits};"
        # elif op_name == "Comment":
        #     txt = op.txt.split("\n")
        #     if op.space:
        #         txt = [f" {t}" if t.strip() != "" else t for t in txt]
        #     if not op.newline:
        #         txt = [f"<same_line>{t}" if t.strip() != "" else t for t in txt]

        #     txt = [f"//{t}" if t.strip() != "" else t for t in txt]
        #     op_str = "\n".join(txt)

        # elif op_name == "Permute":
        #     op_str = process_permute(op)

        # elif op_name == "SET":
        #     stat = True
        #     op_str = self.process_set(op)

        # elif op_name in [
        #     "EQUIV",
        #     "NEQUIV",
        #     "LT",
        #     "GT",
        #     "LE",
        #     "GE",
        #     "MUL",
        #     "DIV",
        #     "XOR",
        #     "AND",
        #     "OR",
        #     "PLUS",
        #     "MINUS",
        #     "RSHIFT",
        #     "LSHIFT",
        # ]:
        #     op_str = self.process_general_binary_op(op)

        # elif op_name in ["NEG", "NOT"]:
        #     op_str = self.process_general_unary_op(op)

        # elif op_name == "Vars":
        #     op_str = None

        # elif op_name in ["CReg", "QReg"]:
        #     op_str = str(op.sym)

        # elif op_name in ["Bit", "Qubit"]:
        #     op_str = f"{op.reg.sym}[{op.index}]"

        # elif isinstance(op, int):
        #     op_str = str(op)

        # elif hasattr(op, "is_qgate") and op.is_qgate:
        #     stat = True
        #     op_str = self.process_qgate(op)

        # elif hasattr(op, "gen"):
        #     op_str = op.gen(self)

        # elif hasattr(op, "qasm"):
        #     stat = True
        #     op_str = op.qasm()

        # else:
        #     msg = f"Operation '{op}' not handled!"
        #     raise NotImplementedError(msg)

        # if self.cond and stat and op_str:
        #     cond = self.cond
        #     if cond.startswith("(") and cond.endswith(")"):
        #         cond = cond[1:-1]
        #     op_list = op_str.split("\n")
        #     op_new = []
        #     for o in op_list:

        #         o = o.strip()
        #         if o != "" and not o.startswith("//"):
        #             for qi in o.split(";"):
        #                 qi = qi.strip()
        #                 if qi != "" and not qi.startswith("//"):
        #                     op_new.append(f"if({cond}) {qi};")
        #         else:
        #             op_new.append(o)

        #     op_str = "\n".join(op_new)

        # return op_str

    def _handle_quantum_gate(self, gate: qgate_base.QGate):
        """Process a quantum gate."""
        # gate.qargs gate.params
        match gate:
            case H(sym, qargs, params):
                h_func = ir.Function(
                    self._module,
                    ir.FunctionType(VoidType, [self._types.qubitPtrType]),
                    name="__quantum__qis__h__body",
                )  # move this to earlier
                index = qargs[0].index
                qubit_index = self.qreg_dict[qargs[0].sym][0] + index
                inttoptr = self.builder.inttoptr(ir.Constant(ir.IntType(64), qubit_index), self._types.qubitPtrType)
                self.builder.call(h_func, [inttoptr], name="")
            case CX(_, _):
                cx_func = ir.Function(
                    self._module,
                    ir.FunctionType(VoidType, [self._types.qubitPtrType, self._types.qubitPtrType]),
                    name="__quantum__qis__cnot__body",
                )

    def get_output(self) -> str:
        return str(self._module)


#         declare void @__quantum__qis__h__body(%Qubit*) local_unnamed_addr

# declare void @__quantum__qis__x__body(%Qubit*) local_unnamed_addr

# declare void @__quantum__qis__cnot__body(%Qubit*, %Qubit*) local_unnamed_addr
