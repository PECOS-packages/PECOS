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

from llvmlite import binding, ir
from llvmlite.ir import IntType, DoubleType, PointerType, Type, VoidType

from pecos import __version__
from pecos.qeclib.qubit import qgate_base
from pecos.qeclib.qubit.measures import Measure
from pecos.qeclib.qubit.sq_hadamards import H
from pecos.qeclib.qubit.tq_cliffords import CX
from pecos.slr import Block, Main
from pecos.slr.gen_codes.generator import Generator
from pecos.slr.misc import Barrier, Comment, Permute
from pecos.slr.vars import CReg, QReg, Reg, Vars, Qubit

class QIRTypes:
    """Class to hold the types used in QIR compilation"""

    def __init__(self, module: ir.Module):
        # Create some useful types to use in compilation later
        qubitTy = module.context.get_identified_type("Qubit")
        resultTy = module.context.get_identified_type("Result")
        self.voidType: VoidType = ir.VoidType()
        self.boolType: IntType = ir.IntType(1)
        self.intType: IntType = ir.IntType(64)
        self.doubleType: DoubleType = ir.DoubleType()
        self.qubitPtrType: PointerType = qubitTy.as_pointer()
        self.resultPtrType: PointerType = resultTy.as_pointer()        


class QIRFunc:
    """Represents a callable function in a QIR program"""
    
    def __init__(self, module: ir.Module, ret_ty: Type, arg_tys: list[Type], name: str):
        self.binding = ir.Function(
            module,
            ir.FunctionType(ret_ty, arg_tys),
            name=name,
        )

    def call(self, builder: ir.IRBuilder, args: list[binding.ValueRef], name: str) -> binding.ValueRef:
        """A helper method to call a QIR Gate"""
        return builder.call(self.binding, args, name)

    def __repr__(self) -> str:
        return self.name
        
    
class QIRGate(QIRFunc):
    """Represents a quantum gate in QIR"""
    
    def __init__(self, module: ir.Module, arg_tys: list[Type], name: str):
        self._arg_tys = arg_tys
        self._mangled_name: str = "__quantum__qis__" + name + "__body"
        self._name:str = name
        super().__init__(module, ir.VoidType(), arg_tys, self._mangled_name)

    @property
    def mangled_name(self) -> str:
        return self._mangled_name

    @property
    def name(self) -> str:
        return self._name

    @property
    def llvm_type_str(self) -> str:
        """returns the llvm typeas a string"""
        
        return f'void @{self_mangle_name}({", ".join(map(str, self._arg_tys))})'

        
class QIRGates:
    """A collection of QIR gates."""
    
    def __init__(self, module: ir.Module, types: QIRTypes):
        self.h_func = QIRGate(
            module,
            [types.qubitPtrType],
            name="h",
        )
        self.cx_func = QIRGate(
            module,
            [types.qubitPtrType, types.qubitPtrType],
            name="cx",
        )

        
class CRegFuncs:
    """A collection of QIR Functions that aren't gates"""
    
    def __init__(self, module: ir.Module, types: QIRTypes):
        self.create_creg_func = QIRFunc(
            module,
            types.boolType.as_pointer(),
            [types.intType],
            "create_creg",
        )
    #TODO: add functions to set and read bits in a creg and also conver to an integer

    
class MzToBit(QIRFunc):
    """Represents a QIR measure call in the Z basis that writes to a bit in a creg"""
    
    def __init__(self, module: ir.Module, types: QIRTypes):
        super().__init__(module,
                         types.voidType,
                         [types.qubitPtrType, types.boolType.as_pointer(), types.intType],
                         "mz_to_creg_bit")


class QIRGenerator(Generator):
    """Class to generate QIR from SLR. This should enable better compilation of conditional programs."""

    def __init__(self, includes: list[str] | None = None):
        self.current_block: Block = None
        self.setup_module()
        # Create a field qreg_list
        self._qreg_dict: dict[str, tuple[int, int]] = OrderedDict()
        self._qubit_count: int = 0
        self._creg_dict: dict[str, object] = {}  # replace 'object' with the correct type

    def setup_module(self):
        self._module = ir.Module(name=__file__)        

        # store them in a read-only object
        self._types = QIRTypes(self._module)

        # setup the quantum gates to be used in compilation
        self._gates = QIRGates(self._module, self._types)

        # setup the measurement function to be used
        self._mz_to_bit = MzToBit(self._module, self._types)

        # setup functions to manipulate cregs
        self._creg_funcs = CRegFuncs(self._module, self._types)

        # declare the main function
        main_fnty = ir.FunctionType(self._types.voidType, [])
        self._main_func = ir.Function(self._module, main_fnty, name="main")        

        # Now implement the function
        self.entry_block = self._main_func.append_basic_block(name="entry")
        self.current_block = self.entry_block
        self._builder = ir.IRBuilder(self.entry_block)
        self._builder.comment(f"// Generated using: PECOS version {__version__}")

    def create_creg(self, creg: CReg):
        """Add a call to create_creg in the current block."""

        
        self._creg_dict[creg.sym] = self._creg_funcs.create_creg_func.call(
            self._builder,
            [ir.Constant(ir.IntType(64), creg.size)],
            f"{creg.sym}",
        )

    def create_qreg(self, qreg: QReg):
        """Uses an OrderedDict to globally flatten quantum registers into a single global register."""
        
        self._qreg_dict[qreg.sym] = (self._qubit_count, self._qubit_count + qreg.size - 1)
        self._qubit_count += qreg.size
        
    def generate_block(self, block: Main) -> None:
        """Primary entry point for generation. Processes an slr Main block."""

        self._handle_main_block(block)
        self._handle_block(block)

        self._builder.ret_void()

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
            case qgate_base.QGate():
                self._handle_quantum_gate(op)

    def _handle_quantum_gate(self, gate: qgate_base.QGate) -> None:
        """Process a quantum gate."""
        
        # Reminder: qgate has sym, qargs, and params properties
        match gate:
            case H():
                self._gates.h_func.call(self._builder, [self._qarg_to_qubit_ptr(gate.qargs[0])], name="")
            case CX():
                    self._gates.cx_func.call(self._builder,
                    [
                        self._qarg_to_qubit_ptr(gate.qargs[0]),
                        self._qarg_to_qubit_ptr(gate.qargs[1]),
                    ],
                    name="",
                )
            case Measure():
                creg = gate.cout[0]
                ll_creg = self._creg_dict[creg.sym]
                for i, q in enumerate(gate.qargs[0]):
                    qubit_ptr = self._qarg_to_qubit_ptr(q)                    
                    self._mz_to_bit.call(self._builder,
                                        [qubit_ptr, ll_creg, ir.Constant(self._types.intType, i)],
                                        name="")    

    def _qarg_to_qubit_ptr(self, qarg: Qubit):
        """Return a pointer to a qubit in the global register, based on the register and index
        passed in the `qarg` param."""
        
        index = qarg.index
        qubit_index = self._qreg_dict[qarg.reg.sym][0] + index
        return ir.Constant(self._types.intType, qubit_index).inttoptr(self._types.qubitPtrType)

    def get_output(self) -> str:
        """Stringify the module as .ll text"""
        
        return str(self._module)
