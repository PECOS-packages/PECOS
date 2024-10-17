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

from dataclasses import dataclass
from llvmlite import ir
from llvmlite.ir import IntType, PointerType

@dataclass
class QIRTypes:
    """Dataclass to hold the types used in QIR compilation"""
    boolType: IntType
    intType: IntType
    qubitPtrType: PointerType
    resultPtrType: PointerType

    
class QIRGenerator:
    """Class to generate QIR from SLR. This should enable better compilation of conditional programs."""
    def __init__(self, includes: list[str] | None = None):
        self.output = []
        self.current_scope = None
        self.includes = includes
        self.cond = None

        
    def setup_module(self):
        # Create some useful types to use in compilation later
        boolTy = ir.IntType(1)
        intTy = ir.IntType(64)
        qubitTy = self.mod.context.get_identified_type('Qubit')
        resultTy = self.mod.context.get_identified_type('Result')
        qubitPtrTy = qubitTy.as_pointer()
        resultPtrTy = resultTy.as_pointer()

        self.types = QIRTypes(boolTy, intTy, qubitPtrTy, resultPtrTy)
        fnty = ir.FunctionType(double, (double, double))

        # Create an empty module...
        self.module = ir.Module(name=__file__)        
        # and declare a function named "fpadd" inside it
        self.main_func = ir.Function(module, fnty, name="main")

        # Now implement the function
        self.entry_block = func.append_basic_block(name="entry")
        self.builder = ir.IRBuilder(block)
        a, b = func.args
        result = builder.fadd(a, b, name="")
        builder.ret(result)

        
    def generate_block(self):
        """Process an slr block."""
        block_name = type(block).__name__

        #TODO: see if I can refactor to remove casing on string name
        # of types
        match block_name:
            case "If":
                self.cond = self.generate_op(block.cond)
                self.block_op_loop(block)
                self.cond = None

            case "Repeat":
                for _ in range(block.cond):
                    self.block_op_loop(block)
            case _:
                self.block_op_loop(block)

        self.current_scope = previous_scope
