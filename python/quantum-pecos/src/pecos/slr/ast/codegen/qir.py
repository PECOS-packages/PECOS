# Copyright 2026 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""AST to QIR (Quantum Intermediate Representation) code generator.

This module transforms AST nodes into QIR using LLVM IR generation.
QIR is an LLVM-based intermediate representation for quantum programs.

Example:
    from pecos.slr.ast import slr_to_ast
    from pecos.slr.ast.codegen import AstToQir

    ast = slr_to_ast(slr_program)
    generator = AstToQir()
    llvm_ir = generator.generate(ast)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pecos as pc
from pecos.slr.ast.nodes import (
    AllocatorDecl,
    AssignOp,
    BarrierOp,
    BinaryExpr,
    BinaryOp,
    BitExpr,
    BitRef,
    ForStmt,
    GateKind,
    GateOp,
    IfStmt,
    LiteralExpr,
    MeasureOp,
    ParallelBlock,
    PermuteOp,
    PrepareOp,
    RegisterDecl,
    RepeatStmt,
    SlotRef,
    UnaryExpr,
    UnaryOp,
    VarExpr,
    WhileStmt,
)

if TYPE_CHECKING:
    from pecos.slr.ast.nodes import (
        Expression,
        Program,
        Statement,
    )

# Optional LLVM dependency - imported at module level for efficiency
try:
    from pecos_rslib_llvm import ir as llvm_ir

    LLVM_AVAILABLE = True
except ImportError:
    llvm_ir = None  # type: ignore[assignment]
    LLVM_AVAILABLE = False

# Mapping from AST GateKind to QIR gate names
GATE_TO_QIR: dict[GateKind, str] = {
    # Single-qubit Paulis
    GateKind.X: "x",
    GateKind.Y: "y",
    GateKind.Z: "z",
    # Hadamard
    GateKind.H: "h",
    # Phase gates
    GateKind.S: "s",
    GateKind.Sdg: "s__adj",
    GateKind.T: "t",
    GateKind.Tdg: "t__adj",
    # Square root gates - mapped to S variants
    GateKind.SZ: "s",
    GateKind.SZdg: "s__adj",
    # Rotation gates
    GateKind.RX: "rx",
    GateKind.RY: "ry",
    GateKind.RZ: "rz",
    # Two-qubit gates
    GateKind.CX: "cnot",
    GateKind.CZ: "cz",
    GateKind.RZZ: "rzz",
    GateKind.SZZ: "zz",
}

# Gates with rotation parameters
PARAMETERIZED_GATES = {GateKind.RX, GateKind.RY, GateKind.RZ, GateKind.RZZ}

# Two-qubit gates
TWO_QUBIT_GATES = {GateKind.CX, GateKind.CZ, GateKind.RZZ, GateKind.SZZ}


@dataclass
class QirCodeGenContext:
    """Context for QIR code generation."""

    qubit_map: dict[tuple[str, int], int] = field(default_factory=dict)
    qubit_count: int = 0
    creg_map: dict[str, int] = field(default_factory=dict)  # name -> size
    measurement_count: int = 0
    allocator_parents: dict[str, str | None] = field(default_factory=dict)
    allocator_offsets: dict[str, int] = field(default_factory=dict)

    def get_root_allocator(self, name: str) -> str:
        """Get the root allocator for a given allocator name."""
        current = name
        while self.allocator_parents.get(current) is not None:
            current = self.allocator_parents[current]
        return current

    def get_absolute_index(self, allocator: str, index: int) -> int:
        """Get the absolute index in the root allocator."""
        offset = self.allocator_offsets.get(allocator, 0)
        return offset + index

    def get_qubit_index(self, allocator: str, index: int) -> int:
        """Get the global qubit index for an allocator slot.

        For child allocators, translates to root allocator with computed offset.
        """
        # Translate to root allocator and absolute index
        root = self.get_root_allocator(allocator)
        abs_index = self.get_absolute_index(allocator, index)

        key = (root, abs_index)
        if key not in self.qubit_map:
            self.qubit_map[key] = self.qubit_count
            self.qubit_count += 1
        return self.qubit_map[key]


class AstToQir:
    """Transforms AST programs into QIR using recursive descent.

    Generates LLVM IR suitable for QIR-compatible execution environments.

    Usage:
        generator = AstToQir()
        llvm_ir = generator.generate(ast_program)
    """

    def __init__(self) -> None:
        """Initialize the generator."""
        self.context = QirCodeGenContext()
        self._module = None
        self._builder = None
        self._types = None
        self._main_func = None
        self._gate_cache: dict[str, Any] = {}
        self._creg_ptrs: dict[str, Any] = {}
        self._creg_funcs = None

    def generate(self, program: Program) -> str:
        """Generate QIR (LLVM IR) for a program.

        Args:
            program: The AST Program to generate code for.

        Returns:
            QIR as an LLVM IR string.

        Raises:
            ImportError: If LLVM dependencies are not available.
        """
        if not LLVM_AVAILABLE:
            msg = "LLVM dependencies not available. Install with 'pip install pecos[qir]'"
            raise ImportError(msg)

        self.context = QirCodeGenContext()
        self._gate_cache = {}
        self._creg_ptrs = {}

        # Setup LLVM module
        self._module = llvm_ir.Module(name="ast_qir_module")

        # Setup types
        qubit_ty = self._module.context.get_identified_type("Qubit")
        result_ty = self._module.context.get_identified_type("Result")

        self._types = {
            "void": llvm_ir.VoidType(),
            "bool": llvm_ir.IntType(1),
            "int": llvm_ir.IntType(64),
            "double": llvm_ir.DoubleType(),
            "qubit_ptr": qubit_ty.as_pointer(),
            "result_ptr": result_ty.as_pointer(),
            "tag": llvm_ir.IntType(8).as_pointer(),
        }

        # Setup creg helper functions
        self._setup_creg_funcs()

        # Setup measurement function
        self._mz_to_bit = self._declare_function(
            "mz_to_creg_bit",
            self._types["void"],
            [
                self._types["qubit_ptr"],
                self._types["bool"].as_pointer(),
                self._types["int"],
            ],
        )

        # Setup main function
        main_fnty = llvm_ir.FunctionType(self._types["void"], [])
        self._main_func = llvm_ir.Function(self._module, main_fnty, name="main")
        entry_block = self._main_func.append_basic_block(name="entry")
        self._builder = llvm_ir.IRBuilder(entry_block)
        self._builder.comment(
            f"// Generated from AST using: PECOS version {pc.__version__}",
        )

        # Setup operator map
        self._setup_op_map()

        # Process declarations
        self._process_declarations(program)

        # Process body statements
        for stmt in program.body:
            self._process_statement(stmt)

        # Generate results output
        self._generate_results()

        # Return void
        self._builder.ret_void()

        # Return the LLVM IR with attributes
        return self._finalize_module()

    def _setup_creg_funcs(self) -> None:
        """Setup classical register helper functions."""

        self._creg_funcs = {
            "create_creg": self._declare_function(
                "create_creg",
                self._types["bool"].as_pointer(),
                [self._types["int"]],
            ),
            "creg_to_int": self._declare_function(
                "get_int_from_creg",
                self._types["int"],
                [self._types["bool"].as_pointer()],
            ),
            "get_creg_bit": self._declare_function(
                "get_creg_bit",
                self._types["bool"],
                [self._types["bool"].as_pointer(), self._types["int"]],
            ),
            "set_creg_bit": self._declare_function(
                "set_creg_bit",
                self._types["void"],
                [
                    self._types["bool"].as_pointer(),
                    self._types["int"],
                    self._types["bool"],
                ],
            ),
            "set_creg": self._declare_function(
                "set_creg_to_int",
                self._types["void"],
                [self._types["bool"].as_pointer(), self._types["int"]],
            ),
            "int_result": self._declare_function(
                "__quantum__rt__int_record_output",
                self._types["void"],
                [self._types["int"], self._types["tag"]],
            ),
        }

    def _declare_function(self, name: str, ret_ty: Any, arg_tys: list) -> Any:
        """Declare an LLVM function."""
        fnty = llvm_ir.FunctionType(ret_ty, arg_tys)
        return llvm_ir.Function(self._module, fnty, name=name)

    def _setup_op_map(self) -> None:
        """Setup binary operator mapping."""
        self._op_map = {
            BinaryOp.EQ: lambda lhs, rhs: self._builder.icmp_signed("==", lhs, rhs),
            BinaryOp.NE: lambda lhs, rhs: self._builder.icmp_signed("!=", lhs, rhs),
            BinaryOp.LT: lambda lhs, rhs: self._builder.icmp_signed("<", lhs, rhs),
            BinaryOp.GT: lambda lhs, rhs: self._builder.icmp_signed(">", lhs, rhs),
            BinaryOp.LE: lambda lhs, rhs: self._builder.icmp_signed("<=", lhs, rhs),
            BinaryOp.GE: lambda lhs, rhs: self._builder.icmp_signed(">=", lhs, rhs),
            BinaryOp.MUL: self._builder.mul,
            BinaryOp.DIV: self._builder.udiv,
            BinaryOp.XOR: self._builder.xor,
            BinaryOp.AND: self._builder.and_,
            BinaryOp.OR: self._builder.or_,
            BinaryOp.ADD: self._builder.add,
            BinaryOp.SUB: self._builder.sub,
            BinaryOp.RSHIFT: self._builder.lshr,
            BinaryOp.LSHIFT: self._builder.shl,
        }

    def _process_declarations(self, program: Program) -> None:
        """Process declarations to allocate qubits and classical registers."""
        # First pass: collect allocator parent info
        for decl in program.declarations:
            if isinstance(decl, AllocatorDecl):
                self.context.allocator_parents[decl.name] = decl.parent

        if program.allocator:
            self.context.allocator_parents[program.allocator.name] = program.allocator.parent

        # Calculate offsets for child allocators
        self._calculate_allocator_offsets(program)

        # Process allocator declarations - only allocate for root allocators
        for decl in program.declarations:
            if isinstance(decl, AllocatorDecl):
                if decl.parent is None:
                    for i in range(decl.capacity):
                        self.context.get_qubit_index(decl.name, i)
            elif isinstance(decl, RegisterDecl):
                self.context.creg_map[decl.name] = decl.size
                # Create classical register
                if decl.size < 64:
                    self._creg_ptrs[decl.name] = self._builder.call(
                        self._creg_funcs["create_creg"],
                        [llvm_ir.Constant(self._types["int"], decl.size)],
                        name=decl.name,
                    )

        if program.allocator and program.allocator.parent is None:
            for i in range(program.allocator.capacity):
                self.context.get_qubit_index(program.allocator.name, i)

    def _calculate_allocator_offsets(self, program: Program) -> None:
        """Calculate the offset of each child allocator within its parent."""
        parent_next_offset: dict[str, int] = {}

        # Root allocators have offset 0
        for decl in program.declarations:
            if isinstance(decl, AllocatorDecl) and decl.parent is None:
                self.context.allocator_offsets[decl.name] = 0

        if program.allocator and program.allocator.parent is None:
            self.context.allocator_offsets[program.allocator.name] = 0

        # Process child allocators in declaration order
        for decl in program.declarations:
            if isinstance(decl, AllocatorDecl) and decl.parent is not None:
                parent = decl.parent
                if parent not in parent_next_offset:
                    parent_next_offset[parent] = 0

                parent_offset = self.context.allocator_offsets.get(parent, 0)
                self.context.allocator_offsets[decl.name] = parent_offset + parent_next_offset[parent]
                parent_next_offset[parent] += decl.capacity

    def _process_statement(self, stmt: Statement) -> None:
        """Process a statement using recursive descent."""
        if isinstance(stmt, GateOp):
            self._process_gate(stmt)
        elif isinstance(stmt, MeasureOp):
            self._process_measure(stmt)
        elif isinstance(stmt, PrepareOp):
            self._process_prepare(stmt)
        elif isinstance(stmt, BarrierOp):
            self._process_barrier(stmt)
        elif isinstance(stmt, AssignOp):
            self._process_assign(stmt)
        elif isinstance(stmt, IfStmt):
            self._process_if(stmt)
        elif isinstance(stmt, WhileStmt):
            self._process_while(stmt)
        elif isinstance(stmt, ForStmt):
            self._process_for(stmt)
        elif isinstance(stmt, RepeatStmt):
            self._process_repeat(stmt)
        elif isinstance(stmt, ParallelBlock):
            self._process_parallel(stmt)
        elif isinstance(stmt, PermuteOp):
            self._process_permute(stmt)

    def _process_gate(self, node: GateOp) -> None:
        """Process a gate operation."""
        qir_name = GATE_TO_QIR.get(node.gate)
        if qir_name is None:
            # Skip unsupported gates
            return

        if node.gate in TWO_QUBIT_GATES:
            self._process_two_qubit_gate(node, qir_name)
        else:
            self._process_single_qubit_gate(node, qir_name)

    def _process_single_qubit_gate(self, node: GateOp, qir_name: str) -> None:
        """Process a single-qubit gate."""
        # Get or create gate function
        gate_func = self._get_or_create_gate(
            qir_name,
            has_params=node.gate in PARAMETERIZED_GATES,
            num_qubits=1,
        )

        for target in node.targets:
            qubit_ptr = self._get_qubit_ptr(target)

            args = []
            if node.gate in PARAMETERIZED_GATES and node.params:
                args.extend(llvm_ir.Constant(self._types["double"], float(p)) for p in node.params)
            args.append(qubit_ptr)

            self._builder.call(gate_func, args, name="")

    def _process_two_qubit_gate(self, node: GateOp, qir_name: str) -> None:
        """Process a two-qubit gate."""
        gate_func = self._get_or_create_gate(
            qir_name,
            has_params=node.gate in PARAMETERIZED_GATES,
            num_qubits=2,
        )

        if len(node.targets) >= 2:
            q0_ptr = self._get_qubit_ptr(node.targets[0])
            q1_ptr = self._get_qubit_ptr(node.targets[1])

            args = []
            if node.gate in PARAMETERIZED_GATES and node.params:
                args.extend(llvm_ir.Constant(self._types["double"], float(p)) for p in node.params)
            args.extend([q0_ptr, q1_ptr])

            self._builder.call(gate_func, args, name="")

    def _get_or_create_gate(
        self,
        qir_name: str,
        *,
        has_params: bool,
        num_qubits: int,
    ) -> Any:
        """Get or create a QIR gate function declaration."""
        cache_key = f"{qir_name}_{has_params}_{num_qubits}"
        if cache_key in self._gate_cache:
            return self._gate_cache[cache_key]

        # Build argument types
        arg_tys = []
        if has_params:
            arg_tys.append(self._types["double"])
        arg_tys.extend([self._types["qubit_ptr"]] * num_qubits)

        # Build mangled name
        suffix = "__body" if "adj" not in qir_name else ""
        mangled_name = f"__quantum__qis__{qir_name}{suffix}"

        fnty = llvm_ir.FunctionType(self._types["void"], arg_tys)
        gate_func = llvm_ir.Function(self._module, fnty, name=mangled_name)

        self._gate_cache[cache_key] = gate_func
        return gate_func

    def _get_qubit_ptr(self, target: SlotRef) -> Any:
        """Get a qubit pointer for a target."""
        qubit_index = self.context.get_qubit_index(target.allocator, target.index)
        return llvm_ir.Constant(self._types["int"], qubit_index).inttoptr(
            self._types["qubit_ptr"],
        )

    def _process_measure(self, node: MeasureOp) -> None:
        """Process a measurement operation."""
        for i, target in enumerate(node.targets):
            self.context.measurement_count += 1
            qubit_ptr = self._get_qubit_ptr(target)

            if i < len(node.results):
                result = node.results[i]
                if result.register in self._creg_ptrs:
                    creg_ptr = self._creg_ptrs[result.register]
                    bit_index = llvm_ir.Constant(self._types["int"], result.index)
                    self._builder.call(
                        self._mz_to_bit,
                        [qubit_ptr, creg_ptr, bit_index],
                        name="",
                    )

    def _process_prepare(self, node: PrepareOp) -> None:
        """Process a prepare/reset operation."""
        if node.slots is None:
            return

        reset_func = self._get_or_create_gate("reset", has_params=False, num_qubits=1)

        for slot in node.slots:
            qubit_ptr = self._get_qubit_ptr(
                SlotRef(allocator=node.allocator, index=slot),
            )
            self._builder.call(reset_func, [qubit_ptr], name="")

    def _process_barrier(self, node: BarrierOp) -> None:
        """Process a barrier operation."""
        # Collect all qubits involved
        qubits = []
        if node.allocators:
            for alloc in node.allocators:
                qubits.extend(
                    self._get_qubit_ptr(SlotRef(allocator=key[0], index=key[1]))
                    for key in self.context.qubit_map
                    if key[0] == alloc
                )

        if not qubits:
            return

        # Create barrier function if needed
        barrier_name = f"__quantum__qis__barrier{len(qubits)}__body"
        fnty = llvm_ir.FunctionType(
            self._types["void"],
            [self._types["qubit_ptr"]] * len(qubits),
        )
        barrier_func = llvm_ir.Function(self._module, fnty, name=barrier_name)
        self._builder.call(barrier_func, qubits, name="")

    def _process_assign(self, node: AssignOp) -> None:
        """Process an assignment operation."""
        if isinstance(node.target, BitRef):
            reg_name = node.target.register
            if reg_name not in self._creg_ptrs:
                return

            creg_ptr = self._creg_ptrs[reg_name]
            bit_index = llvm_ir.Constant(self._types["int"], node.target.index)

            # Evaluate RHS
            rhs = self._eval_expression(node.value)

            self._builder.call(
                self._creg_funcs["set_creg_bit"],
                [creg_ptr, bit_index, rhs],
                name="",
            )

    def _eval_expression(self, expr: Expression) -> Any:
        """Evaluate an expression to an LLVM value."""
        if isinstance(expr, LiteralExpr):
            if isinstance(expr.value, bool):
                return llvm_ir.Constant(self._types["bool"], 1 if expr.value else 0)
            if isinstance(expr.value, int):
                return llvm_ir.Constant(self._types["int"], expr.value)
            if isinstance(expr.value, float):
                return llvm_ir.Constant(self._types["double"], expr.value)
            return llvm_ir.Constant(self._types["int"], expr.value)

        if isinstance(expr, BitExpr):
            reg_name = expr.ref.register
            if reg_name not in self._creg_ptrs:
                return llvm_ir.Constant(self._types["bool"], 0)
            creg_ptr = self._creg_ptrs[reg_name]
            bit_index = llvm_ir.Constant(self._types["int"], expr.ref.index)
            return self._builder.call(
                self._creg_funcs["get_creg_bit"],
                [creg_ptr, bit_index],
                name="",
            )

        if isinstance(expr, VarExpr):
            # Variable lookup - for now just return 0
            return llvm_ir.Constant(self._types["int"], 0)

        if isinstance(expr, BinaryExpr):
            left = self._eval_expression(expr.left)
            right = self._eval_expression(expr.right)
            if expr.op in self._op_map:
                return self._op_map[expr.op](left, right)
            return left

        if isinstance(expr, UnaryExpr):
            operand = self._eval_expression(expr.operand)
            if expr.op == UnaryOp.NEG:
                return self._builder.neg(operand)
            if expr.op == UnaryOp.NOT:
                return self._builder.not_(operand)
            return operand

        return llvm_ir.Constant(self._types["int"], 0)

    def _process_if(self, node: IfStmt) -> None:
        """Process an if statement."""
        pred = self._eval_expression(node.condition)

        if node.else_body:
            with self._builder.if_else(pred) as (then, otherwise):
                with then:
                    for stmt in node.then_body:
                        self._process_statement(stmt)
                with otherwise:
                    for stmt in node.else_body:
                        self._process_statement(stmt)
        else:
            with self._builder.if_then(pred):
                for stmt in node.then_body:
                    self._process_statement(stmt)

    def _process_while(self, node: WhileStmt) -> None:
        """Process a while loop."""
        # QIR supports loops through LLVM branch instructions
        # For simplicity, we process the body once (approximation)
        for stmt in node.body:
            self._process_statement(stmt)

    def _process_for(self, node: ForStmt) -> None:
        """Process a for loop by unrolling."""
        if isinstance(node.start, int) and isinstance(node.stop, int):
            step = node.step if isinstance(node.step, int) else 1
            for _ in range(node.start, node.stop, step):
                for stmt in node.body:
                    self._process_statement(stmt)

    def _process_repeat(self, node: RepeatStmt) -> None:
        """Process a repeat loop by unrolling."""
        if isinstance(node.count, int):
            for _ in range(node.count):
                for stmt in node.body:
                    self._process_statement(stmt)

    def _process_parallel(self, node: ParallelBlock) -> None:
        """Process a parallel block."""
        for stmt in node.body:
            self._process_statement(stmt)

    def _process_permute(self, node: PermuteOp) -> None:
        """Process a permutation operation.

        Updates the internal allocator mapping to swap qubit references.
        QIR doesn't have a permute instruction, so this just updates
        how we map allocator names to qubit indices.
        """
        # Swap the allocator offsets
        for src, tgt in zip(node.sources, node.targets, strict=False):
            # Get current offsets
            src_offset = self.context.allocator_offsets.get(src, 0)
            tgt_offset = self.context.allocator_offsets.get(tgt, 0)
            # Swap them
            self.context.allocator_offsets[src] = tgt_offset
            self.context.allocator_offsets[tgt] = src_offset

    def _generate_results(self) -> None:
        """Generate result output calls."""
        for reg_name, creg_ptr in self._creg_ptrs.items():
            # Create tag for the register name
            reg_name_bytes = bytearray(reg_name.encode("utf-8"))
            tag_type = llvm_ir.ArrayType(llvm_ir.IntType(8), len(reg_name))
            reg_tag = llvm_ir.GlobalVariable(self._module, tag_type, reg_name)
            reg_tag.initializer = llvm_ir.Constant(tag_type, reg_name_bytes)
            reg_tag.global_constant = True
            reg_tag.linkage = "private"

            # Convert creg to int and output
            c_int = self._builder.call(
                self._creg_funcs["creg_to_int"],
                [creg_ptr],
                name="",
            )
            reg_tag_gep = reg_tag.gep(
                (
                    llvm_ir.Constant(llvm_ir.IntType(32), 0),
                    llvm_ir.Constant(llvm_ir.IntType(32), 0),
                ),
            )
            self._builder.call(
                self._creg_funcs["int_result"],
                [c_int, reg_tag_gep],
                name="",
            )

    def _finalize_module(self) -> str:
        """Finalize the module and return LLVM IR with attributes."""
        ll_text = self._fix_internal_consts(str(self._module))
        mod_w_attr = ll_text.replace("@main()", "@main() #0")

        mod_w_attr += '\nattributes #0 = { "entry_point"'
        mod_w_attr += ' "qir_profiles"="custom"'
        mod_w_attr += f' "required_num_qubits"="{self.context.qubit_count}"'
        mod_w_attr += f' "required_num_results"="{self.context.measurement_count}" }}'
        return mod_w_attr

    def _fix_internal_consts(self, llvm_ir: str) -> str:
        """Fix internal constants in LLVM IR."""
        return re.sub('([@%])"([^"]+)"', r"\1\2", llvm_ir)


def ast_to_qir(program: Program) -> str:
    """Convert an AST Program to QIR (LLVM IR).

    Convenience function for simple code generation.

    Args:
        program: The AST Program to convert.

    Returns:
        QIR as an LLVM IR string.
    """
    generator = AstToQir()
    return generator.generate(program)
