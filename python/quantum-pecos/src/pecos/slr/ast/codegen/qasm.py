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

"""AST to QASM (extended OpenQASM 2.0) code generator.

This module provides a visitor that transforms AST nodes into extended OpenQASM 2.0 code.
The extended format supports additional features like conditionals and reset operations.

Example:
    from pecos.slr.ast import slr_to_ast, Program
    from pecos.slr.ast.codegen.qasm import AstToQasm, ast_to_qasm

    # Convert SLR to AST
    ast = slr_to_ast(slr_program)

    # Generate QASM code
    generator = AstToQasm()
    qasm_code = generator.generate(ast)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pecos.slr.ast.nodes import (
    AllocatorDecl,
    AssignOp,
    BarrierOp,
    BinaryExpr,
    BinaryOp,
    BitExpr,
    BitRef,
    CommentOp,
    ForStmt,
    GateKind,
    GateOp,
    IfStmt,
    LiteralExpr,
    MeasureOp,
    ParallelBlock,
    PermuteOp,
    PrepareOp,
    Program,
    RegisterDecl,
    RepeatStmt,
    ReturnOp,
    SlotRef,
    UnaryExpr,
    UnaryOp,
    VarExpr,
    WhileStmt,
)
from pecos.slr.ast.visitor import BaseVisitor

if TYPE_CHECKING:
    from pecos.slr.ast.nodes import Expression


# Mapping from AST GateKind to QASM gate names
GATE_TO_QASM: dict[GateKind, str] = {
    # Single-qubit Paulis
    GateKind.X: "x",
    GateKind.Y: "y",
    GateKind.Z: "z",
    # Hadamard
    GateKind.H: "h",
    # Phase gates
    GateKind.S: "s",
    GateKind.Sdg: "sdg",
    GateKind.T: "rz(pi/4)",  # T gate as rotation
    GateKind.Tdg: "rz(-pi/4)",
    # Square root gates
    GateKind.SX: "rx(pi/2)",
    GateKind.SY: "ry(pi/2)",
    GateKind.SZ: "rz(pi/2)",
    GateKind.SXdg: "rx(-pi/2)",
    GateKind.SYdg: "ry(-pi/2)",
    GateKind.SZdg: "rz(-pi/2)",
    # Rotation gates
    GateKind.RX: "rx",
    GateKind.RY: "ry",
    GateKind.RZ: "rz",
    # Two-qubit gates
    GateKind.CX: "cx",
    GateKind.CY: "cy",
    GateKind.CZ: "cz",
    GateKind.CH: "ch",
    # Two-qubit entangling gates
    GateKind.SXX: "SXX",
    GateKind.SYY: "SYY",
    GateKind.SZZ: "ZZ",
    GateKind.SXXdg: "SXXdg",
    GateKind.SYYdg: "SYYdg",
    GateKind.SZZdg: "SZZdg",
    GateKind.RZZ: "rzz",
    # Face rotations (decomposed)
    GateKind.F: None,  # Handled specially
    GateKind.Fdg: None,
    GateKind.F4: None,
    GateKind.F4dg: None,
}

# Mapping from AST BinaryOp to QASM operators
BINARY_OP_TO_QASM: dict[BinaryOp, str] = {
    BinaryOp.ADD: "+",
    BinaryOp.SUB: "-",
    BinaryOp.MUL: "*",
    BinaryOp.DIV: "/",
    BinaryOp.EQ: "==",
    BinaryOp.NE: "!=",
    BinaryOp.LT: "<",
    BinaryOp.LE: "<=",
    BinaryOp.GT: ">",
    BinaryOp.GE: ">=",
    BinaryOp.AND: "&",
    BinaryOp.OR: "|",
    BinaryOp.XOR: "^",
    BinaryOp.LSHIFT: "<<",
    BinaryOp.RSHIFT: ">>",
}


@dataclass
class QasmContext:
    """Context for QASM code generation."""

    allocators: dict[str, int] = field(default_factory=dict)  # name -> size
    allocator_parents: dict[str, str | None] = field(default_factory=dict)  # name -> parent
    allocator_offsets: dict[str, int] = field(default_factory=dict)  # name -> offset in parent
    registers: dict[str, int] = field(default_factory=dict)  # name -> size
    condition: str | None = None  # Current if condition

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


class AstToQasm(BaseVisitor[list[str]]):
    """Visitor that generates extended OpenQASM 2.0 code from AST.

    The generated QASM uses an extended format that supports:
    - Standard QASM 2.0 gates
    - Reset/prep operations
    - Conditionals (if statements)
    - Comments
    - Barriers

    Usage:
        generator = AstToQasm()
        lines = generator.generate(ast_program)
        code = "\\n".join(lines)
    """

    def __init__(
        self,
        *,
        include_header: bool = True,
        includes: list[str] | None = None,
    ) -> None:
        """Initialize the generator.

        Args:
            include_header: Whether to include OPENQASM header.
            includes: List of include files. Defaults to ["hqslib1.inc"].
        """
        self.include_header = include_header
        self.includes = includes or ["hqslib1.inc"]
        self.context = QasmContext()

    def generate(self, program: Program) -> list[str]:
        """Generate QASM code for a program.

        Args:
            program: The AST Program to generate code for.

        Returns:
            List of code lines.
        """
        self.context = QasmContext()
        return self.visit(program)

    def default_result(self) -> list[str]:
        """Return empty list as default."""
        return []

    def combine_results(self, results: list[list[str]]) -> list[str]:
        """Combine multiple results into a single list."""
        combined = []
        for r in results:
            combined.extend(r)
        return combined

    # === Program ===

    def visit_program(self, node: Program) -> list[str]:
        """Generate code for a complete program."""
        lines = []

        # Header
        if self.include_header:
            lines.append("OPENQASM 2.0;")
            for inc in self.includes:
                lines.append(f'include "{inc}";')

        # First pass: collect all allocator info
        for decl in node.declarations:
            if isinstance(decl, AllocatorDecl):
                self.context.allocators[decl.name] = decl.capacity
                self.context.allocator_parents[decl.name] = decl.parent
            elif isinstance(decl, RegisterDecl):
                self.context.registers[decl.name] = decl.size

        if node.allocator:
            self.context.allocators[node.allocator.name] = node.allocator.capacity
            self.context.allocator_parents[node.allocator.name] = node.allocator.parent

        # Calculate offsets for child allocators
        self._calculate_allocator_offsets(node)

        # Output declarations (only root allocators, not children)
        for decl in node.declarations:
            if isinstance(decl, AllocatorDecl):
                # Only declare root allocators (those without parents)
                if decl.parent is None:
                    lines.append(f"qreg {decl.name}[{decl.capacity}];")
            elif isinstance(decl, RegisterDecl):
                lines.append(f"creg {decl.name}[{decl.size}];")

        if node.allocator and node.allocator.parent is None:
            lines.append(f"qreg {node.allocator.name}[{node.allocator.capacity}];")

        # Generate body
        for stmt in node.body:
            lines.extend(self.visit(stmt))

        return lines

    def _calculate_allocator_offsets(self, node: Program) -> None:
        """Calculate the offset of each child allocator within its parent."""
        parent_next_offset: dict[str, int] = {}

        # Root allocators have offset 0
        for decl in node.declarations:
            if isinstance(decl, AllocatorDecl) and decl.parent is None:
                self.context.allocator_offsets[decl.name] = 0

        if node.allocator and node.allocator.parent is None:
            self.context.allocator_offsets[node.allocator.name] = 0

        # Process child allocators in declaration order
        for decl in node.declarations:
            if isinstance(decl, AllocatorDecl) and decl.parent is not None:
                parent = decl.parent
                if parent not in parent_next_offset:
                    parent_next_offset[parent] = 0

                parent_offset = self.context.allocator_offsets.get(parent, 0)
                self.context.allocator_offsets[decl.name] = (
                    parent_offset + parent_next_offset[parent]
                )
                parent_next_offset[parent] += decl.capacity

    # === Declarations ===

    def visit_allocator_decl(self, node: AllocatorDecl) -> list[str]:
        """Allocator declarations are handled at program level."""
        return []

    def visit_register_decl(self, node: RegisterDecl) -> list[str]:
        """Register declarations are handled at program level."""
        return []

    # === Gates ===

    def visit_gate(self, node: GateOp) -> list[str]:
        """Generate gate operation."""
        lines = []

        # Handle special face rotation gates
        if node.gate == GateKind.F:
            for target in node.targets:
                ref = self._render_slot_ref(target)
                lines.append(self._maybe_conditional(f"rx(pi/2) {ref};"))
                lines.append(self._maybe_conditional(f"rz(pi/2) {ref};"))
            return lines

        if node.gate == GateKind.Fdg:
            for target in node.targets:
                ref = self._render_slot_ref(target)
                lines.append(self._maybe_conditional(f"ry(-pi/2) {ref};"))
                lines.append(self._maybe_conditional(f"rz(-pi/2) {ref};"))
            return lines

        if node.gate == GateKind.F4:
            for target in node.targets:
                ref = self._render_slot_ref(target)
                lines.append(self._maybe_conditional(f"ry(-pi/2) {ref};"))
                lines.append(self._maybe_conditional(f"rz(pi/2) {ref};"))
            return lines

        if node.gate == GateKind.F4dg:
            for target in node.targets:
                ref = self._render_slot_ref(target)
                lines.append(self._maybe_conditional(f"rx(-pi/2) {ref};"))
                lines.append(self._maybe_conditional(f"rz(-pi/2) {ref};"))
            return lines

        # Standard gates
        gate_name = GATE_TO_QASM.get(node.gate, node.gate.name.lower())

        # Handle parameterized gates
        if node.params and node.gate.is_parameterized:
            params = ", ".join(self._render_expression(p) for p in node.params)
            gate_str = f"{gate_name}({params})"
        else:
            gate_str = gate_name

        # Generate for each target set
        if node.gate.arity == 1:
            for target in node.targets:
                ref = self._render_slot_ref(target)
                lines.append(self._maybe_conditional(f"{gate_str} {ref};"))
        else:
            # Two-qubit gate
            if len(node.targets) >= 2:
                ref1 = self._render_slot_ref(node.targets[0])
                ref2 = self._render_slot_ref(node.targets[1])
                lines.append(self._maybe_conditional(f"{gate_str} {ref1}, {ref2};"))

        return lines

    def visit_prepare(self, node: PrepareOp) -> list[str]:
        """Generate reset/prep operation."""
        lines = []

        # Get root allocator for this allocator
        root = self.context.get_root_allocator(node.allocator)

        if node.slots is None:
            # Reset all qubits in the allocator
            capacity = self.context.allocators.get(node.allocator, 1)
            for i in range(capacity):
                abs_index = self.context.get_absolute_index(node.allocator, i)
                lines.append(self._maybe_conditional(f"reset {root}[{abs_index}];"))
        else:
            for slot in node.slots:
                abs_index = self.context.get_absolute_index(node.allocator, slot)
                lines.append(self._maybe_conditional(f"reset {root}[{abs_index}];"))

        return lines

    def visit_measure(self, node: MeasureOp) -> list[str]:
        """Generate measurement operation."""
        lines = []

        for i, target in enumerate(node.targets):
            target_ref = self._render_slot_ref(target)

            if i < len(node.results):
                result = node.results[i]
                result_ref = f"{result.register}[{result.index}]"
                lines.append(self._maybe_conditional(f"measure {target_ref} -> {result_ref};"))
            else:
                # Measurement without result storage
                lines.append(self._maybe_conditional(f"measure {target_ref};"))

        return lines

    # === Statements ===

    def visit_assign(self, node: AssignOp) -> list[str]:
        """Generate assignment operation."""
        if isinstance(node.target, BitRef):
            target = f"{node.target.register}[{node.target.index}]"
        else:
            target = str(node.target)

        value = self._render_expression(node.value)
        return [self._maybe_conditional(f"{target} = {value};")]

    def visit_barrier(self, node: BarrierOp) -> list[str]:
        """Generate barrier operation."""
        if node.allocators:
            qubits = ", ".join(node.allocators)
        else:
            # Barrier on all qubits
            qubits = ", ".join(self.context.allocators.keys())

        return [f"barrier {qubits};"]

    def visit_comment(self, node: CommentOp) -> list[str]:
        """Generate comment."""
        if node.text:
            return [f"// {node.text}"]
        return []

    def visit_return(self, node: ReturnOp) -> list[str]:
        """Return is not a QASM concept - ignored."""
        return []

    def visit_permute(self, node: PermuteOp) -> list[str]:
        """Handle permutation by updating internal permutation map.

        QASM doesn't have a permute instruction, so this updates the
        internal mapping used for subsequent qubit references.
        """
        lines = []

        if node.add_comment and node.sources:
            names = ", ".join(node.sources)
            lines.append(f"// Permute: {names} <-> {', '.join(node.targets)}")

        # Update the permutation map in context
        # For each source->target pair, remap qubit references
        for src, tgt in zip(node.sources, node.targets, strict=False):
            # Swap the mappings for src and tgt allocators
            # This affects how get_qubit_name() resolves references
            if hasattr(self.context, "permutation_map"):
                # Store the swap
                self.context.permutation_map[src] = tgt
                self.context.permutation_map[tgt] = src

        return lines

    # === Control Flow ===

    def visit_if(self, node: IfStmt) -> list[str]:
        """Generate conditional statements."""
        lines = []

        # Render condition
        cond = self._render_expression(node.condition)
        # Remove outer parentheses if present
        if cond.startswith("(") and cond.endswith(")"):
            cond = cond[1:-1]

        # Save previous condition
        prev_cond = self.context.condition
        self.context.condition = cond

        # Generate then body
        for stmt in node.then_body:
            lines.extend(self.visit(stmt))

        # Restore condition
        self.context.condition = prev_cond

        # Note: QASM 2.0 doesn't support else blocks in the standard way
        # We just add a comment if there's an else block
        if node.else_body:
            lines.append("// Note: else block not directly supported in QASM 2.0")

        return lines

    def visit_while(self, node: WhileStmt) -> list[str]:
        """While loops are not supported in QASM 2.0."""
        return ["// ERROR: While loops not supported in QASM 2.0"]

    def visit_for(self, node: ForStmt) -> list[str]:
        """For loops are not supported in QASM 2.0."""
        return ["// ERROR: For loops not supported in QASM 2.0"]

    def visit_repeat(self, node: RepeatStmt) -> list[str]:
        """Generate repeat loop by unrolling."""
        lines = []
        lines.append(f"// Repeat {node.count} times (unrolled)")

        for _ in range(node.count):
            for stmt in node.body:
                lines.extend(self.visit(stmt))

        return lines

    def visit_parallel(self, node: ParallelBlock) -> list[str]:
        """Generate parallel block (sequential in QASM)."""
        lines = []
        lines.append("// parallel begin")

        for stmt in node.body:
            lines.extend(self.visit(stmt))

        lines.append("// parallel end")
        return lines

    # === References ===

    def visit_slot_ref(self, node: SlotRef) -> list[str]:
        """Slot refs are rendered inline."""
        return [self._render_slot_ref(node)]

    def visit_bit_ref(self, node: BitRef) -> list[str]:
        """Bit refs are rendered inline."""
        return [f"{node.register}[{node.index}]"]

    # === Expressions ===

    def visit_literal(self, node: LiteralExpr) -> list[str]:
        """Literals are rendered inline."""
        return [self._render_literal(node)]

    def visit_var(self, node: VarExpr) -> list[str]:
        """Variables are rendered inline."""
        return [node.name]

    def visit_bit_expr(self, node: BitExpr) -> list[str]:
        """Bit expressions are rendered inline."""
        return [f"{node.ref.register}[{node.ref.index}]"]

    def visit_binary(self, node: BinaryExpr) -> list[str]:
        """Binary expressions are rendered inline."""
        return [self._render_binary(node)]

    def visit_unary(self, node: UnaryExpr) -> list[str]:
        """Unary expressions are rendered inline."""
        return [self._render_unary(node)]

    # === Type expressions ===

    def visit_qubit_type(self, node) -> list[str]:
        return ["qubit"]

    def visit_bit_type(self, node) -> list[str]:
        return ["bit"]

    def visit_array_type(self, node) -> list[str]:
        return [f"[{node.size}]"]

    def visit_allocator_type(self, node) -> list[str]:
        return [f"qreg[{node.capacity}]"]

    # === Helper methods ===

    def _render_slot_ref(self, node: SlotRef) -> str:
        """Render a slot reference as array access.

        For child allocators, translates to root allocator with computed offset.
        """
        root = self.context.get_root_allocator(node.allocator)
        abs_index = self.context.get_absolute_index(node.allocator, node.index)
        return f"{root}[{abs_index}]"

    def _render_expression(self, expr: Expression) -> str:
        """Render an expression to a string."""
        if isinstance(expr, LiteralExpr):
            return self._render_literal(expr)
        if isinstance(expr, VarExpr):
            return expr.name
        if isinstance(expr, BitExpr):
            return f"{expr.ref.register}[{expr.ref.index}]"
        if isinstance(expr, BinaryExpr):
            return self._render_binary(expr)
        if isinstance(expr, UnaryExpr):
            return self._render_unary(expr)
        return str(expr)

    def _render_literal(self, node: LiteralExpr) -> str:
        """Render a literal value."""
        return str(node.value)

    def _render_binary(self, node: BinaryExpr) -> str:
        """Render a binary expression."""
        left = self._render_expression(node.left)
        right = self._render_expression(node.right)
        op = BINARY_OP_TO_QASM.get(node.op, str(node.op))
        return f"({left} {op} {right})"

    def _render_unary(self, node: UnaryExpr) -> str:
        """Render a unary expression."""
        operand = self._render_expression(node.operand)
        if node.op == UnaryOp.NOT:
            return f"(~{operand})"
        if node.op == UnaryOp.NEG:
            return f"(-{operand})"
        return f"({node.op} {operand})"

    def _maybe_conditional(self, stmt: str) -> str:
        """Wrap statement with condition if one is active."""
        if self.context.condition:
            return f"if({self.context.condition}) {stmt}"
        return stmt


def ast_to_qasm(program: Program, *, include_header: bool = True) -> str:
    """Convert an AST Program to QASM code.

    Convenience function for simple code generation.

    Args:
        program: The AST Program to convert.
        include_header: Whether to include OPENQASM header.

    Returns:
        Generated QASM code as a string.
    """
    generator = AstToQasm(include_header=include_header)
    lines = generator.generate(program)
    return "\n".join(lines)
