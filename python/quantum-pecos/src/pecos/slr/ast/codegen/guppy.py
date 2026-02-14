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

"""AST to Guppy Python code generator.

This module provides a visitor that transforms AST nodes into Guppy Python code.
Guppy is a quantum programming language that compiles to HUGR.

Example:
    from pecos.slr.ast import slr_to_ast, Program
    from pecos.slr.ast.codegen import AstToGuppy

    # Convert SLR to AST
    ast = slr_to_ast(slr_program)

    # Generate Guppy code
    generator = AstToGuppy()
    guppy_code = generator.generate(ast)
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
    from pecos.slr.ast.nodes import AstNode, Expression


# Mapping from AST GateKind to Guppy function names
GATE_TO_GUPPY: dict[GateKind, str] = {
    # Single-qubit Paulis
    GateKind.X: "quantum.x",
    GateKind.Y: "quantum.y",
    GateKind.Z: "quantum.z",
    # Hadamard
    GateKind.H: "quantum.h",
    # Phase gates
    GateKind.S: "quantum.s",
    GateKind.Sdg: "quantum.sdg",
    GateKind.T: "quantum.t",
    GateKind.Tdg: "quantum.tdg",
    # Square root gates
    GateKind.SX: "quantum.sx",
    GateKind.SY: "quantum.sy",
    GateKind.SZ: "quantum.sz",
    GateKind.SXdg: "quantum.sxdg",
    GateKind.SYdg: "quantum.sydg",
    GateKind.SZdg: "quantum.szdg",
    # Rotation gates
    GateKind.RX: "quantum.rx",
    GateKind.RY: "quantum.ry",
    GateKind.RZ: "quantum.rz",
    # Two-qubit gates
    GateKind.CX: "quantum.cx",
    GateKind.CY: "quantum.cy",
    GateKind.CZ: "quantum.cz",
    GateKind.CH: "quantum.ch",
    # Two-qubit rotation gates
    GateKind.SXX: "quantum.sxx",
    GateKind.SYY: "quantum.syy",
    GateKind.SZZ: "quantum.szz",
    GateKind.SXXdg: "quantum.sxxdg",
    GateKind.SYYdg: "quantum.syydg",
    GateKind.SZZdg: "quantum.szzdg",
    GateKind.RZZ: "quantum.rzz",
    # Face rotations
    GateKind.F: "quantum.f",
    GateKind.Fdg: "quantum.fdg",
    GateKind.F4: "quantum.f4",
    GateKind.F4dg: "quantum.f4dg",
}

# Mapping from AST BinaryOp to Python operators
BINARY_OP_TO_PYTHON: dict[BinaryOp, str] = {
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
    BinaryOp.AND: "and",
    BinaryOp.OR: "or",
    BinaryOp.XOR: "^",
    BinaryOp.LSHIFT: "<<",
    BinaryOp.RSHIFT: ">>",
}

# Mapping from AST UnaryOp to Python operators
UNARY_OP_TO_PYTHON: dict[UnaryOp, str] = {
    UnaryOp.NOT: "not",
    UnaryOp.NEG: "-",
}


@dataclass
class CodeGenContext:
    """Context for code generation."""

    indent_level: int = 0
    allocators: dict[str, int] = field(default_factory=dict)  # name -> capacity
    allocator_parents: dict[str, str | None] = field(default_factory=dict)  # name -> parent
    allocator_offsets: dict[str, int] = field(default_factory=dict)  # name -> offset in parent
    registers: dict[str, int] = field(default_factory=dict)  # name -> size
    measured_slots: set[tuple[str, int]] = field(default_factory=set)  # (allocator, index)
    measurement_vars: list[str] = field(default_factory=list)  # variable names for results

    def indent(self) -> str:
        """Return current indentation string."""
        return "    " * self.indent_level

    def push_indent(self) -> None:
        """Increase indentation level."""
        self.indent_level += 1

    def pop_indent(self) -> None:
        """Decrease indentation level."""
        self.indent_level = max(0, self.indent_level - 1)

    def mark_measured(self, allocator: str, index: int) -> None:
        """Mark a qubit slot as consumed by measurement."""
        self.measured_slots.add((allocator, index))

    def is_allocator_fully_consumed(self, name: str) -> bool:
        """Check if all slots of an allocator have been measured."""
        if name not in self.allocators:
            return False
        capacity = self.allocators[name]
        for i in range(capacity):
            if (name, i) not in self.measured_slots:
                return False
        return True

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


class AstToGuppy(BaseVisitor[list[str]]):
    """Visitor that generates Guppy Python code from AST.

    Generates clean Guppy code that can be compiled to HUGR.

    Usage:
        generator = AstToGuppy()
        lines = generator.generate(ast_program)
        code = "\\n".join(lines)
    """

    def __init__(self) -> None:
        """Initialize the generator."""
        self.context = CodeGenContext()

    def generate(self, program: Program) -> list[str]:
        """Generate Guppy code for a program.

        Args:
            program: The AST Program to generate code for.

        Returns:
            List of code lines.
        """
        self.context = CodeGenContext()
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

        # Standard imports
        lines.append("from guppylang import guppy")
        lines.append("from guppylang.std import quantum")
        lines.append("from guppylang.std.quantum import qubit")
        lines.append("")

        # Process declarations to build context
        for decl in node.declarations:
            if isinstance(decl, AllocatorDecl):
                self.context.allocators[decl.name] = decl.capacity
                self.context.allocator_parents[decl.name] = decl.parent
            elif isinstance(decl, RegisterDecl):
                self.context.registers[decl.name] = decl.size

        if node.allocator:
            self.context.allocators[node.allocator.name] = node.allocator.capacity
            self.context.allocator_parents[node.allocator.name] = node.allocator.parent

        # Calculate offsets for child allocators (sequential allocation within parent)
        self._calculate_allocator_offsets(node)

        # First pass: scan body to find measurements (to determine return type)
        self._scan_for_measurements(node.body)

        # Generate function signature
        func_name = node.name.lower()
        params = self._generate_params(node)
        return_type = self._generate_return_type(node)

        lines.append("@guppy")
        lines.append(f"def {func_name}({params}) -> {return_type}:")

        # Generate body
        self.context.push_indent()

        body_lines = []
        for stmt in node.body:
            body_lines.extend(self.visit(stmt))

        # Add return statement
        return_lines = self._generate_return_statement(node)
        if return_lines:
            body_lines.extend(return_lines)

        if body_lines:
            lines.extend(body_lines)
        else:
            lines.append(f"{self.context.indent()}pass")

        self.context.pop_indent()

        return lines

    def _scan_for_measurements(self, stmts: tuple) -> None:
        """Scan statements to find all measurements and mark consumed qubits.

        Also pre-registers measurement variable names for return type generation.
        """
        for stmt in stmts:
            if isinstance(stmt, MeasureOp):
                for i, target in enumerate(stmt.targets):
                    self.context.mark_measured(target.allocator, target.index)
                    # Pre-register measurement variable name
                    if i < len(stmt.results):
                        result = stmt.results[i]
                        var_name = f"{result.register}_{result.index}"
                    else:
                        var_name = f"_m{len(self.context.measurement_vars)}"
                    self.context.measurement_vars.append(var_name)
            elif isinstance(stmt, IfStmt):
                self._scan_for_measurements(stmt.then_body)
                if stmt.else_body:
                    self._scan_for_measurements(stmt.else_body)
            elif isinstance(stmt, (ForStmt, WhileStmt)):
                self._scan_for_measurements(stmt.body)
            elif isinstance(stmt, RepeatStmt):
                self._scan_for_measurements(stmt.body)
            elif isinstance(stmt, ParallelBlock):
                self._scan_for_measurements(stmt.body)

    def _calculate_allocator_offsets(self, node: Program) -> None:
        """Calculate the offset of each child allocator within its parent.

        Children are allocated sequentially within their parent's capacity.
        This allows translating child[i] to parent[offset + i].
        """
        # Track allocated space per parent
        parent_next_offset: dict[str, int] = {}

        # Root allocators have offset 0
        for decl in node.declarations:
            if isinstance(decl, AllocatorDecl):
                if decl.parent is None:
                    self.context.allocator_offsets[decl.name] = 0

        if node.allocator and node.allocator.parent is None:
            self.context.allocator_offsets[node.allocator.name] = 0

        # Process child allocators in declaration order
        for decl in node.declarations:
            if isinstance(decl, AllocatorDecl) and decl.parent is not None:
                parent = decl.parent
                if parent not in parent_next_offset:
                    parent_next_offset[parent] = 0

                # Get parent's offset (for nested hierarchies)
                parent_offset = self.context.allocator_offsets.get(parent, 0)

                # This child's offset is parent's offset + next available slot
                self.context.allocator_offsets[decl.name] = (
                    parent_offset + parent_next_offset[parent]
                )

                # Reserve space in parent
                parent_next_offset[parent] += decl.capacity

    def _generate_params(self, node: Program) -> str:
        """Generate function parameters from declarations.

        Only includes root allocators (those without parents) as function parameters.
        Child allocators are derived from parent allocators within the function.
        """
        params = []

        # Add allocator parameters (only root allocators without parents)
        for decl in node.declarations:
            if isinstance(decl, AllocatorDecl):
                # Skip child allocators - they're derived from parents
                if decl.parent is not None:
                    continue
                params.append(f"{decl.name}: array[qubit, {decl.capacity}] @owned")

        if node.allocator:
            if node.allocator.parent is None:
                params.append(
                    f"{node.allocator.name}: array[qubit, {node.allocator.capacity}] @owned"
                )

        return ", ".join(params)

    def _generate_return_type(self, node: Program) -> str:
        """Generate return type annotation based on consumed/unconsumed qubits."""
        return_types = []

        # Only include qubit arrays that are NOT fully consumed by measurement
        for decl in node.declarations:
            if isinstance(decl, AllocatorDecl):
                # Skip child allocators - only include root allocators in params/returns
                if decl.parent is not None:
                    continue
                if not self.context.is_allocator_fully_consumed(decl.name):
                    return_types.append(f"array[qubit, {decl.capacity}]")

        if node.allocator:
            if not self.context.is_allocator_fully_consumed(node.allocator.name):
                return_types.append(f"array[qubit, {node.allocator.capacity}]")

        # Add measurement results (bools)
        if self.context.measurement_vars:
            for _ in self.context.measurement_vars:
                return_types.append("bool")

        if not return_types:
            return "None"
        if len(return_types) == 1:
            return return_types[0]
        return f"tuple[{', '.join(return_types)}]"

    def _generate_return_statement(self, node: Program) -> list[str]:
        """Generate return statement with unconsumed qubits and measurement results."""
        return_values = []

        # Return unconsumed qubit arrays
        for decl in node.declarations:
            if isinstance(decl, AllocatorDecl):
                # Skip child allocators
                if decl.parent is not None:
                    continue
                if not self.context.is_allocator_fully_consumed(decl.name):
                    return_values.append(decl.name)

        if node.allocator:
            if not self.context.is_allocator_fully_consumed(node.allocator.name):
                return_values.append(node.allocator.name)

        # Return measurement results
        return_values.extend(self.context.measurement_vars)

        if not return_values:
            return []

        return [f"{self.context.indent()}return {', '.join(return_values)}"]

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
        gate_func = GATE_TO_GUPPY.get(node.gate, f"quantum.{node.gate.name.lower()}")

        # Generate target references
        targets = [self._render_slot_ref(t) for t in node.targets]

        # Handle parameterized gates
        if node.params:
            params = [self._render_expression(p) for p in node.params]
            args = ", ".join(params + targets)
        else:
            args = ", ".join(targets)

        # Single qubit gates need reassignment for linearity
        if node.gate.arity == 1:
            target = targets[0]
            return [f"{self.context.indent()}{target} = {gate_func}({target})"]
        else:
            # Two-qubit gates return a tuple
            return [
                f"{self.context.indent()}{targets[0]}, {targets[1]} = {gate_func}({args})"
            ]

    def visit_prepare(self, node: PrepareOp) -> list[str]:
        """Generate prepare/reset operation."""
        lines = []

        if node.slots is None:
            # Prepare all - would need array iteration
            lines.append(
                f"{self.context.indent()}# Prepare all slots in {node.allocator}"
            )
        else:
            for slot in node.slots:
                ref = f"{node.allocator}[{slot}]"
                # In Guppy, qubits start in |0⟩ state from allocation
                # For re-preparation after measurement, we'd use reset
                lines.append(
                    f"{self.context.indent()}{ref} = quantum.reset({ref})"
                )

        return lines

    def visit_measure(self, node: MeasureOp) -> list[str]:
        """Generate measurement operation.

        In Guppy, quantum.measure() consumes the qubit and returns a bool.
        We use local variable names for measurement results.
        Variable names are pre-registered during scan phase for return type generation.
        """
        lines = []

        for i, target in enumerate(node.targets):
            target_ref = self._render_slot_ref(target)

            if i < len(node.results):
                result = node.results[i]
                # Use a proper local variable name instead of array indexing
                var_name = f"{result.register}_{result.index}"
            else:
                # No result specified - use indexed name
                var_name = f"_m{i}"

            lines.append(
                f"{self.context.indent()}{var_name} = quantum.measure({target_ref})"
            )

        return lines

    # === Statements ===

    def visit_assign(self, node: AssignOp) -> list[str]:
        """Generate assignment operation."""
        if isinstance(node.target, BitRef):
            target = f"{node.target.register}[{node.target.index}]"
        else:
            target = str(node.target)

        value = self._render_expression(node.value)
        return [f"{self.context.indent()}{target} = {value}"]

    def visit_barrier(self, node: BarrierOp) -> list[str]:
        """Generate barrier (as comment - no direct Guppy equivalent)."""
        if node.allocators:
            allocs = ", ".join(node.allocators)
            return [f"{self.context.indent()}# barrier({allocs})"]
        return [f"{self.context.indent()}# barrier"]

    def visit_comment(self, node: CommentOp) -> list[str]:
        """Generate comment."""
        if node.text:
            return [f"{self.context.indent()}# {node.text}"]
        return []

    def visit_return(self, node: ReturnOp) -> list[str]:
        """Generate return statement."""
        if not node.values:
            return [f"{self.context.indent()}return"]

        values = []
        for v in node.values:
            if isinstance(v, str):
                values.append(v)
            else:
                values.append(self._render_expression(v))

        return [f"{self.context.indent()}return {', '.join(values)}"]

    def visit_permute(self, node: PermuteOp) -> list[str]:
        """Generate permutation (register swap) code.

        Generates temp variable assignments to swap register references.
        For Permute(a, b), generates:
            # Swap a and b
            _temp_a = a
            a = b
            b = _temp_a
        """
        lines = []

        if len(node.sources) != len(node.targets):
            lines.append(f"{self.context.indent()}# ERROR: Permute sources/targets length mismatch")
            return lines

        if len(node.sources) == 0:
            return lines

        # Add comment if requested
        if node.add_comment:
            names = " and ".join(node.sources)
            lines.append(f"{self.context.indent()}# Swap {names}")

        # For a simple two-way swap: a, b = b, a
        if len(node.sources) == 1 and node.sources[0] != node.targets[0]:
            src = node.sources[0]
            tgt = node.targets[0]
            temp = f"_temp_{src}"
            lines.append(f"{self.context.indent()}{temp} = {src}")
            lines.append(f"{self.context.indent()}{src} = {tgt}")
            lines.append(f"{self.context.indent()}{tgt} = {temp}")
        elif len(node.sources) == 2 and set(node.sources) == set(node.targets):
            # Simple swap: Permute([a, b], [b, a])
            # Can use Python tuple swap
            a, b = node.sources
            lines.append(f"{self.context.indent()}{a}, {b} = {b}, {a}")
        else:
            # General case: use temp variables
            temps = []
            for src in node.sources:
                temp = f"_temp_{src}"
                temps.append(temp)
                lines.append(f"{self.context.indent()}{temp} = {src}")

            for i, src in enumerate(node.sources):
                tgt = node.targets[i]
                lines.append(f"{self.context.indent()}{src} = {tgt}")

            for i, tgt in enumerate(node.targets):
                lines.append(f"{self.context.indent()}{tgt} = {temps[i]}")

        return lines

    # === Control Flow ===

    def visit_if(self, node: IfStmt) -> list[str]:
        """Generate if statement."""
        lines = []

        cond = self._render_expression(node.condition)
        lines.append(f"{self.context.indent()}if {cond}:")

        # Then block
        self.context.push_indent()
        then_lines = []
        for stmt in node.then_body:
            then_lines.extend(self.visit(stmt))

        if then_lines:
            lines.extend(then_lines)
        else:
            lines.append(f"{self.context.indent()}pass")
        self.context.pop_indent()

        # Else block
        if node.else_body:
            lines.append(f"{self.context.indent()}else:")
            self.context.push_indent()
            else_lines = []
            for stmt in node.else_body:
                else_lines.extend(self.visit(stmt))

            if else_lines:
                lines.extend(else_lines)
            else:
                lines.append(f"{self.context.indent()}pass")
            self.context.pop_indent()

        return lines

    def visit_while(self, node: WhileStmt) -> list[str]:
        """Generate while loop."""
        lines = []

        cond = self._render_expression(node.condition)
        lines.append(f"{self.context.indent()}while {cond}:")

        self.context.push_indent()
        body_lines = []
        for stmt in node.body:
            body_lines.extend(self.visit(stmt))

        if body_lines:
            lines.extend(body_lines)
        else:
            lines.append(f"{self.context.indent()}pass")
        self.context.pop_indent()

        return lines

    def visit_for(self, node: ForStmt) -> list[str]:
        """Generate for loop."""
        lines = []

        start = self._render_expression(node.start)
        stop = self._render_expression(node.stop)

        if node.step:
            step = self._render_expression(node.step)
            lines.append(
                f"{self.context.indent()}for {node.variable} in range({start}, {stop}, {step}):"
            )
        else:
            lines.append(
                f"{self.context.indent()}for {node.variable} in range({start}, {stop}):"
            )

        self.context.push_indent()
        body_lines = []
        for stmt in node.body:
            body_lines.extend(self.visit(stmt))

        if body_lines:
            lines.extend(body_lines)
        else:
            lines.append(f"{self.context.indent()}pass")
        self.context.pop_indent()

        return lines

    def visit_repeat(self, node: RepeatStmt) -> list[str]:
        """Generate repeat loop (as for _ in range(n))."""
        lines = []

        lines.append(f"{self.context.indent()}for _ in range({node.count}):")

        self.context.push_indent()
        body_lines = []
        for stmt in node.body:
            body_lines.extend(self.visit(stmt))

        if body_lines:
            lines.extend(body_lines)
        else:
            lines.append(f"{self.context.indent()}pass")
        self.context.pop_indent()

        return lines

    def visit_parallel(self, node: ParallelBlock) -> list[str]:
        """Generate parallel block (as comment + sequential for now)."""
        lines = []
        lines.append(f"{self.context.indent()}# parallel begin")

        for stmt in node.body:
            lines.extend(self.visit(stmt))

        lines.append(f"{self.context.indent()}# parallel end")
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
        return ["bool"]

    def visit_array_type(self, node) -> list[str]:
        elem = self.visit(node.element)[0] if self.visit(node.element) else "qubit"
        return [f"array[{elem}, {node.size}]"]

    def visit_allocator_type(self, node) -> list[str]:
        return [f"array[qubit, {node.capacity}]"]

    # === Helper methods ===

    def _render_slot_ref(self, node: SlotRef) -> str:
        """Render a slot reference as array access.

        For child allocators, translates to root allocator with computed offset.
        E.g., data[0] -> base[0], ancilla[0] -> base[4]
        """
        # Get the root allocator and absolute index
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
        if isinstance(node.value, bool):
            return "True" if node.value else "False"
        return str(node.value)

    def _render_binary(self, node: BinaryExpr) -> str:
        """Render a binary expression."""
        left = self._render_expression(node.left)
        right = self._render_expression(node.right)
        op = BINARY_OP_TO_PYTHON.get(node.op, str(node.op))
        return f"({left} {op} {right})"

    def _render_unary(self, node: UnaryExpr) -> str:
        """Render a unary expression."""
        operand = self._render_expression(node.operand)
        op = UNARY_OP_TO_PYTHON.get(node.op, str(node.op))
        return f"({op} {operand})"


def ast_to_guppy(program: Program) -> str:
    """Convert an AST Program to Guppy Python code.

    Convenience function for simple code generation.

    Args:
        program: The AST Program to convert.

    Returns:
        Generated Guppy Python code as a string.
    """
    generator = AstToGuppy()
    lines = generator.generate(program)
    return "\n".join(lines)
