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

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pecos.slr.ast.nodes import (
    AllocatorDecl,
    BinaryExpr,
    BinaryOp,
    BitExpr,
    BitRef,
    GateKind,
    LiteralExpr,
    RegisterDecl,
    UnaryExpr,
    UnaryOp,
    VarExpr,
)
from pecos.slr.ast.visitor import BaseVisitor

if TYPE_CHECKING:
    from pecos.slr.ast.nodes import (
        AssignOp,
        BarrierOp,
        CommentOp,
        Expression,
        ForStmt,
        GateOp,
        IfStmt,
        MeasureOp,
        ParallelBlock,
        PermuteOp,
        PrepareOp,
        Program,
        RepeatStmt,
        ReturnOp,
        SlotRef,
        WhileStmt,
    )


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
    # Controlled rotation gates
    GateKind.CRX: "crx",
    GateKind.CRY: "cry",
    GateKind.CRZ: "crz",
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
    allocator_parents: dict[str, str | None] = field(
        default_factory=dict,
    )  # name -> parent
    allocator_offsets: dict[str, int] = field(
        default_factory=dict,
    )  # name -> offset in parent
    registers: dict[str, int] = field(default_factory=dict)  # name -> size
    condition: str | None = None  # Current if condition
    # Permutation map: (allocator, index) -> (new_allocator, new_index)
    permutation_map: dict[tuple[str, int], tuple[str, int]] = field(
        default_factory=dict,
    )

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

    def apply_permutation(self, allocator: str, index: int) -> tuple[str, int]:
        """Apply permutation to get the actual (allocator, index) for a qubit reference.

        Performs a single lookup - the permutation_map maps logical to physical.
        """
        key = (allocator, index)
        if key in self.permutation_map:
            return self.permutation_map[key]
        return key


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
        lines: list[str] = []

        # Header
        if self.include_header:
            lines.append("OPENQASM 2.0;")
            lines.extend(f'include "{inc}";' for inc in self.includes)

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
                self.context.allocator_offsets[decl.name] = parent_offset + parent_next_offset[parent]
                parent_next_offset[parent] += decl.capacity

    # === Declarations ===

    def visit_allocator_decl(self, _node: AllocatorDecl) -> list[str]:
        """Allocator declarations are handled at program level."""
        return []

    def visit_register_decl(self, _node: RegisterDecl) -> list[str]:
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
                lines.append(
                    self._maybe_conditional(f"measure {target_ref} -> {result_ref};"),
                )
            else:
                # Measurement without result storage
                lines.append(self._maybe_conditional(f"measure {target_ref};"))

        return lines

    # === Statements ===

    def visit_assign(self, node: AssignOp) -> list[str]:
        """Generate assignment operation."""
        target = f"{node.target.register}[{node.target.index}]" if isinstance(node.target, BitRef) else str(node.target)

        value = self._render_expression(node.value)
        return [self._maybe_conditional(f"{target} = {value};")]

    def visit_barrier(self, node: BarrierOp) -> list[str]:
        """Generate barrier operation."""
        qubits = ", ".join(node.allocators) if node.allocators else ", ".join(self.context.allocators.keys())
        return [f"barrier {qubits};"]

    def visit_comment(self, node: CommentOp) -> list[str]:
        """Generate comment.

        Handles multi-line comments by splitting on newlines and prefixing
        each non-empty line with '//'. Empty lines become just '//'.
        """
        if not node.text:
            return []

        lines = node.text.split("\n")
        result = []
        for line in lines:
            stripped = line.strip()
            if stripped:
                result.append(f"// {stripped}")
            else:
                result.append("//")
        return result

    def visit_return(self, _node: ReturnOp) -> list[str]:
        """Return is not a QASM concept - ignored."""
        return []

    def visit_permute(self, node: PermuteOp) -> list[str]:
        """Handle permutation.

        For classical bits: generates actual swap code with a temp register.
        For qubits: updates internal permutation map for tracking.
        """
        lines: list[str] = []

        def parse_ref(ref: str) -> tuple[str, int] | None:
            """Parse 'name[index]' into (name, index) tuple.

            Returns None for whole register references (no index).
            """
            match = re.match(r"(\w+)\[(\d+)\]", ref)
            if match:
                return (match.group(1), int(match.group(2)))
            return None  # Whole register reference

        if not node.sources:
            return lines

        # Check if this is a whole register permutation (no indices)
        first_parsed = parse_ref(node.sources[0])
        if first_parsed is None:
            # Whole register permutation
            first_name = node.sources[0]
            if first_name in self.context.registers:
                return self._generate_classical_register_permute(node)
            return self._generate_qubit_register_permute(node, parse_ref)

        # Check if this is a classical bit permutation
        is_classical = first_parsed[0] in self.context.registers

        if is_classical:
            return self._generate_classical_bit_permute(node, parse_ref)
        return self._generate_qubit_permute(node, parse_ref)

    def _generate_classical_register_permute(self, node: PermuteOp) -> list[str]:
        """Generate XOR swap for whole classical register permutation."""
        lines = []

        # For now, handle the simple case of swapping two registers
        if len(node.sources) == 2:
            reg_a = node.sources[0]
            reg_b = node.sources[1]

            # XOR swap
            lines.append(f"{reg_a} = {reg_a} ^ {reg_b};")
            lines.append(f"{reg_b} = {reg_b} ^ {reg_a};")
            lines.append(f"{reg_a} = {reg_a} ^ {reg_b};")

            if node.add_comment:
                lines.append(f"// Permutation: {reg_a} <-> {reg_b}")

        return lines

    def _generate_classical_bit_permute(
        self,
        node: PermuteOp,
        _parse_ref: object,
    ) -> list[str]:
        """Generate actual swap operations for classical bit permutation."""
        lines: list[str] = []

        # Build mapping from source to target
        perm_map = dict(zip(node.sources, node.targets, strict=False))

        # Find all cycles in the permutation
        visited: set[str] = set()
        cycles = []

        for start in node.sources:
            if start in visited:
                continue

            cycle = [start]
            visited.add(start)

            next_elem = perm_map[start]
            while next_elem != start:
                cycle.append(next_elem)
                visited.add(next_elem)
                next_elem = perm_map.get(next_elem, start)

            # Skip cycles of length 1 (elements that map to themselves)
            if len(cycle) > 1:
                cycles.append(cycle)

        # If there are cycles, we need a temporary bit
        if cycles:
            lines.append("creg _bit_swap[1];")

            for cycle in cycles:
                # Save first element to temp
                lines.append(f"_bit_swap[0] = {cycle[0]};")

                # Shift elements: each gets value of next in cycle
                lines.extend(f"{cycle[i]} = {cycle[i + 1]};" for i in range(len(cycle) - 1))

                # Last element gets the saved value
                lines.append(f"{cycle[-1]} = _bit_swap[0];")

        # Add comment describing the permutation
        if node.add_comment:
            perm_strs = [f"{s} -> {t}" for s, t in zip(node.sources, node.targets, strict=False)]
            lines.append(f"// Permutation: {', '.join(perm_strs)}")

        return lines

    def _generate_qubit_register_permute(
        self,
        node: PermuteOp,
        _parse_ref: object,
    ) -> list[str]:
        """Update permutation map for whole qubit register permutation.

        Uses composition semantics to preserve existing mappings.
        """
        lines: list[str] = []

        if node.add_comment and node.sources:
            lines.append(f"// Permutation: {' <-> '.join(node.sources)}")

        # For whole register swap, create mappings for all elements
        if len(node.sources) == 2:
            reg_a = node.sources[0]
            reg_b = node.sources[1]
            size_a = self.context.allocators.get(reg_a, 0)
            size_b = self.context.allocators.get(reg_b, 0)

            if size_a == size_b and size_a > 0:
                # Build new permutation
                new_perm: dict[tuple[str, int], tuple[str, int]] = {}
                for i in range(size_a):
                    new_perm[(reg_a, i)] = (reg_b, i)
                    new_perm[(reg_b, i)] = (reg_a, i)

                # Compose with existing permutation map
                # Update existing mappings
                composed: dict[tuple[str, int], tuple[str, int]] = {
                    src: new_perm.get(intermediate, intermediate)
                    for src, intermediate in self.context.permutation_map.items()
                }

                # Add new mappings only if source is not already mapped
                composed.update(
                    {src: dst for src, dst in new_perm.items() if src not in self.context.permutation_map},
                )

                self.context.permutation_map = composed

        return lines

    def _generate_qubit_permute(
        self,
        node: PermuteOp,
        parse_ref: object,
    ) -> list[str]:
        """Update permutation map for qubit permutation (no physical ops).

        Uses composition semantics: existing mappings are updated if their
        destination is being remapped, but new mappings only added if the
        source is not already mapped.
        """
        lines: list[str] = []

        if node.add_comment and node.sources:
            perm_strs = [f"{s} -> {t}" for s, t in zip(node.sources, node.targets, strict=False)]
            lines.append(f"// Permutation: {', '.join(perm_strs)}")

        # Build the new permutation: src -> where tgt currently points
        new_perm: dict[tuple[str, int], tuple[str, int]] = {}
        for src, tgt in zip(node.sources, node.targets, strict=False):
            src_key = parse_ref(src)
            tgt_key = parse_ref(tgt)

            if src_key is None or tgt_key is None:
                continue

            # Where does tgt currently point? That's where src should now point
            tgt_physical = self.context.apply_permutation(*tgt_key)
            new_perm[src_key] = tgt_physical

        # Compose with existing permutation map
        # Update existing mappings: if destination is in new_perm, follow the chain
        composed: dict[tuple[str, int], tuple[str, int]] = {
            src: new_perm.get(intermediate, intermediate) for src, intermediate in self.context.permutation_map.items()
        }

        # Add new mappings only if source is not already in existing map
        composed.update(
            {src: dst for src, dst in new_perm.items() if src not in self.context.permutation_map},
        )

        self.context.permutation_map = composed

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

    def visit_while(self, _node: WhileStmt) -> list[str]:
        """While loops are not supported in QASM 2.0."""
        return ["// ERROR: While loops not supported in QASM 2.0"]

    def visit_for(self, _node: ForStmt) -> list[str]:
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

    def visit_qubit_type(self, _node: object) -> list[str]:
        return ["qubit"]

    def visit_bit_type(self, _node: object) -> list[str]:
        return ["bit"]

    def visit_array_type(self, node) -> list[str]:
        return [f"[{node.size}]"]

    def visit_allocator_type(self, node) -> list[str]:
        return [f"qreg[{node.capacity}]"]

    # === Helper methods ===

    def _render_slot_ref(self, node: SlotRef) -> str:
        """Render a slot reference as array access.

        For child allocators, translates to root allocator with computed offset.
        Applies permutation if one is in effect.
        """
        # First get the root allocator and absolute index
        root = self.context.get_root_allocator(node.allocator)
        abs_index = self.context.get_absolute_index(node.allocator, node.index)

        # Apply permutation if one exists for this qubit
        actual_root, actual_index = self.context.apply_permutation(root, abs_index)

        return f"{actual_root}[{actual_index}]"

    def _render_expression(self, expr: Expression) -> str:
        """Render an expression to a string."""
        if isinstance(expr, LiteralExpr):
            return self._render_literal(expr)
        if isinstance(expr, VarExpr):
            return expr.name
        if isinstance(expr, BitExpr):
            # Apply permutation to bit references (for mixed qubit/classical permutations)
            reg, idx = self.context.apply_permutation(expr.ref.register, expr.ref.index)
            return f"{reg}[{idx}]"
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
