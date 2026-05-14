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

This emitter lowers SLR's allocator-style AST to Guppy source. Guppy has
linear qubit ownership, so quantum arrays are unpacked to stable local qubit
variables at function entry and the Guppy-only `GuppyLinearityState` tracks
which local owns each logical slot while recursive descent emits statements.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pecos.slr.ast.codegen.guppy_linearity import (
    GuppyLinearityState,
    LinearityError,
    Slot,
    SlotState,
)
from pecos.slr.ast.nodes import (
    AllocatorDecl,
    BinaryExpr,
    BinaryOp,
    BitExpr,
    BitRef,
    BitTypeExpr,
    ForStmt,
    GateKind,
    GateOp,
    IfStmt,
    LiteralExpr,
    MeasureOp,
    ParallelBlock,
    PrepareOp,
    QubitTypeExpr,
    RegisterDecl,
    RepeatStmt,
    ReturnOp,
    UnaryExpr,
    UnaryOp,
    VarExpr,
    WhileStmt,
)

if TYPE_CHECKING:
    from pecos.slr.ast.nodes import (
        AssignOp,
        BarrierOp,
        CommentOp,
        Expression,
        PermuteOp,
        Program,
        SlotRef,
        Statement,
    )


FUNCTIONAL_GATES: dict[GateKind, str] = {
    GateKind.X: "x",
    GateKind.Y: "y",
    GateKind.Z: "z",
    GateKind.H: "h",
    GateKind.S: "s",
    GateKind.Sdg: "sdg",
    GateKind.T: "t",
    GateKind.Tdg: "tdg",
    GateKind.SZ: "s",
    GateKind.SZdg: "sdg",
    GateKind.CX: "cx",
    GateKind.CY: "cy",
    GateKind.CZ: "cz",
    GateKind.CH: "ch",
}

FUNCTIONAL_GATE_IMPORTS = ", ".join(sorted(set(FUNCTIONAL_GATES.values()) | {"reset"}))

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


class GuppyCodegenError(LinearityError):
    """Raised when the v1 AST -> Guppy emitter rejects an unsupported construct."""


@dataclass
class GuppyContext:
    """Mutable state for one Guppy emission run."""

    indent_level: int = 0
    root_allocators: dict[str, int] = field(default_factory=dict)
    child_allocators: set[str] = field(default_factory=set)
    registers: dict[str, RegisterDecl] = field(default_factory=dict)
    linearity: GuppyLinearityState | None = None
    temp_counter: int = 0

    def indent(self) -> str:
        """Return current indentation string."""
        return "    " * self.indent_level

    def push_indent(self) -> None:
        """Increase indentation level."""
        self.indent_level += 1

    def pop_indent(self) -> None:
        """Decrease indentation level."""
        self.indent_level = max(0, self.indent_level - 1)

    def temp(self, prefix: str) -> str:
        """Return a unique temporary local name."""
        name = f"_{prefix}_{self.temp_counter}"
        self.temp_counter += 1
        return name


class AstToGuppy:
    """Recursive-descent Guppy code generator for AST programs."""

    def __init__(self) -> None:
        """Initialize the generator."""
        self.context = GuppyContext()

    def generate(self, program: Program) -> list[str]:
        """Generate Guppy code for a program."""
        self.context = GuppyContext()
        self._collect_declarations(program)
        self.context.linearity = GuppyLinearityState.from_allocators(self.context.root_allocators)
        self._reject_child_allocators()

        body = list(program.body)
        explicit_return = self._validate_return_position(body)
        emitted_body = body[:-1] if explicit_return else body

        lines = self._imports()
        lines.append("")
        lines.append("@guppy")
        lines.append(f"def {program.name.lower()}({self._render_params()}) -> {self._return_type(explicit_return)}:")

        self.context.push_indent()
        body_lines: list[str] = []
        body_lines.extend(self._emit_entry_unpacks())
        body_lines.extend(self._emit_register_initializers())

        for stmt in emitted_body:
            body_lines.extend(self._emit_stmt(stmt))

        if explicit_return is not None:
            body_lines.extend(self._emit_explicit_return(explicit_return))
        else:
            body_lines.extend(self._emit_end_cleanup())
            auto_return = self._auto_return_expr()
            if auto_return is not None:
                body_lines.append(f"{self.context.indent()}return {auto_return}")

        if body_lines:
            lines.extend(body_lines)
        else:
            lines.append(f"{self.context.indent()}pass")

        self.context.pop_indent()
        return lines

    def _collect_declarations(self, program: Program) -> None:
        for decl in program.declarations:
            if isinstance(decl, AllocatorDecl):
                self._add_allocator_decl(decl)
            elif isinstance(decl, RegisterDecl):
                self.context.registers[decl.name] = decl

        if program.allocator is not None:
            self._add_allocator_decl(program.allocator)

    def _add_allocator_decl(self, decl: AllocatorDecl) -> None:
        if decl.parent is not None:
            self.context.child_allocators.add(decl.name)
            return
        self.context.root_allocators.setdefault(decl.name, decl.capacity)

    def _reject_child_allocators(self) -> None:
        if self.context.child_allocators:
            names = ", ".join(sorted(self.context.child_allocators))
            msg = f"AST -> Guppy v1 does not support child allocators: {names}"
            raise GuppyCodegenError(msg)

    def _imports(self) -> list[str]:
        return [
            "from guppylang import guppy",
            "from guppylang.std.builtins import array, owned",
            "from guppylang.std.mem import mem_swap",
            "from guppylang.std.quantum import discard, measure, qubit",
            f"from guppylang.std.quantum.functional import {FUNCTIONAL_GATE_IMPORTS}",
        ]

    def _render_params(self) -> str:
        return ", ".join(f"{name}: array[qubit, {size}] @ owned" for name, size in self.context.root_allocators.items())

    def _return_type(self, explicit_return: ReturnOp | None) -> str:
        if explicit_return is not None:
            types = [self._return_value_type(value) for value in explicit_return.values]
            return self._tuple_type(types)

        types = [f"array[bool, {decl.size}]" for decl in self.context.registers.values() if decl.is_result]
        return self._tuple_type(types)

    def _return_value_type(self, value: Expression | str) -> str:
        if isinstance(value, str):
            if value in self.context.root_allocators:
                return f"array[qubit, {self.context.root_allocators[value]}]"
            if value in self.context.registers:
                return f"array[bool, {self.context.registers[value].size}]"
            msg = f"Unsupported Guppy return value {value!r}"
            raise GuppyCodegenError(msg)

        if isinstance(value, BitExpr):
            return "bool"
        if isinstance(value, LiteralExpr) and isinstance(value.value, bool):
            return "bool"
        if isinstance(value, LiteralExpr) and isinstance(value.value, int):
            return "int"
        msg = f"Unsupported Guppy return expression {value!r}"
        raise GuppyCodegenError(msg)

    def _tuple_type(self, types: list[str]) -> str:
        if not types:
            return "None"
        if len(types) == 1:
            return types[0]
        return f"tuple[{', '.join(types)}]"

    def _emit_entry_unpacks(self) -> list[str]:
        lines: list[str] = []
        linearity = self._linearity()
        for allocator, size in self.context.root_allocators.items():
            if size == 0:
                continue
            locals_for_allocator = [
                binding.local for slot, binding in linearity.bindings() if slot.allocator == allocator
            ]
            lhs = ", ".join(locals_for_allocator)
            if size == 1:
                lhs += ","
            lines.append(f"{self.context.indent()}{lhs} = {allocator}")
        return lines

    def _emit_register_initializers(self) -> list[str]:
        lines: list[str] = []
        for decl in self.context.registers.values():
            values = ", ".join("False" for _ in range(decl.size))
            lines.append(f"{self.context.indent()}{decl.name} = array({values})")
        return lines

    def _validate_return_position(self, body: list[Statement]) -> ReturnOp | None:
        return_count = self._count_returns(body)
        if return_count == 0:
            return None
        if return_count == 1 and body and isinstance(body[-1], ReturnOp):
            return body[-1]
        msg = "AST -> Guppy v1 supports only one final root-level Return"
        raise GuppyCodegenError(msg)

    def _count_returns(self, body: list[Statement] | tuple[Statement, ...]) -> int:
        count = 0
        for stmt in body:
            if isinstance(stmt, ReturnOp):
                count += 1
            elif isinstance(stmt, IfStmt):
                count += self._count_returns(stmt.then_body)
                count += self._count_returns(stmt.else_body)
            elif isinstance(stmt, WhileStmt | ForStmt | RepeatStmt | ParallelBlock):
                count += self._count_returns(stmt.body)
        return count

    def _emit_stmt(self, stmt: Statement) -> list[str]:
        if isinstance(stmt, GateOp):
            return self._emit_gate(stmt)
        if isinstance(stmt, PrepareOp):
            return self._emit_prepare(stmt)
        if isinstance(stmt, MeasureOp):
            return self._emit_measure(stmt)
        if isinstance(stmt, IfStmt):
            return self._emit_if(stmt)
        if isinstance(stmt, RepeatStmt):
            return self._emit_repeat(stmt)
        if isinstance(stmt, ForStmt):
            return self._emit_for(stmt)
        if isinstance(stmt, WhileStmt):
            msg = "AST -> Guppy v1 does not support While loops"
            raise GuppyCodegenError(msg)
        if isinstance(stmt, ParallelBlock):
            return self._emit_parallel(stmt)
        if isinstance(stmt, ReturnOp):
            msg = "AST -> Guppy v1 supports Return only as the final root-level statement"
            raise GuppyCodegenError(msg)

        from pecos.slr.ast.nodes import AssignOp, BarrierOp, CommentOp, PermuteOp  # noqa: PLC0415

        if isinstance(stmt, AssignOp):
            return self._emit_assign(stmt)
        if isinstance(stmt, BarrierOp):
            return self._emit_barrier(stmt)
        if isinstance(stmt, CommentOp):
            return self._emit_comment(stmt)
        if isinstance(stmt, PermuteOp):
            return self._emit_permute(stmt)

        msg = f"Unsupported AST statement for Guppy codegen: {type(stmt).__name__}"
        raise GuppyCodegenError(msg)

    def _emit_gate(self, node: GateOp) -> list[str]:
        gate = FUNCTIONAL_GATES.get(node.gate)
        if gate is None:
            self._raise_unsupported_gate(node.gate)

        if node.params:
            msg = f"AST -> Guppy v1 does not support parameterized gate {node.gate.name}"
            raise GuppyCodegenError(msg)

        slots = [self._slot_from_ref(target) for target in node.targets]
        if len(slots) != len(set(slots)):
            msg = f"Gate {node.gate.name} uses the same qubit slot more than once"
            raise GuppyCodegenError(msg)

        linearity = self._linearity()
        locals_ = [linearity.live(slot) for slot in slots]

        if node.gate.arity == 1:
            local = locals_[0]
            linearity.set_live(slots[0], local)
            return [f"{self.context.indent()}{local} = {gate}({local})"]

        if node.gate.arity == 2:
            left, right = locals_
            linearity.set_live(slots[0], left)
            linearity.set_live(slots[1], right)
            return [f"{self.context.indent()}{left}, {right} = {gate}({left}, {right})"]

        msg = f"AST -> Guppy v1 does not support {node.gate.arity}-qubit gate {node.gate.name}"
        raise GuppyCodegenError(msg)

    def _raise_unsupported_gate(self, gate: GateKind) -> None:
        if gate in {GateKind.SX, GateKind.SXdg, GateKind.SY, GateKind.SYdg}:
            msg = f"AST -> Guppy v1 rejects {gate.name}; decompose it before Guppy emission"
            raise GuppyCodegenError(msg)
        if gate.is_parameterized:
            msg = f"AST -> Guppy v1 does not support parameterized gate {gate.name}"
            raise GuppyCodegenError(msg)
        msg = f"AST -> Guppy v1 does not support gate {gate.name}"
        raise GuppyCodegenError(msg)

    def _emit_prepare(self, node: PrepareOp) -> list[str]:
        lines: list[str] = []
        slots = range(self.context.root_allocators[node.allocator]) if node.slots is None else node.slots
        linearity = self._linearity()
        for index in slots:
            slot = Slot(node.allocator, index)
            local = self._local_name(slot)
            if linearity.status(slot) is SlotState.LIVE:
                old_local = linearity.live(slot)
                lines.append(f"{self.context.indent()}{old_local} = reset({old_local})")
                linearity.set_live(slot, old_local)
            else:
                lines.append(f"{self.context.indent()}{local} = qubit()")
                linearity.set_live(slot, local)
        return lines

    def _emit_measure(self, node: MeasureOp) -> list[str]:
        lines: list[str] = []
        linearity = self._linearity()
        for index, target in enumerate(node.targets):
            slot = self._slot_from_ref(target)
            local = linearity.consume(slot)
            if index < len(node.results):
                result = self._render_bit_ref(node.results[index])
                lines.append(f"{self.context.indent()}{result} = measure({local})")
            else:
                temp = self.context.temp("measurement")
                lines.append(f"{self.context.indent()}{temp} = measure({local})")
        return lines

    def _emit_assign(self, node: AssignOp) -> list[str]:
        target = self._render_bit_ref(node.target) if isinstance(node.target, BitRef) else str(node.target)
        value = self._render_expression(node.value)
        return [f"{self.context.indent()}{target} = {value}"]

    def _emit_barrier(self, _node: BarrierOp) -> list[str]:
        return [f"{self.context.indent()}# barrier"]

    def _emit_comment(self, node: CommentOp) -> list[str]:
        if not node.text:
            return []
        return [f"{self.context.indent()}# {line.strip()}" for line in node.text.splitlines()]

    def _emit_if(self, node: IfStmt) -> list[str]:
        linearity = self._linearity()
        before = linearity.snapshot()

        cond = self._render_expression(node.condition)
        lines = [f"{self.context.indent()}if {cond}:"]

        self.context.push_indent()
        then_lines = self._emit_block(node.then_body)
        lines.extend(then_lines or [f"{self.context.indent()}pass"])
        self.context.pop_indent()
        then_state = linearity.snapshot()

        linearity.restore(before)
        else_state = None
        if node.else_body:
            lines.append(f"{self.context.indent()}else:")
            self.context.push_indent()
            else_lines = self._emit_block(node.else_body)
            lines.extend(else_lines or [f"{self.context.indent()}pass"])
            self.context.pop_indent()
            else_state = linearity.snapshot()

        linearity.merge_if(before, then_state, else_state, label="If")
        return lines

    def _emit_repeat(self, node: RepeatStmt) -> list[str]:
        linearity = self._linearity()
        before = linearity.snapshot()
        lines = [f"{self.context.indent()}for _ in range({node.count}):"]

        self.context.push_indent()
        body_lines = self._emit_block(node.body)
        lines.extend(body_lines or [f"{self.context.indent()}pass"])
        self.context.pop_indent()

        after = linearity.snapshot()
        linearity.assert_same(before, after, label=f"Repeat({node.count})")
        return lines

    def _emit_for(self, node: ForStmt) -> list[str]:
        linearity = self._linearity()
        start = self._render_expression(node.start)
        stop = self._render_expression(node.stop)
        if node.step is not None:
            step = self._render_expression(node.step)
            header = f"for {node.variable} in range({start}, {stop}, {step}):"
        else:
            header = f"for {node.variable} in range({start}, {stop}):"

        before = linearity.snapshot()
        lines = [f"{self.context.indent()}{header}"]
        self.context.push_indent()
        body_lines = self._emit_block(node.body)
        lines.extend(body_lines or [f"{self.context.indent()}pass"])
        self.context.pop_indent()

        after = linearity.snapshot()
        linearity.assert_same(before, after, label=f"For({node.variable})")
        return lines

    def _emit_parallel(self, node: ParallelBlock) -> list[str]:
        return self._emit_block(node.body)

    def _emit_block(self, body: tuple[Statement, ...]) -> list[str]:
        lines: list[str] = []
        for stmt in body:
            lines.extend(self._emit_stmt(stmt))
        return lines

    def _emit_permute(self, node: PermuteOp) -> list[str]:
        if len(node.sources) != len(node.targets):
            msg = "Permute source/target length mismatch"
            raise GuppyCodegenError(msg)

        quantum_mapping: dict[Slot, Slot] = {}
        classical_mapping: dict[BitRef, BitRef] = {}
        for source, target in zip(node.sources, node.targets, strict=True):
            source_refs = self._expand_permute_ref(source)
            target_refs = self._expand_permute_ref(target)
            if len(source_refs) != len(target_refs):
                msg = f"Permute element count mismatch for {source!r} -> {target!r}"
                raise GuppyCodegenError(msg)
            for source_ref, target_ref in zip(source_refs, target_refs, strict=True):
                if isinstance(source_ref, Slot) and isinstance(target_ref, Slot):
                    quantum_mapping[source_ref] = target_ref
                elif isinstance(source_ref, BitRef) and isinstance(target_ref, BitRef):
                    classical_mapping[source_ref] = target_ref
                else:
                    msg = f"Permute cannot map quantum and classical refs together: {source!r} -> {target!r}"
                    raise GuppyCodegenError(msg)

        lines: list[str] = []
        if quantum_mapping:
            self._linearity().permute(quantum_mapping, label="Permute")

        if classical_mapping:
            lines.extend(self._emit_classical_permute(classical_mapping))

        if node.add_comment and (quantum_mapping or classical_mapping):
            pairs = ", ".join(
                f"{source} -> {target}" for source, target in zip(node.sources, node.targets, strict=True)
            )
            lines.insert(0, f"{self.context.indent()}# Permute: {pairs}")
        return lines

    def _expand_permute_ref(self, ref: str) -> list[Slot | BitRef]:
        parsed = self._parse_indexed_ref(ref)
        if parsed is not None:
            name, index = parsed
            if name in self.context.root_allocators:
                return [Slot(name, index)]
            if name in self.context.registers:
                return [BitRef(register=name, index=index)]
            msg = f"Unknown Permute ref {ref!r}"
            raise GuppyCodegenError(msg)

        if ref in self.context.root_allocators:
            return [Slot(ref, index) for index in range(self.context.root_allocators[ref])]
        if ref in self.context.registers:
            return [BitRef(register=ref, index=index) for index in range(self.context.registers[ref].size)]

        msg = f"Unknown Permute ref {ref!r}"
        raise GuppyCodegenError(msg)

    def _emit_classical_permute(self, mapping: dict[BitRef, BitRef]) -> list[str]:
        if set(mapping) != set(mapping.values()):
            msg = "Classical Permute must be bijective over the same bit set"
            raise GuppyCodegenError(msg)

        lines: list[str] = []
        visited: set[BitRef] = set()
        for start, target in mapping.items():
            if start in visited or target == start:
                visited.add(start)
                continue
            cycle = [start]
            visited.add(start)
            current = target
            while current != start:
                if current in visited:
                    msg = "Classical Permute contains a malformed cycle"
                    raise GuppyCodegenError(msg)
                cycle.append(current)
                visited.add(current)
                current = mapping[current]

            lines.extend(
                f"{self.context.indent()}mem_swap({self._render_bit_ref(cycle[index])}, "
                f"{self._render_bit_ref(cycle[index + 1])})"
                for index in range(len(cycle) - 1)
            )
        return lines

    def _emit_end_cleanup(self) -> list[str]:
        return [f"{self.context.indent()}discard({local})" for _slot, local in self._linearity().discard_live()]

    def _auto_return_expr(self) -> str | None:
        values = [decl.name for decl in self.context.registers.values() if decl.is_result]
        if not values:
            return None
        return ", ".join(values)

    def _emit_explicit_return(self, node: ReturnOp) -> list[str]:
        values = [self._return_value_expr(value) for value in node.values]
        lines = self._emit_end_cleanup()
        if values:
            lines.append(f"{self.context.indent()}return {', '.join(values)}")
        else:
            lines.append(f"{self.context.indent()}return")
        return lines

    def _return_value_expr(self, value: Expression | str) -> str:
        if isinstance(value, str):
            if value in self.context.root_allocators:
                return self._consume_allocator_for_return(value)
            if value in self.context.registers:
                return value
            msg = f"Unsupported Guppy return value {value!r}"
            raise GuppyCodegenError(msg)
        return self._render_expression(value)

    def _consume_allocator_for_return(self, allocator: str) -> str:
        linearity = self._linearity()
        locals_ = [
            linearity.consume(Slot(allocator, index)) for index in range(self.context.root_allocators[allocator])
        ]
        return f"array({', '.join(locals_)})"

    def _linearity(self) -> GuppyLinearityState:
        if self.context.linearity is None:
            msg = "Guppy linearity state was not initialized"
            raise GuppyCodegenError(msg)
        return self.context.linearity

    def _slot_from_ref(self, ref: SlotRef) -> Slot:
        if ref.allocator not in self.context.root_allocators:
            msg = f"AST -> Guppy v1 does not support allocator {ref.allocator!r}"
            raise GuppyCodegenError(msg)
        return Slot(ref.allocator, ref.index)

    def _local_name(self, slot: Slot) -> str:
        return f"{slot.allocator}_{slot.index}"

    def _render_bit_ref(self, ref: BitRef) -> str:
        if ref.register not in self.context.registers:
            msg = f"Unknown classical register {ref.register!r}"
            raise GuppyCodegenError(msg)
        return f"{ref.register}[{ref.index}]"

    def _render_expression(self, expr: Expression) -> str:
        if isinstance(expr, LiteralExpr):
            return self._render_literal(expr)
        if isinstance(expr, VarExpr):
            return expr.name
        if isinstance(expr, BitExpr):
            return self._render_bit_ref(expr.ref)
        if isinstance(expr, BinaryExpr):
            return self._render_binary(expr)
        if isinstance(expr, UnaryExpr):
            return self._render_unary(expr)
        msg = f"Unsupported Guppy expression {expr!r}"
        raise GuppyCodegenError(msg)

    def _render_literal(self, expr: LiteralExpr) -> str:
        if isinstance(expr.value, bool):
            return "True" if expr.value else "False"
        return str(expr.value)

    def _render_binary(self, expr: BinaryExpr) -> str:
        left = self._render_expression(expr.left)
        right = self._render_expression(expr.right)
        op = BINARY_OP_TO_PYTHON.get(expr.op)
        if op is None:
            msg = f"Unsupported Guppy binary op {expr.op.name}"
            raise GuppyCodegenError(msg)
        return f"({left} {op} {right})"

    def _render_unary(self, expr: UnaryExpr) -> str:
        operand = self._render_expression(expr.operand)
        if expr.op == UnaryOp.NOT:
            return f"(not {operand})"
        if expr.op == UnaryOp.NEG:
            return f"(-{operand})"
        msg = f"Unsupported Guppy unary op {expr.op.name}"
        raise GuppyCodegenError(msg)

    def _parse_indexed_ref(self, ref: str) -> tuple[str, int] | None:
        match = re.fullmatch(r"([A-Za-z_]\w*)\[(\d+)\]", ref)
        if match is None:
            return None
        return match.group(1), int(match.group(2))

    def visit_qubit_type(self, _node: QubitTypeExpr) -> list[str]:
        """Render a qubit type expression."""
        return ["qubit"]

    def visit_bit_type(self, _node: BitTypeExpr) -> list[str]:
        """Render a bit type expression."""
        return ["bool"]

    def visit_array_type(self, node: object) -> list[str]:
        """Render an array type expression."""
        if isinstance(node.element, QubitTypeExpr):
            elem = "qubit"
        elif isinstance(node.element, BitTypeExpr):
            elem = "bool"
        else:
            elem = "qubit"
        return [f"array[{elem}, {node.size}]"]


def ast_to_guppy(program: Program) -> str:
    """Convert an AST Program to Guppy Python code."""
    generator = AstToGuppy()
    return "\n".join(generator.generate(program))
