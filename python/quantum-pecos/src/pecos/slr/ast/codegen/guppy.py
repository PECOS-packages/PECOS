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
from typing import TYPE_CHECKING, cast

from pecos.slr.ast.codegen._block_flatten import validate_unique_block_decl_names
from pecos.slr.ast.codegen.guppy_linearity import (
    GuppyLinearityState,
    LinearityError,
    Slot,
    SlotState,
)
from pecos.slr.ast.nodes import (
    AllocatorArg,
    AllocatorDecl,
    ArrayTypeExpr,
    BinaryExpr,
    BinaryOp,
    BitExpr,
    BitRef,
    BitTypeExpr,
    BlockCall,
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
    ResourceEffect,
    ReturnOp,
    UnaryExpr,
    UnaryOp,
    VarExpr,
    WhileStmt,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pecos.slr.ast.nodes import (
        AssignOp,
        BarrierOp,
        BlockDecl,
        BlockInput,
        CommentOp,
        Expression,
        PermuteOp,
        PrintOp,
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

BOOL_OPERAND_BINARY_OPS = {BinaryOp.AND, BinaryOp.OR, BinaryOp.XOR}
BOOL_COMPARISON_OPS = {BinaryOp.EQ, BinaryOp.NE}


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
        self._block_decls: dict[str, BlockDecl] = {}

    def generate(self, program: Program) -> list[str]:
        """Generate Guppy code for a program."""
        validate_unique_block_decl_names(program)
        self.context = GuppyContext()
        self._block_decls = {decl.name: decl for decl in program.block_decls}
        for decl in program.block_decls:
            self._validate_block_decl(decl)

        lines = self._imports()

        for decl in program.block_decls:
            lines.append("")
            lines.extend(self._generate_block_decl(decl))

        lines.append("")
        lines.extend(self._generate_main(program))
        return lines

    def _generate_main(self, program: Program) -> list[str]:
        """Emit the main Guppy function for the program body."""
        self.context = GuppyContext()
        self._collect_declarations(program)
        self._collect_implicit_measure_registers(program.body)
        self.context.linearity = GuppyLinearityState.from_allocators(self.context.root_allocators)
        self._reject_child_allocators()

        # Phase 1 Print AST-level validators (run before emission so failures
        # point at the source program, not the generated Guppy).
        self._validate_print_paths(program.body)
        self._validate_print_inline_creg_assignment(program)

        body = list(program.body)
        explicit_return = self._validate_return_position(body)
        emitted_body = body[:-1] if explicit_return else body

        lines: list[str] = ["@guppy"]
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

    def _validate_block_decl(self, decl: BlockDecl) -> None:
        """Reject BlockDecl shapes that the v1 Guppy emitter cannot lower."""
        for inp in decl.inputs:
            if not isinstance(inp.type_expr, ArrayTypeExpr) or not isinstance(inp.type_expr.element, QubitTypeExpr):
                msg = (
                    f"BlockDecl {decl.name!r} input {inp.name!r}: only array[qubit, N] inputs "
                    f"are supported in Phase 3a.1 (got {type(inp.type_expr).__name__})"
                )
                raise GuppyCodegenError(msg)
            if inp.effect not in {ResourceEffect.LIVE_PRESERVED, ResourceEffect.CONSUMED}:
                msg = (
                    f"BlockDecl {decl.name!r} input {inp.name!r}: only LIVE_PRESERVED and "
                    f"CONSUMED effects are supported in Phase 3a.1 (got {inp.effect.name})"
                )
                raise GuppyCodegenError(msg)
        if decl.return_op is not None:
            msg = (
                f"BlockDecl {decl.name!r}: explicit Return inside BlockDecl is not yet "
                "supported in Phase 3a.1; live_preserved inputs are returned implicitly"
            )
            raise GuppyCodegenError(msg)

    def _generate_block_decl(self, decl: BlockDecl) -> list[str]:
        """Emit a BlockDecl as a top-level Guppy @guppy def function."""
        saved_context = self.context
        self.context = GuppyContext()
        # _validate_block_decl guarantees every type_expr is ArrayTypeExpr[QubitTypeExpr].
        input_arrays: list[tuple[str, int]] = [
            (inp.name, cast("ArrayTypeExpr", inp.type_expr).size) for inp in decl.inputs
        ]
        for name, size in input_arrays:
            self.context.root_allocators[name] = size
        self.context.linearity = GuppyLinearityState.from_allocators(self.context.root_allocators)

        live_inputs = tuple(inp for inp in decl.inputs if inp.effect is ResourceEffect.LIVE_PRESERVED)
        live_sizes = [cast("ArrayTypeExpr", inp.type_expr).size for inp in live_inputs]
        return_types = [f"array[qubit, {size}]" for size in live_sizes]
        return_type_str = self._tuple_type(return_types)

        params = ", ".join(f"{name}: array[qubit, {size}] @ owned" for name, size in input_arrays)

        lines: list[str] = ["@guppy", f"def {decl.name}({params}) -> {return_type_str}:"]

        self.context.push_indent()
        body_lines: list[str] = []
        body_lines.extend(self._emit_entry_unpacks())

        for stmt in decl.body:
            body_lines.extend(self._emit_stmt(stmt))

        # Auto-emit return for live_preserved inputs; consume their slots so the
        # linearity tracker reflects ownership leaving the function.
        if live_inputs:
            arrays = [self._consume_allocator_for_return(inp.name) for inp in live_inputs]
            if len(arrays) == 1:
                body_lines.append(f"{self.context.indent()}return {arrays[0]}")
            else:
                body_lines.append(f"{self.context.indent()}return {', '.join(arrays)}")
        # consumed inputs must be discarded inside the body (caller's responsibility
        # to ensure the body does so; the smoke test covers this).
        else:
            body_lines.extend(self._emit_end_cleanup())

        if body_lines:
            lines.extend(body_lines)
        else:
            lines.append(f"{self.context.indent()}pass")

        self.context.pop_indent()
        self.context = saved_context
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

    def _collect_implicit_measure_registers(self, body: tuple[Statement, ...]) -> None:
        """Add result registers introduced only as measurement outputs."""
        max_indices: dict[str, int] = {}
        for stmt in body:
            self._collect_implicit_measure_register_refs(stmt, max_indices)

        for register, max_index in max_indices.items():
            if register not in self.context.registers:
                self.context.registers[register] = RegisterDecl(name=register, size=max_index + 1, is_result=True)

    def _collect_implicit_measure_register_refs(self, stmt: Statement, max_indices: dict[str, int]) -> None:
        if isinstance(stmt, MeasureOp):
            for ref in stmt.results:
                if ref.register not in self.context.registers:
                    max_indices[ref.register] = max(max_indices.get(ref.register, -1), ref.index)
            return

        if isinstance(stmt, IfStmt):
            self._collect_implicit_measure_registers_in_body(stmt.then_body, max_indices)
            self._collect_implicit_measure_registers_in_body(stmt.else_body, max_indices)
            return

        if isinstance(stmt, RepeatStmt | ForStmt | WhileStmt | ParallelBlock):
            self._collect_implicit_measure_registers_in_body(stmt.body, max_indices)

    def _collect_implicit_measure_registers_in_body(
        self,
        body: tuple[Statement, ...],
        max_indices: dict[str, int],
    ) -> None:
        for stmt in body:
            self._collect_implicit_measure_register_refs(stmt, max_indices)

    def _reject_child_allocators(self) -> None:
        if self.context.child_allocators:
            names = ", ".join(sorted(self.context.child_allocators))
            msg = f"AST -> Guppy v1 does not support child allocators: {names}"
            raise GuppyCodegenError(msg)

    def _imports(self) -> list[str]:
        return [
            "from guppylang import guppy",
            "from guppylang.std.builtins import array, owned, result",
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

    def _validate_print_paths(self, body: tuple[Statement, ...]) -> None:
        """Validate Print path-signature consistency across If/Elif branches.

        Walks the body once, descending into nested control flow. For each
        If, both `then_body` and `else_body` (recursively) must emit the
        same ordered sequence of Print events. `Repeat(n)` and static-bound
        `For(name, start, stop[, step])` multiply inner signatures by the
        static trip count. Non-static `For` and `While` reject Prints in
        Phase 1 since the trip count is not statically known.

        Side effect: raises `GuppyCodegenError` if any validation fails.
        """
        from pecos.slr.ast.nodes import PrintOp  # noqa: PLC0415

        self._collect_print_path_signature(body, PrintOp)

    def _collect_print_path_signature(
        self,
        body: tuple[Statement, ...],
        print_op_cls: type,
    ) -> tuple[tuple[str, str, str, int], ...]:
        """Return the ordered Print signature for `body`, validating as we go.

        Each Print emission contributes one signature tuple:
        `(namespace, tag, value_kind, value_shape)` where `value_kind` is
        `"creg"` (whole register) or `"bit"` (single bit) and
        `value_shape` is the register size (or 1 for bit refs).

        Side effects:
        - Raises `GuppyCodegenError` if an `If` body has asymmetric Print
          signatures across `then_body` / `else_body`.
        - Raises `GuppyCodegenError` if a non-static `For` or any `While`
          contains a Print in its body.
        """
        signature: list[tuple[str, str, str, int]] = []
        for stmt in body:
            if isinstance(stmt, print_op_cls):
                signature.append(self._print_op_event(stmt))
            elif isinstance(stmt, IfStmt):
                then_sig = self._collect_print_path_signature(stmt.then_body, print_op_cls)
                else_sig = self._collect_print_path_signature(stmt.else_body, print_op_cls)
                if then_sig != else_sig:
                    msg = (
                        "Print path-signature mismatch across If branches:\n"
                        f"  Then: {then_sig}\n"
                        f"  Else: {else_sig}\n"
                        "Phase 1 requires symmetric Print emission across all branches of an "
                        "If/Elif chain. Either add the missing Print(s) to the lighter branch, "
                        "or move the Print outside the If."
                    )
                    raise GuppyCodegenError(msg)
                signature.extend(then_sig)
            elif isinstance(stmt, RepeatStmt):
                inner = self._collect_print_path_signature(stmt.body, print_op_cls)
                signature.extend(inner * stmt.count)
            elif isinstance(stmt, ForStmt):
                inner = self._collect_print_path_signature(stmt.body, print_op_cls)
                if inner:
                    trip = self._static_for_trip_count(stmt)
                    if trip is None:
                        msg = (
                            "Print inside non-static `For` is not supported in Phase 1. "
                            "Use `Repeat(n)` or `For(name, start, stop)` with literal int "
                            "start/stop/step, or move the Print outside the For body."
                        )
                        raise GuppyCodegenError(msg)
                    signature.extend(inner * trip)
            elif isinstance(stmt, WhileStmt):
                inner = self._collect_print_path_signature(stmt.body, print_op_cls)
                if inner:
                    msg = (
                        "Print inside `While` is not supported in Phase 1 (no static trip "
                        "count). Move the Print outside the While body."
                    )
                    raise GuppyCodegenError(msg)
            elif isinstance(stmt, ParallelBlock):
                signature.extend(self._collect_print_path_signature(stmt.body, print_op_cls))
        return tuple(signature)

    def _print_op_event(self, op) -> tuple[str, str, str, int]:
        if isinstance(op.value, BitRef):
            return (op.namespace, op.tag, "bit", 1)
        # str = whole CReg name
        decl = self.context.registers.get(op.value)
        shape = decl.size if decl is not None else 0
        return (op.namespace, op.tag, "creg", shape)

    def _static_for_trip_count(self, stmt: ForStmt) -> int | None:
        """Compute static trip count for a `For(name, start, stop[, step])`.

        Returns the integer trip count when start/stop/step are all integer
        literals; returns None otherwise (Phase 1 then rejects Print in the
        loop body via `_collect_print_path_signature`).
        """
        start = self._static_int(stmt.start)
        stop = self._static_int(stmt.stop)
        if start is None or stop is None:
            return None
        step = 1
        if stmt.step is not None:
            step_val = self._static_int(stmt.step)
            if step_val is None:
                return None
            step = step_val
        if step == 0:
            return None
        return len(range(start, stop, step))

    def _static_int(self, expr) -> int | None:
        if isinstance(expr, LiteralExpr) and isinstance(expr.value, int) and not isinstance(expr.value, bool):
            return expr.value
        return None

    def _validate_print_inline_creg_assignment(self, program: Program) -> None:
        """Reject Print of an inline CReg bit before Measure has written to it.

        Inline CRegs are those introduced only by `Measure(q) > CReg(...)` --
        they appear in `context.registers` via `_collect_implicit_measure_registers`
        but not in `program.declarations`. Without explicit user declaration in
        `Main(...)`, the runtime sees an auto-initialized all-False register if a
        Print runs before any Measure has populated it. That silent zero-emission
        is the bug; this validator rejects it.

        Declared CRegs (those in `program.declarations`) are NOT validated --
        users who explicitly declare a CReg are acknowledging the zero-init.

        Granularity: **bit-level**. The validator tracks `(register, bit_index)`
        pairs. `Print(c[i])` requires the specific `(c, i)` to be assigned;
        whole-CReg `Print(c)` requires every bit `0..size-1` (inferred size) to
        be assigned. Bit-level tracking also acts as a bounds check: a `Print`
        referencing an index past the inferred size is rejected because that
        `(reg, index)` cannot have been added by any Measure.
        """
        declared = {d.name for d in program.declarations if isinstance(d, RegisterDecl)}
        inline_cregs = {name for name in self.context.registers if name not in declared}
        if not inline_cregs:
            return
        assigned: set[tuple[str, int]] = set()
        self._check_print_inline_assignment(program.body, assigned, inline_cregs)

    def _check_print_inline_assignment(
        self,
        body: tuple[Statement, ...],
        assigned: set[tuple[str, int]],
        inline_cregs: set[str],
    ) -> None:
        """Walk body left-to-right; mutate `assigned` in-place across the path.

        At control-flow joins, merge per-path assignment sets:
        - `If(...)`: definite-after = intersection of `then`/`else` post-states.
        - `Repeat(n>=1)` / static `For(count>=1)`: body runs at least once, so
          inner assignments propagate.
        - `Repeat(0)` / static `For(count<=0)` / non-static `For` / `While`:
          body may not run. Walk for validation (catch unreachable invalid
          Prints), but do NOT propagate inner assignments to the outer scope.
        - `Parallel`: treated as sequential (matches the emitter's flatten
          behavior).
        """
        from pecos.slr.ast.nodes import PrintOp  # noqa: PLC0415

        for stmt in body:
            if isinstance(stmt, MeasureOp):
                for ref in stmt.results:
                    if ref.register in inline_cregs:
                        assigned.add((ref.register, ref.index))
            elif isinstance(stmt, PrintOp):
                self._check_print_inline_read(stmt, assigned, inline_cregs)
            elif isinstance(stmt, IfStmt):
                then_assigned = set(assigned)
                self._check_print_inline_assignment(stmt.then_body, then_assigned, inline_cregs)
                else_assigned = set(assigned)
                self._check_print_inline_assignment(stmt.else_body, else_assigned, inline_cregs)
                assigned.update(then_assigned & else_assigned)
            elif isinstance(stmt, RepeatStmt):
                if stmt.count >= 1:
                    inner_assigned = set(assigned)
                    self._check_print_inline_assignment(stmt.body, inner_assigned, inline_cregs)
                    assigned.update(inner_assigned)
                else:
                    # count == 0: body doesn't run. Walk for validation; do not propagate.
                    self._check_print_inline_assignment(stmt.body, set(assigned), inline_cregs)
            elif isinstance(stmt, ForStmt):
                trip = self._static_for_trip_count(stmt)
                if trip is not None and trip >= 1:
                    inner_assigned = set(assigned)
                    self._check_print_inline_assignment(stmt.body, inner_assigned, inline_cregs)
                    assigned.update(inner_assigned)
                else:
                    self._check_print_inline_assignment(stmt.body, set(assigned), inline_cregs)
            elif isinstance(stmt, WhileStmt):
                self._check_print_inline_assignment(stmt.body, set(assigned), inline_cregs)
            elif isinstance(stmt, ParallelBlock):
                self._check_print_inline_assignment(stmt.body, assigned, inline_cregs)

    def _check_print_inline_read(self, op, assigned: set[tuple[str, int]], inline_cregs: set[str]) -> None:
        if isinstance(op.value, BitRef):
            reg = op.value.register
            if reg not in inline_cregs:
                return
            if (reg, op.value.index) not in assigned:
                msg = (
                    f"Print references inline CReg bit {reg}[{op.value.index}] before any "
                    f"Measure has written to it. Print would emit the auto-initialized False "
                    f"value (or read past the inferred register bound), not a measurement "
                    f"result. Move the Print after a Measure(...) > {reg}[{op.value.index}] "
                    f"that runs on every path, or declare {reg!r} explicitly as a positional "
                    f"in Main(...) if you intend to print the zero-initialized state."
                )
                raise GuppyCodegenError(msg)
        elif isinstance(op.value, str):
            reg = op.value
            if reg not in inline_cregs:
                return
            # Reject whole-CReg Print of an inline CReg outright in Phase 1.
            # The user-stated `CReg(name, size)` size is lost during inline-from-
            # Measure inference (only Measure-targeted bit indices contribute to
            # the inferred RegisterDecl.size). Emitting `result(tag, c)` for the
            # inferred c can silently shrink the register relative to the user's
            # intent. Require either an explicit Main(...) declaration (then the
            # CReg is no longer inline and whole-CReg Print is allowed) or per-bit
            # `Print(c[i], ...)` calls.
            msg = (
                f"Print(whole-CReg) of inline CReg {reg!r} is rejected in Phase 1. "
                "Whole-register Print can silently shrink an inline CReg because the "
                "original `CReg(name, size)` size is lost during inline-from-Measure "
                f"inference. Declare {reg!r} as a positional in `Main(...)` (then whole-"
                f"CReg Print is allowed) or print individual bits via `Print({reg}[i], ...)`."
            )
            raise GuppyCodegenError(msg)

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
        if isinstance(stmt, BlockCall):
            return self._emit_block_call(stmt)
        if isinstance(stmt, ReturnOp):
            msg = "AST -> Guppy v1 supports Return only as the final root-level statement"
            raise GuppyCodegenError(msg)

        from pecos.slr.ast.nodes import AssignOp, BarrierOp, CommentOp, PermuteOp, PrintOp  # noqa: PLC0415

        if isinstance(stmt, AssignOp):
            return self._emit_assign(stmt)
        if isinstance(stmt, BarrierOp):
            return self._emit_barrier(stmt)
        if isinstance(stmt, CommentOp):
            return self._emit_comment(stmt)
        if isinstance(stmt, PermuteOp):
            return self._emit_permute(stmt)
        if isinstance(stmt, PrintOp):
            return self._emit_print(stmt)

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
        is_bit_target = isinstance(node.target, BitRef)
        target = self._render_bit_ref(node.target) if is_bit_target else str(node.target)
        value = self._render_expression(node.value, bool_context=is_bit_target)
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

        cond = self._render_expression(node.condition, bool_context=True)
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

    def _emit_block_call(self, node: BlockCall) -> list[str]:
        """Lower a BlockCall to a packed array(...) call + unpack pattern.

        For each input the caller's outer-allocator locals are packed into an
        array literal. The call's return tuple unpacks back into locals for
        every `LIVE_PRESERVED` input. `CONSUMED` inputs leave their outer
        slots in the CONSUMED state.
        """
        decl = self._block_decls.get(node.callee)
        if decl is None:
            msg = f"BlockCall references undefined block {node.callee!r}"
            raise GuppyCodegenError(msg)

        if len(node.arg_bindings) != len(decl.inputs):
            msg = (
                f"BlockCall {node.callee!r}: {len(node.arg_bindings)} arg_bindings "
                f"but BlockDecl declares {len(decl.inputs)} inputs"
            )
            raise GuppyCodegenError(msg)

        # Phase 1: validate every arg + out binding BEFORE touching linearity state, so
        # a late-raised GuppyCodegenError can't leave the tracker half-consumed (Codex
        # 2026-05-15 review).
        # Phase 3a.3 iter 5a: arg_bindings are typed BlockArg now. This emitter
        # currently only supports AllocatorArg (whole-allocator binding); richer
        # arg shapes (SingleQubitArg/SingleBitArg/QubitBundleArg/BitBundleArg) land
        # with their respective Phase 3a.3 iterations.
        validated_args: list[tuple[BlockInput, str, int]] = []
        live_inputs_out: list[BlockInput] = []
        for inp, arg in zip(decl.inputs, node.arg_bindings, strict=True):
            if not isinstance(arg, AllocatorArg):
                msg = (
                    f"BlockCall {node.callee!r} arg for input {inp.name!r}: only "
                    f"AllocatorArg is supported in Phase 3a.3 iter 5a (got "
                    f"{type(arg).__name__})"
                )
                raise GuppyCodegenError(msg)
            arg_name = arg.name
            input_size = cast("ArrayTypeExpr", inp.type_expr).size
            if arg_name not in self.context.root_allocators:
                msg = (
                    f"BlockCall {node.callee!r} arg {arg_name!r} must be an outer root "
                    "allocator name"
                )
                raise GuppyCodegenError(msg)
            outer_size = self.context.root_allocators[arg_name]
            if outer_size != input_size:
                msg = (
                    f"BlockCall {node.callee!r} arg {arg_name!r} size {outer_size} "
                    f"does not match input {inp.name!r} size {input_size}"
                )
                raise GuppyCodegenError(msg)
            validated_args.append((inp, arg_name, outer_size))
            if inp.effect is ResourceEffect.LIVE_PRESERVED:
                live_inputs_out.append(inp)

        if len(node.out_bindings) != len(live_inputs_out):
            expected = [inp.name for inp in live_inputs_out]
            msg = (
                f"BlockCall {node.callee!r}: out_bindings count "
                f"({len(node.out_bindings)}) does not match expected return positions {expected}"
            )
            raise GuppyCodegenError(msg)

        validated_outs: list[str] = []
        for out, inp in zip(node.out_bindings, live_inputs_out, strict=True):
            if not isinstance(out, AllocatorArg):
                msg = (
                    f"BlockCall {node.callee!r} out_binding for input {inp.name!r}: "
                    f"only AllocatorArg is supported in Phase 3a.3 iter 5a (got "
                    f"{type(out).__name__})"
                )
                raise GuppyCodegenError(msg)
            self._require_out_binding_matches(node.callee, out.name, inp)
            validated_outs.append(out.name)

        # Phase 2: now that every check passed, consume slots and emit code.
        linearity = self._linearity()
        arg_exprs: list[str] = []
        for _inp, arg_name, outer_size in validated_args:
            locals_ = [linearity.consume(Slot(arg_name, i)) for i in range(outer_size)]
            arg_exprs.append(f"array({', '.join(locals_)})")

        call_expr = f"{node.callee}({', '.join(arg_exprs)})"
        lines: list[str] = []

        if not live_inputs_out:
            lines.append(f"{self.context.indent()}{call_expr}")
            return lines

        if len(live_inputs_out) == 1:
            out_name = validated_outs[0]
            ret_temp = self.context.temp("call_ret")
            lines.append(f"{self.context.indent()}{ret_temp} = {call_expr}")
            lines.extend(self._unpack_return_array(out_name, ret_temp))
            return lines

        ret_temps = [self.context.temp("call_ret") for _ in live_inputs_out]
        lines.append(f"{self.context.indent()}{', '.join(ret_temps)} = {call_expr}")
        for ret_temp, out_name in zip(ret_temps, validated_outs, strict=True):
            lines.extend(self._unpack_return_array(out_name, ret_temp))
        return lines

    def _require_out_binding_matches(self, callee: str, out_name: str, inp: BlockInput) -> None:
        # _validate_block_decl guarantees type_expr is ArrayTypeExpr[QubitTypeExpr].
        input_size = cast("ArrayTypeExpr", inp.type_expr).size
        if out_name not in self.context.root_allocators:
            msg = f"BlockCall {callee!r} out_binding {out_name!r} is not an outer allocator"
            raise GuppyCodegenError(msg)
        outer_size = self.context.root_allocators[out_name]
        if outer_size != input_size:
            msg = (
                f"BlockCall {callee!r} out_binding {out_name!r} size {outer_size} "
                f"does not match input {inp.name!r} size {input_size}"
            )
            raise GuppyCodegenError(msg)

    def _unpack_return_array(self, out_name: str, ret_temp: str) -> list[str]:
        size = self.context.root_allocators[out_name]
        new_locals = [f"{out_name}_{i}" for i in range(size)]
        if size == 1:
            line = f"{self.context.indent()}{new_locals[0]}, = {ret_temp}"
        else:
            line = f"{self.context.indent()}{', '.join(new_locals)} = {ret_temp}"
        linearity = self._linearity()
        for i, local in enumerate(new_locals):
            linearity.set_live(Slot(out_name, i), local)
        return [line]

    def _emit_block(self, body: tuple[Statement, ...]) -> list[str]:
        lines: list[str] = []
        for stmt in body:
            lines.extend(self._emit_stmt(stmt))
        return lines

    def _emit_print(self, node: PrintOp) -> list[str]:
        """Lower PrintOp to a Guppy `result(<namespace>.<tag>, <value>)` call.

        Per v2-print.md, Print is scope-orthogonal: it does not allocate, does
        not touch the result-register set, and does not affect main's return
        type. Path-signature consistency for Print inside If branches and
        inline-CReg definite-assignment are enforced by separate validation
        passes; this emitter assumes both have already accepted the AST.
        """
        full_tag = f"{node.namespace}.{node.tag}"

        value_expr: str
        if isinstance(node.value, BitRef):
            register = node.value.register
            if register not in self.context.registers:
                msg = (
                    f"Print(c[{node.value.index}]) references unknown CReg {register!r}; "
                    "declare the CReg or measure into it before Print."
                )
                raise GuppyCodegenError(msg)
            value_expr = f"{register}[{node.value.index}]"
        elif isinstance(node.value, str):
            register = node.value
            if register not in self.context.registers:
                msg = (
                    f"Print({register}) references unknown CReg {register!r}; "
                    "declare the CReg or measure into it before Print."
                )
                raise GuppyCodegenError(msg)
            value_expr = register
        else:
            msg = f"Unsupported Print value type for Guppy codegen: {type(node.value).__name__}"
            raise GuppyCodegenError(msg)

        return [f'{self.context.indent()}result("{full_tag}", {value_expr})']

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

    def _render_expression(self, expr: Expression, *, bool_context: bool = False) -> str:
        if isinstance(expr, LiteralExpr):
            return self._render_literal(expr, bool_context=bool_context)
        if isinstance(expr, VarExpr):
            return expr.name
        if isinstance(expr, BitExpr):
            return self._render_bit_ref(expr.ref)
        if isinstance(expr, BinaryExpr):
            return self._render_binary(expr)
        if isinstance(expr, UnaryExpr):
            return self._render_unary(expr, bool_context=bool_context)
        msg = f"Unsupported Guppy expression {expr!r}"
        raise GuppyCodegenError(msg)

    def _render_literal(self, expr: LiteralExpr, *, bool_context: bool = False) -> str:
        if isinstance(expr.value, bool):
            return "True" if expr.value else "False"
        if bool_context and isinstance(expr.value, int):
            if expr.value in {0, 1}:
                return "True" if expr.value else "False"
            msg = f"Cannot render integer literal {expr.value!r} as a Guppy bool"
            raise GuppyCodegenError(msg)
        return str(expr.value)

    def _render_binary(self, expr: BinaryExpr) -> str:
        op = BINARY_OP_TO_PYTHON.get(expr.op)
        if op is None:
            msg = f"Unsupported Guppy binary op {expr.op.name}"
            raise GuppyCodegenError(msg)

        compares_bool_expression = expr.op in BOOL_COMPARISON_OPS and (
            self._is_bool_expression(expr.left) or self._is_bool_expression(expr.right)
        )
        operand_bool_context = expr.op in BOOL_OPERAND_BINARY_OPS or compares_bool_expression
        left = self._render_expression(expr.left, bool_context=operand_bool_context)
        right = self._render_expression(expr.right, bool_context=operand_bool_context)
        return f"({left} {op} {right})"

    def _render_unary(self, expr: UnaryExpr, *, bool_context: bool = False) -> str:
        operand = self._render_expression(expr.operand, bool_context=bool_context or expr.op == UnaryOp.NOT)
        if expr.op == UnaryOp.NOT:
            return f"(not {operand})"
        if expr.op == UnaryOp.NEG:
            return f"(-{operand})"
        msg = f"Unsupported Guppy unary op {expr.op.name}"
        raise GuppyCodegenError(msg)

    def _is_bool_expression(self, expr: Expression) -> bool:
        if isinstance(expr, BitExpr):
            return True
        if isinstance(expr, LiteralExpr):
            return isinstance(expr.value, bool)
        if isinstance(expr, UnaryExpr):
            return expr.op == UnaryOp.NOT
        return isinstance(expr, BinaryExpr) and expr.op in {
            BinaryOp.AND,
            BinaryOp.OR,
            BinaryOp.XOR,
            BinaryOp.EQ,
            BinaryOp.NE,
            BinaryOp.LT,
            BinaryOp.LE,
            BinaryOp.GT,
            BinaryOp.GE,
        }

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


def validate_slr_for_guppy_v1(block: object | None) -> None:
    """Reject SLR constructs that the v1 AST -> Guppy path cannot represent soundly."""
    if block is None:
        return
    _validate_slr_node_for_guppy_v1(block)


def _validate_slr_node_for_guppy_v1(node: object) -> None:
    node_type = type(node).__name__
    if node_type == "While":
        msg = "AST -> Guppy v1 does not support While loops"
        raise GuppyCodegenError(msg)

    if getattr(node, "is_qgate", False):
        _validate_slr_gate_for_guppy_v1(node)

    for child in getattr(node, "ops", ()) or ():
        _validate_slr_node_for_guppy_v1(child)

    else_block = getattr(node, "else_block", None)
    if else_block is not None:
        _validate_slr_node_for_guppy_v1(else_block)


def _validate_slr_gate_for_guppy_v1(gate: object) -> None:
    qargs = getattr(gate, "qargs", ()) or ()
    cout = getattr(gate, "cout", ()) or ()

    if _contains_symbolic_index(qargs) or _contains_symbolic_index(cout):
        msg = "AST -> Guppy v1 does not support symbolic LoopVar indexing"
        raise GuppyCodegenError(msg)

    if getattr(gate, "sym", None) != "Prep":
        return

    for basis in _string_args(qargs):
        if basis.strip().upper() not in {"Z", "+Z"}:
            msg = "AST -> Guppy v1 supports only Z-basis Prep; use Prep(q); H(q) for X-basis prep"
            raise GuppyCodegenError(msg)


def _contains_symbolic_index(value: object) -> bool:
    for item in _nested_items(value):
        if hasattr(item, "index_var") or type(item).__name__.startswith("Symbolic"):
            return True
    return False


def _string_args(value: object) -> list[str]:
    return [item for item in _nested_items(value) if isinstance(item, str)]


def _nested_items(value: object) -> Iterator[object]:
    if isinstance(value, str):
        yield value
        return
    if isinstance(value, list | tuple):
        for item in value:
            yield from _nested_items(item)
        return
    yield value
