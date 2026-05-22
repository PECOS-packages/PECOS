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

"""No-arg `entry()` wrapper for AST-emitted parameterized `main(...)`.

The AST emitter produces `main(q: array[qubit, N] @ owned, ...)`. Downstream
HUGR consumers (`pecos.Hugr(bytes)`, `pecos_rslib.HugrProgram`, the Selene
runtime) require a no-arg entrypoint, matching the legacy IR generator's
shape. This module builds that wrapper by mirroring the same return-shape
logic the emitter uses, so the wrapper signature matches main's exactly.

Two modes match `AstToGuppy._return_type`:
- Explicit `Return(...)` -> pass through main's return value unchanged.
- No `Return(...)` -> `entry() -> None` and discard (the output model no
  longer implicitly returns result-flagged CRegs).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pecos.slr.ast import (
    AllocatorDecl,
    BitExpr,
    ForStmt,
    IfStmt,
    LiteralExpr,
    MeasureOp,
    ParallelBlock,
    RegisterDecl,
    RepeatStmt,
    ReturnOp,
    WhileStmt,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from pecos.slr.ast import Expression, Program, Statement


@dataclass(frozen=True)
class EntryWrapperInfo:
    """Metadata extracted from the AST program for building the wrapper.

    `all_creg_sizes` mirrors the emitter's `context.registers` lookup view
    used by `AstToGuppy._return_value_type`: every declared CReg plus every
    inline-Measure-introduced CReg. Needed so an explicit `Return(...)`
    referencing a declared or inline CReg resolves the same way the emitter
    does (not as `ValueError`).
    """

    allocator_sizes: dict[str, int]
    explicit_return: ReturnOp | None
    all_creg_sizes: dict[str, int]


# Reserved namespace for the opt-in returned-CReg result tags. Private
# (double-underscore) so it cannot collide with user `Print(..., namespace=
# "result")` -> `result.<tag>` outputs. Single source of truth shared with
# `_selene_harness` so the emit/read sides never drift.
RETURN_TAG_NAMESPACE = "__pecos_return"


def build_no_arg_entry_wrapper(
    program: Program,
    *,
    emit_return_result_tags: bool = False,
) -> tuple[str, EntryWrapperInfo]:
    """Generate the wrapper source and return the metadata used to build it.

    Args:
        program: The AST program to wrap.
        emit_return_result_tags: **Opt-in, test-harness only.** When True and
            the program has an explicit `Return(...)`, the wrapper
            destructures main's return and emits
            `result("__pecos_return.<creg>", <creg>)` per returned CReg
            instead of `return <tuple>`, and is typed `-> None`. This makes
            Selene key the outputs by CReg name (immune to internal,
            non-returned measurements -- e.g. RUS verify). Default False
            keeps the production wrapper byte-identical (the `.hugr()` /
            raw-`measurement_N` consumers must not change).

    Returns:
        A `(source, info)` tuple. `source` is the Guppy snippet defining
        `entry()`; concatenate it after the main source. `info` exposes the
        allocator sizes and explicit Return (if any) that the caller may
        need (e.g., Selene's measurement-key generation).
    """
    info = _collect_info(program)
    source = _render_wrapper(info, emit_return_result_tags=emit_return_result_tags)
    return source, info


def _collect_info(program: Program) -> EntryWrapperInfo:
    allocator_sizes: dict[str, int] = {}
    for decl in getattr(program, "declarations", ()):
        if isinstance(decl, AllocatorDecl) and decl.parent is None:
            allocator_sizes.setdefault(decl.name, decl.capacity)
    top = getattr(program, "allocator", None)
    if isinstance(top, AllocatorDecl) and top.parent is None:
        allocator_sizes.setdefault(top.name, top.capacity)

    declared: dict[str, RegisterDecl] = {}
    for decl in getattr(program, "declarations", ()):
        if isinstance(decl, RegisterDecl):
            declared[decl.name] = decl

    body = getattr(program, "body", ())
    inline_max: dict[str, int] = {}
    _walk_for_measure_results(body, declared, inline_max)

    all_creg_sizes: dict[str, int] = {name: decl.size for name, decl in declared.items()}
    for name, max_index in inline_max.items():
        all_creg_sizes[name] = max_index + 1

    explicit_return = body[-1] if body and isinstance(body[-1], ReturnOp) else None

    return EntryWrapperInfo(
        allocator_sizes=allocator_sizes,
        explicit_return=explicit_return,
        all_creg_sizes=all_creg_sizes,
    )


def _walk_for_measure_results(
    body: Iterable[Statement],
    declared: dict[str, RegisterDecl],
    inline_max: dict[str, int],
) -> None:
    for stmt in body:
        if isinstance(stmt, MeasureOp):
            for ref in stmt.results:
                if ref.register not in declared:
                    inline_max[ref.register] = max(inline_max.get(ref.register, -1), ref.index)
        elif isinstance(stmt, IfStmt):
            _walk_for_measure_results(stmt.then_body, declared, inline_max)
            _walk_for_measure_results(stmt.else_body, declared, inline_max)
        elif isinstance(stmt, (RepeatStmt, ForStmt, WhileStmt, ParallelBlock)):
            _walk_for_measure_results(stmt.body, declared, inline_max)


def _render_wrapper(info: EntryWrapperInfo, *, emit_return_result_tags: bool = False) -> str:
    body_lines: list[str] = [
        f"    {name} = array(qubit() for _ in range({size}))" for name, size in info.allocator_sizes.items()
    ]
    call_args = ", ".join(info.allocator_sizes.keys())
    call_expr = f"main({call_args})"

    if emit_return_result_tags and info.explicit_return is not None:
        # Opt-in: destructure main's return and tag each returned CReg
        # by name so Selene keys outputs by name, not positional
        # measurement_N (which counts internal measurements too).
        targets = [v if isinstance(v, str) else getattr(v, "name", None) for v in info.explicit_return.values]
        if any(t is None for t in targets):
            msg = f"emit_return_result_tags supports only named return values, got {info.explicit_return.values!r}"
            raise ValueError(msg)
        lhs = targets[0] if len(targets) == 1 else ", ".join(targets)
        body_lines.append(f"    {lhs} = {call_expr}")
        body_lines.extend(
            f'    result("{RETURN_TAG_NAMESPACE}.{name}", {name})' for name in targets if name in info.all_creg_sizes
        )
        return_ann = "None"
    elif info.explicit_return is not None:
        body_lines.append(f"    return {call_expr}")
        return_ann = _explicit_return_type(info)
    else:
        body_lines.append(f"    {call_expr}")
        return_ann = "None"

    body = "\n".join(body_lines) if body_lines else "    pass"
    return f"\n\n@guppy\ndef entry() -> {return_ann}:\n{body}\n"


def _explicit_return_type(info: EntryWrapperInfo) -> str:
    assert info.explicit_return is not None  # noqa: S101
    types = [
        _return_value_type(value, info.allocator_sizes, info.all_creg_sizes) for value in info.explicit_return.values
    ]
    return _tuple_type(types)


def _return_value_type(value: Expression | str, allocator_sizes: dict[str, int], creg_sizes: dict[str, int]) -> str:
    if isinstance(value, str):
        if value in allocator_sizes:
            return f"array[qubit, {allocator_sizes[value]}]"
        if value in creg_sizes:
            return f"array[bool, {creg_sizes[value]}]"
        msg = f"Unsupported Guppy return value {value!r}"
        raise ValueError(msg)
    if isinstance(value, BitExpr):
        return "bool"
    if isinstance(value, LiteralExpr) and isinstance(value.value, bool):
        return "bool"
    if isinstance(value, LiteralExpr) and isinstance(value.value, int):
        return "int"
    msg = f"Unsupported Guppy return expression {value!r}"
    raise ValueError(msg)


def _tuple_type(types: list[str]) -> str:
    """Mirror `AstToGuppy._tuple_type`: empty -> None, single -> bare, multi -> tuple[...]."""
    if not types:
        return "None"
    if len(types) == 1:
        return types[0]
    return "tuple[" + ", ".join(types) + "]"


def truncate_source_for_error(source: str, max_lines: int = 80) -> str:
    """Truncate generated Guppy source for inclusion in an error message."""
    lines = source.splitlines()
    if len(lines) <= max_lines:
        return source
    head = lines[: max_lines - 10]
    tail = lines[-10:]
    return "\n".join([*head, f"... ({len(lines) - max_lines} lines elided) ...", *tail])
