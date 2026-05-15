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

"""Phase 3a.2 BlockDecl/BlockCall flattening for non-Guppy codegens.

Non-Guppy codegens (qasm, qir, stim, quantum_circuit) cannot represent
reusable functions, so a `BlockCall` is inlined at its call site by
substituting each input parameter name with the corresponding
`arg_binding` outer-scope allocator name.

The Guppy emitter does NOT use this pass: it lowers `BlockDecl` to
`@guppy def` and `BlockCall` to a packed-array call.

See `~/Repos/pecos-docs/design/slr/v2-blockcall-resource-effects.md`.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from pecos.slr.ast.nodes import (
    AllocatorArg,
    BarrierOp,
    BitBundleArg,
    BitRef,
    BlockCall,
    ForStmt,
    GateOp,
    IfStmt,
    MeasureOp,
    ParallelBlock,
    PermuteOp,
    PrepareOp,
    Program,
    QubitBundleArg,
    RepeatStmt,
    SingleBitArg,
    SingleQubitArg,
    SlotRef,
    WhileStmt,
)

_PERMUTE_REF_RE = re.compile(r"([A-Za-z_]\w*)(\[\d+\])?$")

if TYPE_CHECKING:
    from pecos.slr.ast.nodes import BlockArg, BlockDecl, Statement


def validate_unique_block_decl_names(program: Program) -> None:
    """Raise ValueError if any BlockDecl name appears more than once.

    Shared precondition check (Codex 2026-05-15 review #3): both the Guppy
    emitter and the non-Guppy flatten pass require globally-unique BlockDecl
    names. Keeping the check in one place ensures the contract stays in sync
    across codegens.
    """
    seen: set[str] = set()
    for decl in program.block_decls:
        if decl.name in seen:
            msg = f"Duplicate BlockDecl name {decl.name!r}"
            raise ValueError(msg)
        seen.add(decl.name)


def flatten_block_calls(program: Program) -> Program:
    """Return a new Program with every BlockCall inlined and no BlockDecls left.

    The substitution rule maps each `BlockDecl` input parameter name to the
    `BlockCall.arg_bindings` outer-scope allocator name throughout the body.
    Quantum-only in Phase 3a.1.
    """
    if not program.block_decls:
        return program

    validate_unique_block_decl_names(program)
    decls = {decl.name: decl for decl in program.block_decls}
    new_body = _flatten_stmts(program.body, decls)
    return Program(
        name=program.name,
        declarations=program.declarations,
        body=new_body,
        returns=program.returns,
        allocator=program.allocator,
        block_decls=(),
    )


def _flatten_stmts(body: tuple[Statement, ...], decls: dict[str, BlockDecl]) -> tuple[Statement, ...]:
    out: list[Statement] = []
    for stmt in body:
        if isinstance(stmt, BlockCall):
            inlined = _inline_call(stmt, decls)
            out.extend(_flatten_stmts(inlined, decls))
            continue

        if isinstance(stmt, IfStmt):
            out.append(
                IfStmt(
                    condition=stmt.condition,
                    then_body=_flatten_stmts(stmt.then_body, decls),
                    else_body=_flatten_stmts(stmt.else_body, decls),
                    location=stmt.location,
                ),
            )
            continue

        if isinstance(stmt, RepeatStmt):
            out.append(
                RepeatStmt(count=stmt.count, body=_flatten_stmts(stmt.body, decls), location=stmt.location),
            )
            continue

        if isinstance(stmt, ForStmt):
            out.append(
                ForStmt(
                    variable=stmt.variable,
                    start=stmt.start,
                    stop=stmt.stop,
                    step=stmt.step,
                    body=_flatten_stmts(stmt.body, decls),
                    location=stmt.location,
                ),
            )
            continue

        if isinstance(stmt, WhileStmt):
            out.append(
                WhileStmt(
                    condition=stmt.condition,
                    body=_flatten_stmts(stmt.body, decls),
                    location=stmt.location,
                ),
            )
            continue

        if isinstance(stmt, ParallelBlock):
            out.append(ParallelBlock(body=_flatten_stmts(stmt.body, decls), location=stmt.location))
            continue

        out.append(stmt)
    return tuple(out)


def _inline_call(call: BlockCall, decls: dict[str, BlockDecl]) -> tuple[Statement, ...]:
    decl = decls.get(call.callee)
    if decl is None:
        msg = f"BlockCall references undefined block {call.callee!r}"
        raise ValueError(msg)
    if len(call.arg_bindings) != len(decl.inputs):
        msg = (
            f"BlockCall {call.callee!r}: {len(call.arg_bindings)} arg_bindings but "
            f"BlockDecl declares {len(decl.inputs)} inputs"
        )
        raise ValueError(msg)
    # Build a name-level mapping from BlockDecl input parameter names to the outer
    # binding name. Only AllocatorArg is supported in the iter 5a flatten path;
    # richer args (single qubit/bit, bundles) need full slot-level rewriting
    # which lands with their respective flatten support iterations.
    mapping: dict[str, str] = {}
    for inp, arg in zip(decl.inputs, call.arg_bindings, strict=True):
        if isinstance(arg, AllocatorArg):
            mapping[inp.name] = arg.name
        else:
            msg = (
                f"Flatten pass does not yet support BlockArg {type(arg).__name__} for "
                f"input {inp.name!r} of {call.callee!r}; only AllocatorArg is supported "
                "in Phase 3a.3 iter 5a"
            )
            raise NotImplementedError(msg)
    return tuple(_substitute(stmt, mapping) for stmt in decl.body)


def _substitute(stmt: Statement, mapping: dict[str, str]) -> Statement:
    """Rewrite every SlotRef.allocator and PrepareOp/BarrierOp allocator name."""
    if isinstance(stmt, GateOp):
        return GateOp(
            gate=stmt.gate,
            targets=tuple(_sub_slot(t, mapping) for t in stmt.targets),
            params=stmt.params,
            location=stmt.location,
        )
    if isinstance(stmt, MeasureOp):
        return MeasureOp(
            targets=tuple(_sub_slot(t, mapping) for t in stmt.targets),
            results=stmt.results,
            location=stmt.location,
        )
    if isinstance(stmt, PrepareOp):
        return PrepareOp(
            allocator=mapping.get(stmt.allocator, stmt.allocator),
            slots=stmt.slots,
            location=stmt.location,
        )
    if isinstance(stmt, BarrierOp):
        return BarrierOp(
            allocators=tuple(mapping.get(a, a) for a in stmt.allocators),
            location=stmt.location,
        )
    if isinstance(stmt, IfStmt):
        return IfStmt(
            condition=stmt.condition,
            then_body=tuple(_substitute(s, mapping) for s in stmt.then_body),
            else_body=tuple(_substitute(s, mapping) for s in stmt.else_body),
            location=stmt.location,
        )
    if isinstance(stmt, RepeatStmt):
        return RepeatStmt(
            count=stmt.count,
            body=tuple(_substitute(s, mapping) for s in stmt.body),
            location=stmt.location,
        )
    if isinstance(stmt, ForStmt):
        return ForStmt(
            variable=stmt.variable,
            start=stmt.start,
            stop=stmt.stop,
            step=stmt.step,
            body=tuple(_substitute(s, mapping) for s in stmt.body),
            location=stmt.location,
        )
    if isinstance(stmt, WhileStmt):
        return WhileStmt(
            condition=stmt.condition,
            body=tuple(_substitute(s, mapping) for s in stmt.body),
            location=stmt.location,
        )
    if isinstance(stmt, ParallelBlock):
        return ParallelBlock(body=tuple(_substitute(s, mapping) for s in stmt.body), location=stmt.location)
    if isinstance(stmt, PermuteOp):
        return PermuteOp(
            sources=tuple(_sub_permute_ref(r, mapping) for r in stmt.sources),
            targets=tuple(_sub_permute_ref(r, mapping) for r in stmt.targets),
            add_comment=stmt.add_comment,
            location=stmt.location,
        )
    if isinstance(stmt, BlockCall):
        # Nested BlockCall: rewrite arg_bindings/out_bindings from outer names to
        # input parameter names. Codex 2026-05-15 review #1.
        return BlockCall(
            callee=stmt.callee,
            arg_bindings=tuple(_sub_block_arg(a, mapping) for a in stmt.arg_bindings),
            out_bindings=tuple(_sub_block_arg(a, mapping) for a in stmt.out_bindings),
            location=stmt.location,
        )
    # Pass through statements that don't reference slot allocators directly.
    return stmt


def _sub_block_arg(arg: BlockArg, mapping: dict[str, str]) -> BlockArg:
    """Rewrite a BlockArg's referenced names per the mapping (for nested calls)."""
    if isinstance(arg, AllocatorArg):
        return AllocatorArg(name=mapping.get(arg.name, arg.name), location=arg.location)
    if isinstance(arg, SingleQubitArg):
        return SingleQubitArg(slot=_sub_slot(arg.slot, mapping), location=arg.location)
    if isinstance(arg, SingleBitArg):
        return SingleBitArg(bit=_sub_bit_ref(arg.bit, mapping), location=arg.location)
    if isinstance(arg, QubitBundleArg):
        return QubitBundleArg(
            slots=tuple(_sub_slot(s, mapping) for s in arg.slots),
            location=arg.location,
        )
    if isinstance(arg, BitBundleArg):
        return BitBundleArg(
            bits=tuple(_sub_bit_ref(b, mapping) for b in arg.bits),
            location=arg.location,
        )
    return arg


def _sub_bit_ref(ref: BitRef, mapping: dict[str, str]) -> BitRef:
    if ref.register not in mapping:
        return ref
    return BitRef(register=mapping[ref.register], index=ref.index, location=ref.location)


def _sub_permute_ref(ref: str, mapping: dict[str, str]) -> str:
    """Rewrite PermuteOp source/target strings of the form `name` or `name[idx]`.

    If the ref doesn't match, pass through UNLESS the ref textually mentions a
    mapped allocator -- in that case raise rather than silently leak the outer
    name into the flattened body (Codex 2026-05-15 review).
    """
    match = _PERMUTE_REF_RE.fullmatch(ref)
    if match is None:
        for key in mapping:
            if key in ref:
                msg = (
                    f"Cannot substitute PermuteOp ref {ref!r}: unsupported "
                    f"ref form mentions mapped allocator {key!r}"
                )
                raise ValueError(msg)
        return ref
    name, suffix = match.group(1), match.group(2) or ""
    return f"{mapping.get(name, name)}{suffix}"


def _sub_slot(ref: SlotRef, mapping: dict[str, str]) -> SlotRef:
    if ref.allocator not in mapping:
        return ref
    return SlotRef(allocator=mapping[ref.allocator], index=ref.index, location=ref.location)
