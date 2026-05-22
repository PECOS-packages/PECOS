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

"""BlockDecl/BlockCall flattening for non-Guppy codegens.

Non-Guppy codegens (qasm, qir, stim, quantum_circuit) cannot represent
reusable functions, so a `BlockCall` is inlined at its call site by
substituting each input parameter name with the corresponding
`arg_binding` outer-scope allocator name.

The Guppy emitter does NOT use this pass: it lowers `BlockDecl` to
`@guppy def` and `BlockCall` to a packed-array call.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pecos.slr.ast._block_substitution import BodyRemap, substitute_stmt
from pecos.slr.ast.nodes import (
    AllocatorArg,
    BitBundleArg,
    BlockCall,
    ForStmt,
    IfStmt,
    ParallelBlock,
    Program,
    QubitBundleArg,
    RepeatStmt,
    SingleBitArg,
    SingleQubitArg,
    WhileStmt,
)

if TYPE_CHECKING:
    from pecos.slr.ast.nodes import BlockDecl, Statement


def validate_unique_block_decl_names(program: Program) -> None:
    """Raise ValueError if any BlockDecl name appears more than once.

    Shared precondition check: both the Guppy emitter and the non-Guppy
    flatten pass require globally-unique BlockDecl
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
    typed `BlockCall.arg_bindings` BlockArg. Currently only
    `AllocatorArg` is supported; richer BlockArg shapes raise
    `NotImplementedError`. Quantum-only for now.
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
    # 5e.2: AllocatorArg / SingleQubitArg / SingleBitArg / QubitBundleArg all
    # inline. BitBundleArg is the only still-deferred shape -- reject it (in
    # BOTH arg and out bindings; silently allowing it in out_bindings would
    # be a silent-fallback). The `test_bitbundle_*_rejected` lock-ins depend
    # on this.
    for position, args in (("arg", call.arg_bindings), ("out", call.out_bindings)):
        for arg in args:
            if isinstance(arg, BitBundleArg):
                msg = (
                    f"Flatten pass does not yet support BlockArg "
                    f"{type(arg).__name__} in {position}_bindings of "
                    f"{call.callee!r} (BitBundleArg is still deferred)"
                )
                raise NotImplementedError(msg)
    # Build the PARAM -> OUTER BodyRemap (flatten inlines a BlockDecl body --
    # which references param names -- at the call site, which uses the outer
    # binding). converter builds the inverse OUTER -> PARAM remap; both use
    # the shared `substitute_stmt` (5e.1 unification -- one substitution core,
    # no fix-one-forget-the-mirror drift). Only arg_bindings drive the body
    # rewrite; out_bindings do not contribute (the inlined body writes the
    # outer slots directly -- there is no separate return-unpack in flatten).
    remap = BodyRemap()
    for inp, arg in zip(decl.inputs, call.arg_bindings, strict=True):
        if isinstance(arg, AllocatorArg):
            # AllocatorArg inputs are always array[qubit, N]; the emitter
            # validates this, so flatten can trust inp.type_expr.size.
            size = getattr(inp.type_expr, "size", 0)
            remap.add_whole_alloc(inp.name, arg.name, size)
        elif isinstance(arg, SingleQubitArg):
            remap.add_slot((inp.name, 0), (arg.slot.allocator, arg.slot.index))
        elif isinstance(arg, SingleBitArg):
            remap.add_bit((inp.name, 0), (arg.bit.register, arg.bit.index))
        elif isinstance(arg, QubitBundleArg):
            for k, slot in enumerate(arg.slots):
                remap.add_slot((inp.name, k), (slot.allocator, slot.index))
        else:  # BitBundleArg already rejected above; defensive
            msg = f"Flatten pass: unexpected BlockArg {type(arg).__name__} for input {inp.name!r} of {call.callee!r}"
            raise NotImplementedError(msg)
    return tuple(substitute_stmt(stmt, remap) for stmt in decl.body)
