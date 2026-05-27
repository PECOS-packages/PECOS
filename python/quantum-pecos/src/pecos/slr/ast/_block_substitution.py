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

"""Shared BlockDecl-body reference substitution.

`BodyRemap` is a bidirectional slot/bit/whole-name remap. Two consumers
build it in opposite directions but use the identical substitution core:

- `converter._convert_block_call` builds an OUTER -> PARAM remap, turning a
  Block instance's body into a reusable `BlockDecl` body.
- `_block_flatten._inline_call` builds the inverse PARAM -> OUTER remap,
  inlining a `BlockDecl` body at its call site for non-Guppy codegens.

Unifying both into one `substitute_stmt` eliminates the
fix-one-forget-the-mirror bug class: two parallel copies previously
drifted, leaving PermuteOp and BlockCall substitution gaps.

`substitute_stmt` is **fully recursive** over every node that can carry a
`SlotRef` / `BitRef` / expression, including the spots the pre-5e
allocator-name-only substitution silently skipped (`MeasureOp.results`,
`AssignOp`, `PrintOp`, `ReturnOp`, conditions, `GateOp.params`,
`ForStmt` bounds, nested `BlockCall`).

Name-level refs (`BarrierOp`, bare `PermuteOp`, `PrepareOp(slots=None)`,
`VarExpr`, str `ReturnOp`/`AssignOp` targets) cannot express a partial
(single-qubit / bundle / single-bit) binding; touching a partially-bound
outer name raises `BodySubstitutionError` rather than silently leaking it.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from pecos.slr.ast.nodes import (
    AllocatorArg,
    AssignOp,
    BarrierOp,
    BinaryExpr,
    BitBundleArg,
    BitExpr,
    BitRef,
    BlockCall,
    CommentOp,
    ForStmt,
    GateOp,
    IfStmt,
    LiteralExpr,
    MeasureOp,
    ParallelBlock,
    PermuteOp,
    PrepareOp,
    PrintOp,
    QubitBundleArg,
    RepeatStmt,
    ReturnOp,
    SingleBitArg,
    SingleQubitArg,
    SlotRef,
    UnaryExpr,
    VarExpr,
    WhileStmt,
)

if TYPE_CHECKING:
    from pecos.slr.ast.nodes import BlockArg, Expression, Statement

# A PermuteOp source/target string is either a bare name or `name[idx]`.
_PERMUTE_REF_RE = re.compile(r"([A-Za-z_]\w*)(?:\[(\d+)\])?$")
# Leading identifier token of an arbitrary (possibly unparseable) ref, used
# to decide reject-on-partial WITHOUT a loose substring match -- `"q" in
# "sq[0:2]"` would otherwise falsely reject an unrelated `sq` ref.
_LEADING_IDENT_RE = re.compile(r"[A-Za-z_]\w*")


class BodySubstitutionError(ValueError):
    """A body reference cannot be substituted (e.g. partial-binding name use).

    Subclasses `ValueError` so existing call sites/tests that expect a
    `ValueError` from the old per-helper substitution keep working.
    """


class BodyRemap:
    """Slot/bit/whole-name remap for BlockDecl body rewriting.

    Tables map a source `(name, index)` to a destination `(name, index)`.
    `whole_alloc` records names that are bound *completely* (a whole-`QReg`
    binding); only those may be renamed at the name level. Any name that
    appears in a per-slot/per-bit binding (single qubit, bundle, single
    bit) is recorded as *partial*: name-level use of it is rejected.
    """

    def __init__(self) -> None:
        self._slot: dict[tuple[str, int], tuple[str, int]] = {}
        self._bit: dict[tuple[str, int], tuple[str, int]] = {}
        self._whole_alloc: dict[str, str] = {}
        self._partial_names: set[str] = set()

    # ---- builders ----
    #
    # An outer name may be bound in exactly ONE mode -- whole xor partial --
    # and a whole binding maps exactly one src. Conflicting builder calls
    # (same name bound both whole and partial, or whole-bound twice) are an
    # input-aliasing error: reject at construction time rather than letting
    # whole silently win at lookup. This protects the QASM/flatten path too,
    # where Guppy's own linearity alias check never runs.

    def _reject_conflict(self, name: str, *, mode: str) -> None:
        if name in self._whole_alloc:
            msg = (
                f"BodyRemap: outer name {name!r} is already bound whole; "
                f"cannot also bind it {mode} (input aliasing)"
            )
            raise BodySubstitutionError(msg)
        if name in self._partial_names and mode == "whole":
            msg = (
                f"BodyRemap: outer name {name!r} is already bound partially; "
                f"cannot also bind it whole (input aliasing)"
            )
            raise BodySubstitutionError(msg)

    def add_whole_alloc(self, src: str, dst: str, size: int) -> None:
        """Bind a whole QReg `src` (size N) to `dst`, identity per-slot."""
        self._reject_conflict(src, mode="whole")
        self._whole_alloc[src] = dst
        for i in range(size):
            self._slot[(src, i)] = (dst, i)

    def add_slot(self, src: tuple[str, int], dst: tuple[str, int]) -> None:
        """Bind one outer qubit slot; marks `src` allocator partial.

        A repeated exact `src` slot is rejected: silently overwriting it
        would drop the earlier binding and corrupt the body rewrite -- e.g.
        a `[q[0], q[0]]` bundle, or two single-qubit inputs aliased to the
        same outer slot. Qubit aliasing is invalid anyway (no-cloning);
        fail loudly here so it cannot reach codegen.
        """
        self._reject_conflict(src[0], mode="partial (per-slot)")
        if src in self._slot:
            msg = (
                f"BodyRemap: outer qubit slot {src!r} is already bound "
                f"(to {self._slot[src]!r}); a qubit cannot be aliased to "
                "two block-input positions (no-cloning)"
            )
            raise BodySubstitutionError(msg)
        self._slot[src] = dst
        self._partial_names.add(src[0])

    def add_bit(self, src: tuple[str, int], dst: tuple[str, int]) -> None:
        """Bind one outer classical bit; marks `src` register partial.

        A repeated exact `src` bit is rejected for the same reason as
        `add_slot`: a second binding would silently overwrite the first
        and lose body references to it during substitution.
        """
        self._reject_conflict(src[0], mode="partial (per-bit)")
        if src in self._bit:
            msg = (
                f"BodyRemap: outer bit {src!r} is already bound "
                f"(to {self._bit[src]!r}); the same outer bit cannot back "
                "two block-input positions (lossy substitution)"
            )
            raise BodySubstitutionError(msg)
        self._bit[src] = dst
        self._partial_names.add(src[0])

    # ---- lookups ----

    def slot(self, ref: SlotRef) -> SlotRef:
        dst = self._slot.get((ref.allocator, ref.index))
        if dst is None:
            return ref
        return SlotRef(allocator=dst[0], index=dst[1], location=ref.location)

    def bit(self, ref: BitRef) -> BitRef:
        dst = self._bit.get((ref.register, ref.index))
        if dst is None:
            return ref
        return BitRef(register=dst[0], index=dst[1], location=ref.location)

    def whole_name(self, name: str, *, context: str) -> str:
        """Rename a whole-allocator/register name; reject if partially bound.

        Unmapped names pass through unchanged (they reference allocators not
        bound by any input -- a Block-local register, say).
        """
        if name in self._whole_alloc:
            return self._whole_alloc[name]
        if name in self._partial_names:
            msg = (
                f"{context} references {name!r}, which is only partially bound "
                f"(a single-qubit / bundle / single-bit input). A whole-name "
                f"reference cannot express that binding -- pass the whole "
                f"register, or restructure the Block so the body does not use "
                f"{name!r} by bare name."
            )
            raise BodySubstitutionError(msg)
        return name


def substitute_stmt(stmt: Statement, remap: BodyRemap) -> Statement:
    """Return `stmt` with every slot/bit/expression reference remapped."""
    if isinstance(stmt, GateOp):
        return GateOp(
            gate=stmt.gate,
            targets=tuple(remap.slot(t) for t in stmt.targets),
            params=tuple(_sub_expr(p, remap) for p in stmt.params),
            location=stmt.location,
        )
    if isinstance(stmt, MeasureOp):
        return MeasureOp(
            targets=tuple(remap.slot(t) for t in stmt.targets),
            results=tuple(remap.bit(r) for r in stmt.results),
            location=stmt.location,
        )
    if isinstance(stmt, PrepareOp):
        return _sub_prepare(stmt, remap)
    if isinstance(stmt, BarrierOp):
        return BarrierOp(
            allocators=tuple(remap.whole_name(a, context="Barrier") for a in stmt.allocators),
            location=stmt.location,
        )
    if isinstance(stmt, AssignOp):
        target = stmt.target
        new_target = (
            remap.bit(target) if isinstance(target, BitRef) else remap.whole_name(target, context="assignment target")
        )
        return AssignOp(
            target=new_target,
            value=_sub_expr(stmt.value, remap),
            location=stmt.location,
        )
    if isinstance(stmt, ReturnOp):
        # Substitution remaps names in place (1:1, order/count
        # preserved), so the parallel `value_kinds` provenance still
        # aligns and MUST be carried (dropping it would re-introduce
        # the CReg/QReg name-collision miscompile inside a
        # substituted BlockCall body).
        return ReturnOp(
            values=tuple(
                _sub_expr(v, remap) if not isinstance(v, str) else remap.whole_name(v, context="Return value")
                for v in stmt.values
            ),
            value_kinds=stmt.value_kinds,
            location=stmt.location,
        )
    if isinstance(stmt, PrintOp):
        value = stmt.value
        new_value = remap.bit(value) if isinstance(value, BitRef) else remap.whole_name(value, context="Print value")
        return PrintOp(
            value=new_value,
            tag=stmt.tag,
            namespace=stmt.namespace,
            location=stmt.location,
        )
    if isinstance(stmt, IfStmt):
        return IfStmt(
            condition=_sub_expr(stmt.condition, remap),
            then_body=tuple(substitute_stmt(s, remap) for s in stmt.then_body),
            else_body=tuple(substitute_stmt(s, remap) for s in stmt.else_body),
            location=stmt.location,
        )
    if isinstance(stmt, WhileStmt):
        return WhileStmt(
            condition=_sub_expr(stmt.condition, remap),
            body=tuple(substitute_stmt(s, remap) for s in stmt.body),
            location=stmt.location,
        )
    if isinstance(stmt, ForStmt):
        return ForStmt(
            variable=stmt.variable,
            start=_sub_expr(stmt.start, remap),
            stop=_sub_expr(stmt.stop, remap),
            step=None if stmt.step is None else _sub_expr(stmt.step, remap),
            body=tuple(substitute_stmt(s, remap) for s in stmt.body),
            location=stmt.location,
        )
    if isinstance(stmt, RepeatStmt):
        return RepeatStmt(
            count=stmt.count,  # plain int -- no refs
            body=tuple(substitute_stmt(s, remap) for s in stmt.body),
            location=stmt.location,
        )
    if isinstance(stmt, ParallelBlock):
        return ParallelBlock(
            body=tuple(substitute_stmt(s, remap) for s in stmt.body),
            location=stmt.location,
        )
    if isinstance(stmt, PermuteOp):
        return PermuteOp(
            sources=tuple(_sub_permute_ref(r, remap) for r in stmt.sources),
            targets=tuple(_sub_permute_ref(r, remap) for r in stmt.targets),
            add_comment=stmt.add_comment,
            whole_register=stmt.whole_register,
            location=stmt.location,
        )
    if isinstance(stmt, BlockCall):
        return BlockCall(
            callee=stmt.callee,
            arg_bindings=tuple(_sub_block_arg(a, remap) for a in stmt.arg_bindings),
            out_bindings=tuple(_sub_block_arg(a, remap) for a in stmt.out_bindings),
            location=stmt.location,
        )
    # CommentOp + anything else carrying no slot/bit/expr ref: pass through.
    if not isinstance(stmt, CommentOp):  # defensive: surface unhandled nodes
        # Unknown statement types are passed through unchanged, matching the
        # pre-5e behavior; if a new ref-bearing node is added it must be
        # wired in here (the iter-5e plan's recurring-risk note).
        pass
    return stmt


def _sub_expr(expr: Expression, remap: BodyRemap) -> Expression:
    """Recurse an expression, remapping any BitRef/VarExpr it contains."""
    if isinstance(expr, LiteralExpr):
        return expr
    if isinstance(expr, VarExpr):
        return VarExpr(
            name=remap.whole_name(expr.name, context="variable expression"),
            location=expr.location,
        )
    if isinstance(expr, BitExpr):
        return BitExpr(ref=remap.bit(expr.ref), location=expr.location)
    if isinstance(expr, BinaryExpr):
        return BinaryExpr(
            op=expr.op,
            left=_sub_expr(expr.left, remap),
            right=_sub_expr(expr.right, remap),
            location=expr.location,
        )
    if isinstance(expr, UnaryExpr):
        return UnaryExpr(
            op=expr.op,
            operand=_sub_expr(expr.operand, remap),
            location=expr.location,
        )
    return expr


def _sub_prepare(stmt: PrepareOp, remap: BodyRemap) -> PrepareOp:
    """Remap a PrepareOp.

    `slots=None` (prepare_all) is a whole-register op -> name-level rename
    (reject if partially bound). `slots=(...)` is per-slot: remap each
    `(allocator, i)`; all must land in the same destination allocator.
    """
    if stmt.slots is None:
        return PrepareOp(
            allocator=remap.whole_name(stmt.allocator, context="Prepare-all"),
            slots=None,
            basis=stmt.basis,
            location=stmt.location,
        )
    remapped = [remap.slot(SlotRef(allocator=stmt.allocator, index=i)) for i in stmt.slots]
    dst_allocs = {r.allocator for r in remapped}
    if len(dst_allocs) > 1:
        msg = (
            f"Prepare on {stmt.allocator!r} maps to multiple destination "
            f"allocators {sorted(dst_allocs)} under a bundle binding; a single "
            "Prepare cannot span allocators -- restructure the Block."
        )
        raise BodySubstitutionError(msg)
    (dst_alloc,) = dst_allocs
    return PrepareOp(
        allocator=dst_alloc,
        slots=tuple(r.index for r in remapped),
        basis=stmt.basis,
        location=stmt.location,
    )


def _sub_permute_ref(ref: str, remap: BodyRemap) -> str:
    """Remap a PermuteOp `name` or `name[idx]` string.

    `name[idx]` resolves through the per-slot table (works for whole-alloc
    identity AND bundles). Bare `name` is a whole-register reference ->
    name-level rename, rejected if partially bound. Unparseable refs that
    mention a partially-bound name are rejected rather than silently leaked.
    """
    match = _PERMUTE_REF_RE.fullmatch(ref)
    if match is None:
        # Compare the ref's LEADING identifier token exactly against the
        # partial-name set (not a substring scan) so `sq[0:2]` is not
        # falsely rejected when `q` is partially bound.
        lead = _LEADING_IDENT_RE.match(ref)
        if lead is not None and lead.group(0) in remap._partial_names:  # noqa: SLF001 -- same module
            base = lead.group(0)
            msg = (
                f"Cannot substitute PermuteOp ref {ref!r}: unsupported "
                f"ref form whose base name {base!r} is partially bound"
            )
            raise BodySubstitutionError(msg)
        return ref
    name, idx = match.group(1), match.group(2)
    if idx is None:
        return remap.whole_name(name, context="Permute")
    new = remap.slot(SlotRef(allocator=name, index=int(idx)))
    return f"{new.allocator}[{new.index}]"


def _sub_block_arg(arg: BlockArg, remap: BodyRemap) -> BlockArg:
    """Remap a nested BlockCall's arg/out BlockArg through the remap."""
    if isinstance(arg, AllocatorArg):
        return AllocatorArg(
            name=remap.whole_name(arg.name, context="nested BlockCall AllocatorArg"),
            location=arg.location,
        )
    if isinstance(arg, SingleQubitArg):
        return SingleQubitArg(slot=remap.slot(arg.slot), location=arg.location)
    if isinstance(arg, SingleBitArg):
        return SingleBitArg(bit=remap.bit(arg.bit), location=arg.location)
    if isinstance(arg, QubitBundleArg):
        return QubitBundleArg(
            slots=tuple(remap.slot(s) for s in arg.slots),
            location=arg.location,
        )
    if isinstance(arg, BitBundleArg):
        return BitBundleArg(
            bits=tuple(remap.bit(b) for b in arg.bits),
            location=arg.location,
        )
    return arg
