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

"""Phase 3a.1 smoke tests for BlockDecl / BlockCall.

Builds AST programs directly (without going through SLR) to exercise the new
Guppy emitter codepaths:
- `BlockDecl` lowers to a `@guppy def` top-level function
- `BlockCall` lowers to `array(...)` pack + call + unpack
- `LIVE_PRESERVED` inputs leave outer-scope slots in the LIVE state post-call
- `CONSUMED` inputs leave outer-scope slots in the CONSUMED state post-call

See `~/Repos/pecos-docs/design/slr/v2-blockcall-resource-effects.md`.
"""

from __future__ import annotations

import re
from typing import ClassVar

import pytest
from pecos.slr.ast import (
    AllocatorArg,
    AllocatorDecl,
    ArrayTypeExpr,
    BitRef,
    BlockCall,
    BlockDecl,
    BlockInput,
    GateKind,
    GateOp,
    MeasureOp,
    PrepareOp,
    Program,
    QubitTypeExpr,
    RegisterDecl,
    ResourceEffect,
    SlotRef,
    ast_to_guppy,
)
from pecos.slr.ast.codegen.guppy import GuppyCodegenError


def _bell_program() -> Program:
    """Program with a `bell` BlockDecl that applies H + CX to a 2-qubit array."""
    bell = BlockDecl(
        name="bell",
        inputs=(
            BlockInput(
                name="q",
                effect=ResourceEffect.LIVE_PRESERVED,
                type_expr=ArrayTypeExpr(element=QubitTypeExpr(), size=2),
            ),
        ),
        body=(
            GateOp(gate=GateKind.H, targets=(SlotRef(allocator="q", index=0),)),
            GateOp(
                gate=GateKind.CX,
                targets=(
                    SlotRef(allocator="q", index=0),
                    SlotRef(allocator="q", index=1),
                ),
            ),
        ),
    )
    return Program(
        name="main",
        allocator=AllocatorDecl(name="outer_q", capacity=2),
        declarations=(RegisterDecl(name="c", size=2),),
        block_decls=(bell,),
        body=(
            PrepareOp(allocator="outer_q"),
            BlockCall(
                callee="bell",
                arg_bindings=(AllocatorArg(name="outer_q"),),
                out_bindings=(AllocatorArg(name="outer_q"),),
            ),
            MeasureOp(
                targets=(
                    SlotRef(allocator="outer_q", index=0),
                    SlotRef(allocator="outer_q", index=1),
                ),
                results=(
                    BitRef(register="c", index=0),
                    BitRef(register="c", index=1),
                ),
            ),
        ),
    )


class TestBlockDeclGuppySource:
    """Inspect the generated Guppy source for shape correctness."""

    def test_live_preserved_block_lowers_to_guppy_def_with_array_return(self) -> None:
        source = ast_to_guppy(_bell_program())

        # The BlockDecl emits its own @guppy def above main.
        assert re.search(r"@guppy\s*\n\s*def bell\(q: array\[qubit, 2\] @ owned\) -> array\[qubit, 2\]:", source)
        # Body unpacks the array, applies H + CX, and returns the repacked array.
        assert "q_0, q_1 = q" in source
        assert "q_0 = h(q_0)" in source
        assert "q_0, q_1 = cx(q_0, q_1)" in source
        assert "return array(q_0, q_1)" in source

    def test_block_call_packs_unpacks_around_call(self) -> None:
        source = ast_to_guppy(_bell_program())

        # The call site packs the outer locals, calls bell, and unpacks the return.
        assert re.search(r"_call_ret_\d+\s*=\s*bell\(array\(outer_q_0, outer_q_1\)\)", source)
        assert re.search(r"outer_q_0, outer_q_1\s*=\s*_call_ret_\d+", source)


class TestBlockCallValidation:
    """Edge cases the Guppy emitter must reject with clear errors."""

    def test_undefined_callee_rejected(self) -> None:
        prog = Program(
            name="main",
            allocator=AllocatorDecl(name="outer_q", capacity=2),
            body=(
                BlockCall(
                    callee="missing_block",
                    arg_bindings=(AllocatorArg(name="outer_q"),),
                    out_bindings=(AllocatorArg(name="outer_q"),),
                ),
            ),
        )
        with pytest.raises(GuppyCodegenError, match=r"undefined block 'missing_block'"):
            ast_to_guppy(prog)

    def test_arg_count_mismatch_rejected(self) -> None:
        bell = BlockDecl(
            name="bell",
            inputs=(
                BlockInput(
                    name="q",
                    effect=ResourceEffect.LIVE_PRESERVED,
                    type_expr=ArrayTypeExpr(element=QubitTypeExpr(), size=2),
                ),
            ),
            body=(),
        )
        prog = Program(
            name="main",
            allocator=AllocatorDecl(name="outer_q", capacity=2),
            block_decls=(bell,),
            body=(
                BlockCall(
                    callee="bell",
                    arg_bindings=(AllocatorArg(name="outer_q"), AllocatorArg(name="outer_q")),
                    out_bindings=(AllocatorArg(name="outer_q"),),
                ),
            ),
        )
        with pytest.raises(GuppyCodegenError, match=r"2 arg_bindings but BlockDecl declares 1"):
            ast_to_guppy(prog)

    def test_unsupported_effect_rejected(self) -> None:
        """`PRODUCED` and `DROPPED` effects are not lowered in Phase 3a.1."""
        bell = BlockDecl(
            name="bell",
            inputs=(
                BlockInput(
                    name="q",
                    effect=ResourceEffect.PRODUCED,
                    type_expr=ArrayTypeExpr(element=QubitTypeExpr(), size=2),
                ),
            ),
            body=(),
        )
        prog = Program(
            name="main",
            allocator=AllocatorDecl(name="outer_q", capacity=2),
            block_decls=(bell,),
            body=(),
        )
        with pytest.raises(GuppyCodegenError, match=r"PRODUCED"):
            ast_to_guppy(prog)

    def test_unsupported_input_type_rejected(self) -> None:
        """Only `array[qubit, N]` inputs are supported in Phase 3a.1."""
        from pecos.slr.ast.nodes import BitTypeExpr

        bell = BlockDecl(
            name="bell",
            inputs=(
                BlockInput(
                    name="c",
                    effect=ResourceEffect.LIVE_PRESERVED,
                    type_expr=ArrayTypeExpr(element=BitTypeExpr(), size=2),
                ),
            ),
            body=(),
        )
        prog = Program(
            name="main",
            allocator=AllocatorDecl(name="outer_q", capacity=2),
            block_decls=(bell,),
            body=(),
        )
        with pytest.raises(GuppyCodegenError, match=r"only array\[qubit, N\] inputs"):
            ast_to_guppy(prog)

    def test_size_mismatch_rejected(self) -> None:
        bell = BlockDecl(
            name="bell",
            inputs=(
                BlockInput(
                    name="q",
                    effect=ResourceEffect.LIVE_PRESERVED,
                    type_expr=ArrayTypeExpr(element=QubitTypeExpr(), size=2),
                ),
            ),
            body=(),
        )
        prog = Program(
            name="main",
            allocator=AllocatorDecl(name="outer_q", capacity=3),
            block_decls=(bell,),
            body=(
                BlockCall(
                    callee="bell",
                    arg_bindings=(AllocatorArg(name="outer_q"),),
                    out_bindings=(AllocatorArg(name="outer_q"),),
                ),
            ),
        )
        with pytest.raises(GuppyCodegenError, match=r"size 3 does not match input 'q' size 2"):
            ast_to_guppy(prog)


class TestBlockCallNonGuppyFlatten:
    """Phase 3a.2: non-Guppy codegens inline BlockCall byte-identical to a flat program."""

    def test_qasm_blockcall_matches_inlined_program(self) -> None:
        """QASM output for a BlockCall program matches the hand-flattened program."""
        from pecos.slr.ast.codegen import generate as codegen_generate

        with_block = _bell_program()

        # The same program with the bell body inlined and no BlockDecl/BlockCall.
        flat = Program(
            name="main",
            allocator=AllocatorDecl(name="outer_q", capacity=2),
            declarations=(RegisterDecl(name="c", size=2),),
            body=(
                PrepareOp(allocator="outer_q"),
                GateOp(gate=GateKind.H, targets=(SlotRef(allocator="outer_q", index=0),)),
                GateOp(
                    gate=GateKind.CX,
                    targets=(
                        SlotRef(allocator="outer_q", index=0),
                        SlotRef(allocator="outer_q", index=1),
                    ),
                ),
                MeasureOp(
                    targets=(
                        SlotRef(allocator="outer_q", index=0),
                        SlotRef(allocator="outer_q", index=1),
                    ),
                    results=(
                        BitRef(register="c", index=0),
                        BitRef(register="c", index=1),
                    ),
                ),
            ),
        )

        assert codegen_generate(with_block, "qasm") == codegen_generate(flat, "qasm")


class TestConvertedQeclibBlocksUseBlockCallPath:
    """Lock in that each Phase 3a.3-converted Steane Block actually goes through
    the BlockCall lowering path -- not silently flattening.

    Goes through `SlrConverter(prog).guppy()` end-to-end (not just `slr_to_ast`)
    so this catches regressions in any production transform between SLR and
    AST (e.g., ParallelOptimizer dropping class identity, Codex 2026-05-15
    review #4).
    """

    def _assert_uses_block_call(self, prog: object, expected_callee_prefix: str) -> None:
        from pecos.slr import SlrConverter

        guppy_src = SlrConverter(prog).guppy()
        # The converted Block must emit its own @guppy def with a name starting
        # with `{class_name_lower}_`. Pre-fix, ParallelOptimizer destroyed class
        # identity and the body was inlined into main(), so no such def was
        # emitted.
        assert (
            f"def {expected_callee_prefix}" in guppy_src
        ), f"expected '@guppy def {expected_callee_prefix}...' in source, got:\n{guppy_src}"

    def test_steane_cx_uses_block_call(self) -> None:
        from pecos.slr import CReg, Main, QReg
        from pecos.slr.qeclib.steane.gates_tq import transversal_tq as steane_tq

        prog = Main(a := QReg("a", 7), b := QReg("b", 7), CReg("c", 14), steane_tq.CX(a, b))
        self._assert_uses_block_call(prog, "cx_")

    def test_steane_cy_uses_block_call(self) -> None:
        from pecos.slr import CReg, Main, QReg
        from pecos.slr.qeclib.steane.gates_tq import transversal_tq as steane_tq

        prog = Main(a := QReg("a", 7), b := QReg("b", 7), CReg("c", 14), steane_tq.CY(a, b))
        self._assert_uses_block_call(prog, "cy_")

    def test_steane_cz_uses_block_call(self) -> None:
        from pecos.slr import CReg, Main, QReg
        from pecos.slr.qeclib.steane.gates_tq import transversal_tq as steane_tq

        prog = Main(a := QReg("a", 7), b := QReg("b", 7), CReg("c", 14), steane_tq.CZ(a, b))
        self._assert_uses_block_call(prog, "cz_")

    def test_steane_logical_x_uses_block_call(self) -> None:
        from pecos.slr import CReg, Main, QReg
        from pecos.slr.qeclib.steane.gates_sq import paulis as steane_paulis

        prog = Main(q := QReg("q", 7), CReg("c", 7), steane_paulis.X(q))
        self._assert_uses_block_call(prog, "x_")

    def test_steane_logical_y_uses_block_call(self) -> None:
        from pecos.slr import CReg, Main, QReg
        from pecos.slr.qeclib.steane.gates_sq import paulis as steane_paulis

        prog = Main(q := QReg("q", 7), CReg("c", 7), steane_paulis.Y(q))
        self._assert_uses_block_call(prog, "y_")

    def test_steane_logical_z_uses_block_call(self) -> None:
        from pecos.slr import CReg, Main, QReg
        from pecos.slr.qeclib.steane.gates_sq import paulis as steane_paulis

        prog = Main(q := QReg("q", 7), CReg("c", 7), steane_paulis.Z(q))
        self._assert_uses_block_call(prog, "z_")

    def test_steane_logical_h_uses_block_call(self) -> None:
        from pecos.slr import CReg, Main, QReg
        from pecos.slr.qeclib.steane.gates_sq import hadamards as steane_h

        prog = Main(q := QReg("q", 7), CReg("c", 7), steane_h.H(q))
        self._assert_uses_block_call(prog, "h_")


class TestConsumedEffect:
    """End-to-end coverage for CONSUMED inputs (Codex 2026-05-15 review caught this gap).

    The Phase 3a.1 validator allows CONSUMED in BlockDecl.inputs, and the
    `_emit_block_call` code path marks the outer slot CONSUMED post-call. But
    no existing test confirmed that a subsequent outer-scope reference raises
    a LinearityError. Add a direct-AST test that pins this behavior.
    """

    def _build_consume_then_reuse_program(self) -> Program:
        """A BlockDecl whose `q` input is CONSUMED + a body that measures it; caller
        attempts to use the slot again afterwards.
        """
        consume = BlockDecl(
            name="consume",
            inputs=(
                BlockInput(
                    name="q",
                    effect=ResourceEffect.CONSUMED,
                    type_expr=ArrayTypeExpr(element=QubitTypeExpr(), size=1),
                ),
            ),
            body=(
                # Measure the consumed input so the BlockDecl body itself is sound
                # (otherwise Guppy linearity inside the BlockDecl would complain).
                MeasureOp(targets=(SlotRef(allocator="q", index=0),)),
            ),
        )
        return Program(
            name="main",
            allocator=AllocatorDecl(name="outer_q", capacity=1),
            declarations=(RegisterDecl(name="c", size=1),),
            block_decls=(consume,),
            body=(
                PrepareOp(allocator="outer_q"),
                BlockCall(
                    callee="consume",
                    arg_bindings=(AllocatorArg(name="outer_q"),),
                    out_bindings=(),
                ),
                # After the call outer_q[0] is CONSUMED; reusing it must raise.
                MeasureOp(
                    targets=(SlotRef(allocator="outer_q", index=0),),
                    results=(BitRef(register="c", index=0),),
                ),
            ),
        )

    def test_outer_reuse_after_consumed_raises(self) -> None:
        from pecos.slr.ast.codegen.guppy_linearity import LinearityError

        prog = self._build_consume_then_reuse_program()
        with pytest.raises(LinearityError, match=r"outer_q\[0\] is consumed"):
            ast_to_guppy(prog)


class TestNestedConvertedBlocks:
    """Nested converted Blocks: an Outer Block whose body contains an Inner Block.

    Codex 2026-05-15 review #1+#2 caught two bugs in nested support:
    - `_substitute_stmt` had no BlockCall branch, so nested calls leaked outer
      allocator names into the parent BlockDecl body.
    - Each sub-converter restarted `_decl_counter` at 0, causing name collisions
      when the same Block class appeared both top-level and nested.
    """

    def _build_nested_program(self) -> object:
        from pecos.slr import Block, CReg, Main, QReg
        from pecos.slr.qeclib import qubit as qb
        from pecos.slr.qeclib.qubit.measures import Measure

        class InnerBlock(Block):
            block_inputs: ClassVar[dict[str, str]] = {"q": "live_preserved"}

            def __init__(self, q: QReg) -> None:
                super().__init__()
                self.q = q
                self.extend(qb.H(q[0]))

        class OuterBlock(Block):
            block_inputs: ClassVar[dict[str, str]] = {"q": "live_preserved"}

            def __init__(self, q: QReg) -> None:
                super().__init__()
                self.q = q
                self.extend(InnerBlock(q))

        return Main(
            outer := QReg("outer", 2),
            c := CReg("c", 2),
            qb.Prep(outer),
            OuterBlock(outer),
            Measure(outer) > c,
        )

    def test_nested_block_call_arg_bindings_substitute_to_parent_param(self) -> None:
        from pecos.slr.ast import slr_to_ast

        prog = self._build_nested_program()
        ast = slr_to_ast(prog)

        # Two BlockDecls, both with unique counter-suffixed names.
        decl_names = [d.name for d in ast.block_decls]
        assert len(decl_names) == 2, decl_names
        assert len(set(decl_names)) == 2, f"decl names not unique: {decl_names}"
        inner_decl = next(d for d in ast.block_decls if d.name.startswith("innerblock_"))
        outer_decl = next(d for d in ast.block_decls if d.name.startswith("outerblock_"))

        # The OUTER decl's body must contain a BlockCall to inner whose
        # arg_bindings use the OUTER's parameter name "q", not the user's
        # outer-scope allocator name "outer".
        nested_calls = [s for s in outer_decl.body if isinstance(s, BlockCall)]
        assert len(nested_calls) == 1, nested_calls
        nested = nested_calls[0]
        assert nested.callee == inner_decl.name
        assert nested.arg_bindings == (AllocatorArg(name="q"),), nested.arg_bindings
        assert nested.out_bindings == (AllocatorArg(name="q"),), nested.out_bindings

    def test_top_level_inner_plus_outer_containing_inner_have_unique_names(self) -> None:
        """Codex review #2: same Block class top-level AND nested must not collide."""
        from pecos.slr import Block, CReg, Main, QReg
        from pecos.slr.ast import slr_to_ast
        from pecos.slr.qeclib import qubit as qb
        from pecos.slr.qeclib.qubit.measures import Measure

        class InnerBlock(Block):
            block_inputs: ClassVar[dict[str, str]] = {"q": "live_preserved"}

            def __init__(self, q: QReg) -> None:
                super().__init__()
                self.q = q
                self.extend(qb.H(q[0]))

        class OuterBlock(Block):
            block_inputs: ClassVar[dict[str, str]] = {"q": "live_preserved"}

            def __init__(self, q: QReg) -> None:
                super().__init__()
                self.q = q
                self.extend(InnerBlock(q))

        prog = Main(
            outer := QReg("outer", 2),
            c := CReg("c", 2),
            qb.Prep(outer),
            InnerBlock(outer),
            OuterBlock(outer),
            Measure(outer) > c,
        )
        ast = slr_to_ast(prog)
        decl_names = [d.name for d in ast.block_decls]
        # Three decls: two innerblock_* (top-level + nested) and one outerblock_*.
        assert len(decl_names) == 3, decl_names
        assert len(set(decl_names)) == 3, f"decl names not unique: {decl_names}"

    def test_nested_block_call_compiles_via_guppy(self) -> None:
        """End-to-end: nested BlockCall lowers via Guppy emitter without error."""
        from pecos.slr import SlrConverter

        prog = self._build_nested_program()
        # SlrConverter.guppy() routes through the AST path; if substitution or
        # counter sharing were broken, this would raise GuppyCodegenError.
        guppy_src = SlrConverter(prog).guppy()
        # Sanity: both function definitions are emitted.
        assert "def innerblock_" in guppy_src
        assert "def outerblock_" in guppy_src
        # The nested call inside outer must reference inner by its hoisted name.
        assert re.search(r"innerblock_\d+\(", guppy_src)


class TestPrettyPrintHandlesBlockNodes:
    """Codex 2026-05-15 fix-pass-4 review: `pretty_print` crashed on any program
    containing a BlockCall because the visitor inherited `default_result()` which
    raises NotImplementedError. Lock in that pretty_print emits both BlockDecls
    and BlockCalls cleanly.
    """

    def test_pretty_print_emits_block_decl_and_block_call(self) -> None:
        from pecos.slr.ast.pretty_print import pretty_print

        decl = BlockDecl(
            name="bell",
            inputs=(
                BlockInput(
                    name="q",
                    effect=ResourceEffect.LIVE_PRESERVED,
                    type_expr=ArrayTypeExpr(element=QubitTypeExpr(), size=2),
                ),
            ),
            body=(GateOp(gate=GateKind.H, targets=(SlotRef(allocator="q", index=0),)),),
        )
        prog = Program(
            name="main",
            allocator=AllocatorDecl(name="outer_q", capacity=2),
            block_decls=(decl,),
            body=(
                BlockCall(
                    callee="bell",
                    arg_bindings=(AllocatorArg(name="outer_q"),),
                    out_bindings=(AllocatorArg(name="outer_q"),),
                ),
            ),
        )
        # Pre-fix this raised NotImplementedError from BaseVisitor.default_result().
        rendered = pretty_print(prog)
        assert 'BlockDecl("bell"' in rendered
        assert "q: array[qubit, 2] @ live_preserved" in rendered
        assert "qb.H(q[0])" in rendered
        assert "BlockCall('bell', outer_q)" in rendered


class TestConvertedBlocksInsideParallel:
    """Codex 2026-05-15 fix-pass-3 review #1: a converted Block inside Parallel(...)
    used to silently flatten because ParallelOptimizer's `_collect_operations`
    splatted the Block's body into the surrounding Parallel, destroying its
    scope boundary. The fix bails out of `_can_optimize_parallel` when any
    direct or transitive child is a converted Block.
    """

    def test_parallel_with_converted_block_preserves_block_call_via_slr_converter(self) -> None:
        from pecos.slr import CReg, Main, QReg, SlrConverter
        from pecos.slr.misc import Parallel
        from pecos.slr.qeclib.qubit.measures import Measure
        from pecos.slr.qeclib.steane.gates_sq.hadamards import H as SteaneH

        prog = Main(
            q := QReg("q", 7),
            c := CReg("c", 7),
            Parallel(SteaneH(q)),
            Measure(q) > c,
        )
        guppy_src = SlrConverter(prog).guppy()
        # Pre-fix, Parallel splatted the Steane H body into 7 individual h() calls
        # in main(), and no `def h_0` was emitted.
        assert "def h_" in guppy_src, f"BlockCall path bypassed by Parallel; source:\n{guppy_src}"
        # And the call site references the hoisted def.
        assert re.search(r"h_\d+\(", guppy_src)


class TestAstOptimizationPreservesBlockDecls:
    """Codex 2026-05-15 fix-pass-3 review #2: AST optimization passes were
    reconstructing Program without `block_decls=`, leaving any contained
    BlockCalls dangling. Lock in `block_decls` survival across each pass.
    """

    def _program_with_block_decl(self) -> Program:
        decl = BlockDecl(
            name="b",
            inputs=(
                BlockInput(
                    name="q",
                    effect=ResourceEffect.LIVE_PRESERVED,
                    type_expr=ArrayTypeExpr(element=QubitTypeExpr(), size=1),
                ),
            ),
            body=(GateOp(gate=GateKind.H, targets=(SlotRef(allocator="q", index=0),)),),
        )
        return Program(
            name="main",
            allocator=AllocatorDecl(name="outer_q", capacity=1),
            block_decls=(decl,),
            body=(
                BlockCall(
                    callee="b",
                    arg_bindings=(AllocatorArg(name="outer_q"),),
                    out_bindings=(AllocatorArg(name="outer_q"),),
                ),
            ),
        )

    def test_gate_cancellation_pass_preserves_block_decls(self) -> None:
        from pecos.slr.ast.optimizations.gate_cancellation import GateCancellationPass

        prog = self._program_with_block_decl()
        result = GateCancellationPass().optimize(prog)
        assert result.program.block_decls == prog.block_decls

    def test_identity_removal_pass_preserves_block_decls(self) -> None:
        from pecos.slr.ast.optimizations.identity_removal import IdentityRemovalPass

        prog = self._program_with_block_decl()
        result = IdentityRemovalPass().optimize(prog)
        assert result.program.block_decls == prog.block_decls

    def test_rotation_merging_pass_preserves_block_decls(self) -> None:
        from pecos.slr.ast.optimizations.rotation_merging import RotationMergingPass

        prog = self._program_with_block_decl()
        result = RotationMergingPass().optimize(prog)
        assert result.program.block_decls == prog.block_decls


class TestDeferredBlockArgRejection:
    """Phase 3a.3 iter 5a scope: only `AllocatorArg` is supported by the
    emitter / flatten paths. The four deferred BlockArg subclasses
    (`SingleQubitArg`, `SingleBitArg`, `QubitBundleArg`, `BitBundleArg`) MUST
    raise cleanly in BOTH the Guppy emitter AND the non-Guppy flatten pass --
    silently inlining a deferred shape would mask user errors (Codex
    2026-05-15 fix-pass-5 review caught this on `out_bindings`).
    """

    def _program_with_deferred_arg(self, *, deferred_in_args: bool, arg_subclass: type) -> Program:
        from pecos.slr.ast.nodes import (
            BitBundleArg,
            BitRef,
            QubitBundleArg,
            SingleBitArg,
            SingleQubitArg,
            SlotRef,
        )

        # Build a representative instance of the deferred subclass.
        if arg_subclass is SingleQubitArg:
            deferred = SingleQubitArg(slot=SlotRef(allocator="outer_q", index=0))
        elif arg_subclass is SingleBitArg:
            deferred = SingleBitArg(bit=BitRef(register="c", index=0))
        elif arg_subclass is QubitBundleArg:
            deferred = QubitBundleArg(slots=(SlotRef(allocator="outer_q", index=0),))
        elif arg_subclass is BitBundleArg:
            deferred = BitBundleArg(bits=(BitRef(register="c", index=0),))
        else:
            msg = f"unsupported subclass {arg_subclass}"
            raise AssertionError(msg)

        decl = BlockDecl(
            name="b",
            inputs=(
                BlockInput(
                    name="q",
                    effect=ResourceEffect.LIVE_PRESERVED,
                    type_expr=ArrayTypeExpr(element=QubitTypeExpr(), size=1),
                ),
            ),
            body=(GateOp(gate=GateKind.H, targets=(SlotRef(allocator="q", index=0),)),),
        )
        if deferred_in_args:
            call = BlockCall(
                callee="b",
                arg_bindings=(deferred,),
                out_bindings=(AllocatorArg(name="outer_q"),),
            )
        else:
            call = BlockCall(
                callee="b",
                arg_bindings=(AllocatorArg(name="outer_q"),),
                out_bindings=(deferred,),
            )
        return Program(
            name="main",
            allocator=AllocatorDecl(name="outer_q", capacity=1),
            declarations=(RegisterDecl(name="c", size=1),),
            block_decls=(decl,),
            body=(call,),
        )

    def test_deferred_arg_bindings_raise_in_flatten(self) -> None:
        from pecos.slr.ast.codegen._block_flatten import flatten_block_calls
        from pecos.slr.ast.nodes import (
            BitBundleArg,
            QubitBundleArg,
            SingleBitArg,
            SingleQubitArg,
        )

        for subclass in (SingleQubitArg, SingleBitArg, QubitBundleArg, BitBundleArg):
            prog = self._program_with_deferred_arg(deferred_in_args=True, arg_subclass=subclass)
            with pytest.raises(NotImplementedError, match=subclass.__name__):
                flatten_block_calls(prog)

    def test_deferred_out_bindings_raise_in_flatten(self) -> None:
        """Codex 2026-05-15 fix-pass-5 review caught: out_bindings used to be
        silently accepted by `_inline_call` even when they were a deferred
        BlockArg shape, while Guppy correctly rejected them.
        """
        from pecos.slr.ast.codegen._block_flatten import flatten_block_calls
        from pecos.slr.ast.nodes import (
            BitBundleArg,
            QubitBundleArg,
            SingleBitArg,
            SingleQubitArg,
        )

        for subclass in (SingleQubitArg, SingleBitArg, QubitBundleArg, BitBundleArg):
            prog = self._program_with_deferred_arg(deferred_in_args=False, arg_subclass=subclass)
            with pytest.raises(NotImplementedError, match=subclass.__name__):
                flatten_block_calls(prog)

    def test_deferred_arg_bindings_raise_in_guppy(self) -> None:
        from pecos.slr.ast.nodes import (
            BitBundleArg,
            QubitBundleArg,
            SingleBitArg,
            SingleQubitArg,
        )

        for subclass in (SingleQubitArg, SingleBitArg, QubitBundleArg, BitBundleArg):
            prog = self._program_with_deferred_arg(deferred_in_args=True, arg_subclass=subclass)
            with pytest.raises(GuppyCodegenError, match=subclass.__name__):
                ast_to_guppy(prog)

    def test_deferred_out_bindings_raise_in_guppy(self) -> None:
        from pecos.slr.ast.nodes import (
            BitBundleArg,
            QubitBundleArg,
            SingleBitArg,
            SingleQubitArg,
        )

        for subclass in (SingleQubitArg, SingleBitArg, QubitBundleArg, BitBundleArg):
            prog = self._program_with_deferred_arg(deferred_in_args=False, arg_subclass=subclass)
            with pytest.raises(GuppyCodegenError, match=subclass.__name__):
                ast_to_guppy(prog)


class TestDuplicateBlockDeclNameValidation:
    """Shared validate_unique_block_decl_names contract (Codex review #3)."""

    def _duplicate_decl_program(self) -> Program:
        decl = BlockDecl(
            name="dup",
            inputs=(
                BlockInput(
                    name="q",
                    effect=ResourceEffect.LIVE_PRESERVED,
                    type_expr=ArrayTypeExpr(element=QubitTypeExpr(), size=2),
                ),
            ),
            body=(),
        )
        return Program(
            name="main",
            allocator=AllocatorDecl(name="outer_q", capacity=2),
            block_decls=(decl, decl),
            body=(),
        )

    def test_guppy_emitter_rejects_duplicate_block_decl_names(self) -> None:
        prog = self._duplicate_decl_program()
        with pytest.raises(ValueError, match=r"Duplicate BlockDecl name 'dup'"):
            ast_to_guppy(prog)

    def test_flatten_pass_rejects_duplicate_block_decl_names(self) -> None:
        from pecos.slr.ast.codegen._block_flatten import flatten_block_calls

        prog = self._duplicate_decl_program()
        with pytest.raises(ValueError, match=r"Duplicate BlockDecl name 'dup'"):
            flatten_block_calls(prog)


class TestBlockBodyStatementSubstitution:
    """Phase 3a.1 substitution must cover every SLR statement type that names allocators.

    Codex 2026-05-15 review caught that PermuteOp (which carries source/target
    register names as strings, not SlotRefs) was silently passed through in
    both `converter._substitute_stmt` and `_block_flatten._substitute`. Lock
    in coverage with a regression test.
    """

    def test_permute_inside_block_inputs_substitutes_in_both_paths(self) -> None:
        from pecos.slr import Block, CReg, Main, QReg
        from pecos.slr.ast import slr_to_ast
        from pecos.slr.ast.codegen._block_flatten import flatten_block_calls
        from pecos.slr.ast.nodes import PermuteOp
        from pecos.slr.misc import Permute
        from pecos.slr.qeclib import qubit as qb
        from pecos.slr.qeclib.qubit.measures import Measure

        class SwapBlock(Block):
            block_inputs: ClassVar[dict[str, str]] = {"q": "live_preserved"}

            def __init__(self, q: QReg) -> None:
                super().__init__()
                self.q = q
                self.extend(Permute([q[0], q[1]], [q[1], q[0]]))

        prog = Main(
            outer := QReg("outer", 2),
            c := CReg("c", 2),
            qb.Prep(outer),
            SwapBlock(outer),
            Measure(outer) > c,
        )
        ast = slr_to_ast(prog)

        # BlockDecl body must reference the INPUT parameter name "q", not the outer
        # allocator name "outer". Pre-fix, sources/targets leaked the outer name.
        assert len(ast.block_decls) == 1
        decl_body = ast.block_decls[0].body
        assert len(decl_body) == 1
        assert isinstance(decl_body[0], PermuteOp)
        assert decl_body[0].sources == ("q[0]", "q[1]")
        assert decl_body[0].targets == ("q[1]", "q[0]")

        # Non-Guppy flatten path must rewrite the OTHER direction: input names
        # back to the outer allocator. Pre-fix, the body still said "q[0]"/"q[1]".
        flat = flatten_block_calls(ast)
        permute_stmts = [s for s in flat.body if isinstance(s, PermuteOp)]
        assert len(permute_stmts) == 1
        assert permute_stmts[0].sources == ("outer[0]", "outer[1]")
        assert permute_stmts[0].targets == ("outer[1]", "outer[0]")

    def test_unparseable_permute_ref_mentioning_mapped_key_raises(self) -> None:
        """Defensive raise (Codex 2026-05-15 review question 3): if a PermuteOp ref
        cannot be parsed by the bare-name / `name[idx]` regex AND the ref textually
        mentions a mapped allocator, raise instead of silently leaking outer names.
        """
        from pecos.slr.ast.codegen._block_flatten import _sub_permute_ref
        from pecos.slr.ast.converter import _substitute_permute_ref

        mapping = {"outer": "q"}
        # Slice-form ref mentioning a mapped allocator -- not produced by SLR today
        # but if the codegen layer ever sees one, it must raise rather than pass
        # the unsubstituted string through.
        with pytest.raises(ValueError, match=r"mapped allocator 'outer'"):
            _substitute_permute_ref("outer[0:2]", mapping)
        with pytest.raises(ValueError, match=r"mapped allocator 'outer'"):
            _sub_permute_ref("outer[0:2]", mapping)

        # Unrelated unparseable ref (not in mapping) passes through unchanged.
        assert _substitute_permute_ref("other[0:2]", mapping) == "other[0:2]"
        assert _sub_permute_ref("other[0:2]", mapping) == "other[0:2]"


class TestSlrBlockInputsWiring:
    """Phase 3a.1: an SLR Block with class-level `block_inputs` lowers to BlockDecl/BlockCall."""

    def test_slr_block_with_inputs_emits_block_decl_and_call(self) -> None:
        """slr_to_ast on a Main containing a Block with `block_inputs` produces a BlockDecl + BlockCall."""
        from pecos.slr import CReg, Main, QReg
        from pecos.slr.ast import slr_to_ast
        from pecos.slr.block import Block
        from pecos.slr.qeclib import qubit as qb
        from pecos.slr.qeclib.qubit.measures import Measure

        class BellBlock(Block):
            block_inputs: ClassVar[dict[str, str]] = {"q": "live_preserved"}

            def __init__(self, q: QReg) -> None:
                super().__init__()
                self.q = q
                self.extend(qb.H(q[0]), qb.CX(q[0], q[1]))

        prog = Main(
            outer_q := QReg("outer_q", 2),
            c := CReg("c", 2),
            qb.Prep(outer_q),
            BellBlock(outer_q),
            Measure(outer_q) > c,
        )
        ast = slr_to_ast(prog)

        assert len(ast.block_decls) == 1
        decl = ast.block_decls[0]
        assert decl.name.startswith("bellblock_")
        assert len(decl.inputs) == 1
        assert decl.inputs[0].name == "q"
        assert decl.inputs[0].effect is ResourceEffect.LIVE_PRESERVED
        # Body uses the parameter name "q" not the outer-scope name "outer_q":
        assert isinstance(decl.body[0], GateOp)
        assert decl.body[0].targets[0].allocator == "q"

        # One BlockCall in the Main body referencing the decl with the outer scope name:
        calls = [s for s in ast.body if isinstance(s, BlockCall)]
        assert len(calls) == 1
        assert calls[0].callee == decl.name
        assert calls[0].arg_bindings == (AllocatorArg(name="outer_q"),)
        assert calls[0].out_bindings == (AllocatorArg(name="outer_q"),)

    def test_slr_block_inputs_end_to_end_selene_bell_correlation(self) -> None:
        """SLR Block with block_inputs -> AST -> Guppy -> Hugr -> Selene: Bell-state correlation."""
        import warnings

        from pecos import Hugr, selene_engine, sim
        from pecos.slr import CReg, Main, QReg, SlrConverter
        from pecos.slr.block import Block
        from pecos.slr.qeclib import qubit as qb
        from pecos.slr.qeclib.qubit.measures import Measure

        class BellBlock(Block):
            block_inputs: ClassVar[dict[str, str]] = {"q": "live_preserved"}

            def __init__(self, q: QReg) -> None:
                super().__init__()
                self.q = q
                self.extend(qb.H(q[0]), qb.CX(q[0], q[1]))

        prog = Main(
            outer_q := QReg("outer_q", 2),
            c := CReg("c", 2),
            qb.Prep(outer_q),
            BellBlock(outer_q),
            Measure(outer_q) > c,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            package = SlrConverter(prog).hugr()
            hugr_bytes = package.to_str().encode("utf-8")
            result = sim(Hugr(hugr_bytes)).classical(selene_engine()).qubits(2).seed(42).run(8)
        raw = result.to_dict() if hasattr(result, "to_dict") else result
        m0 = raw["measurement_0"]
        m1 = raw["measurement_1"]
        for shot, (a, b) in enumerate(zip(m0, m1, strict=True)):
            assert a == b, f"Bell correlation violated at shot {shot}: m0={a} m1={b}"
        assert set(m0) == {0, 1}, f"expected both outcomes across 8 shots, got {set(m0)}"


class TestBlockCallSelene:
    """End-to-end Selene roundtrip: the BlockCall must actually run."""

    def test_bell_block_call_produces_correlated_outcomes(self) -> None:
        """After BlockCall to bell, outer_q is still LIVE; Measure shows |00>/|11>."""
        import importlib.util
        import sys
        import tempfile
        import warnings
        from pathlib import Path

        from pecos import Hugr, selene_engine, sim
        from pecos.slr.ast.codegen.entry_wrapper import build_no_arg_entry_wrapper

        program = _bell_program()
        main_source = ast_to_guppy(program)
        entry_source, _info = build_no_arg_entry_wrapper(program)
        full_source = main_source + entry_source

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            path = Path(f.name)
            f.write(full_source)

        spec = importlib.util.spec_from_file_location(f"_bell_smoke_{path.stem}", path)
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException as exc:
            err = f"Generated Guppy failed to import:\n{full_source}\n---\n{exc}"
            raise AssertionError(err) from exc

        package = module.entry.compile()
        hugr_bytes = package.to_str().encode("utf-8")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = sim(Hugr(hugr_bytes)).classical(selene_engine()).qubits(2).seed(42).run(8)
        raw = result.to_dict() if hasattr(result, "to_dict") else result
        assert isinstance(raw, dict)
        m0 = raw["measurement_0"]
        m1 = raw["measurement_1"]
        assert len(m0) == 8
        assert len(m1) == 8
        # Bell-state correlation: every shot should have m0 == m1.
        for shot, (a, b) in enumerate(zip(m0, m1, strict=True)):
            assert a == b, f"Bell correlation violated at shot {shot}: m0={a} m1={b}"
        # And there's at least one 0 and one 1 across 8 shots (sanity: not always 0).
        assert set(m0) == {0, 1}, f"expected both outcomes across 8 shots, got {set(m0)}"
