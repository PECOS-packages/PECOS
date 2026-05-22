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

"""Smoke tests for BlockDecl / BlockCall.

Builds AST programs directly (without going through SLR) to exercise the new
Guppy emitter codepaths:
- `BlockDecl` lowers to a `@guppy def` top-level function
- `BlockCall` lowers to `array(...)` pack + call + unpack
- `LIVE_PRESERVED` inputs leave outer-scope slots in the LIVE state post-call
- `CONSUMED` inputs leave outer-scope slots in the CONSUMED state post-call
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
    BitTypeExpr,
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
        """`PRODUCED` and `DROPPED` effects are not yet lowered."""
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
        """Only `array[qubit, N]` inputs are supported here."""
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
        with pytest.raises(GuppyCodegenError, match=r"only array\[qubit, N\], bare qubit, and bare bit inputs"):
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
    """Non-Guppy codegens inline BlockCall byte-identical to a flat program."""

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
    """Lock in that each converted Steane Block actually goes through
    the BlockCall lowering path -- not silently flattening.

    Goes through `SlrConverter(prog).guppy()` end-to-end (not just `slr_to_ast`)
    so this catches regressions in any production transform between SLR and
    AST (e.g., ParallelOptimizer dropping class identity).
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
        from pecos.slr import Main, QReg
        from pecos.slr.qeclib.steane.gates_tq import transversal_tq as steane_tq

        prog = Main(a := QReg("a", 7), b := QReg("b", 7), steane_tq.CX(a, b))
        self._assert_uses_block_call(prog, "cx_")

    def test_steane_cy_uses_block_call(self) -> None:
        from pecos.slr import Main, QReg
        from pecos.slr.qeclib.steane.gates_tq import transversal_tq as steane_tq

        prog = Main(a := QReg("a", 7), b := QReg("b", 7), steane_tq.CY(a, b))
        self._assert_uses_block_call(prog, "cy_")

    def test_steane_cz_uses_block_call(self) -> None:
        from pecos.slr import Main, QReg
        from pecos.slr.qeclib.steane.gates_tq import transversal_tq as steane_tq

        prog = Main(a := QReg("a", 7), b := QReg("b", 7), steane_tq.CZ(a, b))
        self._assert_uses_block_call(prog, "cz_")

    def test_steane_logical_x_uses_block_call(self) -> None:
        from pecos.slr import Main, QReg
        from pecos.slr.qeclib.steane.gates_sq import paulis as steane_paulis

        prog = Main(q := QReg("q", 7), steane_paulis.X(q))
        self._assert_uses_block_call(prog, "x_")

    def test_steane_logical_y_uses_block_call(self) -> None:
        from pecos.slr import Main, QReg
        from pecos.slr.qeclib.steane.gates_sq import paulis as steane_paulis

        prog = Main(q := QReg("q", 7), steane_paulis.Y(q))
        self._assert_uses_block_call(prog, "y_")

    def test_steane_logical_z_uses_block_call(self) -> None:
        from pecos.slr import Main, QReg
        from pecos.slr.qeclib.steane.gates_sq import paulis as steane_paulis

        prog = Main(q := QReg("q", 7), steane_paulis.Z(q))
        self._assert_uses_block_call(prog, "z_")

    def test_steane_logical_h_uses_block_call(self) -> None:
        from pecos.slr import Main, QReg
        from pecos.slr.qeclib.steane.gates_sq import hadamards as steane_h

        prog = Main(q := QReg("q", 7), steane_h.H(q))
        self._assert_uses_block_call(prog, "h_")


class TestConsumedEffect:
    """End-to-end coverage for CONSUMED inputs.

    The validator allows CONSUMED in BlockDecl.inputs, and the
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

    Two bugs in nested support were caught:
    - `_substitute_stmt` had no BlockCall branch, so nested calls leaked outer
      allocator names into the parent BlockDecl body.
    - Each sub-converter restarted `_decl_counter` at 0, causing name collisions
      when the same Block class appeared both top-level and nested.
    """

    def _build_nested_program(self) -> object:
        from pecos.slr import Block, CReg, Main, QReg, Return
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
            qb.PZ(outer),
            OuterBlock(outer),
            Measure(outer) > c,
            Return(c),
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
        """Same Block class top-level AND nested must not collide."""
        from pecos.slr import Block, CReg, Main, QReg, Return
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
            qb.PZ(outer),
            InnerBlock(outer),
            OuterBlock(outer),
            Measure(outer) > c,
            Return(c),
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
    """`pretty_print` crashed on any program
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
    """A converted Block inside Parallel(...)
    used to silently flatten because ParallelOptimizer's `_collect_operations`
    splatted the Block's body into the surrounding Parallel, destroying its
    scope boundary. The fix bails out of `_can_optimize_parallel` when any
    direct or transitive child is a converted Block.
    """

    def test_parallel_with_converted_block_preserves_block_call_via_slr_converter(self) -> None:
        from pecos.slr import CReg, Main, QReg, Return, SlrConverter
        from pecos.slr.misc import Parallel
        from pecos.slr.qeclib.qubit.measures import Measure
        from pecos.slr.qeclib.steane.gates_sq.hadamards import H as SteaneH

        prog = Main(
            q := QReg("q", 7),
            c := CReg("c", 7),
            Parallel(SteaneH(q)),
            Measure(q) > c,
            Return(c),
        )
        guppy_src = SlrConverter(prog).guppy()
        # Pre-fix, Parallel splatted the Steane H body into 7 individual h() calls
        # in main(), and no `def h_0` was emitted.
        assert "def h_" in guppy_src, f"BlockCall path bypassed by Parallel; source:\n{guppy_src}"
        # And the call site references the hoisted def.
        assert re.search(r"h_\d+\(", guppy_src)


class TestAstOptimizationPreservesBlockDecls:
    """AST optimization passes were
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


class TestSingleQubitInputSupport:
    """Single-qubit (bare `qubit`) input + `SingleQubitArg`
    at the call site. Validator accepts `QubitTypeExpr` as a BlockInput type;
    emitter renders `name: qubit @ owned` and passes the outer slot's local
    directly (no array wrap); LIVE_PRESERVED rebinds the slot from the
    returned single qubit value.
    """

    def _build_single_qubit_program(self, *, consumed: bool) -> Program:
        from pecos.slr.ast.nodes import SingleQubitArg

        effect = ResourceEffect.CONSUMED if consumed else ResourceEffect.LIVE_PRESERVED
        body: tuple = (GateOp(gate=GateKind.H, targets=(SlotRef(allocator="q", index=0),)),)
        if consumed:
            # Consumed inputs must be measured / discarded inside the body.
            body = (*body, MeasureOp(targets=(SlotRef(allocator="q", index=0),)))
        decl = BlockDecl(
            name="b",
            inputs=(BlockInput(name="q", effect=effect, type_expr=QubitTypeExpr()),),
            body=body,
        )
        out_bindings = () if consumed else (SingleQubitArg(slot=SlotRef(allocator="outer_q", index=1)),)
        # For LIVE_PRESERVED, measure outer_q[1] after the call (it's still live).
        # For CONSUMED, outer_q[1] is consumed by the call; measure a different
        # slot (outer_q[0]) so the linearity tracker stays sound.
        trailing_measure = MeasureOp(
            targets=(SlotRef(allocator="outer_q", index=0 if consumed else 1),),
            results=(BitRef(register="c", index=0),),
        )
        return Program(
            name="main",
            allocator=AllocatorDecl(name="outer_q", capacity=3),
            declarations=(RegisterDecl(name="c", size=1),),
            block_decls=(decl,),
            body=(
                PrepareOp(allocator="outer_q"),
                BlockCall(
                    callee="b",
                    arg_bindings=(SingleQubitArg(slot=SlotRef(allocator="outer_q", index=1)),),
                    out_bindings=out_bindings,
                ),
                trailing_measure,
            ),
        )

    def test_single_qubit_live_preserved_renders_bare_qubit_param(self) -> None:
        source = ast_to_guppy(self._build_single_qubit_program(consumed=False))
        # Block decl: `def b(q: qubit @ owned) -> qubit:`
        assert re.search(r"@guppy\s*\n\s*def b\(q: qubit @ owned\) -> qubit:", source)
        # Body: aliased entry, H on q_0, return q_0
        assert "q_0 = q\n" in source
        assert "q_0 = h(q_0)" in source
        assert "return q_0" in source
        # Call site: pass outer_q_1 directly (no array wrap)
        assert re.search(r"_call_ret_\d+\s*=\s*b\(outer_q_1\)", source)
        # Unpack: rebinds outer_q_1 from the returned single qubit
        assert re.search(r"outer_q_1\s*=\s*_call_ret_\d+", source)

    def test_single_qubit_consumed_no_return_type(self) -> None:
        source = ast_to_guppy(self._build_single_qubit_program(consumed=True))
        # No live_preserved input -> return type is None
        assert re.search(r"@guppy\s*\n\s*def b\(q: qubit @ owned\) -> None:", source)
        # Bare call (no ret_temp assignment) since no live outputs
        assert re.search(r"^\s*b\(outer_q_1\)$", source, re.MULTILINE)

    def test_single_qubit_arg_mismatched_input_type_rejected(self) -> None:
        """A SingleQubitArg paired with an array[qubit, N] input must raise."""
        from pecos.slr.ast.nodes import SingleQubitArg

        decl = BlockDecl(
            name="b",
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
            block_decls=(decl,),
            body=(
                BlockCall(
                    callee="b",
                    arg_bindings=(SingleQubitArg(slot=SlotRef(allocator="outer_q", index=1)),),
                    out_bindings=(SingleQubitArg(slot=SlotRef(allocator="outer_q", index=1)),),
                ),
            ),
        )
        with pytest.raises(GuppyCodegenError, match=r"SingleQubitArg requires a bare qubit input"):
            ast_to_guppy(prog)

    def test_allocator_arg_with_single_qubit_input_rejected(self) -> None:
        """Symmetric: AllocatorArg paired with a bare-qubit input must raise."""
        decl = BlockDecl(
            name="b",
            inputs=(
                BlockInput(
                    name="q",
                    effect=ResourceEffect.LIVE_PRESERVED,
                    type_expr=QubitTypeExpr(),
                ),
            ),
            body=(),
        )
        prog = Program(
            name="main",
            allocator=AllocatorDecl(name="outer_q", capacity=2),
            block_decls=(decl,),
            body=(
                BlockCall(
                    callee="b",
                    arg_bindings=(AllocatorArg(name="outer_q"),),
                    out_bindings=(AllocatorArg(name="outer_q"),),
                ),
            ),
        )
        with pytest.raises(GuppyCodegenError, match=r"AllocatorArg requires an array\[qubit, N\] input"):
            ast_to_guppy(prog)

    def test_single_qubit_mismatched_arg_out_slot_rejected(self) -> None:
        """A LIVE_PRESERVED single-qubit input
        whose `arg_binding` and `out_binding` reference DIFFERENT outer slots
        used to produce invalid Guppy (set_live() overwriting a never-consumed
        slot). The emitter must reject this with a clean GuppyCodegenError.
        """
        from pecos.slr.ast.nodes import SingleQubitArg

        decl = BlockDecl(
            name="b",
            inputs=(BlockInput(name="q", effect=ResourceEffect.LIVE_PRESERVED, type_expr=QubitTypeExpr()),),
            body=(),
        )
        prog = Program(
            name="main",
            allocator=AllocatorDecl(name="outer_q", capacity=3),
            block_decls=(decl,),
            body=(
                BlockCall(
                    callee="b",
                    arg_bindings=(SingleQubitArg(slot=SlotRef(allocator="outer_q", index=1)),),
                    out_bindings=(SingleQubitArg(slot=SlotRef(allocator="outer_q", index=2)),),
                ),
            ),
        )
        with pytest.raises(GuppyCodegenError, match=r"must use an identical arg_binding and out_binding"):
            ast_to_guppy(prog)

    def test_allocator_mismatched_arg_out_name_rejected(self) -> None:
        """Symmetric: AllocatorArg arg_binding != out_binding name for a
        LIVE_PRESERVED input must also raise (same bug class).
        """
        decl = BlockDecl(
            name="b",
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
            declarations=(AllocatorDecl(name="other_q", capacity=2),),
            block_decls=(decl,),
            body=(
                BlockCall(
                    callee="b",
                    arg_bindings=(AllocatorArg(name="outer_q"),),
                    out_bindings=(AllocatorArg(name="other_q"),),
                ),
            ),
        )
        with pytest.raises(GuppyCodegenError, match=r"must use an identical arg_binding and out_binding"):
            ast_to_guppy(prog)

    def test_single_qubit_slot_index_out_of_bounds_rejected(self) -> None:
        from pecos.slr.ast.nodes import SingleQubitArg

        decl = BlockDecl(
            name="b",
            inputs=(BlockInput(name="q", effect=ResourceEffect.LIVE_PRESERVED, type_expr=QubitTypeExpr()),),
            body=(),
        )
        prog = Program(
            name="main",
            allocator=AllocatorDecl(name="outer_q", capacity=2),  # only indices 0..1
            block_decls=(decl,),
            body=(
                BlockCall(
                    callee="b",
                    arg_bindings=(SingleQubitArg(slot=SlotRef(allocator="outer_q", index=5)),),
                    out_bindings=(SingleQubitArg(slot=SlotRef(allocator="outer_q", index=5)),),
                ),
            ),
        )
        with pytest.raises(GuppyCodegenError, match=r"slot index 5 out of bounds"):
            ast_to_guppy(prog)


class TestSingleBitInputSupport:
    """Single classical-bit (bare `BitTypeExpr`) input +
    `SingleBitArg` at the call site. The bit is modeled as an
    `array[bool, 1] @ owned` write-back proxy: the callee mutates `name[0]`,
    returns the array, and the caller writes it back into the outer CReg bit.
    """

    def _build_single_bit_program(self) -> Program:
        from pecos.slr.ast.nodes import SingleBitArg, SingleQubitArg

        # Block: measure a borrowed qubit into a single-bit write-back input.
        decl = BlockDecl(
            name="b",
            inputs=(
                BlockInput(name="a", effect=ResourceEffect.CONSUMED, type_expr=QubitTypeExpr()),
                BlockInput(name="out", effect=ResourceEffect.LIVE_PRESERVED, type_expr=BitTypeExpr()),
            ),
            body=(
                GateOp(gate=GateKind.H, targets=(SlotRef(allocator="a", index=0),)),
                MeasureOp(
                    targets=(SlotRef(allocator="a", index=0),),
                    results=(BitRef(register="out", index=0),),
                ),
            ),
        )
        return Program(
            name="main",
            allocator=AllocatorDecl(name="outer_q", capacity=2),
            declarations=(RegisterDecl(name="c", size=2),),
            block_decls=(decl,),
            body=(
                PrepareOp(allocator="outer_q"),
                BlockCall(
                    callee="b",
                    arg_bindings=(
                        SingleQubitArg(slot=SlotRef(allocator="outer_q", index=0)),
                        SingleBitArg(bit=BitRef(register="c", index=1)),
                    ),
                    out_bindings=(SingleBitArg(bit=BitRef(register="c", index=1)),),
                ),
                MeasureOp(
                    targets=(SlotRef(allocator="outer_q", index=1),),
                    results=(BitRef(register="c", index=0),),
                ),
            ),
        )

    def test_single_bit_renders_array_bool_proxy(self) -> None:
        source = ast_to_guppy(self._build_single_bit_program())
        # Param uses the array[bool, 1] write-back proxy.
        assert re.search(r"def b\(a: qubit @ owned, out: array\[bool, 1\] @ owned\) -> array\[bool, 1\]:", source)
        # Body writes the measurement into out[0] and returns the array.
        assert "out[0] = measure(a_0)" in source
        assert "return out" in source
        # Call site wraps the outer CReg bit, then writes it back.
        assert re.search(r"_call_ret_\d+\s*=\s*b\(outer_q_0, array\(c\[1\]\)\)", source)
        assert re.search(r"c\[1\]\s*=\s*_call_ret_\d+\[0\]", source)

    def test_single_bit_must_be_live_preserved(self) -> None:
        decl = BlockDecl(
            name="b",
            inputs=(BlockInput(name="out", effect=ResourceEffect.CONSUMED, type_expr=BitTypeExpr()),),
            body=(),
        )
        prog = Program(
            name="main",
            allocator=AllocatorDecl(name="outer_q", capacity=1),
            block_decls=(decl,),
            body=(),
        )
        with pytest.raises(GuppyCodegenError, match=r"bare bit inputs\s+must be LIVE_PRESERVED"):
            ast_to_guppy(prog)

    def test_single_bit_arg_mismatched_input_type_rejected(self) -> None:
        from pecos.slr.ast.nodes import SingleBitArg

        decl = BlockDecl(
            name="b",
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
            declarations=(RegisterDecl(name="c", size=1),),
            block_decls=(decl,),
            body=(
                BlockCall(
                    callee="b",
                    arg_bindings=(SingleBitArg(bit=BitRef(register="c", index=0)),),
                    out_bindings=(SingleBitArg(bit=BitRef(register="c", index=0)),),
                ),
            ),
        )
        with pytest.raises(GuppyCodegenError, match=r"SingleBitArg requires a bare bit input"):
            ast_to_guppy(prog)

    def test_single_bit_index_out_of_bounds_rejected(self) -> None:
        from pecos.slr.ast.nodes import SingleBitArg

        decl = BlockDecl(
            name="b",
            inputs=(BlockInput(name="out", effect=ResourceEffect.LIVE_PRESERVED, type_expr=BitTypeExpr()),),
            body=(),
        )
        prog = Program(
            name="main",
            allocator=AllocatorDecl(name="outer_q", capacity=1),
            declarations=(RegisterDecl(name="c", size=2),),  # indices 0..1
            block_decls=(decl,),
            body=(
                BlockCall(
                    callee="b",
                    arg_bindings=(SingleBitArg(bit=BitRef(register="c", index=9)),),
                    out_bindings=(SingleBitArg(bit=BitRef(register="c", index=9)),),
                ),
            ),
        )
        with pytest.raises(GuppyCodegenError, match=r"bit index 9 out of bounds"):
            ast_to_guppy(prog)


class TestQubitBundleInputSupport:
    """A single `array[qubit, N]` BlockInput bound at the
    call site to a non-contiguous bundle of N arbitrary outer slots via
    `QubitBundleArg(slots=(...))`. The BlockDecl side is unchanged from the
    AllocatorArg case -- only the caller's slot-bundling differs.
    """

    def _build_bundle_program(self) -> Program:
        from pecos.slr.ast.nodes import QubitBundleArg

        # Block: H on q[0], CX(q[0], q[1]) over a 2-qubit array input.
        decl = BlockDecl(
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
                    targets=(SlotRef(allocator="q", index=0), SlotRef(allocator="q", index=1)),
                ),
            ),
        )
        # Bundle picks non-contiguous slots a[2] and b[0] across two allocators.
        bundle = QubitBundleArg(
            slots=(SlotRef(allocator="a", index=2), SlotRef(allocator="b", index=0)),
        )
        return Program(
            name="main",
            allocator=AllocatorDecl(name="a", capacity=3),
            declarations=(AllocatorDecl(name="b", capacity=2), RegisterDecl(name="c", size=2)),
            block_decls=(decl,),
            body=(
                PrepareOp(allocator="a"),
                PrepareOp(allocator="b"),
                BlockCall(callee="bell", arg_bindings=(bundle,), out_bindings=(bundle,)),
                MeasureOp(
                    targets=(SlotRef(allocator="a", index=2), SlotRef(allocator="b", index=0)),
                    results=(BitRef(register="c", index=0), BitRef(register="c", index=1)),
                ),
            ),
        )

    def test_qubit_bundle_packs_and_unpacks_arbitrary_slots(self) -> None:
        source = ast_to_guppy(self._build_bundle_program())
        # Call site packs the two non-contiguous slot locals into one array.
        assert re.search(r"_call_ret_\d+\s*=\s*bell\(array\(a_2, b_0\)\)", source)
        # Return destructures back into the SAME slots' canonical locals.
        assert re.search(r"a_2, b_0\s*=\s*_call_ret_\d+", source)
        # Downstream measure sees the rebound slots.
        assert "c[0] = measure(a_2)" in source
        assert "c[1] = measure(b_0)" in source

    def test_qubit_bundle_end_to_end_selene_bell_correlation(self) -> None:
        """Compile + run the cross-allocator bundle program through Selene.

        The earlier support test was string-shape only; the iter-5b r1
        blocker proved a string can look right while the
        Guppy fails its own linearity. This compiles the generated Guppy via
        the entry wrapper and pins the seeded Selene records (the bundled
        slots a[2] and b[0] form a Bell pair, so the two measurements are
        perfectly correlated per shot).

        Note: this is a *compile + behavior* gate, not an unpack-order gate.
        Bell measurements are symmetric, so a swapped bundle unpack order
        would still pass here. `test_qubit_bundle_asymmetric_unpack_order`
        below pins unpack order with an asymmetric bundle.
        """
        import importlib.util
        import sys
        import tempfile
        import warnings
        from pathlib import Path

        from pecos import Hugr, selene_engine, sim
        from pecos.slr.ast.codegen.entry_wrapper import build_no_arg_entry_wrapper

        program = self._build_bundle_program()
        main_source = ast_to_guppy(program)
        entry_source, _info = build_no_arg_entry_wrapper(program)
        full_source = main_source + entry_source

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            path = Path(f.name)
            f.write(full_source)

        spec = importlib.util.spec_from_file_location(f"_bundle_smoke_{path.stem}", path)
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
            result = sim(Hugr(hugr_bytes)).classical(selene_engine()).qubits(5).seed(42).run(8)
        raw = result.to_dict() if hasattr(result, "to_dict") else result
        # Empirical probe 2026-05-15:
        assert raw == {
            "measurement_0": [1, 0, 0, 1, 0, 0, 0, 1],
            "measurement_1": [1, 0, 0, 1, 0, 0, 0, 1],
        }, raw
        # Bell correlation: bundled slots a[2] and b[0] measure identically.
        assert raw["measurement_0"] == raw["measurement_1"], raw

    def test_qubit_bundle_asymmetric_unpack_order(self) -> None:
        """Pin bundle unpack ORDER with an asymmetric program.

        The Bell-correlation test is symmetric, so a swapped bundle unpack
        (`b_0, a_2 = ret` instead of
        `a_2, b_0 = ret`) still passes it. Here the block applies X to q[0]
        ONLY, so the two bundled slots end in DIFFERENT states: a[2] (<- q[0],
        X'd) measures 1, b[0] (<- q[1], untouched) measures 0. A swapped
        unpack would flip both records, failing this test.
        """
        import importlib.util
        import sys
        import tempfile
        import warnings
        from pathlib import Path

        from pecos import Hugr, selene_engine, sim
        from pecos.slr.ast.codegen.entry_wrapper import build_no_arg_entry_wrapper
        from pecos.slr.ast.nodes import QubitBundleArg

        decl = BlockDecl(
            name="asym",
            inputs=(
                BlockInput(
                    name="q",
                    effect=ResourceEffect.LIVE_PRESERVED,
                    type_expr=ArrayTypeExpr(element=QubitTypeExpr(), size=2),
                ),
            ),
            # X on q[0] ONLY -- breaks the symmetry between the two bundled slots.
            body=(GateOp(gate=GateKind.X, targets=(SlotRef(allocator="q", index=0),)),),
        )
        bundle = QubitBundleArg(
            slots=(SlotRef(allocator="a", index=2), SlotRef(allocator="b", index=0)),
        )
        program = Program(
            name="main",
            allocator=AllocatorDecl(name="a", capacity=3),
            declarations=(AllocatorDecl(name="b", capacity=2), RegisterDecl(name="c", size=2)),
            block_decls=(decl,),
            body=(
                PrepareOp(allocator="a"),
                PrepareOp(allocator="b"),
                BlockCall(callee="asym", arg_bindings=(bundle,), out_bindings=(bundle,)),
                MeasureOp(
                    targets=(SlotRef(allocator="a", index=2), SlotRef(allocator="b", index=0)),
                    results=(BitRef(register="c", index=0), BitRef(register="c", index=1)),
                ),
            ),
        )
        main_source = ast_to_guppy(program)
        entry_source, _info = build_no_arg_entry_wrapper(program)
        full_source = main_source + entry_source

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            path = Path(f.name)
            f.write(full_source)

        spec = importlib.util.spec_from_file_location(f"_bundle_asym_{path.stem}", path)
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
            result = sim(Hugr(hugr_bytes)).classical(selene_engine()).qubits(5).seed(42).run(4)
        raw = result.to_dict() if hasattr(result, "to_dict") else result
        # Empirical probe 2026-05-15: a[2] <- X'd q[0] -> 1; b[0] <- untouched
        # q[1] -> 0. Swapped unpack order would yield the inverse.
        assert raw == {
            "measurement_0": [1, 1, 1, 1],
            "measurement_1": [0, 0, 0, 0],
        }, raw

    def test_qubit_bundle_cross_input_alias_rejected_pre_consume(self) -> None:
        """A slot referenced by two distinct
        quantum arg_bindings must raise a clean GuppyCodegenError
        (pre-consume), not a mid-Phase-2 LinearityError with the tracker
        half-mutated.
        """
        from pecos.slr.ast.nodes import QubitBundleArg, SingleQubitArg

        decl = BlockDecl(
            name="b",
            inputs=(
                BlockInput(name="x", effect=ResourceEffect.CONSUMED, type_expr=QubitTypeExpr()),
                BlockInput(
                    name="ys",
                    effect=ResourceEffect.LIVE_PRESERVED,
                    type_expr=ArrayTypeExpr(element=QubitTypeExpr(), size=2),
                ),
            ),
            body=(MeasureOp(targets=(SlotRef(allocator="x", index=0),)),),
        )
        # outer_q[0] is bound to BOTH input x (single qubit) and the bundle ys.
        prog = Program(
            name="main",
            allocator=AllocatorDecl(name="outer_q", capacity=3),
            block_decls=(decl,),
            body=(
                PrepareOp(allocator="outer_q"),
                BlockCall(
                    callee="b",
                    arg_bindings=(
                        SingleQubitArg(slot=SlotRef(allocator="outer_q", index=0)),
                        QubitBundleArg(
                            slots=(
                                SlotRef(allocator="outer_q", index=0),
                                SlotRef(allocator="outer_q", index=1),
                            ),
                        ),
                    ),
                    out_bindings=(
                        QubitBundleArg(
                            slots=(
                                SlotRef(allocator="outer_q", index=0),
                                SlotRef(allocator="outer_q", index=1),
                            ),
                        ),
                    ),
                ),
            ),
        )
        with pytest.raises(GuppyCodegenError, match=r"referenced by more than one arg_binding"):
            ast_to_guppy(prog)

    def test_qubit_bundle_size_mismatch_rejected(self) -> None:
        from pecos.slr.ast.nodes import QubitBundleArg

        decl = BlockDecl(
            name="b",
            inputs=(
                BlockInput(
                    name="q",
                    effect=ResourceEffect.LIVE_PRESERVED,
                    type_expr=ArrayTypeExpr(element=QubitTypeExpr(), size=3),
                ),
            ),
            body=(),
        )
        prog = Program(
            name="main",
            allocator=AllocatorDecl(name="outer_q", capacity=3),
            block_decls=(decl,),
            body=(
                BlockCall(
                    callee="b",
                    arg_bindings=(QubitBundleArg(slots=(SlotRef(allocator="outer_q", index=0),)),),
                    out_bindings=(QubitBundleArg(slots=(SlotRef(allocator="outer_q", index=0),)),),
                ),
            ),
        )
        with pytest.raises(GuppyCodegenError, match=r"has 1 slots but input 'q' expects 3"):
            ast_to_guppy(prog)

    def test_qubit_bundle_duplicate_slot_rejected(self) -> None:
        from pecos.slr.ast.nodes import QubitBundleArg

        decl = BlockDecl(
            name="b",
            inputs=(
                BlockInput(
                    name="q",
                    effect=ResourceEffect.LIVE_PRESERVED,
                    type_expr=ArrayTypeExpr(element=QubitTypeExpr(), size=2),
                ),
            ),
            body=(),
        )
        dup = QubitBundleArg(
            slots=(SlotRef(allocator="outer_q", index=1), SlotRef(allocator="outer_q", index=1)),
        )
        prog = Program(
            name="main",
            allocator=AllocatorDecl(name="outer_q", capacity=3),
            block_decls=(decl,),
            body=(BlockCall(callee="b", arg_bindings=(dup,), out_bindings=(dup,)),),
        )
        with pytest.raises(GuppyCodegenError, match=r"more than once \(a qubit cannot be passed twice\)"):
            ast_to_guppy(prog)

    def test_qubit_bundle_out_of_bounds_slot_rejected(self) -> None:
        from pecos.slr.ast.nodes import QubitBundleArg

        decl = BlockDecl(
            name="b",
            inputs=(
                BlockInput(
                    name="q",
                    effect=ResourceEffect.LIVE_PRESERVED,
                    type_expr=ArrayTypeExpr(element=QubitTypeExpr(), size=1),
                ),
            ),
            body=(),
        )
        bad = QubitBundleArg(slots=(SlotRef(allocator="outer_q", index=9),))
        prog = Program(
            name="main",
            allocator=AllocatorDecl(name="outer_q", capacity=2),
            block_decls=(decl,),
            body=(BlockCall(callee="b", arg_bindings=(bad,), out_bindings=(bad,)),),
        )
        with pytest.raises(GuppyCodegenError, match=r"bundle slot index 9 out of bounds"):
            ast_to_guppy(prog)


class TestDeferredBlockArgRejection:
    """Scope:
    - `AllocatorArg`, `SingleQubitArg`, `SingleBitArg`, `QubitBundleArg` are
      now supported in BOTH the Guppy emitter AND the non-Guppy flatten path.
      The former `test_<shape>_arg_rejected_in_flatten_pass` lock-ins are
      flipped into `test_<shape>_arg_flatten_inlines` support tests below
      (they pin that flatten rewrites param refs -> outer refs, no
      NotImplementedError).
    - `BitBundleArg` is the ONLY still-deferred shape: it MUST raise cleanly
      in BOTH paths -- silently inlining a deferred shape would mask user
      errors (this silent-fallback family was caught repeatedly).
    """

    def test_qubit_bundle_arg_flatten_inlines(self) -> None:
        """5e.2: QubitBundleArg now inlines in flatten -- param `q[0]` rewrites
        to the bundled outer slot `outer_q[0]`.
        """
        from pecos.slr.ast.codegen._block_flatten import flatten_block_calls
        from pecos.slr.ast.nodes import QubitBundleArg

        decl = BlockDecl(
            name="b",
            inputs=(
                BlockInput(
                    name="q",
                    effect=ResourceEffect.LIVE_PRESERVED,
                    type_expr=ArrayTypeExpr(element=QubitTypeExpr(), size=2),
                ),
            ),
            body=(GateOp(gate=GateKind.H, targets=(SlotRef(allocator="q", index=1),)),),
        )
        bundle = QubitBundleArg(
            slots=(
                SlotRef(allocator="outer_q", index=0),
                SlotRef(allocator="outer_q", index=2),
            ),
        )
        prog = Program(
            name="main",
            allocator=AllocatorDecl(name="outer_q", capacity=3),
            block_decls=(decl,),
            body=(BlockCall(callee="b", arg_bindings=(bundle,), out_bindings=(bundle,)),),
        )
        flat = flatten_block_calls(prog)
        # q[1] (param) -> bundle slot index 1 -> outer_q[2].
        gates = [s for s in flat.body if isinstance(s, GateOp)]
        assert len(gates) == 1
        assert gates[0].targets == (SlotRef(allocator="outer_q", index=2),)
        assert flat.block_decls == ()

    def test_single_bit_arg_flatten_inlines(self) -> None:
        """5e.2: SingleBitArg now inlines in flatten -- param bit `out[0]`
        rewrites to the outer CReg bit `c[0]`.
        """
        from pecos.slr.ast.codegen._block_flatten import flatten_block_calls
        from pecos.slr.ast.nodes import SingleBitArg, SingleQubitArg

        decl = BlockDecl(
            name="b",
            inputs=(
                BlockInput(name="a", effect=ResourceEffect.CONSUMED, type_expr=QubitTypeExpr()),
                BlockInput(name="out", effect=ResourceEffect.LIVE_PRESERVED, type_expr=BitTypeExpr()),
            ),
            body=(
                MeasureOp(
                    targets=(SlotRef(allocator="a", index=0),),
                    results=(BitRef(register="out", index=0),),
                ),
            ),
        )
        prog = Program(
            name="main",
            allocator=AllocatorDecl(name="outer_q", capacity=1),
            declarations=(RegisterDecl(name="c", size=1),),
            block_decls=(decl,),
            body=(
                BlockCall(
                    callee="b",
                    arg_bindings=(
                        SingleQubitArg(slot=SlotRef(allocator="outer_q", index=0)),
                        SingleBitArg(bit=BitRef(register="c", index=0)),
                    ),
                    out_bindings=(SingleBitArg(bit=BitRef(register="c", index=0)),),
                ),
            ),
        )
        flat = flatten_block_calls(prog)
        meas = [s for s in flat.body if isinstance(s, MeasureOp)]
        assert len(meas) == 1
        assert meas[0].targets == (SlotRef(allocator="outer_q", index=0),)
        assert meas[0].results == (BitRef(register="c", index=0),)
        assert flat.block_decls == ()

    def test_single_qubit_arg_flatten_inlines(self) -> None:
        """5e.2: SingleQubitArg now inlines in flatten -- param `q[0]` rewrites
        to the outer slot `outer_q[1]`.
        """
        from pecos.slr.ast.codegen._block_flatten import flatten_block_calls
        from pecos.slr.ast.nodes import SingleQubitArg

        decl = BlockDecl(
            name="b",
            inputs=(BlockInput(name="q", effect=ResourceEffect.LIVE_PRESERVED, type_expr=QubitTypeExpr()),),
            body=(GateOp(gate=GateKind.H, targets=(SlotRef(allocator="q", index=0),)),),
        )
        sq = SingleQubitArg(slot=SlotRef(allocator="outer_q", index=1))
        prog = Program(
            name="main",
            allocator=AllocatorDecl(name="outer_q", capacity=3),
            block_decls=(decl,),
            body=(BlockCall(callee="b", arg_bindings=(sq,), out_bindings=(sq,)),),
        )
        flat = flatten_block_calls(prog)
        gates = [s for s in flat.body if isinstance(s, GateOp)]
        assert len(gates) == 1
        assert gates[0].targets == (SlotRef(allocator="outer_q", index=1),)
        assert flat.block_decls == ()

    def _program_with_deferred_arg(self, *, deferred_in_args: bool, arg_subclass: type) -> Program:
        from pecos.slr.ast.nodes import BitBundleArg, BitRef

        # Build a representative instance of the deferred subclass.
        if arg_subclass is BitBundleArg:
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
        from pecos.slr.ast.nodes import BitBundleArg

        prog = self._program_with_deferred_arg(deferred_in_args=True, arg_subclass=BitBundleArg)
        with pytest.raises(NotImplementedError, match="BitBundleArg"):
            flatten_block_calls(prog)

    def test_deferred_out_bindings_raise_in_flatten(self) -> None:
        """Regression: out_bindings used to be
        silently accepted by `_inline_call` even when they were a deferred
        BlockArg shape, while Guppy correctly rejected them.
        """
        from pecos.slr.ast.codegen._block_flatten import flatten_block_calls
        from pecos.slr.ast.nodes import BitBundleArg

        prog = self._program_with_deferred_arg(deferred_in_args=False, arg_subclass=BitBundleArg)
        with pytest.raises(NotImplementedError, match="BitBundleArg"):
            flatten_block_calls(prog)

    def test_deferred_arg_bindings_raise_in_guppy(self) -> None:
        from pecos.slr.ast.nodes import BitBundleArg

        prog = self._program_with_deferred_arg(deferred_in_args=True, arg_subclass=BitBundleArg)
        with pytest.raises(GuppyCodegenError, match="BitBundleArg"):
            ast_to_guppy(prog)

    def test_deferred_out_bindings_raise_in_guppy(self) -> None:
        from pecos.slr.ast.nodes import BitBundleArg

        prog = self._program_with_deferred_arg(deferred_in_args=False, arg_subclass=BitBundleArg)
        with pytest.raises(GuppyCodegenError, match="BitBundleArg"):
            ast_to_guppy(prog)


class TestDuplicateBlockDeclNameValidation:
    """Shared validate_unique_block_decl_names contract."""

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
    """Substitution must cover every SLR statement type that names allocators.

    PermuteOp (which carries source/target register names as strings, not
    SlotRefs) was silently passed through in
    both `converter._substitute_stmt` and `_block_flatten._substitute`. Lock
    in coverage with a regression test.
    """

    def test_permute_inside_block_inputs_substitutes_in_both_paths(self) -> None:
        from pecos.slr import Block, CReg, Main, QReg, Return
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
            qb.PZ(outer),
            SwapBlock(outer),
            Measure(outer) > c,
            Return(c),
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

    def test_unparseable_permute_ref_mentioning_partial_name_raises(self) -> None:
        """Defensive raise (5e.1 shared substitution): if a
        PermuteOp ref cannot be parsed by the bare-name / `name[idx]` regex AND
        the ref textually mentions a partially-bound name, raise instead of
        silently leaking. Unrelated unparseable refs pass through unchanged.
        """
        from pecos.slr.ast._block_substitution import (
            BodyRemap,
            BodySubstitutionError,
            _sub_permute_ref,
        )

        remap = BodyRemap()
        # add_slot marks `outer` as partially bound.
        remap.add_slot(("outer", 0), ("q", 0))

        with pytest.raises(BodySubstitutionError, match=r"base name 'outer' is partially bound"):
            _sub_permute_ref("outer[0:2]", remap)

        # Unrelated unparseable refs whose BASE name is not partial pass
        # through -- including the substring-trap `souter`.
        assert _sub_permute_ref("other[0:2]", remap) == "other[0:2]"
        assert _sub_permute_ref("souter[0:2]", remap) == "souter[0:2]"


class TestSlrBlockInputsWiring:
    """An SLR Block with class-level `block_inputs` lowers to BlockDecl/BlockCall."""

    def test_slr_block_with_inputs_emits_block_decl_and_call(self) -> None:
        """slr_to_ast on a Main containing a Block with `block_inputs` produces a BlockDecl + BlockCall."""
        from pecos.slr import CReg, Main, QReg, Return
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
            qb.PZ(outer_q),
            BellBlock(outer_q),
            Measure(outer_q) > c,
            Return(c),
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
        from pecos.slr import CReg, Main, QReg, Return, SlrConverter
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
            qb.PZ(outer_q),
            BellBlock(outer_q),
            Measure(outer_q) > c,
            Return(c),
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


class TestSlrBlockArgShapeDetectionViaConverter:
    """An SLR `Block` subclass whose `block_inputs`
    bind a single `Qubit`, a `list[Qubit]` bundle (same- or cross-QReg), or
    a single `Bit` must drive the full real pipeline:

      SLR Block -> slr_to_ast (`_convert_block_call` shape detection, 5e.2a)
                -> SlrConverter.guppy()  (emitter)
                -> SlrConverter.qasm()   (flatten_block_calls inline, 5e.2b)

    These differ from `TestSingleQubitInputSupport` et al. (which build the
    AST `Program` directly) by exercising the SLR var -> typed BlockArg
    detection AND the bidirectional shared substitution through the public
    `SlrConverter` surface -- the iter-3 "silent flatten" bug class would
    slip past an AST-only test.
    """

    def test_single_qubit_input_detected_and_inlined(self) -> None:
        from pecos.slr import CReg, Main, QReg, Return, SlrConverter
        from pecos.slr.ast import slr_to_ast
        from pecos.slr.ast.nodes import SingleQubitArg, SlotRef
        from pecos.slr.block import Block
        from pecos.slr.qeclib import qubit as qb
        from pecos.slr.qeclib.qubit.measures import Measure

        class SqBlock(Block):
            block_inputs: ClassVar[dict[str, str]] = {"q": "live_preserved"}

            def __init__(self, q) -> None:
                super().__init__()
                self.q = q
                self.extend(qb.H(q))

        prog = Main(
            outer_q := QReg("outer_q", 1),
            c := CReg("c", 1),
            qb.PZ(outer_q),
            SqBlock(outer_q[0]),
            Measure(outer_q) > c,
            Return(c),
        )

        ast = slr_to_ast(prog)
        calls = [s for s in ast.body if isinstance(s, BlockCall)]
        assert len(calls) == 1
        assert calls[0].arg_bindings == (SingleQubitArg(slot=SlotRef(allocator="outer_q", index=0)),)

        guppy_src = SlrConverter(prog).guppy()
        assert re.search(r"def sqblock_\d+\(q: qubit @ owned\) -> qubit:", guppy_src)

        qasm_src = SlrConverter(prog).qasm()
        # Flatten rewrote the param `q[0]` back to the outer slot.
        assert "h outer_q[0];" in qasm_src
        assert "block" not in qasm_src.lower()

    def test_same_qreg_bundle_detected_and_inlined(self) -> None:
        from pecos.slr import CReg, Main, QReg, Return, SlrConverter
        from pecos.slr.ast import slr_to_ast
        from pecos.slr.ast.nodes import QubitBundleArg, SlotRef
        from pecos.slr.block import Block
        from pecos.slr.qeclib import qubit as qb
        from pecos.slr.qeclib.qubit.measures import Measure

        class BundleBlock(Block):
            block_inputs: ClassVar[dict[str, str]] = {"d": "live_preserved"}

            def __init__(self, d) -> None:
                super().__init__()
                self.d = d
                self.extend(qb.H(d[0]), qb.CX(d[0], d[1]))

        prog = Main(
            outer_q := QReg("outer_q", 3),
            c := CReg("c", 3),
            qb.PZ(outer_q),
            BundleBlock([outer_q[0], outer_q[2]]),
            Measure(outer_q) > c,
            Return(c),
        )

        ast = slr_to_ast(prog)
        calls = [s for s in ast.body if isinstance(s, BlockCall)]
        assert calls[0].arg_bindings == (
            QubitBundleArg(
                slots=(
                    SlotRef(allocator="outer_q", index=0),
                    SlotRef(allocator="outer_q", index=2),
                ),
            ),
        )

        qasm_src = SlrConverter(prog).qasm()
        # Bundle slot 0 -> outer_q[0], slot 1 -> outer_q[2] (non-contiguous).
        assert "h outer_q[0];" in qasm_src
        assert "cx outer_q[0], outer_q[2];" in qasm_src
        assert re.search(r"def bundleblock_\d+\(d: array\[qubit, 2\] @ owned\)", SlrConverter(prog).guppy())

    def test_cross_qreg_bundle_detected_and_inlined(self) -> None:
        from pecos.slr import CReg, Main, QReg, Return, SlrConverter
        from pecos.slr.ast import slr_to_ast
        from pecos.slr.ast.nodes import QubitBundleArg, SlotRef
        from pecos.slr.block import Block
        from pecos.slr.qeclib import qubit as qb
        from pecos.slr.qeclib.qubit.measures import Measure

        class BundleBlock(Block):
            block_inputs: ClassVar[dict[str, str]] = {"d": "live_preserved"}

            def __init__(self, d) -> None:
                super().__init__()
                self.d = d
                self.extend(qb.H(d[0]), qb.CX(d[0], d[1]))

        prog = Main(
            qa := QReg("qa", 2),
            qb_reg := QReg("qb", 2),
            c := CReg("c", 4),
            qb.PZ(qa),
            qb.PZ(qb_reg),
            BundleBlock([qa[0], qb_reg[1]]),
            Measure(qa) > c[0:2],
            Measure(qb_reg) > c[2:4],
            Return(c),
        )

        ast = slr_to_ast(prog)
        calls = [s for s in ast.body if isinstance(s, BlockCall)]
        assert calls[0].arg_bindings == (
            QubitBundleArg(
                slots=(
                    SlotRef(allocator="qa", index=0),
                    SlotRef(allocator="qb", index=1),
                ),
            ),
        )

        qasm_src = SlrConverter(prog).qasm()
        # Cross-allocator: d[0] -> qa[0], d[1] -> qb[1].
        assert "h qa[0];" in qasm_src
        assert "cx qa[0], qb[1];" in qasm_src
        # Emitter packs the non-contiguous bundle into one array arg.
        assert "array(qa_0, qb_1)" in SlrConverter(prog).guppy()

    def test_check_shaped_block_all_three_shapes_detected(self) -> None:
        """The realistic qeclib `Check` shape (5e.3's target): a `list[Qubit]`
        data bundle (live_preserved), a single ancilla `Qubit` (consumed,
        measured in-body), and a single `Bit` write-back (live_preserved).
        """
        from pecos.slr import CReg, Main, QReg, Return, SlrConverter
        from pecos.slr.ast import slr_to_ast
        from pecos.slr.ast.nodes import BitRef as AstBitRef
        from pecos.slr.ast.nodes import (
            QubitBundleArg,
            SingleBitArg,
            SingleQubitArg,
            SlotRef,
        )
        from pecos.slr.block import Block
        from pecos.slr.qeclib import qubit as qb
        from pecos.slr.qeclib.qubit.measures import Measure

        class CheckLike(Block):
            block_inputs: ClassVar[dict[str, str]] = {
                "d": "live_preserved",
                "a": "consumed",
                "out": "live_preserved",
            }

            def __init__(self, d, a, out) -> None:
                super().__init__()
                self.d = d
                self.a = a
                self.out = out
                self.extend(
                    qb.H(a),
                    qb.CX(a, d[0]),
                    qb.CX(a, d[1]),
                    qb.H(a),
                    Measure(a) > out,
                )

        prog = Main(
            outer_q := QReg("outer_q", 3),
            c := CReg("c", 3),
            qb.PZ(outer_q),
            CheckLike([outer_q[0], outer_q[1]], outer_q[2], c[0]),
            Measure(outer_q[0]) > c[1],
            Measure(outer_q[1]) > c[2],
            Return(c),
        )

        ast = slr_to_ast(prog)
        call = next(s for s in ast.body if isinstance(s, BlockCall))
        assert call.arg_bindings == (
            QubitBundleArg(
                slots=(
                    SlotRef(allocator="outer_q", index=0),
                    SlotRef(allocator="outer_q", index=1),
                ),
            ),
            SingleQubitArg(slot=SlotRef(allocator="outer_q", index=2)),
            SingleBitArg(bit=AstBitRef(register="c", index=0)),
        )
        # CONSUMED `a` must NOT appear in out_bindings (it is gone post-call);
        # the two LIVE_PRESERVED inputs must.
        assert call.out_bindings == (
            QubitBundleArg(
                slots=(
                    SlotRef(allocator="outer_q", index=0),
                    SlotRef(allocator="outer_q", index=1),
                ),
            ),
            SingleBitArg(bit=AstBitRef(register="c", index=0)),
        )

        qasm_src = SlrConverter(prog).qasm()
        assert "h outer_q[2];" in qasm_src
        assert "cx outer_q[2], outer_q[0];" in qasm_src
        assert "cx outer_q[2], outer_q[1];" in qasm_src
        assert "measure outer_q[2] -> c[0];" in qasm_src

        guppy_src = SlrConverter(prog).guppy()
        assert re.search(
            r"def checklike_\d+\(d: array\[qubit, 2\] @ owned, a: qubit @ owned, "
            r"out: array\[bool, 1\] @ owned\) -> tuple\[array\[qubit, 2\], array\[bool, 1\]\]:",
            guppy_src,
        )

    @pytest.mark.parametrize(
        ("make_block", "match"),
        [
            ("whole_creg", r"whole CReg input is not yet supported"),
            ("list_bit", r"list\[Bit\] .*is not yet supported"),
            ("empty_bundle", r"empty list bundle is not supported"),
            ("mixed_bundle", r"a bundle must be all Qubit"),
            ("symbolic", r"symbolic .*is not supported as a block input"),
        ],
    )
    def test_unsupported_input_shapes_rejected_via_converter(
        self,
        make_block: str,
        match: str,
    ) -> None:
        from pecos.slr import CReg, Main, QReg, Return
        from pecos.slr.ast import slr_to_ast
        from pecos.slr.block import Block
        from pecos.slr.vars import LoopVar

        class WholeCReg(Block):
            block_inputs: ClassVar[dict[str, str]] = {"out": "live_preserved"}

            def __init__(self, out) -> None:
                super().__init__()
                self.out = out

        class ListBit(Block):
            block_inputs: ClassVar[dict[str, str]] = {"out": "live_preserved"}

            def __init__(self, out) -> None:
                super().__init__()
                self.out = out

        class Empty(Block):
            block_inputs: ClassVar[dict[str, str]] = {"d": "live_preserved"}

            def __init__(self, d) -> None:
                super().__init__()
                self.d = d

        class Mixed(Block):
            block_inputs: ClassVar[dict[str, str]] = {"d": "live_preserved"}

            def __init__(self, d) -> None:
                super().__init__()
                self.d = d

        class Sym(Block):
            block_inputs: ClassVar[dict[str, str]] = {"q": "live_preserved"}

            def __init__(self, q) -> None:
                super().__init__()
                self.q = q

        if make_block == "whole_creg":
            prog = Main(
                c := CReg("c", 2),
                WholeCReg(c),
                Return(c),
            )
        elif make_block == "list_bit":
            prog = Main(
                c := CReg("c", 2),
                ListBit([c[0], c[1]]),
                Return(c),
            )
        elif make_block == "empty_bundle":
            prog = Main(QReg("o", 1), Empty([]))
        elif make_block == "mixed_bundle":
            prog = Main(
                o := QReg("o", 1),
                c := CReg("c", 1),
                Mixed([o[0], c[0]]),
                Return(c),
            )
        else:  # symbolic
            prog = Main(o := QReg("o", 2), Sym(o[LoopVar("i")]))

        with pytest.raises(ValueError, match=match):
            slr_to_ast(prog)

    def test_duplicate_qubit_in_bundle_rejected(self) -> None:
        """Regression: `[q[0], q[0]]` corrupted body substitution
        (Guppy rejected on linearity; QASM flatten emitted `cx q[0], q[0];`).
        Must reject at SLR -> AST conversion with a clear message.
        """
        from pecos.slr import Main, QReg, Return
        from pecos.slr.ast import slr_to_ast
        from pecos.slr.block import Block
        from pecos.slr.qeclib import qubit as qb

        class DupBundle(Block):
            block_inputs: ClassVar[dict[str, str]] = {"d": "live_preserved"}

            def __init__(self, d) -> None:
                super().__init__()
                self.d = d
                self.extend(qb.H(d[0]), qb.CX(d[0], d[1]))

        prog = Main(q := QReg("q", 2), DupBundle([q[0], q[0]]))
        with pytest.raises(ValueError, match=r"qubit q\[0\] is also bound .*no-cloning"):
            slr_to_ast(prog)

    def test_two_single_qubit_inputs_aliased_rejected(self) -> None:
        """Two distinct single-qubit inputs bound to the SAME outer slot is
        the same aliasing bug from the cross-input direction.
        """
        from pecos.slr import Main, QReg, Return
        from pecos.slr.ast import slr_to_ast
        from pecos.slr.block import Block
        from pecos.slr.qeclib import qubit as qb

        class TwoSq(Block):
            block_inputs: ClassVar[dict[str, str]] = {
                "a": "live_preserved",
                "b": "live_preserved",
            }

            def __init__(self, a, b) -> None:
                super().__init__()
                self.a = a
                self.b = b
                self.extend(qb.CX(a, b))

        prog = Main(q := QReg("q", 2), TwoSq(q[0], q[0]))
        with pytest.raises(ValueError, match=r"qubit q\[0\] is also bound by input 'a'"):
            slr_to_ast(prog)

    def test_two_bit_inputs_aliased_rejected(self) -> None:
        """Same outer bit backing two single-bit inputs is lossy during
        body substitution -- reject it too.
        """
        from pecos.slr import CReg, Main, Return
        from pecos.slr.ast import slr_to_ast
        from pecos.slr.block import Block

        class TwoBit(Block):
            block_inputs: ClassVar[dict[str, str]] = {
                "x": "live_preserved",
                "y": "live_preserved",
            }

            def __init__(self, x, y) -> None:
                super().__init__()
                self.x = x
                self.y = y

        prog = Main(
            c := CReg("c", 2),
            TwoBit(c[0], c[0]),
            Return(c),
        )
        with pytest.raises(ValueError, match=r"bit c\[0\] is also bound by input 'x'"):
            slr_to_ast(prog)


class TestScratchEffectS1:
    """Scratch-ancilla effect: `ResourceEffect.SCRATCH`
    + converter detection + the mandatory reset-first validator.

    This stage does NOT lower scratch in Guppy (the internal-allocation
    lowering lands later); until then Guppy must
    reject a SCRATCH input loudly (no silent fallback). Flatten/QASM
    already works because the scratch param substitutes to the outer slot
    exactly like any per-slot input.
    """

    @staticmethod
    def _good_check():
        from pecos.slr import Block
        from pecos.slr.qeclib import qubit as qb
        from pecos.slr.qeclib.qubit.measures import Measure

        class GoodCheck(Block):
            block_inputs: ClassVar[dict[str, str]] = {
                "d": "live_preserved",
                "a": "scratch",
                "out": "live_preserved",
            }

            def __init__(self, d, a, out) -> None:
                super().__init__()
                self.d = d
                self.a = a
                self.out = out
                self.extend(qb.PZ(a), qb.CX(a, d[0]), qb.CX(a, d[1]), Measure(a) > out)

        return GoodCheck

    def test_scratch_detected_excluded_from_out_bindings(self) -> None:
        from pecos.slr import CReg, Main, QReg, Return
        from pecos.slr.ast import slr_to_ast
        from pecos.slr.ast.nodes import (
            QubitBundleArg,
            SingleBitArg,
            SingleQubitArg,
        )
        from pecos.slr.qeclib import qubit as qb

        good_check = self._good_check()
        prog = Main(
            q := QReg("q", 3),
            c := CReg("c", 1),
            qb.PZ(q),
            good_check([q[0], q[1]], q[2], c[0]),
            Return(c),
        )
        ast = slr_to_ast(prog)
        call = next(s for s in ast.body if isinstance(s, BlockCall))
        # Scratch `a` is detected as a single-qubit arg and stays in
        # arg_bindings (the alias guard still applies)...
        assert any(isinstance(a, SingleQubitArg) for a in call.arg_bindings)
        # ...but is NOT live-preserved, so it is absent from out_bindings.
        assert call.out_bindings == (
            QubitBundleArg(
                slots=tuple(s for s in call.arg_bindings[0].slots),  # d
            ),
            SingleBitArg(bit=call.arg_bindings[2].bit),  # out
        )

    def test_scratch_flatten_substitutes_to_outer_slot(self) -> None:
        from pecos.slr import CReg, Main, QReg, Return, SlrConverter
        from pecos.slr.qeclib import qubit as qb

        good_check = self._good_check()
        prog = Main(
            q := QReg("q", 3),
            c := CReg("c", 1),
            qb.PZ(q),
            good_check([q[0], q[1]], q[2], c[0]),
            Return(c),
        )
        qasm = SlrConverter(prog).qasm()
        # Param `a` -> outer slot q[2]; data bundle d[0]/d[1] -> q[0]/q[1].
        assert "reset q[2];" in qasm
        assert "cx q[2], q[0];" in qasm
        assert "cx q[2], q[1];" in qasm
        assert "measure q[2] -> c[0];" in qasm

    def test_scratch_guppy_internal_alloc_no_param(self) -> None:
        """A SCRATCH input is allocated INSIDE the @guppy def -- it is
        neither a function parameter nor a positional call argument; the
        body's PZ(scratch) lowers to a fresh internal `qubit()`.
        """
        from pecos.slr import CReg, Main, QReg, Return, SlrConverter
        from pecos.slr.qeclib import qubit as qb

        good_check = self._good_check()
        prog = Main(
            q := QReg("q", 3),
            c := CReg("c", 1),
            qb.PZ(q),
            good_check([q[0], q[1]], q[2], c[0]),
            Return(c),
        )
        guppy_src = SlrConverter(prog).guppy()
        # No `a` parameter on the generated def (scratch is internal).
        assert re.search(
            r"def goodcheck_\d+\(d: array\[qubit, 2\] @ owned, out: array\[bool, 1\] @ owned\) -> ",
            guppy_src,
        )
        assert "a: qubit @ owned" not in guppy_src
        # Body allocates the scratch internally and measures it.
        assert "a_0 = qubit()" in guppy_src
        assert "out[0] = measure(a_0)" in guppy_src
        # The call passes only the non-scratch args (no scratch positional).
        assert re.search(r"goodcheck_\d+\(array\(q_0, q_1\), array\(c\[0\]\)\)", guppy_src)

    def test_scratch_use_before_prep_rejected(self) -> None:
        from pecos.slr import CReg, Main, QReg, Return
        from pecos.slr.ast import slr_to_ast
        from pecos.slr.block import Block
        from pecos.slr.qeclib import qubit as qb
        from pecos.slr.qeclib.qubit.measures import Measure

        class BadFirst(Block):
            block_inputs: ClassVar[dict[str, str]] = {"a": "scratch", "out": "live_preserved"}

            def __init__(self, a, out) -> None:
                super().__init__()
                self.a = a
                self.out = out
                self.extend(qb.H(a), qb.PZ(a), Measure(a) > out)

        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            BadFirst(q[0], c[0]),
            Return(c),
        )
        with pytest.raises(ValueError, match=r"first use is USE.*reset \(PZ\) before"):
            slr_to_ast(prog)

    def test_scratch_use_after_measure_rejected(self) -> None:
        from pecos.slr import CReg, Main, QReg, Return
        from pecos.slr.ast import slr_to_ast
        from pecos.slr.block import Block
        from pecos.slr.qeclib import qubit as qb
        from pecos.slr.qeclib.qubit.measures import Measure

        class BadAfter(Block):
            block_inputs: ClassVar[dict[str, str]] = {"a": "scratch", "out": "live_preserved"}

            def __init__(self, a, out) -> None:
                super().__init__()
                self.a = a
                self.out = out
                self.extend(qb.PZ(a), Measure(a) > out, qb.H(a))

        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            BadAfter(q[0], c[0]),
            Return(c),
        )
        with pytest.raises(ValueError, match=r"used after measurement without re-PZ"):
            slr_to_ast(prog)

    def test_scratch_unused_rejected(self) -> None:
        from pecos.slr import Main, QReg, Return
        from pecos.slr.ast import slr_to_ast
        from pecos.slr.block import Block

        class BadUnused(Block):
            block_inputs: ClassVar[dict[str, str]] = {"a": "scratch"}

            def __init__(self, a) -> None:
                super().__init__()
                self.a = a

        prog = Main(q := QReg("q", 1), BadUnused(q[0]))
        with pytest.raises(ValueError, match=r"declared SCRATCH but never used"):
            slr_to_ast(prog)

    def test_scratch_inside_control_flow_rejected(self) -> None:
        """Conservative scope: a scratch slot touched inside control
        flow cannot be linearized for the reset-first analysis -> reject loudly.
        """
        from pecos.slr import CReg, Main, QReg, Repeat, Return
        from pecos.slr.ast import slr_to_ast
        from pecos.slr.block import Block
        from pecos.slr.qeclib import qubit as qb
        from pecos.slr.qeclib.qubit.measures import Measure

        class LoopScratch(Block):
            block_inputs: ClassVar[dict[str, str]] = {"a": "scratch", "out": "live_preserved"}

            def __init__(self, a, out) -> None:
                super().__init__()
                self.a = a
                self.out = out
                self.extend(
                    qb.PZ(a),
                    Repeat(2).block(qb.H(a)),
                    Measure(a) > out,
                )

        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            LoopScratch(q[0], c[0]),
            Return(c),
        )
        with pytest.raises(ValueError, match=r"flat PZ -> \.\.\. -> Measure lifecycle"):
            slr_to_ast(prog)

    def test_scratch_aliased_to_other_input_rejected(self) -> None:
        """Scratch stays in arg_bindings, so the cross-input
        aliasing guard still rejects a scratch slot aliased to another
        qubit input.
        """
        from pecos.slr import Main, QReg, Return
        from pecos.slr.ast import slr_to_ast
        from pecos.slr.block import Block
        from pecos.slr.qeclib import qubit as qb
        from pecos.slr.qeclib.qubit.measures import Measure

        class Alias(Block):
            block_inputs: ClassVar[dict[str, str]] = {"a": "scratch", "b": "live_preserved"}

            def __init__(self, a, b) -> None:
                super().__init__()
                self.a = a
                self.b = b
                self.extend(qb.PZ(a), qb.CX(a, b), Measure(a))

        prog = Main(q := QReg("q", 2), Alias(q[0], q[0]))
        with pytest.raises(ValueError, match=r"qubit q\[0\] is also bound by input 'a'"):
            slr_to_ast(prog)

    def test_scratch_prep_only_rejected(self) -> None:
        """A PZ with no terminating Measure must be
        rejected -- the Guppy lowering allocates the scratch qubit internally, so an
        unmeasured trailing PZ diverges from the flatten/QASM path.
        """
        from pecos.slr import Main, QReg, Return
        from pecos.slr.ast import slr_to_ast
        from pecos.slr.block import Block
        from pecos.slr.qeclib import qubit as qb

        class PrepOnly(Block):
            block_inputs: ClassVar[dict[str, str]] = {"a": "scratch"}

            def __init__(self, a) -> None:
                super().__init__()
                self.a = a
                self.extend(qb.PZ(a))

        prog = Main(q := QReg("q", 1), PrepOnly(q[0]))
        with pytest.raises(ValueError, match=r"scratch lifecycle not closed"):
            slr_to_ast(prog)

    def test_scratch_prep_use_no_measure_rejected(self) -> None:
        from pecos.slr import Main, QReg, Return
        from pecos.slr.ast import slr_to_ast
        from pecos.slr.block import Block
        from pecos.slr.qeclib import qubit as qb

        class PrepUseNoMeasure(Block):
            block_inputs: ClassVar[dict[str, str]] = {"a": "scratch"}

            def __init__(self, a) -> None:
                super().__init__()
                self.a = a
                self.extend(qb.PZ(a), qb.H(a))

        prog = Main(q := QReg("q", 1), PrepUseNoMeasure(q[0]))
        with pytest.raises(ValueError, match=r"scratch lifecycle not closed"):
            slr_to_ast(prog)

    def test_scratch_prep_measure_prep_unmeasured_rejected(self) -> None:
        """A second PZ with no closing Measure (trailing open lifecycle)."""
        from pecos.slr import CReg, Main, QReg, Return
        from pecos.slr.ast import slr_to_ast
        from pecos.slr.block import Block
        from pecos.slr.qeclib import qubit as qb
        from pecos.slr.qeclib.qubit.measures import Measure

        class PrepMeasurePrep(Block):
            block_inputs: ClassVar[dict[str, str]] = {"a": "scratch", "out": "live_preserved"}

            def __init__(self, a, out) -> None:
                super().__init__()
                self.a = a
                self.out = out
                self.extend(qb.PZ(a), Measure(a) > out, qb.PZ(a))

        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            PrepMeasurePrep(q[0], c[0]),
            Return(c),
        )
        with pytest.raises(ValueError, match=r"scratch lifecycle not closed"):
            slr_to_ast(prog)

    def test_scratch_two_prep_no_measure_between_rejected(self) -> None:
        """Re-PZ before measuring the first lifecycle."""
        from pecos.slr import Main, QReg, Return
        from pecos.slr.ast import slr_to_ast
        from pecos.slr.block import Block
        from pecos.slr.qeclib import qubit as qb

        class PrepPrep(Block):
            block_inputs: ClassVar[dict[str, str]] = {"a": "scratch"}

            def __init__(self, a) -> None:
                super().__init__()
                self.a = a
                self.extend(qb.PZ(a), qb.PZ(a))

        prog = Main(q := QReg("q", 1), PrepPrep(q[0]))
        with pytest.raises(ValueError, match=r"re-Prepped before measuring"):
            slr_to_ast(prog)

    def test_scratch_prep_measure_prep_measure_accepted(self) -> None:
        """Two complete PZ -> Measure lifecycles is valid (each closed)."""
        from pecos.slr import CReg, Main, QReg, Return, SlrConverter
        from pecos.slr.ast import slr_to_ast
        from pecos.slr.block import Block
        from pecos.slr.qeclib import qubit as qb
        from pecos.slr.qeclib.qubit.measures import Measure

        class TwoLifecycles(Block):
            block_inputs: ClassVar[dict[str, str]] = {
                "a": "scratch",
                "out0": "live_preserved",
                "out1": "live_preserved",
            }

            def __init__(self, a, out0, out1) -> None:
                super().__init__()
                self.a = a
                self.out0 = out0
                self.out1 = out1
                self.extend(
                    qb.PZ(a),
                    Measure(a) > out0,
                    qb.PZ(a),
                    Measure(a) > out1,
                )

        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 2),
            TwoLifecycles(q[0], c[0], c[1]),
            Return(c),
        )
        ast = slr_to_ast(prog)  # must not raise
        assert any(isinstance(s, BlockCall) for s in ast.body)
        # Flatten still substitutes the scratch param to the outer slot.
        qasm = SlrConverter(prog).qasm()
        assert qasm.count("reset q[0];") >= 2
        assert "measure q[0] -> c[0];" in qasm
        assert "measure q[0] -> c[1];" in qasm

    def test_scratch_return_rejected(self) -> None:
        """A scratch-bearing block must contain no ReturnOp.

        Post-substitution a returned scratch slot keeps the OUTER name
        (a partial VarExpr passes `whole_name` through unchanged), so it
        is indistinguishable from a returned classical value -- the validator
        conservatively rejects ANY Return in a scratch block (in-scope
        `Check` has none). Covers both returning the scratch qubit and
        returning an unrelated value.
        """
        from pecos.slr import CReg, Main, QReg, Return
        from pecos.slr.ast import slr_to_ast
        from pecos.slr.block import Block
        from pecos.slr.qeclib import qubit as qb
        from pecos.slr.qeclib.qubit.measures import Measure

        class RetScratch(Block):
            block_inputs: ClassVar[dict[str, str]] = {"a": "scratch", "out": "live_preserved"}

            def __init__(self, a, out) -> None:
                super().__init__()
                self.a = a
                self.out = out
                self.extend(qb.PZ(a), Measure(a) > out, Return(a))

        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            RetScratch(q[0], c[0]),
            Return(c),
        )
        with pytest.raises(ValueError, match=r"or any ReturnOp in a scratch-bearing block"):
            slr_to_ast(prog)

        class RetOther(Block):
            block_inputs: ClassVar[dict[str, str]] = {"a": "scratch", "out": "live_preserved"}

            def __init__(self, a, out) -> None:
                super().__init__()
                self.a = a
                self.out = out
                self.extend(qb.PZ(a), Measure(a) > out, Return(out))

        prog2 = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            RetOther(q[0], c[0]),
            Return(c),
        )
        with pytest.raises(ValueError, match=r"or any ReturnOp in a scratch-bearing block"):
            slr_to_ast(prog2)

    def test_scratch_reuse_across_calls_compiles_and_runs(self) -> None:
        """End-to-end: the original blocker. The SAME outer ancilla slot
        feeds two sequential scratch BlockCalls (the production
        `SynExtractBare` reuse pattern). Under `a: consumed` this was a
        Guppy LinearityError; with `a: scratch` each call allocates its
        ancilla internally, so it compiles and runs through Selene.
        """
        from pecos import Hugr, selene_engine, sim
        from pecos.slr import Block, CReg, Main, QReg, Return, SlrConverter
        from pecos.slr.qeclib import qubit as qb
        from pecos.slr.qeclib.qubit.measures import Measure

        class Chk(Block):
            block_inputs: ClassVar[dict[str, str]] = {
                "d": "live_preserved",
                "a": "scratch",
                "out": "live_preserved",
            }

            def __init__(self, d, a, out) -> None:
                super().__init__()
                self.d = d
                self.a = a
                self.out = out
                self.extend(
                    qb.PZ(a),
                    qb.H(a),
                    qb.CX(a, d[0]),
                    qb.CX(a, d[1]),
                    qb.H(a),
                    Measure(a) > out,
                )

        prog = Main(
            q := QReg("q", 3),
            c := CReg("c", 2),
            qb.PZ(q),
            Chk([q[0], q[1]], q[2], c[0]),
            Chk([q[0], q[1]], q[2], c[1]),  # reuses q[2] -- the blocker case
            Return(c),
        )
        guppy_src = SlrConverter(prog).guppy()
        # Two separate scratch internal allocs, no LinearityError.
        assert guppy_src.count("def chk") == 2
        assert guppy_src.count("a_0 = qubit()") == 2

        package = SlrConverter(prog).hugr()
        result = (
            sim(Hugr(package.to_str().encode("utf-8")))
            .classical(selene_engine())
            .qubits(4)  # 3 outer + 1 internally-allocated scratch
            .seed(42)
            .run(4)
        )
        raw = result.to_dict() if hasattr(result, "to_dict") else result
        # Probe-pinned: both Checks see the same |00> data, same seed.
        assert raw == {
            "measurement_0": [1, 0, 0, 1],
            "measurement_1": [1, 0, 0, 1],
        }, raw

    def test_scratch_outer_slot_misuse_rejected(self) -> None:
        """A scratch-bound outer slot also used as
        meaningful caller state (gate + later measure) diverges between
        flatten (mutates it) and Guppy (allocates internally) -- reject.
        """
        from pecos.slr import Block, CReg, Main, QReg, Return, SlrConverter
        from pecos.slr.ast.codegen.guppy import GuppyCodegenError
        from pecos.slr.qeclib import qubit as qb
        from pecos.slr.qeclib.qubit.measures import Measure

        class MS(Block):
            block_inputs: ClassVar[dict[str, str]] = {"a": "scratch", "out": "live_preserved"}

            def __init__(self, a, out) -> None:
                super().__init__()
                self.a = a
                self.out = out
                self.extend(qb.PZ(a), qb.H(a), Measure(a) > out)

        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 2),
            qb.PZ(q[0]),
            qb.X(q[0]),  # meaningful caller state on the scratch slot
            MS(q[0], c[0]),
            Measure(q[0]) > c[1],  # ...observed after the scratch call
            Return(c),
        )
        with pytest.raises(GuppyCodegenError, match=r"Scratch outer slot q\[0\].*meaningful caller state"):
            SlrConverter(prog).guppy()

    def test_scratch_outer_slot_whole_reg_prep_allowed(self) -> None:
        """Carve-out: a bare/whole-register PZ covering the scratch slot
        is allowed (the qeclib corpus does `PZ(q)` then uses `q[i]` as a
        check ancilla -- a reset is unobserved/dead under both lowerings).
        """
        from pecos.slr import Block, CReg, Main, QReg, Return, SlrConverter
        from pecos.slr.qeclib import qubit as qb
        from pecos.slr.qeclib.qubit.measures import Measure

        class Chk(Block):
            block_inputs: ClassVar[dict[str, str]] = {
                "d": "live_preserved",
                "a": "scratch",
                "out": "live_preserved",
            }

            def __init__(self, d, a, out) -> None:
                super().__init__()
                self.d = d
                self.a = a
                self.out = out
                self.extend(qb.PZ(a), qb.CX(a, d[0]), qb.CX(a, d[1]), Measure(a) > out)

        prog = Main(
            q := QReg("q", 3),
            c := CReg("c", 1),
            qb.PZ(q),  # covers q[2] (the scratch slot) -- must NOT reject
            Chk([q[0], q[1]], q[2], c[0]),
            Return(c),
        )
        guppy_src = SlrConverter(prog).guppy()  # must not raise
        assert "a_0 = qubit()" in guppy_src

    def test_scratch_malformed_arg_rejected_direct_ast(self) -> None:
        """A scratch BlockArg referencing an
        out-of-bounds outer slot must fail loudly -- scratch args are now
        validated (then excluded), not skipped before validation.
        """
        from pecos.slr.ast.codegen.guppy import GuppyCodegenError
        from pecos.slr.ast.nodes import SingleQubitArg

        decl = BlockDecl(
            name="b",
            inputs=(BlockInput(name="a", effect=ResourceEffect.SCRATCH, type_expr=QubitTypeExpr()),),
            body=(
                PrepareOp(allocator="a", slots=(0,)),
                MeasureOp(targets=(SlotRef(allocator="a", index=0),)),
            ),
        )
        prog = Program(
            name="main",
            allocator=AllocatorDecl(name="outer_q", capacity=1),
            block_decls=(decl,),
            body=(
                BlockCall(
                    callee="b",
                    arg_bindings=(SingleQubitArg(slot=SlotRef(allocator="outer_q", index=9)),),
                    out_bindings=(),
                ),
            ),
        )
        with pytest.raises(GuppyCodegenError, match=r"slot index 9 out of bounds"):
            ast_to_guppy(prog)

    def test_scratch_outer_slot_permute_rejected(self) -> None:
        """A Permute touching the scratch register
        observes/reorders the scratch-bound outer slot -> reject.
        """
        from pecos.slr import Block, CReg, Main, Permute, QReg, Return, SlrConverter
        from pecos.slr.ast.codegen.guppy import GuppyCodegenError
        from pecos.slr.qeclib import qubit as qb
        from pecos.slr.qeclib.qubit.measures import Measure

        class MS(Block):
            block_inputs: ClassVar[dict[str, str]] = {"a": "scratch", "out": "live_preserved"}

            def __init__(self, a, out) -> None:
                super().__init__()
                self.a = a
                self.out = out
                self.extend(qb.PZ(a), qb.H(a), Measure(a) > out)

        prog = Main(
            q := QReg("q", 2),
            c := CReg("c", 2),
            qb.PZ(q),
            qb.X(q[0]),
            MS(q[1], c[0]),
            Permute([q[0], q[1]], [q[1], q[0]]),
            Measure(q[0]) > c[1],
            Return(c),
        )
        with pytest.raises(GuppyCodegenError, match=r"Scratch outer slot q\[1\].*Permute"):
            SlrConverter(prog).guppy()

    def test_scratch_outer_slot_return_rejected(self) -> None:
        """Returning the register that hosts the
        scratch-bound slot exposes a slot Guppy left untouched -> reject.
        """
        from pecos.slr import Block, CReg, Main, QReg, Return, SlrConverter
        from pecos.slr.ast.codegen.guppy import GuppyCodegenError
        from pecos.slr.qeclib import qubit as qb
        from pecos.slr.qeclib.qubit.measures import Measure

        class MS(Block):
            block_inputs: ClassVar[dict[str, str]] = {"a": "scratch", "out": "live_preserved"}

            def __init__(self, a, out) -> None:
                super().__init__()
                self.a = a
                self.out = out
                self.extend(qb.PZ(a), qb.H(a), Measure(a) > out)

        prog = Main(
            q := QReg("q", 2),
            c := CReg("c", 1),
            qb.PZ(q),
            MS(q[1], c[0]),
            Return(q),
        )
        with pytest.raises(GuppyCodegenError, match=r"Scratch outer slot q\[1\].*Return"):
            SlrConverter(prog).guppy()

    def test_scratch_misuse_inside_block_body_rejected(self) -> None:
        """The purity guard runs per scope -- a
        nested scratch BlockCall + misuse of that slot inside a BlockDecl
        body (not just main) must be caught.
        """
        from pecos.slr import Block, CReg, Main, QReg, Return, SlrConverter
        from pecos.slr.ast.codegen.guppy import GuppyCodegenError
        from pecos.slr.qeclib import qubit as qb
        from pecos.slr.qeclib.qubit.measures import Measure

        class Inner(Block):
            block_inputs: ClassVar[dict[str, str]] = {"a": "scratch", "out": "live_preserved"}

            def __init__(self, a, out) -> None:
                super().__init__()
                self.a = a
                self.out = out
                self.extend(qb.PZ(a), qb.H(a), Measure(a) > out)

        class Outer(Block):
            block_inputs: ClassVar[dict[str, str]] = {
                "q": "live_preserved",
                "o": "live_preserved",
            }

            def __init__(self, q, o) -> None:
                super().__init__()
                self.q = q
                self.o = o
                self.extend(Inner(q[1], o), qb.X(q[1]))  # misuse within this scope

        prog = Main(
            qq := QReg("qq", 2),
            c := CReg("c", 1),
            qb.PZ(qq),
            Outer(qq, c[0]),
            Return(c),
        )
        with pytest.raises(GuppyCodegenError, match=r"Scratch outer slot q\[1\].*meaningful caller state"):
            SlrConverter(prog).guppy()


class TestScratchCheckS4ProductionLockIn:
    """S4: the real qeclib `Check` is converted (`a: scratch`). The
    production caller `SynExtractBare` reuses one ancilla register slot
    across sequential Checks -- under `a: consumed` this was a Guppy
    LinearityError (the original 5e.3 blocker). With the scratch effect
    it must route every Check through a BlockCall, compile to Guppy, and
    run through Selene with stable records, while QASM flatten stays
    byte-identical to the inlined form.
    """

    def test_steane_syn_extract_bare_routes_check_through_blockcall(self) -> None:
        from pecos.slr import CReg, Main, QReg, Return, SlrConverter
        from pecos.slr.ast import slr_to_ast
        from pecos.slr.qeclib.steane.syn_extract.bare import SynExtractBare

        checks = [[2, 1, 3, 0], [5, 2, 1, 4], [6, 5, 2, 3]]
        prog = Main(
            d := QReg("d", 7),
            a := QReg("a", 2),
            syn := CReg("syn", 6),
            SynExtractBare(d, a, checks, syn),
            Return(syn),
        )
        ast = slr_to_ast(prog)
        # 6 Checks (3 Z + 3 X) -> 6 BlockCalls, not silently flattened.
        assert sum(1 for s in ast.body if isinstance(s, BlockCall)) == 6
        assert len(ast.block_decls) == 6

        guppy_src = SlrConverter(prog).guppy()
        # Was the original blocker: `a: consumed` -> LinearityError on the
        # reused ancilla slot. Now each Check def allocates internally.
        assert guppy_src.count("def check") == 6
        assert "LinearityError" not in guppy_src

        # QASM flatten still substitutes the scratch param to the outer
        # ancilla slot (byte-identical inlined form -- a[0]/a[1] reused).
        qasm = SlrConverter(prog).qasm()
        assert "reset a[0];" in qasm
        assert "reset a[1];" in qasm
        assert "measure a[0] -> syn[0];" in qasm

    def test_steane_syn_extract_bare_selene_records_pinned(self) -> None:
        from pecos import Hugr, selene_engine, sim
        from pecos.slr import CReg, Main, QReg, Return, SlrConverter
        from pecos.slr.qeclib.steane.syn_extract.bare import SynExtractBare

        checks = [[2, 1, 3, 0], [5, 2, 1, 4], [6, 5, 2, 3]]
        prog = Main(
            d := QReg("d", 7),
            a := QReg("a", 2),
            syn := CReg("syn", 6),
            SynExtractBare(d, a, checks, syn),
            Return(syn),
        )
        package = SlrConverter(prog).hugr()
        # d(7) + a(2) declared + 1 internally-allocated scratch ancilla
        # (design R2: parity is on behavioral records, not qubit count).
        result = sim(Hugr(package.to_str().encode("utf-8"))).classical(selene_engine()).qubits(10).seed(42).run(4)
        raw = result.to_dict() if hasattr(result, "to_dict") else result
        # Empirical probe 2026-05-16 (post-conversion, scratch internal alloc).
        assert raw == {
            "measurement_0": [0, 0, 0, 0],
            "measurement_1": [0, 0, 0, 0],
            "measurement_2": [0, 0, 0, 0],
            "measurement_3": [1, 1, 0, 1],
            "measurement_4": [0, 0, 1, 0],
            "measurement_5": [0, 0, 1, 0],
        }, raw

    def test_color488_syn_extract_bare_routes_and_runs(self) -> None:
        """Design says the S4 production lock-in is Steane + Color488
        Color488's serial `SynExtractBare` uses the
        identical generic `Check` call shape -- it must also route every
        Check through a BlockCall, compile, and run.
        """
        from pecos import Hugr, selene_engine, sim
        from pecos.slr import CReg, Main, QReg, Return, SlrConverter
        from pecos.slr.ast import slr_to_ast
        from pecos.slr.qeclib.color488.syn_extract.bare import (
            SynExtractBare as Color488SynExtractBare,
        )

        checks = [[2, 1, 3, 0], [5, 2, 1, 4], [6, 5, 2, 3]]
        prog = Main(
            d := QReg("d", 7),
            a := QReg("a", 2),
            syn := CReg("syn", 6),
            Color488SynExtractBare(d, a, checks, syn),
            Return(syn),
        )
        ast = slr_to_ast(prog)
        assert sum(1 for s in ast.body if isinstance(s, BlockCall)) == 6
        guppy_src = SlrConverter(prog).guppy()
        assert guppy_src.count("def check") == 6
        assert "LinearityError" not in guppy_src

        package = SlrConverter(prog).hugr()
        result = sim(Hugr(package.to_str().encode("utf-8"))).classical(selene_engine()).qubits(10).seed(42).run(4)
        raw = result.to_dict() if hasattr(result, "to_dict") else result
        # Same generic-Check mechanism as Steane -> same seed-42 record shape.
        assert raw == {
            "measurement_0": [0, 0, 0, 0],
            "measurement_1": [0, 0, 0, 0],
            "measurement_2": [0, 0, 0, 0],
            "measurement_3": [1, 1, 0, 1],
            "measurement_4": [0, 0, 1, 0],
            "measurement_5": [0, 0, 1, 0],
        }, raw


class TestScratchCheck1FlagS5ProductionLockIn:
    """`Check1Flag` converted (`a`, `flag`: scratch; the body's
    `PZ(a, flag)` split into `PZ(a); PZ(flag)`).
    Steane `SynExtractFlagged` reuses ancilla+flag slots across 6
    Check1Flag calls -- it must route every one through a BlockCall,
    compile (12 internal allocs, no LinearityError), and run.
    """

    def test_steane_syn_extract_flagged_routes_and_runs(self) -> None:
        from pecos import Hugr, selene_engine, sim
        from pecos.slr import CReg, Main, QReg, Return, SlrConverter
        from pecos.slr.ast import slr_to_ast
        from pecos.slr.qeclib.steane.syn_extract.flagged import SynExtractFlagged

        checks = [[2, 1, 3, 0], [5, 2, 1, 4], [6, 5, 2, 3]]
        prog = Main(
            d := QReg("d", 7),
            a := QReg("a", 4),
            sx := CReg("sx", 3),
            sz := CReg("sz", 3),
            fx := CReg("fx", 3),
            fz := CReg("fz", 3),
            SynExtractFlagged(d, a, checks, sx, sz, fx, fz),
            Return(sx, sz, fx, fz),
        )
        ast = slr_to_ast(prog)
        assert sum(1 for s in ast.body if isinstance(s, BlockCall)) == 6
        assert len(ast.block_decls) == 6

        guppy_src = SlrConverter(prog).guppy()
        assert guppy_src.count("def check1flag") == 6
        # 2 internal scratch allocs (a + flag) per Check1Flag * 6.
        assert guppy_src.count("= qubit()") == 12
        assert "LinearityError" not in guppy_src

        package = SlrConverter(prog).hugr()
        result = (
            sim(Hugr(package.to_str().encode("utf-8")))
            .classical(selene_engine())
            .qubits(13)  # d(7)+a(4) declared + 2 concurrent internal scratch
            .seed(42)
            .run(4)
        )
        raw = result.to_dict() if hasattr(result, "to_dict") else result
        # Empirical probe 2026-05-16 (post-conversion, scratch internal alloc).
        assert raw == {
            "measurement_0": [0, 0, 0, 0],
            "measurement_1": [0, 0, 0, 0],
            "measurement_2": [0, 0, 0, 0],
            "measurement_3": [0, 0, 0, 0],
            "measurement_4": [0, 0, 0, 0],
            "measurement_5": [0, 0, 0, 0],
            "measurement_6": [1, 1, 0, 1],
            "measurement_7": [0, 0, 0, 0],
            "measurement_8": [0, 0, 1, 0],
            "measurement_9": [0, 0, 0, 0],
            "measurement_10": [0, 0, 1, 0],
            "measurement_11": [0, 0, 0, 0],
        }, raw

    def test_check1flag_h_branch_compiles_qasm_only(self) -> None:
        """The CH (`ops="H"`) branch hits a non-Clifford R1XY at runtime
        (Selene cannot execute it), so lock in compile + QASM parity
        only: it must still route through a BlockCall and lower cleanly
        (structural check -- not a Selene assertion).
        """
        from pecos.slr import CReg, Main, QReg, Return, SlrConverter
        from pecos.slr.ast import slr_to_ast
        from pecos.slr.qeclib.generic.check_1flag import Check1Flag

        prog = Main(
            q := QReg("q", 5),
            c := CReg("c", 2),
            Check1Flag([q[0], q[1], q[2]], "HHH", q[3], q[4], c[0], c[1], with_barriers=True),
            Return(c),
        )
        ast = slr_to_ast(prog)
        assert sum(1 for s in ast.body if isinstance(s, BlockCall)) == 1

        guppy_src = SlrConverter(prog).guppy()
        assert guppy_src.count("def check1flag") == 1
        assert "LinearityError" not in guppy_src
        assert "ch(" in guppy_src  # the CH branch is exercised

        qasm = SlrConverter(prog).qasm()  # flatten must inline cleanly
        assert "reset q[3];" in qasm
        assert "reset q[4];" in qasm
        assert "measure q[3] -> c[0];" in qasm
        assert "measure q[4] -> c[1];" in qasm
