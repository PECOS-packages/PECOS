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

"""Direct unit tests for the shared body substitution.

The 7 already-converted Steane Blocks only exercise the *whole-allocator
identity* path (`q[i] -> q[i]`). 5e.1 is "dead-code-with-tests" without
coverage of the NON-identity slot/bit maps
(`q[2] -> d[0]`, `c[3] -> out[0]`) and every recursive node kind plus the
reject-on-partial paths. These tests pin exactly that, asserting the
rewritten ref (not just "it compiled").
"""

from __future__ import annotations

import pytest
from pecos.slr.ast._block_substitution import (
    BodyRemap,
    BodySubstitutionError,
    substitute_stmt,
)
from pecos.slr.ast.nodes import (
    AllocatorArg,
    AssignOp,
    BarrierOp,
    BinaryExpr,
    BinaryOp,
    BitExpr,
    BitRef,
    BlockCall,
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
    PrintOp,
    QubitBundleArg,
    RepeatStmt,
    ReturnOp,
    SingleBitArg,
    SingleQubitArg,
    SlotRef,
    UnaryExpr,
    UnaryOp,
    VarExpr,
    WhileStmt,
)


def _bundle_remap() -> BodyRemap:
    """A non-identity remap: q[2]->d[0], q[5]->d[1] (qubit bundle) and
    c[3]->out[0] (single bit). `q` and `c` become partially-bound names.
    """
    remap = BodyRemap()
    remap.add_slot(("q", 2), ("d", 0))
    remap.add_slot(("q", 5), ("d", 1))
    remap.add_bit(("c", 3), ("out", 0))
    return remap


class TestSlotAndBitRemap:
    def test_gate_targets_nonidentity(self) -> None:
        g = GateOp(
            gate=GateKind.CX,
            targets=(SlotRef(allocator="q", index=2), SlotRef(allocator="q", index=5)),
        )
        out = substitute_stmt(g, _bundle_remap())
        assert out.targets == (
            SlotRef(allocator="d", index=0),
            SlotRef(allocator="d", index=1),
        )

    def test_gate_params_expression_remapped(self) -> None:
        # A parameterized gate whose angle expr references a bit via BitExpr.
        g = GateOp(
            gate=GateKind.RZ,
            targets=(SlotRef(allocator="q", index=2),),
            params=(BitExpr(ref=BitRef(register="c", index=3)),),
        )
        out = substitute_stmt(g, _bundle_remap())
        assert out.targets == (SlotRef(allocator="d", index=0),)
        assert out.params == (BitExpr(ref=BitRef(register="out", index=0)),)

    def test_measure_results_remapped(self) -> None:
        m = MeasureOp(
            targets=(SlotRef(allocator="q", index=5),),
            results=(BitRef(register="c", index=3),),
        )
        out = substitute_stmt(m, _bundle_remap())
        assert out.targets == (SlotRef(allocator="d", index=1),)
        assert out.results == (BitRef(register="out", index=0),)

    def test_assign_target_and_value(self) -> None:
        a = AssignOp(
            target=BitRef(register="c", index=3),
            value=BinaryExpr(
                op=BinaryOp.XOR,
                left=BitExpr(ref=BitRef(register="c", index=3)),
                right=LiteralExpr(value=1),
            ),
        )
        out = substitute_stmt(a, _bundle_remap())
        assert out.target == BitRef(register="out", index=0)
        assert out.value.left == BitExpr(ref=BitRef(register="out", index=0))
        assert out.value.right == LiteralExpr(value=1)

    def test_print_bitref_remapped(self) -> None:
        p = PrintOp(value=BitRef(register="c", index=3), tag="t", namespace="result")
        out = substitute_stmt(p, _bundle_remap())
        assert out.value == BitRef(register="out", index=0)
        assert (out.tag, out.namespace) == ("t", "result")

    def test_return_expr_values(self) -> None:
        r = ReturnOp(values=(BitExpr(ref=BitRef(register="c", index=3)),))
        out = substitute_stmt(r, _bundle_remap())
        assert out.values == (BitExpr(ref=BitRef(register="out", index=0)),)

    def test_if_condition_nested_binary_unary(self) -> None:
        cond = BinaryExpr(
            op=BinaryOp.AND,
            left=UnaryExpr(op=UnaryOp.NOT, operand=BitExpr(ref=BitRef(register="c", index=3))),
            right=BitExpr(ref=BitRef(register="c", index=3)),
        )
        node = IfStmt(
            condition=cond,
            then_body=(GateOp(gate=GateKind.X, targets=(SlotRef(allocator="q", index=2),)),),
            else_body=(),
        )
        out = substitute_stmt(node, _bundle_remap())
        assert out.condition.left.operand == BitExpr(ref=BitRef(register="out", index=0))
        assert out.condition.right == BitExpr(ref=BitRef(register="out", index=0))
        assert out.then_body[0].targets == (SlotRef(allocator="d", index=0),)

    def test_while_condition_and_body(self) -> None:
        node = WhileStmt(
            condition=BitExpr(ref=BitRef(register="c", index=3)),
            body=(MeasureOp(targets=(SlotRef(allocator="q", index=5),), results=()),),
        )
        out = substitute_stmt(node, _bundle_remap())
        assert out.condition == BitExpr(ref=BitRef(register="out", index=0))
        assert out.body[0].targets == (SlotRef(allocator="d", index=1),)

    def test_for_bounds_and_body(self) -> None:
        node = ForStmt(
            variable="i",
            start=LiteralExpr(value=0),
            stop=BitExpr(ref=BitRef(register="c", index=3)),
            step=None,
            body=(GateOp(gate=GateKind.X, targets=(SlotRef(allocator="q", index=2),)),),
        )
        out = substitute_stmt(node, _bundle_remap())
        assert out.stop == BitExpr(ref=BitRef(register="out", index=0))
        assert out.body[0].targets == (SlotRef(allocator="d", index=0),)

    def test_repeat_and_parallel_body_recursion(self) -> None:
        inner = GateOp(gate=GateKind.X, targets=(SlotRef(allocator="q", index=2),))
        rep = RepeatStmt(count=3, body=(inner,))
        par = ParallelBlock(body=(inner,))
        ro = substitute_stmt(rep, _bundle_remap())
        po = substitute_stmt(par, _bundle_remap())
        assert ro.body[0].targets == (SlotRef(allocator="d", index=0),)
        assert po.body[0].targets == (SlotRef(allocator="d", index=0),)

    def test_nested_blockcall_args_remapped(self) -> None:
        bc = BlockCall(
            callee="inner",
            arg_bindings=(
                SingleQubitArg(slot=SlotRef(allocator="q", index=2)),
                SingleBitArg(bit=BitRef(register="c", index=3)),
                QubitBundleArg(
                    slots=(SlotRef(allocator="q", index=2), SlotRef(allocator="q", index=5)),
                ),
            ),
            out_bindings=(SingleBitArg(bit=BitRef(register="c", index=3)),),
        )
        out = substitute_stmt(bc, _bundle_remap())
        assert out.arg_bindings[0] == SingleQubitArg(slot=SlotRef(allocator="d", index=0))
        assert out.arg_bindings[1] == SingleBitArg(bit=BitRef(register="out", index=0))
        assert out.arg_bindings[2] == QubitBundleArg(
            slots=(SlotRef(allocator="d", index=0), SlotRef(allocator="d", index=1)),
        )
        assert out.out_bindings[0] == SingleBitArg(bit=BitRef(register="out", index=0))

    def test_comment_passthrough(self) -> None:
        c = CommentOp(text="hello")
        assert substitute_stmt(c, _bundle_remap()) is c


class TestPermuteAndPrepare:
    def test_permute_indexed_ref_via_slot(self) -> None:
        p = PermuteOp(sources=("q[2]", "q[5]"), targets=("q[5]", "q[2]"), add_comment=False)
        out = substitute_stmt(p, _bundle_remap())
        assert out.sources == ("d[0]", "d[1]")
        assert out.targets == ("d[1]", "d[0]")

    def test_permute_whole_name_renamed(self) -> None:
        remap = BodyRemap()
        remap.add_whole_alloc("q1", "qp", 7)
        p = PermuteOp(sources=("q1",), targets=("q1",), add_comment=False)
        out = substitute_stmt(p, remap)
        assert out.sources == ("qp",)

    def test_permute_bare_name_partial_rejected(self) -> None:
        p = PermuteOp(sources=("q",), targets=("q",), add_comment=False)
        with pytest.raises(BodySubstitutionError, match=r"partially bound"):
            substitute_stmt(p, _bundle_remap())

    def test_prepare_all_partial_rejected(self) -> None:
        prep = PrepareOp(allocator="q", slots=None)
        with pytest.raises(BodySubstitutionError, match=r"partially bound"):
            substitute_stmt(prep, _bundle_remap())

    def test_prepare_all_whole_renamed(self) -> None:
        remap = BodyRemap()
        remap.add_whole_alloc("q", "p", 3)
        prep = PrepareOp(allocator="q", slots=None)
        out = substitute_stmt(prep, remap)
        assert out.allocator == "p"
        assert out.slots is None

    def test_prepare_slots_remapped_preserves_source_order(self) -> None:
        # PZ(q[2], q[5]) under q[2]->d[0], q[5]->d[1] -> PZ(d, (0, 1)).
        # Exact order asserted: the lowering preserves source slot order;
        # sorted() would mask an ordering regression.
        prep = PrepareOp(allocator="q", slots=(2, 5))
        out = substitute_stmt(prep, _bundle_remap())
        assert out.allocator == "d"
        assert out.slots == (0, 1)

    def test_prepare_slots_reversed_map_preserves_source_order(self) -> None:
        # Reversed map q[2]->d[1], q[5]->d[0]: source order (2, 5) is kept, so
        # the emitted slots follow the source iteration -> (1, 0), NOT sorted.
        remap = BodyRemap()
        remap.add_slot(("q", 2), ("d", 1))
        remap.add_slot(("q", 5), ("d", 0))
        prep = PrepareOp(allocator="q", slots=(2, 5))
        out = substitute_stmt(prep, remap)
        assert out.allocator == "d"
        assert out.slots == (1, 0)

    def test_prepare_slots_multi_allocator_rejected(self) -> None:
        remap = BodyRemap()
        remap.add_slot(("q", 0), ("d", 0))
        remap.add_slot(("q", 1), ("e", 0))  # different destination allocator
        prep = PrepareOp(allocator="q", slots=(0, 1))
        with pytest.raises(BodySubstitutionError, match=r"multiple destination allocators"):
            substitute_stmt(prep, remap)


class TestBuilderConflictRejection:
    """An outer name bound in conflicting modes
    (whole + partial, or whole twice) is input aliasing -- reject at
    construction time, not silently let whole win at lookup. Protects the
    QASM/flatten path where Guppy's own linearity alias check never runs.
    """

    def test_whole_then_partial_same_name_rejected(self) -> None:
        remap = BodyRemap()
        remap.add_whole_alloc("q", "qp", 3)
        with pytest.raises(BodySubstitutionError, match=r"already bound whole"):
            remap.add_slot(("q", 0), ("d", 0))

    def test_partial_then_whole_same_name_rejected(self) -> None:
        remap = BodyRemap()
        remap.add_slot(("q", 0), ("d", 0))
        with pytest.raises(BodySubstitutionError, match=r"already bound partially"):
            remap.add_whole_alloc("q", "qp", 3)

    def test_whole_bound_twice_rejected(self) -> None:
        remap = BodyRemap()
        remap.add_whole_alloc("q", "a", 2)
        with pytest.raises(BodySubstitutionError, match=r"already bound whole"):
            remap.add_whole_alloc("q", "b", 2)

    def test_partial_then_bit_same_name_ok_distinct_names_ok(self) -> None:
        # Distinct outer names binding independently is NOT a conflict.
        remap = BodyRemap()
        remap.add_slot(("q", 0), ("d", 0))
        remap.add_bit(("c", 3), ("out", 0))  # different name -- fine
        remap.add_whole_alloc("r", "rp", 2)  # different name -- fine

    def test_repeated_exact_slot_source_rejected(self) -> None:
        """A repeated exact source slot must reject, not
        silently overwrite (a `[q[0], q[0]]` bundle or two single-qubit
        inputs aliased to the same outer slot -- corrupts body rewrite;
        Guppy rejects on linearity while QASM flatten emitted bad code).
        """
        remap = BodyRemap()
        remap.add_slot(("q", 0), ("d", 0))
        with pytest.raises(BodySubstitutionError, match=r"qubit slot \('q', 0\) is already bound"):
            remap.add_slot(("q", 0), ("d", 1))

    def test_repeated_exact_bit_source_rejected(self) -> None:
        remap = BodyRemap()
        remap.add_bit(("c", 0), ("out", 0))
        with pytest.raises(BodySubstitutionError, match=r"bit \('c', 0\) is already bound"):
            remap.add_bit(("c", 0), ("flag", 0))

    def test_distinct_slots_same_dst_register_ok(self) -> None:
        # Distinct source slots are fine even if they target the same
        # destination allocator at different indices (the flatten
        # PARAM->OUTER direction does exactly this for a bundle).
        remap = BodyRemap()
        remap.add_slot(("d", 0), ("q", 0))
        remap.add_slot(("d", 1), ("q", 2))


class TestRejectOnPartialNameLevel:
    def test_barrier_partial_name_rejected(self) -> None:
        b = BarrierOp(allocators=("q",))
        with pytest.raises(BodySubstitutionError, match=r"Barrier .*partially bound"):
            substitute_stmt(b, _bundle_remap())

    def test_empty_barrier_passthrough(self) -> None:
        # Barrier(a, d[i]) lowers TODAY to BarrierOp(allocators=()) -- nothing
        # partial left to substitute; global barrier passes through.
        b = BarrierOp(allocators=())
        out = substitute_stmt(b, _bundle_remap())
        assert out.allocators == ()

    def test_varexpr_partial_name_rejected(self) -> None:
        node = AssignOp(
            target=BitRef(register="c", index=3),
            value=VarExpr(name="q"),
        )
        with pytest.raises(BodySubstitutionError, match=r"variable expression .*partially bound"):
            substitute_stmt(node, _bundle_remap())

    def test_str_return_partial_name_rejected(self) -> None:
        r = ReturnOp(values=("q",))
        with pytest.raises(BodySubstitutionError, match=r"Return value .*partially bound"):
            substitute_stmt(r, _bundle_remap())

    def test_whole_name_unmapped_passes_through(self) -> None:
        # A name bound by no input (Block-local) is left unchanged.
        remap = BodyRemap()
        remap.add_slot(("q", 0), ("d", 0))
        b = BarrierOp(allocators=("local_reg",))
        out = substitute_stmt(b, remap)
        assert out.allocators == ("local_reg",)
