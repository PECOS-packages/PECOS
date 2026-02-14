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

"""Tests for AST node definitions."""

import pytest

from pecos.slr.ast import (
    AllocatorDecl,
    AssignOp,
    BarrierOp,
    BinaryExpr,
    BinaryOp,
    BitExpr,
    BitRef,
    CommentOp,
    ForStmt,
    GateKind,
    GateOp,
    IfStmt,
    LiteralExpr,
    MeasureOp,
    ParallelBlock,
    PrepareOp,
    Program,
    RegisterDecl,
    RepeatStmt,
    ReturnOp,
    SlotRef,
    SourceLocation,
    UnaryExpr,
    UnaryOp,
    VarExpr,
    WhileStmt,
)


class TestSourceLocation:
    """Tests for SourceLocation."""

    def test_basic_location(self):
        loc = SourceLocation(line=10, column=5)
        assert loc.line == 10
        assert loc.column == 5
        assert loc.file is None

    def test_location_with_file(self):
        loc = SourceLocation(line=10, column=5, file="test.py")
        assert str(loc) == "test.py:10:5"

    def test_location_without_file(self):
        loc = SourceLocation(line=10, column=5)
        assert str(loc) == "10:5"

    def test_location_is_frozen(self):
        loc = SourceLocation(line=10, column=5)
        with pytest.raises(Exception):  # FrozenInstanceError
            loc.line = 20


class TestReferences:
    """Tests for SlotRef and BitRef."""

    def test_slot_ref(self):
        ref = SlotRef(allocator="q", index=0)
        assert ref.allocator == "q"
        assert ref.index == 0
        assert str(ref) == "q[0]"

    def test_bit_ref(self):
        ref = BitRef(register="c", index=1)
        assert ref.register == "c"
        assert ref.index == 1
        assert str(ref) == "c[1]"

    def test_refs_are_frozen(self):
        ref = SlotRef(allocator="q", index=0)
        with pytest.raises(Exception):
            ref.index = 1


class TestExpressions:
    """Tests for expression nodes."""

    def test_literal_int(self):
        expr = LiteralExpr(value=42)
        assert expr.value == 42

    def test_literal_float(self):
        expr = LiteralExpr(value=3.14)
        assert expr.value == 3.14

    def test_literal_bool(self):
        expr = LiteralExpr(value=True)
        assert expr.value is True

    def test_var_expr(self):
        expr = VarExpr(name="x")
        assert expr.name == "x"

    def test_bit_expr(self):
        ref = BitRef(register="c", index=0)
        expr = BitExpr(ref=ref)
        assert expr.ref == ref
        assert expr.children() == (ref,)

    def test_binary_expr(self):
        left = LiteralExpr(value=1)
        right = LiteralExpr(value=2)
        expr = BinaryExpr(op=BinaryOp.ADD, left=left, right=right)

        assert expr.op == BinaryOp.ADD
        assert expr.left == left
        assert expr.right == right
        assert expr.children() == (left, right)

    def test_unary_expr(self):
        operand = LiteralExpr(value=1)
        expr = UnaryExpr(op=UnaryOp.NEG, operand=operand)

        assert expr.op == UnaryOp.NEG
        assert expr.operand == operand
        assert expr.children() == (operand,)


class TestGateKind:
    """Tests for GateKind enum."""

    def test_single_qubit_gates(self):
        assert GateKind.H.arity == 1
        assert GateKind.X.arity == 1
        assert GateKind.RZ.arity == 1

    def test_two_qubit_gates(self):
        assert GateKind.CX.arity == 2
        assert GateKind.CZ.arity == 2
        assert GateKind.SZZ.arity == 2

    def test_parameterized_gates(self):
        assert GateKind.RX.is_parameterized
        assert GateKind.RY.is_parameterized
        assert GateKind.RZ.is_parameterized
        assert GateKind.RZZ.is_parameterized

    def test_non_parameterized_gates(self):
        assert not GateKind.H.is_parameterized
        assert not GateKind.CX.is_parameterized


class TestStatements:
    """Tests for statement nodes."""

    def test_gate_op_single_qubit(self):
        target = SlotRef(allocator="q", index=0)
        gate = GateOp(gate=GateKind.H, targets=(target,))

        assert gate.gate == GateKind.H
        assert gate.targets == (target,)
        assert gate.params == ()
        assert gate.children() == (target,)

    def test_gate_op_two_qubit(self):
        t1 = SlotRef(allocator="q", index=0)
        t2 = SlotRef(allocator="q", index=1)
        gate = GateOp(gate=GateKind.CX, targets=(t1, t2))

        assert gate.gate == GateKind.CX
        assert gate.targets == (t1, t2)
        assert gate.children() == (t1, t2)

    def test_gate_op_with_params(self):
        target = SlotRef(allocator="q", index=0)
        angle = LiteralExpr(value=3.14)
        gate = GateOp(gate=GateKind.RZ, targets=(target,), params=(angle,))

        assert gate.params == (angle,)
        assert gate.children() == (target, angle)

    def test_prepare_op(self):
        prep = PrepareOp(allocator="q", slots=(0, 1))
        assert prep.allocator == "q"
        assert prep.slots == (0, 1)

    def test_prepare_op_all(self):
        prep = PrepareOp(allocator="q", slots=None)
        assert prep.slots is None

    def test_measure_op(self):
        target = SlotRef(allocator="q", index=0)
        result = BitRef(register="c", index=0)
        measure = MeasureOp(targets=(target,), results=(result,))

        assert measure.targets == (target,)
        assert measure.results == (result,)
        assert measure.children() == (target, result)

    def test_assign_op_to_bit(self):
        target = BitRef(register="c", index=0)
        value = LiteralExpr(value=1)
        assign = AssignOp(target=target, value=value)

        assert assign.target == target
        assert assign.value == value
        assert target in assign.children()
        assert value in assign.children()

    def test_assign_op_to_var(self):
        value = LiteralExpr(value=42)
        assign = AssignOp(target="x", value=value)

        assert assign.target == "x"
        assert assign.children() == [value]

    def test_barrier_op(self):
        barrier = BarrierOp(allocators=("q", "r"))
        assert barrier.allocators == ("q", "r")

    def test_comment_op(self):
        comment = CommentOp(text="This is a comment")
        assert comment.text == "This is a comment"

    def test_return_op(self):
        ret = ReturnOp(values=("q", "c"))
        assert ret.values == ("q", "c")


class TestControlFlow:
    """Tests for control flow nodes."""

    def test_if_stmt_then_only(self):
        cond = LiteralExpr(value=True)
        gate = GateOp(gate=GateKind.H, targets=(SlotRef(allocator="q", index=0),))
        if_stmt = IfStmt(condition=cond, then_body=(gate,))

        assert if_stmt.condition == cond
        assert if_stmt.then_body == (gate,)
        assert if_stmt.else_body == ()

    def test_if_stmt_with_else(self):
        cond = LiteralExpr(value=True)
        then_gate = GateOp(gate=GateKind.H, targets=(SlotRef(allocator="q", index=0),))
        else_gate = GateOp(gate=GateKind.X, targets=(SlotRef(allocator="q", index=0),))
        if_stmt = IfStmt(condition=cond, then_body=(then_gate,), else_body=(else_gate,))

        assert if_stmt.else_body == (else_gate,)
        assert cond in if_stmt.children()
        assert then_gate in if_stmt.children()
        assert else_gate in if_stmt.children()

    def test_while_stmt(self):
        cond = LiteralExpr(value=True)
        gate = GateOp(gate=GateKind.H, targets=(SlotRef(allocator="q", index=0),))
        while_stmt = WhileStmt(condition=cond, body=(gate,))

        assert while_stmt.condition == cond
        assert while_stmt.body == (gate,)

    def test_for_stmt(self):
        start = LiteralExpr(value=0)
        stop = LiteralExpr(value=10)
        gate = GateOp(gate=GateKind.H, targets=(SlotRef(allocator="q", index=0),))
        for_stmt = ForStmt(variable="i", start=start, stop=stop, body=(gate,))

        assert for_stmt.variable == "i"
        assert for_stmt.start == start
        assert for_stmt.stop == stop
        assert for_stmt.step is None
        assert for_stmt.body == (gate,)

    def test_for_stmt_with_step(self):
        start = LiteralExpr(value=0)
        stop = LiteralExpr(value=10)
        step = LiteralExpr(value=2)
        for_stmt = ForStmt(variable="i", start=start, stop=stop, step=step, body=())

        assert for_stmt.step == step
        assert step in for_stmt.children()

    def test_repeat_stmt(self):
        gate = GateOp(gate=GateKind.H, targets=(SlotRef(allocator="q", index=0),))
        repeat = RepeatStmt(count=5, body=(gate,))

        assert repeat.count == 5
        assert repeat.body == (gate,)

    def test_parallel_block(self):
        gate1 = GateOp(gate=GateKind.H, targets=(SlotRef(allocator="q", index=0),))
        gate2 = GateOp(gate=GateKind.H, targets=(SlotRef(allocator="q", index=1),))
        parallel = ParallelBlock(body=(gate1, gate2))

        assert parallel.body == (gate1, gate2)


class TestDeclarations:
    """Tests for declaration nodes."""

    def test_allocator_decl(self):
        decl = AllocatorDecl(name="q", capacity=10)
        assert decl.name == "q"
        assert decl.capacity == 10
        assert decl.parent is None

    def test_allocator_decl_with_parent(self):
        decl = AllocatorDecl(name="data", capacity=7, parent="base")
        assert decl.parent == "base"

    def test_register_decl(self):
        decl = RegisterDecl(name="c", size=5)
        assert decl.name == "c"
        assert decl.size == 5
        assert decl.is_result is True

    def test_register_decl_not_result(self):
        decl = RegisterDecl(name="scratch", size=3, is_result=False)
        assert decl.is_result is False


class TestProgram:
    """Tests for Program node."""

    def test_empty_program(self):
        prog = Program(name="test")
        assert prog.name == "test"
        assert prog.declarations == ()
        assert prog.body == ()
        assert prog.returns == ()
        assert prog.allocator is None

    def test_program_with_declarations(self):
        alloc = AllocatorDecl(name="q", capacity=5)
        reg = RegisterDecl(name="c", size=5)
        prog = Program(name="test", declarations=(alloc, reg))

        assert prog.declarations == (alloc, reg)

    def test_program_with_body(self):
        prep = PrepareOp(allocator="q", slots=(0,))
        gate = GateOp(gate=GateKind.H, targets=(SlotRef(allocator="q", index=0),))
        prog = Program(name="test", body=(prep, gate))

        assert prog.body == (prep, gate)

    def test_program_with_base_allocator(self):
        base = AllocatorDecl(name="base", capacity=100)
        prog = Program(name="test", allocator=base)

        assert prog.allocator == base

    def test_program_get_allocator(self):
        base = AllocatorDecl(name="base", capacity=100)
        child = AllocatorDecl(name="data", capacity=10, parent="base")
        prog = Program(name="test", allocator=base, declarations=(child,))

        assert prog.get_allocator("base") == base
        assert prog.get_allocator("data") == child
        assert prog.get_allocator("unknown") is None

    def test_program_get_register(self):
        reg = RegisterDecl(name="c", size=5)
        prog = Program(name="test", declarations=(reg,))

        assert prog.get_register("c") == reg
        assert prog.get_register("unknown") is None

    def test_program_children(self):
        base = AllocatorDecl(name="q", capacity=5)
        reg = RegisterDecl(name="c", size=5)
        gate = GateOp(gate=GateKind.H, targets=(SlotRef(allocator="q", index=0),))
        prog = Program(
            name="test",
            allocator=base,
            declarations=(reg,),
            body=(gate,),
        )

        children = prog.children()
        assert base in children
        assert reg in children
        assert gate in children
