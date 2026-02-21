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

    def test_basic_location(self) -> None:
        """SourceLocation stores line and column."""
        loc = SourceLocation(line=10, column=5)
        assert loc.line == 10
        assert loc.column == 5
        assert loc.file is None

    def test_location_with_file(self) -> None:
        """SourceLocation with file formats as file:line:column."""
        loc = SourceLocation(line=10, column=5, file="test.py")
        assert str(loc) == "test.py:10:5"

    def test_location_without_file(self) -> None:
        """SourceLocation without file formats as line:column."""
        loc = SourceLocation(line=10, column=5)
        assert str(loc) == "10:5"

    def test_location_is_frozen(self) -> None:
        """SourceLocation is immutable."""
        loc = SourceLocation(line=10, column=5)
        with pytest.raises(AttributeError, match="cannot assign"):
            loc.line = 20


class TestReferences:
    """Tests for SlotRef and BitRef."""

    def test_slot_ref(self) -> None:
        """SlotRef stores allocator name and index."""
        ref = SlotRef(allocator="q", index=0)
        assert ref.allocator == "q"
        assert ref.index == 0
        assert str(ref) == "q[0]"

    def test_bit_ref(self) -> None:
        """BitRef stores register name and index."""
        ref = BitRef(register="c", index=1)
        assert ref.register == "c"
        assert ref.index == 1
        assert str(ref) == "c[1]"

    def test_refs_are_frozen(self) -> None:
        """Reference nodes are immutable."""
        ref = SlotRef(allocator="q", index=0)
        with pytest.raises(AttributeError, match="cannot assign"):
            ref.index = 1


class TestExpressions:
    """Tests for expression nodes."""

    def test_literal_int(self) -> None:
        """LiteralExpr can store integer values."""
        expr = LiteralExpr(value=42)
        assert expr.value == 42

    def test_literal_float(self) -> None:
        """LiteralExpr can store float values."""
        expr = LiteralExpr(value=3.14)
        assert expr.value == 3.14

    def test_literal_bool(self) -> None:
        """LiteralExpr can store boolean values."""
        expr = LiteralExpr(value=True)
        assert expr.value is True

    def test_var_expr(self) -> None:
        """VarExpr stores variable name."""
        expr = VarExpr(name="x")
        assert expr.name == "x"

    def test_bit_expr(self) -> None:
        """BitExpr wraps BitRef and exposes it as child."""
        ref = BitRef(register="c", index=0)
        expr = BitExpr(ref=ref)
        assert expr.ref == ref
        assert expr.children() == (ref,)

    def test_binary_expr(self) -> None:
        """BinaryExpr stores operator and operands."""
        left = LiteralExpr(value=1)
        right = LiteralExpr(value=2)
        expr = BinaryExpr(op=BinaryOp.ADD, left=left, right=right)

        assert expr.op == BinaryOp.ADD
        assert expr.left == left
        assert expr.right == right
        assert expr.children() == (left, right)

    def test_unary_expr(self) -> None:
        """UnaryExpr stores operator and operand."""
        operand = LiteralExpr(value=1)
        expr = UnaryExpr(op=UnaryOp.NEG, operand=operand)

        assert expr.op == UnaryOp.NEG
        assert expr.operand == operand
        assert expr.children() == (operand,)


class TestGateKind:
    """Tests for GateKind enum."""

    def test_single_qubit_gates(self) -> None:
        """Single-qubit gates have arity 1."""
        assert GateKind.H.arity == 1
        assert GateKind.X.arity == 1
        assert GateKind.RZ.arity == 1

    def test_two_qubit_gates(self) -> None:
        """Two-qubit gates have arity 2."""
        assert GateKind.CX.arity == 2
        assert GateKind.CZ.arity == 2
        assert GateKind.SZZ.arity == 2

    def test_parameterized_gates(self) -> None:
        """Parameterized gates have is_parameterized=True."""
        assert GateKind.RX.is_parameterized
        assert GateKind.RY.is_parameterized
        assert GateKind.RZ.is_parameterized
        assert GateKind.RZZ.is_parameterized

    def test_non_parameterized_gates(self) -> None:
        """Non-parameterized gates have is_parameterized=False."""
        assert not GateKind.H.is_parameterized
        assert not GateKind.CX.is_parameterized


class TestStatements:
    """Tests for statement nodes."""

    def test_gate_op_single_qubit(self) -> None:
        """GateOp for single-qubit gate has one target."""
        target = SlotRef(allocator="q", index=0)
        gate = GateOp(gate=GateKind.H, targets=(target,))

        assert gate.gate == GateKind.H
        assert gate.targets == (target,)
        assert gate.params == ()
        assert gate.children() == (target,)

    def test_gate_op_two_qubit(self) -> None:
        """GateOp for two-qubit gate has two targets."""
        t1 = SlotRef(allocator="q", index=0)
        t2 = SlotRef(allocator="q", index=1)
        gate = GateOp(gate=GateKind.CX, targets=(t1, t2))

        assert gate.gate == GateKind.CX
        assert gate.targets == (t1, t2)
        assert gate.children() == (t1, t2)

    def test_gate_op_with_params(self) -> None:
        """GateOp can have parameter expressions."""
        target = SlotRef(allocator="q", index=0)
        angle = LiteralExpr(value=3.14)
        gate = GateOp(gate=GateKind.RZ, targets=(target,), params=(angle,))

        assert gate.params == (angle,)
        assert gate.children() == (target, angle)

    def test_prepare_op(self) -> None:
        """PrepareOp stores allocator and specific slots."""
        prep = PrepareOp(allocator="q", slots=(0, 1))
        assert prep.allocator == "q"
        assert prep.slots == (0, 1)

    def test_prepare_op_all(self) -> None:
        """PrepareOp with slots=None prepares all slots."""
        prep = PrepareOp(allocator="q", slots=None)
        assert prep.slots is None

    def test_measure_op(self) -> None:
        """MeasureOp stores targets and result destinations."""
        target = SlotRef(allocator="q", index=0)
        result = BitRef(register="c", index=0)
        measure = MeasureOp(targets=(target,), results=(result,))

        assert measure.targets == (target,)
        assert measure.results == (result,)
        assert measure.children() == (target, result)

    def test_assign_op_to_bit(self) -> None:
        """AssignOp can assign to BitRef."""
        target = BitRef(register="c", index=0)
        value = LiteralExpr(value=1)
        assign = AssignOp(target=target, value=value)

        assert assign.target == target
        assert assign.value == value
        assert target in assign.children()
        assert value in assign.children()

    def test_assign_op_to_var(self) -> None:
        """AssignOp can assign to variable name."""
        value = LiteralExpr(value=42)
        assign = AssignOp(target="x", value=value)

        assert assign.target == "x"
        assert assign.children() == [value]

    def test_barrier_op(self) -> None:
        """BarrierOp stores allocator names."""
        barrier = BarrierOp(allocators=("q", "r"))
        assert barrier.allocators == ("q", "r")

    def test_comment_op(self) -> None:
        """CommentOp stores comment text."""
        comment = CommentOp(text="This is a comment")
        assert comment.text == "This is a comment"

    def test_return_op(self) -> None:
        """ReturnOp stores return values."""
        ret = ReturnOp(values=("q", "c"))
        assert ret.values == ("q", "c")


class TestControlFlow:
    """Tests for control flow nodes."""

    def test_if_stmt_then_only(self) -> None:
        """IfStmt with only then branch has empty else_body."""
        cond = LiteralExpr(value=True)
        gate = GateOp(gate=GateKind.H, targets=(SlotRef(allocator="q", index=0),))
        if_stmt = IfStmt(condition=cond, then_body=(gate,))

        assert if_stmt.condition == cond
        assert if_stmt.then_body == (gate,)
        assert if_stmt.else_body == ()

    def test_if_stmt_with_else(self) -> None:
        """IfStmt can have both then and else branches."""
        cond = LiteralExpr(value=True)
        then_gate = GateOp(gate=GateKind.H, targets=(SlotRef(allocator="q", index=0),))
        else_gate = GateOp(gate=GateKind.X, targets=(SlotRef(allocator="q", index=0),))
        if_stmt = IfStmt(condition=cond, then_body=(then_gate,), else_body=(else_gate,))

        assert if_stmt.else_body == (else_gate,)
        assert cond in if_stmt.children()
        assert then_gate in if_stmt.children()
        assert else_gate in if_stmt.children()

    def test_while_stmt(self) -> None:
        """WhileStmt stores condition and body."""
        cond = LiteralExpr(value=True)
        gate = GateOp(gate=GateKind.H, targets=(SlotRef(allocator="q", index=0),))
        while_stmt = WhileStmt(condition=cond, body=(gate,))

        assert while_stmt.condition == cond
        assert while_stmt.body == (gate,)

    def test_for_stmt(self) -> None:
        """ForStmt stores variable, range, and body."""
        start = LiteralExpr(value=0)
        stop = LiteralExpr(value=10)
        gate = GateOp(gate=GateKind.H, targets=(SlotRef(allocator="q", index=0),))
        for_stmt = ForStmt(variable="i", start=start, stop=stop, body=(gate,))

        assert for_stmt.variable == "i"
        assert for_stmt.start == start
        assert for_stmt.stop == stop
        assert for_stmt.step is None
        assert for_stmt.body == (gate,)

    def test_for_stmt_with_step(self) -> None:
        """ForStmt can have explicit step value."""
        start = LiteralExpr(value=0)
        stop = LiteralExpr(value=10)
        step = LiteralExpr(value=2)
        for_stmt = ForStmt(variable="i", start=start, stop=stop, step=step, body=())

        assert for_stmt.step == step
        assert step in for_stmt.children()

    def test_repeat_stmt(self) -> None:
        """RepeatStmt stores count and body."""
        gate = GateOp(gate=GateKind.H, targets=(SlotRef(allocator="q", index=0),))
        repeat = RepeatStmt(count=5, body=(gate,))

        assert repeat.count == 5
        assert repeat.body == (gate,)

    def test_parallel_block(self) -> None:
        """ParallelBlock stores operations to execute in parallel."""
        gate1 = GateOp(gate=GateKind.H, targets=(SlotRef(allocator="q", index=0),))
        gate2 = GateOp(gate=GateKind.H, targets=(SlotRef(allocator="q", index=1),))
        parallel = ParallelBlock(body=(gate1, gate2))

        assert parallel.body == (gate1, gate2)


class TestDeclarations:
    """Tests for declaration nodes."""

    def test_allocator_decl(self) -> None:
        """AllocatorDecl stores name and capacity."""
        decl = AllocatorDecl(name="q", capacity=10)
        assert decl.name == "q"
        assert decl.capacity == 10
        assert decl.parent is None

    def test_allocator_decl_with_parent(self) -> None:
        """AllocatorDecl can have parent allocator."""
        decl = AllocatorDecl(name="data", capacity=7, parent="base")
        assert decl.parent == "base"

    def test_register_decl(self) -> None:
        """RegisterDecl stores name and size."""
        decl = RegisterDecl(name="c", size=5)
        assert decl.name == "c"
        assert decl.size == 5
        assert decl.is_result is True

    def test_register_decl_not_result(self) -> None:
        """RegisterDecl can mark register as not a result."""
        decl = RegisterDecl(name="scratch", size=3, is_result=False)
        assert decl.is_result is False


class TestProgram:
    """Tests for Program node."""

    def test_empty_program(self) -> None:
        """Empty program has default empty values."""
        prog = Program(name="test")
        assert prog.name == "test"
        assert prog.declarations == ()
        assert prog.body == ()
        assert prog.returns == ()
        assert prog.allocator is None

    def test_program_with_declarations(self) -> None:
        """Program can have declarations."""
        alloc = AllocatorDecl(name="q", capacity=5)
        reg = RegisterDecl(name="c", size=5)
        prog = Program(name="test", declarations=(alloc, reg))

        assert prog.declarations == (alloc, reg)

    def test_program_with_body(self) -> None:
        """Program can have body statements."""
        prep = PrepareOp(allocator="q", slots=(0,))
        gate = GateOp(gate=GateKind.H, targets=(SlotRef(allocator="q", index=0),))
        prog = Program(name="test", body=(prep, gate))

        assert prog.body == (prep, gate)

    def test_program_with_base_allocator(self) -> None:
        """Program can have base allocator."""
        base = AllocatorDecl(name="base", capacity=100)
        prog = Program(name="test", allocator=base)

        assert prog.allocator == base

    def test_program_get_allocator(self) -> None:
        """Program.get_allocator finds allocators by name."""
        base = AllocatorDecl(name="base", capacity=100)
        child = AllocatorDecl(name="data", capacity=10, parent="base")
        prog = Program(name="test", allocator=base, declarations=(child,))

        assert prog.get_allocator("base") == base
        assert prog.get_allocator("data") == child
        assert prog.get_allocator("unknown") is None

    def test_program_get_register(self) -> None:
        """Program.get_register finds registers by name."""
        reg = RegisterDecl(name="c", size=5)
        prog = Program(name="test", declarations=(reg,))

        assert prog.get_register("c") == reg
        assert prog.get_register("unknown") is None

    def test_program_children(self) -> None:
        """Program.children returns all child nodes."""
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
