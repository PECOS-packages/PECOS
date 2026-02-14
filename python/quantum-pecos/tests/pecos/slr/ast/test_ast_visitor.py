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

"""Tests for AST visitor pattern."""

import pytest

from pecos.slr.ast import (
    AllocatorDecl,
    BaseVisitor,
    CollectingVisitor,
    GateKind,
    GateOp,
    IfStmt,
    LiteralExpr,
    MeasureOp,
    PrepareOp,
    Program,
    SlotRef,
    VoidVisitor,
)


class TestVoidVisitor:
    """Tests for VoidVisitor."""

    def test_void_visitor_returns_none(self):
        visitor = VoidVisitor()
        gate = GateOp(gate=GateKind.H, targets=(SlotRef(allocator="q", index=0),))

        result = visitor.visit(gate)
        assert result is None

    def test_void_visitor_traverses_program(self):
        visitor = VoidVisitor()
        prep = PrepareOp(allocator="q", slots=(0,))
        gate = GateOp(gate=GateKind.H, targets=(SlotRef(allocator="q", index=0),))
        prog = Program(name="test", body=(prep, gate))

        result = visitor.visit(prog)
        assert result is None


class TestCollectingVisitor:
    """Tests for CollectingVisitor."""

    def test_collecting_visitor_empty(self):
        class GateCollector(CollectingVisitor[str]):
            def visit_gate(self, node: GateOp) -> list[str]:
                return [node.gate.name]

        visitor = GateCollector()
        prog = Program(name="empty")

        result = visitor.visit(prog)
        assert result == []

    def test_collecting_visitor_collects_gates(self):
        class GateCollector(CollectingVisitor[str]):
            def visit_gate(self, node: GateOp) -> list[str]:
                return [node.gate.name]

        visitor = GateCollector()
        gate1 = GateOp(gate=GateKind.H, targets=(SlotRef(allocator="q", index=0),))
        gate2 = GateOp(gate=GateKind.CX, targets=(SlotRef(allocator="q", index=0), SlotRef(allocator="q", index=1)))
        prog = Program(name="test", body=(gate1, gate2))

        result = visitor.visit(prog)
        assert "H" in result
        assert "CX" in result


class TestCustomVisitor:
    """Tests for custom visitor implementations."""

    def test_gate_counter(self):
        class GateCounter(BaseVisitor[int]):
            def default_result(self) -> int:
                return 0

            def combine_results(self, results: list[int]) -> int:
                return sum(results)

            def visit_gate(self, node: GateOp) -> int:
                return 1

        visitor = GateCounter()
        gate1 = GateOp(gate=GateKind.H, targets=(SlotRef(allocator="q", index=0),))
        gate2 = GateOp(gate=GateKind.X, targets=(SlotRef(allocator="q", index=1),))
        gate3 = GateOp(gate=GateKind.CX, targets=(SlotRef(allocator="q", index=0), SlotRef(allocator="q", index=1)))
        prog = Program(name="test", body=(gate1, gate2, gate3))

        count = visitor.visit(prog)
        assert count == 3

    def test_depth_calculator(self):
        class DepthCalculator(BaseVisitor[int]):
            def default_result(self) -> int:
                return 0

            def combine_results(self, results: list[int]) -> int:
                return max(results) if results else 0

            def visit_if(self, node: IfStmt) -> int:
                return 1 + self.combine_results(self.visit_children(node))

        visitor = DepthCalculator()

        # Create nested if statements
        inner_gate = GateOp(gate=GateKind.H, targets=(SlotRef(allocator="q", index=0),))
        inner_if = IfStmt(
            condition=LiteralExpr(value=True),
            then_body=(inner_gate,),
        )
        outer_if = IfStmt(
            condition=LiteralExpr(value=True),
            then_body=(inner_if,),
        )
        prog = Program(name="test", body=(outer_if,))

        depth = visitor.visit(prog)
        assert depth == 2

    def test_qubit_usage_tracker(self):
        class QubitTracker(BaseVisitor[set]):
            def default_result(self) -> set:
                return set()

            def combine_results(self, results: list[set]) -> set:
                combined = set()
                for r in results:
                    combined.update(r)
                return combined

            def visit_slot_ref(self, node: SlotRef) -> set:
                return {(node.allocator, node.index)}

        visitor = QubitTracker()
        gate1 = GateOp(gate=GateKind.H, targets=(SlotRef(allocator="q", index=0),))
        gate2 = GateOp(gate=GateKind.CX, targets=(SlotRef(allocator="q", index=0), SlotRef(allocator="q", index=1)))
        gate3 = GateOp(gate=GateKind.X, targets=(SlotRef(allocator="r", index=0),))
        prog = Program(name="test", body=(gate1, gate2, gate3))

        used = visitor.visit(prog)
        assert ("q", 0) in used
        assert ("q", 1) in used
        assert ("r", 0) in used

    def test_code_generator_simple(self):
        class SimpleCodeGen(BaseVisitor[str]):
            def default_result(self) -> str:
                return ""

            def combine_results(self, results: list[str]) -> str:
                return "\n".join(r for r in results if r)

            def visit_gate(self, node: GateOp) -> str:
                targets = ", ".join(str(t) for t in node.targets)
                return f"{node.gate.name}({targets})"

            def visit_prepare(self, node: PrepareOp) -> str:
                if node.slots:
                    slots = ", ".join(str(s) for s in node.slots)
                    return f"Prep({node.allocator}[{slots}])"
                return f"Prep({node.allocator}[*])"

            def visit_measure(self, node: MeasureOp) -> str:
                targets = ", ".join(str(t) for t in node.targets)
                return f"Measure({targets})"

        visitor = SimpleCodeGen()
        prep = PrepareOp(allocator="q", slots=(0, 1))
        gate1 = GateOp(gate=GateKind.H, targets=(SlotRef(allocator="q", index=0),))
        gate2 = GateOp(gate=GateKind.CX, targets=(SlotRef(allocator="q", index=0), SlotRef(allocator="q", index=1)))
        measure = MeasureOp(targets=(SlotRef(allocator="q", index=0),))
        prog = Program(name="test", body=(prep, gate1, gate2, measure))

        code = visitor.visit(prog)
        assert "Prep(q[0, 1])" in code
        assert "H(q[0])" in code
        assert "CX(q[0], q[1])" in code
        assert "Measure(q[0])" in code


class TestVisitorTraversal:
    """Tests for visitor traversal behavior."""

    def test_visits_all_children(self):
        class VisitTracker(VoidVisitor):
            def __init__(self):
                self.visited = []

            def visit_gate(self, node: GateOp) -> None:
                self.visited.append(("gate", node.gate.name))
                return super().visit_gate(node)

            def visit_prepare(self, node: PrepareOp) -> None:
                self.visited.append(("prepare", node.allocator))
                return super().visit_prepare(node)

            def visit_slot_ref(self, node: SlotRef) -> None:
                self.visited.append(("slot", f"{node.allocator}[{node.index}]"))
                return super().visit_slot_ref(node)

        visitor = VisitTracker()
        prep = PrepareOp(allocator="q", slots=(0,))
        gate = GateOp(gate=GateKind.H, targets=(SlotRef(allocator="q", index=0),))
        prog = Program(name="test", body=(prep, gate))

        visitor.visit(prog)

        assert ("prepare", "q") in visitor.visited
        assert ("gate", "H") in visitor.visited
        assert ("slot", "q[0]") in visitor.visited

    def test_visits_control_flow_children(self):
        class VisitTracker(VoidVisitor):
            def __init__(self):
                self.visited = []

            def visit_gate(self, node: GateOp) -> None:
                self.visited.append(node.gate.name)
                return super().visit_gate(node)

        visitor = VisitTracker()
        then_gate = GateOp(gate=GateKind.H, targets=(SlotRef(allocator="q", index=0),))
        else_gate = GateOp(gate=GateKind.X, targets=(SlotRef(allocator="q", index=0),))
        if_stmt = IfStmt(
            condition=LiteralExpr(value=True),
            then_body=(then_gate,),
            else_body=(else_gate,),
        )
        prog = Program(name="test", body=(if_stmt,))

        visitor.visit(prog)

        assert "H" in visitor.visited
        assert "X" in visitor.visited
