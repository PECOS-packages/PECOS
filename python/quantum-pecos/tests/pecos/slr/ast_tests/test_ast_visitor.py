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

    def test_void_visitor_returns_none(self) -> None:
        """VoidVisitor returns None for all visits."""
        visitor = VoidVisitor()
        gate = GateOp(gate=GateKind.H, targets=(SlotRef(allocator="q", index=0),))

        result = visitor.visit(gate)
        assert result is None

    def test_void_visitor_traverses_program(self) -> None:
        """VoidVisitor traverses entire program structure."""
        visitor = VoidVisitor()
        prep = PrepareOp(allocator="q", slots=(0,))
        gate = GateOp(gate=GateKind.H, targets=(SlotRef(allocator="q", index=0),))
        prog = Program(name="test", body=(prep, gate))

        result = visitor.visit(prog)
        assert result is None


class TestCollectingVisitor:
    """Tests for CollectingVisitor."""

    def test_collecting_visitor_empty(self) -> None:
        """CollectingVisitor returns empty list for empty program."""

        class GateCollector(CollectingVisitor[str]):
            def visit_gate(self, node: GateOp) -> list[str]:
                return [node.gate.name]

        visitor = GateCollector()
        prog = Program(name="empty")

        result = visitor.visit(prog)
        assert result == []

    def test_collecting_visitor_collects_gates(self) -> None:
        """CollectingVisitor collects items from all visited nodes."""

        class GateCollector(CollectingVisitor[str]):
            def visit_gate(self, node: GateOp) -> list[str]:
                return [node.gate.name]

        visitor = GateCollector()
        gate1 = GateOp(gate=GateKind.H, targets=(SlotRef(allocator="q", index=0),))
        gate2 = GateOp(
            gate=GateKind.CX,
            targets=(SlotRef(allocator="q", index=0), SlotRef(allocator="q", index=1)),
        )
        prog = Program(name="test", body=(gate1, gate2))

        result = visitor.visit(prog)
        assert "H" in result
        assert "CX" in result


class TestCustomVisitor:
    """Tests for custom visitor implementations."""

    def test_gate_counter(self) -> None:
        """Visitor can count gates in AST."""

        class GateCounter(BaseVisitor[int]):
            def default_result(self) -> int:
                return 0

            def combine_results(self, results: list[int]) -> int:
                return sum(results)

            def visit_gate(self, _node: GateOp) -> int:
                return 1

        visitor = GateCounter()
        gate1 = GateOp(gate=GateKind.H, targets=(SlotRef(allocator="q", index=0),))
        gate2 = GateOp(gate=GateKind.X, targets=(SlotRef(allocator="q", index=1),))
        gate3 = GateOp(
            gate=GateKind.CX,
            targets=(SlotRef(allocator="q", index=0), SlotRef(allocator="q", index=1)),
        )
        prog = Program(name="test", body=(gate1, gate2, gate3))

        count = visitor.visit(prog)
        assert count == 3

    def test_depth_calculator(self) -> None:
        """Visitor can calculate AST depth."""

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

    def test_qubit_usage_tracker(self) -> None:
        """Visitor can track qubit usage across AST."""

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
        gate2 = GateOp(
            gate=GateKind.CX,
            targets=(SlotRef(allocator="q", index=0), SlotRef(allocator="q", index=1)),
        )
        gate3 = GateOp(gate=GateKind.X, targets=(SlotRef(allocator="r", index=0),))
        prog = Program(name="test", body=(gate1, gate2, gate3))

        used = visitor.visit(prog)
        assert ("q", 0) in used
        assert ("q", 1) in used
        assert ("r", 0) in used

    def test_code_generator_simple(self) -> None:
        """Visitor can generate code from AST."""

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
                    return f"PZ({node.allocator}[{slots}])"
                return f"PZ({node.allocator}[*])"

            def visit_measure(self, node: MeasureOp) -> str:
                targets = ", ".join(str(t) for t in node.targets)
                return f"Measure({targets})"

        visitor = SimpleCodeGen()
        prep = PrepareOp(allocator="q", slots=(0, 1))
        gate1 = GateOp(gate=GateKind.H, targets=(SlotRef(allocator="q", index=0),))
        gate2 = GateOp(
            gate=GateKind.CX,
            targets=(SlotRef(allocator="q", index=0), SlotRef(allocator="q", index=1)),
        )
        measure = MeasureOp(targets=(SlotRef(allocator="q", index=0),))
        prog = Program(name="test", body=(prep, gate1, gate2, measure))

        code = visitor.visit(prog)
        assert "PZ(q[0, 1])" in code
        assert "H(q[0])" in code
        assert "CX(q[0], q[1])" in code
        assert "Measure(q[0])" in code


class TestVisitorTraversal:
    """Tests for visitor traversal behavior."""

    def test_visits_all_children(self) -> None:
        """Visitor traverses all child nodes."""

        class VisitTracker(VoidVisitor):
            def __init__(self) -> None:
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

    def test_visits_control_flow_children(self) -> None:
        """Visitor traverses control flow branches."""

        class VisitTracker(VoidVisitor):
            def __init__(self) -> None:
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


class TestVisitorDispatchCompleteness:
    """Safety net for the centralized `_DISPATCH` (replaced per-node
    `accept()`): a new concrete AST node without a dispatch entry must
    fail loudly here -- this is what catching a missing `accept()`
    used to do implicitly.
    """

    @staticmethod
    def _concrete_node_names() -> set[str]:
        import pecos.slr.ast.nodes as nodes_mod

        def all_subclasses(cls: type) -> set[type]:
            out: set[type] = set()
            for sub in cls.__subclasses__():
                out.add(sub)
                out |= all_subclasses(sub)
            return out

        # Only nodes shipped in `pecos.slr.ast.nodes` -- user/test
        # subclasses (e.g. `MyGate(GateOp)`) are intentionally resolved
        # by MRO in BaseVisitor.visit and must NOT be required in
        # _DISPATCH, so scope the enumeration to the nodes module.
        nodes = {c for c in all_subclasses(nodes_mod.AstNode) if c.__module__ == nodes_mod.__name__}
        # Intermediate/abstract bases (AstNode, Expression, Statement,
        # TypeExpr, Declaration, BlockArg) are never instantiated directly
        # and are correctly absent from _DISPATCH.
        bases = {base for cls in nodes for base in cls.__bases__ if base in nodes or base is nodes_mod.AstNode}
        return {cls.__name__ for cls in nodes if cls not in bases}

    def test_every_concrete_node_has_a_dispatch_entry(self) -> None:
        from pecos.slr.ast.visitor import _DISPATCH

        missing = sorted(self._concrete_node_names() - set(_DISPATCH))
        assert (
            not missing
        ), f"concrete AST nodes with no _DISPATCH entry: {missing} (add them to pecos.slr.ast.visitor._DISPATCH)"

    def test_no_stale_or_invalid_dispatch_entries(self) -> None:
        from pecos.slr.ast.visitor import _DISPATCH, BaseVisitor

        concrete = self._concrete_node_names()
        stale = sorted(set(_DISPATCH) - concrete)
        assert not stale, f"_DISPATCH keys that are not concrete nodes: {stale}"
        bad = sorted(v for v in _DISPATCH.values() if not callable(getattr(BaseVisitor, v, None)))
        assert not bad, f"_DISPATCH values not methods on BaseVisitor: {bad}"

    def test_subclass_of_concrete_node_dispatches_via_mro(self) -> None:
        """Visitor-refactor rationale: the old `node.accept(self)`
        double-dispatch was inherited, so a user subclass of a concrete
        node dispatched to the base node's `visit_*`. MRO lookup in
        `BaseVisitor.visit` must preserve that (a bare class-name match
        would wrongly raise).
        """

        class MyGate(GateOp):
            pass

        class Recorder(BaseVisitor[str]):
            def visit_gate(self, node: GateOp) -> str:
                return f"gate:{node.gate.name}"

            def default_result(self) -> str:
                return ""

        node = MyGate(gate=GateKind.H, targets=(SlotRef(allocator="q", index=0),))
        assert Recorder().visit(node) == "gate:H"
