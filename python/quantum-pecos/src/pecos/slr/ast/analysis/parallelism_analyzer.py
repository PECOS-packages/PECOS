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

"""Parallelism analysis for quantum circuits.

This module analyzes potential parallelism in quantum circuits by
identifying operations that can execute concurrently based on
qubit dependencies.

Example:
    from pecos.slr.ast import slr_to_ast
    from pecos.slr.ast.analysis import analyze_parallelism

    ast = slr_to_ast(program)
    result = analyze_parallelism(ast)

    print(f"Parallelism ratio: {result.parallelism_ratio:.2f}")
    print(f"Max parallel gates: {result.max_parallel_gates}")
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pecos.slr.ast.nodes import (
    ForStmt,
    GateOp,
    IfStmt,
    MeasureOp,
    ParallelBlock,
    PrepareOp,
    Program,
    RepeatStmt,
    Statement,
    WhileStmt,
)


@dataclass
class ParallelismResult:
    """Result of parallelism analysis."""

    # Total number of operations
    total_operations: int = 0

    # Circuit depth (number of layers)
    depth: int = 0

    # Parallelism ratio: total_ops / depth (higher = more parallel)
    # A ratio of 1.0 means fully sequential
    parallelism_ratio: float = 0.0

    # Maximum number of gates in any single layer
    max_parallel_gates: int = 0

    # Average gates per layer
    avg_parallel_gates: float = 0.0

    # Schedule: list of layers, each layer contains operation indices
    schedule_layers: list[list[int]] = field(default_factory=list)

    # Operations per layer
    layer_sizes: list[int] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"Parallelism: ratio={self.parallelism_ratio:.2f}, "
            f"depth={self.depth}, max_parallel={self.max_parallel_gates}"
        )


class ParallelismAnalyzer:
    """Analyzes parallelism using ASAP (As Soon As Possible) scheduling.

    Operations are scheduled to the earliest layer where all their
    qubit dependencies are satisfied.
    """

    def __init__(self) -> None:
        # When each qubit becomes available (layer number)
        self.qubit_available: dict[tuple[str, int], int] = {}
        # Operations indexed by their position
        self.operations: list[tuple[int, Statement]] = []
        # Schedule: layer -> list of operation indices
        self.schedule: list[list[int]] = []

    def analyze(self, program: Program) -> ParallelismResult:
        """Analyze parallelism in a program.

        Args:
            program: The AST Program to analyze.

        Returns:
            ParallelismResult with parallelism information.
        """
        self.qubit_available = {}
        self.operations = []
        self.schedule = []

        # Collect and schedule operations
        op_index = 0
        for stmt in program.body:
            op_index = self._collect_operations(stmt, op_index)

        # Build schedule layers
        if self.operations:
            max_layer = max(layer for layer, _ in self.operations)
            self.schedule = [[] for _ in range(max_layer + 1)]
            for layer, stmt in self.operations:
                self.schedule[layer].append(id(stmt))

        # Calculate metrics
        total_ops = len(self.operations)
        depth = len(self.schedule) if self.schedule else 0
        layer_sizes = [len(layer) for layer in self.schedule]
        max_parallel = max(layer_sizes) if layer_sizes else 0
        avg_parallel = total_ops / depth if depth > 0 else 0.0
        parallelism_ratio = total_ops / depth if depth > 0 else 0.0

        return ParallelismResult(
            total_operations=total_ops,
            depth=depth,
            parallelism_ratio=parallelism_ratio,
            max_parallel_gates=max_parallel,
            avg_parallel_gates=avg_parallel,
            schedule_layers=self.schedule,
            layer_sizes=layer_sizes,
        )

    def _get_qubit_layer(self, allocator: str, index: int) -> int:
        """Get the layer when a qubit becomes available."""
        return self.qubit_available.get((allocator, index), 0)

    def _set_qubit_layer(self, allocator: str, index: int, layer: int) -> None:
        """Set when a qubit will be available again."""
        self.qubit_available[(allocator, index)] = layer

    def _collect_operations(self, stmt: Statement, op_index: int) -> int:
        """Collect and schedule an operation.

        Returns the next operation index.
        """
        if isinstance(stmt, GateOp):
            return self._schedule_gate(stmt, op_index)
        elif isinstance(stmt, MeasureOp):
            return self._schedule_measure(stmt, op_index)
        elif isinstance(stmt, PrepareOp):
            return self._schedule_prepare(stmt, op_index)
        elif isinstance(stmt, IfStmt):
            return self._schedule_if(stmt, op_index)
        elif isinstance(stmt, WhileStmt):
            return self._schedule_while(stmt, op_index)
        elif isinstance(stmt, ForStmt):
            return self._schedule_for(stmt, op_index)
        elif isinstance(stmt, RepeatStmt):
            return self._schedule_repeat(stmt, op_index)
        elif isinstance(stmt, ParallelBlock):
            return self._schedule_parallel(stmt, op_index)
        return op_index

    def _schedule_gate(self, node: GateOp, op_index: int) -> int:
        """Schedule a gate operation."""
        # Find earliest layer where all qubits are available
        earliest_layer = 0
        for target in node.targets:
            qubit_layer = self._get_qubit_layer(target.allocator, target.index)
            earliest_layer = max(earliest_layer, qubit_layer)

        # Schedule at this layer
        self.operations.append((earliest_layer, node))

        # Update qubit availability
        next_layer = earliest_layer + 1
        for target in node.targets:
            self._set_qubit_layer(target.allocator, target.index, next_layer)

        return op_index + 1

    def _schedule_measure(self, node: MeasureOp, op_index: int) -> int:
        """Schedule a measurement operation."""
        earliest_layer = 0
        for target in node.targets:
            qubit_layer = self._get_qubit_layer(target.allocator, target.index)
            earliest_layer = max(earliest_layer, qubit_layer)

        self.operations.append((earliest_layer, node))

        next_layer = earliest_layer + 1
        for target in node.targets:
            self._set_qubit_layer(target.allocator, target.index, next_layer)

        return op_index + 1

    def _schedule_prepare(self, node: PrepareOp, op_index: int) -> int:
        """Schedule a prepare operation."""
        if node.slots is not None:
            earliest_layer = 0
            for slot in node.slots:
                qubit_layer = self._get_qubit_layer(node.allocator, slot)
                earliest_layer = max(earliest_layer, qubit_layer)

            self.operations.append((earliest_layer, node))

            next_layer = earliest_layer + 1
            for slot in node.slots:
                self._set_qubit_layer(node.allocator, slot, next_layer)
        else:
            # prepare_all - just record it
            self.operations.append((0, node))

        return op_index + 1

    def _schedule_if(self, node: IfStmt, op_index: int) -> int:
        """Schedule if statement (sequential for now)."""
        # Save state before branches
        state_before = dict(self.qubit_available)

        # Schedule then branch
        for stmt in node.then_body:
            op_index = self._collect_operations(stmt, op_index)

        state_after_then = dict(self.qubit_available)

        # Reset and schedule else branch
        self.qubit_available = dict(state_before)

        if node.else_body:
            for stmt in node.else_body:
                op_index = self._collect_operations(stmt, op_index)

        # Merge: take maximum layer from each branch
        for key in set(state_after_then.keys()) | set(self.qubit_available.keys()):
            then_layer = state_after_then.get(key, 0)
            else_layer = self.qubit_available.get(key, 0)
            self.qubit_available[key] = max(then_layer, else_layer)

        return op_index

    def _schedule_while(self, node: WhileStmt, op_index: int) -> int:
        """Schedule while loop (single iteration)."""
        for stmt in node.body:
            op_index = self._collect_operations(stmt, op_index)
        return op_index

    def _schedule_for(self, node: ForStmt, op_index: int) -> int:
        """Schedule for loop (single iteration)."""
        for stmt in node.body:
            op_index = self._collect_operations(stmt, op_index)
        return op_index

    def _schedule_repeat(self, node: RepeatStmt, op_index: int) -> int:
        """Schedule repeat loop (all iterations)."""
        for _ in range(node.count):
            for stmt in node.body:
                op_index = self._collect_operations(stmt, op_index)
        return op_index

    def _schedule_parallel(self, node: ParallelBlock, op_index: int) -> int:
        """Schedule parallel block.

        In a parallel block, all independent operations can be
        scheduled at the same layer.
        """
        for stmt in node.body:
            op_index = self._collect_operations(stmt, op_index)
        return op_index


def analyze_parallelism(program: Program) -> ParallelismResult:
    """Convenience function to analyze parallelism.

    Args:
        program: The AST Program to analyze.

    Returns:
        ParallelismResult with parallelism information.
    """
    analyzer = ParallelismAnalyzer()
    return analyzer.analyze(program)
