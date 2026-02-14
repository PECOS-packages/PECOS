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

"""Qubit connectivity analysis for quantum circuits.

This module analyzes which qubits interact via two-qubit gates, building
a connectivity graph that can be used for hardware mapping decisions.

Example:
    from pecos.slr.ast import slr_to_ast
    from pecos.slr.ast.analysis import analyze_connectivity

    ast = slr_to_ast(program)
    result = analyze_connectivity(ast)

    print(f"Qubit edges: {result.edges}")
    print(f"Is linearly mappable: {result.is_linear}")
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from pecos.slr.ast.nodes import (
    ForStmt,
    GateKind,
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

# Two-qubit gates that create connectivity requirements
TWO_QUBIT_GATES = frozenset(
    {
        GateKind.CX,
        GateKind.CY,
        GateKind.CZ,
        GateKind.CH,
        GateKind.SXX,
        GateKind.SYY,
        GateKind.SZZ,
        GateKind.SXXdg,
        GateKind.SYYdg,
        GateKind.SZZdg,
        GateKind.RZZ,
    }
)


@dataclass
class ConnectivityResult:
    """Result of connectivity analysis."""

    # Set of qubit pair edges (sorted tuples for consistency)
    edges: set[tuple[tuple[str, int], tuple[str, int]]] = field(default_factory=set)

    # Adjacency list for the connectivity graph
    coupling_map: dict[tuple[str, int], set[tuple[str, int]]] = field(
        default_factory=lambda: defaultdict(set)
    )

    # Count of two-qubit gates per edge
    edge_weights: dict[tuple[tuple[str, int], tuple[str, int]], int] = field(default_factory=dict)

    # All two-qubit gate types used
    gate_types: set[GateKind] = field(default_factory=set)

    # Whether the connectivity graph is linearly mappable (path graph)
    is_linear: bool = True

    # Maximum degree of any qubit (number of neighbors)
    max_degree: int = 0

    def __str__(self) -> str:
        return f"Connectivity: {len(self.edges)} edges, max degree: {self.max_degree}, linear: {self.is_linear}"


class ConnectivityAnalyzer:
    """Analyzes qubit connectivity using recursive descent.

    Builds a graph of qubit interactions from two-qubit gates,
    useful for hardware topology mapping.
    """

    def __init__(self) -> None:
        self.edges: set[tuple[tuple[str, int], tuple[str, int]]] = set()
        self.coupling_map: dict[tuple[str, int], set[tuple[str, int]]] = defaultdict(set)
        self.edge_weights: dict[tuple[tuple[str, int], tuple[str, int]], int] = defaultdict(int)
        self.gate_types: set[GateKind] = set()

    def analyze(self, program: Program) -> ConnectivityResult:
        """Analyze qubit connectivity of a program.

        Args:
            program: The AST Program to analyze.

        Returns:
            ConnectivityResult with connectivity information.
        """
        self.edges = set()
        self.coupling_map = defaultdict(set)
        self.edge_weights = defaultdict(int)
        self.gate_types = set()

        for stmt in program.body:
            self._analyze_statement(stmt)

        # Calculate if graph is linearly mappable
        is_linear = self._check_linear()

        # Calculate max degree
        max_degree = 0
        for neighbors in self.coupling_map.values():
            max_degree = max(max_degree, len(neighbors))

        return ConnectivityResult(
            edges=set(self.edges),
            coupling_map=dict(self.coupling_map),
            edge_weights=dict(self.edge_weights),
            gate_types=set(self.gate_types),
            is_linear=is_linear,
            max_degree=max_degree,
        )

    def _make_edge(
        self, q1: tuple[str, int], q2: tuple[str, int]
    ) -> tuple[tuple[str, int], tuple[str, int]]:
        """Create a canonical edge (sorted for consistency)."""
        return (q1, q2) if (q1[0], q1[1]) <= (q2[0], q2[1]) else (q2, q1)

    def _check_linear(self) -> bool:
        """Check if the connectivity graph is linearly mappable (a path).

        A graph is a path if:
        - It has n-1 edges for n vertices (connected, acyclic)
        - Each vertex has degree <= 2
        - Exactly 0 or 2 vertices have degree 1 (endpoints)
        """
        if not self.coupling_map:
            return True

        n_vertices = len(self.coupling_map)
        n_edges = len(self.edges)

        # Must be connected and acyclic (tree with n-1 edges)
        if n_edges != n_vertices - 1:
            return False

        # Check degrees
        degree_1_count = 0
        for neighbors in self.coupling_map.values():
            degree = len(neighbors)
            if degree > 2:
                return False
            if degree == 1:
                degree_1_count += 1

        # Path has exactly 2 endpoints (degree 1) or is a single vertex
        return degree_1_count in (0, 2) or n_vertices == 1

    def _analyze_statement(self, stmt: Statement) -> None:
        """Analyze a statement using recursive descent."""
        if isinstance(stmt, GateOp):
            self._analyze_gate(stmt)
        elif isinstance(stmt, MeasureOp | PrepareOp):
            pass  # No connectivity contribution
        elif isinstance(stmt, IfStmt):
            self._analyze_if(stmt)
        elif isinstance(stmt, WhileStmt):
            self._analyze_while(stmt)
        elif isinstance(stmt, ForStmt):
            self._analyze_for(stmt)
        elif isinstance(stmt, RepeatStmt):
            self._analyze_repeat(stmt)
        elif isinstance(stmt, ParallelBlock):
            self._analyze_parallel(stmt)

    def _analyze_gate(self, node: GateOp) -> None:
        """Analyze a gate's connectivity contribution."""
        if len(node.targets) >= 2:
            self.gate_types.add(node.gate)

            # Add edges for all pairs of targets
            for i, t1 in enumerate(node.targets):
                q1 = (t1.allocator, t1.index)
                for t2 in node.targets[i + 1 :]:
                    q2 = (t2.allocator, t2.index)
                    edge = self._make_edge(q1, q2)

                    self.edges.add(edge)
                    self.edge_weights[edge] += 1
                    self.coupling_map[q1].add(q2)
                    self.coupling_map[q2].add(q1)

    def _analyze_if(self, node: IfStmt) -> None:
        """Analyze an if statement (both branches)."""
        for stmt in node.then_body:
            self._analyze_statement(stmt)
        if node.else_body:
            for stmt in node.else_body:
                self._analyze_statement(stmt)

    def _analyze_while(self, node: WhileStmt) -> None:
        """Analyze a while loop."""
        for stmt in node.body:
            self._analyze_statement(stmt)

    def _analyze_for(self, node: ForStmt) -> None:
        """Analyze a for loop."""
        for stmt in node.body:
            self._analyze_statement(stmt)

    def _analyze_repeat(self, node: RepeatStmt) -> None:
        """Analyze a repeat loop."""
        for stmt in node.body:
            self._analyze_statement(stmt)

    def _analyze_parallel(self, node: ParallelBlock) -> None:
        """Analyze a parallel block."""
        for stmt in node.body:
            self._analyze_statement(stmt)


def analyze_connectivity(program: Program) -> ConnectivityResult:
    """Convenience function to analyze qubit connectivity.

    Args:
        program: The AST Program to analyze.

    Returns:
        ConnectivityResult with connectivity information.
    """
    analyzer = ConnectivityAnalyzer()
    return analyzer.analyze(program)
