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

"""Tests for connectivity analyzer."""

from pecos.slr import CReg, If, Main, QReg, Repeat
from pecos.slr.ast import slr_to_ast
from pecos.slr.ast.analysis import ConnectivityAnalyzer, analyze_connectivity
from pecos.slr.ast.nodes import GateKind
from pecos.slr.qeclib import qubit as qb


class TestConnectivityBasic:
    """Basic connectivity tests."""

    def test_no_two_qubit_gates(self):
        """Circuit with no two-qubit gates."""
        prog = Main(
            q := QReg("q", 2),
            qb.H(q[0]),
            qb.X(q[1]),
        )

        ast = slr_to_ast(prog)
        result = analyze_connectivity(ast)

        assert len(result.edges) == 0
        assert result.max_degree == 0
        assert result.is_linear is True

    def test_single_cx(self):
        """Single CX gate creates one edge."""
        prog = Main(
            q := QReg("q", 2),
            qb.CX(q[0], q[1]),
        )

        ast = slr_to_ast(prog)
        result = analyze_connectivity(ast)

        assert len(result.edges) == 1
        edge = list(result.edges)[0]
        assert (("q", 0), ("q", 1)) == edge or (("q", 1), ("q", 0)) == edge
        assert result.max_degree == 1
        assert result.is_linear is True

    def test_single_cz(self):
        """Single CZ gate creates one edge."""
        prog = Main(
            q := QReg("q", 2),
            qb.CZ(q[0], q[1]),
        )

        ast = slr_to_ast(prog)
        result = analyze_connectivity(ast)

        assert len(result.edges) == 1
        assert GateKind.CZ in result.gate_types


class TestConnectivityLinear:
    """Linear connectivity tests."""

    def test_bell_state(self):
        """Bell state has linear connectivity."""
        prog = Main(
            q := QReg("q", 2),
            qb.H(q[0]),
            qb.CX(q[0], q[1]),
        )

        ast = slr_to_ast(prog)
        result = analyze_connectivity(ast)

        assert len(result.edges) == 1
        assert result.is_linear is True

    def test_ghz_state(self):
        """GHZ state has linear connectivity."""
        prog = Main(
            q := QReg("q", 3),
            qb.H(q[0]),
            qb.CX(q[0], q[1]),
            qb.CX(q[1], q[2]),
        )

        ast = slr_to_ast(prog)
        result = analyze_connectivity(ast)

        assert len(result.edges) == 2
        assert result.is_linear is True
        assert result.max_degree == 2

    def test_linear_chain(self):
        """Linear chain of CX gates."""
        prog = Main(
            q := QReg("q", 4),
            qb.CX(q[0], q[1]),
            qb.CX(q[1], q[2]),
            qb.CX(q[2], q[3]),
        )

        ast = slr_to_ast(prog)
        result = analyze_connectivity(ast)

        assert len(result.edges) == 3
        assert result.is_linear is True


class TestConnectivityNonLinear:
    """Non-linear connectivity tests."""

    def test_triangle(self):
        """Triangle connectivity is not linear."""
        prog = Main(
            q := QReg("q", 3),
            qb.CX(q[0], q[1]),
            qb.CX(q[1], q[2]),
            qb.CX(q[0], q[2]),
        )

        ast = slr_to_ast(prog)
        result = analyze_connectivity(ast)

        assert len(result.edges) == 3
        assert result.is_linear is False  # Has cycle

    def test_star_topology(self):
        """Star topology (center connected to all others)."""
        prog = Main(
            q := QReg("q", 4),
            qb.CX(q[0], q[1]),
            qb.CX(q[0], q[2]),
            qb.CX(q[0], q[3]),
        )

        ast = slr_to_ast(prog)
        result = analyze_connectivity(ast)

        assert len(result.edges) == 3
        assert result.is_linear is False  # Degree 3 at center
        assert result.max_degree == 3


class TestConnectivityControlFlow:
    """Connectivity with control flow."""

    def test_cx_inside_if(self):
        """CX inside if statement."""
        prog = Main(
            q := QReg("q", 2),
            c := CReg("c", 1),
            If(c[0] == 1).Then(
                qb.CX(q[0], q[1]),
            ),
        )

        ast = slr_to_ast(prog)
        result = analyze_connectivity(ast)

        assert len(result.edges) == 1

    def test_cx_inside_repeat(self):
        """CX inside repeat loop."""
        prog = Main(
            q := QReg("q", 2),
            Repeat(cond=3).block(
                qb.CX(q[0], q[1]),
            ),
        )

        ast = slr_to_ast(prog)
        result = analyze_connectivity(ast)

        assert len(result.edges) == 1
        # Connectivity analyzer only looks at unique edges, not counts
        edge = list(result.edges)[0]
        assert result.edge_weights[edge] >= 1


class TestConnectivityCouplingMap:
    """Coupling map tests."""

    def test_coupling_map_structure(self):
        """Coupling map has correct structure."""
        prog = Main(
            q := QReg("q", 3),
            qb.CX(q[0], q[1]),
            qb.CX(q[1], q[2]),
        )

        ast = slr_to_ast(prog)
        result = analyze_connectivity(ast)

        # Check adjacency
        q0 = ("q", 0)
        q1 = ("q", 1)
        q2 = ("q", 2)

        assert q1 in result.coupling_map[q0]
        assert q0 in result.coupling_map[q1]
        assert q2 in result.coupling_map[q1]
        assert q1 in result.coupling_map[q2]


class TestAnalyzerClass:
    """Tests for ConnectivityAnalyzer class."""

    def test_analyzer_reuse(self):
        """Analyzer can be reused."""
        analyzer = ConnectivityAnalyzer()

        prog1 = Main(q := QReg("q", 2), qb.CX(q[0], q[1]))
        prog2 = Main(q := QReg("q", 3), qb.CX(q[0], q[1]), qb.CX(q[1], q[2]))

        ast1 = slr_to_ast(prog1)
        ast2 = slr_to_ast(prog2)

        result1 = analyzer.analyze(ast1)
        result2 = analyzer.analyze(ast2)

        assert len(result1.edges) == 1
        assert len(result2.edges) == 2

    def test_result_string(self):
        """ConnectivityResult string representation."""
        prog = Main(q := QReg("q", 2), qb.CX(q[0], q[1]))

        ast = slr_to_ast(prog)
        result = analyze_connectivity(ast)

        assert "1 edges" in str(result)
        assert "linear: True" in str(result)
