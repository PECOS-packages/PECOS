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

"""Tests for T-count analyzer."""

from pecos.slr import CReg, If, Main, QReg, Repeat
from pecos.slr.ast import slr_to_ast
from pecos.slr.ast.analysis import TCountAnalyzer, analyze_t_count
from pecos.slr.qeclib import qubit as qb


class TestTCountBasic:
    """Basic T-count tests."""

    def test_no_t_gates(self):
        """Circuit with no T gates."""
        prog = Main(
            q := QReg("q", 1),
            qb.H(q[0]),
            qb.X(q[0]),
        )

        ast = slr_to_ast(prog)
        result = analyze_t_count(ast)

        assert result.t_count == 0
        assert result.t_depth == 0

    def test_single_t_gate(self):
        """Circuit with single T gate."""
        prog = Main(
            q := QReg("q", 1),
            qb.T(q[0]),
        )

        ast = slr_to_ast(prog)
        result = analyze_t_count(ast)

        assert result.t_count == 1
        assert result.t_depth == 1
        assert result.breakdown == {"T": 1}

    def test_single_tdg_gate(self):
        """Circuit with single Tdg gate."""
        prog = Main(
            q := QReg("q", 1),
            qb.Tdg(q[0]),
        )

        ast = slr_to_ast(prog)
        result = analyze_t_count(ast)

        assert result.t_count == 1
        assert result.t_depth == 1
        assert result.breakdown == {"Tdg": 1}

    def test_mixed_t_gates(self):
        """Circuit with T and Tdg gates."""
        prog = Main(
            q := QReg("q", 1),
            qb.T(q[0]),
            qb.Tdg(q[0]),
        )

        ast = slr_to_ast(prog)
        result = analyze_t_count(ast)

        assert result.t_count == 2
        assert result.t_depth == 2
        assert result.breakdown == {"T": 1, "Tdg": 1}


class TestTDepth:
    """T-depth calculation tests."""

    def test_sequential_t_gates(self):
        """Sequential T gates on same qubit have additive T-depth."""
        prog = Main(
            q := QReg("q", 1),
            qb.T(q[0]),
            qb.T(q[0]),
            qb.T(q[0]),
        )

        ast = slr_to_ast(prog)
        result = analyze_t_count(ast)

        assert result.t_count == 3
        assert result.t_depth == 3

    def test_parallel_t_gates(self):
        """T gates on different qubits can be parallel."""
        prog = Main(
            q := QReg("q", 2),
            qb.T(q[0]),
            qb.T(q[1]),
        )

        ast = slr_to_ast(prog)
        result = analyze_t_count(ast)

        assert result.t_count == 2
        assert result.t_depth == 1  # Parallel on different qubits

    def test_mixed_parallel_sequential(self):
        """Mix of parallel and sequential T gates."""
        prog = Main(
            q := QReg("q", 2),
            qb.T(q[0]),  # depth 1 on q[0]
            qb.T(q[1]),  # depth 1 on q[1]
            qb.T(q[0]),  # depth 2 on q[0]
        )

        ast = slr_to_ast(prog)
        result = analyze_t_count(ast)

        assert result.t_count == 3
        assert result.t_depth == 2


class TestTCountControlFlow:
    """T-count with control flow."""

    def test_t_inside_repeat(self):
        """T gates inside repeat loop count all iterations."""
        prog = Main(
            q := QReg("q", 1),
            Repeat(cond=3).block(
                qb.T(q[0]),
            ),
        )

        ast = slr_to_ast(prog)
        result = analyze_t_count(ast)

        assert result.t_count == 3
        assert result.t_depth == 3

    def test_t_inside_if(self):
        """T gates inside if statement."""
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            If(c[0] == 1).Then(
                qb.T(q[0]),
            ),
        )

        ast = slr_to_ast(prog)
        result = analyze_t_count(ast)

        assert result.t_count == 1
        assert result.t_depth == 1


class TestTCountWithClifford:
    """T-count with Clifford gates."""

    def test_t_with_clifford(self):
        """T gates with Clifford gates."""
        prog = Main(
            q := QReg("q", 1),
            qb.H(q[0]),
            qb.T(q[0]),
            qb.H(q[0]),
            qb.T(q[0]),
        )

        ast = slr_to_ast(prog)
        result = analyze_t_count(ast)

        assert result.t_count == 2
        assert result.t_depth == 2

    def test_toffoli_decomposition_pattern(self):
        """Pattern similar to Toffoli gate T-count."""
        prog = Main(
            q := QReg("q", 3),
            qb.T(q[0]),
            qb.T(q[1]),
            qb.T(q[2]),
            qb.Tdg(q[0]),
            qb.Tdg(q[1]),
            qb.Tdg(q[2]),
            qb.T(q[2]),
        )

        ast = slr_to_ast(prog)
        result = analyze_t_count(ast)

        assert result.t_count == 7
        assert result.breakdown["T"] == 4
        assert result.breakdown["Tdg"] == 3


class TestAnalyzerClass:
    """Tests for TCountAnalyzer class."""

    def test_analyzer_reuse(self):
        """Analyzer can be reused."""
        analyzer = TCountAnalyzer()

        prog1 = Main(q := QReg("q", 1), qb.T(q[0]))
        prog2 = Main(q := QReg("q", 1), qb.T(q[0]), qb.T(q[0]))

        ast1 = slr_to_ast(prog1)
        ast2 = slr_to_ast(prog2)

        result1 = analyzer.analyze(ast1)
        result2 = analyzer.analyze(ast2)

        assert result1.t_count == 1
        assert result2.t_count == 2

    def test_result_string(self):
        """TCountResult string representation."""
        prog = Main(q := QReg("q", 1), qb.T(q[0]))

        ast = slr_to_ast(prog)
        result = analyze_t_count(ast)

        assert "T-count: 1" in str(result)
        assert "T-depth: 1" in str(result)
