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

"""Tests for parallelism analyzer."""

from pecos.slr import CReg, If, Main, QReg, Repeat
from pecos.slr.ast import slr_to_ast
from pecos.slr.ast.analysis import ParallelismAnalyzer, analyze_parallelism
from pecos.slr.qeclib import qubit as qb


class TestParallelismBasic:
    """Basic parallelism tests."""

    def test_empty_circuit(self) -> None:
        """Empty circuit has no parallelism."""
        prog = Main(
            _q := QReg("q", 1),
        )

        ast = slr_to_ast(prog)
        result = analyze_parallelism(ast)

        assert result.total_operations == 0
        assert result.depth == 0

    def test_single_gate(self) -> None:
        """Single gate has parallelism ratio of 1."""
        prog = Main(
            q := QReg("q", 1),
            qb.H(q[0]),
        )

        ast = slr_to_ast(prog)
        result = analyze_parallelism(ast)

        assert result.total_operations == 1
        assert result.depth == 1
        assert result.parallelism_ratio == 1.0

    def test_sequential_gates(self) -> None:
        """Sequential gates on same qubit have ratio 1."""
        prog = Main(
            q := QReg("q", 1),
            qb.H(q[0]),
            qb.X(q[0]),
            qb.Y(q[0]),
        )

        ast = slr_to_ast(prog)
        result = analyze_parallelism(ast)

        assert result.total_operations == 3
        assert result.depth == 3
        assert result.parallelism_ratio == 1.0


class TestParallelismParallel:
    """Tests for parallel operations."""

    def test_parallel_gates(self) -> None:
        """Gates on different qubits can be parallel."""
        prog = Main(
            q := QReg("q", 2),
            qb.H(q[0]),
            qb.X(q[1]),
        )

        ast = slr_to_ast(prog)
        result = analyze_parallelism(ast)

        assert result.total_operations == 2
        assert result.depth == 1
        assert result.parallelism_ratio == 2.0
        assert result.max_parallel_gates == 2

    def test_three_parallel_gates(self) -> None:
        """Three gates on different qubits."""
        prog = Main(
            q := QReg("q", 3),
            qb.H(q[0]),
            qb.X(q[1]),
            qb.Y(q[2]),
        )

        ast = slr_to_ast(prog)
        result = analyze_parallelism(ast)

        assert result.total_operations == 3
        assert result.depth == 1
        assert result.parallelism_ratio == 3.0
        assert result.max_parallel_gates == 3

    def test_mixed_parallelism(self) -> None:
        """Mix of parallel and sequential."""
        prog = Main(
            q := QReg("q", 2),
            qb.H(q[0]),  # depth 0
            qb.X(q[1]),  # depth 0 (parallel)
            qb.Y(q[0]),  # depth 1
        )

        ast = slr_to_ast(prog)
        result = analyze_parallelism(ast)

        assert result.total_operations == 3
        assert result.depth == 2
        assert result.parallelism_ratio == 1.5


class TestParallelismTwoQubit:
    """Two-qubit gate parallelism tests."""

    def test_cx_blocks_both_qubits(self) -> None:
        """CX gate blocks both qubits."""
        prog = Main(
            q := QReg("q", 2),
            qb.CX(q[0], q[1]),
            qb.H(q[0]),
            qb.X(q[1]),
        )

        ast = slr_to_ast(prog)
        result = analyze_parallelism(ast)

        # CX at depth 0, H and X at depth 1 (parallel)
        assert result.total_operations == 3
        assert result.depth == 2

    def test_parallel_cx_gates(self) -> None:
        """CX gates on disjoint qubits can be parallel."""
        prog = Main(
            q := QReg("q", 4),
            qb.CX(q[0], q[1]),
            qb.CX(q[2], q[3]),
        )

        ast = slr_to_ast(prog)
        result = analyze_parallelism(ast)

        assert result.total_operations == 2
        assert result.depth == 1
        assert result.parallelism_ratio == 2.0


class TestParallelismControlFlow:
    """Parallelism with control flow."""

    def test_repeat_loop(self) -> None:
        """Repeat loop unrolls for parallelism."""
        prog = Main(
            q := QReg("q", 1),
            Repeat(cond=3).block(
                qb.H(q[0]),
            ),
        )

        ast = slr_to_ast(prog)
        result = analyze_parallelism(ast)

        # 3 H gates, all sequential on same qubit
        assert result.total_operations == 3
        assert result.depth == 3

    def test_if_branches(self) -> None:
        """If statement considers both branches."""
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            If(c[0] == 1).Then(
                qb.H(q[0]),
            ),
        )

        ast = slr_to_ast(prog)
        result = analyze_parallelism(ast)

        assert result.total_operations == 1
        assert result.depth == 1


class TestParallelismMetrics:
    """Tests for parallelism metrics."""

    def test_avg_parallel_gates(self) -> None:
        """Average parallel gates calculation."""
        prog = Main(
            q := QReg("q", 2),
            qb.H(q[0]),
            qb.X(q[1]),  # 2 in layer 0
            qb.Y(q[0]),  # 1 in layer 1
        )

        ast = slr_to_ast(prog)
        result = analyze_parallelism(ast)

        assert result.avg_parallel_gates == 1.5

    def test_layer_sizes(self) -> None:
        """Layer size tracking."""
        prog = Main(
            q := QReg("q", 3),
            qb.H(q[0]),
            qb.X(q[1]),
            qb.Y(q[2]),  # 3 in layer 0
            qb.Z(q[0]),  # 1 in layer 1
        )

        ast = slr_to_ast(prog)
        result = analyze_parallelism(ast)

        assert result.layer_sizes == [3, 1]
        assert result.max_parallel_gates == 3


class TestAnalyzerClass:
    """Tests for ParallelismAnalyzer class."""

    def test_analyzer_reuse(self) -> None:
        """Analyzer can be reused."""
        analyzer = ParallelismAnalyzer()

        prog1 = Main(q := QReg("q", 1), qb.H(q[0]))
        prog2 = Main(q := QReg("q", 2), qb.H(q[0]), qb.X(q[1]))

        ast1 = slr_to_ast(prog1)
        ast2 = slr_to_ast(prog2)

        result1 = analyzer.analyze(ast1)
        result2 = analyzer.analyze(ast2)

        assert result1.parallelism_ratio == 1.0
        assert result2.parallelism_ratio == 2.0

    def test_result_string(self) -> None:
        """ParallelismResult string representation."""
        prog = Main(q := QReg("q", 2), qb.H(q[0]), qb.X(q[1]))

        ast = slr_to_ast(prog)
        result = analyze_parallelism(ast)

        assert "ratio=" in str(result)
        assert "depth=" in str(result)
