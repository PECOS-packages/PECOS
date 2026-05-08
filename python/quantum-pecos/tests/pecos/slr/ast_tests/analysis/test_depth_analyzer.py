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

"""Tests for AST depth analyzer."""

import pytest
from pecos.slr import CReg, Main, QReg, Repeat
from pecos.slr.ast import slr_to_ast
from pecos.slr.ast.analysis import DepthAnalyzer, analyze_depth
from pecos.slr.qeclib import qubit as qb


class TestDepthAnalyzerBasic:
    """Basic depth analysis tests."""

    def test_empty_program(self) -> None:
        """Empty program has depth 0."""
        prog = Main()
        ast = slr_to_ast(prog)

        result = analyze_depth(ast)

        assert result.depth == 0

    def test_single_gate(self) -> None:
        """Single gate has depth 1."""
        prog = Main(
            q := QReg("q", 1),
            qb.Prep(q[0]),
            qb.H(q[0]),
        )
        ast = slr_to_ast(prog)

        result = analyze_depth(ast)

        assert result.depth == 2  # Prep + H

    def test_sequential_gates_same_qubit(self) -> None:
        """Sequential gates on same qubit add to depth."""
        prog = Main(
            q := QReg("q", 1),
            qb.Prep(q[0]),
            qb.H(q[0]),
            qb.X(q[0]),
            qb.Z(q[0]),
        )
        ast = slr_to_ast(prog)

        result = analyze_depth(ast)

        assert result.depth == 4  # Prep + H + X + Z

    def test_parallel_gates_different_qubits(self) -> None:
        """Gates on different qubits can run in parallel."""
        prog = Main(
            q := QReg("q", 2),
            qb.Prep(q[0]),
            qb.Prep(q[1]),
            qb.H(q[0]),
            qb.X(q[1]),
        )
        ast = slr_to_ast(prog)

        result = analyze_depth(ast)

        # q[0]: Prep(1) -> H(2)
        # q[1]: Prep(1) -> X(2)
        # Both paths have depth 2
        assert result.depth == 2


class TestDepthAnalyzerTwoQubit:
    """Two-qubit gate depth tests."""

    def test_two_qubit_gate_depth(self) -> None:
        """Two-qubit gate increases depth for both qubits."""
        prog = Main(
            q := QReg("q", 2),
            qb.Prep(q[0]),
            qb.Prep(q[1]),
            qb.CX(q[0], q[1]),
        )
        ast = slr_to_ast(prog)

        result = analyze_depth(ast)

        # Preps at depth 1, CX at depth 2
        assert result.depth == 2
        assert result.two_qubit_depth == 2

    def test_two_qubit_gate_waits_for_both(self) -> None:
        """Two-qubit gate waits for both qubits to be ready."""
        prog = Main(
            q := QReg("q", 2),
            qb.Prep(q[0]),
            qb.Prep(q[1]),
            qb.H(q[0]),  # q[0] now at depth 2
            qb.CX(q[0], q[1]),  # Must wait for q[0]
        )
        ast = slr_to_ast(prog)

        result = analyze_depth(ast)

        # q[0]: Prep(1) -> H(2) -> CX(3)
        # q[1]: Prep(1) -> (wait) -> CX(3)
        assert result.depth == 3

    def test_chain_of_two_qubit_gates(self) -> None:
        """Chain of two-qubit gates increases depth."""
        prog = Main(
            q := QReg("q", 3),
            qb.Prep(q[0]),
            qb.Prep(q[1]),
            qb.Prep(q[2]),
            qb.CX(q[0], q[1]),  # Depth 2
            qb.CX(q[1], q[2]),  # Depth 3 (waits for q[1])
        )
        ast = slr_to_ast(prog)

        result = analyze_depth(ast)

        assert result.depth == 3
        assert result.two_qubit_depth == 3


class TestDepthAnalyzerBellState:
    """Bell state circuit depth tests."""

    def test_bell_state_depth(self) -> None:
        """Bell state has depth 3 (prep + H + CX)."""
        prog = Main(
            q := QReg("q", 2),
            qb.Prep(q[0]),
            qb.Prep(q[1]),
            qb.H(q[0]),
            qb.CX(q[0], q[1]),
        )
        ast = slr_to_ast(prog)

        result = analyze_depth(ast)

        # q[0]: Prep(1) -> H(2) -> CX(3)
        # q[1]: Prep(1) -> (wait) -> CX(3)
        assert result.depth == 3


class TestDepthAnalyzerControlFlow:
    """Control flow depth tests."""

    def test_repeat_adds_depth(self) -> None:
        """Repeat loop adds depth for each iteration."""
        prog = Main(
            q := QReg("q", 1),
            qb.Prep(q[0]),
            Repeat(cond=3).block(
                qb.H(q[0]),
            ),
        )
        ast = slr_to_ast(prog)

        result = analyze_depth(ast)

        # Prep(1) + H(2) + H(3) + H(4)
        assert result.depth == 4


class TestDepthAnalyzerQEC:
    """QEC pattern depth tests."""

    def test_syndrome_extraction_depth(self) -> None:
        """Syndrome extraction depth is computed correctly."""
        prog = Main(
            data := QReg("data", 2),
            ancilla := QReg("ancilla", 1),
            c := CReg("c", 1),
            qb.Prep(data[0]),
            qb.Prep(data[1]),
            qb.Prep(ancilla[0]),
            qb.CX(data[0], ancilla[0]),
            qb.CX(data[1], ancilla[0]),
            qb.Measure(ancilla[0]) > c[0],
        )
        ast = slr_to_ast(prog)

        result = analyze_depth(ast)

        # All preps at depth 1
        # CX(data[0], ancilla[0]) at depth 2
        # CX(data[1], ancilla[0]) at depth 3 (waits for ancilla)
        # Measure at depth 4
        assert result.depth == 4
        assert result.two_qubit_depth == 3


class TestDepthAnalyzerClass:
    """Tests for the DepthAnalyzer class."""

    def test_analyzer_reusable(self) -> None:
        """Analyzer can be reused for multiple programs."""
        analyzer = DepthAnalyzer()

        prog1 = Main(
            q := QReg("q", 1),
            qb.Prep(q[0]),
            qb.H(q[0]),
        )
        prog2 = Main(
            q := QReg("q", 1),
            qb.Prep(q[0]),
            qb.H(q[0]),
            qb.X(q[0]),
        )

        ast1 = slr_to_ast(prog1)
        ast2 = slr_to_ast(prog2)

        result1 = analyzer.analyze(ast1)
        result2 = analyzer.analyze(ast2)

        assert result1.depth == 2
        assert result2.depth == 3

    def test_result_string_representation(self) -> None:
        """DepthResult has useful string representation."""
        prog = Main(
            q := QReg("q", 2),
            qb.Prep(q[0]),
            qb.Prep(q[1]),
            qb.H(q[0]),
            qb.CX(q[0], q[1]),
        )
        ast = slr_to_ast(prog)

        result = analyze_depth(ast)
        result_str = str(result)

        assert "Depth: 3" in result_str
        assert "2Q depth: 3" in result_str
