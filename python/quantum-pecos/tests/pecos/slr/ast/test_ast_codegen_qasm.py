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

"""Tests for AST to QASM code generator."""

import pytest
from pecos.slr import Barrier, CReg, If, Main, QReg, Repeat
from pecos.slr.ast import AstToQasm, ast_to_qasm, slr_to_ast
from pecos.slr.qeclib import qubit as qb


class TestAstToQasmBasic:
    """Basic code generation tests."""

    def test_empty_program(self) -> None:
        """Empty program generates QASM header."""
        prog = Main()
        ast = slr_to_ast(prog)

        code = ast_to_qasm(ast)

        assert "OPENQASM 2.0;" in code
        assert 'include "hqslib1.inc";' in code

    def test_no_header(self) -> None:
        """Program without header excludes version and includes."""
        prog = Main()
        ast = slr_to_ast(prog)

        code = ast_to_qasm(ast, include_header=False)

        assert "OPENQASM" not in code
        assert "include" not in code

    def test_program_with_qreg(self) -> None:
        """Program with QReg generates qreg declaration."""
        prog = Main(
            _q := QReg("q", 2),
        )
        ast = slr_to_ast(prog)

        code = ast_to_qasm(ast)

        assert "qreg q[2];" in code

    def test_program_with_creg(self) -> None:
        """Program with CReg generates both qreg and creg declarations."""
        prog = Main(
            _q := QReg("q", 1),
            _c := CReg("c", 1),
        )
        ast = slr_to_ast(prog)

        code = ast_to_qasm(ast)

        assert "qreg q[1];" in code
        assert "creg c[1];" in code


class TestAstToQasmGates:
    """Gate code generation tests."""

    def test_single_qubit_gate(self) -> None:
        """Single-qubit gate generates correct syntax."""
        prog = Main(
            q := QReg("q", 1),
            qb.H(q[0]),
        )
        ast = slr_to_ast(prog)

        code = ast_to_qasm(ast)

        assert "h q[0];" in code

    def test_two_qubit_gate(self) -> None:
        """Two-qubit gate generates correct syntax with both targets."""
        prog = Main(
            q := QReg("q", 2),
            qb.CX(q[0], q[1]),
        )
        ast = slr_to_ast(prog)

        code = ast_to_qasm(ast)

        assert "cx q[0], q[1];" in code

    def test_multiple_gates(self) -> None:
        """Multiple gates generate in sequence."""
        prog = Main(
            q := QReg("q", 2),
            qb.H(q[0]),
            qb.X(q[1]),
            qb.CZ(q[0], q[1]),
        )
        ast = slr_to_ast(prog)

        code = ast_to_qasm(ast)

        assert "h q[0];" in code
        assert "x q[1];" in code
        assert "cz q[0], q[1];" in code

    def test_pauli_gates(self) -> None:
        """Pauli gates (X, Y, Z) generate correctly."""
        prog = Main(
            q := QReg("q", 1),
            qb.X(q[0]),
            qb.Y(q[0]),
            qb.Z(q[0]),
        )
        ast = slr_to_ast(prog)

        code = ast_to_qasm(ast)

        assert "x q[0];" in code
        assert "y q[0];" in code
        assert "z q[0];" in code

    def test_phase_gates(self) -> None:
        """Phase gates (SZ, SZdg) generate as rz rotations."""
        prog = Main(
            q := QReg("q", 1),
            qb.SZ(q[0]),
            qb.SZdg(q[0]),
        )
        ast = slr_to_ast(prog)

        code = ast_to_qasm(ast)

        # SZ is rz(pi/2), SZdg is rz(-pi/2)
        assert "rz(pi/2) q[0];" in code
        assert "rz(-pi/2) q[0];" in code


class TestAstToQasmPrepMeasure:
    """Prep and measure code generation tests."""

    def test_measure_with_result(self) -> None:
        """Measure with result generates arrow syntax."""
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            qb.Measure(q[0]) > c[0],
        )
        ast = slr_to_ast(prog)

        code = ast_to_qasm(ast)

        assert "measure q[0] -> c[0];" in code

    def test_prep_reset(self) -> None:
        """Prep generates reset operation."""
        prog = Main(
            q := QReg("q", 1),
            qb.Prep(q[0]),
        )
        ast = slr_to_ast(prog)

        code = ast_to_qasm(ast)

        assert "reset q[0];" in code


class TestAstToQasmControlFlow:
    """Control flow code generation tests."""

    def test_if_statement(self) -> None:
        """If statement generates conditional gate syntax."""
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            If(c[0] == 1).Then(
                qb.H(q[0]),
            ),
        )
        ast = slr_to_ast(prog)

        code = ast_to_qasm(ast)

        # Should generate conditional gate
        assert "if(c[0] == 1) h q[0];" in code

    def test_repeat_unrolled(self) -> None:
        """Repeat generates unrolled loop with comment."""
        prog = Main(
            q := QReg("q", 1),
            Repeat(cond=3).block(
                qb.H(q[0]),
            ),
        )
        ast = slr_to_ast(prog)

        code = ast_to_qasm(ast)

        # Repeat should be unrolled with a comment
        assert "// Repeat 3 times (unrolled)" in code
        # Should have 3 H gates
        assert code.count("h q[0];") == 3


class TestAstToQasmQEC:
    """QEC pattern code generation tests."""

    def test_syndrome_extraction(self) -> None:
        """Syndrome extraction pattern generates correct operations."""
        prog = Main(
            data := QReg("data", 2),
            ancilla := QReg("ancilla", 1),
            c := CReg("c", 1),
            qb.CX(data[0], ancilla[0]),
            qb.CX(data[1], ancilla[0]),
            qb.Measure(ancilla[0]) > c[0],
        )
        ast = slr_to_ast(prog)

        code = ast_to_qasm(ast)

        # Check register declarations
        assert "qreg data[2];" in code
        assert "qreg ancilla[1];" in code
        assert "creg c[1];" in code

        # Check operations
        assert "cx data[0], ancilla[0];" in code
        assert "cx data[1], ancilla[0];" in code
        assert "measure ancilla[0] -> c[0];" in code


class TestAstToQasmGenerator:
    """Tests for AstToQasm generator class."""

    def test_generator_reusable(self) -> None:
        """Generator can be reused for multiple programs."""
        generator = AstToQasm(include_header=False)

        prog1 = Main(
            q := QReg("q", 1),
            qb.H(q[0]),
        )

        prog2 = Main(
            r := QReg("r", 2),
            qb.X(r[0]),
        )

        ast1 = slr_to_ast(prog1)
        ast2 = slr_to_ast(prog2)

        code1 = "\n".join(generator.generate(ast1))
        code2 = "\n".join(generator.generate(ast2))

        assert "q[0]" in code1
        assert "r[0]" in code2

    def test_custom_includes(self) -> None:
        """Generator can use custom include files."""
        generator = AstToQasm(includes=["custom.inc", "other.inc"])

        prog = Main()
        ast = slr_to_ast(prog)

        code = "\n".join(generator.generate(ast))

        assert 'include "custom.inc";' in code
        assert 'include "other.inc";' in code


class TestAstToQasmFullPipeline:
    """End-to-end tests: SLR -> AST -> QASM."""

    def test_full_pipeline(self) -> None:
        """Full SLR to QASM pipeline generates valid code."""
        # Create SLR program
        prog = Main(
            q := QReg("q", 3),
            c := CReg("c", 1),
            # Bell state
            qb.H(q[0]),
            qb.CX(q[0], q[1]),
            # Conditional
            If(c[0] == 1).Then(
                qb.X(q[2]),
            ),
        )

        # Convert to AST
        ast = slr_to_ast(prog)

        # Generate QASM code
        code = ast_to_qasm(ast)

        # Verify structure
        assert "OPENQASM 2.0;" in code
        assert "qreg q[3];" in code
        assert "creg c[1];" in code
        assert "h q[0];" in code
        assert "cx q[0], q[1];" in code
        assert "if(c[0] == 1) x q[2];" in code

    def test_bell_state_circuit(self) -> None:
        """Test a simple Bell state circuit."""
        prog = Main(
            q := QReg("q", 2),
            qb.H(q[0]),
            qb.CX(q[0], q[1]),
        )

        ast = slr_to_ast(prog)
        code = ast_to_qasm(ast)

        lines = code.split("\n")

        # Check header is first
        assert lines[0] == "OPENQASM 2.0;"

        # Check all expected lines exist
        assert any("qreg q[2];" in line for line in lines)
        assert any("h q[0];" in line for line in lines)
        assert any("cx q[0], q[1];" in line for line in lines)

    def test_t_gates(self) -> None:
        """Test T and Tdg gate generation."""
        prog = Main(
            q := QReg("q", 1),
            qb.T(q[0]),
            qb.Tdg(q[0]),
        )

        ast = slr_to_ast(prog)
        code = ast_to_qasm(ast)

        # T is rz(pi/4), Tdg is rz(-pi/4)
        assert "rz(pi/4) q[0];" in code
        assert "rz(-pi/4) q[0];" in code

    def test_barrier_operation(self) -> None:
        """Test barrier generation."""
        prog = Main(
            q := QReg("q", 2),
            qb.H(q[0]),
            Barrier(q),
            qb.CX(q[0], q[1]),
        )

        ast = slr_to_ast(prog)
        code = ast_to_qasm(ast)

        assert "barrier q;" in code
