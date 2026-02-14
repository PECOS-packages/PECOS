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

"""Tests for AST to Guppy code generator."""

import pytest

from pecos.slr import CReg, If, Main, QReg, Repeat
from pecos.slr.ast import AstToGuppy, ast_to_guppy, slr_to_ast
from pecos.slr.qeclib import qubit as qb


class TestAstToGuppyBasic:
    """Basic code generation tests."""

    def test_empty_program(self):
        prog = Main()
        ast = slr_to_ast(prog)

        code = ast_to_guppy(ast)

        assert "from guppylang import guppy" in code
        assert "from guppylang.std import quantum" in code
        assert "@guppy" in code
        assert "def main" in code

    def test_program_with_qreg(self):
        prog = Main(
            q := QReg("q", 2),
        )
        ast = slr_to_ast(prog)

        code = ast_to_guppy(ast)

        assert "q: array[qubit, 2]" in code

    def test_program_with_creg(self):
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
        )
        ast = slr_to_ast(prog)

        # CRegs are handled differently in Guppy
        code = ast_to_guppy(ast)

        # Should still generate valid code
        assert "@guppy" in code


class TestAstToGuppyGates:
    """Gate code generation tests."""

    def test_single_qubit_gate(self):
        prog = Main(
            q := QReg("q", 1),
            qb.H(q[0]),
        )
        ast = slr_to_ast(prog)

        code = ast_to_guppy(ast)

        # Should generate gate with reassignment for linearity
        assert "quantum.h" in code
        assert "q[0] = quantum.h(q[0])" in code

    def test_two_qubit_gate(self):
        prog = Main(
            q := QReg("q", 2),
            qb.CX(q[0], q[1]),
        )
        ast = slr_to_ast(prog)

        code = ast_to_guppy(ast)

        # Two-qubit gates return tuple
        assert "quantum.cx" in code
        assert "q[0], q[1] = quantum.cx" in code

    def test_multiple_gates(self):
        prog = Main(
            q := QReg("q", 2),
            qb.H(q[0]),
            qb.X(q[1]),
            qb.CZ(q[0], q[1]),
        )
        ast = slr_to_ast(prog)

        code = ast_to_guppy(ast)

        assert "quantum.h" in code
        assert "quantum.x" in code
        assert "quantum.cz" in code


class TestAstToGuppyPrepMeasure:
    """Prep and measure code generation tests."""

    def test_measure_with_result(self):
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            qb.Measure(q[0]) > c[0],
        )
        ast = slr_to_ast(prog)

        code = ast_to_guppy(ast)

        assert "quantum.measure" in code
        # Measurement results use local variable names (c_0 instead of c[0])
        assert "c_0 = quantum.measure(q[0])" in code
        # Return type should be bool since all qubits are measured
        assert "-> bool:" in code
        assert "return c_0" in code


class TestAstToGuppyControlFlow:
    """Control flow code generation tests."""

    def test_if_statement(self):
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            If(c[0] == 1).Then(
                qb.H(q[0]),
            ),
        )
        ast = slr_to_ast(prog)

        code = ast_to_guppy(ast)

        assert "if" in code
        assert "quantum.h" in code

    def test_if_else_statement(self):
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            If(c[0] == 1).Then(
                qb.H(q[0]),
            ).Else(
                qb.X(q[0]),
            ),
        )
        ast = slr_to_ast(prog)

        code = ast_to_guppy(ast)

        assert "if" in code
        assert "else:" in code
        assert "quantum.h" in code
        assert "quantum.x" in code

    def test_repeat_statement(self):
        prog = Main(
            q := QReg("q", 1),
            Repeat(cond=3).block(
                qb.H(q[0]),
            ),
        )
        ast = slr_to_ast(prog)

        code = ast_to_guppy(ast)

        # Repeat becomes for _ in range(n)
        assert "for _ in range(3):" in code
        assert "quantum.h" in code


class TestAstToGuppyQEC:
    """QEC pattern code generation tests."""

    def test_syndrome_extraction(self):
        prog = Main(
            data := QReg("data", 2),
            ancilla := QReg("ancilla", 1),
            c := CReg("c", 1),
            qb.CX(data[0], ancilla[0]),
            qb.CX(data[1], ancilla[0]),
            qb.Measure(ancilla[0]) > c[0],
        )
        ast = slr_to_ast(prog)

        code = ast_to_guppy(ast)

        # Check function parameters include both arrays
        assert "data: array[qubit, 2]" in code
        assert "ancilla: array[qubit, 1]" in code

        # Check operations
        assert "quantum.cx" in code
        assert "quantum.measure" in code


class TestAstToGuppyGenerator:
    """Tests for AstToGuppy generator class."""

    def test_generator_reusable(self):
        generator = AstToGuppy()

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

    def test_indentation(self):
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            If(c[0] == 1).Then(
                qb.H(q[0]),
            ),
        )
        ast = slr_to_ast(prog)

        code = ast_to_guppy(ast)
        lines = code.split("\n")

        # Find the if line and the line after
        for i, line in enumerate(lines):
            if "if " in line and ":" in line:
                # Next line should be indented
                if i + 1 < len(lines):
                    next_line = lines[i + 1]
                    # Should have more leading spaces than the if line
                    assert next_line.startswith("        ") or next_line.startswith(
                        "    "
                    )
                break


class TestAstToGuppyFullPipeline:
    """End-to-end tests: SLR -> AST -> Guppy."""

    def test_full_pipeline(self):
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

        # Generate Guppy code
        code = ast_to_guppy(ast)

        # Verify structure
        assert "from guppylang import guppy" in code
        assert "@guppy" in code
        assert "def main" in code
        assert "quantum.h" in code
        assert "quantum.cx" in code
        assert "if" in code
        assert "quantum.x" in code

    def test_bell_state_circuit(self):
        """Test a simple Bell state circuit."""
        prog = Main(
            q := QReg("q", 2),
            qb.H(q[0]),
            qb.CX(q[0], q[1]),
        )

        ast = slr_to_ast(prog)
        code = ast_to_guppy(ast)

        # Should have proper Guppy structure
        lines = code.split("\n")

        # Check imports are at the top
        assert lines[0].startswith("from guppylang")

        # Check decorator and function
        assert any("@guppy" in line for line in lines)
        assert any("def main" in line for line in lines)

        # Check gates are in function body (indented)
        gate_lines = [l for l in lines if "quantum." in l]
        assert all(l.startswith("    ") for l in gate_lines)
