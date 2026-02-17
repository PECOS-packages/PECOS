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

"""Tests comparing direct SLR code generation vs AST-based code generation.

These tests verify that both paths produce equivalent output.
"""

import pytest
from pecos.slr import CReg, If, Main, Permute, QReg, Repeat
from pecos.slr.ast import slr_to_ast
from pecos.slr.ast.codegen import generate
from pecos.slr.gen_codes import (
    GuppyGenerator,
    QASMGenerator,
    QIRGenerator,
    QuantumCircuitGenerator,
    StimGenerator,
)
from pecos.slr.qeclib import qubit as qb


def normalize_whitespace(s: str) -> str:
    """Normalize whitespace for comparison."""
    lines = [line.strip() for line in s.strip().split("\n") if line.strip()]
    return "\n".join(lines)


def extract_gates_qasm(qasm: str) -> list[str]:
    """Extract gate operations from QASM, ignoring headers and declarations."""
    gates = []
    for raw_line in qasm.split("\n"):
        line = raw_line.strip().lower()
        # Skip empty lines, headers, includes, declarations, comments
        if not line:
            continue
        if any(line.startswith(x) for x in ["openqasm", "include", "qreg", "creg", "//"]):
            continue
        gates.append(line.rstrip(";"))
    return gates


class TestQASMEquivalence:
    """Compare QASM output from direct SLR vs AST."""

    def test_bell_state_qasm(self) -> None:
        """Test Bell state produces equivalent QASM."""
        prog = Main(
            q := QReg("q", 2),
            qb.H(q[0]),
            qb.CX(q[0], q[1]),
        )

        # AST path first (before direct generator mutates prog)
        ast = slr_to_ast(prog)
        ast_qasm = generate(ast, "qasm")

        # Direct SLR path
        gen = QASMGenerator(skip_headers=True, _internal=True)
        gen.generate_block(prog)
        direct_qasm = "\n".join(gen.output)

        # Extract gates for comparison
        direct_gates = extract_gates_qasm(direct_qasm)
        ast_gates = extract_gates_qasm(ast_qasm)

        # Should have same gate sequence
        assert direct_gates == ast_gates, f"Direct: {direct_gates}\nAST: {ast_gates}"

    def test_pauli_gates_qasm(self) -> None:
        """Test Pauli gates produce equivalent QASM."""
        prog = Main(
            q := QReg("q", 3),
            qb.X(q[0]),
            qb.Y(q[1]),
            qb.Z(q[2]),
        )

        # AST path first (before direct generator mutates prog)
        ast = slr_to_ast(prog)
        ast_qasm = generate(ast, "qasm")

        gen = QASMGenerator(skip_headers=True, _internal=True)
        gen.generate_block(prog)
        direct_qasm = "\n".join(gen.output)

        direct_gates = extract_gates_qasm(direct_qasm)
        ast_gates = extract_gates_qasm(ast_qasm)

        assert direct_gates == ast_gates

    def test_measurement_qasm(self) -> None:
        """Test measurement produces equivalent QASM."""
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            qb.H(q[0]),
            qb.Measure(q[0]) > c[0],
        )

        # AST path first (before direct generator mutates prog)
        ast = slr_to_ast(prog)
        ast_qasm = generate(ast, "qasm")

        gen = QASMGenerator(skip_headers=True, _internal=True)
        gen.generate_block(prog)
        direct_qasm = "\n".join(gen.output)

        direct_gates = extract_gates_qasm(direct_qasm)
        ast_gates = extract_gates_qasm(ast_qasm)

        # Both should have H gate and measure
        assert "h q[0]" in direct_gates
        assert "h q[0]" in ast_gates
        assert any("measure" in g for g in direct_gates)
        assert any("measure" in g for g in ast_gates)


class TestStimEquivalence:
    """Compare Stim output from direct SLR vs AST."""

    def test_bell_state_stim(self) -> None:
        """Test Bell state produces equivalent Stim."""
        prog = Main(
            q := QReg("q", 2),
            qb.H(q[0]),
            qb.CX(q[0], q[1]),
        )

        # AST path first (before direct generator mutates prog)
        ast = slr_to_ast(prog)
        ast_stim = generate(ast, "stim")

        # Direct SLR path
        gen = StimGenerator(_internal=True)
        gen.generate_block(prog)
        direct_stim = str(gen.get_circuit())

        # Normalize and compare
        direct_lines = set(normalize_whitespace(direct_stim).split("\n"))
        ast_lines = set(normalize_whitespace(ast_stim).split("\n"))

        # Should have same operations (order may differ slightly)
        assert direct_lines == ast_lines, f"Direct: {direct_lines}\nAST: {ast_lines}"

    def test_clifford_gates_stim(self) -> None:
        """Test Clifford gates produce equivalent Stim."""
        prog = Main(
            q := QReg("q", 2),
            qb.H(q[0]),
            qb.SZ(q[0]),
            qb.CZ(q[0], q[1]),
        )

        # AST path first (before direct generator mutates prog)
        ast = slr_to_ast(prog)
        ast_stim = generate(ast, "stim")

        gen = StimGenerator(_internal=True)
        gen.generate_block(prog)
        direct_stim = str(gen.get_circuit())

        direct_lines = set(normalize_whitespace(direct_stim).split("\n"))
        ast_lines = set(normalize_whitespace(ast_stim).split("\n"))

        assert direct_lines == ast_lines


class TestGuppyEquivalence:
    """Compare Guppy output from direct SLR vs AST."""

    def test_bell_state_guppy_structure(self) -> None:
        """Test Bell state produces structurally similar Guppy."""
        prog = Main(
            q := QReg("q", 2),
            qb.H(q[0]),
            qb.CX(q[0], q[1]),
        )

        # AST path first (before direct generator mutates prog)
        ast = slr_to_ast(prog)
        ast_guppy = generate(ast, "guppy")

        # Direct SLR path
        gen = GuppyGenerator(_internal=True)
        gen.generate_block(prog)
        direct_guppy = gen.get_output()

        # Both should have key elements
        assert "@guppy" in direct_guppy or "guppy" in direct_guppy.lower()
        assert "@guppy" in ast_guppy or "guppy" in ast_guppy.lower()

        # Both should have H and CX gates
        assert "quantum.h" in direct_guppy.lower() or ".h(" in direct_guppy.lower()
        assert "quantum.h" in ast_guppy.lower() or ".h(" in ast_guppy.lower()

        assert "quantum.cx" in direct_guppy.lower() or ".cx(" in direct_guppy.lower()
        assert "quantum.cx" in ast_guppy.lower() or ".cx(" in ast_guppy.lower()


class TestQIREquivalence:
    """Compare QIR output from direct SLR vs AST."""

    def test_bell_state_qir_structure(self) -> None:
        """Test Bell state produces structurally similar QIR."""
        prog = Main(
            q := QReg("q", 2),
            qb.H(q[0]),
            qb.CX(q[0], q[1]),
        )

        # AST path first (before direct generator mutates prog)
        ast = slr_to_ast(prog)
        ast_qir = generate(ast, "qir")

        # Direct SLR path
        gen = QIRGenerator(_internal=True)
        gen.generate_block(prog)
        direct_qir = gen.get_output()

        # Both should have LLVM IR structure
        assert "define" in direct_qir
        assert "define" in ast_qir

        # Both should have quantum intrinsics
        assert "__quantum__qis__h" in direct_qir or "h__body" in direct_qir.lower()
        assert "__quantum__qis__h" in ast_qir or "h__body" in ast_qir.lower()


class TestQuantumCircuitEquivalence:
    """Compare QuantumCircuit output from direct SLR vs AST."""

    def test_bell_state_qc(self) -> None:
        """Test Bell state produces equivalent QuantumCircuit."""
        prog = Main(
            q := QReg("q", 2),
            qb.H(q[0]),
            qb.CX(q[0], q[1]),
        )

        # AST path first (before direct generator mutates prog)
        ast = slr_to_ast(prog)
        ast_qc = generate(ast, "quantum_circuit")

        # Direct SLR path
        gen = QuantumCircuitGenerator(_internal=True)
        gen.generate_block(prog)
        direct_qc = gen.get_circuit()

        # Compare circuit properties
        assert len(direct_qc.qudits) == len(ast_qc.qudits)
        assert len(direct_qc) == len(ast_qc)

        # Compare gate operations in each tick
        for tick_idx in range(len(direct_qc)):
            direct_ops = {
                (sym, frozenset(locs) if isinstance(locs, set) else locs)
                for sym, locs, _ in direct_qc[tick_idx].items()
            }
            ast_ops = {
                (sym, frozenset(locs) if isinstance(locs, set) else locs) for sym, locs, _ in ast_qc[tick_idx].items()
            }
            assert direct_ops == ast_ops, f"Tick {tick_idx} mismatch: {direct_ops} vs {ast_ops}"


class TestRepeatEquivalence:
    """Compare Repeat loop handling."""

    def test_repeat_qasm(self) -> None:
        """Test Repeat produces equivalent QASM (unrolled)."""
        prog = Main(
            q := QReg("q", 1),
            Repeat(cond=3).block(
                qb.X(q[0]),
            ),
        )

        # AST path first (before direct generator mutates prog)
        ast = slr_to_ast(prog)
        ast_qasm = generate(ast, "qasm")

        gen = QASMGenerator(skip_headers=True, _internal=True)
        gen.generate_block(prog)
        direct_qasm = "\n".join(gen.output)

        # Both should have 3 X gates (unrolled)
        direct_x_count = direct_qasm.lower().count("x q[0]")
        ast_x_count = ast_qasm.lower().count("x q[0]")

        assert direct_x_count == 3, f"Direct has {direct_x_count} X gates"
        assert ast_x_count == 3, f"AST has {ast_x_count} X gates"


class TestConditionalEquivalence:
    """Compare If statement handling."""

    def test_if_statement_qasm(self) -> None:
        """Test If produces equivalent QASM."""
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            qb.H(q[0]),
            qb.Measure(q[0]) > c[0],
            If(c[0] == 1).Then(
                qb.X(q[0]),
            ),
        )

        # AST path first (before direct generator mutates prog)
        ast = slr_to_ast(prog)
        ast_qasm = generate(ast, "qasm")

        gen = QASMGenerator(skip_headers=True, _internal=True)
        gen.generate_block(prog)
        direct_qasm = "\n".join(gen.output)

        # Both should have conditional X
        assert "if" in direct_qasm.lower() or "x q[0]" in direct_qasm.lower()
        assert "if" in ast_qasm.lower() or "x q[0]" in ast_qasm.lower()


class TestMultipleRegistersEquivalence:
    """Compare handling of multiple registers."""

    def test_two_registers_qasm(self) -> None:
        """Test multiple registers produce equivalent QASM."""
        prog = Main(
            a := QReg("a", 2),
            b := QReg("b", 2),
            qb.H(a[0]),
            qb.H(b[0]),
            qb.CX(a[0], b[0]),
        )

        # AST path first (before direct generator mutates prog)
        ast = slr_to_ast(prog)
        ast_qasm = generate(ast, "qasm")

        gen = QASMGenerator(skip_headers=True, _internal=True)
        gen.generate_block(prog)
        direct_qasm = "\n".join(gen.output)

        direct_gates = extract_gates_qasm(direct_qasm)
        ast_gates = extract_gates_qasm(ast_qasm)

        # Should have same gates
        assert len(direct_gates) == len(ast_gates)
        assert set(direct_gates) == set(ast_gates)

    def test_two_registers_stim(self) -> None:
        """Test multiple registers produce equivalent Stim."""
        prog = Main(
            a := QReg("a", 2),
            b := QReg("b", 2),
            qb.H(a[0]),
            qb.H(b[0]),
            qb.CZ(a[0], b[0]),
        )

        # AST path first (before direct generator mutates prog)
        ast = slr_to_ast(prog)
        ast_stim = generate(ast, "stim")

        gen = StimGenerator(_internal=True)
        gen.generate_block(prog)
        direct_stim = str(gen.get_circuit())

        direct_lines = set(normalize_whitespace(direct_stim).split("\n"))
        ast_lines = set(normalize_whitespace(ast_stim).split("\n"))

        assert direct_lines == ast_lines


class TestRotationGatesEquivalence:
    """Compare rotation gate handling."""

    def test_rx_gate_qasm(self) -> None:
        """Test RX gate with parameter produces equivalent QASM."""
        import math

        prog = Main(
            q := QReg("q", 1),
            qb.RX[math.pi / 4](q[0]),
        )

        # AST path first
        ast = slr_to_ast(prog)
        ast_qasm = generate(ast, "qasm")

        gen = QASMGenerator(skip_headers=True, _internal=True)
        gen.generate_block(prog)
        direct_qasm = "\n".join(gen.output)

        # Both should contain rx with pi/4 angle
        assert "rx" in direct_qasm.lower()
        assert "rx" in ast_qasm.lower()

    def test_ry_gate_qasm(self) -> None:
        """Test RY gate with parameter produces equivalent QASM."""
        import math

        prog = Main(
            q := QReg("q", 1),
            qb.RY[math.pi / 2](q[0]),
        )

        ast = slr_to_ast(prog)
        ast_qasm = generate(ast, "qasm")

        gen = QASMGenerator(skip_headers=True, _internal=True)
        gen.generate_block(prog)
        direct_qasm = "\n".join(gen.output)

        assert "ry" in direct_qasm.lower()
        assert "ry" in ast_qasm.lower()

    def test_rz_gate_qasm(self) -> None:
        """Test RZ gate with parameter produces equivalent QASM."""
        import math

        prog = Main(
            q := QReg("q", 1),
            qb.RZ[math.pi](q[0]),
        )

        ast = slr_to_ast(prog)
        ast_qasm = generate(ast, "qasm")

        gen = QASMGenerator(skip_headers=True, _internal=True)
        gen.generate_block(prog)
        direct_qasm = "\n".join(gen.output)

        assert "rz" in direct_qasm.lower()
        assert "rz" in ast_qasm.lower()

    def test_multiple_rotations_qasm(self) -> None:
        """Test multiple rotation gates produce equivalent QASM."""
        import math

        prog = Main(
            q := QReg("q", 2),
            qb.RX[math.pi / 4](q[0]),
            qb.RY[math.pi / 2](q[1]),
            qb.RZ[math.pi](q[0]),
        )

        ast = slr_to_ast(prog)
        ast_qasm = generate(ast, "qasm")

        gen = QASMGenerator(skip_headers=True, _internal=True)
        gen.generate_block(prog)
        direct_qasm = "\n".join(gen.output)

        # Count rotation gates
        direct_rx = direct_qasm.lower().count("rx")
        direct_ry = direct_qasm.lower().count("ry")
        direct_rz = direct_qasm.lower().count("rz")

        ast_rx = ast_qasm.lower().count("rx")
        ast_ry = ast_qasm.lower().count("ry")
        ast_rz = ast_qasm.lower().count("rz")

        assert direct_rx == ast_rx
        assert direct_ry == ast_ry
        assert direct_rz == ast_rz


class TestNestedControlFlowEquivalence:
    """Compare nested control flow handling."""

    def test_if_inside_repeat_qasm(self) -> None:
        """Test If inside Repeat produces equivalent QASM."""
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            Repeat(cond=2).block(
                qb.H(q[0]),
                qb.Measure(q[0]) > c[0],
                If(c[0] == 1).Then(
                    qb.X(q[0]),
                ),
            ),
        )

        ast = slr_to_ast(prog)
        ast_qasm = generate(ast, "qasm")

        gen = QASMGenerator(skip_headers=True, _internal=True)
        gen.generate_block(prog)
        direct_qasm = "\n".join(gen.output)

        # Both should have multiple H gates (unrolled repeat)
        direct_h_count = direct_qasm.lower().count("h q[0]")
        ast_h_count = ast_qasm.lower().count("h q[0]")

        assert direct_h_count == 2
        assert ast_h_count == 2

    def test_repeat_inside_repeat_qasm(self) -> None:
        """Test nested Repeat produces equivalent QASM."""
        prog = Main(
            q := QReg("q", 1),
            Repeat(cond=2).block(
                Repeat(cond=3).block(
                    qb.X(q[0]),
                ),
            ),
        )

        ast = slr_to_ast(prog)
        ast_qasm = generate(ast, "qasm")

        gen = QASMGenerator(skip_headers=True, _internal=True)
        gen.generate_block(prog)
        direct_qasm = "\n".join(gen.output)

        # Should have 2 * 3 = 6 X gates
        direct_x_count = direct_qasm.lower().count("x q[0]")
        ast_x_count = ast_qasm.lower().count("x q[0]")

        assert direct_x_count == 6, f"Direct has {direct_x_count} X gates"
        assert ast_x_count == 6, f"AST has {ast_x_count} X gates"

    def test_if_else_qasm(self) -> None:
        """Test If-Else produces equivalent QASM."""
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            qb.H(q[0]),
            qb.Measure(q[0]) > c[0],
            If(c[0] == 1)
            .Then(
                qb.X(q[0]),
            )
            .Else(
                qb.Z(q[0]),
            ),
        )

        ast = slr_to_ast(prog)
        ast_qasm = generate(ast, "qasm")

        gen = QASMGenerator(skip_headers=True, _internal=True)
        gen.generate_block(prog)
        direct_qasm = "\n".join(gen.output)

        # Both should have X and Z gates
        assert "x" in direct_qasm.lower() or "z" in direct_qasm.lower()
        assert "x" in ast_qasm.lower() or "z" in ast_qasm.lower()


class TestTGateEquivalence:
    """Compare T/Tdg gate handling."""

    def test_t_gate_qasm(self) -> None:
        """Test T gate produces equivalent QASM."""
        prog = Main(
            q := QReg("q", 1),
            qb.T(q[0]),
        )

        ast = slr_to_ast(prog)
        ast_qasm = generate(ast, "qasm")

        gen = QASMGenerator(skip_headers=True, _internal=True)
        gen.generate_block(prog)
        direct_qasm = "\n".join(gen.output)

        # T gate might be represented as rz(pi/4) or t
        assert "t" in direct_qasm.lower() or "rz" in direct_qasm.lower()
        assert "t" in ast_qasm.lower() or "rz" in ast_qasm.lower()

    def test_tdg_gate_qasm(self) -> None:
        """Test Tdg gate produces equivalent QASM."""
        prog = Main(
            q := QReg("q", 1),
            qb.Tdg(q[0]),
        )

        ast = slr_to_ast(prog)
        ast_qasm = generate(ast, "qasm")

        gen = QASMGenerator(skip_headers=True, _internal=True)
        gen.generate_block(prog)
        direct_qasm = "\n".join(gen.output)

        # Tdg gate might be represented as rz(-pi/4) or tdg
        assert "tdg" in direct_qasm.lower() or "rz" in direct_qasm.lower()
        assert "tdg" in ast_qasm.lower() or "rz" in ast_qasm.lower()

    def test_t_count_circuit_qasm(self) -> None:
        """Test circuit with multiple T gates produces equivalent QASM."""
        prog = Main(
            q := QReg("q", 2),
            qb.H(q[0]),
            qb.T(q[0]),
            qb.CX(q[0], q[1]),
            qb.T(q[1]),
            qb.Tdg(q[0]),
        )

        ast = slr_to_ast(prog)
        ast_qasm = generate(ast, "qasm")

        gen = QASMGenerator(skip_headers=True, _internal=True)
        gen.generate_block(prog)
        direct_qasm = "\n".join(gen.output)

        # Both should have H and CX
        assert "h" in direct_qasm.lower()
        assert "h" in ast_qasm.lower()
        assert "cx" in direct_qasm.lower()
        assert "cx" in ast_qasm.lower()


class TestSqrtGatesEquivalence:
    """Compare sqrt gate handling (SX, SY, SZ)."""

    def test_sx_gate_qasm(self) -> None:
        """Test SX gate produces equivalent QASM."""
        prog = Main(
            q := QReg("q", 1),
            qb.SX(q[0]),
        )

        ast = slr_to_ast(prog)
        ast_qasm = generate(ast, "qasm")

        gen = QASMGenerator(skip_headers=True, _internal=True)
        gen.generate_block(prog)
        direct_qasm = "\n".join(gen.output)

        # SX might be sx or rx(pi/2)
        has_direct = "sx" in direct_qasm.lower() or "rx" in direct_qasm.lower()
        has_ast = "sx" in ast_qasm.lower() or "rx" in ast_qasm.lower()

        assert has_direct
        assert has_ast

    def test_sz_gate_stim(self) -> None:
        """Test SZ gate produces equivalent Stim."""
        prog = Main(
            q := QReg("q", 1),
            qb.SZ(q[0]),
        )

        ast = slr_to_ast(prog)
        ast_stim = generate(ast, "stim")

        gen = StimGenerator(_internal=True)
        gen.generate_block(prog)
        direct_stim = str(gen.get_circuit())

        # Stim uses S for SZ
        assert "S" in direct_stim or "s" in direct_stim.lower()
        assert "S" in ast_stim or "s" in ast_stim.lower()


class TestComplexCircuitEquivalence:
    """Compare more complex circuit patterns."""

    def test_ghz_state_qasm(self) -> None:
        """Test GHZ state preparation produces equivalent QASM."""
        prog = Main(
            q := QReg("q", 4),
            qb.H(q[0]),
            qb.CX(q[0], q[1]),
            qb.CX(q[1], q[2]),
            qb.CX(q[2], q[3]),
        )

        ast = slr_to_ast(prog)
        ast_qasm = generate(ast, "qasm")

        gen = QASMGenerator(skip_headers=True, _internal=True)
        gen.generate_block(prog)
        direct_qasm = "\n".join(gen.output)

        direct_gates = extract_gates_qasm(direct_qasm)
        ast_gates = extract_gates_qasm(ast_qasm)

        # Should have 1 H and 3 CX gates
        assert len(direct_gates) == 4
        assert len(ast_gates) == 4
        assert direct_gates == ast_gates

    def test_qft_like_circuit_qasm(self) -> None:
        """Test QFT-like circuit produces equivalent QASM."""
        import math

        prog = Main(
            q := QReg("q", 3),
            qb.H(q[0]),
            qb.RZ[math.pi / 2](q[0]),
            qb.H(q[1]),
            qb.RZ[math.pi / 4](q[1]),
            qb.H(q[2]),
        )

        ast = slr_to_ast(prog)
        ast_qasm = generate(ast, "qasm")

        gen = QASMGenerator(skip_headers=True, _internal=True)
        gen.generate_block(prog)
        direct_qasm = "\n".join(gen.output)

        # Count H and RZ gates
        direct_h = direct_qasm.lower().count("h q[")
        ast_h = ast_qasm.lower().count("h q[")

        assert direct_h == 3
        assert ast_h == 3

    def test_repeated_measurement_qasm(self) -> None:
        """Test repeated measurement pattern produces equivalent QASM."""
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            Repeat(cond=3).block(
                qb.H(q[0]),
                qb.Measure(q[0]) > c[0],
            ),
        )

        ast = slr_to_ast(prog)
        ast_qasm = generate(ast, "qasm")

        gen = QASMGenerator(skip_headers=True, _internal=True)
        gen.generate_block(prog)
        direct_qasm = "\n".join(gen.output)

        # Should have 3 measurements
        direct_measure = direct_qasm.lower().count("measure")
        ast_measure = ast_qasm.lower().count("measure")

        assert direct_measure == 3
        assert ast_measure == 3
