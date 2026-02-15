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

"""Tests for AST to QIR code generator."""

import pytest

# QIR requires pecos_rslib.llvm which may not be available in all environments
pytest.importorskip("pecos_rslib.llvm")

from pecos.slr import Barrier, CReg, If, Main, QReg, Repeat
from pecos.slr.ast import slr_to_ast
from pecos.slr.ast.codegen import AstToQir, ast_to_qir
from pecos.slr.qeclib import qubit as qb


class TestAstToQirBasic:
    """Basic code generation tests."""

    def test_empty_program(self) -> None:
        """Empty program generates main function definition."""
        prog = Main()
        ast = slr_to_ast(prog)

        llvm_ir = ast_to_qir(ast)

        assert isinstance(llvm_ir, str)
        assert "define void @main()" in llvm_ir

    def test_program_with_qreg(self) -> None:
        """Program with QReg generates qubit operations."""
        prog = Main(
            q := QReg("q", 2),
            qb.H(q[0]),
        )
        ast = slr_to_ast(prog)

        llvm_ir = ast_to_qir(ast)

        assert "define void @main()" in llvm_ir
        assert "__quantum__qis__h__body" in llvm_ir

    def test_has_entry_point_attribute(self) -> None:
        """Program generates QIR entry point and profile attributes."""
        prog = Main(
            _q := QReg("q", 1),
        )
        ast = slr_to_ast(prog)

        llvm_ir = ast_to_qir(ast)

        assert '"entry_point"' in llvm_ir
        assert '"qir_profiles"' in llvm_ir


class TestAstToQirGates:
    """Gate code generation tests."""

    def test_hadamard_gate(self) -> None:
        """Hadamard gate generates h_body call."""
        prog = Main(
            q := QReg("q", 1),
            qb.H(q[0]),
        )
        ast = slr_to_ast(prog)

        llvm_ir = ast_to_qir(ast)

        assert "__quantum__qis__h__body" in llvm_ir

    def test_pauli_gates(self) -> None:
        """Pauli gates (X, Y, Z) generate correct QIS calls."""
        prog = Main(
            q := QReg("q", 1),
            qb.X(q[0]),
            qb.Y(q[0]),
            qb.Z(q[0]),
        )
        ast = slr_to_ast(prog)

        llvm_ir = ast_to_qir(ast)

        assert "__quantum__qis__x__body" in llvm_ir
        assert "__quantum__qis__y__body" in llvm_ir
        assert "__quantum__qis__z__body" in llvm_ir

    def test_phase_gates(self) -> None:
        """Phase gates (SZ, SZdg) generate s_body and s_adj calls."""
        prog = Main(
            q := QReg("q", 1),
            qb.SZ(q[0]),
            qb.SZdg(q[0]),
        )
        ast = slr_to_ast(prog)

        llvm_ir = ast_to_qir(ast)

        assert "__quantum__qis__s__body" in llvm_ir
        assert "__quantum__qis__s__adj" in llvm_ir

    def test_t_gates(self) -> None:
        """T gates (T, Tdg) generate t_body and t_adj calls."""
        prog = Main(
            q := QReg("q", 1),
            qb.T(q[0]),
            qb.Tdg(q[0]),
        )
        ast = slr_to_ast(prog)

        llvm_ir = ast_to_qir(ast)

        assert "__quantum__qis__t__body" in llvm_ir
        assert "__quantum__qis__t__adj" in llvm_ir

    def test_two_qubit_cx_gate(self) -> None:
        """CX gate generates cnot_body call."""
        prog = Main(
            q := QReg("q", 2),
            qb.CX(q[0], q[1]),
        )
        ast = slr_to_ast(prog)

        llvm_ir = ast_to_qir(ast)

        assert "__quantum__qis__cnot__body" in llvm_ir

    def test_two_qubit_cz_gate(self) -> None:
        """CZ gate generates cz_body call."""
        prog = Main(
            q := QReg("q", 2),
            qb.CZ(q[0], q[1]),
        )
        ast = slr_to_ast(prog)

        llvm_ir = ast_to_qir(ast)

        assert "__quantum__qis__cz__body" in llvm_ir


class TestAstToQirPrepMeasure:
    """Prep and measure code generation tests."""

    def test_measurement(self) -> None:
        """Measurement generates mz_to_creg_bit call."""
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            qb.Measure(q[0]) > c[0],
        )
        ast = slr_to_ast(prog)

        llvm_ir = ast_to_qir(ast)

        # Measurement uses mz_to_creg_bit
        assert "mz_to_creg_bit" in llvm_ir

    def test_prep_reset(self) -> None:
        """Prep generates reset_body call."""
        prog = Main(
            q := QReg("q", 1),
            qb.Prep(q[0]),
        )
        ast = slr_to_ast(prog)

        llvm_ir = ast_to_qir(ast)

        assert "__quantum__qis__reset__body" in llvm_ir


class TestAstToQirControlFlow:
    """Control flow code generation tests."""

    def test_if_statement(self) -> None:
        """If statement generates conditional branch."""
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            If(c[0] == 1).Then(
                qb.H(q[0]),
            ),
        )
        ast = slr_to_ast(prog)

        llvm_ir = ast_to_qir(ast)

        # LLVM IR should have conditional branch
        assert "br i1" in llvm_ir
        assert "__quantum__qis__h__body" in llvm_ir

    def test_repeat_unrolled(self) -> None:
        """Repeat generates unrolled gate calls."""
        prog = Main(
            q := QReg("q", 1),
            Repeat(cond=3).block(
                qb.H(q[0]),
            ),
        )
        ast = slr_to_ast(prog)

        llvm_ir = ast_to_qir(ast)

        # Repeat should be unrolled - multiple H gate calls
        # Count occurrences of the call
        assert llvm_ir.count("call void @__quantum__qis__h__body") == 3


class TestAstToQirClassicalRegisters:
    """Classical register tests."""

    def test_creg_creation(self) -> None:
        """CReg generates create_creg call."""
        prog = Main(
            _q := QReg("q", 1),
            _c := CReg("c", 4),
        )
        ast = slr_to_ast(prog)

        llvm_ir = ast_to_qir(ast)

        # Should call create_creg
        assert "create_creg" in llvm_ir

    def test_results_output(self) -> None:
        """Result CReg generates int_record_output call."""
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1, result=True),
            qb.Measure(q[0]) > c[0],
        )
        ast = slr_to_ast(prog)

        llvm_ir = ast_to_qir(ast)

        # Should have int_record_output for results
        assert "__quantum__rt__int_record_output" in llvm_ir


class TestAstToQirQEC:
    """QEC pattern code generation tests."""

    def test_syndrome_extraction(self) -> None:
        """Syndrome extraction generates correct CNOT and measurement calls."""
        prog = Main(
            data := QReg("data", 2),
            ancilla := QReg("ancilla", 1),
            c := CReg("c", 1),
            qb.CX(data[0], ancilla[0]),
            qb.CX(data[1], ancilla[0]),
            qb.Measure(ancilla[0]) > c[0],
        )
        ast = slr_to_ast(prog)

        llvm_ir = ast_to_qir(ast)

        # Two CNOT gate calls
        assert llvm_ir.count("call void @__quantum__qis__cnot__body") == 2
        # One measurement
        assert "mz_to_creg_bit" in llvm_ir


class TestAstToQirGenerator:
    """Tests for AstToQir generator class."""

    def test_generator_reusable(self) -> None:
        """Generator can be reused for multiple programs."""
        generator = AstToQir()

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

        llvm_ir1 = generator.generate(ast1)
        llvm_ir2 = generator.generate(ast2)

        assert "__quantum__qis__h__body" in llvm_ir1
        assert "__quantum__qis__x__body" in llvm_ir2

    def test_qubit_count_tracked(self) -> None:
        """Generator tracks total qubit count across registers."""
        prog = Main(
            _a := QReg("a", 3),
            _b := QReg("b", 2),
        )
        ast = slr_to_ast(prog)

        generator = AstToQir()
        llvm_ir = generator.generate(ast)

        assert generator.context.qubit_count == 5
        assert 'required_num_qubits"="5"' in llvm_ir

    def test_measurement_count_tracked(self) -> None:
        """Generator tracks measurement count."""
        prog = Main(
            q := QReg("q", 3),
            c := CReg("c", 3),
            qb.Measure(q[0]) > c[0],
            qb.Measure(q[1]) > c[1],
            qb.Measure(q[2]) > c[2],
        )
        ast = slr_to_ast(prog)

        generator = AstToQir()
        llvm_ir = generator.generate(ast)

        assert generator.context.measurement_count == 3
        assert 'required_num_results"="3"' in llvm_ir


class TestAstToQirFullPipeline:
    """End-to-end tests: SLR -> AST -> QIR."""

    def test_bell_state_circuit(self) -> None:
        """Bell state generates H and CNOT calls."""
        prog = Main(
            q := QReg("q", 2),
            qb.H(q[0]),
            qb.CX(q[0], q[1]),
        )

        ast = slr_to_ast(prog)
        llvm_ir = ast_to_qir(ast)

        # Verify structure
        assert "define void @main()" in llvm_ir
        assert "__quantum__qis__h__body" in llvm_ir
        assert "__quantum__qis__cnot__body" in llvm_ir
        assert "ret void" in llvm_ir

    def test_ghz_state_circuit(self) -> None:
        """GHZ state generates H and two CNOT calls."""
        prog = Main(
            q := QReg("q", 3),
            qb.H(q[0]),
            qb.CX(q[0], q[1]),
            qb.CX(q[1], q[2]),
        )

        ast = slr_to_ast(prog)
        llvm_ir = ast_to_qir(ast)

        assert "__quantum__qis__h__body" in llvm_ir
        # Two CNOT calls - count occurrences of the call instruction
        assert llvm_ir.count("call void @__quantum__qis__cnot__body") == 2

    def test_valid_llvm_ir_structure(self) -> None:
        """Test that generated IR has valid LLVM structure."""
        prog = Main(
            q := QReg("q", 2),
            c := CReg("c", 2),
            qb.H(q[0]),
            qb.CX(q[0], q[1]),
            qb.Measure(q[0]) > c[0],
            qb.Measure(q[1]) > c[1],
        )

        ast = slr_to_ast(prog)
        llvm_ir = ast_to_qir(ast)

        # Check essential LLVM IR components
        assert llvm_ir.startswith("; ModuleID")
        assert "define void @main()" in llvm_ir
        assert "entry:" in llvm_ir
        assert "ret void" in llvm_ir
        assert "attributes #0" in llvm_ir
