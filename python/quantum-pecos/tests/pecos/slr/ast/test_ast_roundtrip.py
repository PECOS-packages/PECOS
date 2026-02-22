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

"""Round-trip tests for AST code generation.

These tests verify that the full pipeline works correctly:
SLR → AST → Target Code → Execution/Verification

This catches subtle bugs where generated code compiles but produces wrong results.
"""

import pytest
from pecos.slr import Barrier, CReg, If, Main, Parallel, QReg, Repeat
from pecos.slr.ast import slr_to_ast
from pecos.slr.ast.codegen import (
    ast_to_guppy,
    ast_to_qasm,
)
from pecos.slr.qeclib import qubit as qb


def tick_to_dict(tick: object) -> dict[str, set]:
    """Convert TickView to a dict {gate_symbol: set(locations)}."""
    return {symbol: locations for symbol, locations, _ in tick.items()}


class TestRoundTripStructure:
    """Tests that verify structure is preserved through the pipeline."""

    def test_gate_count_preserved(self) -> None:
        """Verify gate count is preserved through conversion."""
        prog = Main(
            q := QReg("q", 3),
            qb.H(q[0]),
            qb.H(q[1]),
            qb.H(q[2]),
            qb.CX(q[0], q[1]),
            qb.CX(q[1], q[2]),
        )

        ast = slr_to_ast(prog)

        # Count gates in AST
        from pecos.slr.ast.analysis import count_resources

        resources = count_resources(ast)

        assert resources.single_qubit_gates == 3  # 3 H gates
        assert resources.two_qubit_gates == 2  # 2 CX gates
        assert resources.total_gates == 5

    def test_qubit_count_preserved(self) -> None:
        """Verify qubit count is preserved through conversion."""
        prog = Main(
            _a := QReg("a", 3),
            _b := QReg("b", 2),
        )

        ast = slr_to_ast(prog)

        from pecos.slr.ast.analysis import count_resources

        resources = count_resources(ast)

        assert resources.qubit_count == 5

    def test_measurement_count_preserved(self) -> None:
        """Verify measurement count is preserved through conversion."""
        prog = Main(
            q := QReg("q", 3),
            c := CReg("c", 3),
            qb.Measure(q[0]) > c[0],
            qb.Measure(q[1]) > c[1],
            qb.Measure(q[2]) > c[2],
        )

        ast = slr_to_ast(prog)

        from pecos.slr.ast.analysis import count_resources

        resources = count_resources(ast)

        assert resources.measurement_count == 3


class TestRoundTripQASM:
    """Round-trip tests through QASM generation."""

    def test_bell_state_qasm_structure(self) -> None:
        """Test Bell state generates correct QASM structure."""
        prog = Main(
            q := QReg("q", 2),
            qb.H(q[0]),
            qb.CX(q[0], q[1]),
        )

        ast = slr_to_ast(prog)
        qasm = ast_to_qasm(ast)

        # Verify essential QASM elements
        lines = qasm.strip().split("\n")
        non_empty = [line.strip() for line in lines if line.strip()]

        # Should have: header, include, qreg, h, cx
        assert any("OPENQASM" in line for line in non_empty)
        assert any("qreg q[2]" in line for line in non_empty)
        assert any("h q[0]" in line for line in non_empty)
        assert any("cx q[0], q[1]" in line for line in non_empty)

    def test_ghz_state_qasm_structure(self) -> None:
        """Test GHZ state generates correct QASM structure."""
        prog = Main(
            q := QReg("q", 3),
            qb.H(q[0]),
            qb.CX(q[0], q[1]),
            qb.CX(q[1], q[2]),
        )

        ast = slr_to_ast(prog)
        qasm = ast_to_qasm(ast)

        assert "h q[0]" in qasm
        assert "cx q[0], q[1]" in qasm
        assert "cx q[1], q[2]" in qasm

    def test_qec_syndrome_qasm_structure(self) -> None:
        """Test QEC syndrome extraction generates correct QASM."""
        prog = Main(
            data := QReg("data", 2),
            ancilla := QReg("ancilla", 1),
            c := CReg("c", 1),
            qb.Prep(ancilla[0]),
            qb.CX(data[0], ancilla[0]),
            qb.CX(data[1], ancilla[0]),
            qb.Measure(ancilla[0]) > c[0],
        )

        ast = slr_to_ast(prog)
        qasm = ast_to_qasm(ast)

        # Verify all operations present
        assert "qreg data[2]" in qasm
        assert "qreg ancilla[1]" in qasm
        assert "creg c[1]" in qasm
        assert "reset ancilla[0]" in qasm
        assert "cx data[0], ancilla[0]" in qasm
        assert "cx data[1], ancilla[0]" in qasm
        assert "measure ancilla[0] -> c[0]" in qasm

    def test_conditional_qasm_structure(self) -> None:
        """Test conditional operations generate correct QASM."""
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            If(c[0] == 1).Then(
                qb.X(q[0]),
            ),
        )

        ast = slr_to_ast(prog)
        qasm = ast_to_qasm(ast)

        assert "if(c[0] == 1) x q[0]" in qasm

    def test_repeat_unrolled_qasm(self) -> None:
        """Test repeat loops are unrolled in QASM."""
        prog = Main(
            q := QReg("q", 1),
            Repeat(cond=5).block(
                qb.H(q[0]),
            ),
        )

        ast = slr_to_ast(prog)
        qasm = ast_to_qasm(ast)

        # Should have 5 H gates
        assert qasm.count("h q[0]") == 5


class TestRoundTripGuppy:
    """Round-trip tests through Guppy generation."""

    def test_bell_state_guppy_structure(self) -> None:
        """Test Bell state generates correct Guppy structure."""
        prog = Main(
            q := QReg("q", 2),
            qb.H(q[0]),
            qb.CX(q[0], q[1]),
        )

        ast = slr_to_ast(prog)
        guppy = ast_to_guppy(ast)

        # Verify Guppy imports and structure
        assert "from guppylang import guppy" in guppy
        assert "from guppylang.std import quantum" in guppy
        assert "@guppy" in guppy
        assert "def main" in guppy.lower()
        assert "quantum.h" in guppy
        assert "quantum.cx" in guppy

    def test_measurement_guppy_structure(self) -> None:
        """Test measurement generates correct Guppy structure."""
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            qb.Measure(q[0]) > c[0],
        )

        ast = slr_to_ast(prog)
        guppy = ast_to_guppy(ast)

        assert "quantum.measure" in guppy


class TestRoundTripStim:
    """Round-trip tests through Stim generation."""

    @pytest.fixture
    def _require_stim(self) -> object:
        """Skip tests if stim is not available."""
        return pytest.importorskip("stim")

    @pytest.mark.usefixtures("_require_stim")
    def test_bell_state_stim_simulation(self) -> None:
        """Test Bell state can be simulated with Stim."""
        from pecos.slr.ast.codegen import ast_to_stim

        prog = Main(
            q := QReg("q", 2),
            c := CReg("c", 2),
            qb.H(q[0]),
            qb.CX(q[0], q[1]),
            qb.Measure(q[0]) > c[0],
            qb.Measure(q[1]) > c[1],
        )

        ast = slr_to_ast(prog)
        circuit = ast_to_stim(ast)

        # Sample from the circuit
        sampler = circuit.compile_sampler()
        samples = sampler.sample(shots=100)

        # Bell state should have correlated measurements
        # Both qubits should measure the same
        for sample in samples:
            assert sample[0] == sample[1], "Bell state qubits should be correlated"

    @pytest.mark.usefixtures("_require_stim")
    def test_ghz_state_stim_simulation(self) -> None:
        """Test GHZ state can be simulated with Stim."""
        from pecos.slr.ast.codegen import ast_to_stim

        prog = Main(
            q := QReg("q", 3),
            c := CReg("c", 3),
            qb.H(q[0]),
            qb.CX(q[0], q[1]),
            qb.CX(q[1], q[2]),
            qb.Measure(q[0]) > c[0],
            qb.Measure(q[1]) > c[1],
            qb.Measure(q[2]) > c[2],
        )

        ast = slr_to_ast(prog)
        circuit = ast_to_stim(ast)

        # Sample from the circuit
        sampler = circuit.compile_sampler()
        samples = sampler.sample(shots=100)

        # GHZ state should have all qubits the same
        for sample in samples:
            assert sample[0] == sample[1] == sample[2], "GHZ state qubits should all be correlated"

    @pytest.mark.usefixtures("_require_stim")
    def test_repeat_block_preserved(self) -> None:
        """Test repeat blocks are preserved in Stim."""
        from pecos.slr.ast.codegen import ast_to_stim_str

        prog = Main(
            q := QReg("q", 1),
            Repeat(cond=10).block(
                qb.H(q[0]),
            ),
        )

        ast = slr_to_ast(prog)
        stim_str = ast_to_stim_str(ast)

        # Should use REPEAT block, not unroll
        assert "REPEAT 10" in stim_str


class TestRoundTripQuantumCircuit:
    """Round-trip tests through QuantumCircuit generation."""

    def test_bell_state_quantum_circuit_structure(self) -> None:
        """Test Bell state generates correct QuantumCircuit structure."""
        from pecos.slr.ast.codegen import ast_to_quantum_circuit

        prog = Main(
            q := QReg("q", 2),
            qb.H(q[0]),
            qb.CX(q[0], q[1]),
        )

        ast = slr_to_ast(prog)
        circuit = ast_to_quantum_circuit(ast)

        # Should have 2 ticks
        assert len(circuit) == 2

        # First tick: H on qubit 0
        tick0 = tick_to_dict(circuit[0])
        assert "H" in tick0
        assert 0 in tick0["H"]

        # Second tick: CX on (0, 1)
        tick1 = tick_to_dict(circuit[1])
        assert "CX" in tick1
        assert (0, 1) in tick1["CX"]

    def test_parallel_operations_same_tick(self) -> None:
        """Test parallel operations are in same tick."""
        from pecos.slr.ast.codegen import ast_to_quantum_circuit

        prog = Main(
            q := QReg("q", 2),
            Parallel(
                qb.H(q[0]),
                qb.H(q[1]),
            ),
        )

        ast = slr_to_ast(prog)
        circuit = ast_to_quantum_circuit(ast)

        # Parallel operations in single tick
        assert len(circuit) == 1
        tick = tick_to_dict(circuit[0])
        assert "H" in tick
        assert 0 in tick["H"]
        assert 1 in tick["H"]


class TestRoundTripQIR:
    """Round-trip tests through QIR generation."""

    @pytest.fixture
    def _require_llvm(self) -> object:
        """Skip tests if LLVM is not available."""
        return pytest.importorskip("pecos_rslib.llvm")

    @pytest.mark.usefixtures("_require_llvm")
    def test_bell_state_qir_structure(self) -> None:
        """Test Bell state generates valid QIR."""
        from pecos.slr.ast.codegen import ast_to_qir

        prog = Main(
            q := QReg("q", 2),
            qb.H(q[0]),
            qb.CX(q[0], q[1]),
        )

        ast = slr_to_ast(prog)
        qir = ast_to_qir(ast)

        # Verify essential QIR elements
        assert "define void @main()" in qir
        assert "__quantum__qis__h__body" in qir
        assert "__quantum__qis__cnot__body" in qir
        assert "ret void" in qir

    @pytest.mark.usefixtures("_require_llvm")
    def test_qir_qubit_count_attribute(self) -> None:
        """Test QIR includes correct qubit count attribute."""
        from pecos.slr.ast.codegen import ast_to_qir

        prog = Main(
            q := QReg("q", 5),
            qb.H(q[0]),
        )

        ast = slr_to_ast(prog)
        qir = ast_to_qir(ast)

        assert 'required_num_qubits"="5"' in qir

    @pytest.mark.usefixtures("_require_llvm")
    def test_qir_measurement_count_attribute(self) -> None:
        """Test QIR includes correct measurement count attribute."""
        from pecos.slr.ast.codegen import ast_to_qir

        prog = Main(
            q := QReg("q", 3),
            c := CReg("c", 3),
            qb.Measure(q[0]) > c[0],
            qb.Measure(q[1]) > c[1],
            qb.Measure(q[2]) > c[2],
        )

        ast = slr_to_ast(prog)
        qir = ast_to_qir(ast)

        assert 'required_num_results"="3"' in qir


class TestRoundTripConsistency:
    """Tests that verify consistency across different generators."""

    def test_same_qubit_order_all_generators(self) -> None:
        """Test that all generators use consistent qubit ordering."""
        prog = Main(
            a := QReg("a", 2),
            b := QReg("b", 2),
            qb.H(a[0]),
            qb.H(b[0]),
            qb.CX(a[0], b[0]),
        )

        ast = slr_to_ast(prog)

        # QASM
        qasm = ast_to_qasm(ast, include_header=False)
        assert "h a[0]" in qasm
        assert "h b[0]" in qasm
        assert "cx a[0], b[0]" in qasm

        # Guppy
        guppy = ast_to_guppy(ast)
        assert "a[0]" in guppy
        assert "b[0]" in guppy

    def test_gate_sequence_preserved_all_generators(self) -> None:
        """Test that gate sequence is preserved in all generators."""
        prog = Main(
            q := QReg("q", 1),
            qb.H(q[0]),
            qb.X(q[0]),
            qb.Z(q[0]),
        )

        ast = slr_to_ast(prog)

        # QASM - check order
        qasm = ast_to_qasm(ast, include_header=False)
        h_pos = qasm.find("h q[0]")
        x_pos = qasm.find("x q[0]")
        z_pos = qasm.find("z q[0]")
        assert h_pos < x_pos < z_pos, "Gate order not preserved in QASM"

        # Guppy - check order
        guppy = ast_to_guppy(ast)
        h_pos = guppy.find("quantum.h")
        x_pos = guppy.find("quantum.x")
        z_pos = guppy.find("quantum.z")
        assert h_pos < x_pos < z_pos, "Gate order not preserved in Guppy"


class TestRoundTripEdgeCases:
    """Edge case tests for round-trip generation."""

    def test_empty_program_all_generators(self) -> None:
        """Test empty program works with all generators."""
        from pecos.slr.ast.codegen import (
            ast_to_quantum_circuit,
        )

        prog = Main()
        ast = slr_to_ast(prog)

        # Should not raise errors
        qasm = ast_to_qasm(ast)
        guppy = ast_to_guppy(ast)
        qc = ast_to_quantum_circuit(ast)

        assert isinstance(qasm, str)
        assert isinstance(guppy, str)
        assert len(qc) == 0

    def test_deeply_nested_repeat(self) -> None:
        """Test deeply nested repeat blocks."""
        prog = Main(
            q := QReg("q", 1),
            Repeat(cond=2).block(
                Repeat(cond=2).block(
                    qb.H(q[0]),
                ),
            ),
        )

        ast = slr_to_ast(prog)
        qasm = ast_to_qasm(ast)

        # Should have 2 * 2 = 4 H gates when unrolled
        assert qasm.count("h q[0]") == 4

    def test_all_single_qubit_gates(self) -> None:
        """Test all supported single qubit gates."""
        prog = Main(
            q := QReg("q", 1),
            qb.H(q[0]),
            qb.X(q[0]),
            qb.Y(q[0]),
            qb.Z(q[0]),
            qb.SZ(q[0]),
            qb.SZdg(q[0]),
            qb.T(q[0]),
            qb.Tdg(q[0]),
        )

        ast = slr_to_ast(prog)

        # Should not raise errors
        qasm = ast_to_qasm(ast)
        guppy = ast_to_guppy(ast)

        assert isinstance(qasm, str)
        assert isinstance(guppy, str)

    def test_all_two_qubit_gates(self) -> None:
        """Test all supported two qubit gates."""
        prog = Main(
            q := QReg("q", 2),
            qb.CX(q[0], q[1]),
            qb.CZ(q[0], q[1]),
            qb.CY(q[0], q[1]),
        )

        ast = slr_to_ast(prog)

        # Should not raise errors
        qasm = ast_to_qasm(ast)
        ast_to_guppy(ast)

        assert "cx" in qasm.lower()
        assert "cz" in qasm.lower()
        assert "cy" in qasm.lower()
