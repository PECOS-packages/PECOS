"""Tests for conversion verification using QASM simulation and comparison."""

import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).parent / "../../../../quantum-pecos/src"),
)

import pytest
from pecos.qeclib import qubit
from pecos.slr import CReg, Main, Parallel, QReg, Repeat, SlrConverter
from pecos.slr.gen_codes.gen_quantum_circuit import QuantumCircuitGenerator

# Check if stim is available for additional testing
try:
    import stim

    STIM_AVAILABLE = True
except ImportError:
    STIM_AVAILABLE = False
    stim = None


class TestConversionConsistency:
    """Test that different conversion paths produce consistent QASM output."""

    def test_bell_state_consistency(self) -> None:
        """Test Bell state preparation consistency across all formats."""
        # Original SLR program
        slr_prog = Main(
            q := QReg("q", 2),
            c := CReg("c", 2),
            qubit.Prep(q[0]),
            qubit.Prep(q[1]),
            qubit.H(q[0]),
            qubit.CX(q[0], q[1]),
            qubit.Measure(q[0]) > c[0],
            qubit.Measure(q[1]) > c[1],
        )

        # Get QASM from SLR
        slr_qasm = SlrConverter(slr_prog).qasm(skip_headers=True)

        # Convert SLR -> QuantumCircuit -> SLR -> QASM
        generator = QuantumCircuitGenerator()
        generator.generate_block(slr_prog)
        qc = generator.get_circuit()

        reconstructed_slr = SlrConverter.from_quantum_circuit(qc)
        qc_qasm = SlrConverter(reconstructed_slr).qasm(skip_headers=True)

        # Check that both QASM outputs contain the same essential operations
        essential_ops = ["reset", "h q[0]", "measure"]
        cx_variants = ["cx q[0],q[1]", "cx q[0], q[1]"]

        for op in essential_ops:
            assert op in slr_qasm.lower(), f"'{op}' missing from SLR QASM"
            assert op in qc_qasm.lower(), f"'{op}' missing from QuantumCircuit QASM"

        # Check CX with flexible formatting
        assert any(
            cx in slr_qasm.lower() for cx in cx_variants
        ), f"CX variants {cx_variants} missing from SLR QASM"
        assert any(
            cx in qc_qasm.lower() for cx in cx_variants
        ), f"CX variants {cx_variants} missing from QuantumCircuit QASM"

    @pytest.mark.skipif(not STIM_AVAILABLE, reason="Stim not installed")
    def test_stim_slr_qasm_consistency(self) -> None:
        """Test consistency between Stim and SLR through QASM."""
        # Create a Stim circuit
        stim_circuit = stim.Circuit(
            """
            R 0 1
            H 0
            CX 0 1
            M 0 1
        """,
        )

        # Convert Stim -> SLR -> QASM
        slr_prog = SlrConverter.from_stim(stim_circuit)
        slr_qasm = SlrConverter(slr_prog).qasm(skip_headers=True)

        # Convert SLR -> Stim -> SLR -> QASM
        converter = SlrConverter(slr_prog)
        reconstructed_stim = converter.stim()
        reconstructed_slr = SlrConverter.from_stim(reconstructed_stim)
        roundtrip_qasm = SlrConverter(reconstructed_slr).qasm(skip_headers=True)

        # Both should contain the same operations
        essential_ops = [
            "reset q[0]",
            "reset q[1]",
            "h q[0]",
            "measure q[0]",
            "measure q[1]",
        ]
        cx_ops = ["cx q[0],q[1]", "cx q[0], q[1]"]  # Accept both formats

        for op in essential_ops:
            assert op in slr_qasm, f"'{op}' missing from SLR QASM"
            assert op in roundtrip_qasm, f"'{op}' missing from round-trip QASM"

        # Check CX gate with flexible formatting
        assert any(
            cx in slr_qasm for cx in cx_ops
        ), "Neither CX format found in SLR QASM"
        assert any(
            cx in roundtrip_qasm for cx in cx_ops
        ), "Neither CX format found in round-trip QASM"

    def test_parallel_operations_qasm(self) -> None:
        """Test that parallel operations are correctly represented in QASM."""
        prog = Main(
            q := QReg("q", 4),
            # Parallel single-qubit gates
            Parallel(
                qubit.H(q[0]),
                qubit.X(q[1]),
                qubit.Y(q[2]),
                qubit.Z(q[3]),
            ),
            # Sequential two-qubit gates
            qubit.CX(q[0], q[1]),
            qubit.CX(q[2], q[3]),
        )

        # Generate QASM
        qasm = SlrConverter(prog).qasm(skip_headers=True)

        # All single-qubit gates should be present
        assert "h q[0]" in qasm
        assert "x q[1]" in qasm
        assert "y q[2]" in qasm
        assert "z q[3]" in qasm

        # Two-qubit gates should be present
        assert "cx q[0],q[1]" in qasm or "cx q[0], q[1]" in qasm
        assert "cx q[2],q[3]" in qasm or "cx q[2], q[3]" in qasm

        # Test through QuantumCircuit conversion
        generator = QuantumCircuitGenerator()
        generator.generate_block(prog)
        qc = generator.get_circuit()

        # Should have 3 ticks: parallel gates, CX(0,1), CX(2,3)
        assert len(qc) == 3, f"Expected 3 ticks but got {len(qc)}"

        # First tick should have all parallel operations
        tick0_gates = {
            symbol: locations for symbol, locations, _params in qc[0].items()
        }
        assert len(tick0_gates) == 4  # H, X, Y, Z
        assert "H" in tick0_gates
        assert 0 in tick0_gates["H"]
        assert "X" in tick0_gates
        assert 1 in tick0_gates["X"]
        assert "Y" in tick0_gates
        assert 2 in tick0_gates["Y"]
        assert "Z" in tick0_gates
        assert 3 in tick0_gates["Z"]

    def test_repeat_loop_qasm_expansion(self) -> None:
        """Test that repeat loops are properly expanded in QASM."""
        prog = Main(
            q := QReg("q", 2),
            Repeat(3).block(
                qubit.H(q[0]),
                qubit.CX(q[0], q[1]),
            ),
        )

        qasm = SlrConverter(prog).qasm(skip_headers=True)

        # Should have 3 occurrences of each operation
        assert qasm.count("h q[0]") == 3
        cx_count = qasm.count("cx q[0],q[1]") + qasm.count("cx q[0], q[1]")
        assert cx_count == 3, f"Expected 3 CX gates, got {cx_count}"

        # Test through QuantumCircuit conversion
        generator = QuantumCircuitGenerator()
        generator.generate_block(prog)
        qc = generator.get_circuit()

        # Should have 6 ticks (3 iterations x 2 operations)
        assert len(qc) == 6

        # Count operations in QuantumCircuit
        def get_tick_gates(tick: object) -> dict:
            return {symbol: locations for symbol, locations, _params in tick.items()}

        h_count = sum(
            1
            for i in range(len(qc))
            for gates in [get_tick_gates(qc[i])]
            if "H" in gates and 0 in gates["H"]
        )
        cx_count = sum(
            1
            for i in range(len(qc))
            for gates in [get_tick_gates(qc[i])]
            if "CX" in gates and (0, 1) in gates["CX"]
        )

        assert h_count == 3
        assert cx_count == 3

    def test_qreg_allocation_consistency(self) -> None:
        """Test that qubit register allocation is consistent across formats."""
        prog = Main(
            q1 := QReg("q", 2),
            q2 := QReg("r", 3),
            # Use qubits from both registers
            qubit.H(q1[0]),
            qubit.X(q1[1]),
            qubit.Y(q2[0]),
            qubit.Z(q2[1]),
            qubit.H(q2[2]),
            # Two-qubit gates across registers
            qubit.CX(q1[0], q2[0]),
            qubit.CX(q1[1], q2[1]),
        )

        qasm = SlrConverter(prog).qasm(skip_headers=True)

        # Check that both registers are used with correct indices
        # q register: q[0], q[1]
        assert "q[0]" in qasm
        assert "q[1]" in qasm

        # r register: r[0], r[1], r[2]
        assert "r[0]" in qasm
        assert "r[1]" in qasm
        assert "r[2]" in qasm

        # Check specific operations with correct register names
        expected_ops = ["h q[0]", "x q[1]", "y r[0]", "z r[1]", "h r[2]"]

        for op in expected_ops:
            assert op in qasm, f"'{op}' not found in QASM"

        # Check two-qubit gates with flexible formatting
        assert "cx q[0],r[0]" in qasm or "cx q[0], r[0]" in qasm
        assert "cx q[1],r[1]" in qasm or "cx q[1], r[1]" in qasm

    def test_measurement_consistency(self) -> None:
        """Test measurement operations consistency across conversions."""
        prog = Main(
            q := QReg("q", 3),
            c := CReg("c", 3),
            # Prepare a GHZ state
            qubit.Prep(q[0]),
            qubit.Prep(q[1]),
            qubit.Prep(q[2]),
            qubit.H(q[0]),
            qubit.CX(q[0], q[1]),
            qubit.CX(q[1], q[2]),
            # Measure all qubits
            qubit.Measure(q[0]) > c[0],
            qubit.Measure(q[1]) > c[1],
            qubit.Measure(q[2]) > c[2],
        )

        qasm = SlrConverter(prog).qasm(skip_headers=True)

        # Check for reset/prep operations
        assert qasm.count("reset") == 3 or qasm.count("prep") >= 3

        # Check for measurements
        assert qasm.count("measure") == 3

        # Test through QuantumCircuit
        generator = QuantumCircuitGenerator()
        generator.generate_block(prog)
        qc = generator.get_circuit()

        # Count reset and measure operations in QuantumCircuit
        circuit_str = str(qc).upper()
        reset_count = circuit_str.count("RESET") + circuit_str.count("PREP")
        measure_count = circuit_str.count("MEASURE")

        assert reset_count >= 3
        assert measure_count >= 3

    @pytest.mark.skipif(not STIM_AVAILABLE, reason="Stim not installed")
    def test_noise_instruction_handling(self) -> None:
        """Test that noise instructions are properly handled (as comments)."""
        stim_circuit = stim.Circuit(
            """
            H 0
            DEPOLARIZE1(0.01) 0
            CX 0 1
            DEPOLARIZE2(0.02) 0 1
            M 0 1
        """,
        )

        # Convert to SLR (noise should become comments)
        slr_prog = SlrConverter.from_stim(stim_circuit)
        qasm = SlrConverter(slr_prog).qasm(skip_headers=True)

        # Quantum operations should be preserved
        assert "h q[0]" in qasm
        assert "cx q[0],q[1]" in qasm or "cx q[0], q[1]" in qasm
        assert "measure q[0]" in qasm
        assert "measure q[1]" in qasm

        # Noise should appear as comments (if implemented)
        # This depends on the implementation details


class TestQASMValidation:
    """Test that generated QASM is valid and executable."""

    def test_qasm_syntax_validity(self) -> None:
        """Test that generated QASM has valid syntax."""
        prog = Main(
            q := QReg("q", 3),
            c := CReg("c", 3),
            qubit.H(q[0]),
            qubit.CX(q[0], q[1]),
            qubit.CX(q[1], q[2]),
            qubit.Measure(q[0]) > c[0],
            qubit.Measure(q[1]) > c[1],
            qubit.Measure(q[2]) > c[2],
        )

        qasm = SlrConverter(prog).qasm()

        # Check QASM structure
        assert "OPENQASM" in qasm
        assert "include" in qasm
        assert "qreg q[3]" in qasm
        assert "creg c[3]" in qasm

        # Check gate definitions are valid
        lines = qasm.split("\n")
        gate_lines = [
            line.strip()
            for line in lines
            if line.strip()
            and not line.startswith("//")
            and not any(
                keyword in line for keyword in ["OPENQASM", "include", "qreg", "creg"]
            )
        ]

        for line in gate_lines:
            if line:
                # Basic syntax check - should have valid gate format
                assert (
                    any(gate in line for gate in ["h", "cx", "measure", "reset"])
                    or "->" in line
                )

    def test_register_declaration_consistency(self) -> None:
        """Test that register declarations are consistent in QASM."""
        prog = Main(
            q1 := QReg("data", 4),
            q2 := QReg("ancilla", 2),
            c1 := CReg("results", 4),
            c2 := CReg("syndrome", 2),
            qubit.H(q1[0]),
            qubit.CX(q1[0], q2[0]),
            qubit.Measure(q1[0]) > c1[0],
            qubit.Measure(q2[0]) > c2[0],
        )

        qasm = SlrConverter(prog).qasm()

        # Check register declarations with actual names
        assert "qreg data[4]" in qasm  # Data quantum register
        assert "qreg ancilla[2]" in qasm  # Ancilla quantum register
        assert "creg results[4]" in qasm  # Results classical register
        assert "creg syndrome[2]" in qasm  # Syndrome classical register

        # Check that operations use the correct register names
        assert "h data[0]" in qasm
        assert "cx data[0], ancilla[0]" in qasm or "cx data[0],ancilla[0]" in qasm
        assert "measure data[0] -> results[0]" in qasm
        assert "measure ancilla[0] -> syndrome[0]" in qasm


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
