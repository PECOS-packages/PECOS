"""Tests for QuantumCircuit to/from SLR conversion."""

import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).parent / "../../../../quantum-pecos/src"),
)

import pytest
from pecos.circuits.quantum_circuit import QuantumCircuit
from pecos.qeclib import qubit
from pecos.slr import CReg, For, Main, Parallel, QReg, Repeat, SlrConverter
from pecos.slr.gen_codes.gen_quantum_circuit import QuantumCircuitGenerator


class TestQuantumCircuitToSLR:
    """Test conversion from QuantumCircuit to SLR format."""

    def test_basic_gates(self) -> None:
        """Test conversion of basic single-qubit gates."""
        qc = QuantumCircuit()
        qc.append({"H": {0, 1, 2}})  # Hadamards on qubits 0, 1, 2
        qc.append({"X": {0}, "Y": {1}, "Z": {2}})  # Different gates
        qc.append({"S": {0}, "SDG": {1}, "T": {2}})  # Phase gates

        slr_prog = SlrConverter.from_quantum_circuit(qc)

        # Convert to QASM to verify structure
        qasm = SlrConverter(slr_prog).qasm(skip_headers=True)

        # First tick - all H gates
        assert "h q[0]" in qasm
        assert "h q[1]" in qasm
        assert "h q[2]" in qasm

        # Second tick
        assert "x q[0]" in qasm
        assert "y q[1]" in qasm
        assert "z q[2]" in qasm

        # Third tick
        assert "s q[0]" in qasm or "rz(pi/2) q[0]" in qasm
        assert "sdg q[1]" in qasm or "rz(-pi/2) q[1]" in qasm
        assert "t q[2]" in qasm or "rz(pi/4) q[2]" in qasm

    def test_two_qubit_gates(self) -> None:
        """Test conversion of two-qubit gates."""
        qc = QuantumCircuit()
        qc.append({"CX": {(0, 1), (2, 3)}})  # Two CNOT gates in parallel
        qc.append({"CY": {(1, 2)}})
        qc.append({"CZ": {(0, 3)}})

        slr_prog = SlrConverter.from_quantum_circuit(qc)
        qasm = SlrConverter(slr_prog).qasm(skip_headers=True)

        assert "cx q[0],q[1]" in qasm or "cx q[0], q[1]" in qasm
        assert "cx q[2],q[3]" in qasm or "cx q[2], q[3]" in qasm
        assert "cy q[1],q[2]" in qasm or "cy q[1], q[2]" in qasm
        assert "cz q[0],q[3]" in qasm or "cz q[0], q[3]" in qasm

    def test_measurements(self) -> None:
        """Test conversion of measurement operations."""
        qc = QuantumCircuit()
        qc.append({"RESET": {0, 1}})  # Reset/prep
        qc.append({"H": {0}})
        qc.append({"CX": {(0, 1)}})
        qc.append({"Measure": {0, 1}})

        slr_prog = SlrConverter.from_quantum_circuit(qc)
        qasm = SlrConverter(slr_prog).qasm(skip_headers=True)

        assert "reset q[0]" in qasm
        assert "reset q[1]" in qasm
        assert "h q[0]" in qasm
        assert "cx q[0],q[1]" in qasm or "cx q[0], q[1]" in qasm
        assert "measure q[0]" in qasm
        assert "measure q[1]" in qasm

    def test_parallel_detection(self) -> None:
        """Test that parallel operations in same tick are detected."""
        qc = QuantumCircuit()
        # All gates in one tick - should become a Parallel block
        qc.append({"H": {0}, "X": {1}, "Y": {2}})
        qc.append({"CX": {(0, 1)}})

        slr_prog = SlrConverter.from_quantum_circuit(qc, optimize_parallel=True)

        # Check for Parallel block (either direct Parallel or Block containing multiple ops)
        def has_parallel_structure(op: object) -> bool:
            if op.__class__.__name__ == "Parallel":
                return True
            # If it's a Block with multiple operations, it came from a Parallel optimization
            return bool(
                op.__class__.__name__ == "Block"
                and hasattr(op, "ops")
                and len(op.ops) > 1,
            )

        has_parallel = any(has_parallel_structure(op) for op in slr_prog.ops)
        assert has_parallel, "Should have detected parallel operations"

    def test_empty_circuit(self) -> None:
        """Test conversion of empty circuit."""
        qc = QuantumCircuit()

        slr_prog = SlrConverter.from_quantum_circuit(qc)

        # Should have minimal structure
        assert hasattr(slr_prog, "vars")
        assert hasattr(slr_prog, "ops")


class TestSLRToQuantumCircuit:
    """Test conversion from SLR format to QuantumCircuit."""

    def test_basic_gates_to_qc(self) -> None:
        """Test conversion of basic gates from SLR to QuantumCircuit."""
        prog = Main(
            q := QReg("q", 3),
            qubit.H(q[0]),
            qubit.X(q[1]),
            qubit.Y(q[2]),
            qubit.Z(q[0]),
            qubit.CX(q[0], q[1]),
        )

        # Use the already imported generator

        generator = QuantumCircuitGenerator()
        generator.generate_block(prog)
        qc = generator.get_circuit()

        # Check the circuit structure
        assert len(qc) == 5  # 5 separate ticks (no parallel optimization)

        # Check specific gates
        tick0_gates = {
            symbol: locations for symbol, locations, _params in qc[0].items()
        }
        assert "H" in tick0_gates
        assert 0 in tick0_gates["H"]

        tick1_gates = {
            symbol: locations for symbol, locations, _params in qc[1].items()
        }
        assert "X" in tick1_gates
        assert 1 in tick1_gates["X"]

        tick2_gates = {
            symbol: locations for symbol, locations, _params in qc[2].items()
        }
        assert "Y" in tick2_gates
        assert 2 in tick2_gates["Y"]

        tick3_gates = {
            symbol: locations for symbol, locations, _params in qc[3].items()
        }
        assert "Z" in tick3_gates
        assert 0 in tick3_gates["Z"]

        tick4_gates = {
            symbol: locations for symbol, locations, _params in qc[4].items()
        }
        assert "CX" in tick4_gates
        assert (0, 1) in tick4_gates["CX"]

    def test_measurements_to_qc(self) -> None:
        """Test conversion of measurements from SLR to QuantumCircuit."""
        prog = Main(
            q := QReg("q", 2),
            c := CReg("c", 2),
            qubit.Prep(q[0]),
            qubit.Prep(q[1]),
            qubit.H(q[0]),
            qubit.CX(q[0], q[1]),
            qubit.Measure(q[0]) > c[0],
            qubit.Measure(q[1]) > c[1],
        )

        generator = QuantumCircuitGenerator()
        generator.generate_block(prog)
        qc = generator.get_circuit()

        # Check for reset and measure operations
        circuit_str = str(qc)
        assert "RESET" in circuit_str or "Prep" in circuit_str
        assert "Measure" in circuit_str

    def test_parallel_block_to_qc(self) -> None:
        """Test conversion of Parallel blocks from SLR to QuantumCircuit."""
        prog = Main(
            q := QReg("q", 3),
            Parallel(
                qubit.H(q[0]),
                qubit.X(q[1]),
                qubit.Y(q[2]),
            ),
            qubit.CX(q[0], q[1]),
        )

        generator = QuantumCircuitGenerator()
        generator.generate_block(prog)
        qc = generator.get_circuit()

        # Should have exactly 2 ticks
        assert len(qc) == 2, f"Expected 2 ticks but got {len(qc)}"

        # First tick should have all three gates
        tick0_gates = {
            symbol: locations for symbol, locations, _params in qc[0].items()
        }

        assert "H" in tick0_gates
        assert 0 in tick0_gates["H"]
        assert "X" in tick0_gates
        assert 1 in tick0_gates["X"]
        assert "Y" in tick0_gates
        assert 2 in tick0_gates["Y"]

        # Second tick should have CX
        tick1_gates = {
            symbol: locations for symbol, locations, _params in qc[1].items()
        }

        assert "CX" in tick1_gates
        assert (0, 1) in tick1_gates["CX"]

    def test_repeat_block_to_qc(self) -> None:
        """Test conversion of Repeat blocks from SLR to QuantumCircuit."""
        prog = Main(
            q := QReg("q", 2),
            Repeat(3).block(
                qubit.H(q[0]),
                qubit.CX(q[0], q[1]),
            ),
        )

        generator = QuantumCircuitGenerator()
        generator.generate_block(prog)
        qc = generator.get_circuit()

        # Should have 6 ticks (3 repetitions x 2 gates)
        assert len(qc) == 6, f"Expected 6 ticks but got {len(qc)}"

        # Check pattern repeats
        def get_tick_gates(tick: object) -> dict:
            return {symbol: locations for symbol, locations, _params in tick.items()}

        for i in range(3):
            tick_h = get_tick_gates(qc[i * 2])
            tick_cx = get_tick_gates(qc[i * 2 + 1])
            assert "H" in tick_h
            assert 0 in tick_h["H"]
            assert "CX" in tick_cx
            assert (0, 1) in tick_cx["CX"]

    def test_for_loop_to_qc(self) -> None:
        """Test conversion of For loops from SLR to QuantumCircuit."""
        prog = Main(
            q := QReg("q", 2),
            For("i", range(2)).Do(
                qubit.H(q[0]),
                qubit.X(q[1]),
            ),
        )

        generator = QuantumCircuitGenerator()
        generator.generate_block(prog)
        qc = generator.get_circuit()

        # Should unroll the loop
        assert len(qc) == 4, f"Expected 4 ticks but got {len(qc)}"


class TestQuantumCircuitRoundTrip:
    """Test round-trip conversions between QuantumCircuit and SLR."""

    def test_qc_round_trip(self) -> None:
        """Test QuantumCircuit -> SLR -> QuantumCircuit preserves structure."""
        original = QuantumCircuit()
        original.append({"H": {0, 1}})
        original.append({"CX": {(0, 1)}})
        original.append({"Measure": {0, 1}})

        # Convert to SLR
        slr_prog = SlrConverter.from_quantum_circuit(original)

        # Convert back to QuantumCircuit
        generator = QuantumCircuitGenerator()
        generator.generate_block(slr_prog)
        reconstructed = generator.get_circuit()

        # Both should have same number of ticks
        assert len(original) == len(reconstructed)

        # Check each tick matches
        def get_tick_gates(tick: object) -> dict:
            return {symbol: locations for symbol, locations, _params in tick.items()}

        for i in range(len(original)):
            orig_tick = get_tick_gates(original[i])
            recon_tick = get_tick_gates(reconstructed[i])

            # Same gates in each tick
            assert set(orig_tick.keys()) == set(recon_tick.keys())

            # Same targets for each gate
            for gate in orig_tick:
                assert orig_tick[gate] == recon_tick[gate]

    def test_slr_to_qc_round_trip(self) -> None:
        """Test SLR -> QuantumCircuit -> SLR preserves program structure."""
        original = Main(
            q := QReg("q", 3),
            Parallel(
                qubit.H(q[0]),
                qubit.H(q[1]),
                qubit.H(q[2]),
            ),
            qubit.CX(q[0], q[1]),
            qubit.CX(q[1], q[2]),
        )

        # Convert to QuantumCircuit
        generator = QuantumCircuitGenerator()
        generator.generate_block(original)
        qc = generator.get_circuit()

        # Convert back to SLR
        reconstructed = SlrConverter.from_quantum_circuit(qc, optimize_parallel=True)

        # Convert both to QASM for comparison
        orig_qasm = SlrConverter(original).qasm(skip_headers=True)
        recon_qasm = SlrConverter(reconstructed).qasm(skip_headers=True)

        # Check key operations are preserved
        single_qubit_ops = ["h q[0]", "h q[1]", "h q[2]"]
        for op in single_qubit_ops:
            assert op in orig_qasm, f"'{op}' not in original QASM"
            assert op in recon_qasm, f"'{op}' not in reconstructed QASM"

        # Check CX gates with flexible formatting
        cx_ops = [("cx q[0],q[1]", "cx q[0], q[1]"), ("cx q[1],q[2]", "cx q[1], q[2]")]
        for op_nospace, op_space in cx_ops:
            assert (
                op_nospace in orig_qasm or op_space in orig_qasm
            ), f"Neither '{op_nospace}' nor '{op_space}' in original QASM"
            assert (
                op_nospace in recon_qasm or op_space in recon_qasm
            ), f"Neither '{op_nospace}' nor '{op_space}' in reconstructed QASM"

    def test_complex_circuit_preservation(self) -> None:
        """Test that complex circuit features are preserved."""
        prog = Main(
            q := QReg("q", 4),
            c := CReg("c", 4),
            # Initialize
            qubit.Prep(q[0]),
            qubit.Prep(q[1]),
            qubit.Prep(q[2]),
            qubit.Prep(q[3]),
            # Create entanglement
            qubit.H(q[0]),
            qubit.CX(q[0], q[1]),
            qubit.CX(q[1], q[2]),
            qubit.CX(q[2], q[3]),
            # Measure
            qubit.Measure(q[0]) > c[0],
            qubit.Measure(q[1]) > c[1],
            qubit.Measure(q[2]) > c[2],
            qubit.Measure(q[3]) > c[3],
        )

        # Convert to QuantumCircuit and back
        generator = QuantumCircuitGenerator()
        generator.generate_block(prog)
        qc = generator.get_circuit()

        reconstructed = SlrConverter.from_quantum_circuit(qc)

        # Both should produce similar QASM
        orig_qasm = SlrConverter(prog).qasm(skip_headers=True)
        recon_qasm = SlrConverter(reconstructed).qasm(skip_headers=True)

        # Check all major operations are present
        for op in ["reset", "h q[0]", "measure"]:
            assert op in orig_qasm.lower()
            assert op in recon_qasm.lower()

        # Check CX gates with flexible formatting
        cx_gates = [
            ("cx q[0],q[1]", "cx q[0], q[1]"),
            ("cx q[1],q[2]", "cx q[1], q[2]"),
            ("cx q[2],q[3]", "cx q[2], q[3]"),
        ]
        for op_nospace, op_space in cx_gates:
            assert op_nospace in orig_qasm.lower() or op_space in orig_qasm.lower()
            assert op_nospace in recon_qasm.lower() or op_space in recon_qasm.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
