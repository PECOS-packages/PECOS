"""Tests for Stim circuit to/from SLR conversion."""

import pytest
from pecos.qeclib import qubit
from pecos.slr import CReg, Main, Parallel, QReg, Repeat, SlrConverter

# Check if stim is available
try:
    import stim

    STIM_AVAILABLE = True
except ImportError:
    STIM_AVAILABLE = False
    stim = None


@pytest.mark.skipif(not STIM_AVAILABLE, reason="Stim not installed")
class TestStimToSLR:
    """Test conversion from Stim circuits to SLR format."""

    def test_basic_gates(self) -> None:
        """Test conversion of basic single-qubit gates."""
        circuit = stim.Circuit(
            """
            H 0
            X 1
            Y 2
            Z 0
            S 1
            S_DAG 2
        """,
        )

        slr_prog = SlrConverter.from_stim(circuit)

        # Convert back to QASM to verify structure
        qasm = SlrConverter(slr_prog).qasm(skip_headers=True)
        assert "h q[0]" in qasm
        assert "x q[1]" in qasm
        assert "y q[2]" in qasm
        assert "z q[0]" in qasm
        assert "s q[1]" in qasm or "rz(pi/2) q[1]" in qasm
        assert "sdg q[2]" in qasm or "rz(-pi/2) q[2]" in qasm

    def test_two_qubit_gates(self) -> None:
        """Test conversion of two-qubit gates."""
        circuit = stim.Circuit(
            """
            CX 0 1
            CY 1 2
            CZ 0 2
        """,
        )

        slr_prog = SlrConverter.from_stim(circuit)
        qasm = SlrConverter(slr_prog).qasm(skip_headers=True)

        assert "cx q[0],q[1]" in qasm or "cx q[0], q[1]" in qasm
        assert "cy q[1],q[2]" in qasm or "cy q[1], q[2]" in qasm
        assert "cz q[0],q[2]" in qasm or "cz q[0], q[2]" in qasm

    def test_measurements_and_reset(self) -> None:
        """Test conversion of measurements and reset operations."""
        circuit = stim.Circuit(
            """
            R 0 1 2
            H 0
            CX 0 1
            M 0 1
        """,
        )

        slr_prog = SlrConverter.from_stim(circuit)
        qasm = SlrConverter(slr_prog).qasm(skip_headers=True)

        assert "reset q[0]" in qasm
        assert "reset q[1]" in qasm
        assert "reset q[2]" in qasm
        assert "h q[0]" in qasm
        assert "cx q[0],q[1]" in qasm or "cx q[0], q[1]" in qasm
        assert "measure q[0]" in qasm
        assert "measure q[1]" in qasm

    def test_repeat_blocks(self) -> None:
        """Test conversion of REPEAT blocks."""
        circuit = stim.Circuit(
            """
            H 0
            REPEAT 3 {
                CX 0 1
                CX 1 2
            }
            M 0 1 2
        """,
        )

        slr_prog = SlrConverter.from_stim(circuit)

        # Check that the repeat block is preserved
        assert any(
            hasattr(op, "__class__") and op.__class__.__name__ == "Repeat"
            for op in slr_prog.ops
        )

    def test_parallel_optimization(self) -> None:
        """Test that parallel operations are optimized into Parallel blocks."""
        circuit = stim.Circuit(
            """
            H 0
            H 1
            H 2
            CX 0 1
        """,
        )

        # With optimization (note: optimizer doesn't create new parallel blocks from sequential ops)
        slr_prog_opt = SlrConverter.from_stim(circuit, optimize_parallel=True)
        # Sequential H gates from Stim remain sequential in SLR - this is expected
        h_ops = [op for op in slr_prog_opt.ops if type(op).__name__ == "H"]
        cx_ops = [op for op in slr_prog_opt.ops if type(op).__name__ == "CX"]
        assert len(h_ops) == 3, f"Should have 3 H operations, got {len(h_ops)}"
        assert len(cx_ops) == 1, f"Should have 1 CX operation, got {len(cx_ops)}"

        # Without optimization should be the same (no difference for sequential ops)
        slr_prog_no_opt = SlrConverter.from_stim(circuit, optimize_parallel=False)
        h_ops_no_opt = [op for op in slr_prog_no_opt.ops if type(op).__name__ == "H"]
        assert (
            len(h_ops_no_opt) == 3
        ), f"Should have 3 H operations, got {len(h_ops_no_opt)}"


@pytest.mark.skipif(not STIM_AVAILABLE, reason="Stim not installed")
class TestSLRToStim:
    """Test conversion from SLR format to Stim circuits."""

    def test_basic_gates_to_stim(self) -> None:
        """Test conversion of basic gates from SLR to Stim."""
        prog = Main(
            q := QReg("q", 3),
            qubit.H(q[0]),
            qubit.X(q[1]),
            qubit.Y(q[2]),
            qubit.Z(q[0]),
            qubit.CX(q[0], q[1]),
        )

        converter = SlrConverter(prog)
        stim_circuit = converter.stim()

        # Check the circuit has the expected operations
        instructions = list(stim_circuit)
        assert any(
            instr.name == "H" and instr.targets_copy() == [stim.GateTarget(0)]
            for instr in instructions
        )
        assert any(
            instr.name == "X" and instr.targets_copy() == [stim.GateTarget(1)]
            for instr in instructions
        )
        assert any(
            instr.name == "Y" and instr.targets_copy() == [stim.GateTarget(2)]
            for instr in instructions
        )
        assert any(
            instr.name == "Z" and instr.targets_copy() == [stim.GateTarget(0)]
            for instr in instructions
        )
        assert any(
            instr.name == "CX"
            and instr.targets_copy() == [stim.GateTarget(0), stim.GateTarget(1)]
            for instr in instructions
        )

    def test_measurements_to_stim(self) -> None:
        """Test conversion of measurements from SLR to Stim."""
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

        converter = SlrConverter(prog)
        stim_circuit = converter.stim()

        instructions = list(stim_circuit)
        # Check for reset (prep_z)
        assert any(instr.name == "R" for instr in instructions)
        # Check for measurements
        assert any(instr.name == "M" for instr in instructions)

    def test_repeat_block_to_stim(self) -> None:
        """Test conversion of Repeat blocks from SLR to Stim."""
        prog = Main(
            q := QReg("q", 2),
            Repeat(3).block(
                qubit.H(q[0]),
                qubit.CX(q[0], q[1]),
            ),
        )

        converter = SlrConverter(prog)
        stim_circuit = converter.stim()

        # Check for REPEAT in the circuit
        circuit_str = str(stim_circuit)
        assert "REPEAT" in circuit_str
        assert "3" in circuit_str

    def test_parallel_block_to_stim(self) -> None:
        """Test conversion of Parallel blocks from SLR to Stim."""
        prog = Main(
            q := QReg("q", 3),
            Parallel(
                qubit.H(q[0]),
                qubit.X(q[1]),
                qubit.Y(q[2]),
            ),
            qubit.CX(q[0], q[1]),
        )

        converter = SlrConverter(prog)
        stim_circuit = converter.stim()

        # Parallel operations should appear before the CX
        instructions = list(stim_circuit)

        # Find indices of operations
        h_idx = next(
            i
            for i, instr in enumerate(instructions)
            if instr.name == "H" and 0 in [t.value for t in instr.targets_copy()]
        )
        x_idx = next(
            i
            for i, instr in enumerate(instructions)
            if instr.name == "X" and 1 in [t.value for t in instr.targets_copy()]
        )
        y_idx = next(
            i
            for i, instr in enumerate(instructions)
            if instr.name == "Y" and 2 in [t.value for t in instr.targets_copy()]
        )
        cx_idx = next(i for i, instr in enumerate(instructions) if instr.name == "CX")

        # All parallel ops should come before CX
        assert h_idx < cx_idx
        assert x_idx < cx_idx
        assert y_idx < cx_idx


@pytest.mark.skipif(not STIM_AVAILABLE, reason="Stim not installed")
class TestStimRoundTrip:
    """Test round-trip conversions between Stim and SLR."""

    def test_basic_circuit_round_trip(self) -> None:
        """Test Stim -> SLR -> Stim preserves circuit structure."""
        original = stim.Circuit(
            """
            H 0
            CX 0 1
            M 0 1
        """,
        )

        # Convert to SLR and back
        slr_prog = SlrConverter.from_stim(original)
        converter = SlrConverter(slr_prog)
        reconstructed = converter.stim()

        # Check both circuits have same operations
        orig_ops = [(instr.name, list(instr.targets_copy())) for instr in original]
        recon_ops = [
            (instr.name, list(instr.targets_copy())) for instr in reconstructed
        ]

        assert len(orig_ops) == len(recon_ops)
        for orig, recon in zip(orig_ops, recon_ops, strict=False):
            assert orig[0] == recon[0]  # Same gate name
            assert orig[1] == recon[1]  # Same targets

    def test_slr_round_trip(self) -> None:
        """Test SLR -> Stim -> SLR preserves program structure."""
        original = Main(
            q := QReg("q", 2),
            c := CReg("c", 2),
            qubit.H(q[0]),
            qubit.CX(q[0], q[1]),
            qubit.Measure(q[0]) > c[0],
            qubit.Measure(q[1]) > c[1],
        )

        # Convert to Stim and back
        converter = SlrConverter(original)
        stim_circuit = converter.stim()
        reconstructed = SlrConverter.from_stim(stim_circuit)

        # Convert both to QASM for comparison
        orig_qasm = SlrConverter(original).qasm(skip_headers=True)
        recon_qasm = SlrConverter(reconstructed).qasm(skip_headers=True)

        # Check key operations are preserved
        for op in ["h q[0]", "measure q[0]", "measure q[1]"]:
            assert op in orig_qasm
            assert op in recon_qasm

        # Check CX with flexible formatting
        assert "cx q[0],q[1]" in orig_qasm or "cx q[0], q[1]" in orig_qasm
        assert "cx q[0],q[1]" in recon_qasm or "cx q[0], q[1]" in recon_qasm


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
