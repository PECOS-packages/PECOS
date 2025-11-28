"""Test the Stim REPEAT -> SLR Repeat -> Guppy for loop pipeline."""

import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).parent / "../../../../quantum-pecos/src"),
)

import pytest
from pecos.slr.slr_converter import SlrConverter

# Check if stim is available
try:
    import stim

    STIM_AVAILABLE = True
except ImportError:
    STIM_AVAILABLE = False
    stim = None


@pytest.mark.skipif(not STIM_AVAILABLE, reason="Stim not installed")
class TestRepeatToGuppyPipeline:
    """Test that Stim REPEAT blocks become Guppy for loops."""

    def test_simple_repeat_to_guppy_for_loop(self) -> None:
        """Test basic REPEAT block becomes a for loop in Guppy."""
        stim_circuit = stim.Circuit(
            """
            REPEAT 3 {
                CX 0 1
                CX 1 2
            }
        """,
        )

        # Convert Stim -> SLR
        slr_prog = SlrConverter.from_stim(stim_circuit)

        # Verify SLR has Repeat block
        repeat_blocks = [op for op in slr_prog.ops if type(op).__name__ == "Repeat"]
        assert len(repeat_blocks) == 1, "Should have exactly one Repeat block"

        repeat_block = repeat_blocks[0]
        assert hasattr(repeat_block, "cond"), "Repeat block should have cond attribute"
        assert (
            repeat_block.cond == 3
        ), f"Repeat count should be 3, got {repeat_block.cond}"
        assert (
            len(repeat_block.ops) == 2
        ), f"Should have 2 operations, got {len(repeat_block.ops)}"

        # Convert SLR -> Guppy
        converter = SlrConverter(slr_prog)
        guppy_code = converter.guppy()

        # Verify Guppy contains for loop with correct range
        assert (
            "for _ in range(3):" in guppy_code
        ), "Guppy code should contain 'for _ in range(3):'"
        assert "quantum.cx(" in guppy_code, "Guppy code should contain CX operations"

        # Count for loops and range calls
        for_count = guppy_code.count("for _ in range(3):")
        assert (
            for_count == 1
        ), f"Should have exactly 1 'for _ in range(3):' loop, got {for_count}"

    def test_nested_operations_in_repeat(self) -> None:
        """Test REPEAT block with various gate types."""
        stim_circuit = stim.Circuit(
            """
            H 0
            REPEAT 2 {
                CX 0 1
                H 1
                M 1
            }
        """,
        )

        slr_prog = SlrConverter.from_stim(stim_circuit)
        converter = SlrConverter(slr_prog)
        guppy_code = converter.guppy()

        # Should have for loop with range(2)
        assert "for _ in range(2):" in guppy_code

        # Should contain all the gate types within the loop
        lines = guppy_code.split("\n")
        for_line_idx = None
        for i, line in enumerate(lines):
            if "for _ in range(2):" in line:
                for_line_idx = i
                break

        assert for_line_idx is not None, "Should find the for loop"

        # Check the next few lines after the for loop contain the expected operations
        loop_body = "\n".join(lines[for_line_idx + 1 : for_line_idx + 5])
        assert "quantum.cx(" in loop_body, "Loop body should contain CX"
        assert "quantum.h(" in loop_body, "Loop body should contain H"
        assert "quantum.measure(" in loop_body, "Loop body should contain measurement"

    def test_multiple_repeat_blocks(self) -> None:
        """Test circuit with multiple REPEAT blocks."""
        stim_circuit = stim.Circuit(
            """
            REPEAT 2 {
                H 0
            }
            REPEAT 3 {
                CX 0 1
            }
        """,
        )

        slr_prog = SlrConverter.from_stim(stim_circuit)

        # Should have 2 Repeat blocks in SLR
        repeat_blocks = [op for op in slr_prog.ops if type(op).__name__ == "Repeat"]
        assert (
            len(repeat_blocks) == 2
        ), f"Should have 2 Repeat blocks, got {len(repeat_blocks)}"

        # Check repeat counts
        counts = [block.cond for block in repeat_blocks]
        assert 2 in counts, f"Should have count 2, got {counts}"
        assert 3 in counts, f"Should have count 3, got {counts}"

        # Check Guppy has both for loops
        converter = SlrConverter(slr_prog)
        guppy_code = converter.guppy()
        assert "for _ in range(2):" in guppy_code, "Should have range(2) loop"
        assert "for _ in range(3):" in guppy_code, "Should have range(3) loop"

        # Count for loops from REPEAT blocks (not including array initialization)
        # Split by lines and count quantum operation loops
        lines = guppy_code.split("\n")
        quantum_for_loops = 0
        for i, line in enumerate(lines):
            if "for _ in range(" in line:
                # Check if next non-empty line contains quantum operations
                for j in range(i + 1, min(i + 5, len(lines))):
                    if lines[j].strip():
                        if "quantum." in lines[j] and "array" not in lines[j]:
                            quantum_for_loops += 1
                        break
        assert (
            quantum_for_loops == 2
        ), f"Should have 2 quantum operation for loops, got {quantum_for_loops}"

    def test_qasm_unrolling_vs_guppy_loops(self) -> None:
        """Test that QASM unrolls loops while Guppy keeps them as loops."""
        stim_circuit = stim.Circuit(
            """
            REPEAT 4 {
                H 0
                CX 0 1
            }
        """,
        )

        slr_prog = SlrConverter.from_stim(stim_circuit)

        # QASM should unroll the loop
        converter = SlrConverter(slr_prog)
        qasm_code = converter.qasm(skip_headers=True)
        h_count_qasm = qasm_code.count("h q[0]")
        cx_count_qasm = qasm_code.count("cx q[0],q[1]") + qasm_code.count(
            "cx q[0], q[1]",
        )

        assert h_count_qasm == 4, f"QASM should have 4 H gates, got {h_count_qasm}"
        assert cx_count_qasm == 4, f"QASM should have 4 CX gates, got {cx_count_qasm}"
        assert "for" not in qasm_code.lower(), "QASM should not contain for loops"

        # Guppy should keep it as a loop
        converter = SlrConverter(slr_prog)

        # QASM should unroll the loop
        qasm_code = converter.qasm(skip_headers=True)
        h_count_qasm = qasm_code.count("h q[0]")
        cx_count_qasm = qasm_code.count("cx q[0],q[1]") + qasm_code.count(
            "cx q[0], q[1]",
        )

        assert h_count_qasm == 4, f"QASM should have 4 H gates, got {h_count_qasm}"
        assert cx_count_qasm == 4, f"QASM should have 4 CX gates, got {cx_count_qasm}"
        assert "for" not in qasm_code.lower(), "QASM should not contain for loops"

        # Guppy should keep it as a loop
        guppy_code = converter.guppy()
        assert "for _ in range(4):" in guppy_code, "Guppy should contain range(4) loop"

        # Count quantum operations in Guppy (should be 1 each, inside loop)
        h_count_guppy = guppy_code.count("quantum.h(")
        cx_count_guppy = guppy_code.count("quantum.cx(")

        assert (
            h_count_guppy == 1
        ), f"Guppy should have 1 H call (in loop), got {h_count_guppy}"
        assert (
            cx_count_guppy == 1
        ), f"Guppy should have 1 CX call (in loop), got {cx_count_guppy}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
