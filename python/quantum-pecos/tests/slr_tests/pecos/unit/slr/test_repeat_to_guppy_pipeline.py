"""Test the Stim REPEAT -> SLR Repeat -> Guppy ``for _ in range(...)`` pipeline.

The legacy assertions on ``quantum.cx(`` were the buggy form and have
been removed. The pipeline structure (Stim REPEAT -> SLR Repeat ->
Guppy for-loop with the body inside the loop, vs QASM's unrolled
expansion) is the load-bearing claim and is asserted here. Whole-
program compile is verified via the v1 acceptance harness for the
state-preserving cases.
"""

import sys
from pathlib import Path

import pytest

# Bridge to the v1 compile harness; see test_guppy_generation.py for rationale.
_SLR_TESTS_ROOT = Path(__file__).resolve().parents[3]
if str(_SLR_TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_SLR_TESTS_ROOT))

from ast_guppy._harness import assert_ast_guppy_compiles  # noqa: E402
from pecos.slr.slr_converter import SlrConverter  # noqa: E402

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
        """Stim REPEAT 3 -> SLR Repeat -> Guppy ``for _ in range(3):``."""
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
        assert repeat_block.cond == 3, f"Repeat count should be 3, got {repeat_block.cond}"
        assert len(repeat_block.ops) == 2, f"Should have 2 operations, got {len(repeat_block.ops)}"

        # Convert SLR -> Guppy and verify the loop survives
        converter = SlrConverter(slr_prog)
        guppy_code = converter.guppy()
        assert "for _ in range(3):" in guppy_code

        # Whole-program compile through the AST emitter
        assert_ast_guppy_compiles(slr_prog)

    def test_multiple_repeat_blocks(self) -> None:
        """Two REPEAT blocks become two ``for _ in range(...)`` loops."""
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
        assert len(repeat_blocks) == 2, f"Should have 2 Repeat blocks, got {len(repeat_blocks)}"

        # Check repeat counts
        counts = [block.cond for block in repeat_blocks]
        assert 2 in counts, f"Should have count 2, got {counts}"
        assert 3 in counts, f"Should have count 3, got {counts}"

        # Check Guppy has both for loops
        guppy_code = SlrConverter(slr_prog).guppy()
        assert "for _ in range(2):" in guppy_code, "Should have range(2) loop"
        assert "for _ in range(3):" in guppy_code, "Should have range(3) loop"

        assert_ast_guppy_compiles(slr_prog)

    def test_qasm_unrolling_vs_guppy_loops(self) -> None:
        """QASM unrolls REPEAT bodies; Guppy preserves the ``for`` loop."""
        stim_circuit = stim.Circuit(
            """
            REPEAT 4 {
                H 0
                CX 0 1
            }
        """,
        )

        slr_prog = SlrConverter.from_stim(stim_circuit)
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

        assert_ast_guppy_compiles(slr_prog)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
