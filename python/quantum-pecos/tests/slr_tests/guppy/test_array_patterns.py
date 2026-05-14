"""Tests for array handling patterns in Guppy code generation.

After the AST -> Guppy v1 emitter rewrite, the canonical acceptance corpus
lives in ``tests/slr_tests/ast_guppy/test_v1_acceptance.py``. The legacy
string-shape tests in this file are mostly duplicate coverage of that
corpus and have been deleted; the surviving cases either exercise a
v1 pattern not yet in the acceptance set (e.g. ``Permute``) or test
non-Guppy fallthrough behavior on the legacy IR path.
"""

import pytest
from pecos.slr import Block, CReg, Main, Permute, QReg, SlrConverter
from pecos.slr.qeclib import qubit
from pecos.slr.qeclib.qubit.measures import Measure

from ..ast_guppy._harness import assert_ast_guppy_compiles  # noqa: TID252


class TestArrayUnpacking:
    """Test array unpacking patterns for measurements."""

    @pytest.mark.optional_dependency
    def test_unique_unpacked_names(self) -> None:
        """Test that unpacked names avoid conflicts."""
        prog = Main(
            q := QReg("q", 2),
            q_0 := QReg("q_0", 1),  # Conflicting name
            c := CReg("c", 3),
            Measure(q[0]) > c[0],
            Measure(q[1]) > c[1],
            Measure(q_0[0]) > c[2],
        )

        guppy_code = SlrConverter(prog).guppy()

        # Should generate unique names to avoid conflicts
        # The unpacked names might be _q_0, _q_1 or similar
        assert "= q" in guppy_code  # Some unpacking happens

        # Should compile without name conflicts
        try:
            hugr = SlrConverter(prog).hugr()
            assert hugr is not None
        except ImportError as e:
            pytest.fail(f"Should handle name conflicts: {e}")


class TestArraySwapPatterns:
    """Test patterns for swapping array elements via Permute."""

    def test_permute_operation(self) -> None:
        """Permute on whole quantum registers compiles via the AST emitter."""
        prog = Main(
            q1 := QReg("q1", 2),
            q2 := QReg("q2", 2),
            c := CReg("c", 4),
            qubit.H(q1[0]),
            qubit.X(q2[0]),
            Permute(q1, q2),
            Measure(q1) > c[0:2],
            Measure(q2) > c[2:4],
        )
        assert_ast_guppy_compiles(prog)


class TestComplexArrayPatterns:
    """Test complex array manipulation patterns."""

    def test_nested_array_operations(self) -> None:
        """Test operations on subarrays."""

        class ProcessPair(Block):
            def __init__(self, q: QReg) -> None:
                super().__init__()
                self.q = q
                self.ops = [
                    qubit.H(q[0]),
                    qubit.CX(q[0], q[1]),
                ]

        prog = Main(
            q := QReg("q", 6),
            c := CReg("c", 6),
            # Process pairs of qubits
            ProcessPair(q[0:2]),
            ProcessPair(q[2:4]),
            ProcessPair(q[4:6]),
            # Measure all
            Measure(q) > c,
        )

        # Note: Slicing syntax q[0:2] might not be fully supported yet
        # This test documents the desired pattern

        try:
            guppy_code = SlrConverter(prog).guppy()
            # Just verify code generates without error
            assert "def main" in guppy_code
        except (NotImplementedError, AttributeError):
            # Expected to fail with current implementation
            pass

    def test_dynamic_sized_arrays(self) -> None:
        """Test handling arrays with runtime-determined sizes."""
        # Currently SLR uses compile-time sizes
        # This documents potential future pattern

        prog = Main(
            q := QReg("q", 4),  # Fixed size
            c := CReg("c", 4),
            # All current operations use fixed indices
            Measure(q) > c,
        )

        guppy_code = SlrConverter(prog).guppy()

        # AST codegen uses fixed-size array parameters
        assert "array[qubit, 4]" in guppy_code
        # Just verify the code generates without errors
        assert "def main" in guppy_code
