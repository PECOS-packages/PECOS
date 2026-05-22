"""Tests for array handling patterns in Guppy code generation.

After the AST -> Guppy v1 emitter rewrite, the canonical acceptance corpus
lives in ``tests/slr_tests/ast_guppy/test_v1_acceptance.py``. The legacy
string-shape tests in this file are mostly duplicate coverage of that
corpus and have been deleted; the surviving cases either exercise a
v1 pattern not yet in the acceptance set (e.g. ``Permute``) or test
non-Guppy fallthrough behavior on the legacy IR path.
"""

import pytest
from pecos.slr import Block, CReg, Main, Permute, QReg, Return, SlrConverter
from pecos.slr.qeclib import qubit
from pecos.slr.qeclib.qubit.measures import Measure

from ..ast_guppy._harness import assert_ast_guppy_compiles  # noqa: TID252


class TestArrayUnpacking:
    """Test array unpacking patterns for measurements."""

    @pytest.mark.optional_dependency
    def test_unique_unpacked_names(self) -> None:
        """Slot-locals are disambiguated against declared register names.

        Before the fix, the Guppy emitter's slot-local formula
        `f"{allocator}_{index}"` would generate `q_0` for `q[0]`, which
        shadows a separately declared `QReg("q_0", ...)` parameter: the
        entry-unpack LHS `q_0, q_1 = q` rebinds the `q_0` param to the
        first qubit of `q`, and the next line `q_0_0, = q_0` then tries
        to unpack a single qubit, raising UnpackableError.

        After the fix, `GuppyContext.populate_slot_locals` builds a
        single namespace-wide table that is read by both
        `GuppyLinearityState.from_allocators(..., slot_locals=...)` and
        the emitter's `_local_name`; colliding candidates get `_`-suffixed
        until unique. So `q[0]` becomes `q_0_` (one underscore appended,
        since `q_0` is taken by the other register), and `q_0[0]` stays
        `q_0_0`. The compiled Guppy + HUGR build cleanly.
        """
        prog = Main(
            q := QReg("q", 2),
            q_0 := QReg("q_0", 1),  # Same name as the q[0] slot-local would default to.
            c := CReg("c", 3),
            Measure(q[0]) > c[0],
            Measure(q[1]) > c[1],
            Measure(q_0[0]) > c[2],
            Return(c),
        )

        guppy_code = SlrConverter(prog).guppy()

        # The fix produces `q_0_, q_1 = q` (q[0] disambiguated against
        # the register `q_0`); `q_0_0, = q_0` is unchanged.
        assert "q_0_, q_1 = q" in guppy_code, guppy_code
        assert "q_0_0, = q_0" in guppy_code, guppy_code
        # The slot-local must NOT be the bare `q_0`, which would shadow
        # the parameter:
        assert "q_0, q_1 = q" not in guppy_code, guppy_code

        # HUGR build must succeed (this is what raised UnpackableError pre-fix).
        hugr = SlrConverter(prog).hugr()
        assert hugr is not None


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
            Return(c),
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
            Return(c),
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
            Return(c),
        )

        guppy_code = SlrConverter(prog).guppy()

        # AST codegen uses fixed-size array parameters
        assert "array[qubit, 4]" in guppy_code
        # Just verify the code generates without errors
        assert "def main" in guppy_code
