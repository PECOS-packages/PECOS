"""Tests for SLR patterns that challenge Guppy's linearity requirements.

The v1 emitter rejects divergent post-states at control-flow joins, so the
legacy ``conditional_consumption`` / ``all_paths_consume_resources``
tests have been deleted (they exercised behavior the v1 design
explicitly disallows). The remaining cases cover Block-flattening
patterns that are in v1 scope but not part of the v1 acceptance set in
``tests/slr_tests/ast_guppy/test_v1_acceptance.py``.
"""

import pytest
from pecos.slr import Block, CReg, Main, QReg, Return, SlrConverter
from pecos.slr.qeclib import qubit
from pecos.slr.qeclib.qubit.measures import Measure

from ..ast_guppy._harness import assert_ast_guppy_compiles  # noqa: TID252


class TestLinearityPatterns:
    """Test patterns that challenge Guppy's linear type system."""

    def test_function_modifies_but_returns_qubits(self) -> None:
        """Block subclass that modifies (but does not consume) qubits."""

        class PrepareGHZ(Block):
            """Prepare a GHZ state - modifies qubits but doesn't measure them."""

            def __init__(self, q: QReg) -> None:
                super().__init__()
                self.q = q
                self.ops = [
                    qubit.H(q[0]),
                    qubit.CX(q[0], q[1]),
                    qubit.CX(q[1], q[2]),
                ]

        prog = Main(
            q := QReg("q", 3),
            c := CReg("c", 3),
            PrepareGHZ(q),
            Measure(q) > c,
            Return(c),
        )
        assert_ast_guppy_compiles(prog)

    def test_multiple_functions_passing_qubits(self) -> None:
        """Multiple Block subclasses sharing the same QReg."""

        class ApplyH(Block):
            def __init__(self, q: QReg) -> None:
                super().__init__()
                self.q = q
                self.ops = [qubit.H(q[0])]

        class ApplyCNOT(Block):
            def __init__(self, q: QReg) -> None:
                super().__init__()
                self.q = q
                self.ops = [qubit.CX(q[0], q[1])]

        prog = Main(
            q := QReg("q", 2),
            c := CReg("c", 2),
            ApplyH(q),
            ApplyCNOT(q),
            Measure(q) > c,
            Return(c),
        )
        assert_ast_guppy_compiles(prog)

    def test_partial_array_in_function(self) -> None:
        """Block consumes part of a QReg; the remainder is consumed at root."""

        class MeasureHalf(Block):
            def __init__(self, q: QReg, c: CReg) -> None:
                super().__init__()
                self.q = q
                self.c = c
                self.ops = [
                    Measure(q[0]) > c[0],
                    Measure(q[1]) > c[1],
                ]

        prog = Main(
            q := QReg("q", 4),
            partial := CReg("partial", 2),
            rest := CReg("rest", 2),
            MeasureHalf(q, partial),
            Measure(q[2]) > rest[0],
            Measure(q[3]) > rest[1],
            Return(partial, rest),
        )
        assert_ast_guppy_compiles(prog)

    @pytest.mark.optional_dependency
    def test_empty_main_linearity(self) -> None:
        """Test empty main function satisfies linearity."""
        prog = Main()

        guppy_code = SlrConverter(prog).guppy()

        # Should have a valid main function
        assert "def main" in guppy_code

        # Should compile to HUGR without errors
        try:
            hugr = SlrConverter(prog).hugr()
            assert hugr is not None
        except ImportError as e:
            pytest.fail(f"Empty main should compile: {e}")

    def test_nested_blocks_linearity(self) -> None:
        """Nested Block subclasses are flattened into main and compile."""

        class Inner(Block):
            def __init__(self, q: QReg, c: CReg) -> None:
                super().__init__()
                self.q = q
                self.c = c
                self.ops = [
                    Measure(q[0]) > c[0],
                ]

        class Outer(Block):
            def __init__(self, q: QReg, c: CReg) -> None:
                super().__init__()
                self.q = q
                self.c = c
                self.ops = [
                    qubit.H(q[0]),
                    Inner(q, c),
                ]

        prog = Main(
            q := QReg("q", 2),
            c := CReg("c", 2),
            Outer(q, c),
            Measure(q[1]) > c[1],
            Return(c),
        )
        assert_ast_guppy_compiles(prog)


class TestResourceManagement:
    """Test quantum resource allocation and deallocation patterns."""

    def test_function_with_local_qubits(self) -> None:
        """Block uses an ancilla register that is consumed inside the block."""

        class UseAncilla(Block):
            def __init__(self, data: QReg, ancilla: QReg, result: CReg) -> None:
                super().__init__()
                self.data = data
                self.ancilla = ancilla
                self.result = result
                self.ops = [
                    qubit.CX(data[0], ancilla[0]),
                    Measure(ancilla[0]) > result[0],
                ]

        prog = Main(
            data := QReg("data", 1),
            ancilla := QReg("ancilla", 1),
            result := CReg("result", 1),
            final := CReg("final", 1),
            UseAncilla(data, ancilla, result),
            Measure(data[0]) > final[0],
            Return(result, final),
        )
        assert_ast_guppy_compiles(prog)
