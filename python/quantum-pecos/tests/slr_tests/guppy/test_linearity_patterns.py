"""Tests for SLR patterns that challenge Guppy's linearity requirements.

These tests verify that the Guppy generator correctly handles:
- Functions that modify but don't consume qubits
- Partial measurements in main function
- Conditional consumption patterns
- Resource cleanup for linearity
"""

import pytest
from pecos.slr import Block, CReg, If, Main, QReg, SlrConverter
from pecos.slr.qeclib import qubit
from pecos.slr.qeclib.qubit.measures import Measure


class TestLinearityPatterns:
    """Test patterns that challenge Guppy's linear type system."""

    def test_function_modifies_but_returns_qubits(self) -> None:
        """Test function that modifies qubits and returns them."""

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
            # Use q after function call
            Measure(q) > c,
        )

        guppy_code = SlrConverter(prog).guppy()

        # AST codegen flattens blocks into main
        # Check that GHZ operations are present
        assert "quantum.h(q[0])" in guppy_code
        assert "quantum.cx(q[0], q[1])" in guppy_code
        assert "quantum.cx(q[1], q[2])" in guppy_code

        # Measurements should follow
        assert "quantum.measure(q[0])" in guppy_code

    def test_main_with_unmeasured_qubits(self) -> None:
        """Test main function that doesn't measure all qubits."""
        prog = Main(
            q := QReg("q", 5),
            c := CReg("c", 2),
            qubit.H(q[0]),
            qubit.CX(q[0], q[1]),
            # Only measure first two qubits
            Measure(q[0]) > c[0],
            Measure(q[1]) > c[1],
            # q[2], q[3], q[4] are not measured
        )

        guppy_code = SlrConverter(prog).guppy()

        # AST codegen returns unconsumed qubits
        # Function should return the array since not all consumed
        assert "array[qubit, 5]" in guppy_code
        # Measurements are present
        assert "quantum.measure(q[0])" in guppy_code
        assert "quantum.measure(q[1])" in guppy_code

    def test_conditional_consumption(self) -> None:
        """Test conditional consumption of quantum resources."""
        prog = Main(
            q := QReg("q", 2),
            flag := CReg("flag", 1),
            result := CReg("result", 1),
            # Set flag based on some condition
            Measure(q[0]) > flag[0],
            # Conditionally measure second qubit
            If(flag[0]).Then(
                Measure(q[1]) > result[0],
            ),
            # Note: q[1] might not be consumed if flag[0] is False
        )

        guppy_code = SlrConverter(prog).guppy()

        # Should handle conditional consumption
        assert "if flag_0:" in guppy_code

    def test_multiple_functions_passing_qubits(self) -> None:
        """Test passing qubits through multiple functions."""

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
        )

        guppy_code = SlrConverter(prog).guppy()

        # AST codegen flattens blocks into main
        assert "quantum.h(q[0])" in guppy_code
        assert "quantum.cx(q[0], q[1])" in guppy_code
        assert "quantum.measure(q[0])" in guppy_code
        assert "quantum.measure(q[1])" in guppy_code

    def test_partial_array_in_function(self) -> None:
        """Test function that consumes part of an array."""

        class MeasureHalf(Block):
            def __init__(self, q: QReg, c: CReg) -> None:
                super().__init__()
                self.q = q
                self.c = c
                self.ops = [
                    Measure(q[0]) > c[0],
                    Measure(q[1]) > c[1],
                    # q[2] and q[3] remain unmeasured
                ]

        prog = Main(
            q := QReg("q", 4),
            partial := CReg("partial", 2),
            rest := CReg("rest", 2),
            MeasureHalf(q, partial),
            # Measure remaining qubits
            Measure(q[2]) > rest[0],
            Measure(q[3]) > rest[1],
        )

        guppy_code = SlrConverter(prog).guppy()

        # AST codegen flattens blocks - verify all measurements present
        assert "partial_0 = quantum.measure(q[0])" in guppy_code
        assert "partial_1 = quantum.measure(q[1])" in guppy_code
        assert "rest_0 = quantum.measure(q[2])" in guppy_code
        assert "rest_1 = quantum.measure(q[3])" in guppy_code

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
        """Test nested blocks handle linearity correctly."""

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
                    # q[1] still needs to be handled
                ]

        prog = Main(
            q := QReg("q", 2),
            c := CReg("c", 2),
            Outer(q, c),
            Measure(q[1]) > c[1],
        )

        guppy_code = SlrConverter(prog).guppy()

        # AST codegen flattens nested blocks
        assert "quantum.h(q[0])" in guppy_code
        assert "c_0 = quantum.measure(q[0])" in guppy_code
        assert "c_1 = quantum.measure(q[1])" in guppy_code


class TestResourceManagement:
    """Test quantum resource allocation and deallocation patterns."""

    def test_function_with_local_qubits(self) -> None:
        """Test function that allocates and consumes local qubits."""

        class UseAncilla(Block):
            def __init__(self, data: QReg, ancilla: QReg, result: CReg) -> None:
                super().__init__()
                self.data = data
                self.ancilla = ancilla
                self.result = result
                self.ops = [
                    qubit.CX(data[0], ancilla[0]),
                    Measure(ancilla[0]) > result[0],
                    # ancilla consumed, data returned
                ]

        prog = Main(
            data := QReg("data", 1),
            ancilla := QReg("ancilla", 1),
            result := CReg("result", 1),
            final := CReg("final", 1),
            UseAncilla(data, ancilla, result),
            Measure(data[0]) > final[0],
        )

        guppy_code = SlrConverter(prog).guppy()

        # AST codegen flattens blocks
        assert "quantum.cx(data[0], ancilla[0])" in guppy_code
        assert "result_0 = quantum.measure(ancilla[0])" in guppy_code
        assert "final_0 = quantum.measure(data[0])" in guppy_code

    def test_all_paths_consume_resources(self) -> None:
        """Test that all execution paths consume quantum resources."""
        prog = Main(
            q := QReg("q", 2),
            flag := CReg("flag", 1),
            result := CReg("result", 2),
            # Get a flag
            Measure(q[0]) > flag[0],
            If(flag[0])
            .Then(
                qubit.X(q[1]),
                Measure(q[1]) > result[1],
            )
            .Else(
                qubit.Z(q[1]),
                Measure(q[1]) > result[0],  # Different index
            ),
        )

        guppy_code = SlrConverter(prog).guppy()

        # Both branches should consume q[1]
        assert "if flag_0:" in guppy_code
        assert "else:" in guppy_code
        assert "quantum.x(q[1])" in guppy_code
        assert "quantum.z(q[1])" in guppy_code
