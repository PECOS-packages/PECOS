"""Tests for successful HUGR compilation of Guppy code.

These tests verify that various SLR patterns compile all the way to HUGR,
ensuring linearity constraints are satisfied.
"""

import pytest
from pecos.qeclib import qubit
from pecos.qeclib.qubit.measures import Measure
from pecos.slr import Block, CReg, If, Main, QReg, SlrConverter


@pytest.mark.optional_dependency
class TestHugrCompilation:
    """Test that various patterns compile successfully to HUGR."""

    def test_basic_measurement_compiles(self) -> None:
        """Test basic measurement pattern compiles to HUGR."""
        prog = Main(
            q := QReg("q", 3),
            c := CReg("c", 3),
            qubit.H(q[0]),
            qubit.CX(q[0], q[1]),
            Measure(q) > c,
        )

        # Should compile without errors
        hugr = SlrConverter(prog).hugr()
        assert hugr is not None
        assert hasattr(hugr, "__class__")
        assert "Package" in str(type(hugr))

    def test_partial_consumption_compiles(self) -> None:
        """Test partial consumption pattern compiles to HUGR."""

        class MeasureAncillas(Block):
            def __init__(self, data: QReg, ancilla: QReg, syndrome: CReg) -> None:
                super().__init__()
                self.data = data
                self.ancilla = ancilla
                self.syndrome = syndrome
                self.ops = [
                    qubit.CX(data[0], ancilla[0]),
                    Measure(ancilla) > syndrome,
                ]

        prog = Main(
            data := QReg("data", 2),
            ancilla := QReg("ancilla", 1),
            syndrome := CReg("syndrome", 1),
            result := CReg("result", 2),
            MeasureAncillas(data, ancilla, syndrome),
            qubit.H(data[0]),
            Measure(data) > result,
        )

        hugr = SlrConverter(prog).hugr()
        assert hugr is not None

    def test_individual_measurements_compile(self) -> None:
        """Test individual measurements with unpacking compile to HUGR."""
        prog = Main(
            q := QReg("q", 4),
            c := CReg("c", 4),
            # Individual measurements
            Measure(q[0]) > c[0],
            Measure(q[1]) > c[1],
            Measure(q[2]) > c[2],
            Measure(q[3]) > c[3],
        )

        hugr = SlrConverter(prog).hugr()
        assert hugr is not None

    def test_function_with_returns_compiles(self) -> None:
        """Test function returning quantum resources compiles to HUGR."""

        class ProcessQubits(Block):
            def __init__(self, q: QReg) -> None:
                super().__init__()
                self.q = q
                self.ops = [
                    qubit.H(q[0]),
                    qubit.CX(q[0], q[1]),
                    # Return without measuring
                ]

        prog = Main(
            q := QReg("q", 2),
            c := CReg("c", 2),
            ProcessQubits(q),
            Measure(q) > c,
        )

        hugr = SlrConverter(prog).hugr()
        assert hugr is not None

    def test_conditional_measurements_compile(self) -> None:
        """Test measurements with conditionals compile to HUGR."""
        prog = Main(
            q := QReg("q", 2),
            c := CReg("c", 2),
            Measure(q[0]) > c[0],
            If(c[0]).Then(
                qubit.X(q[1]),
            ),
            Measure(q[1]) > c[1],
        )

        hugr = SlrConverter(prog).hugr()
        assert hugr is not None

    def test_nested_blocks_compile(self) -> None:
        """Test nested block structures compile to HUGR."""

        class InnerBlock(Block):
            def __init__(self, q: QReg, c: CReg) -> None:
                super().__init__()
                self.q = q
                self.c = c
                self.ops = [
                    Measure(q[0]) > c[0],
                ]

        class OuterBlock(Block):
            def __init__(self, q: QReg, c: CReg) -> None:
                super().__init__()
                self.q = q
                self.c = c
                self.ops = [
                    qubit.H(q[0]),
                    InnerBlock(q, c),
                    qubit.H(q[1]),
                    Measure(q[1]) > c[1],
                ]

        prog = Main(
            q := QReg("q", 2),
            c := CReg("c", 2),
            OuterBlock(q, c),
        )

        hugr = SlrConverter(prog).hugr()
        assert hugr is not None

    def test_multiple_qregs_compile(self) -> None:
        """Test multiple quantum registers compile to HUGR."""
        prog = Main(
            q1 := QReg("q1", 2),
            q2 := QReg("q2", 2),
            c1 := CReg("c1", 2),
            c2 := CReg("c2", 2),
            qubit.H(q1[0]),
            qubit.H(q2[0]),
            qubit.CX(q1[0], q2[0]),
            Measure(q1) > c1,
            Measure(q2) > c2,
        )

        hugr = SlrConverter(prog).hugr()
        assert hugr is not None

    def test_empty_main_compiles(self) -> None:
        """Test empty main function compiles to HUGR."""
        prog = Main()

        hugr = SlrConverter(prog).hugr()
        assert hugr is not None

    def test_gates_only_with_cleanup_compiles(self) -> None:
        """Test program with only gates (no explicit measurements) compiles."""
        prog = Main(
            q := QReg("q", 3),
            qubit.H(q[0]),
            qubit.CX(q[0], q[1]),
            qubit.CX(q[1], q[2]),
            # Automatic cleanup should handle unconsumed qubits
        )

        hugr = SlrConverter(prog).hugr()
        assert hugr is not None


@pytest.mark.optional_dependency
class TestHugrCompilationFailures:
    """Test cases that should fail HUGR compilation with clear errors."""

    @pytest.mark.xfail(reason="Expected to fail - demonstrates linearity error")
    def test_double_measurement_fails(self) -> None:
        """Test that measuring a qubit twice fails compilation."""
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 2),
            Measure(q[0]) > c[0],
            Measure(q[0]) > c[1],  # Error: q[0] already consumed
        )

        # This should raise an error
        with pytest.raises(RuntimeError) as exc_info:
            SlrConverter(prog).hugr()

        assert "already consumed" in str(exc_info.value).lower()

    @pytest.mark.xfail(reason="Expected to fail - demonstrates missing return")
    def test_function_not_returning_qubits_fails(self) -> None:
        """Test function that doesn't return live qubits fails."""

        class UseButDontReturn(Block):
            def __init__(self, q: QReg) -> None:
                super().__init__()
                self.q = q
                self.ops = [
                    qubit.H(q[0]),
                    # Should return q but doesn't
                ]

        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            UseButDontReturn(q),
            # Try to use q after function that didn't return it
            Measure(q[0]) > c[0],
        )

        with pytest.raises(RuntimeError) as exc_info:
            SlrConverter(prog).hugr()

        assert (
            "linearity" in str(exc_info.value).lower()
            or "not defined" in str(exc_info.value).lower()
        )


@pytest.mark.optional_dependency
class TestQECPatternCompilation:
    """Test real QEC patterns compile to HUGR."""

    def test_steane_code_syndrome_extraction(self) -> None:
        """Test Steane code syndrome extraction compiles."""

        class SteaneXSyndrome(Block):
            def __init__(self, data: QReg, ancilla: QReg, syndrome: CReg) -> None:
                super().__init__()
                self.data = data
                self.ancilla = ancilla
                self.syndrome = syndrome
                self.ops = [
                    # X stabilizers for Steane code
                    qubit.H(ancilla[0]),
                    qubit.CX(ancilla[0], data[0]),
                    qubit.CX(ancilla[0], data[2]),
                    qubit.CX(ancilla[0], data[4]),
                    qubit.CX(ancilla[0], data[6]),
                    qubit.H(ancilla[0]),
                    qubit.H(ancilla[1]),
                    qubit.CX(ancilla[1], data[1]),
                    qubit.CX(ancilla[1], data[2]),
                    qubit.CX(ancilla[1], data[5]),
                    qubit.CX(ancilla[1], data[6]),
                    qubit.H(ancilla[1]),
                    qubit.H(ancilla[2]),
                    qubit.CX(ancilla[2], data[3]),
                    qubit.CX(ancilla[2], data[4]),
                    qubit.CX(ancilla[2], data[5]),
                    qubit.CX(ancilla[2], data[6]),
                    qubit.H(ancilla[2]),
                    Measure(ancilla) > syndrome,
                ]

        prog = Main(
            data := QReg("data", 7),
            ancilla := QReg("ancilla", 3),
            syndrome := CReg("syndrome", 3),
            # Initialize logical state
            qubit.H(data[0]),
            # Extract syndrome
            SteaneXSyndrome(data, ancilla, syndrome),
            # Could apply corrections based on syndrome
            # For now just measure data
            result := CReg("result", 7),
            Measure(data) > result,
        )

        hugr = SlrConverter(prog).hugr()
        assert hugr is not None

    def test_repetition_code_round(self) -> None:
        """Test repetition code error correction round compiles."""

        class RepetitionRound(Block):
            def __init__(self, data: QReg, ancilla: QReg, syndrome: CReg) -> None:
                super().__init__()
                self.data = data
                self.ancilla = ancilla
                self.syndrome = syndrome
                self.ops = [
                    # Measure stabilizers
                    qubit.CX(data[0], ancilla[0]),
                    qubit.CX(data[1], ancilla[0]),
                    qubit.CX(data[1], ancilla[1]),
                    qubit.CX(data[2], ancilla[1]),
                    Measure(ancilla) > syndrome,
                ]

        prog = Main(
            data := QReg("data", 3),
            ancilla := QReg("ancilla", 2),
            syndrome := CReg("syndrome", 2),
            # Multiple rounds of syndrome extraction
            RepetitionRound(data, ancilla, syndrome),
            # Apply corrections
            If(syndrome[0]).Then(
                qubit.X(data[1]),
            ),
            # Another round
            ancilla2 := QReg("ancilla2", 2),
            syndrome2 := CReg("syndrome2", 2),
            RepetitionRound(data, ancilla2, syndrome2),
            # Final measurement
            result := CReg("result", 3),
            Measure(data) > result,
        )

        hugr = SlrConverter(prog).hugr()
        assert hugr is not None
