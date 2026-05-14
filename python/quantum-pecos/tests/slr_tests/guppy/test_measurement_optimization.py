"""Tests for measurement optimization in Guppy code generation.

These tests verify that the Guppy generator handles measurement patterns:
- Full array measurements
- Selective measurements
- Mixed measurement patterns
"""

from pecos.slr import Block, CReg, If, Main, QReg, SlrConverter
from pecos.slr.qeclib import qubit
from pecos.slr.qeclib.qubit.measures import Measure


class TestMeasurementOptimization:
    """Test measurement pattern handling."""

    def test_full_array_measurement(self) -> None:
        """Test that full array measurements are handled."""
        prog = Main(
            q := QReg("q", 5),
            c := CReg("c", 5),
            # Full array measurement
            Measure(q) > c,
        )

        guppy_code = SlrConverter(prog).guppy()

        # AST codegen measures each qubit individually
        assert "quantum.measure(q[0])" in guppy_code
        assert "quantum.measure(q[1])" in guppy_code
        assert "quantum.measure(q[4])" in guppy_code

    def test_selective_measurements_force_unpacking(self) -> None:
        """Test that selective measurements are handled correctly."""
        prog = Main(
            q := QReg("q", 5),
            c := CReg("c", 5),
            qubit.H(q[0]),
            # Selective measurements with operations between
            Measure(q[0]) > c[0],
            qubit.CX(q[1], q[2]),
            Measure(q[1]) > c[1],
            Measure(q[2]) > c[2],
            Measure(q[3]) > c[3],
            Measure(q[4]) > c[4],
        )

        guppy_code = SlrConverter(prog).guppy()

        # AST codegen uses array indexing
        assert "quantum.h(q[0])" in guppy_code
        assert "c_0 = quantum.measure(q[0])" in guppy_code
        assert "quantum.cx(q[1], q[2])" in guppy_code

    def test_block_all_measurements_together(self) -> None:
        """Test optimization when all measurements are consecutive in a block."""

        class MeasureAll(Block):
            def __init__(self, q: QReg, c: CReg) -> None:
                super().__init__()
                self.q = q
                self.c = c
                self.ops = [
                    Measure(q[0]) > c[0],
                    Measure(q[1]) > c[1],
                    Measure(q[2]) > c[2],
                    Measure(q[3]) > c[3],
                ]

        prog = Main(
            q := QReg("q", 4),
            c := CReg("c", 4),
            qubit.H(q[0]),
            MeasureAll(q, c),
        )

        guppy_code = SlrConverter(prog).guppy()

        # AST codegen flattens blocks and uses array indexing
        assert "quantum.h(q[0])" in guppy_code
        assert "c_0 = quantum.measure(q[0])" in guppy_code
        assert "c_3 = quantum.measure(q[3])" in guppy_code

    def test_non_contiguous_measurements(self) -> None:
        """Test handling of non-contiguous index measurements."""
        prog = Main(
            q := QReg("q", 6),
            c := CReg("c", 3),
            # Measure non-contiguous indices
            Measure(q[0]) > c[0],
            Measure(q[2]) > c[1],
            Measure(q[4]) > c[2],
        )

        guppy_code = SlrConverter(prog).guppy()

        # AST codegen uses array indexing
        assert "c_0 = quantum.measure(q[0])" in guppy_code
        assert "c_1 = quantum.measure(q[2])" in guppy_code
        assert "c_2 = quantum.measure(q[4])" in guppy_code

    def test_measurement_with_conditionals(self) -> None:
        """Test measurements interleaved with conditionals."""
        prog = Main(
            q := QReg("q", 3),
            c := CReg("c", 3),
            Measure(q[0]) > c[0],
            If(c[0]).Then(
                qubit.X(q[1]),
            ),
            Measure(q[1]) > c[1],
            Measure(q[2]) > c[2],
        )

        guppy_code = SlrConverter(prog).guppy()

        # AST codegen uses array indexing for qubits, underscore naming for bits
        assert "c_0 = quantum.measure(q[0])" in guppy_code
        assert "if c_0:" in guppy_code
        assert "quantum.x(q[1])" in guppy_code
        assert "c_1 = quantum.measure(q[1])" in guppy_code
        assert "c_2 = quantum.measure(q[2])" in guppy_code

    def test_multiple_qreg_measurements(self) -> None:
        """Test measurements across multiple quantum registers."""
        prog = Main(
            q1 := QReg("q1", 2),
            q2 := QReg("q2", 2),
            c1 := CReg("c1", 2),
            c2 := CReg("c2", 2),
            # Measure both registers fully
            Measure(q1) > c1,
            Measure(q2) > c2,
        )

        guppy_code = SlrConverter(prog).guppy()

        # AST codegen measures each qubit individually
        assert "c1_0 = quantum.measure(q1[0])" in guppy_code
        assert "c1_1 = quantum.measure(q1[1])" in guppy_code
        assert "c2_0 = quantum.measure(q2[0])" in guppy_code
        assert "c2_1 = quantum.measure(q2[1])" in guppy_code

    def test_partial_then_full_measurement(self) -> None:
        """Test partial measurements followed by full measurement."""

        class MeasureFirst(Block):
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
            MeasureFirst(q, partial),
            # Measure remaining qubits
            Measure(q[2]) > rest[0],
            Measure(q[3]) > rest[1],
        )

        guppy_code = SlrConverter(prog).guppy()

        # AST codegen flattens blocks and measures all
        assert "partial_0 = quantum.measure(q[0])" in guppy_code
        assert "partial_1 = quantum.measure(q[1])" in guppy_code
        assert "rest_0 = quantum.measure(q[2])" in guppy_code
        assert "rest_1 = quantum.measure(q[3])" in guppy_code


class TestMeasurementResultPacking:
    """Test handling of measurement results."""

    def test_pack_individual_results(self) -> None:
        """Test individual measurement results are handled."""
        prog = Main(
            q := QReg("q", 3),
            c := CReg("c", 3),
            # Force individual measurements with operations between
            Measure(q[0]) > c[0],
            qubit.H(q[1]),
            Measure(q[1]) > c[1],
            qubit.H(q[2]),
            Measure(q[2]) > c[2],
        )

        guppy_code = SlrConverter(prog).guppy()

        # AST codegen uses array indexing
        assert "c_0 = quantum.measure(q[0])" in guppy_code
        assert "quantum.h(q[1])" in guppy_code
        assert "c_1 = quantum.measure(q[1])" in guppy_code
        assert "quantum.h(q[2])" in guppy_code
        assert "c_2 = quantum.measure(q[2])" in guppy_code

    def test_no_packing_for_partial_measurements(self) -> None:
        """Test that partial measurements are handled correctly."""
        prog = Main(
            q := QReg("q", 4),
            c := CReg("c", 4),
            # Only measure some qubits
            Measure(q[0]) > c[0],
            Measure(q[1]) > c[1],
            # c[2] and c[3] remain unset
        )

        guppy_code = SlrConverter(prog).guppy()

        # AST codegen uses array indexing
        assert "c_0 = quantum.measure(q[0])" in guppy_code
        assert "c_1 = quantum.measure(q[1])" in guppy_code


class TestComplexPatterns:
    """Test complex measurement patterns from real QEC code."""

    def test_syndrome_extraction_pattern(self) -> None:
        """Test typical syndrome extraction pattern."""

        class ExtractSyndrome(Block):
            def __init__(self, data: QReg, ancilla: QReg, syndrome: CReg) -> None:
                super().__init__()
                self.data = data
                self.ancilla = ancilla
                self.syndrome = syndrome
                self.ops = [
                    # Syndrome extraction circuit
                    qubit.H(ancilla[0]),
                    qubit.CX(data[0], ancilla[0]),
                    qubit.CX(data[1], ancilla[0]),
                    qubit.CX(data[2], ancilla[0]),
                    qubit.H(ancilla[0]),
                    qubit.H(ancilla[1]),
                    qubit.CX(data[3], ancilla[1]),
                    qubit.CX(data[4], ancilla[1]),
                    qubit.CX(data[5], ancilla[1]),
                    qubit.H(ancilla[1]),
                    # Measure ancillas
                    Measure(ancilla) > syndrome,
                ]

        prog = Main(
            data := QReg("data", 7),
            ancilla := QReg("ancilla", 2),
            syndrome := CReg("syndrome", 2),
            ExtractSyndrome(data, ancilla, syndrome),
            # Apply correction based on syndrome
            If(syndrome[0]).Then(
                qubit.X(data[0]),
            ),
            If(syndrome[1]).Then(
                qubit.X(data[3]),
            ),
        )

        guppy_code = SlrConverter(prog).guppy()

        # AST codegen flattens blocks
        assert "quantum.h(ancilla[0])" in guppy_code
        assert "quantum.cx(data[0], ancilla[0])" in guppy_code
        assert "syndrome_0 = quantum.measure(ancilla[0])" in guppy_code
        assert "syndrome_1 = quantum.measure(ancilla[1])" in guppy_code

        # Should have conditionals for corrections
        assert "if syndrome_0:" in guppy_code
        assert "if syndrome_1:" in guppy_code
        assert "quantum.x(data[0])" in guppy_code
        assert "quantum.x(data[3])" in guppy_code
