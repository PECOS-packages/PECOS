"""Tests for measurement optimization in Guppy code generation.

These tests verify that the Guppy generator optimizes measurement patterns:
- Uses measure_array when all qubits measured together
- Detects consecutive individual measurements
- Handles mixed measurement patterns efficiently
"""

from pecos.qeclib import qubit
from pecos.qeclib.qubit.measures import Measure
from pecos.slr import Block, CReg, If, Main, QReg, SlrConverter


class TestMeasurementOptimization:
    """Test measurement pattern optimization."""

    def test_full_array_measurement(self) -> None:
        """Test that full array measurements use measure_array."""
        prog = Main(
            q := QReg("q", 5),
            c := CReg("c", 5),
            # Full array measurement
            Measure(q) > c,
        )

        guppy_code = SlrConverter(prog).guppy()

        # Should use measure_array directly
        assert "c = quantum.measure_array(q)" in guppy_code

        # Should not unpack
        assert "q_0" not in guppy_code

    def test_selective_measurements_force_unpacking(self) -> None:
        """Test that selective measurements force array unpacking."""
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

        # With dynamic allocation, qubits are allocated individually
        # Check that individual qubit variables are used
        assert "q_0" in guppy_code
        assert "q_1" in guppy_code

        # Should use individual qubit variables
        assert (
            "c_0 = quantum.measure(q_0)" in guppy_code
            or "quantum.measure(q_0)" in guppy_code
        )
        assert "quantum.cx(q_1, q_2)" in guppy_code

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

        # Function should generate a block function that measures individually
        assert "measure_all" in guppy_code
        # Measurements use individual qubit variables (q_0, q_1, etc.)
        assert "quantum.measure(q_0)" in guppy_code
        assert "quantum.measure(q_3)" in guppy_code

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

        # Should unpack and measure individually
        assert "q_0, q_1, q_2, q_3, q_4, q_5 = q" in guppy_code
        assert "c_0 = quantum.measure(q_0)" in guppy_code
        assert "c_1 = quantum.measure(q_2)" in guppy_code
        assert "c_2 = quantum.measure(q_4)" in guppy_code

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

        # Should handle measurements individually
        assert "c_0 = quantum.measure(q_0)" in guppy_code
        assert "if c_0:" in guppy_code  # Fixed: using unpacked variable
        assert "quantum.x(q_1)" in guppy_code  # Fixed: using unpacked variable
        assert "c_1 = quantum.measure(q_1)" in guppy_code
        assert "c_2 = quantum.measure(q_2)" in guppy_code

    def test_multiple_qreg_measurements(self) -> None:
        """Test optimizing measurements across multiple quantum registers."""
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

        # Each should use measure_array
        assert "c1 = quantum.measure_array(q1)" in guppy_code
        assert "c2 = quantum.measure_array(q2)" in guppy_code

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

        # Function should be generated and measure first two
        assert "measure_first" in guppy_code
        # With dynamic allocation, may not need unpacking
        assert (
            "q_0, q_1, q_2, q_3 = q" in guppy_code
            or "q_0 = quantum.qubit()" in guppy_code
        )
        # Functions may use array indexing (q[0]) or unpacked vars (q_0)
        assert (
            "partial[0] = quantum.measure(q_0)" in guppy_code
            or "partial_0 = quantum.measure(q_0)" in guppy_code
            or "partial[0] = quantum.measure(q[0])" in guppy_code
        )
        assert (
            "partial[1] = quantum.measure(q_1)" in guppy_code
            or "partial_1 = quantum.measure(q_1)" in guppy_code
            or "partial[1] = quantum.measure(q[1])" in guppy_code
        )

        # Main should handle remaining measurements
        assert (
            "rest[0] = quantum.measure(" in guppy_code
            or "rest_0 = quantum.measure(" in guppy_code
        )
        assert (
            "rest[1] = quantum.measure(" in guppy_code
            or "rest_1 = quantum.measure(" in guppy_code
        )


class TestMeasurementResultPacking:
    """Test packing of individual measurement results into arrays."""

    def test_pack_individual_results(self) -> None:
        """Test packing individual measurement results into CReg."""
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

        # Should create individual variables
        assert "c_0 = quantum.measure(q_0)" in guppy_code
        assert "c_1 = quantum.measure(q_1)" in guppy_code
        assert "c_2 = quantum.measure(q_2)" in guppy_code

        # IR generator unpacks c at the beginning, so no packing needed
        # The unpacked variables are used directly
        assert "c_0, c_1, c_2 = c" in guppy_code

    def test_no_packing_for_partial_measurements(self) -> None:
        """Test that partial measurements don't force packing."""
        prog = Main(
            q := QReg("q", 4),
            c := CReg("c", 4),
            # Only measure some qubits
            Measure(q[0]) > c[0],
            Measure(q[1]) > c[1],
            # c[2] and c[3] remain unset
        )

        guppy_code = SlrConverter(prog).guppy()

        # Should not pack partial results
        assert "# Pack measurement results" not in guppy_code

        # Should use direct assignment
        assert "c_0 = quantum.measure(q_0)" in guppy_code
        assert "c_1 = quantum.measure(q_1)" in guppy_code


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

        # Function should be generated
        assert "extract_syndrome" in guppy_code

        # Should measure ancilla qubits into syndrome
        assert "syndrome" in guppy_code
        assert "quantum.measure" in guppy_code

        # Should have conditionals for corrections
        # With unpacking, uses individual syndrome variables
        assert "if syndrome[0]:" in guppy_code or "if syndrome_0:" in guppy_code
        # Return values from functions may use _ret suffix when unpacked
        assert (
            "quantum.x(data[0])" in guppy_code
            or "quantum.x(data_0)" in guppy_code
            or "quantum.x(data_0_ret)" in guppy_code
        )
