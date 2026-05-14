"""Tests for partial array consumption patterns in Guppy code generation.

Note: The AST codegen flattens Block subclasses into the main function
rather than generating nested functions. These tests verify that blocks
are correctly flattened and operations are properly sequenced.
"""

import pytest
from pecos.slr import Block, CReg, Main, QReg, SlrConverter
from pecos.slr.qeclib import qubit
from pecos.slr.qeclib.qubit.measures import Measure


class TestPartialConsumption:
    """Test cases for partial quantum array consumption."""

    def test_measure_ancillas_preserve_data(self) -> None:
        """Test measuring ancilla qubits while preserving data qubits."""

        class MeasureAncillas(Block):
            """Measure ancilla qubits but keep data qubits."""

            def __init__(self, data: QReg, ancilla: QReg, syndrome: CReg) -> None:
                super().__init__()
                self.data = data
                self.ancilla = ancilla
                self.syndrome = syndrome
                self.ops = [
                    # Measure all ancillas
                    Measure(ancilla[0]) > syndrome[0],
                    Measure(ancilla[1]) > syndrome[1],
                    Measure(ancilla[2]) > syndrome[2],
                    Measure(ancilla[3]) > syndrome[3],
                    # Data qubits remain unmeasured
                ]

        prog = Main(
            data := QReg("data", 7),
            ancilla := QReg("ancilla", 4),
            syndrome := CReg("syndrome", 4),
            data_result := CReg("data_result", 7),
            # Prepare some state
            qubit.H(data[0]),
            qubit.CX(data[0], ancilla[0]),
            # Measure ancillas but keep data
            MeasureAncillas(data, ancilla, syndrome),
            # Continue using data
            qubit.X(data[0]),
            # Eventually measure data
            Measure(data) > data_result,
        )

        guppy_code = SlrConverter(prog).guppy()

        # AST codegen flattens blocks
        # Check ancilla measurements
        assert "syndrome_0 = quantum.measure(ancilla[0])" in guppy_code
        assert "syndrome_1 = quantum.measure(ancilla[1])" in guppy_code
        assert "syndrome_2 = quantum.measure(ancilla[2])" in guppy_code
        assert "syndrome_3 = quantum.measure(ancilla[3])" in guppy_code

        # Check that data operations are present
        assert "quantum.x(data[0])" in guppy_code

        # Check data measurements
        assert "data_result_0 = quantum.measure(data[0])" in guppy_code

    def test_consume_subset_of_qubits(self) -> None:
        """Test consuming only part of a qubit array."""

        class MeasureFirstHalf(Block):
            """Measure first half of a qubit array."""

            def __init__(self, qubits: QReg, results: CReg) -> None:
                super().__init__()
                self.qubits = qubits
                self.results = results
                self.ops = [
                    Measure(qubits[0]) > results[0],
                    Measure(qubits[1]) > results[1],
                    Measure(qubits[2]) > results[2],
                    # qubits[3], [4], [5] remain unmeasured
                ]

        prog = Main(
            q := QReg("q", 6),
            c_first := CReg("c_first", 3),
            c_second := CReg("c_second", 3),
            # Prepare
            qubit.H(q[0]),
            qubit.CX(q[0], q[1]),
            # Measure first half
            MeasureFirstHalf(q, c_first),
            # Continue with second half
            qubit.H(q[3]),
            Measure(q[3]) > c_second[0],
            Measure(q[4]) > c_second[1],
            Measure(q[5]) > c_second[2],
        )

        guppy_code = SlrConverter(prog).guppy()

        # AST codegen flattens blocks
        # Check first half measurements
        assert "c_first_0 = quantum.measure(q[0])" in guppy_code
        assert "c_first_1 = quantum.measure(q[1])" in guppy_code
        assert "c_first_2 = quantum.measure(q[2])" in guppy_code

        # Check operations on second half
        assert "quantum.h(q[3])" in guppy_code
        assert "c_second_0 = quantum.measure(q[3])" in guppy_code
        assert "c_second_1 = quantum.measure(q[4])" in guppy_code
        assert "c_second_2 = quantum.measure(q[5])" in guppy_code

    def test_block_operations_flattened(self) -> None:
        """Test that block operations are flattened into main."""

        class StabilizerMeasurement(Block):
            """Measure stabilizer, return data qubits."""

            def __init__(self, data: QReg, ancilla: QReg, syndrome: CReg) -> None:
                super().__init__()
                self.data = data
                self.ancilla = ancilla
                self.syndrome = syndrome
                self.ops = [
                    # Stabilizer circuit
                    qubit.H(ancilla[0]),
                    qubit.CX(data[0], ancilla[0]),
                    qubit.CX(data[1], ancilla[0]),
                    qubit.H(ancilla[0]),
                    # Measure ancilla to get syndrome
                    Measure(ancilla[0]) > syndrome[0],
                ]

        prog = Main(
            data := QReg("data", 2),
            ancilla := QReg("ancilla", 1),
            syndrome := CReg("syndrome", 1),
            final := CReg("final", 2),
            # Run stabilizer measurement
            StabilizerMeasurement(data, ancilla, syndrome),
            # Continue with data
            qubit.Z(data[0]),
            # Final measurements
            Measure(data) > final,
        )

        guppy_code = SlrConverter(prog).guppy()

        # AST codegen flattens blocks
        # Check stabilizer operations
        assert "quantum.h(ancilla[0])" in guppy_code
        assert "quantum.cx(data[0], ancilla[0])" in guppy_code
        assert "quantum.cx(data[1], ancilla[0])" in guppy_code
        assert "syndrome_0 = quantum.measure(ancilla[0])" in guppy_code

        # Check operations after block
        assert "quantum.z(data[0])" in guppy_code
        assert "final_0 = quantum.measure(data[0])" in guppy_code
        assert "final_1 = quantum.measure(data[1])" in guppy_code

    def test_consecutive_measurements(self) -> None:
        """Test that consecutive measurements use array indexing."""
        prog = Main(
            q := QReg("q", 4),
            c := CReg("c", 4),
            # Consecutive measurements
            Measure(q[0]) > c[0],
            Measure(q[1]) > c[1],
            Measure(q[2]) > c[2],
            Measure(q[3]) > c[3],
        )

        guppy_code = SlrConverter(prog).guppy()

        # AST codegen uses array indexing
        assert "c_0 = quantum.measure(q[0])" in guppy_code
        assert "c_1 = quantum.measure(q[1])" in guppy_code
        assert "c_2 = quantum.measure(q[2])" in guppy_code
        assert "c_3 = quantum.measure(q[3])" in guppy_code

    def test_mixed_destination_measurements(self) -> None:
        """Test measurements to different classical registers."""
        prog = Main(
            q := QReg("q", 4),
            c1 := CReg("c1", 2),
            c2 := CReg("c2", 2),
            # Measurements to different registers
            Measure(q[0]) > c1[0],
            Measure(q[1]) > c1[1],
            Measure(q[2]) > c2[0],
            Measure(q[3]) > c2[1],
        )

        guppy_code = SlrConverter(prog).guppy()

        # Results distributed to correct destinations
        assert "c1_0 = quantum.measure(q[0])" in guppy_code
        assert "c1_1 = quantum.measure(q[1])" in guppy_code
        assert "c2_0 = quantum.measure(q[2])" in guppy_code
        assert "c2_1 = quantum.measure(q[3])" in guppy_code

    def test_gates_with_array_indexing(self) -> None:
        """Test that gates work correctly with array indexing."""
        prog = Main(
            q := QReg("q", 3),
            c := CReg("c", 3),
            # Apply gates
            qubit.H(q[0]),
            qubit.CX(q[0], q[1]),
            # Then measure
            Measure(q[0]) > c[0],
            qubit.X(q[1]),  # Gate between measurements
            Measure(q[1]) > c[1],
            Measure(q[2]) > c[2],
        )

        guppy_code = SlrConverter(prog).guppy()

        # AST codegen uses array indexing
        assert "quantum.h(q[0])" in guppy_code
        assert "quantum.cx(q[0], q[1])" in guppy_code
        assert "c_0 = quantum.measure(q[0])" in guppy_code
        assert "quantum.x(q[1])" in guppy_code
        assert "c_1 = quantum.measure(q[1])" in guppy_code
        assert "c_2 = quantum.measure(q[2])" in guppy_code

    def test_single_element_block(self) -> None:
        """Test block with single operation."""

        class MeasureSingle(Block):
            def __init__(self, q: QReg, c: CReg) -> None:
                super().__init__()
                self.q = q
                self.c = c
                self.ops = [
                    Measure(q[0]) > c[0],
                ]

        prog = Main(
            single := QReg("single", 1),
            result := CReg("result", 1),
            MeasureSingle(single, result),
        )

        guppy_code = SlrConverter(prog).guppy()

        # AST codegen flattens blocks
        assert "result_0 = quantum.measure(single[0])" in guppy_code

    @pytest.mark.optional_dependency
    def test_hugr_compilation(self) -> None:
        """Test that patterns compile to HUGR."""
        prog = Main(
            q := QReg("q", 3),
            c := CReg("c", 3),
            # Individual measurements
            Measure(q[0]) > c[0],
            Measure(q[1]) > c[1],
            Measure(q[2]) > c[2],
        )

        # This should compile without errors
        try:
            hugr = SlrConverter(prog).hugr()
            assert hugr is not None
        except ImportError as e:
            pytest.fail(f"HUGR compilation failed: {e}")


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_block(self) -> None:
        """Test block with no operations."""

        class DoNothing(Block):
            def __init__(self, q: QReg) -> None:
                super().__init__()
                self.q = q
                self.ops = []

        prog = Main(
            q := QReg("q", 2),
            c := CReg("c", 2),
            DoNothing(q),
            Measure(q) > c,
        )

        guppy_code = SlrConverter(prog).guppy()

        # Empty blocks are just skipped, measurements should work
        assert "c_0 = quantum.measure(q[0])" in guppy_code
        assert "c_1 = quantum.measure(q[1])" in guppy_code

    def test_unmeasured_qubits_returned(self) -> None:
        """Test handling of unconsumed qubits in main return type."""
        prog = Main(
            q := QReg("q", 2),
            # Apply gates but don't measure
            qubit.H(q[0]),
            qubit.CX(q[0], q[1]),
        )

        guppy_code = SlrConverter(prog).guppy()

        # Main should have qubit operations
        assert "quantum.h(q[0])" in guppy_code
        assert "quantum.cx(q[0], q[1])" in guppy_code

        # Return type should include qubit array since they're not consumed
        assert "array[qubit, 2]" in guppy_code
