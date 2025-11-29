"""Tests for partial array consumption patterns in Guppy code generation.

These tests verify that the Guppy generator correctly handles QEC patterns where:
- Some qubits are measured while others are preserved
- Functions return unconsumed quantum resources
- Arrays are properly unpacked for individual measurements
"""

import pytest
from pecos.qeclib import qubit
from pecos.qeclib.qubit.measures import Measure
from pecos.slr import Block, CReg, Main, QReg, SlrConverter


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

        # Check function is generated
        assert "measure_ancillas" in guppy_code

        # Function should measure ancillas into syndrome
        assert "syndrome" in guppy_code
        assert "quantum.measure" in guppy_code

        # Ensure data qubits are still available in main
        assert "quantum.x(data[0])" in guppy_code
        assert "data_result = quantum.measure_array(data)" in guppy_code

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

        # Check that function is generated
        assert "measure_first_half" in guppy_code

        # Should measure first half - with unpacking uses individual variables
        assert (
            "c_first[0] = quantum.measure(" in guppy_code
            or "c_first_0 = quantum.measure(" in guppy_code
        )
        assert (
            "c_first[1] = quantum.measure(" in guppy_code
            or "c_first_1 = quantum.measure(" in guppy_code
        )
        assert (
            "c_first[2] = quantum.measure(" in guppy_code
            or "c_first_2 = quantum.measure(" in guppy_code
        )

        # Check main function measures remaining qubits
        # Array access patterns may vary with allocation strategy
        assert "quantum.h(" in guppy_code
        assert (
            "c_second[0] = quantum.measure(" in guppy_code
            or "c_second_0 = quantum.measure(" in guppy_code
        )
        assert (
            "c_second[1] = quantum.measure(" in guppy_code
            or "c_second_1 = quantum.measure(" in guppy_code
        )
        assert (
            "c_second[2] = quantum.measure(" in guppy_code
            or "c_second_2 = quantum.measure(" in guppy_code
        )

    def test_function_returning_quantum_resources(self) -> None:
        """Test functions that return unconsumed quantum resources."""

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

        # Check function is generated
        assert "stabilizer_measurement" in guppy_code
        # Array may be unpacked for element access, then reconstructed for return
        assert "return data" in guppy_code or "return array(data_" in guppy_code

        # Check function call captures returned resources
        # With dynamic allocation, ancilla is constructed as array(ancilla_0)
        assert (
            "data = test_partial_consumption_stabilizer_measurement(ancilla, data, syndrome)"
            in guppy_code
            or "data = test_partial_consumption_stabilizer_measurement(array(ancilla_0), data, syndrome)"
            in guppy_code
        )

        # Should measure ancilla
        assert "syndrome" in guppy_code
        assert "quantum.measure" in guppy_code

    def test_consecutive_measurements_optimization(self) -> None:
        """Test that consecutive individual measurements use measure_array."""
        prog = Main(
            q := QReg("q", 4),
            c := CReg("c", 4),
            # Consecutive measurements that should be optimized
            Measure(q[0]) > c[0],
            Measure(q[1]) > c[1],
            Measure(q[2]) > c[2],
            Measure(q[3]) > c[3],
        )

        guppy_code = SlrConverter(prog).guppy()

        # Should measure individually after unpacking
        assert "c_0 = quantum.measure(q_0)" in guppy_code
        assert "c_1 = quantum.measure(q_1)" in guppy_code
        assert "c_2 = quantum.measure(q_2)" in guppy_code
        assert "c_3 = quantum.measure(q_3)" in guppy_code

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

        # With dynamic allocation, individual qubits are allocated and measured
        # No unpacking needed since they're allocated individually

        # Results distributed to correct destinations
        assert "c1_0 = quantum.measure(q_0)" in guppy_code
        assert "c1_1 = quantum.measure(q_1)" in guppy_code
        assert "c2_0 = quantum.measure(q_2)" in guppy_code
        assert "c2_1 = quantum.measure(q_3)" in guppy_code

    def test_array_unpacking_with_gates(self) -> None:
        """Test that gates work correctly with unpacked arrays."""
        prog = Main(
            q := QReg("q", 3),
            c := CReg("c", 3),
            # Apply gates
            qubit.H(q[0]),
            qubit.CX(q[0], q[1]),
            # Then measure individually (forces unpacking)
            Measure(q[0]) > c[0],
            qubit.X(q[1]),  # Gate between measurements
            Measure(q[1]) > c[1],
            Measure(q[2]) > c[2],
        )

        guppy_code = SlrConverter(prog).guppy()

        # Should either unpack or use local allocation
        # With local allocation: individual qubits created as needed
        # With pre-allocation: array created then unpacked
        has_unpacking = "q_0, q_1, q_2 = q" in guppy_code
        has_local_alloc = "q_0 = quantum.qubit()" in guppy_code

        assert (
            has_unpacking or has_local_alloc
        ), "Should use either unpacking or local allocation"

        # Gates should use unpacked names (q_1)
        assert "quantum.x(q_1)" in guppy_code

        # Measurements use unpacked names
        assert "c_0 = quantum.measure(q_0)" in guppy_code

    def test_single_element_array_unpacking(self) -> None:
        """Test correct unpacking syntax for single-element arrays."""

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

        # Should generate a function
        assert "measure_single" in guppy_code
        # With dynamic allocation, no special unpacking syntax needed
        # Just check that measurement happens
        assert (
            "single_0, = single" in guppy_code  # Pre-allocated with unpacking
            or "single_0 = quantum.qubit()" in guppy_code
        )  # Dynamic allocation
        assert (
            "result_reg[0] = quantum.measure(single_0)" in guppy_code
            or "result[0] = quantum.measure(single_0)" in guppy_code
        )

    @pytest.mark.optional_dependency
    def test_hugr_compilation(self) -> None:
        """Test that partial consumption patterns compile to HUGR."""
        # Simple test that should definitely compile
        prog = Main(
            q := QReg("q", 3),
            c := CReg("c", 3),
            # Individual measurements that will use measure_array optimization
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

    def test_empty_function_body(self) -> None:
        """Test function with no operations."""

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

        # Should have pass in empty function
        # Note: function names are prefixed with module name
        assert "def test_partial_consumption_do_nothing" in guppy_code
        assert "pass" in guppy_code

    def test_no_measurements(self) -> None:
        """Test handling of unconsumed qubits at end of main."""
        prog = Main(
            q := QReg("q", 2),
            # Apply gates but don't measure
            qubit.H(q[0]),
            qubit.CX(q[0], q[1]),
        )

        guppy_code = SlrConverter(prog).guppy()

        # Should automatically discard unconsumed qubits
        assert "# Discard q" in guppy_code
        assert "quantum.discard_array(q)" in guppy_code
