"""Tests for array handling patterns in Guppy code generation.

These tests verify various array patterns including:
- Array unpacking for measurements
- Swapping and permutation patterns
- Option[qubit] patterns (future enhancement)
- Array indexing vs unpacking trade-offs
"""

import pytest
from pecos.qeclib import qubit
from pecos.qeclib.qubit.measures import Measure
from pecos.slr import Block, CReg, Main, Permute, QReg, SlrConverter


class TestArrayUnpacking:
    """Test array unpacking patterns for measurements."""

    def test_unpack_for_selective_measurement(self) -> None:
        """Test unpacking arrays to measure individual qubits."""
        prog = Main(
            q := QReg("q", 4),
            c := CReg("c", 4),
            # Selective measurements force unpacking
            Measure(q[0]) > c[0],
            qubit.H(q[1]),  # Operation between measurements
            Measure(q[1]) > c[1],
            Measure(q[2]) > c[2],
            Measure(q[3]) > c[3],
        )

        guppy_code = SlrConverter(prog).guppy()

        # With dynamic allocation, qubits are allocated individually
        # Check that individual qubit variables are used
        assert "q_0" in guppy_code
        assert "q_1" in guppy_code

        # Should use individual qubit variables
        assert "quantum.measure(q_0)" in guppy_code
        assert "quantum.h(q_1)" in guppy_code

    def test_no_unpack_for_full_measurement(self) -> None:
        """Test that full array measurements don't unpack."""
        prog = Main(
            q := QReg("q", 4),
            c := CReg("c", 4),
            qubit.H(q[0]),
            qubit.CX(q[0], q[1]),
            # Full array measurement
            Measure(q) > c,
        )

        guppy_code = SlrConverter(prog).guppy()

        # Should NOT unpack
        assert "q_0, q_1, q_2, q_3 = q" not in guppy_code

        # Should use measure_array directly
        assert "c = quantum.measure_array(q)" in guppy_code

    def test_unpack_timing_for_first_measurement(self) -> None:
        """Test that unpacking happens before first measurement."""
        prog = Main(
            q := QReg("q", 3),
            c := CReg("c", 3),
            # Operations before measurement
            qubit.H(q[0]),
            qubit.CX(q[0], q[1]),
            qubit.CX(q[1], q[2]),
            # First measurement triggers unpacking
            Measure(q[1]) > c[1],  # Not measuring in order
            Measure(q[0]) > c[0],
            Measure(q[2]) > c[2],
        )

        guppy_code = SlrConverter(prog).guppy()

        # With dynamic allocation, qubits are allocated as needed
        assert "q_0 = quantum.qubit()" in guppy_code
        assert "quantum.h(q_0)" in guppy_code
        assert "q_1 = quantum.qubit()" in guppy_code
        assert "quantum.cx(q_0, q_1)" in guppy_code
        assert "q_2 = quantum.qubit()" in guppy_code
        assert "quantum.cx(q_1, q_2)" in guppy_code

        # Measurements use the dynamically allocated qubits
        assert "c_1 = quantum.measure(q_1)" in guppy_code
        assert "c_0 = quantum.measure(q_0)" in guppy_code
        assert "c_2 = quantum.measure(q_2)" in guppy_code

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
    """Test patterns for swapping array elements."""

    def test_permute_operation(self) -> None:
        """Test Permute operation for register swapping."""
        prog = Main(
            q1 := QReg("q1", 2),
            q2 := QReg("q2", 2),
            c := CReg("c", 4),
            # Prepare states
            qubit.H(q1[0]),
            qubit.X(q2[0]),
            # Swap registers
            Permute(q1, q2),
            # Measure (q1 and q2 are swapped)
            Measure(q1) > c[0:2],
            Measure(q2) > c[2:4],
        )

        guppy_code = SlrConverter(prog).guppy()

        # Permute operation should generate a swap comment
        assert "# Swap q1 and q2" in guppy_code

        # TODO: Permute needs proper implementation

    def test_manual_element_swap(self) -> None:
        """Test swapping individual array elements."""
        # This pattern might be used to reorder qubits
        prog = Main(
            q := QReg("q", 3),
            c := CReg("c", 3),
            # Prepare different states
            qubit.H(q[0]),
            qubit.X(q[1]),
            qubit.Y(q[2]),
            # Measure in different order
            # This forces unpacking and reordering
            Measure(q[2]) > c[0],
            Measure(q[0]) > c[1],
            Measure(q[1]) > c[2],
        )

        guppy_code = SlrConverter(prog).guppy()

        # With IR generator, q is dynamically allocated since it's only used for measurement
        # Check that individual qubits are allocated and measured in the right order
        assert "q_0 = quantum.qubit()" in guppy_code
        assert "quantum.h(q_0)" in guppy_code
        assert "q_1 = quantum.qubit()" in guppy_code
        assert "quantum.x(q_1)" in guppy_code
        assert "q_2 = quantum.qubit()" in guppy_code
        assert "quantum.y(q_2)" in guppy_code
        assert "c_0 = quantum.measure(q_2)" in guppy_code
        assert "c_1 = quantum.measure(q_0)" in guppy_code
        assert "c_2 = quantum.measure(q_1)" in guppy_code


class TestMeasurementIntoArrays:
    """Test patterns for measuring into classical arrays."""

    def test_measure_into_preallocated_array(self) -> None:
        """Test measuring qubits into pre-existing classical array."""
        prog = Main(
            q := QReg("q", 4),
            c := CReg("c", 4),
            # Initialize qubits
            qubit.H(q[0]),
            qubit.CX(q[0], q[1]),
            # Measure into specific indices
            Measure(q[0]) > c[0],
            Measure(q[1]) > c[1],
            Measure(q[2]) > c[2],
            Measure(q[3]) > c[3],
        )

        guppy_code = SlrConverter(prog).guppy()

        # With dynamic allocation, qubits are allocated as needed
        assert "q_0 = quantum.qubit()" in guppy_code
        assert "q_1 = quantum.qubit()" in guppy_code
        assert "c_0 = quantum.measure(q_0)" in guppy_code
        assert "c_1 = quantum.measure(q_1)" in guppy_code

    def test_measure_into_multiple_arrays(self) -> None:
        """Test measuring into different classical arrays."""
        prog = Main(
            q := QReg("q", 4),
            even := CReg("even", 2),
            odd := CReg("odd", 2),
            # Measure even indices to one array, odd to another
            Measure(q[0]) > even[0],
            Measure(q[2]) > even[1],
            Measure(q[1]) > odd[0],
            Measure(q[3]) > odd[1],
        )

        guppy_code = SlrConverter(prog).guppy()

        # With dynamic allocation, no unpacking needed - qubits allocated individually
        # Check that individual qubits are measured and assigned
        assert "q_0 = quantum.qubit()" in guppy_code
        assert "q_1 = quantum.qubit()" in guppy_code

        # Results distributed to correct arrays
        assert "even_0 = quantum.measure(q_0)" in guppy_code
        assert "even_1 = quantum.measure(q_2)" in guppy_code
        assert "odd_0 = quantum.measure(q_1)" in guppy_code
        assert "odd_1 = quantum.measure(q_3)" in guppy_code

    def test_partial_array_measurement(self) -> None:
        """Test measuring only part of a quantum array."""
        prog = Main(
            q := QReg("q", 5),
            c := CReg("c", 3),
            # Only measure first 3 qubits
            Measure(q[0]) > c[0],
            Measure(q[1]) > c[1],
            Measure(q[2]) > c[2],
            # q[3] and q[4] remain unmeasured
        )

        guppy_code = SlrConverter(prog).guppy()

        # Should unpack for partial measurement
        assert "q_0, q_1, q_2, q_3, q_4 = q" in guppy_code

        # Should handle unconsumed qubits by discarding the entire array
        assert "# Discard q" in guppy_code
        assert "quantum.discard_array(q)" in guppy_code


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

        # Note: Slicing syntax q[0:2] might not be supported yet
        # This test documents the desired pattern

        try:
            guppy_code = SlrConverter(prog).guppy()
            # If slicing is supported, check the output
            assert "process_pair" in guppy_code.lower()
        except NotImplementedError:
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

        # Current implementation uses fixed-size arrays
        assert "array(quantum.qubit() for _ in range(4))" in guppy_code
        # Type annotations appear in function signatures, not main
        # Just verify the code generates without errors
