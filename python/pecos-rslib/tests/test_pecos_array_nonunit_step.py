"""Tests for Array non-unit step slicing functionality.

This module tests Array's support for non-unit step slicing operations
(e.g., arr[::2], arr[::-1], arr[1:10:3]) against NumPy to ensure correct
drop-in replacement behavior.
"""

import numpy as np

from pecos_rslib import Array


class TestNonUnitStepSlicing1D:
    """Test non-unit step slicing for 1D arrays."""

    def test_every_other_element(self):
        """Test arr[::2] - every other element."""
        np_arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        pa_arr = Array(np_arr.copy())

        # Test assignment
        pa_arr[::2] = 99.0
        np_arr[::2] = 99.0

        # Verify results match
        np.testing.assert_array_equal(np.asarray(pa_arr), np_arr)

    def test_every_other_element_with_start(self):
        """Test arr[1::2] - every other element starting at index 1."""
        np_arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        pa_arr = Array(np_arr.copy())

        # Test assignment
        pa_arr[1::2] = 88.0
        np_arr[1::2] = 88.0

        # Verify results match
        np.testing.assert_array_equal(np.asarray(pa_arr), np_arr)

    def test_step_with_start_and_stop(self):
        """Test arr[1:10:3] - step by 3 from index 1 to 10."""
        np_arr = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        pa_arr = Array(np_arr.copy())

        # Test assignment
        pa_arr[1:10:3] = 77.0
        np_arr[1:10:3] = 77.0

        # Verify results match
        np.testing.assert_array_equal(np.asarray(pa_arr), np_arr)

    def test_reverse_order(self):
        """Test arr[::-1] - reverse order."""
        np_arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        pa_arr = Array(np_arr.copy())

        # Test assignment
        pa_arr[::-1] = 11.0
        np_arr[::-1] = 11.0

        # Verify results match
        np.testing.assert_array_equal(np.asarray(pa_arr), np_arr)

    def test_every_other_element_reverse(self):
        """Test arr[::-2] - every other element in reverse."""
        np_arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        pa_arr = Array(np_arr.copy())

        # Test assignment
        pa_arr[::-2] = 22.0
        np_arr[::-2] = 22.0

        # Verify results match
        np.testing.assert_array_equal(np.asarray(pa_arr), np_arr)

    def test_negative_step_with_explicit_bounds(self):
        """Test arr[10:0:-2] - reverse with step -2 from 10 to 0."""
        np_arr = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        pa_arr = Array(np_arr.copy())

        # Test assignment
        pa_arr[10:0:-2] = 33.0
        np_arr[10:0:-2] = 33.0

        # Verify results match
        np.testing.assert_array_equal(np.asarray(pa_arr), np_arr)


class TestNonUnitStepArrayAssignment:
    """Test array assignment with non-unit step slicing."""

    def test_array_assignment_with_step(self):
        """Test assigning an array to a non-unit step slice."""
        np_arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        pa_arr = Array(np_arr.copy())

        assignment_arr = np.array([100.0, 200.0, 300.0])

        # Test array assignment
        pa_arr[::2] = assignment_arr
        np_arr[::2] = assignment_arr

        # Verify results match
        np.testing.assert_array_equal(np.asarray(pa_arr), np_arr)

    def test_array_assignment_reverse_step(self):
        """Test assigning an array to a reverse non-unit step slice."""
        np_arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        pa_arr = Array(np_arr.copy())

        assignment_arr = np.array([100.0, 200.0, 300.0])

        # Test array assignment with reverse step
        pa_arr[::-2] = assignment_arr
        np_arr[::-2] = assignment_arr

        # Verify results match
        np.testing.assert_array_equal(np.asarray(pa_arr), np_arr)


class TestNonUnitStepDifferentDtypes:
    """Test non-unit step slicing with different data types."""

    def test_int64_non_unit_step(self):
        """Test non-unit step slicing with int64 array."""
        np_arr = np.array([10, 20, 30, 40, 50, 60])
        pa_arr = Array(np_arr.copy())

        # Test assignment
        pa_arr[1::2] = 99
        np_arr[1::2] = 99

        # Verify results match
        np.testing.assert_array_equal(np.asarray(pa_arr), np_arr)
        assert np.asarray(pa_arr).dtype == np.int64

    def test_complex128_non_unit_step(self):
        """Test non-unit step slicing with complex128 array."""
        np_arr = np.array([1 + 2j, 3 + 4j, 5 + 6j, 7 + 8j, 9 + 10j, 11 + 12j])
        pa_arr = Array(np_arr.copy())

        # Test assignment
        pa_arr[::2] = 100 + 200j
        np_arr[::2] = 100 + 200j

        # Verify results match
        np.testing.assert_array_equal(np.asarray(pa_arr), np_arr)
        assert np.asarray(pa_arr).dtype == np.complex128

    def test_int32_non_unit_step(self):
        """Test non-unit step slicing with int32 array."""
        np_arr = np.array([10, 20, 30, 40, 50, 60], dtype=np.int32)
        pa_arr = Array(np_arr.copy())

        # Test assignment
        pa_arr[::2] = 99
        np_arr[::2] = 99

        # Verify results match
        np.testing.assert_array_equal(np.asarray(pa_arr), np_arr)
        assert np.asarray(pa_arr).dtype == np.int32

    def test_float32_non_unit_step(self):
        """Test non-unit step slicing with float32 array."""
        np_arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
        pa_arr = Array(np_arr.copy())

        # Test assignment
        pa_arr[1::2] = 88.0
        np_arr[1::2] = 88.0

        # Verify results match
        np.testing.assert_array_equal(np.asarray(pa_arr), np_arr)
        assert np.asarray(pa_arr).dtype == np.float32


class TestNonUnitStepEdgeCases:
    """Test edge cases for non-unit step slicing."""

    def test_step_larger_than_array(self):
        """Test edge case - step larger than array size."""
        np_arr = np.array([1.0, 2.0, 3.0])
        pa_arr = Array(np_arr.copy())

        # Test assignment (only affects index 0)
        pa_arr[::10] = 555.0
        np_arr[::10] = 555.0

        # Verify results match
        np.testing.assert_array_equal(np.asarray(pa_arr), np_arr)

    def test_empty_slice_result(self):
        """Test when slice produces empty result."""
        np_arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        pa_arr = Array(np_arr.copy())

        # Test assignment to empty slice (should do nothing)
        pa_arr[5:10:2] = 99.0
        np_arr[5:10:2] = 99.0

        # Verify results match (should be unchanged)
        np.testing.assert_array_equal(np.asarray(pa_arr), np_arr)

    def test_single_element_step(self):
        """Test when step results in single element."""
        np_arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        pa_arr = Array(np_arr.copy())

        # Test assignment to single-element slice
        pa_arr[0:1:5] = 999.0
        np_arr[0:1:5] = 999.0

        # Verify results match
        np.testing.assert_array_equal(np.asarray(pa_arr), np_arr)

    def test_negative_indices_with_step(self):
        """Test negative indices combined with non-unit step."""
        np_arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        pa_arr = Array(np_arr.copy())

        # Test assignment with negative start
        pa_arr[-4::2] = 77.0
        np_arr[-4::2] = 77.0

        # Verify results match
        np.testing.assert_array_equal(np.asarray(pa_arr), np_arr)


class TestNonUnitStepReproducibility:
    """Test that non-unit step operations are reproducible and consistent."""

    def test_multiple_operations(self):
        """Test multiple non-unit step operations on same array."""
        np_arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        pa_arr = Array(np_arr.copy())

        # First operation
        pa_arr[::2] = 10.0
        np_arr[::2] = 10.0
        np.testing.assert_array_equal(np.asarray(pa_arr), np_arr)

        # Second operation
        pa_arr[1::3] = 20.0
        np_arr[1::3] = 20.0
        np.testing.assert_array_equal(np.asarray(pa_arr), np_arr)

        # Third operation (reverse)
        pa_arr[::-2] = 30.0
        np_arr[::-2] = 30.0
        np.testing.assert_array_equal(np.asarray(pa_arr), np_arr)

    def test_unit_step_still_works(self):
        """Verify that unit-step slicing still works after non-unit implementation."""
        np_arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        pa_arr = Array(np_arr.copy())

        # Test unit-step assignment (should use optimized path)
        pa_arr[1:4:1] = 99.0
        np_arr[1:4:1] = 99.0

        # Verify results match
        np.testing.assert_array_equal(np.asarray(pa_arr), np_arr)

    def test_conversion_to_numpy_preserves_values(self):
        """Test that conversion to NumPy preserves values after non-unit step operations."""
        np_arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        pa_arr = Array(np_arr.copy())

        # Perform operation
        pa_arr[::2] = 100.0
        np_arr[::2] = 100.0

        # Convert to NumPy and verify
        result = np.asarray(pa_arr)
        np.testing.assert_array_equal(result, np_arr)
        assert result.dtype == np_arr.dtype


class TestNonUnitStepWithUnitStep:
    """Test interaction between non-unit step and unit-step slicing."""

    def test_alternating_unit_nonunit_steps(self):
        """Test alternating between unit and non-unit step operations."""
        np_arr = np.arange(20, dtype=np.float64)
        pa_arr = Array(np_arr.copy())

        # Unit step
        pa_arr[0:5] = 1.0
        np_arr[0:5] = 1.0
        np.testing.assert_array_equal(np.asarray(pa_arr), np_arr)

        # Non-unit step
        pa_arr[5::3] = 2.0
        np_arr[5::3] = 2.0
        np.testing.assert_array_equal(np.asarray(pa_arr), np_arr)

        # Unit step
        pa_arr[10:15] = 3.0
        np_arr[10:15] = 3.0
        np.testing.assert_array_equal(np.asarray(pa_arr), np_arr)

        # Negative non-unit step
        pa_arr[::-2] = 4.0
        np_arr[::-2] = 4.0
        np.testing.assert_array_equal(np.asarray(pa_arr), np_arr)
