"""Tests for Array multi-dimensional non-unit step slicing functionality.

This module tests Array's support for non-unit step slicing operations
on multi-dimensional arrays (2D, 3D, etc.) against NumPy to ensure correct
drop-in replacement behavior.
"""

import numpy as np

from pecos_rslib import Array


class TestNonUnitStep2D:
    """Test non-unit step slicing for 2D arrays."""

    def test_step_on_first_dimension(self) -> None:
        """Test arr[::2, :] - step on first dimension."""
        np_arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])
        pa_arr = Array(np_arr.copy())

        # Test assignment
        pa_arr[::2, :] = 99.0
        np_arr[::2, :] = 99.0

        # Verify results match
        np.testing.assert_array_equal(np.asarray(pa_arr), np_arr)

    def test_step_on_second_dimension(self) -> None:
        """Test arr[:, ::2] - step on second dimension."""
        np_arr = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]])
        pa_arr = Array(np_arr.copy())

        # Test assignment
        pa_arr[:, ::2] = 88.0
        np_arr[:, ::2] = 88.0

        # Verify results match
        np.testing.assert_array_equal(np.asarray(pa_arr), np_arr)

    def test_step_on_both_dimensions(self) -> None:
        """Test arr[::2, ::2] - step on both dimensions."""
        np_arr = np.array(
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
                [13.0, 14.0, 15.0, 16.0],
            ]
        )
        pa_arr = Array(np_arr.copy())

        # Test assignment
        pa_arr[::2, ::2] = 77.0
        np_arr[::2, ::2] = 77.0

        # Verify results match
        np.testing.assert_array_equal(np.asarray(pa_arr), np_arr)

    def test_reverse_first_dimension(self) -> None:
        """Test arr[::-1, :] - reverse first dimension."""
        np_arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        pa_arr = Array(np_arr.copy())

        # Test assignment
        pa_arr[::-1, :] = 11.0
        np_arr[::-1, :] = 11.0

        # Verify results match
        np.testing.assert_array_equal(np.asarray(pa_arr), np_arr)

    def test_reverse_second_dimension(self) -> None:
        """Test arr[:, ::-1] - reverse second dimension."""
        np_arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        pa_arr = Array(np_arr.copy())

        # Test assignment
        pa_arr[:, ::-1] = 22.0
        np_arr[:, ::-1] = 22.0

        # Verify results match
        np.testing.assert_array_equal(np.asarray(pa_arr), np_arr)

    def test_step_with_start_and_stop(self) -> None:
        """Test arr[0:3:2, 1:4:2] - step with start and stop on both dimensions."""
        np_arr = np.arange(20).reshape(4, 5).astype(np.float64)
        pa_arr = Array(np_arr.copy())

        # Test assignment
        pa_arr[0:3:2, 1:4:2] = 555.0
        np_arr[0:3:2, 1:4:2] = 555.0

        # Verify results match
        np.testing.assert_array_equal(np.asarray(pa_arr), np_arr)


class TestNonUnitStep2DArrayAssignment:
    """Test array assignment with non-unit step slicing for 2D arrays."""

    def test_array_assignment_with_step(self) -> None:
        """Test assigning an array to a 2D non-unit step slice."""
        np_arr = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]])
        pa_arr = Array(np_arr.copy())

        assignment_arr = np.array([[100.0, 200.0], [300.0, 400.0], [500.0, 600.0]])

        # Test array assignment
        pa_arr[:, ::2] = assignment_arr
        np_arr[:, ::2] = assignment_arr

        # Verify results match
        np.testing.assert_array_equal(np.asarray(pa_arr), np_arr)

    def test_array_assignment_both_dimensions_step(self) -> None:
        """Test assigning an array to a 2D slice with steps on both dimensions."""
        np_arr = np.array(
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
                [13.0, 14.0, 15.0, 16.0],
            ]
        )
        pa_arr = Array(np_arr.copy())

        assignment_arr = np.array([[100.0, 200.0], [300.0, 400.0]])

        # Test array assignment
        pa_arr[::2, ::2] = assignment_arr
        np_arr[::2, ::2] = assignment_arr

        # Verify results match
        np.testing.assert_array_equal(np.asarray(pa_arr), np_arr)


class TestNonUnitStep3D:
    """Test non-unit step slicing for 3D arrays."""

    def test_step_on_first_dimension(self) -> None:
        """Test arr[::2, :, :] - step on first dimension."""
        np_arr = np.arange(24).reshape(4, 3, 2).astype(np.float64)
        pa_arr = Array(np_arr.copy())

        # Test assignment
        pa_arr[::2, :, :] = 99.0
        np_arr[::2, :, :] = 99.0

        # Verify results match
        np.testing.assert_array_equal(np.asarray(pa_arr), np_arr)

    def test_step_on_all_dimensions(self) -> None:
        """Test arr[::2, ::2, ::2] - step on all dimensions."""
        np_arr = np.arange(64).reshape(4, 4, 4).astype(np.float64)
        pa_arr = Array(np_arr.copy())

        # Test assignment
        pa_arr[::2, ::2, ::2] = 88.0
        np_arr[::2, ::2, ::2] = 88.0

        # Verify results match
        np.testing.assert_array_equal(np.asarray(pa_arr), np_arr)
        # Should affect 8 elements (2x2x2 subset)
        assert np.sum(np.asarray(pa_arr) == 88.0) == 8

    def test_reverse_first_dimension(self) -> None:
        """Test arr[::-1, :, :] - reverse first dimension."""
        np_arr = np.arange(8).reshape(2, 2, 2).astype(np.float64)
        pa_arr = Array(np_arr.copy())

        # Test assignment
        pa_arr[::-1, :, :] = 22.0
        np_arr[::-1, :, :] = 22.0

        # Verify results match
        np.testing.assert_array_equal(np.asarray(pa_arr), np_arr)

    def test_step_on_second_dimension(self) -> None:
        """Test arr[:, ::2, :] - step on second dimension."""
        np_arr = np.arange(24).reshape(2, 6, 2).astype(np.float64)
        pa_arr = Array(np_arr.copy())

        # Test assignment
        pa_arr[:, ::2, :] = 33.0
        np_arr[:, ::2, :] = 33.0

        # Verify results match
        np.testing.assert_array_equal(np.asarray(pa_arr), np_arr)

    def test_step_on_third_dimension(self) -> None:
        """Test arr[:, :, ::2] - step on third dimension."""
        np_arr = np.arange(24).reshape(2, 3, 4).astype(np.float64)
        pa_arr = Array(np_arr.copy())

        # Test assignment
        pa_arr[:, :, ::2] = 44.0
        np_arr[:, :, ::2] = 44.0

        # Verify results match
        np.testing.assert_array_equal(np.asarray(pa_arr), np_arr)


class TestNonUnitStepDifferentDtypes:
    """Test non-unit step slicing with different data types on multi-dimensional arrays."""

    def test_int64_2d_non_unit_step(self) -> None:
        """Test non-unit step slicing with int64 2D array."""
        np_arr = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]], dtype=np.int64)
        pa_arr = Array(np_arr.copy())

        # Test assignment
        pa_arr[::2, :] = 99
        np_arr[::2, :] = 99

        # Verify results match
        np.testing.assert_array_equal(np.asarray(pa_arr), np_arr)
        assert np.asarray(pa_arr).dtype == np.int64

    def test_complex128_2d_non_unit_step(self) -> None:
        """Test non-unit step slicing with complex128 2D array."""
        np_arr = np.array([[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j], [9 + 10j, 11 + 12j]])
        pa_arr = Array(np_arr.copy())

        # Test assignment
        pa_arr[::2, :] = 100 + 200j
        np_arr[::2, :] = 100 + 200j

        # Verify results match
        np.testing.assert_array_equal(np.asarray(pa_arr), np_arr)
        assert np.asarray(pa_arr).dtype == np.complex128

    # Note: Float32 not yet implemented in N-dimensional non-unit step slicing
    # def test_float32_2d_non_unit_step(self):
    #     """Test non-unit step slicing with float32 2D array."""
    #     np_arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    #     pa_arr = PecosArray(np_arr.copy())
    #
    #     # Test assignment
    #     pa_arr[:, ::2] = 99.0
    #     np_arr[:, ::2] = 99.0
    #
    #     # Verify results match
    #     np.testing.assert_array_equal(np.asarray(pa_arr), np_arr)
    #     assert np.asarray(pa_arr).dtype == np.float32


class TestNonUnitStepEdgeCases:
    """Test edge cases for multi-dimensional non-unit step slicing."""

    def test_step_larger_than_dimension(self) -> None:
        """Test edge case - step larger than array dimension."""
        np_arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        pa_arr = Array(np_arr.copy())

        # Test assignment (only affects first row)
        pa_arr[::10, :] = 555.0
        np_arr[::10, :] = 555.0

        # Verify results match
        np.testing.assert_array_equal(np.asarray(pa_arr), np_arr)

    def test_empty_slice_result(self) -> None:
        """Test when slice produces empty result."""
        np_arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        pa_arr = Array(np_arr.copy())

        # Test assignment to empty slice (should do nothing)
        pa_arr[5:10:2, :] = 99.0
        np_arr[5:10:2, :] = 99.0

        # Verify results match (should be unchanged)
        np.testing.assert_array_equal(np.asarray(pa_arr), np_arr)

    def test_negative_indices_with_step(self) -> None:
        """Test negative indices combined with non-unit step."""
        np_arr = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]])
        pa_arr = Array(np_arr.copy())

        # Test assignment with negative start
        pa_arr[-2:, ::2] = 77.0
        np_arr[-2:, ::2] = 77.0

        # Verify results match
        np.testing.assert_array_equal(np.asarray(pa_arr), np_arr)


class TestNonUnitStepReproducibility:
    """Test that multi-dimensional non-unit step operations are reproducible and consistent."""

    def test_multiple_operations_2d(self) -> None:
        """Test multiple non-unit step operations on same 2D array."""
        np_arr = np.arange(24).reshape(4, 6).astype(np.float64)
        pa_arr = Array(np_arr.copy())

        # First operation
        pa_arr[::2, :] = 10.0
        np_arr[::2, :] = 10.0
        np.testing.assert_array_equal(np.asarray(pa_arr), np_arr)

        # Second operation
        pa_arr[:, ::3] = 20.0
        np_arr[:, ::3] = 20.0
        np.testing.assert_array_equal(np.asarray(pa_arr), np_arr)

        # Third operation (reverse)
        pa_arr[::-1, :] = 30.0
        np_arr[::-1, :] = 30.0
        np.testing.assert_array_equal(np.asarray(pa_arr), np_arr)

    def test_unit_step_still_works_after_nonunit(self) -> None:
        """Verify that unit-step slicing still works after non-unit implementation."""
        np_arr = np.arange(20).reshape(4, 5).astype(np.float64)
        pa_arr = Array(np_arr.copy())

        # Non-unit step operation
        pa_arr[::2, :] = 10.0
        np_arr[::2, :] = 10.0
        np.testing.assert_array_equal(np.asarray(pa_arr), np_arr)

        # Unit-step operation (should use optimized path)
        pa_arr[1:3, 1:4] = 99.0
        np_arr[1:3, 1:4] = 99.0
        np.testing.assert_array_equal(np.asarray(pa_arr), np_arr)

    def test_conversion_to_numpy_preserves_values(self) -> None:
        """Test that conversion to NumPy preserves values after non-unit step operations."""
        np_arr = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]])
        pa_arr = Array(np_arr.copy())

        # Perform operation
        pa_arr[::2, ::2] = 100.0
        np_arr[::2, ::2] = 100.0

        # Convert to NumPy and verify
        result = np.asarray(pa_arr)
        np.testing.assert_array_equal(result, np_arr)
        assert result.dtype == np_arr.dtype


class TestNonUnitStepCombinations:
    """Test combinations of unit and non-unit step slicing across different dimensions."""

    def test_unit_step_first_nonunit_second(self) -> None:
        """Test arr[1:3, ::2] - unit step on first, non-unit on second dimension."""
        np_arr = np.arange(24).reshape(4, 6).astype(np.float64)
        pa_arr = Array(np_arr.copy())

        # Test assignment
        pa_arr[1:3, ::2] = 77.0
        np_arr[1:3, ::2] = 77.0

        # Verify results match
        np.testing.assert_array_equal(np.asarray(pa_arr), np_arr)

    def test_nonunit_step_first_unit_second(self) -> None:
        """Test arr[::2, 1:5] - non-unit step on first, unit on second dimension."""
        np_arr = np.arange(24).reshape(4, 6).astype(np.float64)
        pa_arr = Array(np_arr.copy())

        # Test assignment
        pa_arr[::2, 1:5] = 88.0
        np_arr[::2, 1:5] = 88.0

        # Verify results match
        np.testing.assert_array_equal(np.asarray(pa_arr), np_arr)

    def test_3d_mixed_steps(self) -> None:
        """Test 3D array with mix of unit and non-unit steps."""
        np_arr = np.arange(48).reshape(4, 4, 3).astype(np.float64)
        pa_arr = Array(np_arr.copy())

        # Test assignment: unit on first, non-unit on second and third
        pa_arr[1:3, ::2, ::2] = 99.0
        np_arr[1:3, ::2, ::2] = 99.0

        # Verify results match
        np.testing.assert_array_equal(np.asarray(pa_arr), np_arr)
