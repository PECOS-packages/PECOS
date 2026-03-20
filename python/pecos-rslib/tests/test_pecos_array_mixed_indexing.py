"""Tests for Array mixed integer/slice indexing functionality.

This module tests Array's support for mixed integer/slice indexing operations
(e.g., arr[0, 1:3], arr[:, 0], arr[1:3, 0, :]) against NumPy to ensure correct
drop-in replacement behavior.
"""

import numpy as np
import pytest

from pecos_rslib import Array


class TestMixedIndexing2D:
    """Test mixed integer/slice indexing for 2D arrays."""

    def test_integer_first_slice_second(self) -> None:
        """Test arr[0, 1:3] - integer first, slice second."""
        np_arr = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]])
        pa_arr = Array(np_arr.copy())

        # Test mixed indexing
        result = pa_arr[0, 1:3]
        expected = np_arr[0, 1:3]
        result_np = np.asarray(result)

        # Verify shape and values
        assert result.shape == expected.shape, f"Shape mismatch: {result.shape} vs {expected.shape}"
        np.testing.assert_array_equal(result_np, expected)

    def test_slice_first_integer_second(self) -> None:
        """Test arr[1:3, 0] - slice first, integer second."""
        np_arr = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]])
        pa_arr = Array(np_arr.copy())

        # Test mixed indexing
        result = pa_arr[1:3, 0]
        expected = np_arr[1:3, 0]
        result_np = np.asarray(result)

        # Verify shape and values
        assert result.shape == expected.shape, f"Shape mismatch: {result.shape} vs {expected.shape}"
        np.testing.assert_array_equal(result_np, expected)

    def test_full_slice_integer(self) -> None:
        """Test arr[:, 0] - full slice with integer."""
        np_arr = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]])
        pa_arr = Array(np_arr.copy())

        # Test mixed indexing
        result = pa_arr[:, 0]
        expected = np_arr[:, 0]
        result_np = np.asarray(result)

        # Verify shape and values
        assert result.shape == expected.shape, f"Shape mismatch: {result.shape} vs {expected.shape}"
        np.testing.assert_array_equal(result_np, expected)

    def test_integer_full_slice(self) -> None:
        """Test arr[0, :] - integer with full slice."""
        np_arr = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]])
        pa_arr = Array(np_arr.copy())

        # Test mixed indexing
        result = pa_arr[0, :]
        expected = np_arr[0, :]
        result_np = np.asarray(result)

        # Verify shape and values
        assert result.shape == expected.shape, f"Shape mismatch: {result.shape} vs {expected.shape}"
        np.testing.assert_array_equal(result_np, expected)

    def test_partial_slices_with_integer(self) -> None:
        """Test arr[1:3, 1] - partial slice with integer."""
        np_arr = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]])
        pa_arr = Array(np_arr.copy())

        # Test mixed indexing
        result = pa_arr[1:3, 1]
        expected = np_arr[1:3, 1]
        result_np = np.asarray(result)

        # Verify shape and values
        assert result.shape == expected.shape, f"Shape mismatch: {result.shape} vs {expected.shape}"
        np.testing.assert_array_equal(result_np, expected)


class TestMixedIndexing3D:
    """Test mixed integer/slice indexing for 3D arrays."""

    def test_int_slice_int(self) -> None:
        """Test arr[0, 1:3, 2] - int, slice, int."""
        np_arr = np.arange(24).reshape(3, 4, 2).astype(np.float64)
        pa_arr = Array(np_arr.copy())

        # Test mixed indexing
        result = pa_arr[0, 1:3, 1]
        expected = np_arr[0, 1:3, 1]
        result_np = np.asarray(result)

        # Verify shape and values
        assert result.shape == expected.shape, f"Shape mismatch: {result.shape} vs {expected.shape}"
        np.testing.assert_array_equal(result_np, expected)

    def test_slice_int_slice(self) -> None:
        """Test arr[:, 0, 1:3] - slice, int, slice."""
        np_arr = np.arange(24).reshape(3, 4, 2).astype(np.float64)
        pa_arr = Array(np_arr.copy())

        # Test mixed indexing
        result = pa_arr[:, 1, 0:2]
        expected = np_arr[:, 1, 0:2]
        result_np = np.asarray(result)

        # Verify shape and values
        assert result.shape == expected.shape, f"Shape mismatch: {result.shape} vs {expected.shape}"
        np.testing.assert_array_equal(result_np, expected)

    def test_int_int_slice(self) -> None:
        """Test arr[0, 1, :] - int, int, slice."""
        np_arr = np.arange(24).reshape(3, 4, 2).astype(np.float64)
        pa_arr = Array(np_arr.copy())

        # Test mixed indexing
        result = pa_arr[0, 1, :]
        expected = np_arr[0, 1, :]
        result_np = np.asarray(result)

        # Verify shape and values
        assert result.shape == expected.shape, f"Shape mismatch: {result.shape} vs {expected.shape}"
        np.testing.assert_array_equal(result_np, expected)

    def test_slice_slice_int(self) -> None:
        """Test arr[0:2, 1:3, 1] - slice, slice, int."""
        np_arr = np.arange(24).reshape(3, 4, 2).astype(np.float64)
        pa_arr = Array(np_arr.copy())

        # Test mixed indexing
        result = pa_arr[0:2, 1:3, 1]
        expected = np_arr[0:2, 1:3, 1]
        result_np = np.asarray(result)

        # Verify shape and values
        assert result.shape == expected.shape, f"Shape mismatch: {result.shape} vs {expected.shape}"
        np.testing.assert_array_equal(result_np, expected)

    def test_int_slice_slice(self) -> None:
        """Test arr[1, :, 0:2] - int, slice, slice."""
        np_arr = np.arange(24).reshape(3, 4, 2).astype(np.float64)
        pa_arr = Array(np_arr.copy())

        # Test mixed indexing
        result = pa_arr[1, :, 0:2]
        expected = np_arr[1, :, 0:2]
        result_np = np.asarray(result)

        # Verify shape and values
        assert result.shape == expected.shape, f"Shape mismatch: {result.shape} vs {expected.shape}"
        np.testing.assert_array_equal(result_np, expected)


class TestMixedIndexingNegativeIndices:
    """Test mixed indexing with negative integer indices."""

    def test_negative_integer_with_slice(self) -> None:
        """Test arr[-1, 1:3] - negative integer with slice."""
        np_arr = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]])
        pa_arr = Array(np_arr.copy())

        # Test mixed indexing
        result = pa_arr[-1, 1:3]
        expected = np_arr[-1, 1:3]
        result_np = np.asarray(result)

        # Verify shape and values
        assert result.shape == expected.shape, f"Shape mismatch: {result.shape} vs {expected.shape}"
        np.testing.assert_array_equal(result_np, expected)

    def test_slice_with_negative_integer(self) -> None:
        """Test arr[0:2, -1] - slice with negative integer."""
        np_arr = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]])
        pa_arr = Array(np_arr.copy())

        # Test mixed indexing
        result = pa_arr[0:2, -1]
        expected = np_arr[0:2, -1]
        result_np = np.asarray(result)

        # Verify shape and values
        assert result.shape == expected.shape, f"Shape mismatch: {result.shape} vs {expected.shape}"
        np.testing.assert_array_equal(result_np, expected)

    def test_negative_integer_full_slice(self) -> None:
        """Test arr[-2, :] - negative integer with full slice."""
        np_arr = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]])
        pa_arr = Array(np_arr.copy())

        # Test mixed indexing
        result = pa_arr[-2, :]
        expected = np_arr[-2, :]
        result_np = np.asarray(result)

        # Verify shape and values
        assert result.shape == expected.shape, f"Shape mismatch: {result.shape} vs {expected.shape}"
        np.testing.assert_array_equal(result_np, expected)

    def test_multiple_negative_integers_with_slice(self) -> None:
        """Test arr[-1, -2, :] - multiple negative integers with slice (3D)."""
        np_arr = np.arange(24).reshape(3, 4, 2).astype(np.float64)
        pa_arr = Array(np_arr.copy())

        # Test mixed indexing
        result = pa_arr[-1, -2, :]
        expected = np_arr[-1, -2, :]
        result_np = np.asarray(result)

        # Verify shape and values
        assert result.shape == expected.shape, f"Shape mismatch: {result.shape} vs {expected.shape}"
        np.testing.assert_array_equal(result_np, expected)


class TestMixedIndexingNonUnitStep:
    """Test mixed indexing with non-unit step slices."""

    def test_integer_with_step_slice(self) -> None:
        """Test arr[0, ::2] - integer with step slice."""
        np_arr = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]])
        pa_arr = Array(np_arr.copy())

        # Test mixed indexing
        result = pa_arr[0, ::2]
        expected = np_arr[0, ::2]
        result_np = np.asarray(result)

        # Verify shape and values
        assert result.shape == expected.shape, f"Shape mismatch: {result.shape} vs {expected.shape}"
        np.testing.assert_array_equal(result_np, expected)

    def test_step_slice_with_integer(self) -> None:
        """Test arr[::2, 1] - step slice with integer."""
        np_arr = np.array(
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
                [13.0, 14.0, 15.0, 16.0],
            ]
        )
        pa_arr = Array(np_arr.copy())

        # Test mixed indexing
        result = pa_arr[::2, 1]
        expected = np_arr[::2, 1]
        result_np = np.asarray(result)

        # Verify shape and values
        assert result.shape == expected.shape, f"Shape mismatch: {result.shape} vs {expected.shape}"
        np.testing.assert_array_equal(result_np, expected)

    def test_integer_reverse_slice(self) -> None:
        """Test arr[1, ::-1] - integer with reverse slice."""
        np_arr = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]])
        pa_arr = Array(np_arr.copy())

        # Test mixed indexing
        result = pa_arr[1, ::-1]
        expected = np_arr[1, ::-1]
        result_np = np.asarray(result)

        # Verify shape and values
        assert result.shape == expected.shape, f"Shape mismatch: {result.shape} vs {expected.shape}"
        np.testing.assert_array_equal(result_np, expected)

    def test_reverse_slice_with_integer(self) -> None:
        """Test arr[::-1, 2] - reverse slice with integer."""
        np_arr = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]])
        pa_arr = Array(np_arr.copy())

        # Test mixed indexing
        result = pa_arr[::-1, 2]
        expected = np_arr[::-1, 2]
        result_np = np.asarray(result)

        # Verify shape and values
        assert result.shape == expected.shape, f"Shape mismatch: {result.shape} vs {expected.shape}"
        np.testing.assert_array_equal(result_np, expected)


class TestMixedIndexingDifferentDtypes:
    """Test mixed indexing with different data types."""

    def test_int64_mixed_indexing(self) -> None:
        """Test mixed indexing with int64 array."""
        np_arr = np.array([[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120]])
        pa_arr = Array(np_arr.copy())

        # Test mixed indexing
        result = pa_arr[0, 1:3]
        expected = np_arr[0, 1:3]
        result_np = np.asarray(result)

        # Verify results
        assert result_np.dtype == np.int64, f"Expected int64, got {result_np.dtype}"
        np.testing.assert_array_equal(result_np, expected)

    def test_int32_mixed_indexing(self) -> None:
        """Test mixed indexing with int32 array."""
        np_arr = np.array([[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120]], dtype=np.int32)
        pa_arr = Array(np_arr.copy())

        # Test mixed indexing
        result = pa_arr[1:3, 0]
        expected = np_arr[1:3, 0]
        result_np = np.asarray(result)

        # Verify results
        assert result_np.dtype == np.int32, f"Expected int32, got {result_np.dtype}"
        np.testing.assert_array_equal(result_np, expected)

    def test_int16_mixed_indexing(self) -> None:
        """Test mixed indexing with int16 array."""
        np_arr = np.array([[10, 20, 30, 40], [50, 60, 70, 80]], dtype=np.int16)
        pa_arr = Array(np_arr.copy())

        # Test mixed indexing
        result = pa_arr[:, 1]
        expected = np_arr[:, 1]
        result_np = np.asarray(result)

        # Verify results
        assert result_np.dtype == np.int16, f"Expected int16, got {result_np.dtype}"
        np.testing.assert_array_equal(result_np, expected)

    def test_int8_mixed_indexing(self) -> None:
        """Test mixed indexing with int8 array."""
        np_arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int8)
        pa_arr = Array(np_arr.copy())

        # Test mixed indexing
        result = pa_arr[0, :]
        expected = np_arr[0, :]
        result_np = np.asarray(result)

        # Verify results
        assert result_np.dtype == np.int8, f"Expected int8, got {result_np.dtype}"
        np.testing.assert_array_equal(result_np, expected)

    def test_float32_mixed_indexing(self) -> None:
        """Test mixed indexing with float32 array."""
        np_arr = np.array(
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]],
            dtype=np.float32,
        )
        pa_arr = Array(np_arr.copy())

        # Test mixed indexing
        result = pa_arr[1, 1:3]
        expected = np_arr[1, 1:3]
        result_np = np.asarray(result)

        # Verify results
        assert result_np.dtype == np.float32, f"Expected float32, got {result_np.dtype}"
        np.testing.assert_array_equal(result_np, expected)

    def test_complex128_mixed_indexing(self) -> None:
        """Test mixed indexing with complex128 array."""
        np_arr = np.array([[1 + 2j, 3 + 4j, 5 + 6j], [7 + 8j, 9 + 10j, 11 + 12j]])
        pa_arr = Array(np_arr.copy())

        # Test mixed indexing
        result = pa_arr[0, 1:]
        expected = np_arr[0, 1:]
        result_np = np.asarray(result)

        # Verify results
        assert result_np.dtype == np.complex128, f"Expected complex128, got {result_np.dtype}"
        np.testing.assert_array_equal(result_np, expected)

    def test_complex64_mixed_indexing(self) -> None:
        """Test mixed indexing with complex64 array."""
        np_arr = np.array([[1 + 2j, 3 + 4j, 5 + 6j], [7 + 8j, 9 + 10j, 11 + 12j]], dtype=np.complex64)
        pa_arr = Array(np_arr.copy())

        # Test mixed indexing
        result = pa_arr[:, 1]
        expected = np_arr[:, 1]
        result_np = np.asarray(result)

        # Verify results
        assert result_np.dtype == np.complex64, f"Expected complex64, got {result_np.dtype}"
        np.testing.assert_array_equal(result_np, expected)


class TestMixedIndexingEdgeCases:
    """Test edge cases for mixed indexing."""

    def test_single_element_result(self) -> None:
        """Test when result is a single-element array."""
        np_arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        pa_arr = Array(np_arr.copy())

        # Test mixed indexing that results in single element
        result = pa_arr[0, 1:2]
        expected = np_arr[0, 1:2]
        result_np = np.asarray(result)

        # Verify shape and values
        assert result.shape == expected.shape, f"Shape mismatch: {result.shape} vs {expected.shape}"
        assert result.shape == (1,), f"Expected shape (1,), got {result.shape}"
        np.testing.assert_array_equal(result_np, expected)

    def test_empty_slice_result(self) -> None:
        """Test when slice produces empty result."""
        np_arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        pa_arr = Array(np_arr.copy())

        # Test mixed indexing with empty slice
        result = pa_arr[0, 5:10]
        expected = np_arr[0, 5:10]
        result_np = np.asarray(result)

        # Verify empty result
        assert result.shape == expected.shape, f"Shape mismatch: {result.shape} vs {expected.shape}"
        assert result.shape == (0,), f"Expected shape (0,), got {result.shape}"
        np.testing.assert_array_equal(result_np, expected)

    def test_full_array_slice_with_integer(self) -> None:
        """Test arr[:, :] would be all slices, but arr[0, :] is mixed."""
        np_arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        pa_arr = Array(np_arr.copy())

        # Test that integer collapses one dimension
        result = pa_arr[0, :]
        expected = np_arr[0, :]
        result_np = np.asarray(result)

        # Verify dimensionality reduction
        assert result.ndim == 1, f"Expected ndim=1, got {result.ndim}"
        assert result.shape == expected.shape, f"Shape mismatch: {result.shape} vs {expected.shape}"
        np.testing.assert_array_equal(result_np, expected)

    def test_out_of_bounds_integer_index(self) -> None:
        """Test out of bounds integer index with slice."""
        np_arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        pa_arr = Array(np_arr.copy())

        # Test out of bounds - should raise IndexError
        with pytest.raises(IndexError):
            _ = pa_arr[10, 1:2]

    def test_negative_out_of_bounds(self) -> None:
        """Test negative out of bounds integer index."""
        np_arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        pa_arr = Array(np_arr.copy())

        # Test negative out of bounds - should raise IndexError
        with pytest.raises(IndexError):
            _ = pa_arr[-10, 1:2]


class TestMixedIndexingConsistency:
    """Test that mixed indexing is consistent with NumPy."""

    def test_mixed_vs_pure_integer_indexing(self) -> None:
        """Verify mixed indexing matches sequential pure integer indexing."""
        np_arr = np.arange(24).reshape(3, 4, 2).astype(np.float64)
        pa_arr = Array(np_arr.copy())

        # Mixed indexing: arr[1, 2, :]
        mixed_result = pa_arr[1, 2, :]
        # Sequential: arr[1][2][:]
        seq_result_step1 = pa_arr[1, :, :]  # Shape (4, 2)
        seq_result_step2 = seq_result_step1[2, :]  # Shape (2,)

        mixed_np = np.asarray(mixed_result)
        seq_np = np.asarray(seq_result_step2)

        # Results should match
        np.testing.assert_array_equal(mixed_np, seq_np)

    def test_order_independence_verification(self) -> None:
        """Verify that the order of operations matches NumPy."""
        np_arr = np.arange(24).reshape(3, 4, 2).astype(np.float64)
        pa_arr = Array(np_arr.copy())

        # Different mixed indexing patterns should produce predictable results
        result1 = pa_arr[0, :, 1]  # Shape (4,)
        expected1 = np_arr[0, :, 1]
        np.testing.assert_array_equal(np.asarray(result1), expected1)

        result2 = pa_arr[:, 0, 1]  # Shape (3,)
        expected2 = np_arr[:, 0, 1]
        np.testing.assert_array_equal(np.asarray(result2), expected2)

    def test_multiple_operations_preserve_values(self) -> None:
        """Test multiple mixed indexing operations on same array."""
        np_arr = np.arange(60).reshape(5, 4, 3).astype(np.float64)
        pa_arr = Array(np_arr.copy())

        # First operation
        result1 = pa_arr[0, 1:3, 1]
        expected1 = np_arr[0, 1:3, 1]
        np.testing.assert_array_equal(np.asarray(result1), expected1)

        # Second operation on same array
        result2 = pa_arr[2:4, 0, :]
        expected2 = np_arr[2:4, 0, :]
        np.testing.assert_array_equal(np.asarray(result2), expected2)

        # Third operation
        result3 = pa_arr[:, 2, 1:3]
        expected3 = np_arr[:, 2, 1:3]
        np.testing.assert_array_equal(np.asarray(result3), expected3)

    def test_conversion_to_numpy_preserves_values(self) -> None:
        """Test that conversion to NumPy preserves values after mixed indexing."""
        np_arr = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]])
        pa_arr = Array(np_arr.copy())

        # Perform mixed indexing
        result = pa_arr[1, 1:3]

        # Convert to NumPy and verify
        result_np = np.asarray(result)
        expected = np_arr[1, 1:3]

        np.testing.assert_array_equal(result_np, expected)
        assert result_np.dtype == expected.dtype
