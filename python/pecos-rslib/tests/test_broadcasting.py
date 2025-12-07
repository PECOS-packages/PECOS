"""
Comprehensive tests for array broadcasting in pecos_rslib.

This module tests that our Array implementation follows NumPy's broadcasting rules:
- Arrays with different shapes can be operated on if they are compatible
- Broadcasting allows element-wise operations between arrays of different sizes
- Rules: dimensions are compatible if they are equal or one of them is 1
"""

import numpy as np
import pytest

from pecos_rslib import Array


class TestBasicBroadcasting:
    """Test basic broadcasting scenarios."""

    def test_scalar_broadcast(self):
        """Test scalar broadcasting (already working, but verify)."""
        np_arr = np.array([1.0, 2.0, 3.0])
        pa_arr = Array(np_arr)

        # Scalar + array
        np_result = np_arr + 5.0
        pa_result = pa_arr + 5.0

        np.testing.assert_array_almost_equal(np.asarray(pa_result), np_result)

    def test_1d_to_2d_broadcast(self):
        """Test broadcasting 1D array to 2D array."""
        # (3,) + (3, 4) -> should broadcast to (3, 4)
        np_a = np.array([1.0, 2.0, 3.0])
        np_b = np.ones((3, 4))

        pa_a = Array(np_a)
        pa_b = Array(np_b)

        # NumPy result
        np_result = np_a[:, np.newaxis] + np_b  # Need to reshape for NumPy

        # PECOS result - should handle broadcasting automatically
        # Actually, (3,) and (3,4) are NOT compatible in NumPy broadcasting
        # (3,) needs to match the last dimension
        # Let's test the correct case: (4,) + (3, 4) -> (3, 4)

        np_a = np.array([1.0, 2.0, 3.0, 4.0])  # (4,)
        np_b = np.ones((3, 4))  # (3, 4)

        pa_a = Array(np_a)
        pa_b = Array(np_b)

        np_result = np_a + np_b  # NumPy broadcasts (4,) to (3, 4)
        pa_result = pa_a + pa_b

        np.testing.assert_array_almost_equal(np.asarray(pa_result), np_result)

    def test_column_vector_broadcast(self):
        """Test broadcasting a column vector (n, 1) with a matrix (n, m)."""
        np_col = np.array([[1.0], [2.0], [3.0]])  # (3, 1)
        np_mat = np.ones((3, 4))  # (3, 4)

        pa_col = Array(np_col)
        pa_mat = Array(np_mat)

        np_result = np_col + np_mat
        pa_result = pa_col + pa_mat

        np.testing.assert_array_almost_equal(np.asarray(pa_result), np_result)
        assert np.asarray(pa_result).shape == (3, 4)

    def test_row_vector_broadcast(self):
        """Test broadcasting a row vector (1, m) with a matrix (n, m)."""
        np_row = np.array([[1.0, 2.0, 3.0, 4.0]])  # (1, 4)
        np_mat = np.ones((3, 4))  # (3, 4)

        pa_row = Array(np_row)
        pa_mat = Array(np_mat)

        np_result = np_row + np_mat
        pa_result = pa_row + pa_mat

        np.testing.assert_array_almost_equal(np.asarray(pa_result), np_result)
        assert np.asarray(pa_result).shape == (3, 4)


class TestBroadcastingAllOperations:
    """Test that broadcasting works for all arithmetic operations."""

    def test_broadcast_addition(self):
        """Test broadcasting with addition."""
        np_a = np.array([[1.0], [2.0], [3.0]])  # (3, 1)
        np_b = np.array([10.0, 20.0, 30.0, 40.0])  # (4,)

        pa_a = Array(np_a)
        pa_b = Array(np_b)

        np_result = np_a + np_b
        pa_result = pa_a + pa_b

        np.testing.assert_array_almost_equal(np.asarray(pa_result), np_result)
        assert np.asarray(pa_result).shape == (3, 4)

    def test_broadcast_subtraction(self):
        """Test broadcasting with subtraction."""
        np_a = np.array([[1.0], [2.0], [3.0]])  # (3, 1)
        np_b = np.array([10.0, 20.0, 30.0, 40.0])  # (4,)

        pa_a = Array(np_a)
        pa_b = Array(np_b)

        np_result = np_a - np_b
        pa_result = pa_a - pa_b

        np.testing.assert_array_almost_equal(np.asarray(pa_result), np_result)

    def test_broadcast_multiplication(self):
        """Test broadcasting with multiplication."""
        np_a = np.array([[2.0], [3.0], [4.0]])  # (3, 1)
        np_b = np.array([10.0, 20.0, 30.0])  # (3,)

        pa_a = Array(np_a)
        pa_b = Array(np_b)

        np_result = np_a * np_b
        pa_result = pa_a * pa_b

        np.testing.assert_array_almost_equal(np.asarray(pa_result), np_result)

    def test_broadcast_division(self):
        """Test broadcasting with division."""
        np_a = np.array([[10.0], [20.0], [30.0]])  # (3, 1)
        np_b = np.array([2.0, 4.0, 5.0])  # (3,)

        pa_a = Array(np_a)
        pa_b = Array(np_b)

        np_result = np_a / np_b
        pa_result = pa_a / pa_b

        np.testing.assert_array_almost_equal(np.asarray(pa_result), np_result)


class TestBroadcastingComplex:
    """Test broadcasting with different data types."""

    def test_broadcast_int64(self):
        """Test broadcasting with int64 arrays."""
        np_a = np.array([[1], [2], [3]], dtype=np.int64)  # (3, 1)
        np_b = np.array([10, 20, 30], dtype=np.int64)  # (3,)

        pa_a = Array(np_a)
        pa_b = Array(np_b)

        np_result = np_a + np_b
        pa_result = pa_a + pa_b

        np.testing.assert_array_equal(np.asarray(pa_result), np_result)

    def test_broadcast_complex128(self):
        """Test broadcasting with complex128 arrays."""
        np_a = np.array([[1 + 2j], [3 + 4j]], dtype=np.complex128)  # (2, 1)
        np_b = np.array([10 + 0j, 20 + 0j, 30 + 0j], dtype=np.complex128)  # (3,)

        pa_a = Array(np_a)
        pa_b = Array(np_b)

        np_result = np_a + np_b
        pa_result = pa_a + pa_b

        np.testing.assert_array_almost_equal(np.asarray(pa_result), np_result)


class TestBroadcastingEdgeCases:
    """Test edge cases and error conditions."""

    def test_incompatible_shapes_error(self):
        """Test that incompatible shapes raise an error."""
        np_a = np.ones((3, 4))
        np_b = np.ones((2, 4))

        pa_a = Array(np_a)
        pa_b = Array(np_b)

        # These shapes are incompatible for broadcasting
        with pytest.raises(ValueError, match="cannot broadcast"):
            pa_a + pa_b

    def test_same_shape_no_broadcast(self):
        """Test that same-shaped arrays work (no broadcasting needed)."""
        np_a = np.ones((3, 4))
        np_b = np.ones((3, 4)) * 2.0

        pa_a = Array(np_a)
        pa_b = Array(np_b)

        np_result = np_a + np_b
        pa_result = pa_a + pa_b

        np.testing.assert_array_almost_equal(np.asarray(pa_result), np_result)

    def test_broadcast_single_element(self):
        """Test broadcasting a single element (1,1) array."""
        np_a = np.array([[5.0]])  # (1, 1)
        np_b = np.ones((3, 4))  # (3, 4)

        pa_a = Array(np_a)
        pa_b = Array(np_b)

        np_result = np_a + np_b
        pa_result = pa_a + pa_b

        np.testing.assert_array_almost_equal(np.asarray(pa_result), np_result)
        assert np.asarray(pa_result).shape == (3, 4)


class TestBroadcastingWithNumPy:
    """Test broadcasting when one operand is a NumPy array."""

    def test_pecos_array_plus_numpy_broadcast(self):
        """Test PECOS Array + NumPy array with broadcasting."""
        np_a = np.array([[1.0], [2.0], [3.0]])  # (3, 1)
        np_b = np.array([10.0, 20.0, 30.0, 40.0])  # (4,)

        pa_a = Array(np_a)

        np_result = np_a + np_b
        pa_result = pa_a + np_b  # NumPy array on right side

        np.testing.assert_array_almost_equal(np.asarray(pa_result), np_result)


class TestBroadcastingHigherDimensions:
    """Test broadcasting with 3D and higher dimensional arrays."""

    def test_3d_broadcast(self):
        """Test broadcasting with 3D arrays."""
        np_a = np.ones((2, 3, 1))  # (2, 3, 1)
        np_b = np.ones((1, 3, 4))  # (1, 3, 4)

        pa_a = Array(np_a)
        pa_b = Array(np_b)

        np_result = np_a + np_b  # Should broadcast to (2, 3, 4)
        pa_result = pa_a + pa_b

        np.testing.assert_array_almost_equal(np.asarray(pa_result), np_result)
        assert np.asarray(pa_result).shape == (2, 3, 4)

    def test_4d_broadcast(self):
        """Test broadcasting with 4D arrays."""
        # Simulating batch_size × channels × height × width
        np_a = np.ones((2, 1, 3, 4))  # (2, 1, 3, 4)
        np_b = np.ones((5, 3, 1))  # (5, 3, 1) - will broadcast to (2, 5, 3, 4)

        pa_a = Array(np_a)
        pa_b = Array(np_b)

        np_result = np_a + np_b  # Should broadcast to (2, 5, 3, 4)
        pa_result = pa_a + pa_b

        np.testing.assert_array_almost_equal(np.asarray(pa_result), np_result)
        assert np.asarray(pa_result).shape == (2, 5, 3, 4)

    def test_5d_broadcast(self):
        """Test broadcasting with 5D arrays."""
        # Simulating batch × time × qubits × gates × parameters
        np_a = np.ones((2, 3, 1, 4, 5))  # (2, 3, 1, 4, 5)
        np_b = np.ones((1, 6, 1, 5))  # (1, 6, 1, 5)

        pa_a = Array(np_a)
        pa_b = Array(np_b)

        np_result = np_a + np_b  # Should broadcast to (2, 3, 6, 4, 5)
        pa_result = pa_a + pa_b

        np.testing.assert_array_almost_equal(np.asarray(pa_result), np_result)
        assert np.asarray(pa_result).shape == (2, 3, 6, 4, 5)

    def test_6d_broadcast_extreme(self):
        """Test broadcasting with 6D arrays to verify truly general ND support."""
        # Extreme case: 6D tensors
        np_a = np.ones((1, 2, 1, 3, 1, 4))  # (1, 2, 1, 3, 1, 4)
        np_b = np.ones((2, 1, 2, 1, 3, 1))  # (2, 1, 2, 1, 3, 1)

        pa_a = Array(np_a)
        pa_b = Array(np_b)

        np_result = np_a + np_b  # Should broadcast to (2, 2, 2, 3, 3, 4)
        pa_result = pa_a + pa_b

        np.testing.assert_array_almost_equal(np.asarray(pa_result), np_result)
        assert np.asarray(pa_result).shape == (2, 2, 2, 3, 3, 4)
