"""Comprehensive tests comparing Array arithmetic operations with numpy.

This test suite ensures our Array arithmetic operations (+, -, *, /)
match numpy's behavior across all operand combinations:
- Array + scalar, scalar + array
- Array + array (Array, numpy array)
- Different dtypes (int64, float64, complex128)
- Broadcasting behavior
- Commutative operations (addition, multiplication)
- Non-commutative operations (subtraction, division)
"""

import numpy as np
import pytest

from pecos_rslib import Array


class TestPecosArrayAddition:
    """Test Array addition against numpy arrays."""

    def test_array_plus_scalar_float(self):
        """Test: array + scalar (float)."""
        np_arr = np.array([1.0, 2.0, 3.0, 4.0])
        pa_arr = Array(np_arr)

        np_result = np_arr + 10.0
        pa_result = pa_arr + 10.0

        pa_result_np = np.asarray(pa_result)
        np.testing.assert_array_equal(pa_result_np, np_result)
        # Verify dtype compatibility via buffer protocol conversion
        assert pa_result_np.dtype == np_result.dtype

    def test_scalar_plus_array_float(self):
        """Test: scalar + array (reverse operation)."""
        np_arr = np.array([1.0, 2.0, 3.0, 4.0])
        pa_arr = Array(np_arr)

        np_result = 10.0 + np_arr
        pa_result = 10.0 + pa_arr

        pa_result_np = np.asarray(pa_result)
        np.testing.assert_array_equal(pa_result_np, np_result)

    def test_array_plus_array_float(self):
        """Test: array + array (both PecosArray)."""
        np_arr1 = np.array([1.0, 2.0, 3.0])
        np_arr2 = np.array([10.0, 20.0, 30.0])
        pa_arr1 = Array(np_arr1)
        pa_arr2 = Array(np_arr2)

        np_result = np_arr1 + np_arr2
        pa_result = pa_arr1 + pa_arr2

        pa_result_np = np.asarray(pa_result)
        np.testing.assert_array_equal(pa_result_np, np_result)

    def test_pecos_array_plus_numpy_array(self):
        """Test: Array + numpy array."""
        np_arr1 = np.array([1.0, 2.0, 3.0])
        np_arr2 = np.array([10.0, 20.0, 30.0])
        pa_arr = Array(np_arr1)

        np_result = np_arr1 + np_arr2
        pa_result = pa_arr + np_arr2

        pa_result_np = np.asarray(pa_result)
        np.testing.assert_array_equal(pa_result_np, np_result)

    def test_array_plus_scalar_int(self):
        """Test: int array + scalar."""
        np_arr = np.array([1, 2, 3, 4])
        pa_arr = Array(np_arr)

        np_result = np_arr + 10.0
        pa_result = pa_arr + 10.0

        pa_result_np = np.asarray(pa_result)
        # Note: type conversion may differ, just check values
        np.testing.assert_array_almost_equal(pa_result_np, np_result)

    def test_array_plus_scalar_complex(self):
        """Test: complex array + scalar."""
        np_arr = np.array([1 + 2j, 3 + 4j, 5 + 6j])
        pa_arr = Array(np_arr)

        np_result = np_arr + 10.0
        pa_result = pa_arr + 10.0

        pa_result_np = np.asarray(pa_result)
        np.testing.assert_array_almost_equal(pa_result_np, np_result)

    def test_commutative_property(self):
        """Test: a + b == b + a (commutativity)."""
        np_arr = np.array([1.0, 2.0, 3.0])
        pa_arr = Array(np_arr)
        scalar = 5.0

        result1 = pa_arr + scalar
        result2 = scalar + pa_arr

        np.testing.assert_array_equal(np.asarray(result1), np.asarray(result2))

    def test_2d_array_plus_scalar(self):
        """Test: 2D array + scalar."""
        np_arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        pa_arr = Array(np_arr)

        np_result = np_arr + 100.0
        pa_result = pa_arr + 100.0

        pa_result_np = np.asarray(pa_result)
        np.testing.assert_array_equal(pa_result_np, np_result)


class TestPecosArraySubtraction:
    """Test Array subtraction against numpy arrays."""

    def test_array_minus_scalar(self):
        """Test: array - scalar."""
        np_arr = np.array([10.0, 20.0, 30.0, 40.0])
        pa_arr = Array(np_arr)

        np_result = np_arr - 5.0
        pa_result = pa_arr - 5.0

        pa_result_np = np.asarray(pa_result)
        np.testing.assert_array_equal(pa_result_np, np_result)

    def test_scalar_minus_array(self):
        """Test: scalar - array (reverse operation)."""
        np_arr = np.array([1.0, 2.0, 3.0, 4.0])
        pa_arr = Array(np_arr)

        np_result = 10.0 - np_arr
        pa_result = 10.0 - pa_arr

        pa_result_np = np.asarray(pa_result)
        np.testing.assert_array_equal(pa_result_np, np_result)
        # Expected: [9.0, 8.0, 7.0, 6.0]

    def test_array_minus_array(self):
        """Test: array - array."""
        np_arr1 = np.array([10.0, 20.0, 30.0])
        np_arr2 = np.array([1.0, 2.0, 3.0])
        pa_arr1 = Array(np_arr1)
        pa_arr2 = Array(np_arr2)

        np_result = np_arr1 - np_arr2
        pa_result = pa_arr1 - pa_arr2

        pa_result_np = np.asarray(pa_result)
        np.testing.assert_array_equal(pa_result_np, np_result)

    def test_non_commutative_property(self):
        """Test: a - b != b - a (non-commutativity)."""
        np_arr = np.array([10.0, 20.0, 30.0])
        pa_arr = Array(np_arr)
        scalar = 5.0

        result1 = pa_arr - scalar  # [5, 15, 25]
        result2 = scalar - pa_arr  # [-5, -15, -25]

        result1_np = np.asarray(result1)
        result2_np = np.asarray(result2)

        # Should NOT be equal
        assert not np.array_equal(result1_np, result2_np)

        # Verify against numpy
        np.testing.assert_array_equal(result1_np, np_arr - scalar)
        np.testing.assert_array_equal(result2_np, scalar - np_arr)

    def test_complex_subtraction(self):
        """Test: complex array - scalar."""
        np_arr = np.array([1 + 2j, 3 + 4j, 5 + 6j])
        pa_arr = Array(np_arr)

        np_result = np_arr - (1 + 1j)
        pa_result = pa_arr - (1 + 1j)

        pa_result_np = np.asarray(pa_result)
        np.testing.assert_array_almost_equal(pa_result_np, np_result)


class TestPecosArrayMultiplication:
    """Test Array multiplication against numpy arrays."""

    def test_array_times_scalar(self):
        """Test: array * scalar."""
        np_arr = np.array([1.0, 2.0, 3.0, 4.0])
        pa_arr = Array(np_arr)

        np_result = np_arr * 2.0
        pa_result = pa_arr * 2.0

        pa_result_np = np.asarray(pa_result)
        np.testing.assert_array_equal(pa_result_np, np_result)

    def test_scalar_times_array(self):
        """Test: scalar * array (reverse operation)."""
        np_arr = np.array([1.0, 2.0, 3.0, 4.0])
        pa_arr = Array(np_arr)

        np_result = 3.0 * np_arr
        pa_result = 3.0 * pa_arr

        pa_result_np = np.asarray(pa_result)
        np.testing.assert_array_equal(pa_result_np, np_result)

    def test_array_times_array(self):
        """Test: array * array (element-wise)."""
        np_arr1 = np.array([1.0, 2.0, 3.0])
        np_arr2 = np.array([10.0, 20.0, 30.0])
        pa_arr1 = Array(np_arr1)
        pa_arr2 = Array(np_arr2)

        np_result = np_arr1 * np_arr2
        pa_result = pa_arr1 * pa_arr2

        pa_result_np = np.asarray(pa_result)
        np.testing.assert_array_equal(pa_result_np, np_result)
        # Expected: [10.0, 40.0, 90.0]

    def test_commutative_property(self):
        """Test: a * b == b * a (commutativity)."""
        np_arr = np.array([1.0, 2.0, 3.0])
        pa_arr = Array(np_arr)
        scalar = 5.0

        result1 = pa_arr * scalar
        result2 = scalar * pa_arr

        np.testing.assert_array_equal(np.asarray(result1), np.asarray(result2))

    def test_complex_multiplication(self):
        """Test: complex array * scalar."""
        np_arr = np.array([1 + 2j, 3 + 4j])
        pa_arr = Array(np_arr)

        np_result = np_arr * 2.0
        pa_result = pa_arr * 2.0

        pa_result_np = np.asarray(pa_result)
        np.testing.assert_array_almost_equal(pa_result_np, np_result)

    def test_int_array_multiplication(self):
        """Test: int array * scalar."""
        np_arr = np.array([1, 2, 3, 4])
        pa_arr = Array(np_arr)

        np_result = np_arr * 5.0
        pa_result = pa_arr * 5.0

        pa_result_np = np.asarray(pa_result)
        np.testing.assert_array_almost_equal(pa_result_np, np_result)


class TestPecosArrayDivision:
    """Test Array division against numpy arrays."""

    def test_array_divided_by_scalar(self):
        """Test: array / scalar."""
        np_arr = np.array([10.0, 20.0, 30.0, 40.0])
        pa_arr = Array(np_arr)

        np_result = np_arr / 2.0
        pa_result = pa_arr / 2.0

        pa_result_np = np.asarray(pa_result)
        np.testing.assert_array_almost_equal(pa_result_np, np_result)

    def test_scalar_divided_by_array(self):
        """Test: scalar / array (reverse operation)."""
        np_arr = np.array([1.0, 2.0, 4.0, 5.0])
        pa_arr = Array(np_arr)

        np_result = 10.0 / np_arr
        pa_result = 10.0 / pa_arr

        pa_result_np = np.asarray(pa_result)
        np.testing.assert_array_almost_equal(pa_result_np, np_result)
        # Expected: [10.0, 5.0, 2.5, 2.0]

    def test_array_divided_by_array(self):
        """Test: array / array (element-wise)."""
        np_arr1 = np.array([10.0, 20.0, 30.0])
        np_arr2 = np.array([2.0, 4.0, 5.0])
        pa_arr1 = Array(np_arr1)
        pa_arr2 = Array(np_arr2)

        np_result = np_arr1 / np_arr2
        pa_result = pa_arr1 / pa_arr2

        pa_result_np = np.asarray(pa_result)
        np.testing.assert_array_almost_equal(pa_result_np, np_result)
        # Expected: [5.0, 5.0, 6.0]

    def test_non_commutative_property(self):
        """Test: a / b != b / a (non-commutativity)."""
        np_arr = np.array([10.0, 20.0, 40.0])
        pa_arr = Array(np_arr)
        scalar = 2.0

        result1 = pa_arr / scalar  # [5, 10, 20]
        result2 = scalar / pa_arr  # [0.2, 0.1, 0.05]

        result1_np = np.asarray(result1)
        result2_np = np.asarray(result2)

        # Should NOT be equal
        assert not np.allclose(result1_np, result2_np)

        # Verify against numpy
        np.testing.assert_array_almost_equal(result1_np, np_arr / scalar)
        np.testing.assert_array_almost_equal(result2_np, scalar / np_arr)

    def test_complex_division(self):
        """Test: complex array / scalar."""
        np_arr = np.array([2 + 4j, 6 + 8j])
        pa_arr = Array(np_arr)

        np_result = np_arr / 2.0
        pa_result = pa_arr / 2.0

        pa_result_np = np.asarray(pa_result)
        np.testing.assert_array_almost_equal(pa_result_np, np_result)

    def test_int_array_division(self):
        """Test: int array / scalar (results in float)."""
        np_arr = np.array([10, 20, 30, 40])
        pa_arr = Array(np_arr)

        np_result = np_arr / 2.0
        pa_result = pa_arr / 2.0

        pa_result_np = np.asarray(pa_result)
        np.testing.assert_array_almost_equal(pa_result_np, np_result)


class TestArrayShapeMismatch:
    """Test error handling for shape mismatches."""

    def test_shape_mismatch_addition(self):
        """Test: array + array with mismatched shapes should raise error."""
        np_arr1 = np.array([1.0, 2.0, 3.0])
        np_arr2 = np.array([1.0, 2.0, 3.0, 4.0])
        pa_arr1 = Array(np_arr1)
        pa_arr2 = Array(np_arr2)

        with pytest.raises(ValueError, match="Shape mismatch"):
            pa_arr1 + pa_arr2

    def test_shape_mismatch_subtraction(self):
        """Test: array - array with mismatched shapes should raise error."""
        np_arr1 = np.array([1.0, 2.0])
        np_arr2 = np.array([1.0, 2.0, 3.0])
        pa_arr1 = Array(np_arr1)
        pa_arr2 = Array(np_arr2)

        with pytest.raises(ValueError, match="Shape mismatch"):
            pa_arr1 - pa_arr2


class TestArrayCombinedOperations:
    """Test combined arithmetic operations."""

    def test_multiple_operations(self):
        """Test: (array + scalar) * scalar - scalar."""
        np_arr = np.array([1.0, 2.0, 3.0])
        pa_arr = Array(np_arr)

        np_result = (np_arr + 10.0) * 2.0 - 5.0
        pa_result = (pa_arr + 10.0) * 2.0 - 5.0

        pa_result_np = np.asarray(pa_result)
        np.testing.assert_array_almost_equal(pa_result_np, np_result)

    def test_array_operations_chain(self):
        """Test: chained array operations."""
        np_arr1 = np.array([10.0, 20.0, 30.0])
        np_arr2 = np.array([1.0, 2.0, 3.0])
        pa_arr1 = Array(np_arr1)
        pa_arr2 = Array(np_arr2)

        np_result = (np_arr1 + np_arr2) * 2.0 / 4.0
        pa_result = (pa_arr1 + pa_arr2) * 2.0 / 4.0

        pa_result_np = np.asarray(pa_result)
        np.testing.assert_array_almost_equal(pa_result_np, np_result)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
