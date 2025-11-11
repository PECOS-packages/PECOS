"""Tests for zeros() and ones() functions.

This module tests the Rust implementations of zeros() and ones() against NumPy
to ensure they are drop-in replacements.
"""

import numpy as np

from pecos_rslib import ones, zeros


class TestZeros:
    """Test zeros() function against numpy.zeros()."""

    def test_zeros_1d_float(self):
        """Test 1D float array creation."""
        # Rust implementation
        rust_result = zeros(5)

        # NumPy reference
        numpy_result = np.zeros(5)

        # Check shape and dtype
        assert rust_result.shape == numpy_result.shape
        assert rust_result.dtype == numpy_result.dtype

        # Check values
        np.testing.assert_array_equal(rust_result, numpy_result)

    def test_zeros_2d_float(self):
        """Test 2D float array creation."""
        # Rust implementation
        rust_result = zeros((3, 4))

        # NumPy reference
        numpy_result = np.zeros((3, 4))

        # Check shape and dtype
        assert rust_result.shape == numpy_result.shape
        assert rust_result.dtype == numpy_result.dtype

        # Check values
        np.testing.assert_array_equal(rust_result, numpy_result)

    def test_zeros_3d_float(self):
        """Test 3D float array creation."""
        # Rust implementation
        rust_result = zeros((2, 3, 4))

        # NumPy reference
        numpy_result = np.zeros((2, 3, 4))

        # Check shape and dtype
        assert rust_result.shape == numpy_result.shape
        assert rust_result.dtype == numpy_result.dtype

        # Check values
        np.testing.assert_array_equal(rust_result, numpy_result)

    def test_zeros_1d_complex(self):
        """Test 1D complex array creation."""
        # Rust implementation
        rust_result = zeros(5, dtype="complex128")

        # NumPy reference
        numpy_result = np.zeros(5, dtype=np.complex128)

        # Check shape and dtype
        assert rust_result.shape == numpy_result.shape
        assert rust_result.dtype == numpy_result.dtype

        # Check values
        np.testing.assert_array_equal(rust_result, numpy_result)

    def test_zeros_2d_complex(self):
        """Test 2D complex array creation."""
        # Rust implementation
        rust_result = zeros((3, 4), dtype="complex128")

        # NumPy reference
        numpy_result = np.zeros((3, 4), dtype=np.complex128)

        # Check shape and dtype
        assert rust_result.shape == numpy_result.shape
        assert rust_result.dtype == numpy_result.dtype

        # Check values
        np.testing.assert_array_equal(rust_result, numpy_result)

    def test_zeros_1d_int(self):
        """Test 1D integer array creation."""
        # Rust implementation
        rust_result = zeros(5, dtype="int64")

        # NumPy reference
        numpy_result = np.zeros(5, dtype=np.int64)

        # Check shape and dtype
        assert rust_result.shape == numpy_result.shape
        assert rust_result.dtype == numpy_result.dtype

        # Check values
        np.testing.assert_array_equal(rust_result, numpy_result)

    def test_zeros_2d_int(self):
        """Test 2D integer array creation."""
        # Rust implementation
        rust_result = zeros((3, 4), dtype="int64")

        # NumPy reference
        numpy_result = np.zeros((3, 4), dtype=np.int64)

        # Check shape and dtype
        assert rust_result.shape == numpy_result.shape
        assert rust_result.dtype == numpy_result.dtype

        # Check values
        np.testing.assert_array_equal(rust_result, numpy_result)

    def test_zeros_dtype_aliases(self):
        """Test that dtype aliases work (float, complex, int)."""
        # float alias
        result_float = zeros(3, dtype="float")
        assert result_float.dtype == np.float64

        # complex alias
        result_complex = zeros(3, dtype="complex")
        assert result_complex.dtype == np.complex128

        # int alias
        result_int = zeros(3, dtype="int")
        assert result_int.dtype == np.int64

    def test_zeros_shape_as_list(self):
        """Test that shape can be provided as a list."""
        # Shape as list
        rust_result = zeros([3, 4])

        # NumPy reference
        numpy_result = np.zeros((3, 4))

        # Check shape and dtype
        assert rust_result.shape == numpy_result.shape
        np.testing.assert_array_equal(rust_result, numpy_result)

    def test_zeros_edge_cases(self):
        """Test edge cases like empty arrays."""
        # Empty 1D array
        result = zeros(0)
        assert result.shape == (0,)
        assert len(result) == 0

        # Single element
        result = zeros(1)
        assert result.shape == (1,)
        assert result[0] == 0.0


class TestOnes:
    """Test ones() function against numpy.ones()."""

    def test_ones_1d_float(self):
        """Test 1D float array creation."""
        # Rust implementation
        rust_result = ones(5)

        # NumPy reference
        numpy_result = np.ones(5)

        # Check shape and dtype
        assert rust_result.shape == numpy_result.shape
        assert rust_result.dtype == numpy_result.dtype

        # Check values
        np.testing.assert_array_equal(rust_result, numpy_result)

    def test_ones_2d_float(self):
        """Test 2D float array creation."""
        # Rust implementation
        rust_result = ones((3, 4))

        # NumPy reference
        numpy_result = np.ones((3, 4))

        # Check shape and dtype
        assert rust_result.shape == numpy_result.shape
        assert rust_result.dtype == numpy_result.dtype

        # Check values
        np.testing.assert_array_equal(rust_result, numpy_result)

    def test_ones_3d_float(self):
        """Test 3D float array creation."""
        # Rust implementation
        rust_result = ones((2, 3, 4))

        # NumPy reference
        numpy_result = np.ones((2, 3, 4))

        # Check shape and dtype
        assert rust_result.shape == numpy_result.shape
        assert rust_result.dtype == numpy_result.dtype

        # Check values
        np.testing.assert_array_equal(rust_result, numpy_result)

    def test_ones_1d_complex(self):
        """Test 1D complex array creation."""
        # Rust implementation
        rust_result = ones(5, dtype="complex128")

        # NumPy reference
        numpy_result = np.ones(5, dtype=np.complex128)

        # Check shape and dtype
        assert rust_result.shape == numpy_result.shape
        assert rust_result.dtype == numpy_result.dtype

        # Check values
        np.testing.assert_array_equal(rust_result, numpy_result)

    def test_ones_2d_complex(self):
        """Test 2D complex array creation."""
        # Rust implementation
        rust_result = ones((3, 4), dtype="complex128")

        # NumPy reference
        numpy_result = np.ones((3, 4), dtype=np.complex128)

        # Check shape and dtype
        assert rust_result.shape == numpy_result.shape
        assert rust_result.dtype == numpy_result.dtype

        # Check values
        np.testing.assert_array_equal(rust_result, numpy_result)

    def test_ones_1d_int(self):
        """Test 1D integer array creation."""
        # Rust implementation
        rust_result = ones(5, dtype="int64")

        # NumPy reference
        numpy_result = np.ones(5, dtype=np.int64)

        # Check shape and dtype
        assert rust_result.shape == numpy_result.shape
        assert rust_result.dtype == numpy_result.dtype

        # Check values
        np.testing.assert_array_equal(rust_result, numpy_result)

    def test_ones_2d_int(self):
        """Test 2D integer array creation."""
        # Rust implementation
        rust_result = ones((3, 4), dtype="int64")

        # NumPy reference
        numpy_result = np.ones((3, 4), dtype=np.int64)

        # Check shape and dtype
        assert rust_result.shape == numpy_result.shape
        assert rust_result.dtype == numpy_result.dtype

        # Check values
        np.testing.assert_array_equal(rust_result, numpy_result)

    def test_ones_dtype_aliases(self):
        """Test that dtype aliases work (float, complex, int)."""
        # float alias
        result_float = ones(3, dtype="float")
        assert result_float.dtype == np.float64

        # complex alias
        result_complex = ones(3, dtype="complex")
        assert result_complex.dtype == np.complex128

        # int alias
        result_int = ones(3, dtype="int")
        assert result_int.dtype == np.int64

    def test_ones_shape_as_list(self):
        """Test that shape can be provided as a list."""
        # Shape as list
        rust_result = ones([3, 4])

        # NumPy reference
        numpy_result = np.ones((3, 4))

        # Check shape and dtype
        assert rust_result.shape == numpy_result.shape
        np.testing.assert_array_equal(rust_result, numpy_result)

    def test_ones_edge_cases(self):
        """Test edge cases like empty arrays."""
        # Empty 1D array
        result = ones(0)
        assert result.shape == (0,)
        assert len(result) == 0

        # Single element
        result = ones(1)
        assert result.shape == (1,)
        assert result[0] == 1.0


class TestZerosOnesInteraction:
    """Test that zeros() and ones() work well with other NumPy operations."""

    def test_zeros_plus_ones(self):
        """Test that zeros + ones = ones."""
        z = zeros(5)
        o = ones(5)
        result = z + o

        expected = np.ones(5)
        np.testing.assert_array_equal(result, expected)

    def test_zeros_complex_arithmetic(self):
        """Test complex number arithmetic with zeros."""
        z = zeros(3, dtype="complex128")
        o = ones(3, dtype="complex128")

        # zeros + ones should equal ones
        result = z + o
        np.testing.assert_array_equal(result, np.ones(3, dtype=np.complex128))

        # zeros * anything should be zeros
        result = z * (1 + 2j)
        np.testing.assert_array_equal(result, np.zeros(3, dtype=np.complex128))

    def test_zeros_ones_matrix_operations(self):
        """Test matrix operations with zeros and ones."""
        z = zeros((3, 3))
        o = ones((3, 3))

        # Matrix multiplication with zeros
        result = np.dot(z, o)
        np.testing.assert_array_equal(result, np.zeros((3, 3)))

        # Matrix addition
        result = z + o
        np.testing.assert_array_equal(result, np.ones((3, 3)))

    def test_import_from_pecos_num(self):
        """Test that zeros/ones can be imported from pecos.num."""
        from pecos.num import ones as pecos_ones
        from pecos.num import zeros as pecos_zeros

        # Test basic functionality
        z = pecos_zeros(5)
        o = pecos_ones(5)

        assert z.shape == (5,)
        assert o.shape == (5,)
        np.testing.assert_array_equal(z, np.zeros(5))
        np.testing.assert_array_equal(o, np.ones(5))
