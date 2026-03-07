"""Tests for zeros() and ones() functions.

This module tests the Rust implementations of zeros() and ones() against NumPy
to ensure they are drop-in replacements.
"""

import numpy as np

import pecos as pc


class TestZeros:
    """Test zeros() function against numpy.zeros()."""

    def test_zeros_1d_float(self) -> None:
        """Test 1D float array creation."""
        # Rust implementation
        rust_result = pc.zeros(5)

        # NumPy reference
        numpy_result = np.zeros(5)

        # Check shape and dtype
        assert rust_result.shape == numpy_result.shape
        assert rust_result.dtype == pc.dtypes.f64

        # Check values
        np.testing.assert_array_equal(rust_result, numpy_result)

    def test_zeros_2d_float(self) -> None:
        """Test 2D float array creation."""
        # Rust implementation
        rust_result = pc.zeros((3, 4))

        # NumPy reference
        numpy_result = np.zeros((3, 4))

        # Check shape and dtype
        assert rust_result.shape == numpy_result.shape
        assert rust_result.dtype == pc.dtypes.f64

        # Check values
        np.testing.assert_array_equal(rust_result, numpy_result)

    def test_zeros_3d_float(self) -> None:
        """Test 3D float array creation."""
        # Rust implementation
        rust_result = pc.zeros((2, 3, 4))

        # NumPy reference
        numpy_result = np.zeros((2, 3, 4))

        # Check shape and dtype
        assert rust_result.shape == numpy_result.shape
        assert rust_result.dtype == pc.dtypes.f64

        # Check values
        np.testing.assert_array_equal(rust_result, numpy_result)

    def test_zeros_1d_complex(self) -> None:
        """Test 1D complex array creation."""
        # Rust implementation
        rust_result = pc.zeros(5, dtype="complex128")

        # NumPy reference
        numpy_result = np.zeros(5, dtype=np.complex128)

        # Check shape and dtype
        assert rust_result.shape == numpy_result.shape
        assert rust_result.dtype == pc.dtypes.complex128

        # Check values
        np.testing.assert_array_equal(rust_result, numpy_result)

    def test_zeros_2d_complex(self) -> None:
        """Test 2D complex array creation."""
        # Rust implementation
        rust_result = pc.zeros((3, 4), dtype="complex128")

        # NumPy reference
        numpy_result = np.zeros((3, 4), dtype=np.complex128)

        # Check shape and dtype
        assert rust_result.shape == numpy_result.shape
        assert rust_result.dtype == pc.dtypes.complex128

        # Check values
        np.testing.assert_array_equal(rust_result, numpy_result)

    def test_zeros_1d_int(self) -> None:
        """Test 1D integer array creation."""
        # Rust implementation
        rust_result = pc.zeros(5, dtype="int64")

        # NumPy reference
        numpy_result = np.zeros(5, dtype=np.int64)

        # Check shape and dtype
        assert rust_result.shape == numpy_result.shape
        assert rust_result.dtype == pc.dtypes.i64

        # Check values
        np.testing.assert_array_equal(rust_result, numpy_result)

    def test_zeros_2d_int(self) -> None:
        """Test 2D integer array creation."""
        # Rust implementation
        rust_result = pc.zeros((3, 4), dtype="int64")

        # NumPy reference
        numpy_result = np.zeros((3, 4), dtype=np.int64)

        # Check shape and dtype
        assert rust_result.shape == numpy_result.shape
        assert rust_result.dtype == pc.dtypes.i64

        # Check values
        np.testing.assert_array_equal(rust_result, numpy_result)

    def test_zeros_dtype_aliases(self) -> None:
        """Test that dtype aliases work (float, complex, int)."""
        # float alias
        result_float = pc.zeros(3, dtype="float")
        assert result_float.dtype == pc.dtypes.f64

        # complex alias
        result_complex = pc.zeros(3, dtype="complex")
        assert result_complex.dtype == pc.dtypes.complex128

        # int alias
        result_int = pc.zeros(3, dtype="int")
        assert result_int.dtype == pc.dtypes.i64

    def test_zeros_shape_as_list(self) -> None:
        """Test that shape can be provided as a list."""
        # Shape as list
        rust_result = pc.zeros([3, 4])

        # NumPy reference
        numpy_result = np.zeros((3, 4))

        # Check shape and dtype
        assert rust_result.shape == numpy_result.shape
        np.testing.assert_array_equal(rust_result, numpy_result)

    def test_zeros_edge_cases(self) -> None:
        """Test edge cases like empty arrays."""
        # Empty 1D array
        result = pc.zeros(0)
        assert result.shape == (0,)
        assert len(result) == 0

        # Single element
        result = pc.zeros(1)
        assert result.shape == (1,)
        assert result[0] == 0.0


class TestOnes:
    """Test ones() function against numpy.ones()."""

    def test_ones_1d_float(self) -> None:
        """Test 1D float array creation."""
        # Rust implementation
        rust_result = pc.ones(5)

        # NumPy reference
        numpy_result = np.ones(5)

        # Check shape and dtype
        assert rust_result.shape == numpy_result.shape
        assert rust_result.dtype == pc.dtypes.f64

        # Check values
        np.testing.assert_array_equal(rust_result, numpy_result)

    def test_ones_2d_float(self) -> None:
        """Test 2D float array creation."""
        # Rust implementation
        rust_result = pc.ones((3, 4))

        # NumPy reference
        numpy_result = np.ones((3, 4))

        # Check shape and dtype
        assert rust_result.shape == numpy_result.shape
        assert rust_result.dtype == pc.dtypes.f64

        # Check values
        np.testing.assert_array_equal(rust_result, numpy_result)

    def test_ones_3d_float(self) -> None:
        """Test 3D float array creation."""
        # Rust implementation
        rust_result = pc.ones((2, 3, 4))

        # NumPy reference
        numpy_result = np.ones((2, 3, 4))

        # Check shape and dtype
        assert rust_result.shape == numpy_result.shape
        assert rust_result.dtype == pc.dtypes.f64

        # Check values
        np.testing.assert_array_equal(rust_result, numpy_result)

    def test_ones_1d_complex(self) -> None:
        """Test 1D complex array creation."""
        # Rust implementation
        rust_result = pc.ones(5, dtype="complex128")

        # NumPy reference
        numpy_result = np.ones(5, dtype=np.complex128)

        # Check shape and dtype
        assert rust_result.shape == numpy_result.shape
        assert rust_result.dtype == pc.dtypes.complex128

        # Check values
        np.testing.assert_array_equal(rust_result, numpy_result)

    def test_ones_2d_complex(self) -> None:
        """Test 2D complex array creation."""
        # Rust implementation
        rust_result = pc.ones((3, 4), dtype="complex128")

        # NumPy reference
        numpy_result = np.ones((3, 4), dtype=np.complex128)

        # Check shape and dtype
        assert rust_result.shape == numpy_result.shape
        assert rust_result.dtype == pc.dtypes.complex128

        # Check values
        np.testing.assert_array_equal(rust_result, numpy_result)

    def test_ones_1d_int(self) -> None:
        """Test 1D integer array creation."""
        # Rust implementation
        rust_result = pc.ones(5, dtype="int64")

        # NumPy reference
        numpy_result = np.ones(5, dtype=np.int64)

        # Check shape and dtype
        assert rust_result.shape == numpy_result.shape
        assert rust_result.dtype == pc.dtypes.i64

        # Check values
        np.testing.assert_array_equal(rust_result, numpy_result)

    def test_ones_2d_int(self) -> None:
        """Test 2D integer array creation."""
        # Rust implementation
        rust_result = pc.ones((3, 4), dtype="int64")

        # NumPy reference
        numpy_result = np.ones((3, 4), dtype=np.int64)

        # Check shape and dtype
        assert rust_result.shape == numpy_result.shape
        assert rust_result.dtype == pc.dtypes.i64

        # Check values
        np.testing.assert_array_equal(rust_result, numpy_result)

    def test_ones_dtype_aliases(self) -> None:
        """Test that dtype aliases work (float, complex, int)."""
        # float alias
        result_float = pc.ones(3, dtype="float")
        assert result_float.dtype == pc.dtypes.f64

        # complex alias
        result_complex = pc.ones(3, dtype="complex")
        assert result_complex.dtype == pc.dtypes.complex128

        # int alias
        result_int = pc.ones(3, dtype="int")
        assert result_int.dtype == pc.dtypes.i64

    def test_ones_shape_as_list(self) -> None:
        """Test that shape can be provided as a list."""
        # Shape as list
        rust_result = pc.ones([3, 4])

        # NumPy reference
        numpy_result = np.ones((3, 4))

        # Check shape and dtype
        assert rust_result.shape == numpy_result.shape
        np.testing.assert_array_equal(rust_result, numpy_result)

    def test_ones_edge_cases(self) -> None:
        """Test edge cases like empty arrays."""
        # Empty 1D array
        result = pc.ones(0)
        assert result.shape == (0,)
        assert len(result) == 0

        # Single element
        result = pc.ones(1)
        assert result.shape == (1,)
        assert result[0] == 1.0


class TestZerosOnesInteraction:
    """Test that zeros() and ones() work well with other NumPy operations."""

    def test_zeros_plus_ones(self) -> None:
        """Test that zeros + ones = ones."""
        z = pc.zeros(5)
        o = pc.ones(5)
        result = z + o

        expected = np.ones(5)
        np.testing.assert_array_equal(result, expected)

    def test_zeros_complex_arithmetic(self) -> None:
        """Test complex number arithmetic with zeros."""
        z = pc.zeros(3, dtype="complex128")
        o = pc.ones(3, dtype="complex128")

        # zeros + ones should equal ones
        result = z + o
        np.testing.assert_array_equal(result, np.ones(3, dtype=np.complex128))

        # zeros * anything should be zeros
        result = z * (1 + 2j)
        np.testing.assert_array_equal(result, np.zeros(3, dtype=np.complex128))

    def test_zeros_ones_matrix_operations(self) -> None:
        """Test matrix operations with zeros and ones."""
        z = pc.zeros((3, 3))
        o = pc.ones((3, 3))

        # Matrix multiplication with zeros
        result = np.dot(z, o)
        np.testing.assert_array_equal(result, np.zeros((3, 3)))

        # Matrix addition
        result = z + o
        np.testing.assert_array_equal(result, np.ones((3, 3)))

    def test_import_from_pecos(self) -> None:
        """Test that zeros/ones can be imported from pecos."""
        # Already imported at top: import pecos as pc

        # Test basic functionality
        z = pc.zeros(5)
        o = pc.ones(5)

        assert z.shape == (5,)
        assert o.shape == (5,)
        np.testing.assert_array_equal(z, np.zeros(5))
        np.testing.assert_array_equal(o, np.ones(5))
