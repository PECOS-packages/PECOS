"""Tests for polymorphic math functions (exp, cos, sin, isnan, isclose).

These tests verify that the polymorphic dispatch works correctly for:
- Scalar inputs (float, complex)
- Array inputs (numpy arrays of float and complex)
- List inputs (converted to arrays)
- Type checking and error handling
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from pecos_rslib import Array, array_equal, cos, exp, isclose, isnan, sin


class TestExpPolymorphic:
    """Test exp() with various input types."""

    def test_exp_scalar_float(self):
        """Test exp with scalar float input."""
        result = exp(1.0)
        assert isinstance(result, float)
        assert abs(result - math.e) < 1e-10

    def test_exp_scalar_complex(self):
        """Test exp with scalar complex input (Euler's formula)."""
        # e^(iπ) = -1
        result = exp(1j * math.pi)
        assert isinstance(result, complex)
        assert abs(result - (-1.0 + 0j)) < 1e-10

    def test_exp_array_float(self):
        """Test exp with float array input returns Array."""
        arr = np.array([0.0, 1.0, 2.0])
        result = exp(arr)
        assert isinstance(result, Array)
        assert str(result.dtype) == "float64"
        expected = np.array([1.0, math.e, math.e**2])
        assert np.allclose(result, expected)

    def test_exp_array_complex(self):
        """Test exp with complex array input returns Array."""
        arr = np.array([0 + 0j, 1j * math.pi, 2 + 0j])
        result = exp(arr)
        assert isinstance(result, Array)
        # exp(0) = 1, exp(iπ) = -1, exp(2) = e^2
        expected = np.array([1.0 + 0j, -1.0 + 0j, math.e**2 + 0j])
        assert np.allclose(result, expected, atol=1e-10)

    def test_exp_list_input(self):
        """Test exp accepts list input and returns Array."""
        result = exp([0.0, 1.0, 2.0])
        assert isinstance(result, Array)
        expected = np.array([1.0, math.e, math.e**2])
        assert np.allclose(result, expected)

    def test_exp_2d_array(self):
        """Test exp preserves 2D array shape."""
        arr = np.array([[0.0, 1.0], [2.0, 3.0]])
        result = exp(arr)
        assert result.shape == (2, 2)
        expected = np.exp(arr)
        assert np.allclose(result, expected)


class TestCosPolymorphic:
    """Test cos() with various input types."""

    def test_cos_scalar_float(self):
        """Test cos with scalar float input."""
        result = cos(0.0)
        assert isinstance(result, float)
        assert abs(result - 1.0) < 1e-10

        result_pi = cos(math.pi)
        assert abs(result_pi - (-1.0)) < 1e-10

    def test_cos_scalar_complex(self):
        """Test cos supports complex scalars."""
        # cos now supports complex numbers via ComplexFloat trait
        result = cos(0 + 0j)
        assert isinstance(result, complex)
        assert abs(result - 1.0) < 1e-10

    def test_cos_array_float(self):
        """Test cos with float array input returns Array."""
        arr = np.array([0.0, math.pi / 2, math.pi])
        result = cos(arr)
        assert isinstance(result, Array)
        assert str(result.dtype) == "float64"
        expected = np.array([1.0, 0.0, -1.0])
        assert np.allclose(result, expected, atol=1e-10)

    def test_cos_array_complex(self):
        """Test cos supports complex arrays."""
        arr_complex = np.array([0 + 0j, math.pi + 0j])
        result = cos(arr_complex)
        assert isinstance(result, Array)
        # cos(0) = 1, cos(π) = -1
        assert abs(result[0] - 1.0) < 1e-10
        assert abs(result[1] - (-1.0)) < 1e-10

    def test_cos_list_input(self):
        """Test cos accepts list input and returns Array."""
        result = cos([0.0, math.pi / 2, math.pi])
        assert isinstance(result, Array)
        expected = np.array([1.0, 0.0, -1.0])
        assert np.allclose(result, expected, atol=1e-10)

    def test_cos_2d_array(self):
        """Test cos preserves 2D array shape."""
        arr = np.array([[0.0, math.pi / 2], [math.pi, 2 * math.pi]])
        result = cos(arr)
        assert result.shape == (2, 2)
        expected = np.cos(arr)
        assert np.allclose(result, expected, atol=1e-10)


class TestSinPolymorphic:
    """Test sin() with various input types."""

    def test_sin_scalar_float(self):
        """Test sin with scalar float input."""
        result = sin(0.0)
        assert isinstance(result, float)
        assert abs(result - 0.0) < 1e-10

        result_pi2 = sin(math.pi / 2)
        assert abs(result_pi2 - 1.0) < 1e-10

    def test_sin_scalar_complex(self):
        """Test sin supports complex scalars."""
        # sin now supports complex numbers via ComplexFloat trait
        result = sin(0 + 0j)
        assert isinstance(result, complex)
        assert abs(result) < 1e-10

    def test_sin_array_float(self):
        """Test sin with float array input returns Array."""
        arr = np.array([0.0, math.pi / 2, math.pi])
        result = sin(arr)
        assert isinstance(result, Array)
        # Array.dtype returns a dtype object, check string representation
        assert str(result.dtype) == "float64"
        expected = np.array([0.0, 1.0, 0.0])
        assert np.allclose(result, expected, atol=1e-10)

    def test_sin_array_complex(self):
        """Test sin supports complex arrays."""
        arr_complex = np.array([0 + 0j, math.pi / 2 + 0j])
        result = sin(arr_complex)
        assert isinstance(result, Array)
        # Verify it computes correctly for complex input
        assert abs(result[0]) < 1e-10  # sin(0) = 0
        assert abs(result[1] - 1.0) < 1e-10  # sin(π/2) = 1

    def test_sin_list_input(self):
        """Test sin accepts list input and returns Array."""
        result = sin([0.0, math.pi / 2, math.pi])
        assert isinstance(result, Array)
        expected = np.array([0.0, 1.0, 0.0])
        assert np.allclose(result, expected, atol=1e-10)

    def test_sin_2d_array(self):
        """Test sin preserves 2D array shape."""
        arr = np.array([[0.0, math.pi / 2], [math.pi, 2 * math.pi]])
        result = sin(arr)
        # Array.shape returns a list, NumPy returns tuple
        assert result.shape == [2, 2] or result.shape == (2, 2)
        expected = np.sin(arr)
        assert np.allclose(result, expected, atol=1e-10)


class TestIsNanPolymorphic:
    """Test isnan() with various input types."""

    def test_isnan_scalar_normal(self):
        """Test isnan with normal scalar."""
        result = isnan(1.0)
        assert isinstance(result, bool)
        assert result is False

    def test_isnan_scalar_nan(self):
        """Test isnan with NaN scalar."""
        result = isnan(float("nan"))
        assert isinstance(result, bool)
        assert result is True

    def test_isnan_scalar_complex(self):
        """Test isnan with complex scalar."""
        result = isnan(1.0 + 2.0j)
        assert isinstance(result, bool)
        assert result is False

        result_nan = isnan(complex(float("nan"), 0))
        assert result_nan is True

    def test_isnan_array_float(self):
        """Test isnan with float array."""
        arr = np.array([1.0, float("nan"), 3.0])
        result = isnan(arr)
        # isnan returns BoolArrayView (bool array view)
        expected = np.array([False, True, False])
        assert array_equal(result, expected)

    def test_isnan_array_complex(self):
        """Test isnan with complex array."""
        arr = np.array([1.0 + 0j, complex(float("nan"), 0), 3.0 + 0j])
        result = isnan(arr)
        # isnan returns BoolArrayView (bool array view)
        expected = np.array([False, True, False])
        assert array_equal(result, expected)

    def test_isnan_list_input(self):
        """Test isnan accepts list input."""
        result = isnan([1.0, float("nan"), 3.0])
        # isnan returns BoolArrayView (bool array view)
        expected = np.array([False, True, False])
        assert array_equal(result, expected)

    def test_isnan_2d_array(self):
        """Test isnan preserves 2D array shape."""
        arr = np.array([[1.0, float("nan")], [3.0, 4.0]])
        result = isnan(arr)
        assert result.shape == [2, 2] or result.shape == (2, 2)
        expected = np.array([[False, True], [False, False]])
        assert array_equal(result, expected)


class TestIsClosePolymorphic:
    """Test isclose() with various input types."""

    def test_isclose_scalar_equal(self):
        """Test isclose with equal scalars."""
        result = isclose(1.0, 1.0)
        assert isinstance(result, bool)
        assert result is True

    def test_isclose_scalar_close(self):
        """Test isclose with close scalars."""
        result = isclose(1.0, 1.0 + 1e-9)
        assert result is True

        result_far = isclose(1.0, 1.1)
        assert result_far is False

    def test_isclose_scalar_complex(self):
        """Test isclose with complex scalars."""
        result = isclose(1.0 + 2.0j, 1.0 + 2.0j)
        assert isinstance(result, bool)
        assert result is True

        result_close = isclose(1.0 + 2.0j, 1.0 + 2.0j + 1e-10)
        assert result_close is True

    def test_isclose_array_float(self):
        """Test isclose with float arrays."""
        arr1 = np.array([1.0, 2.0, 3.0])
        arr2 = np.array([1.0, 2.0 + 1e-9, 3.1])
        result = isclose(arr1, arr2)
        # isclose now returns PECOS Array objects
        assert isinstance(result, Array)
        assert str(result.dtype) == "bool"
        expected = np.array([True, True, False])
        assert array_equal(result, expected)

    def test_isclose_array_complex(self):
        """Test isclose with complex arrays."""
        arr1 = np.array([1.0 + 0j, 2.0 + 1.0j])
        arr2 = np.array([1.0 + 0j, 2.0 + 1.0j + 1e-10])
        result = isclose(arr1, arr2)
        # isclose now returns PECOS Array objects
        assert isinstance(result, Array)
        assert str(result.dtype) == "bool"
        # Use array_equal or result.all() to check if all elements are True
        assert result.all()

    def test_isclose_list_input(self):
        """Test isclose requires numpy arrays (not lists)."""
        arr1 = np.array([1.0, 2.0])
        arr2 = np.array([1.0, 2.0 + 1e-9])
        result = isclose(arr1, arr2)
        # isclose now returns PECOS Array objects
        assert isinstance(result, Array)
        assert result.all()

    def test_isclose_2d_array(self):
        """Test isclose preserves 2D array shape."""
        arr1 = np.array([[1.0, 2.0], [3.0, 4.0]])
        arr2 = np.array([[1.0, 2.0 + 1e-9], [3.1, 4.0]])
        result = isclose(arr1, arr2)
        assert result.shape == (2, 2)
        expected = np.array([[True, True], [False, True]])
        assert array_equal(result, expected)

    def test_isclose_no_broadcasting(self):
        """Test isclose doesn't support scalar-array broadcasting."""
        # isclose requires both arguments to be same type (both scalars or both arrays)
        arr = np.array([1.0, 2.0, 3.0])
        scalar = 2.0
        with pytest.raises(TypeError, match="Input must be a numpy array"):
            isclose(arr, scalar)

        # Workaround: broadcast manually first
        result = isclose(arr, np.full_like(arr, scalar))
        expected = np.array([False, True, False])
        assert array_equal(result, expected)


class TestRealWorldUseCases:
    """Test polymorphic functions in realistic quantum simulation scenarios."""

    def test_quantum_gate_matrix_r1xy(self):
        """Test using exp, cos, sin for R1XY gate matrix construction."""
        # From find_cliffs.py
        theta = math.pi / 2  # 90 degree rotation
        phi = 0.0

        c = cos(theta * 0.5)
        s = sin(theta * 0.5)

        # Construct R1XY matrix elements
        elem_00 = c
        elem_01 = -1j * exp(-1j * phi) * s
        elem_10 = -1j * exp(1j * phi) * s
        elem_11 = c

        # Verify values
        assert isinstance(elem_00, float)
        assert abs(elem_00 - math.cos(math.pi / 4)) < 1e-10
        assert isinstance(elem_01, complex)
        assert isinstance(elem_10, complex)
        assert isinstance(elem_11, float)

    def test_quantum_gate_matrix_rz(self):
        """Test using exp for RZ gate matrix construction."""
        # From find_cliffs.py
        theta = math.pi

        elem_00 = exp(-1j * theta * 0.5)
        elem_11 = exp(1j * theta * 0.5)

        # Verify these are on the unit circle
        assert abs(abs(elem_00) - 1.0) < 1e-10
        assert abs(abs(elem_11) - 1.0) < 1e-10

        # exp(-iπ/2) should equal -i
        assert abs(elem_00 - (-1j)) < 1e-10
        # exp(iπ/2) should equal i
        assert abs(elem_11 - 1j) < 1e-10

    def test_error_filtering_with_isnan(self):
        """Test using isnan to filter invalid simulation results."""
        # Simulate some computation results with potential NaNs
        results = np.array([0.95, 0.98, float("nan"), 0.97, 0.96])

        # Filter out NaN values - convert to NumPy for indexing
        valid_mask = ~np.array(isnan(results))
        valid_results = results[valid_mask]

        assert len(valid_results) == 4
        assert not np.any(np.isnan(valid_results))

    def test_threshold_comparison_with_isclose(self):
        """Test using isclose for threshold convergence checks."""
        # Simulate iterative threshold fitting
        old_threshold = 0.01234567
        new_threshold = 0.01234568

        # Check if converged (within tolerance)
        converged = isclose(old_threshold, new_threshold, rtol=1e-6, atol=1e-8)

        assert isinstance(converged, bool)
        assert converged is True


class TestIsCloseMixedTypes:
    """Test isclose() with mixed complex/float types to match NumPy behavior.

    NumPy's isclose uses magnitude-based comparison for complex numbers:
        |a - b| <= (atol + rtol * |b|)
    where |z| = sqrt(real² + imag²) for complex z.
    """

    def test_isclose_complex_vs_real_zero(self):
        """Test pure imaginary vs real zero."""
        # This was the original bug: isclose(1j, 0.0) should be False
        result = isclose(1j, 0.0, rtol=1e-5, atol=1e-12)
        expected = np.isclose(1j, 0.0, rtol=1e-5, atol=1e-12)
        assert result == expected
        assert result is False  # |1j - 0| = 1.0, threshold = 1e-12

    def test_isclose_real_vs_complex_zero(self):
        """Test real zero vs pure imaginary."""
        result = isclose(0.0, 1j, rtol=1e-5, atol=1e-12)
        expected = np.isclose(0.0, 1j, rtol=1e-5, atol=1e-12)
        assert result == expected
        assert result is False

    def test_isclose_small_imaginary_vs_real(self):
        """Test small imaginary part vs real number."""
        result = isclose(1.0 + 1e-9j, 1.0, rtol=1e-5, atol=1e-8)
        expected = np.isclose(1.0 + 1e-9j, 1.0, rtol=1e-5, atol=1e-8)
        assert result == expected
        assert result is True  # |1e-9j| = 1e-9, threshold ≈ 1e-5

    def test_isclose_real_vs_small_imaginary(self):
        """Test real number vs small imaginary part."""
        result = isclose(1.0, 1.0 + 1e-9j, rtol=1e-5, atol=1e-8)
        expected = np.isclose(1.0, 1.0 + 1e-9j, rtol=1e-5, atol=1e-8)
        assert result == expected
        assert result is True

    def test_isclose_magnitude_based_not_component_wise(self):
        """Test that comparison is magnitude-based, not component-wise."""
        # Both components differ by 0.01
        # Component-wise rtol=0.01 would pass (each component within 1%)
        # But magnitude-based: |diff| = sqrt(0.01² + 0.01²) ≈ 0.0141
        # threshold = 0 + 0.01 * sqrt(1.01² + 1.01²) ≈ 0.0143
        a = 1.0 + 1.0j
        b = 1.01 + 1.01j
        result = isclose(a, b, rtol=0.01, atol=0)
        expected = np.isclose(a, b, rtol=0.01, atol=0)
        assert result == expected
        # Magnitude-based should pass
        assert result is True

    def test_isclose_complex_arrays_mixed_types(self):
        """Test isclose with arrays of mixed complex/float."""
        # Array with both complex and real-valued elements
        arr1 = np.array([1.0 + 0j, 2.0 + 1j, 3.0 + 0j])
        arr2 = np.array([1.0, 2.0 + 1j + 1e-10, 3.0])

        result = isclose(arr1, arr2)
        expected = np.isclose(arr1, arr2)
        assert array_equal(result, expected)
        assert np.all(result)

    def test_isclose_pure_imaginary_array(self):
        """Test isclose with pure imaginary numbers."""
        arr1 = np.array([1j, 2j, 3j])
        arr2 = np.array([1j + 1e-10, 2j, 3j + 1e-5])

        result = isclose(arr1, arr2, rtol=1e-5, atol=1e-8)
        expected = np.isclose(arr1, arr2, rtol=1e-5, atol=1e-8)
        assert array_equal(result, expected)

    def test_isclose_quantum_gate_comparison(self):
        """Test realistic quantum gate matrix element comparison."""
        # exp(iπ/4) vs manually computed value
        angle = math.pi / 4
        elem1 = exp(1j * angle)
        elem2 = complex(math.cos(angle), math.sin(angle))

        result = isclose(elem1, elem2, rtol=1e-10, atol=1e-12)
        assert result is True

        # Verify against NumPy
        expected = np.isclose(elem1, elem2, rtol=1e-10, atol=1e-12)
        assert result == expected


class TestIsCloseMixedArrays:
    """Test isclose() with mixed float/complex array types.

    NumPy seamlessly handles mixed array types by promoting floats to complex.
    Our implementation should do the same.
    """

    def test_complex_array_vs_float_array(self):
        """Test complex array vs float array."""
        arr1 = np.array([1 + 0j, 2 + 0j, 3 + 0j], dtype=complex)
        arr2 = np.array([1.0, 2.0, 3.0], dtype=float)

        result = isclose(arr1, arr2)
        expected = np.isclose(arr1, arr2)
        assert array_equal(result, expected)
        assert np.all(result)

    def test_float_array_vs_complex_array(self):
        """Test float array vs complex array."""
        arr1 = np.array([1.0, 2.0, 3.0], dtype=float)
        arr2 = np.array([1 + 0j, 2 + 0j, 3 + 0j], dtype=complex)

        result = isclose(arr1, arr2)
        expected = np.isclose(arr1, arr2)
        assert array_equal(result, expected)
        assert np.all(result)

    def test_pure_imaginary_vs_float_zero(self):
        """Test pure imaginary array vs float zero array."""
        arr1 = np.array([1j, 2j, 3j], dtype=complex)
        arr2 = np.array([0.0, 0.0, 0.0], dtype=float)

        result = isclose(arr1, arr2, rtol=1e-5, atol=1e-12)
        expected = np.isclose(arr1, arr2, rtol=1e-5, atol=1e-12)
        assert array_equal(result, expected)
        assert not np.any(result)  # All should be False

    def test_float_zero_vs_pure_imaginary(self):
        """Test float zero array vs pure imaginary array."""
        arr1 = np.array([0.0, 0.0, 0.0], dtype=float)
        arr2 = np.array([1j, 2j, 3j], dtype=complex)

        result = isclose(arr1, arr2, rtol=1e-5, atol=1e-12)
        expected = np.isclose(arr1, arr2, rtol=1e-5, atol=1e-12)
        assert array_equal(result, expected)
        assert not np.any(result)  # All should be False

    def test_float_vs_small_imaginary_array(self):
        """Test float array vs complex with small imaginary parts."""
        arr1 = np.array([1.0, 2.0, 3.0], dtype=float)
        arr2 = np.array([1 + 1e-9j, 2 + 1e-9j, 3 + 1e-9j], dtype=complex)

        result = isclose(arr1, arr2)
        expected = np.isclose(arr1, arr2)
        assert array_equal(result, expected)
        assert np.all(result)  # All should be True

    def test_2d_mixed_arrays(self):
        """Test 2D mixed arrays."""
        arr1 = np.array([[1 + 0j, 2 + 0j], [3 + 0j, 4 + 0j]], dtype=complex)
        arr2 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)

        result = isclose(arr1, arr2)
        expected = np.isclose(arr1, arr2)
        assert array_equal(result, expected)
        assert result.shape == (2, 2)
        assert np.all(result)

    def test_mixed_with_differences(self):
        """Test mixed arrays where some elements differ."""
        arr1 = np.array([1.0, 2.0, 3.0], dtype=float)
        arr2 = np.array([1 + 1e-9j, 2.1 + 0j, 3 + 1e-9j], dtype=complex)

        result = isclose(arr1, arr2)
        expected = np.isclose(arr1, arr2)
        assert array_equal(result, expected)
        # First and third should be close, middle should be far
        assert result[0]
        assert not result[1]
        assert result[2]
