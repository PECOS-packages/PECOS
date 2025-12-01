# Copyright 2025 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.

"""Tests for newly implemented NumPy replacement features.

This module tests:
1. Boolean array support in sum()
2. asarray() function (copy avoidance)
3. assert_allclose() function (detailed error messages)
"""

import numpy as np
import pytest
from _pecos_rslib.num import array, asarray, assert_allclose, sum as pecos_sum


class TestBooleanSum:
    """Test sum() function with boolean arrays."""

    def test_sum_bool_1d_basic(self) -> None:
        """Test basic 1D boolean array sum."""
        arr = array([True, False, True, True, False])
        result = pecos_sum(arr)
        assert result == 3

    def test_sum_bool_1d_all_true(self) -> None:
        """Test sum with all True values."""
        arr = array([True, True, True, True])
        result = pecos_sum(arr)
        assert result == 4

    def test_sum_bool_1d_all_false(self) -> None:
        """Test sum with all False values."""
        arr = array([False, False, False])
        result = pecos_sum(arr)
        assert result == 0

    def test_sum_bool_1d_empty(self) -> None:
        """Test sum with empty boolean array."""
        arr = array([], dtype="bool")
        result = pecos_sum(arr)
        assert result == 0

    def test_sum_bool_2d_no_axis(self) -> None:
        """Test 2D boolean array sum without axis parameter."""
        # Note: sum() currently requires NumPy arrays for multidimensional boolean arrays
        arr = np.array([[True, False, True], [False, True, False]])
        result = pecos_sum(arr)
        assert result == 3

    def test_sum_bool_2d_axis_0(self) -> None:
        """Test 2D boolean array sum along axis 0."""
        arr = array([[True, False, True], [False, True, False]], dtype="bool")
        result = pecos_sum(arr, axis=0)
        # Should sum columns: [1, 1, 1]
        expected = np.array([1, 1, 1])
        np.testing.assert_array_equal(result, expected)

    def test_sum_bool_2d_axis_1(self) -> None:
        """Test 2D boolean array sum along axis 1."""
        arr = array([[True, False, True], [False, True, False]], dtype="bool")
        result = pecos_sum(arr, axis=1)
        # Should sum rows: [2, 1]
        expected = np.array([2, 1])
        np.testing.assert_array_equal(result, expected)

    def test_sum_bool_3d_axis_0(self) -> None:
        """Test 3D boolean array sum along axis 0."""
        arr = array(
            [[[True, False], [False, True]], [[True, True], [False, False]]],
            dtype="bool",
        )
        result = pecos_sum(arr, axis=0)
        expected = np.array([[2, 1], [0, 1]])
        np.testing.assert_array_equal(result, expected)

    def test_sum_bool_3d_axis_1(self) -> None:
        """Test 3D boolean array sum along axis 1."""
        arr = array(
            [[[True, False], [False, True]], [[True, True], [False, False]]],
            dtype="bool",
        )
        result = pecos_sum(arr, axis=1)
        expected = np.array([[1, 1], [1, 1]])
        np.testing.assert_array_equal(result, expected)

    def test_sum_bool_3d_axis_2(self) -> None:
        """Test 3D boolean array sum along axis 2."""
        arr = array(
            [[[True, False], [False, True]], [[True, True], [False, False]]],
            dtype="bool",
        )
        result = pecos_sum(arr, axis=2)
        expected = np.array([[1, 1], [2, 0]])
        np.testing.assert_array_equal(result, expected)

    def test_sum_bool_negative_axis(self) -> None:
        """Test boolean sum with negative axis."""
        arr = array([[True, False, True], [False, True, False]], dtype="bool")
        result = pecos_sum(arr, axis=-1)
        expected = np.array([2, 1])
        np.testing.assert_array_equal(result, expected)

    def test_sum_bool_comparison_with_numpy(self) -> None:
        """Test that boolean sum matches NumPy behavior."""
        np_arr = np.array([True, False, True, True, False])
        pecos_arr = array([True, False, True, True, False])

        np_result = np.sum(np_arr)
        pecos_result = pecos_sum(pecos_arr)

        assert pecos_result == np_result

    def test_sum_bool_2d_comparison_with_numpy(self) -> None:
        """Test that 2D boolean sum matches NumPy behavior."""
        # Note: sum() currently requires NumPy arrays for multidimensional boolean arrays
        np_arr = np.array([[True, False, True], [False, True, False]])

        # Test axis=None
        np_result = np.sum(np_arr)
        pecos_result = pecos_sum(np_arr)
        assert pecos_result == np_result

        # Test axis=0
        np_result_0 = np.sum(np_arr, axis=0)
        pecos_result_0 = pecos_sum(np_arr, axis=0)
        np.testing.assert_array_equal(pecos_result_0, np_result_0)

        # Test axis=1
        np_result_1 = np.sum(np_arr, axis=1)
        pecos_result_1 = pecos_sum(np_arr, axis=1)
        np.testing.assert_array_equal(pecos_result_1, np_result_1)


class TestAsarray:
    """Test asarray() function for copy avoidance."""

    def test_asarray_from_list(self) -> None:
        """Test asarray creates array from list."""
        result = asarray([1.0, 2.0, 3.0])
        expected = array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(result, expected)

    def test_asarray_from_tuple(self) -> None:
        """Test asarray creates array from tuple."""
        result = asarray((1.0, 2.0, 3.0))
        expected = array((1.0, 2.0, 3.0))
        np.testing.assert_allclose(result, expected)

    def test_asarray_from_numpy_array(self) -> None:
        """Test asarray creates array from NumPy array."""
        np_arr = np.array([1.0, 2.0, 3.0])
        result = asarray(np_arr)
        np.testing.assert_allclose(result, np_arr)

    def test_asarray_no_copy_same_dtype(self) -> None:
        """Test asarray doesn't copy when dtype matches."""
        original = array([1.0, 2.0, 3.0])
        result = asarray(original)

        # Should be the same object (no copy)
        assert result is original

    def test_asarray_no_copy_no_dtype_param(self) -> None:
        """Test asarray doesn't copy when no dtype specified."""
        original = array([1, 2, 3], dtype="int64")
        result = asarray(original)

        # Should be the same object (no copy)
        assert result is original

    def test_asarray_copy_different_dtype(self) -> None:
        """Test asarray copies when dtype conversion needed."""
        original = array([1.0, 2.0, 3.0], dtype="float64")
        result = asarray(original, dtype="int64")

        # Should be different objects (copy occurred)
        assert result is not original

        # Values should be converted
        expected = array([1, 2, 3], dtype="int64")
        np.testing.assert_array_equal(result, expected)

    def test_asarray_f64_to_i64_conversion(self) -> None:
        """Test asarray converts float64 to int64."""
        original = array([1.5, 2.7, 3.2], dtype="float64")
        result = asarray(original, dtype="int64")

        assert result is not original
        expected = array([1, 2, 3], dtype="int64")
        np.testing.assert_array_equal(result, expected)

    def test_asarray_i64_to_f64_conversion(self) -> None:
        """Test asarray converts int64 to float64."""
        original = array([1, 2, 3], dtype="int64")
        result = asarray(original, dtype="float64")

        assert result is not original
        expected = array([1.0, 2.0, 3.0], dtype="float64")
        np.testing.assert_allclose(result, expected)

    def test_asarray_2d_no_copy(self) -> None:
        """Test asarray doesn't copy 2D arrays when dtype matches."""
        original = array([[1.0, 2.0], [3.0, 4.0]], dtype="float64")
        result = asarray(original)

        assert result is original

    def test_asarray_2d_with_conversion(self) -> None:
        """Test asarray copies 2D arrays when dtype conversion needed."""
        original = array([[1.0, 2.0], [3.0, 4.0]], dtype="float64")
        result = asarray(original, dtype="int64")

        assert result is not original
        expected = array([[1, 2], [3, 4]], dtype="int64")
        np.testing.assert_array_equal(result, expected)

    def test_asarray_complex_no_copy(self) -> None:
        """Test asarray doesn't copy complex arrays when dtype matches."""
        original = array([1 + 2j, 3 + 4j], dtype="complex128")
        result = asarray(original)

        assert result is original

    def test_asarray_bool_no_copy(self) -> None:
        """Test asarray doesn't copy boolean arrays when dtype matches."""
        original = array([True, False, True], dtype="bool")
        result = asarray(original)

        assert result is original

    def test_asarray_vs_array_copy_behavior(self) -> None:
        """Test that asarray() avoids copies while array() always copies."""
        original = array([1.0, 2.0, 3.0])

        # asarray should NOT copy
        asarray_result = asarray(original)
        assert asarray_result is original

        # array should ALWAYS copy
        array_result = array(original)
        assert array_result is not original


class TestAssertAllclose:
    """Test assert_allclose() function for detailed error messages."""

    def test_assert_allclose_exact_match(self) -> None:
        """Test assert_allclose passes with exact match."""
        a = array([1.0, 2.0, 3.0])
        b = array([1.0, 2.0, 3.0])

        # Should not raise
        assert_allclose(a, b)

    def test_assert_allclose_within_tolerance(self) -> None:
        """Test assert_allclose passes when values within tolerance."""
        a = array([1.0, 2.0, 3.0])
        b = array([1.00001, 2.00001, 3.00001])

        # Should not raise with default tolerances
        assert_allclose(a, b, rtol=1e-4, atol=1e-8)

    def test_assert_allclose_fails_outside_tolerance(self) -> None:
        """Test assert_allclose raises when values outside tolerance."""
        a = array([1.0, 2.0, 3.0])
        b = array([1.0, 2.0, 4.0])

        with pytest.raises(AssertionError) as exc_info:
            assert_allclose(a, b, rtol=1e-5, atol=1e-8)

        # Check that error message contains useful info
        error_msg = str(exc_info.value)
        assert "Not equal to tolerance" in error_msg
        assert "Mismatched elements" in error_msg
        assert "Max absolute difference" in error_msg
        assert "Max relative difference" in error_msg

    def test_assert_allclose_error_shows_tolerances(self) -> None:
        """Test error message shows the tolerances used."""
        a = array([1.0, 2.0])
        b = array([1.5, 2.5])

        with pytest.raises(AssertionError) as exc_info:
            assert_allclose(a, b, rtol=1e-3, atol=1e-6)

        error_msg = str(exc_info.value)
        assert "rtol=0.001" in error_msg
        assert "atol=0.000001" in error_msg or "atol=1e-06" in error_msg

    def test_assert_allclose_error_shows_mismatch_count(self) -> None:
        """Test error message shows number of mismatched elements."""
        a = array([1.0, 2.0, 3.0, 4.0])
        b = array([1.0, 2.5, 3.5, 4.0])

        with pytest.raises(AssertionError) as exc_info:
            assert_allclose(a, b, rtol=1e-5, atol=1e-8)

        error_msg = str(exc_info.value)
        assert "2 / 4" in error_msg  # 2 mismatched out of 4 total

    def test_assert_allclose_error_shows_first_mismatch(self) -> None:
        """Test error message shows first mismatch values."""
        a = array([1.0, 2.0, 3.0])
        b = array([1.0, 2.5, 3.5])

        with pytest.raises(AssertionError) as exc_info:
            assert_allclose(a, b, rtol=1e-5, atol=1e-8)

        error_msg = str(exc_info.value)
        assert "First mismatch" in error_msg
        # Should show the first mismatched values
        assert "2.0" in error_msg or "2." in error_msg
        assert "2.5" in error_msg

    def test_assert_allclose_shape_mismatch(self) -> None:
        """Test assert_allclose raises on shape mismatch."""
        a = array([1.0, 2.0, 3.0])
        b = array([1.0, 2.0])

        with pytest.raises(AssertionError) as exc_info:
            assert_allclose(a, b)

        error_msg = str(exc_info.value)
        assert "shape" in error_msg.lower()

    def test_assert_allclose_2d_arrays(self) -> None:
        """Test assert_allclose works with 2D arrays."""
        a = array([[1.0, 2.0], [3.0, 4.0]])
        b = array([[1.00001, 2.00001], [3.00001, 4.00001]])

        # Should not raise
        assert_allclose(a, b, rtol=1e-4, atol=1e-8)

    def test_assert_allclose_2d_arrays_fail(self) -> None:
        """Test assert_allclose fails correctly with 2D arrays."""
        a = array([[1.0, 2.0], [3.0, 4.0]])
        b = array([[1.0, 2.0], [3.0, 5.0]])

        with pytest.raises(AssertionError) as exc_info:
            assert_allclose(a, b, rtol=1e-5, atol=1e-8)

        error_msg = str(exc_info.value)
        assert "Mismatched elements: 1 / 4" in error_msg

    def test_assert_allclose_complex_arrays(self) -> None:
        """Test assert_allclose works with complex arrays."""
        a = array([1 + 2j, 3 + 4j])
        b = array([1.00001 + 2.00001j, 3.00001 + 4.00001j])

        # Should not raise
        assert_allclose(a, b, rtol=1e-4, atol=1e-8)

    def test_assert_allclose_complex_arrays_fail(self) -> None:
        """Test assert_allclose fails correctly with complex arrays."""
        a = array([1 + 2j, 3 + 4j])
        b = array([1 + 2j, 3 + 5j])

        with pytest.raises(AssertionError):
            assert_allclose(a, b, rtol=1e-5, atol=1e-8)

    def test_assert_allclose_mixed_real_complex(self) -> None:
        """Test assert_allclose with mixed real/complex arrays."""
        a = array([1.0, 2.0, 3.0])
        b = array([1.0 + 0j, 2.0 + 0j, 3.0 + 0j])

        # Should not raise - real numbers can be compared to complex
        assert_allclose(a, b, rtol=1e-5, atol=1e-8)

    def test_assert_allclose_nan_equal_nan_false(self) -> None:
        """Test assert_allclose fails when NaNs present (equal_nan=False)."""
        a = array([1.0, float("nan"), 3.0])
        b = array([1.0, float("nan"), 3.0])

        with pytest.raises(AssertionError):
            assert_allclose(a, b, equal_nan=False)

    def test_assert_allclose_nan_equal_nan_true(self) -> None:
        """Test assert_allclose passes when NaNs in same position (equal_nan=True)."""
        a = array([1.0, float("nan"), 3.0])
        b = array([1.0, float("nan"), 3.0])

        # Should not raise
        assert_allclose(a, b, equal_nan=True)

    def test_assert_allclose_different_nans_positions(self) -> None:
        """Test assert_allclose fails when NaNs in different positions."""
        a = array([1.0, float("nan"), 3.0])
        b = array([1.0, 2.0, float("nan")])

        with pytest.raises(AssertionError):
            assert_allclose(a, b, equal_nan=True)

    def test_assert_allclose_default_tolerances(self) -> None:
        """Test assert_allclose uses correct default tolerances."""
        a = array([1.0, 2.0])
        b = array([1.000001, 2.000001])

        # Should pass with default rtol=1e-5, atol=1e-8
        assert_allclose(a, b)

    def test_assert_allclose_strict_tolerance(self) -> None:
        """Test assert_allclose with very strict tolerance."""
        a = array([1.0, 2.0])
        b = array([1.0000001, 2.0000001])

        # Should fail with rtol=1e-8
        with pytest.raises(AssertionError):
            assert_allclose(a, b, rtol=1e-8, atol=1e-10)

    def test_assert_allclose_loose_tolerance(self) -> None:
        """Test assert_allclose with loose tolerance."""
        a = array([1.0, 2.0])
        b = array([1.01, 2.01])

        # Should pass with rtol=1e-2
        assert_allclose(a, b, rtol=1e-2, atol=1e-8)

    def test_assert_allclose_zero_values(self) -> None:
        """Test assert_allclose handles zero values correctly."""
        a = array([0.0, 1.0, 2.0])
        b = array([0.0, 1.0, 2.0])

        # Should pass
        assert_allclose(a, b)

    def test_assert_allclose_near_zero_absolute_tolerance(self) -> None:
        """Test assert_allclose uses absolute tolerance near zero."""
        a = array([0.0, 1e-10])
        b = array([1e-9, 2e-10])

        # Should pass with atol=1e-8
        assert_allclose(a, b, rtol=1e-5, atol=1e-8)

    def test_assert_allclose_large_values(self) -> None:
        """Test assert_allclose with large values uses relative tolerance."""
        a = array([1e10, 2e10])
        b = array([1e10 + 1e5, 2e10 + 2e5])

        # Should pass - 1e5 difference is small relative to 1e10
        assert_allclose(a, b, rtol=1e-4, atol=1e-8)

    def test_assert_allclose_numpy_array_inputs(self) -> None:
        """Test assert_allclose accepts NumPy arrays as input."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.00001, 2.00001, 3.00001])

        # Should work with NumPy arrays
        assert_allclose(a, b, rtol=1e-4, atol=1e-8)

    def test_assert_allclose_list_inputs(self) -> None:
        """Test assert_allclose accepts lists as input."""
        a = [1.0, 2.0, 3.0]
        b = [1.00001, 2.00001, 3.00001]

        # Should work with lists
        assert_allclose(a, b, rtol=1e-4, atol=1e-8)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
