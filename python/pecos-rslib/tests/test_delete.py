"""Tests for delete() function.

This module tests the Rust implementation of delete() against NumPy
to ensure it's a drop-in replacement.
"""

import numpy as np
import pytest

import pecos as pc


class TestDeleteBasic:
    """Test basic delete() functionality."""

    def test_delete_middle_float(self) -> None:
        """Test deleting middle element from float array."""
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = pc.delete(arr, 2)
        expected = np.delete(arr, 2)

        np.testing.assert_array_equal(result, expected)
        assert result.dtype == expected.dtype

    def test_delete_first_float(self) -> None:
        """Test deleting first element from float array."""
        arr = np.array([10.0, 20.0, 30.0])
        result = pc.delete(arr, 0)
        expected = np.delete(arr, 0)

        np.testing.assert_array_equal(result, expected)

    def test_delete_last_float(self) -> None:
        """Test deleting last element from float array."""
        arr = np.array([10.0, 20.0, 30.0])
        result = pc.delete(arr, 2)
        expected = np.delete(arr, 2)

        np.testing.assert_array_equal(result, expected)

    def test_delete_complex(self) -> None:
        """Test deleting from complex array."""
        arr = np.array([1 + 2j, 3 + 4j, 5 + 6j, 7 + 8j])
        result = pc.delete(arr, 1)
        expected = np.delete(arr, 1)

        np.testing.assert_array_equal(result, expected)
        assert result.dtype == expected.dtype

    def test_delete_int(self) -> None:
        """Test deleting from integer array."""
        arr = np.array([10, 20, 30, 40, 50], dtype=np.int64)
        result = pc.delete(arr, 3)
        expected = np.delete(arr, 3)

        np.testing.assert_array_equal(result, expected)
        assert result.dtype == expected.dtype


class TestDeleteEdgeCases:
    """Test edge cases for delete()."""

    def test_delete_two_elements(self) -> None:
        """Test deleting from 2-element array."""
        arr = np.array([1.0, 2.0])

        # Delete first
        result = pc.delete(arr, 0)
        expected = np.delete(arr, 0)
        np.testing.assert_array_equal(result, expected)

        # Delete second
        result = pc.delete(arr, 1)
        expected = np.delete(arr, 1)
        np.testing.assert_array_equal(result, expected)

    def test_delete_single_element(self) -> None:
        """Test that deleting from single-element array returns empty array."""
        arr = np.array([42.0])

        # NumPy allows this and returns an empty array
        result = pc.delete(arr, 0)
        expected = np.delete(arr, 0)

        np.testing.assert_array_equal(result, expected)
        assert len(result) == 0
        assert result.shape == (0,)

    def test_delete_out_of_bounds(self) -> None:
        """Test deleting with out-of-bounds index."""
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        with pytest.raises(IndexError):
            pc.delete(arr, 5)

        with pytest.raises(IndexError):
            pc.delete(arr, 10)


class TestDeleteJackknifeUseCase:
    """Test the jackknife resampling use case (leave-one-out)."""

    def test_jackknife_simple(self) -> None:
        """Test jackknife resampling on simple array."""
        plist = np.array([0.01, 0.02, 0.03, 0.04, 0.05])

        # Leave-one-out: remove each element in turn
        for i in range(len(plist)):
            rust_result = pc.delete(plist, i)
            numpy_result = np.delete(plist, i)

            np.testing.assert_array_equal(rust_result, numpy_result)
            assert len(rust_result) == len(plist) - 1

            # Verify the removed element is not in the result
            assert plist[i] not in rust_result

    def test_jackknife_threshold_curve_use_case(self) -> None:
        """Test the actual use case from threshold_curve.py."""
        # Simulating the threshold curve fitting scenario
        plist = np.array([0.001, 0.002, 0.003, 0.004, 0.005, 0.006])
        plog = np.log(plist)
        dlist = np.array([3, 5, 7, 9, 11, 13])

        results = []
        for i in range(len(plog)):
            # This is exactly what threshold_curve.py does
            p_copy = pc.delete(plist, i)
            plog_copy = pc.delete(plog, i)
            dlist_copy = pc.delete(dlist, i)

            # Verify all arrays have correct length
            assert len(p_copy) == len(plist) - 1
            assert len(plog_copy) == len(plog) - 1
            assert len(dlist_copy) == len(dlist) - 1

            # Verify correspondence is maintained
            for j in range(len(p_copy)):
                assert np.isclose(plog_copy[j], np.log(p_copy[j]))

            results.append((p_copy, plog_copy, dlist_copy))

        # Verify we processed all iterations
        assert len(results) == len(plist)


class TestDeleteWithLists:
    """Test delete() with Python lists (should convert automatically)."""

    def test_delete_from_list(self) -> None:
        """Test deleting from Python list."""
        lst = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = pc.delete(lst, 2)
        expected = np.delete(np.array(lst), 2)

        np.testing.assert_array_equal(result, expected)

    def test_delete_from_complex_list(self) -> None:
        """Test deleting from list of complex numbers."""
        lst = [1 + 2j, 3 + 4j, 5 + 6j]
        result = pc.delete(lst, 1)
        expected = np.delete(np.array(lst), 1)

        np.testing.assert_array_equal(result, expected)


class TestDeleteTypePreservation:
    """Test that delete() preserves array dtype."""

    def test_float64_preserved(self) -> None:
        """Test float64 dtype is preserved."""
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        result = pc.delete(arr, 1)

        assert result.dtype == np.float64
        np.testing.assert_array_equal(result, np.array([1.0, 3.0]))

    def test_complex128_preserved(self) -> None:
        """Test complex128 dtype is preserved."""
        arr = np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex128)
        result = pc.delete(arr, 0)

        assert result.dtype == np.complex128
        np.testing.assert_array_equal(result, np.array([3 + 4j, 5 + 6j]))

    def test_int64_preserved(self) -> None:
        """Test int64 dtype is preserved."""
        arr = np.array([10, 20, 30, 40], dtype=np.int64)
        result = pc.delete(arr, 2)

        assert result.dtype == np.int64
        np.testing.assert_array_equal(result, np.array([10, 20, 40]))


class TestDeletePerformance:
    """Test delete() performance characteristics."""

    def test_delete_maintains_order(self) -> None:
        """Test that delete() maintains element order."""
        arr = np.array([5.0, 3.0, 8.0, 1.0, 9.0, 2.0])
        result = pc.delete(arr, 2)

        # Element order should be preserved (just element at index 2 removed)
        expected = np.array([5.0, 3.0, 1.0, 9.0, 2.0])
        np.testing.assert_array_equal(result, expected)

    def test_delete_from_pecos_num(self) -> None:
        """Test that delete() is accessible from pecos."""
        # Already imported at top: import pecos as pc

        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = pc.delete(arr, 2)
        expected = np.delete(arr, 2)

        np.testing.assert_array_equal(result, expected)
