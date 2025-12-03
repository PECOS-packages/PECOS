"""Test Array negative step slicing to match NumPy behavior."""

import numpy as np
import pytest

from _pecos_rslib import Array


class TestPecosArrayNegativeSlicing:
    """Test Array negative step slicing matches NumPy."""

    def test_basic_reverse(self):
        """Test: arr[::-1] - basic full reverse."""
        np_arr = np.array([0.0, 1.0, 2.0, 3.0])
        pa_arr = Array(np_arr)

        np_result = np_arr[::-1]
        pa_result = pa_arr[::-1]

        pa_result_np = np.asarray(pa_result)
        np.testing.assert_array_equal(pa_result_np, np_result)
        print(f"arr[::-1]: {pa_result_np} == {np_result}")

    def test_reverse_from_index(self):
        """Test: arr[3::-1] - reverse from index 3 to beginning."""
        np_arr = np.array([0.0, 1.0, 2.0, 3.0])
        pa_arr = Array(np_arr)

        np_result = np_arr[3::-1]
        pa_result = pa_arr[3::-1]

        pa_result_np = np.asarray(pa_result)
        np.testing.assert_array_equal(pa_result_np, np_result)
        print(f"arr[3::-1]: {pa_result_np} == {np_result}")

    def test_reverse_with_negative_start(self):
        """Test: arr[-1::-1] - reverse from last element."""
        np_arr = np.array([0.0, 1.0, 2.0, 3.0])
        pa_arr = Array(np_arr)

        np_result = np_arr[-1::-1]
        pa_result = pa_arr[-1::-1]

        pa_result_np = np.asarray(pa_result)
        np.testing.assert_array_equal(pa_result_np, np_result)
        print(f"arr[-1::-1]: {pa_result_np} == {np_result}")

    def test_reverse_with_explicit_stop_negative_5(self):
        """Test: arr[3:-5:-1] - should give full reverse (stop becomes -1 sentinel)."""
        np_arr = np.array([0.0, 1.0, 2.0, 3.0])
        pa_arr = Array(np_arr)

        np_result = np_arr[3:-5:-1]
        pa_result = pa_arr[3:-5:-1]

        pa_result_np = np.asarray(pa_result)
        np.testing.assert_array_equal(pa_result_np, np_result)
        print(f"arr[3:-5:-1]: {pa_result_np} == {np_result}")

    def test_reverse_with_explicit_stop_negative_100(self):
        """Test: arr[3:-100:-1] - very negative stop, should give full reverse."""
        np_arr = np.array([0.0, 1.0, 2.0, 3.0])
        pa_arr = Array(np_arr)

        np_result = np_arr[3:-100:-1]
        pa_result = pa_arr[3:-100:-1]

        pa_result_np = np.asarray(pa_result)
        np.testing.assert_array_equal(pa_result_np, np_result)
        print(f"arr[3:-100:-1]: {pa_result_np} == {np_result}")

    def test_reverse_partial_stop_negative_2(self):
        """Test: arr[3:-2:-1] - partial reverse (stop at index 2)."""
        np_arr = np.array([0.0, 1.0, 2.0, 3.0])
        pa_arr = Array(np_arr)

        np_result = np_arr[3:-2:-1]
        pa_result = pa_arr[3:-2:-1]

        pa_result_np = np.asarray(pa_result)
        np.testing.assert_array_equal(pa_result_np, np_result)
        print(f"arr[3:-2:-1]: {pa_result_np} == {np_result} (should be [3.0])")

    def test_reverse_partial_stop_negative_3(self):
        """Test: arr[3:-3:-1] - partial reverse (stop at index 1)."""
        np_arr = np.array([0.0, 1.0, 2.0, 3.0])
        pa_arr = Array(np_arr)

        np_result = np_arr[3:-3:-1]
        pa_result = pa_arr[3:-3:-1]

        pa_result_np = np.asarray(pa_result)
        np.testing.assert_array_equal(pa_result_np, np_result)
        print(f"arr[3:-3:-1]: {pa_result_np} == {np_result} (should be [3.0, 2.0])")

    def test_reverse_partial_stop_negative_4(self):
        """Test: arr[3:-4:-1] - partial reverse (stop at index 0)."""
        np_arr = np.array([0.0, 1.0, 2.0, 3.0])
        pa_arr = Array(np_arr)

        np_result = np_arr[3:-4:-1]
        pa_result = pa_arr[3:-4:-1]

        pa_result_np = np.asarray(pa_result)
        np.testing.assert_array_equal(pa_result_np, np_result)
        print(
            f"arr[3:-4:-1]: {pa_result_np} == {np_result} (should be [3.0, 2.0, 1.0])"
        )

    def test_reverse_empty_stop_negative_1(self):
        """Test: arr[3:-1:-1] - should be empty (start==stop after normalization)."""
        np_arr = np.array([0.0, 1.0, 2.0, 3.0])
        pa_arr = Array(np_arr)

        np_result = np_arr[3:-1:-1]
        pa_result = pa_arr[3:-1:-1]

        pa_result_np = np.asarray(pa_result)
        np.testing.assert_array_equal(pa_result_np, np_result)
        print(f"arr[3:-1:-1]: {pa_result_np} == {np_result} (should be empty)")

    def test_reverse_with_step_minus_2(self):
        """Test: arr[::-2] - reverse with step -2."""
        np_arr = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        pa_arr = Array(np_arr)

        np_result = np_arr[::-2]
        pa_result = pa_arr[::-2]

        pa_result_np = np.asarray(pa_result)
        np.testing.assert_array_equal(pa_result_np, np_result)
        print(f"arr[::-2]: {pa_result_np} == {np_result} (should be [4.0, 2.0, 0.0])")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
