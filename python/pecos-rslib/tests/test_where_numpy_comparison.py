"""Comprehensive tests comparing pecos_rslib.where() with numpy.where().

This test suite ensures our where() implementation matches numpy's behavior
across all parameter combinations:
- Scalar vs array for condition, x, y
- Broadcasting behavior
- Different dtypes and shapes
"""

import numpy as np

from pecos_rslib import where as pecos_where


class TestWhereNumPyComparison:
    """Test pecos where() against numpy.where() for all combinations."""

    def test_scalar_condition_scalar_values(self) -> None:
        """Test: bool condition, scalar x, scalar y."""
        # True condition
        np_result = np.where(True, 10.0, 20.0)
        pecos_result = pecos_where(True, 10.0, 20.0)
        assert pecos_result == np_result

        # False condition
        np_result = np.where(False, 10.0, 20.0)
        pecos_result = pecos_where(False, 10.0, 20.0)
        assert pecos_result == np_result

    def test_scalar_condition_array_values(self) -> None:
        """Test: bool condition, array x, array y."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([10.0, 20.0, 30.0])

        # True condition - should return x
        np_result = np.where(True, x, y)
        pecos_result = pecos_where(True, x, y)
        np.testing.assert_array_equal(pecos_result, np_result)

        # False condition - should return y
        np_result = np.where(False, x, y)
        pecos_result = pecos_where(False, x, y)
        np.testing.assert_array_equal(pecos_result, np_result)

    def test_array_condition_scalar_values(self) -> None:
        """Test: array condition, scalar x, scalar y (broadcasting)."""
        condition = np.array([True, False, True, False])

        np_result = np.where(condition, 10.0, 20.0)
        pecos_result = pecos_where(condition, 10.0, 20.0)
        np.testing.assert_array_equal(pecos_result, np_result)

    def test_array_condition_array_values_same_shape(self) -> None:
        """Test: array condition, array x, array y (all same shape)."""
        condition = np.array([True, False, True, False])
        x = np.array([10.0, 20.0, 30.0, 40.0])
        y = np.array([100.0, 200.0, 300.0, 400.0])

        np_result = np.where(condition, x, y)
        pecos_result = pecos_where(condition, x, y)
        np.testing.assert_array_equal(pecos_result, np_result)
        # Expected: [10.0, 200.0, 30.0, 400.0]

    def test_array_condition_mixed_scalar_array(self) -> None:
        """Test: array condition, array x, scalar y (broadcasting)."""
        condition = np.array([True, False, True, False])
        x = np.array([10.0, 20.0, 30.0, 40.0])
        y_scalar = -1.0

        np_result = np.where(condition, x, y_scalar)
        pecos_result = pecos_where(condition, x, y_scalar)
        np.testing.assert_array_equal(pecos_result, np_result)
        # Expected: [10.0, -1.0, 30.0, -1.0]

    def test_array_condition_scalar_x_array_y(self) -> None:
        """Test: array condition, scalar x, array y (broadcasting)."""
        condition = np.array([True, False, True, False])
        x_scalar = 999.0
        y = np.array([100.0, 200.0, 300.0, 400.0])

        np_result = np.where(condition, x_scalar, y)
        pecos_result = pecos_where(condition, x_scalar, y)
        np.testing.assert_array_equal(pecos_result, np_result)
        # Expected: [999.0, 200.0, 999.0, 400.0]

    def test_2d_array_condition_and_values(self) -> None:
        """Test: 2D arrays for condition, x, y."""
        condition = np.array([[True, False], [False, True]])
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array([[10.0, 20.0], [30.0, 40.0]])

        np_result = np.where(condition, x, y)
        pecos_result = pecos_where(condition, x, y)
        np.testing.assert_array_equal(pecos_result, np_result)
        # Expected: [[1.0, 20.0], [30.0, 4.0]]

    def test_2d_condition_with_scalar_values(self) -> None:
        """Test: 2D condition, scalar x and y (broadcasting)."""
        condition = np.array([[True, False], [False, True]])

        np_result = np.where(condition, 100.0, -100.0)
        pecos_result = pecos_where(condition, 100.0, -100.0)
        np.testing.assert_array_equal(pecos_result, np_result)
        # Expected: [[100.0, -100.0], [-100.0, 100.0]]

    def test_broadcasting_1d_to_2d(self) -> None:
        """Test: broadcasting 1D arrays to 2D."""
        # Condition is 2D, x and y are 1D (should broadcast)
        condition = np.array([[True, False], [False, True]])
        x = np.array([1.0, 2.0])  # 1D
        y = np.array([10.0, 20.0])  # 1D

        np_result = np.where(condition, x, y)
        pecos_result = pecos_where(condition, x, y)
        np.testing.assert_array_equal(pecos_result, np_result)

    def test_all_true_condition(self) -> None:
        """Test: all True condition (should return x)."""
        condition = np.array([True, True, True])
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([10.0, 20.0, 30.0])

        np_result = np.where(condition, x, y)
        pecos_result = pecos_where(condition, x, y)
        np.testing.assert_array_equal(pecos_result, np_result)
        np.testing.assert_array_equal(pecos_result, x)

    def test_all_false_condition(self) -> None:
        """Test: all False condition (should return y)."""
        condition = np.array([False, False, False])
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([10.0, 20.0, 30.0])

        np_result = np.where(condition, x, y)
        pecos_result = pecos_where(condition, x, y)
        np.testing.assert_array_equal(pecos_result, np_result)
        np.testing.assert_array_equal(pecos_result, y)

    def test_empty_arrays(self) -> None:
        """Test: empty arrays."""
        condition = np.array([], dtype=bool)
        x = np.array([], dtype=np.float64)
        y = np.array([], dtype=np.float64)

        np_result = np.where(condition, x, y)
        pecos_result = pecos_where(condition, x, y)
        np.testing.assert_array_equal(pecos_result, np_result)

    def test_list_inputs(self) -> None:
        """Test: Python lists as inputs (should convert to arrays)."""
        condition = [True, False, True, False]
        x = [1.0, 2.0, 3.0, 4.0]
        y = [10.0, 20.0, 30.0, 40.0]

        np_result = np.where(condition, x, y)
        pecos_result = pecos_where(condition, x, y)
        np.testing.assert_array_equal(pecos_result, np_result)

    def test_tuple_inputs(self) -> None:
        """Test: Python tuples as inputs."""
        condition = (True, False, True)
        x = (1.0, 2.0, 3.0)
        y = (10.0, 20.0, 30.0)

        np_result = np.where(condition, x, y)
        pecos_result = pecos_where(condition, x, y)
        np.testing.assert_array_equal(pecos_result, np_result)

    def test_integer_arrays(self) -> None:
        """Test: integer arrays (type preservation)."""
        condition = np.array([True, False, True, False])
        x = np.array([1, 2, 3, 4])  # integers
        y = np.array([10, 20, 30, 40])  # integers

        np_result = np.where(condition, x, y)
        pecos_result = pecos_where(condition, x, y)
        np.testing.assert_array_equal(pecos_result, np_result)

    def test_comparison_condition(self) -> None:
        """Test: condition from comparison operation."""
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # Example from numpy docs: np.where(a < 5, a, 10*a)
        np_result = np.where(a < 5, a, 10 * a)
        pecos_result = pecos_where(a < 5, a, 10 * a)
        np.testing.assert_array_equal(pecos_result, np_result)
        # Expected: [1.0, 2.0, 3.0, 4.0, 50.0]

    def test_comparison_with_scalar_y(self) -> None:
        """Test: Example from numpy docs with scalar y."""
        a = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.float64)

        # Example from docs: np.where(a < 4, a, -1)
        np_result = np.where(a < 4, a, -1)
        pecos_result = pecos_where(a < 4, a, -1)
        np.testing.assert_array_equal(pecos_result, np_result)
        # Expected: [0.0, 1.0, 2.0, 3.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]

    def test_3d_arrays(self) -> None:
        """Test: 3D arrays."""
        condition = np.array([[[True, False], [False, True]]])
        x = np.array([[[1.0, 2.0], [3.0, 4.0]]])
        y = np.array([[[10.0, 20.0], [30.0, 40.0]]])

        np_result = np.where(condition, x, y)
        pecos_result = pecos_where(condition, x, y)
        np.testing.assert_array_equal(pecos_result, np_result)
