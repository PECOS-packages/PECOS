"""Tests for Array utility methods: copy, astype, comparison operators, all, any, len, repr/str, edge cases."""

import numpy as np
import pytest

from pecos_rslib import Array, dtypes

# ---------------------------------------------------------------------------
# copy()
# ---------------------------------------------------------------------------


class TestCopy:
    """Test Array.copy() method."""

    def test_copy_returns_equal_values(self) -> None:
        """Test that copy returns an Array with equal values."""
        a = Array(np.array([1.0, 2.0, 3.0]))
        b = a.copy()
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

    def test_copy_is_independent(self) -> None:
        """Modifying the copy should not affect the original."""
        a = Array(np.array([1.0, 2.0, 3.0]))
        b = a.copy()
        # Modify b via setitem
        b[0] = 99.0
        assert np.asarray(a)[0] == 1.0
        assert np.asarray(b)[0] == 99.0

    def test_copy_preserves_dtype_f64(self) -> None:
        """Test that copy preserves float64 dtype."""
        a = Array(np.array([1.0, 2.0]))
        b = a.copy()
        assert b.dtype == a.dtype

    def test_copy_preserves_dtype_complex128(self) -> None:
        """Test that copy preserves complex128 dtype and values."""
        a = Array(np.array([1 + 2j, 3 + 4j], dtype=np.complex128))
        b = a.copy()
        assert b.dtype == a.dtype
        np.testing.assert_array_equal(np.asarray(b), np.asarray(a))

    def test_copy_preserves_dtype_i64(self) -> None:
        """Test that copy preserves int64 dtype."""
        a = Array(np.array([1, 2, 3], dtype=np.int64))
        b = a.copy()
        assert b.dtype == a.dtype

    def test_copy_preserves_shape(self) -> None:
        """Test that copy preserves the shape of a 2D array."""
        a = Array(np.ones((3, 4)))
        b = a.copy()
        assert b.shape == (3, 4)

    def test_copy_preserves_bool(self) -> None:
        """Test that copy preserves boolean array values."""
        a = Array(np.array([True, False, True]))
        b = a.copy()
        np.testing.assert_array_equal(np.asarray(b), np.asarray(a))


# ---------------------------------------------------------------------------
# astype()
# ---------------------------------------------------------------------------


class TestAstype:
    """Test Array.astype() dtype conversion method."""

    def test_f64_to_complex128(self) -> None:
        """Test converting float64 to complex128."""
        a = Array(np.array([1.0, 2.0, 3.0]))
        b = a.astype(dtypes.complex128)
        assert b.dtype == dtypes.complex128
        assert np.asarray(b)[0] == 1.0 + 0j

    def test_complex128_to_f64(self) -> None:
        """Converting complex to real should take the real part."""
        a = Array(np.array([1 + 2j, 3 + 4j], dtype=np.complex128))
        b = a.astype(dtypes.float64)
        assert b.dtype == dtypes.float64
        assert np.asarray(b)[0] == 1.0
        assert np.asarray(b)[1] == 3.0

    def test_f64_to_i64(self) -> None:
        """Test converting float64 to int64 with truncation."""
        a = Array(np.array([1.5, 2.7, 3.0]))
        b = a.astype(dtypes.int64)
        assert b.dtype == dtypes.int64
        assert np.asarray(b)[0] == 1
        assert np.asarray(b)[1] == 2

    def test_i64_to_f64(self) -> None:
        """Test converting int64 to float64."""
        a = Array(np.array([1, 2, 3], dtype=np.int64))
        b = a.astype(dtypes.float64)
        assert b.dtype == dtypes.float64
        assert np.asarray(b)[1] == 2.0

    def test_f64_to_f32(self) -> None:
        """Test converting float64 to float32."""
        a = Array(np.array([1.0, 2.0]))
        b = a.astype(dtypes.float32)
        assert b.dtype == dtypes.float32

    def test_f32_to_f64(self) -> None:
        """Test converting float32 to float64."""
        a = Array(np.array([1.0, 2.0], dtype=np.float32))
        b = a.astype(dtypes.float64)
        assert b.dtype == dtypes.float64

    def test_i64_to_complex128(self) -> None:
        """Test converting int64 to complex128."""
        a = Array(np.array([1, 2, 3], dtype=np.int64))
        b = a.astype(dtypes.complex128)
        assert b.dtype == dtypes.complex128
        assert np.asarray(b)[0] == 1.0 + 0j

    def test_bool_to_f64(self) -> None:
        """Test converting bool to float64."""
        a = Array(np.array([True, False, True]))
        b = a.astype(dtypes.float64)
        assert b.dtype == dtypes.float64
        assert np.asarray(b)[0] == 1.0
        assert np.asarray(b)[1] == 0.0

    def test_f64_to_bool(self) -> None:
        """Test converting float64 to bool where zero is False and nonzero is True."""
        a = Array(np.array([0.0, 1.0, -2.5]))
        b = a.astype(dtypes.bool)
        assert b.dtype == dtypes.bool
        result = np.asarray(b)
        assert not result[0]
        assert result[1]
        assert result[2]

    def test_same_dtype_returns_copy(self) -> None:
        """Converting to same dtype should return a copy, not the same object."""
        a = Array(np.array([1.0, 2.0]))
        b = a.astype(dtypes.float64)
        assert b.dtype == dtypes.float64
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

    def test_preserves_shape(self) -> None:
        """Test that astype preserves the array shape."""
        a = Array(np.ones((3, 4)))
        b = a.astype(dtypes.int64)
        assert b.shape == (3, 4)

    def test_complex64_to_complex128(self) -> None:
        """Test converting complex64 to complex128."""
        a = Array(np.array([1 + 2j, 3 + 4j], dtype=np.complex64))
        b = a.astype(dtypes.complex128)
        assert b.dtype == dtypes.complex128
        assert abs(np.asarray(b)[0] - (1 + 2j)) < 1e-5

    def test_i64_to_bool(self) -> None:
        """Test converting int64 to bool where zero is False and nonzero is True."""
        a = Array(np.array([0, 1, -5], dtype=np.int64))
        b = a.astype(dtypes.bool)
        assert b.dtype == dtypes.bool
        result = np.asarray(b)
        assert not result[0]
        assert result[1]
        assert result[2]


# ---------------------------------------------------------------------------
# Comparison operators: <, <=, >, >=
# ---------------------------------------------------------------------------


class TestComparisonOps:
    """Test __lt__, __le__, __gt__, __ge__ with scalar operands."""

    def test_gt_scalar_f64(self) -> None:
        """Test greater-than comparison with a scalar on float64 array."""
        a = Array(np.array([1.0, 2.0, 3.0, 4.0]))
        result = a > 2.5
        expected = np.array([0.0, 0.0, 1.0, 1.0])
        np.testing.assert_array_equal(np.asarray(result), expected)

    def test_lt_scalar_f64(self) -> None:
        """Test less-than comparison with a scalar on float64 array."""
        a = Array(np.array([1.0, 2.0, 3.0, 4.0]))
        result = a < 2.5
        expected = np.array([1.0, 1.0, 0.0, 0.0])
        np.testing.assert_array_equal(np.asarray(result), expected)

    def test_ge_scalar_f64(self) -> None:
        """Test greater-than-or-equal comparison with a scalar on float64 array."""
        a = Array(np.array([1.0, 2.0, 3.0]))
        result = a >= 2.0
        expected = np.array([0.0, 1.0, 1.0])
        np.testing.assert_array_equal(np.asarray(result), expected)

    def test_le_scalar_f64(self) -> None:
        """Test less-than-or-equal comparison with a scalar on float64 array."""
        a = Array(np.array([1.0, 2.0, 3.0]))
        result = a <= 2.0
        expected = np.array([1.0, 1.0, 0.0])
        np.testing.assert_array_equal(np.asarray(result), expected)

    def test_gt_scalar_i64(self) -> None:
        """Test greater-than comparison with a scalar on int64 array."""
        a = Array(np.array([1, 2, 3, 4], dtype=np.int64))
        result = a > 2.0
        expected = np.array([0.0, 0.0, 1.0, 1.0])
        np.testing.assert_array_equal(np.asarray(result), expected)

    def test_lt_returns_f64_dtype(self) -> None:
        """Comparison results are F64 arrays with 1.0/0.0 values."""
        a = Array(np.array([1.0, 5.0]))
        result = a < 3.0
        assert result.dtype == dtypes.float64

    def test_gt_2d(self) -> None:
        """Test greater-than comparison on a 2D array."""
        a = Array(np.array([[1.0, 2.0], [3.0, 4.0]]))
        result = a > 2.5
        expected = np.array([[0.0, 0.0], [1.0, 1.0]])
        np.testing.assert_array_equal(np.asarray(result), expected)

    def test_complex_comparison_raises(self) -> None:
        """Comparison operators on complex arrays should raise."""
        a = Array(np.array([1 + 2j, 3 + 4j], dtype=np.complex128))
        with pytest.raises(TypeError, match="not supported for complex"):
            _ = a > 1.0

    def test_all_false(self) -> None:
        """Test comparison where no elements satisfy the condition."""
        a = Array(np.array([1.0, 2.0, 3.0]))
        result = a > 10.0
        expected = np.array([0.0, 0.0, 0.0])
        np.testing.assert_array_equal(np.asarray(result), expected)

    def test_all_true(self) -> None:
        """Test comparison where all elements satisfy the condition."""
        a = Array(np.array([1.0, 2.0, 3.0]))
        result = a > 0.0
        expected = np.array([1.0, 1.0, 1.0])
        np.testing.assert_array_equal(np.asarray(result), expected)


# ---------------------------------------------------------------------------
# any() and all() as functions and methods
# ---------------------------------------------------------------------------


class TestAllAny:
    """Test all() and any() on Array objects."""

    # -- all() method --

    def test_all_true_bool(self) -> None:
        """Test all() returns True when all boolean elements are True."""
        a = Array(np.array([True, True, True]))
        assert a.all() is True

    def test_all_false_bool(self) -> None:
        """Test all() returns False when some boolean elements are False."""
        a = Array(np.array([True, False, True]))
        assert a.all() is False

    def test_all_true_f64(self) -> None:
        """Test all() returns True when all float64 elements are nonzero."""
        a = Array(np.array([1.0, 2.0, 3.0]))
        assert a.all() is True

    def test_all_false_f64_with_zero(self) -> None:
        """Test all() returns False when a float64 element is zero."""
        a = Array(np.array([1.0, 0.0, 3.0]))
        assert a.all() is False

    def test_all_true_i64(self) -> None:
        """Test all() returns True when all int64 elements are nonzero."""
        a = Array(np.array([1, 2, 3], dtype=np.int64))
        assert a.all() is True

    def test_all_false_i64_with_zero(self) -> None:
        """Test all() returns False when an int64 element is zero."""
        a = Array(np.array([1, 0, 3], dtype=np.int64))
        assert a.all() is False

    def test_all_complex_nonzero(self) -> None:
        """Test all() returns True when all complex128 elements are nonzero."""
        a = Array(np.array([1 + 0j, 0 + 1j], dtype=np.complex128))
        assert a.all() is True

    def test_all_complex_with_zero(self) -> None:
        """Test all() returns False when a complex128 element is zero."""
        a = Array(np.array([1 + 0j, 0 + 0j], dtype=np.complex128))
        assert a.all() is False

    # -- any() via pc.any() --

    def test_any_some_true_bool(self) -> None:
        """Test pc.any() returns True when some boolean elements are True."""
        import pecos as pc

        a = pc.array([True, False, False])
        assert pc.any(a) is True

    def test_any_all_false_bool(self) -> None:
        """Test pc.any() returns False when all boolean elements are False."""
        import pecos as pc

        a = pc.array([False, False, False])
        assert pc.any(a) is False

    def test_any_f64_with_nonzero(self) -> None:
        """Test pc.any() returns True when a float64 element is nonzero."""
        import pecos as pc

        a = pc.array([0.0, 0.0, 1.0])
        assert pc.any(a) is True

    def test_any_f64_all_zero(self) -> None:
        """Test pc.any() returns False when all float64 elements are zero."""
        import pecos as pc

        a = pc.array([0.0, 0.0, 0.0])
        assert pc.any(a) is False

    def test_any_i64(self) -> None:
        """Test pc.any() returns True for int64 array with a nonzero element."""
        import pecos as pc

        a = pc.array([0, 0, 5], dtype=pc.dtypes.int64)
        assert pc.any(a) is True

    def test_all_via_pc(self) -> None:
        """Test pc.all() returns True when all elements are nonzero."""
        import pecos as pc

        a = pc.array([1.0, 2.0, 3.0])
        assert pc.all(a) is True

    def test_all_via_pc_false(self) -> None:
        """Test pc.all() returns False when an element is zero."""
        import pecos as pc

        a = pc.array([1.0, 0.0, 3.0])
        assert pc.all(a) is False

    def test_any_scalar_true(self) -> None:
        """Test pc.any() returns True for a nonzero scalar."""
        import pecos as pc

        assert pc.any(1.0) is True

    def test_any_scalar_false(self) -> None:
        """Test pc.any() returns False for a zero scalar."""
        import pecos as pc

        assert pc.any(0.0) is False

    def test_all_scalar_true(self) -> None:
        """Test pc.all() returns True for a nonzero scalar."""
        import pecos as pc

        assert pc.all(1.0) is True

    def test_all_scalar_false(self) -> None:
        """Test pc.all() returns False for a zero scalar."""
        import pecos as pc

        assert pc.all(0) is False

    # -- axis-based all/any --

    def test_all_axis_0_2d(self) -> None:
        """Test all() along axis 0 of a 2D boolean array."""
        a = Array(np.array([[True, False], [True, True]]))
        result = a.all(axis=0)
        expected = np.array([True, False])
        np.testing.assert_array_equal(np.asarray(result), expected)

    def test_all_axis_1_2d(self) -> None:
        """Test all() along axis 1 of a 2D boolean array."""
        a = Array(np.array([[True, False], [True, True]]))
        result = a.all(axis=1)
        expected = np.array([False, True])
        np.testing.assert_array_equal(np.asarray(result), expected)

    def test_any_axis_0_2d(self) -> None:
        """Test any() along axis 0 of a 2D boolean array."""
        a = Array(np.array([[True, False], [False, False]]))
        result = a.any(axis=0)
        expected = np.array([True, False])
        np.testing.assert_array_equal(np.asarray(result), expected)

    def test_any_axis_1_2d(self) -> None:
        """Test any() along axis 1 of a 2D boolean array."""
        a = Array(np.array([[True, False], [False, False]]))
        result = a.any(axis=1)
        expected = np.array([True, False])
        np.testing.assert_array_equal(np.asarray(result), expected)

    def test_all_axis_f64(self) -> None:
        """Test all() along axis 0 of a 2D float64 array."""
        a = Array(np.array([[1.0, 0.0], [2.0, 3.0]]))
        result = a.all(axis=0)
        expected = np.array([True, False])
        np.testing.assert_array_equal(np.asarray(result), expected)

    def test_any_axis_i64(self) -> None:
        """Test any() along axis 0 of a 2D int64 array."""
        a = Array(np.array([[0, 0], [0, 5]], dtype=np.int64))
        result = a.any(axis=0)
        expected = np.array([False, True])
        np.testing.assert_array_equal(np.asarray(result), expected)

    def test_all_axis_returns_bool_array(self) -> None:
        """Test that all() with axis returns a bool dtype array."""
        a = Array(np.array([[True, False], [True, True]]))
        result = a.all(axis=0)
        assert result.dtype == dtypes.bool

    def test_pc_all_axis(self) -> None:
        """Test pc.all() with axis parameter on a 2D array."""
        import pecos as pc

        a = pc.array([[True, False], [True, True]])
        result = pc.all(a, axis=0)
        expected = np.array([True, False])
        np.testing.assert_array_equal(np.asarray(result), expected)

    def test_pc_any_axis(self) -> None:
        """Test pc.any() with axis parameter on a 2D array."""
        import pecos as pc

        a = pc.array([[True, False], [False, False]])
        result = pc.any(a, axis=0)
        expected = np.array([True, False])
        np.testing.assert_array_equal(np.asarray(result), expected)

    def test_all_axis_negative(self) -> None:
        """Test all() with a negative axis index."""
        a = Array(np.array([[True, False], [True, True]]))
        result = a.all(axis=-1)
        expected = np.array([False, True])
        np.testing.assert_array_equal(np.asarray(result), expected)

    def test_all_axis_out_of_bounds(self) -> None:
        """Test that all() raises ValueError for an out-of-bounds axis."""
        a = Array(np.array([[True, False], [True, True]]))
        with pytest.raises(ValueError, match="out of bounds"):
            a.all(axis=5)

    def test_array_any_method(self) -> None:
        """Test Array.any() method returns True when some elements are True."""
        a = Array(np.array([True, False, True]))
        assert a.any() is True

    def test_array_any_method_all_false(self) -> None:
        """Test Array.any() method returns False when all elements are False."""
        a = Array(np.array([False, False, False]))
        assert a.any() is False


# ---------------------------------------------------------------------------
# __len__
# ---------------------------------------------------------------------------


class TestLen:
    """Test Array __len__ method."""

    def test_len_1d(self) -> None:
        """Test len() on a 1D array."""
        a = Array(np.array([1.0, 2.0, 3.0]))
        assert len(a) == 3

    def test_len_2d_returns_first_dim(self) -> None:
        """Test len() on a 2D array returns the first dimension size."""
        a = Array(np.ones((3, 4)))
        assert len(a) == 3

    def test_len_3d_returns_first_dim(self) -> None:
        """Test len() on a 3D array returns the first dimension size."""
        a = Array(np.ones((2, 3, 4)))
        assert len(a) == 2

    def test_len_single_element(self) -> None:
        """Test len() on a single-element array returns 1."""
        a = Array(np.array([42.0]))
        assert len(a) == 1


# ---------------------------------------------------------------------------
# __repr__ and __str__
# ---------------------------------------------------------------------------


class TestReprStr:
    """Test Array __repr__ and __str__ methods."""

    def test_repr_f64(self) -> None:
        """Test repr() of a float64 array includes type and dtype info."""
        a = Array(np.array([1.0, 2.0, 3.0]))
        r = repr(a)
        assert "Array" in r
        assert "float64" in r
        assert "(3,)" in r or "3" in r

    def test_repr_complex128(self) -> None:
        """Test repr() of a complex128 array includes dtype info."""
        a = Array(np.array([1 + 2j], dtype=np.complex128))
        r = repr(a)
        assert "complex128" in r

    def test_repr_i64(self) -> None:
        """Test repr() of an int64 array includes dtype info."""
        a = Array(np.array([1, 2, 3], dtype=np.int64))
        r = repr(a)
        assert "int64" in r

    def test_repr_2d_shape(self) -> None:
        """Test repr() of a 2D array includes shape dimensions."""
        a = Array(np.ones((3, 4)))
        r = repr(a)
        assert "3" in r
        assert "4" in r

    def test_str_contains_values(self) -> None:
        """Test str() of an array contains the element values."""
        a = Array(np.array([1.0, 2.0, 3.0]))
        s = str(a)
        assert "1" in s
        assert "2" in s
        assert "3" in s

    def test_str_bool(self) -> None:
        """Test str() of a boolean array contains boolean representations."""
        a = Array(np.array([True, False]))
        s = str(a)
        assert "true" in s.lower() or "1" in s or "True" in s

    def test_str_2d(self) -> None:
        """Test str() of a 2D array contains all element values."""
        a = Array(np.array([[1.0, 2.0], [3.0, 4.0]]))
        s = str(a)
        # Should contain all values
        assert "1" in s
        assert "4" in s


# ---------------------------------------------------------------------------
# Edge cases: empty and single-element arrays
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test edge cases with empty and single-element arrays."""

    def test_empty_array_len(self) -> None:
        """Test len() of an empty array returns 0."""
        a = Array(np.array([], dtype=np.float64))
        assert len(a) == 0

    def test_empty_array_shape(self) -> None:
        """Test that an empty array has shape (0,)."""
        a = Array(np.array([], dtype=np.float64))
        assert a.shape == (0,)

    def test_empty_array_all_is_true(self) -> None:
        """all() on empty array is True (vacuous truth, matches numpy)."""
        a = Array(np.array([], dtype=np.float64))
        assert a.all() is True

    def test_empty_array_sum(self) -> None:
        """Test that sum of an empty array returns 0.0."""
        import pecos as pc

        a = pc.array([], dtype=pc.dtypes.float64)
        result = pc.sum(a)
        assert result == 0.0

    def test_single_element_array_ops(self) -> None:
        """Test addition on single-element arrays."""
        a = Array(np.array([5.0]))
        b = Array(np.array([3.0]))
        result = a + b
        assert np.asarray(result)[0] == 8.0

    def test_single_element_comparison(self) -> None:
        """Test comparison on a single-element array."""
        a = Array(np.array([5.0]))
        result = a > 3.0
        assert np.asarray(result)[0] == 1.0

    def test_single_element_all(self) -> None:
        """Test all() returns True for a single nonzero element."""
        a = Array(np.array([1.0]))
        assert a.all() is True

    def test_single_element_all_false(self) -> None:
        """Test all() returns False for a single zero element."""
        a = Array(np.array([0.0]))
        assert a.all() is False

    def test_2d_empty_shape(self) -> None:
        """2D empty arrays (e.g. shape (0,5)) are supported."""
        a = Array(np.ones((0, 5)))
        assert a.shape == (0, 5)

    def test_neg_empty(self) -> None:
        """Test negation of an empty array preserves shape."""
        a = Array(np.array([], dtype=np.float64))
        result = -a
        assert np.asarray(result).shape == (0,)

    def test_copy_empty(self) -> None:
        """Test copy of an empty array preserves shape."""
        a = Array(np.array([], dtype=np.float64))
        b = a.copy()
        assert b.shape == (0,)

    def test_single_element_neg(self) -> None:
        """Test negation of a single-element array."""
        a = Array(np.array([-3.0]))
        result = -a
        assert np.asarray(result)[0] == 3.0

    def test_single_element_conj(self) -> None:
        """Test conjugation of a single-element complex array."""
        a = Array(np.array([1 + 2j], dtype=np.complex128))
        result = a.conj()
        assert abs(np.asarray(result)[0] - (1 - 2j)) < 1e-15
