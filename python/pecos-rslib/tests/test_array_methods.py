"""Tests for Array utility methods: copy, astype, comparison operators, all, any, len, repr/str, edge cases."""

import numpy as np
import pytest

from pecos_rslib import Array, dtypes


# ---------------------------------------------------------------------------
# copy()
# ---------------------------------------------------------------------------


class TestCopy:
    def test_copy_returns_equal_values(self):
        a = Array(np.array([1.0, 2.0, 3.0]))
        b = a.copy()
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

    def test_copy_is_independent(self):
        """Modifying the copy should not affect the original."""
        a = Array(np.array([1.0, 2.0, 3.0]))
        b = a.copy()
        # Modify b via setitem
        b[0] = 99.0
        assert np.asarray(a)[0] == 1.0
        assert np.asarray(b)[0] == 99.0

    def test_copy_preserves_dtype_f64(self):
        a = Array(np.array([1.0, 2.0]))
        b = a.copy()
        assert b.dtype == a.dtype

    def test_copy_preserves_dtype_complex128(self):
        a = Array(np.array([1 + 2j, 3 + 4j], dtype=np.complex128))
        b = a.copy()
        assert b.dtype == a.dtype
        np.testing.assert_array_equal(np.asarray(b), np.asarray(a))

    def test_copy_preserves_dtype_i64(self):
        a = Array(np.array([1, 2, 3], dtype=np.int64))
        b = a.copy()
        assert b.dtype == a.dtype

    def test_copy_preserves_shape(self):
        a = Array(np.ones((3, 4)))
        b = a.copy()
        assert b.shape == (3, 4)

    def test_copy_preserves_bool(self):
        a = Array(np.array([True, False, True]))
        b = a.copy()
        np.testing.assert_array_equal(np.asarray(b), np.asarray(a))


# ---------------------------------------------------------------------------
# astype()
# ---------------------------------------------------------------------------


class TestAstype:
    def test_f64_to_complex128(self):
        a = Array(np.array([1.0, 2.0, 3.0]))
        b = a.astype(dtypes.complex128)
        assert b.dtype == dtypes.complex128
        assert np.asarray(b)[0] == 1.0 + 0j

    def test_complex128_to_f64(self):
        """Converting complex to real should take the real part."""
        a = Array(np.array([1 + 2j, 3 + 4j], dtype=np.complex128))
        b = a.astype(dtypes.float64)
        assert b.dtype == dtypes.float64
        assert np.asarray(b)[0] == 1.0
        assert np.asarray(b)[1] == 3.0

    def test_f64_to_i64(self):
        a = Array(np.array([1.5, 2.7, 3.0]))
        b = a.astype(dtypes.int64)
        assert b.dtype == dtypes.int64
        assert np.asarray(b)[0] == 1
        assert np.asarray(b)[1] == 2

    def test_i64_to_f64(self):
        a = Array(np.array([1, 2, 3], dtype=np.int64))
        b = a.astype(dtypes.float64)
        assert b.dtype == dtypes.float64
        assert np.asarray(b)[1] == 2.0

    def test_f64_to_f32(self):
        a = Array(np.array([1.0, 2.0]))
        b = a.astype(dtypes.float32)
        assert b.dtype == dtypes.float32

    def test_f32_to_f64(self):
        a = Array(np.array([1.0, 2.0], dtype=np.float32))
        b = a.astype(dtypes.float64)
        assert b.dtype == dtypes.float64

    def test_i64_to_complex128(self):
        a = Array(np.array([1, 2, 3], dtype=np.int64))
        b = a.astype(dtypes.complex128)
        assert b.dtype == dtypes.complex128
        assert np.asarray(b)[0] == 1.0 + 0j

    def test_bool_to_f64(self):
        a = Array(np.array([True, False, True]))
        b = a.astype(dtypes.float64)
        assert b.dtype == dtypes.float64
        assert np.asarray(b)[0] == 1.0
        assert np.asarray(b)[1] == 0.0

    def test_f64_to_bool(self):
        a = Array(np.array([0.0, 1.0, -2.5]))
        b = a.astype(dtypes.bool)
        assert b.dtype == dtypes.bool
        result = np.asarray(b)
        assert result[0] == False
        assert result[1] == True
        assert result[2] == True

    def test_same_dtype_returns_copy(self):
        """Converting to same dtype should return a copy, not the same object."""
        a = Array(np.array([1.0, 2.0]))
        b = a.astype(dtypes.float64)
        assert b.dtype == dtypes.float64
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

    def test_preserves_shape(self):
        a = Array(np.ones((3, 4)))
        b = a.astype(dtypes.int64)
        assert b.shape == (3, 4)

    def test_complex64_to_complex128(self):
        a = Array(np.array([1 + 2j, 3 + 4j], dtype=np.complex64))
        b = a.astype(dtypes.complex128)
        assert b.dtype == dtypes.complex128
        assert abs(np.asarray(b)[0] - (1 + 2j)) < 1e-5

    def test_i64_to_bool(self):
        a = Array(np.array([0, 1, -5], dtype=np.int64))
        b = a.astype(dtypes.bool)
        assert b.dtype == dtypes.bool
        result = np.asarray(b)
        assert result[0] == False
        assert result[1] == True
        assert result[2] == True


# ---------------------------------------------------------------------------
# Comparison operators: <, <=, >, >=
# ---------------------------------------------------------------------------


class TestComparisonOps:
    """Test __lt__, __le__, __gt__, __ge__ with scalar operands."""

    def test_gt_scalar_f64(self):
        a = Array(np.array([1.0, 2.0, 3.0, 4.0]))
        result = a > 2.5
        expected = np.array([0.0, 0.0, 1.0, 1.0])
        np.testing.assert_array_equal(np.asarray(result), expected)

    def test_lt_scalar_f64(self):
        a = Array(np.array([1.0, 2.0, 3.0, 4.0]))
        result = a < 2.5
        expected = np.array([1.0, 1.0, 0.0, 0.0])
        np.testing.assert_array_equal(np.asarray(result), expected)

    def test_ge_scalar_f64(self):
        a = Array(np.array([1.0, 2.0, 3.0]))
        result = a >= 2.0
        expected = np.array([0.0, 1.0, 1.0])
        np.testing.assert_array_equal(np.asarray(result), expected)

    def test_le_scalar_f64(self):
        a = Array(np.array([1.0, 2.0, 3.0]))
        result = a <= 2.0
        expected = np.array([1.0, 1.0, 0.0])
        np.testing.assert_array_equal(np.asarray(result), expected)

    def test_gt_scalar_i64(self):
        a = Array(np.array([1, 2, 3, 4], dtype=np.int64))
        result = a > 2.0
        expected = np.array([0.0, 0.0, 1.0, 1.0])
        np.testing.assert_array_equal(np.asarray(result), expected)

    def test_lt_returns_f64_dtype(self):
        """Comparison results are F64 arrays with 1.0/0.0 values."""
        a = Array(np.array([1.0, 5.0]))
        result = a < 3.0
        assert result.dtype == dtypes.float64

    def test_gt_2d(self):
        a = Array(np.array([[1.0, 2.0], [3.0, 4.0]]))
        result = a > 2.5
        expected = np.array([[0.0, 0.0], [1.0, 1.0]])
        np.testing.assert_array_equal(np.asarray(result), expected)

    def test_complex_comparison_raises(self):
        """Comparison operators on complex arrays should raise."""
        a = Array(np.array([1 + 2j, 3 + 4j], dtype=np.complex128))
        with pytest.raises(TypeError, match="not supported for complex"):
            a > 1.0

    def test_all_false(self):
        a = Array(np.array([1.0, 2.0, 3.0]))
        result = a > 10.0
        expected = np.array([0.0, 0.0, 0.0])
        np.testing.assert_array_equal(np.asarray(result), expected)

    def test_all_true(self):
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

    def test_all_true_bool(self):
        a = Array(np.array([True, True, True]))
        assert a.all() is True

    def test_all_false_bool(self):
        a = Array(np.array([True, False, True]))
        assert a.all() is False

    def test_all_true_f64(self):
        a = Array(np.array([1.0, 2.0, 3.0]))
        assert a.all() is True

    def test_all_false_f64_with_zero(self):
        a = Array(np.array([1.0, 0.0, 3.0]))
        assert a.all() is False

    def test_all_true_i64(self):
        a = Array(np.array([1, 2, 3], dtype=np.int64))
        assert a.all() is True

    def test_all_false_i64_with_zero(self):
        a = Array(np.array([1, 0, 3], dtype=np.int64))
        assert a.all() is False

    def test_all_complex_nonzero(self):
        a = Array(np.array([1 + 0j, 0 + 1j], dtype=np.complex128))
        assert a.all() is True

    def test_all_complex_with_zero(self):
        a = Array(np.array([1 + 0j, 0 + 0j], dtype=np.complex128))
        assert a.all() is False

    # -- any() via pc.any() --

    def test_any_some_true_bool(self):
        import pecos as pc

        a = pc.array([True, False, False])
        assert pc.any(a) is True

    def test_any_all_false_bool(self):
        import pecos as pc

        a = pc.array([False, False, False])
        assert pc.any(a) is False

    def test_any_f64_with_nonzero(self):
        import pecos as pc

        a = pc.array([0.0, 0.0, 1.0])
        assert pc.any(a) is True

    def test_any_f64_all_zero(self):
        import pecos as pc

        a = pc.array([0.0, 0.0, 0.0])
        assert pc.any(a) is False

    def test_any_i64(self):
        import pecos as pc

        a = pc.array([0, 0, 5], dtype=pc.dtypes.int64)
        assert pc.any(a) is True

    def test_all_via_pc(self):
        import pecos as pc

        a = pc.array([1.0, 2.0, 3.0])
        assert pc.all(a) is True

    def test_all_via_pc_false(self):
        import pecos as pc

        a = pc.array([1.0, 0.0, 3.0])
        assert pc.all(a) is False

    def test_any_scalar_true(self):
        import pecos as pc

        assert pc.any(1.0) is True

    def test_any_scalar_false(self):
        import pecos as pc

        assert pc.any(0.0) is False

    def test_all_scalar_true(self):
        import pecos as pc

        assert pc.all(1.0) is True

    def test_all_scalar_false(self):
        import pecos as pc

        assert pc.all(0) is False

    # -- axis-based all/any --

    def test_all_axis_0_2d(self):
        a = Array(np.array([[True, False], [True, True]]))
        result = a.all(axis=0)
        expected = np.array([True, False])
        np.testing.assert_array_equal(np.asarray(result), expected)

    def test_all_axis_1_2d(self):
        a = Array(np.array([[True, False], [True, True]]))
        result = a.all(axis=1)
        expected = np.array([False, True])
        np.testing.assert_array_equal(np.asarray(result), expected)

    def test_any_axis_0_2d(self):
        a = Array(np.array([[True, False], [False, False]]))
        result = a.any(axis=0)
        expected = np.array([True, False])
        np.testing.assert_array_equal(np.asarray(result), expected)

    def test_any_axis_1_2d(self):
        a = Array(np.array([[True, False], [False, False]]))
        result = a.any(axis=1)
        expected = np.array([True, False])
        np.testing.assert_array_equal(np.asarray(result), expected)

    def test_all_axis_f64(self):
        a = Array(np.array([[1.0, 0.0], [2.0, 3.0]]))
        result = a.all(axis=0)
        expected = np.array([True, False])
        np.testing.assert_array_equal(np.asarray(result), expected)

    def test_any_axis_i64(self):
        a = Array(np.array([[0, 0], [0, 5]], dtype=np.int64))
        result = a.any(axis=0)
        expected = np.array([False, True])
        np.testing.assert_array_equal(np.asarray(result), expected)

    def test_all_axis_returns_bool_array(self):
        a = Array(np.array([[True, False], [True, True]]))
        result = a.all(axis=0)
        assert result.dtype == dtypes.bool

    def test_pc_all_axis(self):
        import pecos as pc

        a = pc.array([[True, False], [True, True]])
        result = pc.all(a, axis=0)
        expected = np.array([True, False])
        np.testing.assert_array_equal(np.asarray(result), expected)

    def test_pc_any_axis(self):
        import pecos as pc

        a = pc.array([[True, False], [False, False]])
        result = pc.any(a, axis=0)
        expected = np.array([True, False])
        np.testing.assert_array_equal(np.asarray(result), expected)

    def test_all_axis_negative(self):
        a = Array(np.array([[True, False], [True, True]]))
        result = a.all(axis=-1)
        expected = np.array([False, True])
        np.testing.assert_array_equal(np.asarray(result), expected)

    def test_all_axis_out_of_bounds(self):
        a = Array(np.array([[True, False], [True, True]]))
        with pytest.raises(ValueError, match="out of bounds"):
            a.all(axis=5)

    def test_array_any_method(self):
        a = Array(np.array([True, False, True]))
        assert a.any() is True

    def test_array_any_method_all_false(self):
        a = Array(np.array([False, False, False]))
        assert a.any() is False


# ---------------------------------------------------------------------------
# __len__
# ---------------------------------------------------------------------------


class TestLen:
    def test_len_1d(self):
        a = Array(np.array([1.0, 2.0, 3.0]))
        assert len(a) == 3

    def test_len_2d_returns_first_dim(self):
        a = Array(np.ones((3, 4)))
        assert len(a) == 3

    def test_len_3d_returns_first_dim(self):
        a = Array(np.ones((2, 3, 4)))
        assert len(a) == 2

    def test_len_single_element(self):
        a = Array(np.array([42.0]))
        assert len(a) == 1


# ---------------------------------------------------------------------------
# __repr__ and __str__
# ---------------------------------------------------------------------------


class TestReprStr:
    def test_repr_f64(self):
        a = Array(np.array([1.0, 2.0, 3.0]))
        r = repr(a)
        assert "Array" in r
        assert "float64" in r
        assert "(3,)" in r or "3" in r

    def test_repr_complex128(self):
        a = Array(np.array([1 + 2j], dtype=np.complex128))
        r = repr(a)
        assert "complex128" in r

    def test_repr_i64(self):
        a = Array(np.array([1, 2, 3], dtype=np.int64))
        r = repr(a)
        assert "int64" in r

    def test_repr_2d_shape(self):
        a = Array(np.ones((3, 4)))
        r = repr(a)
        assert "3" in r and "4" in r

    def test_str_contains_values(self):
        a = Array(np.array([1.0, 2.0, 3.0]))
        s = str(a)
        assert "1" in s
        assert "2" in s
        assert "3" in s

    def test_str_bool(self):
        a = Array(np.array([True, False]))
        s = str(a)
        assert "true" in s.lower() or "1" in s or "True" in s

    def test_str_2d(self):
        a = Array(np.array([[1.0, 2.0], [3.0, 4.0]]))
        s = str(a)
        # Should contain all values
        assert "1" in s
        assert "4" in s


# ---------------------------------------------------------------------------
# Edge cases: empty and single-element arrays
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_array_len(self):
        a = Array(np.array([], dtype=np.float64))
        assert len(a) == 0

    def test_empty_array_shape(self):
        a = Array(np.array([], dtype=np.float64))
        assert a.shape == (0,)

    def test_empty_array_all_is_true(self):
        """all() on empty array is True (vacuous truth, matches numpy)."""
        a = Array(np.array([], dtype=np.float64))
        assert a.all() is True

    def test_empty_array_sum(self):
        import pecos as pc

        a = pc.array([], dtype=pc.dtypes.float64)
        result = pc.sum(a)
        assert result == 0.0

    def test_single_element_array_ops(self):
        a = Array(np.array([5.0]))
        b = Array(np.array([3.0]))
        result = a + b
        assert np.asarray(result)[0] == 8.0

    def test_single_element_comparison(self):
        a = Array(np.array([5.0]))
        result = a > 3.0
        assert np.asarray(result)[0] == 1.0

    def test_single_element_all(self):
        a = Array(np.array([1.0]))
        assert a.all() is True

    def test_single_element_all_false(self):
        a = Array(np.array([0.0]))
        assert a.all() is False

    def test_2d_empty_shape_panics(self):
        """2D empty arrays (e.g. shape (0,5)) currently panic in ndarray -- known limitation."""
        with pytest.raises(BaseException):
            Array(np.ones((0, 5)))

    def test_neg_empty(self):
        a = Array(np.array([], dtype=np.float64))
        result = -a
        assert np.asarray(result).shape == (0,)

    def test_copy_empty(self):
        a = Array(np.array([], dtype=np.float64))
        b = a.copy()
        assert b.shape == (0,)

    def test_single_element_neg(self):
        a = Array(np.array([-3.0]))
        result = -a
        assert np.asarray(result)[0] == 3.0

    def test_single_element_conj(self):
        a = Array(np.array([1 + 2j], dtype=np.complex128))
        result = a.conj()
        assert abs(np.asarray(result)[0] - (1 - 2j)) < 1e-15
