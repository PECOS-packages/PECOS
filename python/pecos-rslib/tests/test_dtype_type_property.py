"""Test that DType.type property works for NumPy compatibility.

NumPy provides arr.dtype.type to get the scalar class. This test ensures
PECOS implements the same interface for drop-in NumPy replacement.
"""

import numpy as np
import pytest

from pecos_rslib import Array, dtypes


class TestDTypeTypeProperty:
    """Test the .type property on DType objects."""

    def test_dtype_has_type_property(self) -> None:
        """Test that dtype has a .type attribute."""
        arr = Array([1, 2, 3], dtype="int64")
        assert hasattr(arr.dtype, "type")
        assert arr.dtype.type is not None

    def test_type_property_returns_class(self) -> None:
        """Test that .type returns a class (type)."""
        arr = Array([1, 2, 3], dtype="int64")
        scalar_type = arr.dtype.type
        assert isinstance(scalar_type, type)

    def test_type_property_is_callable(self) -> None:
        """Test that the returned type can be called to create scalars."""
        arr = Array([1, 2, 3], dtype="int64")
        ScalarType = arr.dtype.type
        val = ScalarType(99)
        assert val == 99

    def test_numpy_compatibility_pattern_type_of_scalar(self) -> None:
        """Test NumPy pattern: dtype = type(a); b = dtype(1)"""
        # This is a common pattern in NumPy code
        pecos_scalar = dtypes.i64(5)
        ScalarClass = type(pecos_scalar)
        new_val = ScalarClass(42)

        assert new_val == 42
        assert type(new_val).__name__ == "i64"

    def test_numpy_compatibility_pattern_array_dtype_type(self) -> None:
        """Test NumPy pattern: dtype = arr.dtype.type; b = dtype(1)"""
        # This is another common NumPy pattern
        arr = Array([1, 2, 3], dtype="int64")
        ScalarType = arr.dtype.type
        val = ScalarType(99)

        assert val == 99
        assert type(val).__name__ == "i64"

    def test_both_patterns_return_same_class(self) -> None:
        """Test that both patterns return the same scalar class."""
        # Pattern 1: type(scalar_instance)
        scalar = dtypes.i64(5)
        class1 = type(scalar)

        # Pattern 2: arr.dtype.type
        arr = Array([1], dtype="int64")
        class2 = arr.dtype.type

        assert class1 is class2

    @pytest.mark.parametrize(
        ("dtype_str", "test_value"),
        [
            ("int64", 42),
            ("int32", 42),
            ("int16", 42),
            ("int8", 42),
            ("uint64", 42),
            ("uint32", 42),
            ("uint16", 42),
            ("uint8", 42),
            ("float64", 3.14),
            ("float32", 3.14),
            ("complex128", 1 + 2j),
            ("bool", True),
        ],
    )
    def test_type_property_for_all_dtypes(self, dtype_str, test_value) -> None:
        """Test .type property works for all supported dtypes."""
        arr = Array([1], dtype=dtype_str)
        ScalarType = arr.dtype.type
        result = ScalarType(test_value)

        # Just verify it doesn't raise and returns something
        assert result is not None

    def test_comparison_with_numpy(self) -> None:
        """Compare PECOS behavior with NumPy behavior."""
        # NumPy behavior
        np_arr = np.array([1, 2, 3], dtype=np.int64)
        np_scalar_type = np_arr.dtype.type
        np_val = np_scalar_type(99)
        assert np_val == 99

        # PECOS behavior (should match)
        pecos_arr = Array([1, 2, 3], dtype="int64")
        pecos_scalar_type = pecos_arr.dtype.type
        pecos_val = pecos_scalar_type(99)
        assert pecos_val == 99

        # Both should produce values that compare equal
        assert np_val == pecos_val
