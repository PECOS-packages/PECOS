# Copyright 2026 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.

"""Regression coverage for the native numeric prerequisites tracked by PECOS #458."""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

import pecos as pc
import pecos_rslib
from pecos_rslib import Array, dtypes, num

UNSIGNED_DTYPES = (
    ("uint8", "u8", dtypes.uint8, np.uint8),
    ("uint16", "u16", dtypes.uint16, np.uint16),
    ("uint32", "u32", dtypes.uint32, np.uint32),
    ("uint64", "u64", dtypes.uint64, np.uint64),
)

CONSTRUCTORS: tuple[tuple[str, Callable[[str], Array]], ...] = (
    ("Array", lambda dtype: Array([0, 1], dtype=dtype)),
    ("pecos_rslib.array", lambda dtype: pecos_rslib.array([0, 1], dtype=dtype)),
    ("num.array", lambda dtype: num.array([0, 1], dtype=dtype)),
    ("num.asarray", lambda dtype: num.asarray([0, 1], dtype=dtype)),
    ("num.zeros", lambda dtype: num.zeros(2, dtype=dtype)),
    ("num.ones", lambda dtype: num.ones(2, dtype=dtype)),
    ("num.arange", lambda dtype: num.arange(2, dtype=dtype)),
)

SUPPORTED_DTYPES = (
    "'float64'/'float', 'float32'/'f32', 'complex128'/'complex', "
    "'int64'/'int'/'i64', 'int32'/'i32', 'int16'/'i16', 'int8'/'i8', "
    "'uint64'/'u64', 'uint32'/'u32', 'uint16'/'u16', 'uint8'/'u8', 'bool'"
)


@pytest.mark.parametrize(("constructor_name", "constructor"), CONSTRUCTORS)
@pytest.mark.parametrize(("canonical", "short", "expected", "_numpy_dtype"), UNSIGNED_DTYPES)
def test_every_integer_constructor_accepts_both_unsigned_spellings(
    constructor_name: str,
    constructor: Callable[[str], Array],
    canonical: str,
    short: str,
    expected: Any,
    _numpy_dtype: Any,
) -> None:
    """Every constructor that accepts signed widths accepts both unsigned spellings."""
    assert constructor(canonical).dtype == expected, constructor_name
    assert constructor(short).dtype == expected, constructor_name


@pytest.mark.parametrize(("canonical", "_short", "expected", "_numpy_dtype"), UNSIGNED_DTYPES)
@pytest.mark.parametrize("constructor", [num.array, num.asarray, pecos_rslib.array, Array])
def test_direct_unsigned_construction_matches_astype(
    constructor: Callable[..., Array],
    canonical: str,
    _short: str,
    expected: Any,
    _numpy_dtype: Any,
) -> None:
    """Direct construction and the previously-working astype path are equivalent."""
    direct = constructor([0, 1, 255], dtype=canonical)
    via_cast = constructor([0, 1, 255]).astype(expected)
    assert direct.dtype == via_cast.dtype == expected
    assert direct.tolist() == via_cast.tolist()


def test_uint8_honors_full_range_without_becoming_negative() -> None:
    result = num.array([0, 255], dtype=dtypes.uint8)
    assert result.tolist() == [0, 255]
    assert result[1] == 255
    assert isinstance(result[1], int)


def test_uint64_honors_full_range() -> None:
    maximum = 2**64 - 1
    assert num.array([maximum], dtype=dtypes.uint64).tolist() == [maximum]
    assert Array([maximum], dtype=dtypes.uint64).tolist() == [maximum]


@pytest.mark.parametrize(("_canonical", "_short", "expected", "numpy_dtype"), UNSIGNED_DTYPES)
def test_array_class_accepts_unsigned_numpy_buffers(
    _canonical: str,
    _short: str,
    expected: Any,
    numpy_dtype: Any,
) -> None:
    source = np.array([0, np.iinfo(numpy_dtype).max], dtype=numpy_dtype)
    actual = Array(source)
    assert actual.dtype == expected
    assert actual.tolist() == source.tolist()


@pytest.mark.parametrize("constructor", [num.zeros, num.ones])
def test_constructor_supported_dtype_message_is_complete(constructor: Callable[..., Array]) -> None:
    with pytest.raises(ValueError, match="^unsupported dtype: not-a-dtype") as exc_info:
        constructor(1, dtype="not-a-dtype")
    assert str(exc_info.value) == (f"unsupported dtype: not-a-dtype. Supported: {SUPPORTED_DTYPES}")


@pytest.mark.parametrize("constructor", [num.array, num.asarray])
def test_array_typestring_error_lists_unsigned_kind(constructor: Callable[..., Array]) -> None:
    with pytest.raises(TypeError) as exc_info:
        constructor(["2026-01-01"], dtype="datetime64[D]")
    assert str(exc_info.value).endswith("Supported typestring kinds: 'b', 'i', 'u', 'f', 'c'")


def test_array_class_typestring_error_lists_unsigned_kind() -> None:
    source = np.array(["2026-01-01"], dtype="datetime64[D]")
    with pytest.raises(TypeError) as exc_info:
        Array(source)
    assert str(exc_info.value).endswith("Supported typestring kinds: 'b', 'i', 'u', 'f', 'c'")


@pytest.mark.parametrize(
    ("reduction", "supported"),
    [
        (num.sum, "'b', 'i', 'u', 'f', 'c'"),
        (num.max, "'b', 'i', 'u', 'f'"),
        (num.min, "'b', 'i', 'u', 'f'"),
    ],
)
def test_reduction_dtype_kind_error_lists_unsigned_kind(reduction: Callable[..., Any], supported: str) -> None:
    source = np.array(["2026-01-01"], dtype="datetime64[D]")
    with pytest.raises(TypeError) as exc_info:
        reduction(source)
    assert str(exc_info.value) == f"Unsupported dtype kind: M. Supported: {supported}"


@pytest.mark.parametrize("value", [-1, 256])
def test_pure_constructor_rejects_out_of_range_uint8(value: int) -> None:
    with pytest.raises((OverflowError, TypeError)):
        Array([value], dtype=dtypes.uint8)


def test_arange_rejects_out_of_range_unsigned_values() -> None:
    with pytest.raises(OverflowError, match="out of range for uint8"):
        num.arange(255, 257, dtype=dtypes.uint8)


def test_unsigned_cast_directly() -> None:
    result = num.array([0, 254, 255], dtype=dtypes.uint8)
    assert result.astype(dtypes.uint16).tolist() == [0, 254, 255]
    assert result.astype(dtypes.int64).tolist() == [0, 254, 255]


def test_unsigned_indexing_directly() -> None:
    result = num.array([0, 254, 255], dtype=dtypes.uint8)
    assert result[2] == 255


def test_unsigned_arithmetic_directly() -> None:
    result = num.array([0, 254, 255], dtype=dtypes.uint8)
    assert (result + 0).tolist() == [0, 254, 255]


def test_unsigned_comparison_directly() -> None:
    result = num.array([0, 254, 255], dtype=dtypes.uint8)
    assert (result == num.array([0, 254, 255], dtype=dtypes.uint8)).tolist() == [True, True, True]


def test_unsigned_buffer_export_directly() -> None:
    result = num.array([0, 254, 255], dtype=dtypes.uint8)
    exported = np.asarray(result)
    assert exported.dtype == np.dtype(np.uint8)
    assert exported.tolist() == [0, 254, 255]


def test_unsigned_sum_directly() -> None:
    result = num.array([0, 254, 255], dtype=dtypes.uint8)
    assert num.sum(result) == 509


def test_unsigned_max_directly() -> None:
    result = num.array([0, 254, 255], dtype=dtypes.uint8)
    assert num.max(result) == 255


def test_unsigned_min_directly() -> None:
    result = num.array([0, 254, 255], dtype=dtypes.uint8)
    assert num.min(result) == 0


def test_unsigned_sum_rejects_uint64_overflow() -> None:
    maximum = 2**64 - 1
    with pytest.raises(OverflowError, match="sum exceeds uint64"):
        num.sum(num.array([maximum, 1], dtype=dtypes.uint64))


def test_unsigned_axis_reductions() -> None:
    result = num.array([[1, 255], [2, 3]], dtype=dtypes.uint8)
    assert num.sum(result, axis=0).dtype == dtypes.uint64
    assert num.sum(result, axis=0).tolist() == [3, 258]
    assert num.max(result, axis=0).tolist() == [2, 255]
    assert num.min(result, axis=0).tolist() == [1, 3]


def test_nonzero_matches_numpy() -> None:
    values = [[0, 2, 0], [3, 0, 4]]
    expected = np.nonzero(np.array(values, dtype=np.int64))
    actual = num.nonzero(num.array(values, dtype=dtypes.int64))
    assert isinstance(actual, tuple)
    assert len(actual) == len(expected)
    for actual_axis, expected_axis in zip(actual, expected, strict=True):
        assert actual_axis.dtype == dtypes.int64
        assert actual_axis.tolist() == expected_axis.tolist()


def test_nonzero_rejects_zero_dimensional_input_like_numpy() -> None:
    with pytest.raises(ValueError, match="nonzero on 0d arrays"):
        num.nonzero(1)


@pytest.mark.parametrize(("canonical", "_short", "expected", "numpy_dtype"), UNSIGNED_DTYPES)
def test_zeros_like_matches_numpy(
    canonical: str,
    _short: str,
    expected: Any,
    numpy_dtype: Any,
) -> None:
    source = num.array([[1, 2], [3, 4]], dtype=canonical)
    actual = num.zeros_like(source)
    oracle = np.zeros_like(np.array([[1, 2], [3, 4]], dtype=numpy_dtype))
    assert actual.shape == oracle.shape
    assert actual.dtype == expected
    assert actual.tolist() == oracle.tolist()


def test_issubdtype_integer_hierarchy_matches_numpy() -> None:
    dtype_pairs = (
        (dtypes.int8, np.int8),
        (dtypes.int16, np.int16),
        (dtypes.int32, np.int32),
        (dtypes.int64, np.int64),
        (dtypes.uint8, np.uint8),
        (dtypes.uint16, np.uint16),
        (dtypes.uint32, np.uint32),
        (dtypes.uint64, np.uint64),
        (dtypes.bool, np.bool_),
        (dtypes.float64, np.float64),
    )
    for pecos_dtype, numpy_dtype in dtype_pairs:
        assert num.issubdtype(pecos_dtype, num.integer) == np.issubdtype(numpy_dtype, np.integer)
    assert num.issubdtype(dtypes.uint8, dtypes.uint8) == np.issubdtype(np.uint8, np.uint8)
    assert num.issubdtype(dtypes.uint8, dtypes.uint16) == np.issubdtype(np.uint8, np.uint16)


def test_bool_scalar_dtype_matches_numpy_boolean_behavior() -> None:
    assert num.bool_ == dtypes.bool == dtypes.bool_
    assert num.array([0, 1], dtype=num.bool_).tolist() == [False, True]
    assert num.bool_(0) is bool(np.bool_(0))
    assert num.bool_(1) is bool(np.bool_(1))


def test_tolist_matches_numpy_for_native_scalar_types() -> None:
    cases = (
        ([1, 2], dtypes.int16, np.int16),
        ([1, 255], dtypes.uint8, np.uint8),
        ([1.25, 2.5], dtypes.float32, np.float32),
        ([True, False], dtypes.bool, np.bool_),
    )
    for values, pecos_dtype, numpy_dtype in cases:
        actual = num.array(values, dtype=pecos_dtype).tolist()
        expected = np.array(values, dtype=numpy_dtype).tolist()
        assert actual == expected
        assert [type(value) for value in actual] == [type(value) for value in expected]

    values_2d = [[1, 2], [3, 4]]
    assert num.array(values_2d, dtype=dtypes.uint16).tolist() == np.array(values_2d, dtype=np.uint16).tolist()


@pytest.mark.parametrize(
    "name",
    ["asarray", "nonzero", "zeros_like", "issubdtype", "integer"],
)
def test_pecos_reexports_complete_migration_surface(name: str) -> None:
    assert getattr(pc, name) is getattr(num, name)


def test_pecos_reexports_bool_dtype() -> None:
    assert pc.bool_ == num.bool_


def test_pecos_reexported_asarray_works() -> None:
    assert pc.asarray([1, 2], dtype=dtypes.uint8).tolist() == [1, 2]


@pytest.mark.parametrize("dtype_name", ["uint8", "uint64"])
@pytest.mark.parametrize("values", [[1, 0], [0, 0], [1, 1]])
def test_any_and_all_accept_unsigned_arrays(dtype_name: str, values: list[int]) -> None:
    """Widening the constructors made unsigned arrays reachable by any()/all().

    Both previously raised TypeError for unsigned input while accepting the
    matching signed array -- exactly the newly-reachable-range failure the
    downstream sweep exists to catch.
    """
    pecos_arr = num.array(values, dtype=getattr(dtypes, dtype_name))
    numpy_arr = np.array(values, dtype=getattr(np, dtype_name))

    assert num.any(pecos_arr) == bool(numpy_arr.any())
    assert num.all(pecos_arr) == bool(numpy_arr.all())
