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
from pecos_rslib import Array, Pauli, PauliString, dtypes, num

UNSIGNED_DTYPES = (
    ("uint8", "u8", dtypes.uint8, np.uint8),
    ("uint16", "u16", dtypes.uint16, np.uint16),
    ("uint32", "u32", dtypes.uint32, np.uint32),
    ("uint64", "u64", dtypes.uint64, np.uint64),
)

ARRAY_INTERFACE_DTYPES = (
    ("bool", np.bool_, [False, True]),
    ("int8", np.int8, [1, 127]),
    ("int16", np.int16, [1, 256]),
    ("int32", np.int32, [1, 256]),
    ("int64", np.int64, [1, 256]),
    ("uint8", np.uint8, [1, 255]),
    ("uint16", np.uint16, [1, 256]),
    ("uint32", np.uint32, [1, 256]),
    ("uint64", np.uint64, [1, 256]),
    ("float32", np.float32, [1.5, 2.5]),
    ("float64", np.float64, [1.5, 2.5]),
    ("complex64", np.complex64, [1.5 + 2.25j, 2.5 - 3.25j]),
    ("complex128", np.complex128, [1.5 + 2.25j, 2.5 - 3.25j]),
)

ARRAY_EQUAL_CUSTOM_DTYPES = (
    ("pauli", [Pauli.I, Pauli.X, Pauli.Z], [Pauli.I, Pauli.Y, Pauli.Z]),
    (
        "paulistring",
        [PauliString.from_dense_str(value) for value in ("I", "X", "Z")],
        [PauliString.from_dense_str(value) for value in ("I", "Y", "Z")],
    ),
)

SCALAR_DTYPES = (
    ("bool", np.bool_, True),
    ("int8", np.int8, 7),
    ("int16", np.int16, 7),
    ("int32", np.int32, 7),
    ("int64", np.int64, 7),
    ("uint8", np.uint8, 7),
    ("uint16", np.uint16, 7),
    ("uint32", np.uint32, 7),
    ("uint64", np.uint64, 7),
    ("float32", np.float32, 7.5),
    ("float64", np.float64, 7.5),
    ("complex64", np.complex64, 7.5 + 2.25j),
    ("complex128", np.complex128, 7.5 + 2.25j),
)

TOLIST_SHAPES = ((), (3,), (2, 3), (2, 2, 2), (0,), (2, 0), (2, 0, 3))

INDEX_ASSIGNMENT_DTYPES = (
    ("bool", dtypes.bool, np.bool_, True, [True, False, True]),
    ("int8", dtypes.int8, np.int8, 7, [2, 3, 4]),
    ("int16", dtypes.int16, np.int16, 7, [2, 3, 4]),
    ("int32", dtypes.int32, np.int32, 7, [2, 3, 4]),
    ("int64", dtypes.int64, np.int64, 7, [2, 3, 4]),
    ("uint8", dtypes.uint8, np.uint8, 7, [2, 3, 4]),
    ("uint16", dtypes.uint16, np.uint16, 7, [2, 3, 4]),
    ("uint32", dtypes.uint32, np.uint32, 7, [2, 3, 4]),
    ("uint64", dtypes.uint64, np.uint64, 7, [2, 3, 4]),
    ("float32", dtypes.float32, np.float32, 7.5, [2.5, 3.5, 4.5]),
    ("float64", dtypes.float64, np.float64, 7.5, [2.5, 3.5, 4.5]),
    (
        "complex64",
        dtypes.complex64,
        np.complex64,
        7.5 + 1.25j,
        [2.5 + 1j, 3.5 - 2j, 4.5 + 3j],
    ),
    (
        "complex128",
        dtypes.complex128,
        np.complex128,
        7.5 + 1.25j,
        [2.5 + 1j, 3.5 - 2j, 4.5 + 3j],
    ),
)

AXIS_REDUCTIONS = (
    ("sum", num.sum, np.sum),
    ("max", num.max, np.max),
    ("min", num.min, np.min),
    ("any", num.any, np.any),
    ("all", num.all, np.all),
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
@pytest.mark.parametrize(("canonical", "_short", "expected", "_numpy_dtype"), UNSIGNED_DTYPES)
def test_every_integer_constructor_accepts_long_unsigned_spelling(
    constructor_name: str,
    constructor: Callable[[str], Array],
    canonical: str,
    _short: str,
    expected: Any,
    _numpy_dtype: Any,
) -> None:
    """Every constructor that accepts signed widths accepts long unsigned names."""
    assert constructor(canonical).dtype == expected, constructor_name


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


@pytest.mark.parametrize("constructor", [Array, pecos_rslib.array, num.array, num.asarray])
@pytest.mark.parametrize(
    "values",
    [
        [0, -1, 2, 2**31],
        [0.0, -1.5, 2.25, float("inf")],
    ],
    ids=["builtin-int-list", "builtin-float-list"],
)
def test_flat_builtin_list_fast_path_matches_numpy(
    constructor: Callable[..., Array], values: list[int] | list[float]
) -> None:
    expected = np.array(values)
    actual = constructor(values)

    assert np.asarray(actual).dtype == expected.dtype
    np.testing.assert_array_equal(np.asarray(actual), expected)


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
def test_constructor_supported_dtype_message_is_complete(
    constructor: Callable[..., Array],
) -> None:
    with pytest.raises(ValueError, match="^unsupported dtype: not-a-dtype") as exc_info:
        constructor(1, dtype="not-a-dtype")
    assert str(exc_info.value) == (f"unsupported dtype: not-a-dtype. Supported: {SUPPORTED_DTYPES}")


@pytest.mark.parametrize("constructor", [num.array, num.asarray])
def test_array_typestring_error_lists_unsigned_kind(
    constructor: Callable[..., Array],
) -> None:
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
    assert (result == num.array([0, 254, 255], dtype=dtypes.uint8)).tolist() == [
        True,
        True,
        True,
    ]


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


@pytest.mark.parametrize(("name", "numpy_dtype", "_value"), SCALAR_DTYPES)
@pytest.mark.parametrize("shape", TOLIST_SHAPES)
def test_tolist_matches_numpy_for_every_numeric_dtype_and_shape(
    name: str,
    numpy_dtype: Any,
    _value: Any,
    shape: tuple[int, ...],
) -> None:
    size = int(np.prod(shape, dtype=np.int64)) if shape else 1
    values = np.arange(size, dtype=np.float64)
    if np.issubdtype(numpy_dtype, np.complexfloating):
        values = values + (values + 0.5) * 1j
    elif numpy_dtype is np.bool_:
        values = values % 2 == 0
    expected_array = np.asarray(values, dtype=numpy_dtype).reshape(shape)

    actual = Array(expected_array, dtype=getattr(dtypes, name)).tolist()
    expected = expected_array.tolist()

    def scalar_types(value: Any) -> Any:
        if isinstance(value, list):
            return [scalar_types(item) for item in value]
        return type(value)

    assert actual == expected
    assert scalar_types(actual) == scalar_types(expected)


@pytest.mark.parametrize("shape", TOLIST_SHAPES)
@pytest.mark.parametrize("kind", ["pauli", "paulistring"])
def test_tolist_preserves_custom_dtype_objects_and_shape(
    shape: tuple[int, ...],
    kind: str,
) -> None:
    size = int(np.prod(shape, dtype=np.int64)) if shape else 1
    if kind == "pauli":
        values = [Pauli.I, Pauli.X, Pauli.Y, Pauli.Z]
        dtype = dtypes.pauli
    else:
        values = [
            PauliString.from_dense_str("I"),
            PauliString.from_dense_str("X"),
            PauliString.from_dense_str("Y"),
            PauliString.from_dense_str("Z"),
        ]
        dtype = dtypes.paulistring
    flat_values = [values[index % len(values)] for index in range(size)]
    expected = np.asarray(flat_values, dtype=object).reshape(shape).tolist()

    actual = Array(flat_values, dtype=dtype).reshape(shape).tolist()

    assert actual == expected


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


@pytest.mark.parametrize(("dtype_name", "numpy_dtype", "values"), ARRAY_INTERFACE_DTYPES)
@pytest.mark.parametrize("input_kind", ["native", "numpy", "list"])
def test_array_equal_same_numeric_dtype_matches_numpy_for_every_input_kind(
    dtype_name: str,
    numpy_dtype: Any,
    values: list[Any],
    input_kind: str,
) -> None:
    equal_left = np.asarray(values, dtype=numpy_dtype)
    equal_right = equal_left.copy()
    unequal_right = equal_left.copy()
    unequal_right[-1] = equal_left[0]

    def as_input(value: np.ndarray) -> Any:
        if input_kind == "native":
            return Array(value, dtype=getattr(dtypes, dtype_name))
        if input_kind == "list":
            return value.tolist()
        return value

    assert np.array_equal(equal_left, equal_right)
    assert not np.array_equal(equal_left, unequal_right)
    assert pc.array_equal(as_input(equal_left), as_input(equal_right))
    assert not pc.array_equal(as_input(equal_left), as_input(unequal_right))


@pytest.mark.parametrize(("dtype_name", "equal_values", "unequal_values"), ARRAY_EQUAL_CUSTOM_DTYPES)
@pytest.mark.parametrize("input_kind", ["native", "numpy", "list"])
def test_array_equal_custom_dtype_matches_numpy_object_oracle(
    dtype_name: str,
    equal_values: list[Any],
    unequal_values: list[Any],
    input_kind: str,
) -> None:
    equal_left = np.asarray(equal_values, dtype=object)
    equal_right = np.asarray(equal_values, dtype=object)
    unequal_right = np.asarray(unequal_values, dtype=object)

    def as_input(value: np.ndarray) -> Any:
        values = value.tolist()
        if input_kind == "native":
            return Array(values, dtype=getattr(dtypes, dtype_name))
        if input_kind == "list":
            return values
        return value

    assert np.array_equal(equal_left, equal_right)
    assert not np.array_equal(equal_left, unequal_right)
    assert pc.array_equal(as_input(equal_left), as_input(equal_right))
    assert not pc.array_equal(as_input(equal_left), as_input(unequal_right))


@pytest.mark.parametrize(("left_name", "left_dtype", "_left_values"), ARRAY_INTERFACE_DTYPES)
@pytest.mark.parametrize(("right_name", "right_dtype", "_right_values"), ARRAY_INTERFACE_DTYPES)
def test_array_equal_all_numeric_dtype_pairs_match_numpy_for_equal_values(
    left_name: str,
    left_dtype: Any,
    _left_values: list[Any],
    right_name: str,
    right_dtype: Any,
    _right_values: list[Any],
) -> None:
    left = np.asarray([0, 1], dtype=left_dtype)
    right = np.asarray([0, 1], dtype=right_dtype)
    expected = np.array_equal(left, right)

    assert pc.array_equal(left, right) is expected, (left_name, right_name, "numpy")
    assert pc.array_equal(Array(left), Array(right)) is expected, (left_name, right_name, "native")


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (np.asarray([0, 255], dtype=np.uint8), np.asarray([0, 255], dtype=np.int64)),
        (np.asarray([-2, 3], dtype=np.int32), np.asarray([-2.0, 3.0], dtype=np.float64)),
    ],
)
def test_array_equal_mixed_dtype_true_cases_match_numpy(left: np.ndarray, right: np.ndarray) -> None:
    assert np.array_equal(left, right)
    assert pc.array_equal(Array(left), Array(right))
    assert pc.array_equal(left, right)


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (np.asarray([1, 2], dtype=np.uint8), np.asarray([1, 3], dtype=np.int64)),
        (np.asarray([[1, 2]], dtype=np.int32), np.asarray([1, 2], dtype=np.int64)),
        (np.asarray([16_777_217], dtype=np.int32), np.asarray([16_777_216], dtype=np.float32)),
        (np.asarray([2**63], dtype=np.uint64), np.asarray([2**63 - 1], dtype=np.int64)),
    ],
)
def test_array_equal_mixed_dtype_false_cases_match_numpy(left: np.ndarray, right: np.ndarray) -> None:
    assert not np.array_equal(left, right)
    assert not pc.array_equal(Array(left), Array(right))
    assert not pc.array_equal(left, right)


def test_array_equal_uses_numpy_promotion_before_value_comparison() -> None:
    left = np.asarray([2**53 + 1], dtype=np.int64)
    right = np.asarray([float(2**53)], dtype=np.float64)

    assert np.array_equal(left, right)
    assert pc.array_equal(Array(left), Array(right))
    assert (Array(left) == Array(right)).tolist() == (left == right).tolist()


@pytest.mark.parametrize("numpy_dtype", [np.float32, np.float64, np.complex64, np.complex128])
@pytest.mark.parametrize("input_kind", ["native", "numpy", "list"])
def test_array_equal_equal_nan_matches_numpy(
    numpy_dtype: Any,
    input_kind: str,
) -> None:
    if np.issubdtype(numpy_dtype, np.complexfloating):
        left = np.asarray([complex(np.nan, 1), complex(2, np.nan)], dtype=numpy_dtype)
        right = np.asarray([complex(np.nan, 9), complex(7, np.nan)], dtype=numpy_dtype)
    else:
        left = np.asarray([1, np.nan], dtype=numpy_dtype)
        right = left.copy()

    def as_input(value: np.ndarray) -> Any:
        if input_kind == "native":
            return Array(value)
        if input_kind == "list":
            return value.tolist()
        return value

    assert pc.array_equal(as_input(left), as_input(right), equal_nan=False) is np.array_equal(
        left, right, equal_nan=False
    )
    assert pc.array_equal(as_input(left), as_input(right), equal_nan=True) is np.array_equal(
        left, right, equal_nan=True
    )


@pytest.mark.parametrize(
    "custom_values",
    [
        [Pauli.I, Pauli.X],
        [PauliString.from_dense_str("I"), PauliString.from_dense_str("X")],
    ],
)
def test_array_equal_pauli_vs_numeric_is_false_in_both_orders(custom_values: list[Any]) -> None:
    custom = Array(custom_values)
    numeric = Array([0, 1], dtype=dtypes.uint8)

    assert not pc.array_equal(custom, numeric)
    assert not pc.array_equal(numeric, custom)


def test_array_equal_different_pauli_kinds_are_unequal() -> None:
    pauli = Array([Pauli.X])
    pauli_string = Array([PauliString.from_dense_str("X")])

    assert not pc.array_equal(pauli, pauli_string)
    assert not pc.array_equal(pauli_string, pauli)


def test_array_equal_uint8_check_matrix_regression() -> None:
    left = Array([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=dtypes.uint8)
    equal = Array([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=dtypes.uint8)
    bit_flipped = Array([[1, 0, 1, 0], [0, 1, 1, 1]], dtype=dtypes.uint8)

    assert pc.array_equal(left, equal)
    assert not pc.array_equal(left, bit_flipped)


@pytest.mark.parametrize("dtype_name", ["uint8", "uint16", "uint32", "uint64"])
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


@pytest.mark.parametrize(("_dtype_name", "numpy_dtype", "values"), ARRAY_INTERFACE_DTYPES)
@pytest.mark.parametrize("constructor", [Array, num.array])
def test_big_endian_ingest_matches_little_endian_and_round_trips(
    constructor: Callable[..., Array],
    _dtype_name: str,
    numpy_dtype: Any,
    values: list[Any],
) -> None:
    little_endian = np.array(values, dtype=np.dtype(numpy_dtype).newbyteorder("<"))
    big_endian = np.array(values, dtype=np.dtype(numpy_dtype).newbyteorder(">"))

    little_actual = constructor(little_endian)
    big_actual = constructor(big_endian)

    assert big_actual.dtype == little_actual.dtype
    assert big_actual.tolist() == little_actual.tolist() == big_endian.tolist()
    np.testing.assert_array_equal(np.asarray(big_actual), little_endian)


@pytest.mark.parametrize("constructor", [Array, num.array])
def test_float16_array_interface_remains_rejected(
    constructor: Callable[..., Array],
) -> None:
    with pytest.raises(TypeError, match="Unsupported dtype"):
        constructor(np.array([1.5], dtype=np.float16))


@pytest.mark.parametrize(("_canonical", "_short", "target", "_numpy_dtype"), UNSIGNED_DTYPES)
def test_native_array_unsigned_conversions_use_sequence_range_checks(
    _canonical: str,
    _short: str,
    target: Any,
    _numpy_dtype: Any,
) -> None:
    with pytest.raises(OverflowError) as sequence_error:
        Array([-1], dtype=target)

    source = Array([-1], dtype=dtypes.int64)
    for conversion in (
        lambda: Array(source, dtype=target),
        lambda: source.astype(target),
    ):
        with pytest.raises(type(sequence_error.value)) as native_error:
            conversion()
        assert str(native_error.value) == str(sequence_error.value)


def test_native_uint64_to_int64_conversion_rejects_wrap() -> None:
    maximum = 2**64 - 1
    with pytest.raises(OverflowError, match=f"value {maximum} is out of range for int64"):
        Array(Array([maximum], dtype=dtypes.uint64), dtype=dtypes.int64)


@pytest.mark.parametrize(("canonical", "_short", "target", "_numpy_dtype"), UNSIGNED_DTYPES)
@pytest.mark.parametrize("constructor", [Array, pecos_rslib.array, num.array, num.asarray])
def test_integer_like_elements_share_checked_unsigned_conversion(
    constructor: Callable[..., Array],
    canonical: str,
    _short: str,
    target: Any,
    _numpy_dtype: Any,
) -> None:
    errors = []
    for value in (-1, np.int64(-1)):
        with pytest.raises(OverflowError) as error:
            constructor([value], dtype=target)
        errors.append(str(error.value))
    assert errors[0] == errors[1] == f"value -1 is out of range for {canonical}"
    assert constructor([False, True], dtype=target).tolist() == [0, 1]


@pytest.mark.parametrize(("reduction", "numpy_reduction"), [(num.max, np.max), (num.min, np.min)])
@pytest.mark.parametrize(("_canonical", "_short", "target", "numpy_dtype"), UNSIGNED_DTYPES)
def test_empty_unsigned_axis_extreme_raises_like_axisless_case(
    reduction: Callable[..., Any],
    numpy_reduction: Callable[..., Any],
    _canonical: str,
    _short: str,
    target: Any,
    numpy_dtype: Any,
) -> None:
    source = num.array(np.empty((2, 0), dtype=numpy_dtype), dtype=target)
    with pytest.raises(ValueError, match="^zero-size array reduction has no identity$"):
        reduction(source, axis=1)
    with pytest.raises(ValueError, match="zero-size array to reduction operation"):
        numpy_reduction(np.empty((2, 0), dtype=numpy_dtype), axis=1)


@pytest.mark.parametrize(("reduction", "numpy_reduction"), [(num.max, np.max), (num.min, np.min)])
@pytest.mark.parametrize("numpy_dtype", [np.bool_, np.int64, np.float32, np.float64])
def test_empty_native_axis_extreme_raises_for_every_ordered_dtype(
    reduction: Callable[..., Any], numpy_reduction: Callable[..., Any], numpy_dtype: Any
) -> None:
    source = Array(np.empty((2, 0), dtype=numpy_dtype))
    with pytest.raises(ValueError, match="^zero-size array reduction has no identity$"):
        reduction(source, axis=1)
    with pytest.raises(ValueError, match="zero-size array to reduction operation"):
        numpy_reduction(np.empty((2, 0), dtype=numpy_dtype), axis=1)


@pytest.mark.parametrize(("_canonical", "_short", "expected", "numpy_dtype"), UNSIGNED_DTYPES)
def test_reversed_unsigned_numpy_views_are_copied_in_logical_order(
    _canonical: str,
    _short: str,
    expected: Any,
    numpy_dtype: Any,
) -> None:
    source = np.arange(6, dtype=numpy_dtype)[::-1]
    actual = Array(source)
    assert actual.dtype == expected
    assert actual.tolist() == source.tolist() == [5, 4, 3, 2, 1, 0]


def test_numpy_ingest_applies_requested_dtype_with_checked_conversion() -> None:
    actual = Array(np.array([255], dtype=np.uint8), dtype=dtypes.int16)
    assert actual.dtype == dtypes.int16
    assert actual.tolist() == [255]

    with pytest.raises(OverflowError, match="out of range for int8"):
        Array(np.array([255], dtype=np.uint8), dtype=dtypes.int8)


@pytest.mark.parametrize(("_canonical", "_short", "pecos_dtype", "numpy_dtype"), UNSIGNED_DTYPES)
def test_all_unsigned_reductions_match_numpy(
    _canonical: str,
    _short: str,
    pecos_dtype: Any,
    numpy_dtype: Any,
) -> None:
    values = [[0, 1], [2, 3]]
    actual = num.array(values, dtype=pecos_dtype)
    expected = np.array(values, dtype=numpy_dtype)
    assert num.sum(actual) == int(np.sum(expected))
    assert num.max(actual) == int(np.max(expected))
    assert num.min(actual) == int(np.min(expected))
    assert num.sum(actual, axis=0).tolist() == np.sum(expected, axis=0).tolist()
    assert num.max(actual, axis=0).tolist() == np.max(expected, axis=0).tolist()
    assert num.min(actual, axis=0).tolist() == np.min(expected, axis=0).tolist()


@pytest.mark.parametrize(("_canonical", "_short", "pecos_dtype", "numpy_dtype"), UNSIGNED_DTYPES)
def test_all_unsigned_any_all_axis_and_axisless_match_numpy(
    _canonical: str,
    _short: str,
    pecos_dtype: Any,
    numpy_dtype: Any,
) -> None:
    values = [[0, 1], [1, 1]]
    actual = num.array(values, dtype=pecos_dtype)
    expected = np.array(values, dtype=numpy_dtype)
    assert num.any(actual) == bool(np.any(expected))
    assert num.all(actual) == bool(np.all(expected))
    assert num.any(actual, axis=1).tolist() == np.any(expected, axis=1).tolist()
    assert num.all(actual, axis=1).tolist() == np.all(expected, axis=1).tolist()


@pytest.mark.parametrize(("dtype_name", "numpy_dtype", "_values"), ARRAY_INTERFACE_DTYPES)
def test_native_array_sum_uses_all_numeric_dtypes(
    dtype_name: str,
    numpy_dtype: Any,
    _values: list[Any],
) -> None:
    expected = np.array([[1, 2], [3, 4]], dtype=numpy_dtype)
    actual = Array(expected)

    assert num.sum(actual) == np.sum(expected).item(), dtype_name
    np.testing.assert_array_equal(np.asarray(num.sum(actual, axis=0)), np.sum(expected, axis=0))


@pytest.mark.parametrize(
    ("dtype_name", "numpy_dtype", "_values"),
    [case for case in ARRAY_INTERFACE_DTYPES if not case[0].startswith("complex")],
)
@pytest.mark.parametrize(("reduction", "numpy_reduction"), [(num.max, np.max), (num.min, np.min)])
def test_native_array_extremes_use_all_ordered_numeric_dtypes(
    dtype_name: str,
    numpy_dtype: Any,
    _values: list[Any],
    reduction: Callable[..., Any],
    numpy_reduction: Callable[..., Any],
) -> None:
    expected = np.array([[1, 2], [3, 4]], dtype=numpy_dtype)
    actual = Array(expected)

    assert reduction(actual) == numpy_reduction(expected).item(), dtype_name
    np.testing.assert_array_equal(np.asarray(reduction(actual, axis=1)), numpy_reduction(expected, axis=1))


@pytest.mark.parametrize(("reduction", "numpy_reduction"), [(num.max, np.max), (num.min, np.min)])
def test_native_int64_axis0_row_sweep_matches_numpy(
    reduction: Callable[..., Any], numpy_reduction: Callable[..., Any]
) -> None:
    expected = np.array(
        [
            [np.iinfo(np.int64).min, 5, 9, -3],
            [4, np.iinfo(np.int64).max, -8, 7],
            [2, -6, 3, 0],
        ],
        dtype=np.int64,
    )

    np.testing.assert_array_equal(np.asarray(reduction(Array(expected), axis=0)), numpy_reduction(expected, axis=0))


@pytest.mark.parametrize(("dtype_name", "numpy_dtype", "_values"), ARRAY_INTERFACE_DTYPES)
@pytest.mark.parametrize(("reduction", "numpy_reduction"), [(num.any, np.any), (num.all, np.all)])
def test_native_array_truth_reductions_use_all_numeric_dtypes(
    dtype_name: str,
    numpy_dtype: Any,
    _values: list[Any],
    reduction: Callable[..., Any],
    numpy_reduction: Callable[..., Any],
) -> None:
    expected = np.array([[0, 1], [2, 0]], dtype=numpy_dtype)
    actual = Array(expected)

    assert reduction(actual) == numpy_reduction(expected).item(), dtype_name
    np.testing.assert_array_equal(np.asarray(reduction(actual, axis=0)), numpy_reduction(expected, axis=0))


@pytest.mark.parametrize(
    ("dtype_name", "numpy_dtype", "_values"),
    [case for case in ARRAY_INTERFACE_DTYPES if case[0] != "bool" and not case[0].startswith("complex")],
)
def test_native_array_mean_uses_all_supported_numeric_dtypes(
    dtype_name: str,
    numpy_dtype: Any,
    _values: list[Any],
) -> None:
    expected = np.array([[1, 2], [3, 4]], dtype=numpy_dtype)
    actual = Array(expected)

    assert num.mean(actual) == np.mean(expected).item(), dtype_name
    np.testing.assert_array_equal(np.asarray(num.mean(actual, axis=0)), np.mean(expected, axis=0))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_native_float_reductions_preserve_nan_inf_and_axis_results(dtype: Any) -> None:
    values = np.array(
        [
            [np.nan, np.inf, -np.inf, 1.0],
            [2.0, np.nan, 3.0, -4.0],
            [5.0, 6.0, 7.0, 8.0],
        ],
        dtype=dtype,
    )
    actual = Array(values)

    assert np.isnan(num.sum(actual))
    assert np.isnan(num.max(actual))
    assert np.isnan(num.min(actual))
    for axis in (0, 1):
        np.testing.assert_array_equal(np.asarray(num.sum(actual, axis=axis)), np.sum(values, axis=axis))
        np.testing.assert_array_equal(np.asarray(num.max(actual, axis=axis)), np.max(values, axis=axis))
        np.testing.assert_array_equal(np.asarray(num.min(actual, axis=axis)), np.min(values, axis=axis))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("nan_index", [0, 1, 2])
def test_native_float_extremes_propagate_nan_independent_of_position(dtype: Any, nan_index: int) -> None:
    values = np.array([1.0, 2.0, 3.0], dtype=dtype)
    values[nan_index] = np.nan
    actual = Array(values)

    assert np.isnan(num.max(actual))
    assert np.isnan(num.min(actual))

    if dtype == np.float64:
        assert np.isnan(num.max(values))
        assert np.isnan(num.min(values))
        assert np.isnan(num.max(values.tolist()))
        assert np.isnan(num.min(values.tolist()))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("values", [[0.0, -0.0], [-0.0, 0.0]])
@pytest.mark.parametrize(("reduction", "numpy_reduction"), [(num.max, np.max), (num.min, np.min)])
def test_native_float_extremes_preserve_numpy_signed_zero(
    dtype: Any,
    values: list[float],
    reduction: Callable[..., Any],
    numpy_reduction: Callable[..., Any],
) -> None:
    expected = np.array(values, dtype=dtype)
    actual = reduction(Array(expected))

    assert actual == 0.0
    assert np.signbit(actual) == np.signbit(numpy_reduction(expected))

    expected_axis = np.array([values, values[::-1]], dtype=dtype)
    actual_axis = np.asarray(reduction(Array(expected_axis), axis=0))
    np.testing.assert_array_equal(np.signbit(actual_axis), np.signbit(numpy_reduction(expected_axis, axis=0)))


@pytest.mark.performance
def test_native_float64_sum_does_not_reenter_python_per_element() -> None:
    """Release-only guard for the native-Array dispatch regression in PECOS #505."""
    import time

    values = Array(np.arange(1_000_000, dtype=np.float64))
    num.sum(values)
    samples = []
    for _ in range(3):
        start = time.perf_counter()
        num.sum(values)
        samples.append(time.perf_counter() - start)

    assert min(samples) < 0.05, f"native float64 sum took {min(samples) * 1000:.1f} ms"


@pytest.mark.parametrize("value", [0, 1, False, True, np.int64(0), np.int64(1)])
def test_bool_dtype_scalar_likes_match_numpy(value: Any) -> None:
    assert num.bool_(value) is bool(np.bool_(value))


@pytest.mark.parametrize("values", [[], [0, 1]])
def test_bool_dtype_rejects_ambiguous_native_arrays_like_numpy_truth(
    values: list[int],
) -> None:
    oracle = np.array(values, dtype=np.uint8)
    with pytest.raises(ValueError, match="truth value") as numpy_error:
        bool(oracle)
    with pytest.raises(type(numpy_error.value)) as pecos_error:
        num.bool_(Array(values, dtype=dtypes.uint8))
    assert str(pecos_error.value) == str(numpy_error.value)


@pytest.mark.parametrize("value", [0, 1])
def test_bool_dtype_size_one_native_array_uses_element_truth(value: int) -> None:
    assert num.bool_(Array([value], dtype=dtypes.uint8)) is bool(np.array([value], dtype=np.uint8))


@pytest.mark.parametrize(
    ("spelling", "expected"),
    [
        ("i1", dtypes.int8),
        ("i2", dtypes.int16),
        ("i4", dtypes.int32),
        ("i8", dtypes.int64),
        ("u1", dtypes.uint8),
        ("u2", dtypes.uint16),
        ("u4", dtypes.uint32),
        ("u8", dtypes.uint64),
        ("int8", dtypes.int8),
        ("uint8", dtypes.uint8),
        ("i64", dtypes.int64),
        ("u64", dtypes.uint64),
    ],
)
@pytest.mark.parametrize("constructor", [num.array, num.asarray])
def test_num_array_dtype_spelling_compatibility(
    constructor: Callable[..., Array], spelling: str, expected: Any
) -> None:
    actual = constructor([1], dtype=spelling)
    assert actual.dtype == expected


def test_i8_string_keeps_dev_numpy_typestring_behavior() -> None:
    assert num.array([128], dtype="i8").tolist() == [128]
    assert num.array([128], dtype="i8").dtype == dtypes.int64


SHAPE_METHOD_DTYPES = (
    (dtypes.bool, np.bool_),
    (dtypes.int8, np.int8),
    (dtypes.int64, np.int64),
    (dtypes.uint8, np.uint8),
    (dtypes.uint64, np.uint64),
    (dtypes.float32, np.float32),
    (dtypes.float64, np.float64),
    (dtypes.complex128, np.complex128),
)

INTEGER_DTYPES = (
    ("int8", dtypes.int8),
    ("int16", dtypes.int16),
    ("int32", dtypes.int32),
    ("int64", dtypes.int64),
    ("uint8", dtypes.uint8),
    ("uint16", dtypes.uint16),
    ("uint32", dtypes.uint32),
    ("uint64", dtypes.uint64),
)

SHAPE_METHOD_CASES = (
    ((6,), (2, 3)),
    ((2, 3), (3, 2)),
    ((1, 2, 3), (3, 2)),
    ((2, 0, 3), (0, 6)),
    ((1, 1, 1), (1,)),
)


def shape_method_values(shape: tuple[int, ...], numpy_dtype: Any) -> np.ndarray[Any, Any]:
    size = int(np.prod(shape, dtype=np.int64))
    values = np.arange(size, dtype=np.float64).reshape(shape)
    if numpy_dtype is np.bool_:
        values = values % 2 == 0
    elif numpy_dtype is np.complex128:
        values = values + 0.5j
    return values.astype(numpy_dtype)


def fill_value_for_dtype(numpy_dtype: Any) -> Any:
    if numpy_dtype is np.bool_:
        return False
    if numpy_dtype is np.complex128:
        return 3 + 2j
    return 3


@pytest.mark.parametrize(("pecos_dtype", "numpy_dtype"), SHAPE_METHOD_DTYPES)
@pytest.mark.parametrize(("source_shape", "target_shape"), SHAPE_METHOD_CASES)
def test_array_shape_methods_match_numpy_across_ranks_sizes_and_dtypes(
    pecos_dtype: Any,
    numpy_dtype: Any,
    source_shape: tuple[int, ...],
    target_shape: tuple[int, ...],
) -> None:
    source = shape_method_values(source_shape, numpy_dtype)
    actual = Array(source, dtype=pecos_dtype)

    assert actual.flatten().tolist() == source.flatten().tolist()
    assert actual.ravel().tolist() == source.ravel().tolist()
    assert actual.reshape(target_shape).tolist() == source.reshape(target_shape).tolist()

    expected_fill = source.copy()
    expected_fill.fill(fill_value_for_dtype(numpy_dtype))
    assert actual.fill(fill_value_for_dtype(numpy_dtype)) is None
    assert actual.tolist() == expected_fill.tolist()


@pytest.mark.parametrize("method_name", ["flatten", "ravel"])
def test_flatten_and_ravel_return_independent_copies(method_name: str) -> None:
    original = Array([0, 1, 2, 3, 4, 5], dtype=dtypes.int64)
    result = getattr(original, method_name)()

    result[0] = 99

    assert result.tolist() == [99, 1, 2, 3, 4, 5]
    assert original.tolist() == [0, 1, 2, 3, 4, 5]


@pytest.mark.parametrize("method_name", ["flatten", "ravel"])
def test_flatten_and_ravel_use_logical_c_order_for_transposed_arrays(
    method_name: str,
) -> None:
    source = np.arange(12, dtype=np.int64).reshape(3, 4).T
    actual = Array(source).T
    expected = np.asarray(source).T

    assert getattr(actual, method_name)().tolist() == getattr(expected, method_name)().tolist()


@pytest.mark.parametrize("target_shape", [(3, 8), (2, 3, 4), (24,), (1, 4, 2, 3)])
def test_reshape_valid_shapes_match_numpy(target_shape: tuple[int, ...]) -> None:
    source = np.arange(24, dtype=np.int64).reshape(2, 3, 4)
    actual = Array(source)

    assert actual.reshape(target_shape).tolist() == source.reshape(target_shape).tolist()
    assert actual.reshape(*target_shape).tolist() == source.reshape(*target_shape).tolist()


def test_reshape_sequence_and_varargs_forms_match_numpy() -> None:
    source = np.arange(4, dtype=np.int64)
    actual = Array(source)

    for target_shape in ([2, 2], (2, 2)):
        result = actual.reshape(target_shape)
        expected = source.reshape(target_shape)
        assert result.shape == expected.shape
        assert result.tolist() == expected.tolist()

    result = actual.reshape(2, 2)
    expected = source.reshape(2, 2)
    assert result.shape == expected.shape
    assert result.tolist() == expected.tolist()


@pytest.mark.parametrize("target_shape", [(-1,), (2, -1), (-1, 3, 2)])
def test_reshape_minus_one_inference_matches_numpy(
    target_shape: tuple[int, ...],
) -> None:
    source = np.arange(24, dtype=np.int64).reshape(2, 3, 4)
    actual = Array(source)
    expected = source.reshape(target_shape)

    result = actual.reshape(target_shape)
    assert result.shape == expected.shape
    assert result.tolist() == expected.tolist()


def test_reshape_minus_one_infers_zero_for_empty_array_like_numpy() -> None:
    source = np.empty((2, 0, 3), dtype=np.float64)
    actual = Array(source)

    assert actual.reshape(-1).shape == source.reshape(-1).shape == (0,)


def test_reshape_mismatched_element_count_names_size_and_requested_shape() -> None:
    actual = Array([[0, 1, 2], [3, 4, 5]], dtype=dtypes.int64)

    with pytest.raises(ValueError, match="size 6.*requested shape") as error:
        actual.reshape(4, 2)

    assert "size 6" in str(error.value)
    assert "requested shape (4, 2)" in str(error.value)


def test_reshape_rejects_two_inferred_dimensions() -> None:
    with pytest.raises(ValueError, match="only specify one unknown dimension"):
        Array([0, 1, 2, 3]).reshape(-1, -1)


def test_reshape_rejects_non_minus_one_negative_dimension() -> None:
    """PECOS documents literal -1 inference; NumPy accepts any single negative value."""
    assert np.arange(4).reshape(-2).tolist() == [0, 1, 2, 3]

    with pytest.raises(ValueError, match="negative dimensions are not allowed"):
        Array([0, 1, 2, 3]).reshape(-2)


def test_reshape_rejects_inference_that_does_not_divide_evenly() -> None:
    with pytest.raises(ValueError, match="size 5.*requested shape") as error:
        Array([0, 1, 2, 3, 4]).reshape(2, -1)

    assert "size 5" in str(error.value)
    assert "requested shape (2, -1)" in str(error.value)


@pytest.mark.parametrize("bad_shape", [1.5, "6", True, (2, 1.5)])
def test_reshape_rejects_non_integer_shape(bad_shape: object) -> None:
    with pytest.raises(TypeError, match="shape dimensions must be integers"):
        Array([0, 1, 2, 3, 4, 5]).reshape(bad_shape)


def test_reshape_rejects_no_shape_arguments() -> None:
    with pytest.raises(TypeError, match="at least one shape argument"):
        Array([1]).reshape()


def test_reshape_empty_tuple_matches_numpy_scalar_shape() -> None:
    actual = Array([7]).reshape(())
    expected = np.array([7]).reshape(())

    assert actual.shape == expected.shape == ()
    assert actual.tolist() == expected.tolist() == 7


def test_fill_is_in_place_returns_none_and_sets_every_element() -> None:
    actual = Array([[1, 2], [3, 4]], dtype=dtypes.int64)

    result = actual.fill(7)

    assert result is None
    assert actual.tolist() == [[7, 7], [7, 7]]


@pytest.mark.parametrize(("dtype_name", "dtype"), INTEGER_DTYPES)
def test_fill_rejects_float_for_every_integer_dtype_without_mutation(dtype_name: str, dtype: Any) -> None:
    actual = Array([1, 2, 3], dtype=dtype)

    with pytest.raises(TypeError) as error:
        actual.fill(3.75)

    assert "value 3.75" in str(error.value)
    assert f"{dtype_name} array requires an integer" in str(error.value)
    assert "cast explicitly" in str(error.value)
    assert actual.tolist() == [1, 2, 3]


@pytest.mark.parametrize(
    "constructor",
    [Array, pecos_rslib.array, num.array, num.asarray, pc.array, pc.asarray],
)
def test_float_to_integer_constructor_matches_fill_error(
    constructor: Callable[..., Array],
) -> None:
    actual = Array([1], dtype=dtypes.uint8)
    with pytest.raises(TypeError) as fill_error:
        actual.fill(1.5)

    with pytest.raises(TypeError) as constructor_error:
        constructor([1.5], dtype=dtypes.uint8)

    assert str(constructor_error.value) == str(fill_error.value)
    assert actual.tolist() == [1]


@pytest.mark.parametrize(("dtype", "value"), [(dtypes.uint8, 256), (dtypes.uint64, -1)])
def test_fill_rejects_unsigned_out_of_range_without_mutation(dtype: Any, value: int) -> None:
    actual = Array([1, 2, 3], dtype=dtype)

    with pytest.raises(OverflowError, match="out of range"):
        actual.fill(value)

    assert actual.tolist() == [1, 2, 3]


def test_fill_rejects_wrong_type_without_mutation() -> None:
    actual = Array([1, 2, 3], dtype=dtypes.int64)

    with pytest.raises(TypeError):
        actual.fill("not an integer")

    assert actual.tolist() == [1, 2, 3]


def test_bool_fill_rejects_string_truthiness() -> None:
    """PECOS rejects truthiness coercion because NumPy invents a boolean value."""
    expected = np.array([False], dtype=np.bool_)
    expected.fill("x")
    assert expected.tolist() == [True]

    actual = Array([False], dtype=dtypes.bool)
    with pytest.raises(TypeError):
        actual.fill("x")
    assert actual.tolist() == [False]


def test_fill_rejects_non_scalar_without_mutation() -> None:
    actual = Array([1, 2, 3], dtype=dtypes.int64)

    with pytest.raises(TypeError, match="fill value must be a scalar"):
        actual.fill([4, 5])

    assert actual.tolist() == [1, 2, 3]


def test_fill_integer_into_float_array_matches_measured_numpy_behavior() -> None:
    actual = Array([1.5, 2.5], dtype=dtypes.float32)
    expected = np.array([1.5, 2.5], dtype=np.float32)

    actual_result = actual.fill(3)
    expected_result = expected.fill(3)

    assert actual_result is expected_result is None
    assert actual.tolist() == expected.tolist() == [3.0, 3.0]


def test_uint64_fill_overflow_names_rejected_value() -> None:
    value = 2**64
    actual = Array([1], dtype=dtypes.uint64)

    with pytest.raises(OverflowError, match=str(value)):
        actual.fill(value)

    assert actual.tolist() == [1]


def test_ravel_docstring_states_copy_divergence_from_numpy() -> None:
    docstring = Array.ravel.__doc__
    assert docstring is not None
    assert "differs from NumPy" in docstring
    assert "always returns an independent" in docstring


def test_fill_and_reshape_docstrings_state_deliberate_divergences() -> None:
    fill_docstring = Array.fill.__doc__
    reshape_docstring = Array.reshape.__doc__
    assert fill_docstring is not None
    assert reshape_docstring is not None
    assert "do not fill by truthiness" in fill_docstring
    assert "do not parse strings" in fill_docstring
    assert "only the literal -1" in reshape_docstring


@pytest.mark.parametrize("shape", [(2, 12), (3, 2, 4), (24,)])
def test_reshape_round_trips_and_flatten_restores_original_shape(
    shape: tuple[int, ...],
) -> None:
    source = np.arange(24, dtype=np.int64).reshape(2, 3, 4)
    actual = Array(source)

    assert actual.reshape(shape).reshape(actual.shape).tolist() == actual.tolist()
    assert actual.flatten().reshape(actual.shape).tolist() == actual.tolist()


def test_pauli_shape_methods_and_fill_are_supported() -> None:
    actual = Array([[Pauli.I, Pauli.X], [Pauli.Y, Pauli.Z]])

    assert actual.flatten().tolist() == [Pauli.I, Pauli.X, Pauli.Y, Pauli.Z]
    assert actual.ravel().tolist() == [Pauli.I, Pauli.X, Pauli.Y, Pauli.Z]
    assert actual.reshape(4, 1).tolist() == [[Pauli.I], [Pauli.X], [Pauli.Y], [Pauli.Z]]
    assert actual.fill(Pauli.Z) is None
    assert actual.tolist() == [[Pauli.Z, Pauli.Z], [Pauli.Z, Pauli.Z]]


def test_pauli_string_shape_methods_and_fill_are_supported() -> None:
    x = PauliString.from_dense_str("X")
    y = PauliString.from_dense_str("Y")
    z = PauliString.from_dense_str("Z")
    actual = Array([[x, y], [z, x]])

    assert actual.flatten().tolist() == [x, y, z, x]
    assert actual.ravel().tolist() == [x, y, z, x]
    assert actual.reshape(4, 1).tolist() == [[x], [y], [z], [x]]
    assert actual.fill(z) is None
    assert actual.tolist() == [[z, z], [z, z]]


@pytest.mark.parametrize(("_canonical", "_short", "pecos_dtype", "numpy_dtype"), UNSIGNED_DTYPES)
def test_issubdtype_accepts_classes_instances_and_pecos_dtypes(
    _canonical: str,
    _short: str,
    pecos_dtype: Any,
    numpy_dtype: Any,
) -> None:
    for candidate in (numpy_dtype, np.dtype(numpy_dtype), pecos_dtype):
        assert num.issubdtype(candidate, num.integer) == np.issubdtype(numpy_dtype, np.integer)
        assert num.issubdtype(candidate, pecos_dtype) == np.issubdtype(numpy_dtype, numpy_dtype)


@pytest.mark.parametrize(("dtype_name", "numpy_dtype", "value"), SCALAR_DTYPES)
@pytest.mark.parametrize(("reduction_name", "reduction", "numpy_reduction"), AXIS_REDUCTIONS)
@pytest.mark.parametrize("axis", [0, -1])
def test_zero_dimensional_reductions_match_numpy(
    dtype_name: str,
    numpy_dtype: Any,
    value: Any,
    reduction_name: str,
    reduction: Callable[..., Any],
    numpy_reduction: Callable[..., Any],
    axis: int,
) -> None:
    numpy_array = np.array(value, dtype=numpy_dtype)
    pecos_array = num.array(numpy_array)
    expected = numpy_reduction(numpy_array, axis=axis)

    actual = reduction(pecos_array, axis=axis)
    assert not isinstance(actual, Array)
    assert actual == expected.item(), dtype_name

    if reduction_name in {"any", "all"}:
        method_actual = getattr(pecos_array, reduction_name)(axis=axis)
        assert not isinstance(method_actual, Array)
        assert method_actual == expected.item(), dtype_name


@pytest.mark.parametrize(("_dtype_name", "numpy_dtype", "value"), SCALAR_DTYPES)
@pytest.mark.parametrize(("reduction_name", "reduction", "numpy_reduction"), AXIS_REDUCTIONS)
def test_zero_dimensional_reductions_reject_out_of_range_axis(
    _dtype_name: str,
    numpy_dtype: Any,
    value: Any,
    reduction_name: str,
    reduction: Callable[..., Any],
    numpy_reduction: Callable[..., Any],
) -> None:
    numpy_array = np.array(value, dtype=numpy_dtype)
    pecos_array = num.array(numpy_array)
    error_message = "axis 1 is out of bounds for array of dimension 0"

    with pytest.raises(ValueError, match=f"^{error_message}$") as numpy_error:
        numpy_reduction(numpy_array, axis=1)
    with pytest.raises(ValueError, match=f"^{error_message}$") as pecos_error:
        reduction(pecos_array, axis=1)
    assert str(pecos_error.value) == str(numpy_error.value)

    if reduction_name in {"any", "all"}:
        with pytest.raises(ValueError, match=f"^{error_message}$") as method_error:
            getattr(pecos_array, reduction_name)(axis=1)
        assert str(method_error.value) == str(numpy_error.value)


@pytest.mark.parametrize("dtype_name", ["int8", "int64", "float32", "complex128", "uint8", "bool_"])
def test_zero_dimensional_default_axis_reductions_match_numpy(dtype_name: str) -> None:
    """axis=None on a 0-d array must reduce to the sole element, as NumPy does.

    The explicit-axis fix routed only axis=0/-1 through the scalar path; the
    default axis=None stayed on the legacy kind-dispatch, where a 0-d buffer
    extracts as empty -- sum silently returned 0 and max/min raised.
    """
    value = 1.25 + 2.5j if dtype_name == "complex128" else 1
    np_arr = np.array(value, dtype=getattr(np, dtype_name))
    pecos_arr = Array(np_arr)

    assert num.sum(pecos_arr) == np.sum(np_arr)
    assert num.any(pecos_arr) == bool(np_arr.any())
    assert num.all(pecos_arr) == bool(np_arr.all())
    if dtype_name != "complex128":
        assert num.max(pecos_arr) == np.max(np_arr).item()
        assert num.min(pecos_arr) == np.min(np_arr).item()
    else:
        assert num.max(pecos_arr) == complex(np.max(np_arr))
        assert num.min(pecos_arr) == complex(np.min(np_arr))


@pytest.mark.parametrize(
    ("dtype_name", "pecos_dtype", "numpy_dtype", "scalar", "sequence"),
    INDEX_ASSIGNMENT_DTYPES,
)
def test_indexed_assignment_1d_matches_numpy(
    dtype_name: str,
    pecos_dtype: Any,
    numpy_dtype: Any,
    scalar: Any,
    sequence: list[Any],
) -> None:
    expected = np.zeros(5, dtype=numpy_dtype)
    actual = Array([0] * 5, dtype=pecos_dtype)

    expected[[0, -1]] = scalar
    actual[[0, -1]] = scalar
    np.testing.assert_array_equal(np.asarray(actual), expected, err_msg=dtype_name)

    duplicate_indices = [1, 3, 1]
    expected[duplicate_indices] = sequence
    actual[duplicate_indices] = sequence
    np.testing.assert_array_equal(np.asarray(actual), expected, err_msg=dtype_name)
    assert actual[duplicate_indices].tolist() == expected[duplicate_indices].tolist()

    expected[[]] = scalar
    actual[[]] = scalar
    expected[[]] = []
    actual[[]] = []
    np.testing.assert_array_equal(np.asarray(actual), expected, err_msg=dtype_name)


@pytest.mark.parametrize(
    ("dtype_name", "pecos_dtype", "numpy_dtype", "scalar", "sequence"),
    INDEX_ASSIGNMENT_DTYPES,
)
def test_indexed_assignment_2d_matches_numpy(
    dtype_name: str,
    pecos_dtype: Any,
    numpy_dtype: Any,
    scalar: Any,
    sequence: list[Any],
) -> None:
    expected = np.zeros((2, 4), dtype=numpy_dtype)
    actual = Array([[0] * 4 for _ in range(2)], dtype=pecos_dtype)

    expected[0, [0, -1]] = scalar
    actual[0, [0, -1]] = scalar
    expected[1, [0, 2, 0]] = sequence
    actual[1, [0, 2, 0]] = sequence

    np.testing.assert_array_equal(np.asarray(actual), expected, err_msg=dtype_name)
    assert actual[1, [0, 2]].tolist() == expected[1, [0, 2]].tolist()


def test_indexed_assignment_rejects_out_of_bounds_with_index_and_length() -> None:
    actual = Array([0, 0, 0], dtype=dtypes.int64)

    with pytest.raises(IndexError, match=r"index 3 .* length 3"):
        actual[[0, 3]] = 1
    actual_2d = Array([[0, 0, 0], [0, 0, 0]], dtype=dtypes.int64)
    with pytest.raises(IndexError, match=r"index -4 .* length 3"):
        actual_2d[1, [-4]] = 1


def test_indexed_assignment_rejects_mismatched_rhs_length() -> None:
    actual = Array([0, 0, 0], dtype=dtypes.int64)

    with pytest.raises(ValueError, match=r"selection has shape \[2\].*source has shape \[1\]"):
        actual[[0, 2]] = [7]


def test_indexed_assignment_uses_checked_value_conversion() -> None:
    actual = Array([0, 0], dtype=dtypes.uint8)

    with pytest.raises(OverflowError, match=r"value 256 is out of range for uint8"):
        actual[[0, 1]] = 256
    with pytest.raises(OverflowError, match=r"value 256 is out of range for uint8"):
        actual[[0, 1]] = [1, 256]


@pytest.mark.parametrize(
    "mask",
    [
        [True, True, True, True],
        [False, False, False, False],
        [False, True, False, True],
    ],
)
@pytest.mark.parametrize("as_array", [False, True])
def test_boolean_mask_reads_match_numpy(mask: list[bool], as_array: bool) -> None:
    values = np.array([10, 20, 30, 40], dtype=np.int64)
    actual = Array(values)
    actual_mask: Any = Array(mask, dtype=dtypes.bool) if as_array else mask

    np.testing.assert_array_equal(np.asarray(actual[actual_mask]), values[np.array(mask)])


@pytest.mark.parametrize("as_array", [False, True])
def test_boolean_mask_length_mismatch_raises(as_array: bool) -> None:
    actual = Array([10, 20, 30], dtype=dtypes.int64)
    mask: Any = Array([True, False], dtype=dtypes.bool) if as_array else [True, False]

    with pytest.raises(IndexError, match=r"mask length 2 .* axis 0 length 3"):
        _ = actual[mask]


def test_one_dimensional_boolean_mask_selects_rows_like_numpy() -> None:
    values = np.arange(12, dtype=np.int64).reshape(3, 4)
    mask = [True, False, True]

    actual = Array(values)[Array(mask, dtype=dtypes.bool)]
    np.testing.assert_array_equal(np.asarray(actual), values[np.array(mask)])


def test_nonstandard_layout_boolean_mask_read_and_write_match_numpy() -> None:
    expected = np.arange(12, dtype=np.int64).reshape(3, 4)
    actual = Array(expected)
    mask_source = np.array([[True, False, True], [False, True, False], [True, False, False], [False, True, True]])
    expected_mask = mask_source.T
    actual_mask = Array(mask_source).T

    np.testing.assert_array_equal(np.asarray(actual[actual_mask]), expected[expected_mask])

    replacement = np.arange(expected[expected_mask].size, dtype=np.int64)
    expected[expected_mask] = replacement
    actual[actual_mask] = replacement
    np.testing.assert_array_equal(np.asarray(actual), expected)


@pytest.mark.parametrize("mask_ndim", [2, 3])
@pytest.mark.parametrize("as_array", [False, True])
def test_multidimensional_boolean_mask_read_and_write_match_numpy(mask_ndim: int, as_array: bool) -> None:
    shape = (2, 3, 4)
    expected = np.arange(np.prod(shape), dtype=np.int64).reshape(shape)
    actual = Array(expected)
    mask_shape = shape[:mask_ndim]
    mask = np.arange(np.prod(mask_shape)).reshape(mask_shape) % 2 == 0
    actual_mask: Any = Array(mask) if as_array else mask.tolist()

    expected_selection = expected[mask]
    actual_selection = np.asarray(actual[actual_mask])
    assert actual_selection.shape == expected_selection.shape
    np.testing.assert_array_equal(actual_selection, expected_selection)

    replacement = np.arange(expected_selection.size, dtype=np.int64).reshape(expected_selection.shape)
    expected[mask] = replacement
    actual[actual_mask] = replacement
    np.testing.assert_array_equal(np.asarray(actual), expected)


def test_boolean_mask_assignment_falls_out_of_indexed_assignment() -> None:
    expected = np.zeros(4, dtype=np.uint8)
    actual = Array([0, 0, 0, 0], dtype=dtypes.uint8)
    mask = [False, True, False, True]

    expected[np.array(mask)] = [3, 5]
    actual[Array(mask, dtype=dtypes.bool)] = [3, 5]
    np.testing.assert_array_equal(np.asarray(actual), expected)


def test_detection_events_detector_sequence_assignment() -> None:
    detection_events = num.zeros((3, 6), dtype=dtypes.uint8)
    index = 1
    detectors = [0, 2, 5]

    detection_events[index, detectors] = 1

    assert detection_events.tolist() == [
        [0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 1],
        [0, 0, 0, 0, 0, 0],
    ]


@pytest.mark.parametrize("index_form", ["sequence", "tuple", "bool_sequence", "bool_array"])
def test_indexed_assignment_round_trip(index_form: str) -> None:
    if index_form == "tuple":
        actual = Array([[0, 0, 0], [0, 0, 0]], dtype=dtypes.int64)
        actual[1, [0, 2]] = [7, 9]
        assert actual[1, [0, 2]].tolist() == [7, 9]
        return

    actual = Array([0, 0, 0], dtype=dtypes.int64)
    index: Any
    if index_form == "sequence":
        index = [0, 2]
    elif index_form == "bool_sequence":
        index = [True, False, True]
    else:
        index = Array([True, False, True], dtype=dtypes.bool)
    actual[index] = [7, 9]
    assert actual[index].tolist() == [7, 9]


def test_separated_advanced_indices_read_and_write_match_numpy() -> None:
    values = np.arange(18, dtype=np.int64).reshape(2, 3, 3)
    actual = Array(values)
    index = (1, slice(0, 2), [0, 2])

    # NumPy moves the advanced-index dimension to the front for this layout.
    assert values[index].shape == (2, 2)
    assert actual[index].shape == values[index].shape
    np.testing.assert_array_equal(np.asarray(actual[index]), values[index])
    replacement = np.arange(4, dtype=np.int64).reshape(2, 2) + 100
    values[index] = replacement
    actual[index] = replacement
    np.testing.assert_array_equal(np.asarray(actual), values)

    unequal_length_index = (1, slice(0, 3), [0, 2])
    assert values[unequal_length_index].shape == (2, 3)
    assert actual[unequal_length_index].shape == (2, 3)
    np.testing.assert_array_equal(np.asarray(actual[unequal_length_index]), values[unequal_length_index])


@pytest.mark.parametrize(
    "index",
    [
        (1, slice(0, 2), [0, 2], slice(1, 5, 2)),
        ([0, 1], slice(0, 3), [1, 3], slice(None)),
    ],
)
def test_separated_advanced_indices_in_four_dimensions_match_numpy(
    index: tuple[Any, ...],
) -> None:
    shape = (2, 3, 4, 5)
    expected = np.arange(np.prod(shape), dtype=np.int64).reshape(shape)
    actual = Array(expected)
    expected_selection = expected[index]

    assert actual[index].shape == expected_selection.shape
    np.testing.assert_array_equal(np.asarray(actual[index]), expected_selection)
    replacement = np.arange(expected_selection.size, dtype=np.int64).reshape(expected_selection.shape)
    expected[index] = replacement
    actual[index] = replacement
    np.testing.assert_array_equal(np.asarray(actual), expected)


@pytest.mark.parametrize(
    ("index", "expected_shape"),
    [
        ((slice(None), [0, 2, 1], [1, 3, 0], slice(None)), (2, 3, 5)),
        ((slice(None), [0, 2, 1], slice(None), [1, 3, 4]), (3, 2, 4)),
    ],
    ids=["adjacent", "separated"],
)
def test_advanced_index_axis_placement_matches_numpy_shape(
    index: tuple[Any, ...], expected_shape: tuple[int, ...]
) -> None:
    values = np.arange(2 * 3 * 4 * 5, dtype=np.int64).reshape(2, 3, 4, 5)
    actual = Array(values)

    assert values[index].shape == expected_shape
    assert actual[index].shape == expected_shape
    np.testing.assert_array_equal(np.asarray(actual[index]), values[index])


@pytest.mark.parametrize(
    "index",
    [
        ([0, 2], [1, 3]),
        ([[0], [2]], [[1, 3, 0]]),
    ],
    ids=["equal-shape", "broadcastable-shapes"],
)
def test_multiple_advanced_indices_read_and_write_match_numpy(
    index: tuple[Any, ...],
) -> None:
    shape = (3, 4, 5)
    expected = np.arange(np.prod(shape), dtype=np.int64).reshape(shape)
    actual = Array(expected)
    expected_selection = expected[index]

    assert actual[index].shape == expected_selection.shape
    np.testing.assert_array_equal(np.asarray(actual[index]), expected_selection)
    replacement = np.arange(expected_selection.size, dtype=np.int64).reshape(expected_selection.shape)
    expected[index] = replacement
    actual[index] = replacement
    np.testing.assert_array_equal(np.asarray(actual), expected)


@pytest.mark.parametrize("as_native", [False, True])
def test_multidimensional_integer_array_index_matches_numpy(as_native: bool) -> None:
    expected = np.arange(3 * 4, dtype=np.int64).reshape(3, 4)
    actual = Array(expected)
    index = np.array([[0], [2]], dtype=np.int32)
    actual_index: Any = Array(index) if as_native else index

    assert actual[actual_index].shape == expected[index].shape
    np.testing.assert_array_equal(np.asarray(actual[actual_index]), expected[index])
    replacement = np.full(expected[index].shape, -1, dtype=np.int64)
    expected[index] = replacement
    actual[actual_index] = replacement
    np.testing.assert_array_equal(np.asarray(actual), expected)


@pytest.mark.parametrize(
    ("shape", "index"),
    [
        ((5,), ([0, -1, 2],)),
        ((3, 4), (1, [0, -1, 2])),
        ((2, 3, 4), (slice(None), 1, [0, -1, 2])),
        ((2, 3, 4, 5), ([0, 1], slice(None), 2, [0, -1])),
    ],
)
def test_scalar_and_sequence_assignment_through_four_dimensions(shape: tuple[int, ...], index: tuple[Any, ...]) -> None:
    expected = np.zeros(shape, dtype=np.int64)
    actual = Array(expected)

    expected[index] = 7
    actual[index] = 7
    np.testing.assert_array_equal(np.asarray(actual), expected)

    replacement = np.arange(expected[index].size, dtype=np.int64).reshape(expected[index].shape)
    expected[index] = replacement
    actual[index] = replacement
    np.testing.assert_array_equal(np.asarray(actual), expected)


@pytest.mark.parametrize("ndim", [3, 4])
def test_boolean_row_mask_assignment_through_four_dimensions(ndim: int) -> None:
    shape = (4, *([3] * (ndim - 1)))
    expected = np.arange(np.prod(shape), dtype=np.int64).reshape(shape)
    actual = Array(expected)
    mask = np.array([False, True, False, True])
    replacement = np.full(expected[mask].shape, -1, dtype=np.int64)

    expected[mask] = replacement
    actual[Array(mask)] = replacement
    np.testing.assert_array_equal(np.asarray(actual), expected)


@pytest.mark.parametrize(
    ("shape", "index"),
    [
        ((0,), ([],)),
        ((0, 3), ([], slice(None))),
        ((3, 0), ([0, 2], slice(None))),
    ],
)
def test_empty_and_zero_length_advanced_selections_match_numpy(shape: tuple[int, ...], index: tuple[Any, ...]) -> None:
    expected = np.zeros(shape, dtype=np.int64)
    actual = Array(expected)

    assert actual[index].shape == expected[index].shape
    np.testing.assert_array_equal(np.asarray(actual[index]), expected[index])
    actual[index] = np.empty(expected[index].shape, dtype=np.int64)
    np.testing.assert_array_equal(np.asarray(actual), expected)


def test_all_false_multidimensional_mask_matches_numpy() -> None:
    expected = np.arange(12, dtype=np.int64).reshape(3, 4)
    actual = Array(expected)
    mask = np.zeros((3, 4), dtype=np.bool_)

    assert actual[Array(mask)].shape == expected[mask].shape == (0,)
    actual[Array(mask)] = np.empty((0,), dtype=np.int64)
    np.testing.assert_array_equal(np.asarray(actual), expected)


@pytest.mark.parametrize(
    ("target", "source"),
    [
        ([3, 2, 1, 0], [0, 1, 2, 3]),
        ([0, 1, 2, 3], [3, 2, 1, 0]),
    ],
)
def test_selection_based_aliasing_and_overlap_match_numpy(target: list[int], source: list[int]) -> None:
    expected = np.arange(4, dtype=np.int64)
    actual = Array(expected)

    expected[target] = expected[source]
    actual[target] = actual[source]
    np.testing.assert_array_equal(np.asarray(actual), expected)


@pytest.mark.parametrize(
    "values",
    [
        np.arange(24, dtype=np.int64).reshape(4, 6).T,
        np.arange(48, dtype=np.int64).reshape(6, 8)[::2, 1::2],
    ],
    ids=["transpose", "stepped-slice"],
)
def test_advanced_assignment_on_noncontiguous_input_matches_numpy(
    values: np.ndarray[Any, Any],
) -> None:
    expected = values.copy(order="K")
    actual = Array(values)
    index = ([0, expected.shape[0] - 1], [0, expected.shape[1] - 1])

    expected[index] = [100, 200]
    actual[index] = [100, 200]
    np.testing.assert_array_equal(np.asarray(actual), expected)


def test_advanced_indices_reject_nonbroadcastable_shapes() -> None:
    actual = Array(np.arange(3 * 4, dtype=np.int64).reshape(3, 4))

    with pytest.raises(IndexError, match=r"could not be broadcast.*\[2\].*\[3\]"):
        _ = actual[[0, 1], [0, 1, 2]]
    with pytest.raises(IndexError, match=r"could not be broadcast.*\[2\].*\[3\]"):
        actual[[0, 1], [0, 1, 2]] = 0


def test_advanced_indices_check_bounds_before_execution() -> None:
    actual = Array(np.arange(3 * 4, dtype=np.int64).reshape(3, 4))

    with pytest.raises(IndexError, match=r"index 4 .* axis 1 .* length 4"):
        _ = actual[[0, 1], [0, 4]]
    with pytest.raises(IndexError, match=r"index -5 .* axis 1 .* length 4"):
        actual[[0, 1], [0, -5]] = 0


@pytest.mark.parametrize("index", [(Ellipsis, 0), (None, 0)])
def test_unverified_index_expansions_remain_explicitly_unsupported(
    index: tuple[Any, ...],
) -> None:
    actual = Array(np.arange(6, dtype=np.int64).reshape(2, 3))

    with pytest.raises(NotImplementedError, match=r"Ellipsis|newaxis"):
        _ = actual[index]
    with pytest.raises(NotImplementedError, match=r"Ellipsis|newaxis"):
        actual[index] = 0


def test_contiguous_advanced_indices_continue_to_match_numpy() -> None:
    expected = np.arange(18, dtype=np.int64).reshape(2, 3, 3)
    actual = Array(expected)
    index = (slice(0, 2), 1, [0, 2])

    np.testing.assert_array_equal(np.asarray(actual[index]), expected[index])
    expected[index] = [[100, 101], [102, 103]]
    actual[index] = [[100, 101], [102, 103]]
    np.testing.assert_array_equal(np.asarray(actual), expected)


@pytest.mark.parametrize(
    ("index", "expected"),
    [
        ([1, 0, 3, 2], [1, 0, 3, 2]),
        (slice(None, None, -1), [3, 2, 1, 0]),
    ],
)
def test_direct_self_assignment_snapshots_rhs(index: Any, expected: list[int]) -> None:
    actual = Array([0, 1, 2, 3], dtype=dtypes.int64)

    actual[index] = actual

    assert actual.tolist() == expected


def test_pauli_indexed_assignment_accepts_array_like_rhs() -> None:
    actual = Array([Pauli.I, Pauli.I, Pauli.I])
    native_rhs = Array([Pauli.X, Pauli.Z])
    numpy_rhs = np.array([Pauli.Y, Pauli.X], dtype=object)

    actual[[0, 2]] = native_rhs
    assert actual.tolist() == [Pauli.X, Pauli.I, Pauli.Z]
    actual[[1, 2]] = numpy_rhs
    assert actual.tolist() == [Pauli.X, Pauli.Y, Pauli.X]


def test_pauli_string_indexed_assignment_accepts_array_like_rhs() -> None:
    identity = PauliString.from_dense_str("I")
    x = PauliString.from_dense_str("X")
    y = PauliString.from_dense_str("Y")
    z = PauliString.from_dense_str("Z")
    actual = Array([identity, identity, identity])
    native_rhs = Array([x, z])
    numpy_rhs = np.array([y, x], dtype=object)

    actual[[0, 2]] = native_rhs
    assert actual.tolist() == [x, identity, z]
    actual[[1, 2]] = numpy_rhs
    assert actual.tolist() == [x, y, x]


@pytest.mark.parametrize(
    ("alias", "expected"),
    [
        (float, "float64"),
        ("float", "float64"),
        ("double", "float64"),
        (int, "int64"),
        ("int", "int64"),
        (bool, "bool"),
        (complex, "complex128"),
        ("complex", "complex128"),
    ],
)
def test_dtype_aliases_resolve_to_the_same_width_as_python(alias: object, expected: str) -> None:
    """Builtin and string dtype aliases must match Python's own widths.

    `from_str("float")` returned F32 while the `dtypes.float` member was F64, so
    `asarray(x, dtype=float)` silently halved precision -- the module disagreed
    with itself depending on which spelling reached it (#483).
    """
    assert str(pc.asarray([1], dtype=alias).dtype) == expected


@pytest.mark.parametrize("alias", [float, "float", "double"])
def test_float_aliases_preserve_double_precision(alias: object) -> None:
    """A value needing more than 24 bits of mantissa must survive."""
    value = 1.0 + 2**-30
    assert pc.asarray([value], dtype=alias).tolist() == [value]
    assert np.asarray([value], dtype=float).tolist() == [value]


def test_dtype_member_and_builtin_agree_for_every_alias() -> None:
    """The member spelling and the builtin must never diverge again."""
    for name, builtin in [
        ("float", float),
        ("int", int),
        ("bool", bool),
        ("complex", complex),
    ]:
        member = getattr(dtypes, name)
        assert str(pc.asarray([1], dtype=member).dtype) == str(pc.asarray([1], dtype=builtin).dtype)
