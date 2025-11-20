# Copyright 2025 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Type aliases for PECOS types.

This module provides type aliases for use in type hints, analogous to NumPy's typing module.
These types are purely for static type checking and have no runtime overhead.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Import directly from _pecos_rslib to avoid circular import
    from pecos_rslib._pecos_rslib import Array as _Array
    from pecos_rslib._pecos_rslib import dtypes

    # Access scalar classes through dtypes dtype instances using .type attribute
    ScalarI8 = dtypes.i8.type
    ScalarI16 = dtypes.i16.type
    ScalarI32 = dtypes.i32.type
    ScalarI64 = dtypes.i64.type
    ScalarU8 = dtypes.u8.type
    ScalarU16 = dtypes.u16.type
    ScalarU32 = dtypes.u32.type
    ScalarU64 = dtypes.u64.type
    ScalarF32 = dtypes.f32.type
    ScalarF64 = dtypes.f64.type
    ScalarComplex64 = dtypes.complex64.type
    ScalarComplex128 = dtypes.complex128.type

    # Integer type aliases for type hints (analogous to numpy.integer)
    Integer = (
        type[ScalarI8]
        | type[ScalarI16]
        | type[ScalarI32]
        | type[ScalarI64]
        | type[ScalarU8]
        | type[ScalarU16]
        | type[ScalarU32]
        | type[ScalarU64]
    )
    SignedInteger = type[ScalarI8] | type[ScalarI16] | type[ScalarI32] | type[ScalarI64]
    UnsignedInteger = (
        type[ScalarU8] | type[ScalarU16] | type[ScalarU32] | type[ScalarU64]
    )

    # Float type aliases - now includes both f32 and f64
    Float = type[ScalarF32] | type[ScalarF64]

    # Complex type aliases - now includes both complex64 and complex128
    Complex = type[ScalarComplex64] | type[ScalarComplex128]

    # Numeric type (any integer or float)
    Numeric = Integer | Float

    # Inexact type (any float or complex)
    Inexact = Float | Complex
else:
    # At runtime, these are just placeholders - they're never actually used
    # because they're only for type checking
    Integer = "Integer"
    SignedInteger = "SignedInteger"
    UnsignedInteger = "UnsignedInteger"
    Float = "Float"
    Complex = "Complex"
    Numeric = "Numeric"
    Inexact = "Inexact"

    # Import the real Array for runtime use
    from pecos_rslib._pecos_rslib import Array as _Array


# Generic Array type for type hints (analogous to numpy.typing.NDArray)
class ArrayType(type):
    """Metaclass to add __class_getitem__ support for Array type hints.

    This allows syntax like:
        Array[f64]  # Array with float64 dtype
        Array[i32]  # Array with int32 dtype

    This is purely for type checking and has no runtime effect.
    """

    def __getitem__(cls, dtype_hint: Any) -> type[_Array]:
        """Support Array[dtype] syntax for type hints.

        Args:
            dtype_hint: The dtype type hint (e.g., f64, i32)

        Returns:
            The Array type (dtype information is for type checkers only)
        """
        # Return the actual Array class - the dtype parameter is only for type checkers
        return _Array


# We can't subclass the Rust-backed Array class directly (PyO3 limitation),
# but we can add __class_getitem__ to the class itself to enable Array[dtype] syntax


def _array_class_getitem(dtype_hint: Any) -> type[_Array]:
    """Support Array[dtype] syntax for type hints.

    Args:
        dtype_hint: The dtype type hint (e.g., f64, i32)

    Returns:
        The Array type (dtype information is for type checkers only)
    """
    # Return the actual Array class - the dtype parameter is only for type checkers
    return _Array


# Add __class_getitem__ directly to the Array type
# This is a special method that doesn't need self/cls when called as Class[arg]
type.__setattr__(_Array, "__class_getitem__", _array_class_getitem)

# Re-export as Array
Array = _Array


__all__ = [
    "Integer",
    "SignedInteger",
    "UnsignedInteger",
    "Float",
    "Complex",
    "Numeric",
    "Inexact",
    "Array",
]
