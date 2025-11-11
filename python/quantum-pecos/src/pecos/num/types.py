# Copyright 2025 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generic type support for Array and numerical operations.

This module provides type hints and generic types for PECOS numerical arrays,
enabling better IDE support and static type checking. Follows NumPy's typing
conventions using Array[dtype] pattern (similar to numpy.typing.NDArray).

Examples:
    NumPy-style type annotations (recommended)::

        from pecos.num.types import Array
        from pecos_rslib import dtypes

        def normalize(vec: Array[dtypes.complex128]) -> Array[dtypes.complex128]:
            norm = sqrt(sum(abs(vec)**2))
            return vec / norm

        # Using dtype aliases for convenience
        def process(data: Array[dtypes.float]) -> Array[dtypes.complex]:
            return data * (1+2j)

    Alternative: Using Array directly::

        from pecos.num.types import Array
        from pecos_rslib import dtypes

        def process(data: Array[dtypes.f64]) -> Array[dtypes.f64]:
            return data * 2.0

    Type-safe array construction::

        from pecos.num import array, zeros
        from pecos.num.types import Array
        from pecos_rslib import dtypes

        # These type annotations help IDEs provide better autocomplete
        float_data: Array[dtypes.f64] = array([1.0, 2.0, 3.0])
        complex_state: Array[dtypes.complex128] = zeros(8, dtype=dtypes.complex128)
"""

from __future__ import annotations

from typing import Generic, TypeVar

# Type variable for dtype
DType = TypeVar("DType")


# Generic Array type that can be parameterized by dtype
# This allows type annotations like: Array[dtypes.complex128]
class Array(Generic[DType]):
    """Generic type for Array with dtype parameter support.

    This is a typing stub that enables generic type annotations for Array.
    At runtime, use the actual Array from pecos_rslib.

    Type Parameters:
        DType: The dtype of the array (from pecos_rslib.dtypes)

    Examples:
        >>> from pecos.num.types import Array
        >>> from pecos_rslib import dtypes
        >>>
        >>> def get_state_vector() -> Array[dtypes.complex128]:
        ...     return array([1 + 0j, 0 + 0j], dtype=dtypes.complex128)
        ...
        >>>
        >>> def multiply_floats(
        ...     a: Array[dtypes.f64], b: Array[dtypes.f64]
        ... ) -> Array[dtypes.f64]:
        ...     return a * b

    Note:
        This is a type hint only. At runtime, import Array from pecos_rslib:
        >>> from pecos_rslib import Array  # Runtime usage
        >>> from pecos.num.types import Array  # Type hints only
    """

    # Typing stubs - these methods exist on the real Array
    @property
    def dtype(self) -> DType:
        """The dtype of the array elements."""

    @property
    def shape(self) -> tuple[int, ...]:
        """The shape of the array."""

    @property
    def ndim(self) -> int:
        """The number of dimensions."""

    @property
    def size(self) -> int:
        """The total number of elements."""

    def __len__(self) -> int:
        """The length of the first dimension."""

    def __getitem__(self, key: int | tuple | slice) -> Array:  # type: ignore[misc]
        """Get array element(s) by index or slice."""

    def __setitem__(self, key: int | tuple | slice, value: Array | complex) -> None:
        """Set array element(s) by index or slice."""


__all__ = [
    # Main type (recommended - follows NumPy pattern)
    "Array",
    # Type variable
    "DType",
]
