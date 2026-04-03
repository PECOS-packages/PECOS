# Copyright 2025 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Type stubs for pecos_rslib.

This module provides type information for the pecos_rslib Rust extension.
"""

from __future__ import annotations

import os
from typing import (
    Callable,
    Generic,
    Iterator,
    Sequence,
    TypeVar,
    overload,
)

# =============================================================================
# Type Variables
# =============================================================================
_T = TypeVar("_T")
_DType = TypeVar("_DType", bound="DType")
_ScalarT = TypeVar("_ScalarT", bound="Scalar")

# =============================================================================
# Scalar Types (NumPy-like)
# =============================================================================
class Scalar:
    """Base class for scalar numeric types."""

    def __init__(self, value: int | float | complex) -> None: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    def __int__(self) -> int: ...
    def __float__(self) -> float: ...
    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __lt__(self, other: Scalar | int | float) -> bool: ...
    def __le__(self, other: Scalar | int | float) -> bool: ...
    def __gt__(self, other: Scalar | int | float) -> bool: ...
    def __ge__(self, other: Scalar | int | float) -> bool: ...
    def __add__(self, other: Scalar | int | float) -> Scalar: ...
    def __radd__(self, other: Scalar | int | float) -> Scalar: ...
    def __sub__(self, other: Scalar | int | float) -> Scalar: ...
    def __rsub__(self, other: Scalar | int | float) -> Scalar: ...
    def __mul__(self, other: Scalar | int | float) -> Scalar: ...
    def __rmul__(self, other: Scalar | int | float) -> Scalar: ...
    def __truediv__(self, other: Scalar | int | float) -> Scalar: ...
    def __rtruediv__(self, other: Scalar | int | float) -> Scalar: ...
    def __floordiv__(self, other: Scalar | int | float) -> Scalar: ...
    def __rfloordiv__(self, other: Scalar | int | float) -> Scalar: ...
    def __mod__(self, other: Scalar | int | float) -> Scalar: ...
    def __rmod__(self, other: Scalar | int | float) -> Scalar: ...
    def __neg__(self) -> Scalar: ...
    def __pos__(self) -> Scalar: ...
    def __abs__(self) -> Scalar: ...
    # Bitwise operations (for integer scalar types)
    def __and__(self, other: Scalar | int) -> Scalar: ...
    def __rand__(self, other: Scalar | int) -> Scalar: ...
    def __or__(self, other: Scalar | int) -> Scalar: ...
    def __ror__(self, other: Scalar | int) -> Scalar: ...
    def __xor__(self, other: Scalar | int) -> Scalar: ...
    def __rxor__(self, other: Scalar | int) -> Scalar: ...
    def __lshift__(self, other: Scalar | int) -> Scalar: ...
    def __rlshift__(self, other: Scalar | int) -> Scalar: ...
    def __rshift__(self, other: Scalar | int) -> Scalar: ...
    def __rrshift__(self, other: Scalar | int) -> Scalar: ...
    def __invert__(self) -> Scalar: ...

class ScalarI8(Scalar):
    """8-bit signed integer scalar."""

    ...

class ScalarI16(Scalar):
    """16-bit signed integer scalar."""

    ...

class ScalarI32(Scalar):
    """32-bit signed integer scalar."""

    ...

class ScalarI64(Scalar):
    """64-bit signed integer scalar."""

    ...

class ScalarU8(Scalar):
    """8-bit unsigned integer scalar."""

    ...

class ScalarU16(Scalar):
    """16-bit unsigned integer scalar."""

    ...

class ScalarU32(Scalar):
    """32-bit unsigned integer scalar."""

    ...

class ScalarU64(Scalar):
    """64-bit unsigned integer scalar."""

    ...

class ScalarF32(Scalar):
    """32-bit floating point scalar."""

    ...

class ScalarF64(Scalar):
    """64-bit floating point scalar."""

    ...

class ScalarComplex64(Scalar):
    """64-bit complex number (32-bit real + 32-bit imag)."""

    @property
    def real(self) -> float: ...
    @property
    def imag(self) -> float: ...
    def __complex__(self) -> complex: ...

class ScalarComplex128(Scalar):
    """128-bit complex number (64-bit real + 64-bit imag)."""

    @property
    def real(self) -> float: ...
    @property
    def imag(self) -> float: ...
    def __complex__(self) -> complex: ...

# Scalar type shortcuts
i8: type[ScalarI8]
i16: type[ScalarI16]
i32: type[ScalarI32]
i64: type[ScalarI64]
u8: type[ScalarU8]
u16: type[ScalarU16]
u32: type[ScalarU32]
u64: type[ScalarU64]
f32: type[ScalarF32]
f64: type[ScalarF64]
complex64: type[ScalarComplex64]
complex128: type[ScalarComplex128]

# Note: Type aliases (Integer, Float, Complex, Numeric, Inexact, etc.) are defined
# in quantum-pecos (pecos.typing module) as they are Python TypeAlias constructs.

# =============================================================================
# BitInt Type
# =============================================================================
class BitInt:
    """Fixed-width signed integer type.

    Always signed. Internally wraps BitUInt(N+1) where the extra bit is the sign bit.
    BitInt(1, 1) returns 1 (not -1). BitInt(1, -1) returns -1.

    Examples:
        >>> b = BitInt(8, 42)   # 8-bit signed integer with value 42
        >>> b = BitInt("1010")  # 4-bit signed from binary string (value 10, sign bit 0)
        >>> b = BitInt(1, -1)   # 1-bit signed with value -1
    """

    @property
    def size(self) -> int:
        """Number of user-visible bits (not including internal sign bit)."""
        ...

    @property
    def signed(self) -> bool:
        """Always True."""
        ...

    @overload
    def __init__(self, binary_str: str, value: None = None) -> None: ...
    @overload
    def __init__(self, size: int, value: int = 0) -> None: ...
    def __init__(self, size: str | int, value: int | None = None) -> None: ...
    @staticmethod
    def from_binary(s: str) -> BitInt: ...
    @staticmethod
    def zeros(size: int) -> BitInt: ...
    @staticmethod
    def ones(size: int) -> BitInt: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    def __int__(self) -> int: ...
    def __index__(self) -> int: ...
    def __len__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __bool__(self) -> bool: ...
    def to_binary_str(self, reverse_bits: bool = False, separator: str | None = None) -> str: ...
    def __getitem__(self, index: int) -> int: ...
    def __setitem__(self, index: int, value: int) -> None: ...
    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __lt__(self, other: BitInt | int | str) -> bool: ...
    def __le__(self, other: BitInt | int | str) -> bool: ...
    def __gt__(self, other: BitInt | int | str) -> bool: ...
    def __ge__(self, other: BitInt | int | str) -> bool: ...
    def __and__(self, other: BitInt | int | str) -> BitInt: ...
    def __rand__(self, other: BitInt | int | str) -> BitInt: ...
    def __or__(self, other: BitInt | int | str) -> BitInt: ...
    def __ror__(self, other: BitInt | int | str) -> BitInt: ...
    def __xor__(self, other: BitInt | int | str) -> BitInt: ...
    def __rxor__(self, other: BitInt | int | str) -> BitInt: ...
    def __invert__(self) -> BitInt: ...
    def __lshift__(self, other: BitInt | int) -> BitInt: ...
    def __rshift__(self, other: BitInt | int) -> BitInt: ...
    def __add__(self, other: BitInt | int | str) -> BitInt: ...
    def __radd__(self, other: BitInt | int | str) -> BitInt: ...
    def __sub__(self, other: BitInt | int | str) -> BitInt: ...
    def __rsub__(self, other: BitInt | int | str) -> BitInt: ...
    def __mul__(self, other: BitInt | int | str) -> BitInt: ...
    def __rmul__(self, other: BitInt | int | str) -> BitInt: ...
    def __floordiv__(self, other: BitInt | int | str) -> BitInt: ...
    def __mod__(self, other: BitInt | int | str) -> BitInt: ...
    def set(self, other: BitInt | int | str) -> None: ...
    def set_clip(self, other: BitInt | int | str) -> None: ...
    def clamp(self, size: int) -> None: ...
    def get_bit(self, index: int) -> bool: ...
    def set_bit(self, index: int, value: bool) -> None: ...
    def count_ones(self) -> int: ...
    def count_zeros(self) -> int: ...
    def is_zero(self) -> bool: ...
    def num_bits(self) -> int: ...

class BitUInt:
    """Fixed-width unsigned integer type.

    Always unsigned. int() always returns a non-negative value.

    Examples:
        >>> u = BitUInt(8, 42)   # 8-bit unsigned with value 42
        >>> u = BitUInt("1010")  # 4-bit unsigned from binary string
        >>> u = BitUInt(1, 1)    # 1-bit: int() returns 1, not -1
    """

    @property
    def size(self) -> int:
        """Number of bits."""
        ...

    @property
    def signed(self) -> bool:
        """Always False."""
        ...

    @overload
    def __init__(self, binary_str: str, value: None = None) -> None: ...
    @overload
    def __init__(self, size: int, value: int = 0) -> None: ...
    def __init__(self, size: str | int, value: int | None = None) -> None: ...
    @staticmethod
    def from_binary(s: str) -> BitUInt: ...
    @staticmethod
    def zeros(size: int) -> BitUInt: ...
    @staticmethod
    def ones(size: int) -> BitUInt: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    def __int__(self) -> int: ...
    def __index__(self) -> int: ...
    def __len__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __bool__(self) -> bool: ...
    def to_binary_str(self, reverse_bits: bool = False, separator: str | None = None) -> str: ...
    def __getitem__(self, index: int) -> int: ...
    def __setitem__(self, index: int, value: int) -> None: ...
    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __lt__(self, other: BitUInt | int | str) -> bool: ...
    def __le__(self, other: BitUInt | int | str) -> bool: ...
    def __gt__(self, other: BitUInt | int | str) -> bool: ...
    def __ge__(self, other: BitUInt | int | str) -> bool: ...
    def __and__(self, other: BitUInt | int | str) -> BitUInt: ...
    def __rand__(self, other: BitUInt | int | str) -> BitUInt: ...
    def __or__(self, other: BitUInt | int | str) -> BitUInt: ...
    def __ror__(self, other: BitUInt | int | str) -> BitUInt: ...
    def __xor__(self, other: BitUInt | int | str) -> BitUInt: ...
    def __rxor__(self, other: BitUInt | int | str) -> BitUInt: ...
    def __invert__(self) -> BitUInt: ...
    def __lshift__(self, other: BitUInt | int) -> BitUInt: ...
    def __rshift__(self, other: BitUInt | int) -> BitUInt: ...
    def __add__(self, other: BitUInt | int | str) -> BitUInt: ...
    def __radd__(self, other: BitUInt | int | str) -> BitUInt: ...
    def __sub__(self, other: BitUInt | int | str) -> BitUInt: ...
    def __rsub__(self, other: BitUInt | int | str) -> BitUInt: ...
    def __mul__(self, other: BitUInt | int | str) -> BitUInt: ...
    def __rmul__(self, other: BitUInt | int | str) -> BitUInt: ...
    def __floordiv__(self, other: BitUInt | int | str) -> BitUInt: ...
    def __mod__(self, other: BitUInt | int | str) -> BitUInt: ...
    def set(self, other: BitUInt | int | str) -> None: ...
    def set_clip(self, other: BitUInt | int | str) -> None: ...
    def clamp(self, size: int) -> None: ...
    def get_bit(self, index: int) -> bool: ...
    def set_bit(self, index: int, value: bool) -> None: ...
    def count_ones(self) -> int: ...
    def count_zeros(self) -> int: ...
    def is_zero(self) -> bool: ...
    def num_bits(self) -> int: ...

# =============================================================================
# DType System
# =============================================================================
class DType:
    """Data type descriptor."""

    @property
    def name(self) -> str: ...
    @property
    def type(self) -> type[Scalar]: ...
    def __repr__(self) -> str: ...

class DTypes:
    """Container for dtype instances."""

    @property
    def i8(self) -> DType: ...
    @property
    def i16(self) -> DType: ...
    @property
    def i32(self) -> DType: ...
    @property
    def i64(self) -> DType: ...
    @property
    def u8(self) -> DType: ...
    @property
    def u16(self) -> DType: ...
    @property
    def u32(self) -> DType: ...
    @property
    def u64(self) -> DType: ...
    @property
    def f32(self) -> DType: ...
    @property
    def f64(self) -> DType: ...
    @property
    def complex64(self) -> DType: ...
    @property
    def complex128(self) -> DType: ...
    @property
    def bool(self) -> DType: ...

dtypes: DTypes

# =============================================================================
# Array Type
# =============================================================================
class Array(Generic[_ScalarT]):
    """N-dimensional array with NumPy-like interface."""

    def __class_getitem__(cls, item: type[Scalar]) -> type[Array[Scalar]]: ...
    @property
    def shape(self) -> tuple[int, ...]: ...
    @property
    def ndim(self) -> int: ...
    @property
    def dtype(self) -> DType: ...
    @property
    def size(self) -> int: ...
    @property
    def T(self) -> Array[_ScalarT]: ...
    def __init__(
        self,
        data: Sequence[int | float | complex] | Sequence[Sequence[int | float | complex]],
        dtype: type[Scalar] | DType | None = None,
    ) -> None: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[Scalar | Array[_ScalarT]]: ...

    # Indexing
    @overload
    def __getitem__(self, key: int) -> Scalar | Array[_ScalarT]: ...
    @overload
    def __getitem__(self, key: slice) -> Array[_ScalarT]: ...
    @overload
    def __getitem__(self, key: tuple[int | slice, ...]) -> Scalar | Array[_ScalarT]: ...
    def __setitem__(
        self,
        key: int | slice | tuple[int | slice, ...],
        value: Scalar | int | float | complex | Array[Scalar],
    ) -> None: ...

    # Arithmetic
    def __add__(self, other: Array[Scalar] | Scalar | int | float | complex) -> Array[_ScalarT]: ...
    def __radd__(self, other: Array[Scalar] | Scalar | int | float | complex) -> Array[_ScalarT]: ...
    def __sub__(self, other: Array[Scalar] | Scalar | int | float | complex) -> Array[_ScalarT]: ...
    def __rsub__(self, other: Array[Scalar] | Scalar | int | float | complex) -> Array[_ScalarT]: ...
    def __mul__(
        self, other: Array[Scalar] | Scalar | int | float | complex
    ) -> Array[_ScalarT]: ...  # matmul if Array, scaling if scalar
    def __rmul__(
        self, other: Array[Scalar] | Scalar | int | float | complex
    ) -> Array[_ScalarT]: ...  # matmul if Array, scaling if scalar
    def __truediv__(self, other: Array[Scalar] | Scalar | int | float | complex) -> Array[ScalarF64]: ...
    def __rtruediv__(self, other: Array[Scalar] | Scalar | int | float | complex) -> Array[ScalarF64]: ...
    def __floordiv__(self, other: Array[Scalar] | Scalar | int | float) -> Array[_ScalarT]: ...
    def __rfloordiv__(self, other: Array[Scalar] | Scalar | int | float) -> Array[_ScalarT]: ...
    def __mod__(self, other: Array[Scalar] | Scalar | int | float) -> Array[_ScalarT]: ...
    def __rmod__(self, other: Array[Scalar] | Scalar | int | float) -> Array[_ScalarT]: ...
    def __pow__(self, other: Array[Scalar] | Scalar | int | float) -> Array[ScalarF64]: ...
    def __rpow__(self, other: Array[Scalar] | Scalar | int | float) -> Array[ScalarF64]: ...
    def __neg__(self) -> Array[_ScalarT]: ...
    def __pos__(self) -> Array[_ScalarT]: ...
    def __abs__(self) -> Array[_ScalarT]: ...
    def __and__(self, other: Array[Scalar]) -> Array[_ScalarT]: ...  # Kronecker product
    def __rand__(self, other: Array[Scalar]) -> Array[_ScalarT]: ...  # Kronecker product (reverse)

    # Comparison
    def __eq__(self, other: object) -> Array[ScalarU8]: ...  # type: ignore[override]
    def __ne__(self, other: object) -> Array[ScalarU8]: ...  # type: ignore[override]
    def __lt__(self, other: Array[Scalar] | Scalar | int | float) -> Array[ScalarU8]: ...
    def __le__(self, other: Array[Scalar] | Scalar | int | float) -> Array[ScalarU8]: ...
    def __gt__(self, other: Array[Scalar] | Scalar | int | float) -> Array[ScalarU8]: ...
    def __ge__(self, other: Array[Scalar] | Scalar | int | float) -> Array[ScalarU8]: ...

    # Linear algebra
    def dot(self, other: Array[Scalar]) -> Array[_ScalarT]: ...
    def elemwise_mul(self, other: Array[Scalar] | Scalar | int | float | complex) -> Array[_ScalarT]: ...
    def conj(self) -> Array[_ScalarT]: ...

    # Methods
    def reshape(self, *shape: int) -> Array[_ScalarT]: ...
    def flatten(self) -> Array[_ScalarT]: ...
    def ravel(self) -> Array[_ScalarT]: ...
    def transpose(self, *axes: int) -> Array[_ScalarT]: ...
    def sum(self, axis: int | None = None) -> Scalar | Array[_ScalarT]: ...
    def mean(self, axis: int | None = None) -> ScalarF64 | Array[ScalarF64]: ...
    def std(self, axis: int | None = None, ddof: int = 0) -> ScalarF64 | Array[ScalarF64]: ...
    def max(self, axis: int | None = None) -> Scalar | Array[_ScalarT]: ...
    def min(self, axis: int | None = None) -> Scalar | Array[_ScalarT]: ...
    def argmax(self, axis: int | None = None) -> ScalarI64 | Array[ScalarI64]: ...
    def argmin(self, axis: int | None = None) -> ScalarI64 | Array[ScalarI64]: ...
    def copy(self) -> Array[_ScalarT]: ...
    def astype(self, dtype: type[Scalar] | DType) -> Array[Scalar]: ...
    def tolist(
        self,
    ) -> list[int | float | complex] | list[list[int | float | complex]]: ...

# Array factory function
def array(
    data: Sequence[int | float | complex] | Sequence[Sequence[int | float | complex]],
    dtype: type[Scalar] | DType | None = None,
) -> Array[Scalar]: ...

# =============================================================================
# Array Creation Functions
# =============================================================================
def zeros(shape: int | tuple[int, ...], dtype: type[Scalar] | DType | None = None) -> Array[Scalar]: ...
def ones(shape: int | tuple[int, ...], dtype: type[Scalar] | DType | None = None) -> Array[Scalar]: ...
def linspace(
    start: float, stop: float, num: int = 50, dtype: type[Scalar] | DType | None = None
) -> Array[ScalarF64]: ...
def arange(
    start: float,
    stop: float | None = None,
    step: float = 1.0,
    dtype: type[Scalar] | DType | None = None,
) -> Array[Scalar]: ...
def diag(v: Array[Scalar], k: int = 0) -> Array[Scalar]: ...
def delete(arr: Array[Scalar], indices: int | Sequence[int], axis: int | None = None) -> Array[Scalar]: ...
def kron(a: Array[Scalar], b: Array[Scalar]) -> Array[Scalar]: ...

# =============================================================================
# Mathematical Functions
# =============================================================================
def mean(a: Array[Scalar], axis: int | None = None) -> ScalarF64 | Array[ScalarF64]: ...
def std(a: Array[Scalar], axis: int | None = None, ddof: int = 0) -> ScalarF64 | Array[ScalarF64]: ...
def sum(a: Array[Scalar], axis: int | None = None) -> Scalar | Array[Scalar]: ...  # noqa: A001
def max(a: Array[Scalar], axis: int | None = None) -> Scalar | Array[Scalar]: ...  # noqa: A001
def min(a: Array[Scalar], axis: int | None = None) -> Scalar | Array[Scalar]: ...  # noqa: A001
def power(x: Array[Scalar] | Scalar | float, y: Array[Scalar] | Scalar | float) -> Array[ScalarF64] | ScalarF64: ...
def sqrt(x: Array[Scalar] | Scalar | float) -> Array[ScalarF64] | ScalarF64: ...
def exp(x: Array[Scalar] | Scalar | float) -> Array[ScalarF64] | ScalarF64: ...
def ln(x: Array[Scalar] | Scalar | float) -> Array[ScalarF64] | ScalarF64: ...
def log(x: Array[Scalar] | Scalar | float, base: float | None = None) -> Array[ScalarF64] | ScalarF64: ...
def abs(x: Array[Scalar] | Scalar | float) -> Array[Scalar] | Scalar: ...  # noqa: A001
def floor(x: Array[Scalar] | Scalar | float) -> Array[ScalarF64] | ScalarF64: ...
def ceil(x: Array[Scalar] | Scalar | float) -> Array[ScalarF64] | ScalarF64: ...
def round(x: Array[Scalar] | Scalar | float, decimals: int = 0) -> Array[ScalarF64] | ScalarF64: ...  # noqa: A001

# Trigonometric functions
def cos(x: Array[Scalar] | Scalar | float) -> Array[ScalarF64] | ScalarF64: ...
def sin(x: Array[Scalar] | Scalar | float) -> Array[ScalarF64] | ScalarF64: ...
def tan(x: Array[Scalar] | Scalar | float) -> Array[ScalarF64] | ScalarF64: ...
def acos(x: Array[Scalar] | Scalar | float) -> Array[ScalarF64] | ScalarF64: ...
def asin(x: Array[Scalar] | Scalar | float) -> Array[ScalarF64] | ScalarF64: ...
def atan(x: Array[Scalar] | Scalar | float) -> Array[ScalarF64] | ScalarF64: ...
def atan2(y: Array[Scalar] | Scalar | float, x: Array[Scalar] | Scalar | float) -> Array[ScalarF64] | ScalarF64: ...

# Hyperbolic functions
def sinh(x: Array[Scalar] | Scalar | float) -> Array[ScalarF64] | ScalarF64: ...
def cosh(x: Array[Scalar] | Scalar | float) -> Array[ScalarF64] | ScalarF64: ...
def tanh(x: Array[Scalar] | Scalar | float) -> Array[ScalarF64] | ScalarF64: ...
def asinh(x: Array[Scalar] | Scalar | float) -> Array[ScalarF64] | ScalarF64: ...
def acosh(x: Array[Scalar] | Scalar | float) -> Array[ScalarF64] | ScalarF64: ...
def atanh(x: Array[Scalar] | Scalar | float) -> Array[ScalarF64] | ScalarF64: ...

# Comparison functions
def isnan(x: Array[Scalar] | Scalar | float) -> Array[ScalarU8] | bool: ...
def isclose(
    a: Array[Scalar] | Scalar | float,
    b: Array[Scalar] | Scalar | float,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> Array[ScalarU8] | bool: ...
def allclose(
    a: Array[Scalar] | Scalar | float,
    b: Array[Scalar] | Scalar | float,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> bool: ...
def array_equal(a: Array[Scalar], b: Array[Scalar]) -> bool: ...
def all(a: Array[Scalar], axis: int | None = None) -> bool | Array[ScalarU8]: ...  # noqa: A001
def any(a: Array[Scalar], axis: int | None = None) -> bool | Array[ScalarU8]: ...  # noqa: A001
def where(
    condition: Array[Scalar],
    x: Array[Scalar] | Scalar | float,
    y: Array[Scalar] | Scalar | float,
) -> Array[Scalar]: ...

# Constants
inf: float
nan: float

# =============================================================================
# Linear Algebra Module
# =============================================================================
class linalg:
    """Linear algebra functions module."""

    @staticmethod
    def norm(x: Array[Scalar] | Sequence[int | float | complex], ord: float | None = None) -> float: ...
    @staticmethod
    def expm(a: Array[Scalar]) -> Array[Scalar]: ...
    @staticmethod
    def matrix_power(a: Array[Scalar], n: int) -> Array[Scalar]: ...

# =============================================================================
# Optimization Functions
# =============================================================================
def brentq(
    f: Callable[[float], float],
    a: float,
    b: float,
    xtol: float = 2e-12,
    rtol: float = 8.881784197001252e-16,
    maxiter: int = 100,
) -> float: ...
def newton(
    func: Callable[[float], float],
    x0: float,
    fprime: Callable[[float], float] | None = None,
    tol: float = 1.48e-8,
    maxiter: int = 50,
) -> float: ...
def curve_fit(
    f: Callable[..., float | Array[Scalar]],
    xdata: Array[Scalar],
    ydata: Array[Scalar],
    p0: Sequence[float] | None = None,
    sigma: Array[Scalar] | None = None,
    absolute_sigma: bool = False,
    bounds: tuple[Sequence[float], Sequence[float]] | None = None,
) -> tuple[Array[ScalarF64], Array[ScalarF64]]: ...
def polyfit(x: Array[Scalar], y: Array[Scalar], deg: int) -> Array[ScalarF64]: ...

class Poly1d:
    """Polynomial class for evaluation and manipulation."""

    def __init__(self, coeffs: Sequence[float] | Array[Scalar]) -> None: ...
    def __call__(self, x: float | Array[Scalar]) -> float | Array[ScalarF64]: ...
    @property
    def coeffs(self) -> Array[ScalarF64]: ...
    @property
    def order(self) -> int: ...
    def __repr__(self) -> str: ...

# =============================================================================
# Random Module
# =============================================================================
class random:
    """Random number generation module."""

    @staticmethod
    def seed(seed: int | None = None) -> None: ...
    @staticmethod
    def random(
        size: int | tuple[int, ...] | None = None,
    ) -> float | Array[ScalarF64]: ...
    @staticmethod
    def uniform(
        low: float = 0.0, high: float = 1.0, size: int | tuple[int, ...] | None = None
    ) -> float | Array[ScalarF64]: ...
    @staticmethod
    def normal(
        loc: float = 0.0, scale: float = 1.0, size: int | tuple[int, ...] | None = None
    ) -> float | Array[ScalarF64]: ...
    @staticmethod
    def randint(
        low: int, high: int | None = None, size: int | tuple[int, ...] | None = None
    ) -> int | Array[ScalarI64]: ...
    @staticmethod
    def choice(
        a: int | Sequence[_T] | Array[Scalar],
        size: int | tuple[int, ...] | None = None,
        replace: bool = True,
        p: Sequence[float] | Array[Scalar] | None = None,
    ) -> _T | Array[Scalar]: ...
    @staticmethod
    def permutation(x: int | Sequence[_T] | Array[Scalar]) -> Array[Scalar]: ...
    @staticmethod
    def shuffle(x: list[_T] | Array[Scalar]) -> None: ...

# =============================================================================
# Statistics Module
# =============================================================================
class stats:
    """Statistical functions module."""

    class norm:
        """Normal distribution."""

        @staticmethod
        def pdf(x: float | Array[Scalar], loc: float = 0.0, scale: float = 1.0) -> float | Array[ScalarF64]: ...
        @staticmethod
        def cdf(x: float | Array[Scalar], loc: float = 0.0, scale: float = 1.0) -> float | Array[ScalarF64]: ...
        @staticmethod
        def ppf(q: float | Array[Scalar], loc: float = 0.0, scale: float = 1.0) -> float | Array[ScalarF64]: ...
        @staticmethod
        def rvs(
            loc: float = 0.0,
            scale: float = 1.0,
            size: int | tuple[int, ...] | None = None,
        ) -> float | Array[ScalarF64]: ...

# =============================================================================
# Num Module (namespace for numerical functions)
# =============================================================================
class num:
    """Numerical computing module."""

    # Re-export all functions
    mean = mean
    std = std
    sum = sum
    max = max
    min = min
    power = power
    sqrt = sqrt
    exp = exp
    ln = ln
    log = log
    abs = abs
    floor = floor
    ceil = ceil
    round = round
    cos = cos
    sin = sin
    tan = tan
    acos = acos
    asin = asin
    atan = atan
    atan2 = atan2
    sinh = sinh
    cosh = cosh
    tanh = tanh
    asinh = asinh
    acosh = acosh
    atanh = atanh
    isnan = isnan
    isclose = isclose
    allclose = allclose
    array_equal = array_equal
    all = all
    any = any
    where_array = where
    brentq = brentq
    newton = newton
    curve_fit = curve_fit
    polyfit = polyfit
    Poly1d = Poly1d
    diag = diag
    linspace = linspace
    arange = arange
    zeros = zeros
    ones = ones
    delete = delete
    inf = inf
    nan = nan
    random = random
    stats = stats

    class math:
        """Math submodule."""

        power = power
        sqrt = sqrt
        exp = exp
        abs = abs
        cos = cos
        sin = sin
        tan = tan
        acos = acos
        asin = asin
        atan = atan
        atan2 = atan2
        sinh = sinh
        cosh = cosh
        tanh = tanh
        asinh = asinh
        acosh = acosh
        atanh = atanh

    class compare:
        """Comparison submodule."""

        isnan = isnan
        isclose = isclose
        allclose = allclose
        array_equal = array_equal

# =============================================================================
# Quantum Simulators
# =============================================================================
class TableauWrapper:
    """Wrapper for accessing stabilizer/destabilizer tableaus from simulators."""

    def __init__(self, sim: object, *, is_stab: bool) -> None: ...
    def print_tableau(self, *, verbose: bool = False) -> list[str]: ...
    @property
    def col_x(self) -> list[list[int]]: ...
    @property
    def col_z(self) -> list[list[int]]: ...
    @property
    def row_x(self) -> list[list[int]]: ...
    @property
    def row_z(self) -> list[list[int]]: ...

class GateBindingsDict:
    """Special dict that delegates gate lookups to run_gate()."""

    def __init__(self, sim: object) -> None: ...
    def __getitem__(self, key: str) -> object: ...
    def __setitem__(self, key: str, value: object) -> None: ...
    def __contains__(self, key: str) -> bool: ...
    def get(self, key: str, default: object | None = None) -> object: ...
    def __len__(self) -> int: ...
    def keys(self) -> list[str]: ...

class SparseSim:
    """Sparse stabilizer simulator."""

    def __init__(self, num_qubits: int, seed: int | None = None) -> None: ...
    def set_seed(self, seed: int) -> None: ...
    def reset(self) -> SparseSim: ...
    @property
    def num_qubits(self) -> int: ...
    @property
    def stabs(self) -> TableauWrapper: ...
    @property
    def destabs(self) -> TableauWrapper: ...
    @property
    def gens(self) -> tuple[TableauWrapper, TableauWrapper]: ...
    @property
    def bindings(self) -> GateBindingsDict: ...
    def __repr__(self) -> str: ...

class Stab:
    """Generic stabilizer simulator (recommended)."""

    def __init__(self, num_qubits: int, seed: int | None = None) -> None: ...
    def set_seed(self, seed: int) -> None: ...
    def reset(self) -> Stab: ...
    @property
    def num_qubits(self) -> int: ...
    def stab_tableau(self) -> str: ...
    def destab_tableau(self) -> str: ...
    @property
    def stabs(self) -> TableauWrapper: ...
    @property
    def destabs(self) -> TableauWrapper: ...
    @property
    def gens(self) -> tuple[TableauWrapper, TableauWrapper]: ...
    @property
    def bindings(self) -> GateBindingsDict: ...

class StateVec:
    """Rust state vector simulator."""

    def __init__(self, num_qubits: int) -> None: ...
    def reset(self) -> StateVec: ...
    @property
    def num_qubits(self) -> int: ...
    @property
    def vector(self) -> Array: ...
    @property
    def probabilities(self) -> Array: ...
    def probability(self, basis_state: int) -> float: ...
    def vector_big_endian(self) -> Array: ...

class Qulacs:
    """Rust Qulacs state vector simulator."""

    def __init__(self, num_qubits: int, *, seed: int | None = None) -> None: ...
    def reset(self) -> Qulacs: ...
    @property
    def num_qubits(self) -> int: ...
    @property
    def probabilities(self) -> list[float]: ...

class SparseStab:
    """Rust sparse stabilizer simulator."""

    def __init__(self, num_qubits: int, *, seed: int | None = None) -> None: ...
    @property
    def num_qubits(self) -> int: ...

class Stabilizer:
    """Rust stabilizer simulator."""

    def __init__(self, num_qubits: int, *, seed: int | None = None) -> None: ...
    @property
    def num_qubits(self) -> int: ...

class CliffordRz:
    """Rust Clifford+RZ simulator."""

    def __init__(
        self,
        num_qubits: int,
        seed: int | None = None,
        pruning_threshold: float | None = None,
        mc_threshold: int | None = 2048,
    ) -> None: ...
    @property
    def num_qubits(self) -> int: ...
    @property
    def num_terms(self) -> int: ...

class CoinToss:
    """Coin toss simulator for random measurement outcomes."""

    def __init__(self, num_qubits: int, prob: float = 0.5, seed: int | None = None) -> None: ...
    @property
    def num_qubits(self) -> int: ...

class QuestStateVec:
    """QuEST state vector simulator."""

    def __init__(self, num_qubits: int) -> None: ...
    def reset(self) -> QuestStateVec: ...
    @property
    def num_qubits(self) -> int: ...

class QuestDensityMatrix:
    """QuEST density matrix simulator."""

    def __init__(self, num_qubits: int) -> None: ...
    def reset(self) -> QuestDensityMatrix: ...
    @property
    def num_qubits(self) -> int: ...

# =============================================================================
# Engine Types
# =============================================================================
class SparseStabEngine:
    """Sparse stabilizer engine."""

    ...

class StateVecEngine:
    """State vector engine."""

    ...

# =============================================================================
# Program Types
# =============================================================================
class QasmProgram:
    """OpenQASM program representation."""

    ...

class QisProgram:
    """QIS program representation."""

    ...

class HugrProgram:
    """HUGR program representation."""

    ...

class PhirJsonProgram:
    """PHIR JSON program representation."""

    ...

class WasmProgram:
    """WebAssembly program representation."""

    ...

class WatProgram:
    """WebAssembly Text format program representation."""

    ...

# =============================================================================
# Engine Builders
# =============================================================================
class QasmEngineBuilder:
    """Builder for QASM engines."""

    ...

class QisEngineBuilder:
    """Builder for QIS engines."""

    ...

class PhirJsonEngineBuilder:
    """Builder for PHIR JSON engines."""

    ...

class SimBuilder:
    """General simulation builder."""

    ...

class StateVectorEngineBuilder:
    """Builder for state vector engines."""

    ...

class SparseStabEngineBuilder:
    """Builder for sparse stabilizer engines."""

    ...

class StabilizerEngineBuilder:
    """Builder for stabilizer engines."""

    ...

class CliffordRzEngineBuilder:
    """Builder for Clifford+RZ engines."""

    ...

class DensityMatrixEngineBuilder:
    """Builder for density matrix engines."""

    ...

class CoinTossEngineBuilder:
    """Builder for coin toss engines."""

    ...

class QisInterfaceBuilder:
    """Builder for QIS interfaces."""

    ...

# =============================================================================
# Noise Model Builders
# =============================================================================
class GeneralNoiseModelBuilder:
    """Builder for general noise models."""

    ...

class DepolarizingNoiseModelBuilder:
    """Builder for depolarizing noise models."""

    ...

class BiasedDepolarizingNoiseModelBuilder:
    """Builder for biased depolarizing noise models."""

    ...

# =============================================================================
# PHIR Types
# =============================================================================
class PhirJsonEngine:
    """PHIR JSON execution engine."""

    ...

class PhirJsonSimulation:
    """PHIR JSON simulation instance."""

    ...

# =============================================================================
# Result Types
# =============================================================================
class ByteMessage:
    """Binary message type for efficient data transfer."""

    ...

class ByteMessageBuilder:
    """Builder for ByteMessage objects."""

    ...

class ShotMap:
    """Map of measurement outcomes."""

    ...

class ShotVec:
    """Vector of measurement outcomes."""

    ...

# =============================================================================
# Quantum Types
# =============================================================================
class Pauli:
    """Single Pauli operator (I, X, Y, Z)."""

    ...

class PauliString:
    """String of Pauli operators."""

    ...

class PauliStabilizerGroup:
    """A group of commuting Pauli operators with real phases."""

    ...

class PauliSequence:
    """Ordered sequence of Pauli operators with symplectic analysis."""

    ...

class CliffordRep:
    """Clifford gate in the Heisenberg picture."""

    # Core
    @staticmethod
    def identity(num_qubits: int) -> CliffordRep: ...

    # Pauli gates
    @staticmethod
    def x(q: int) -> CliffordRep: ...
    @staticmethod
    def y(q: int) -> CliffordRep: ...
    @staticmethod
    def z(q: int) -> CliffordRep: ...

    # Hadamard variants
    @staticmethod
    def h(q: int) -> CliffordRep: ...
    @staticmethod
    def h2(q: int) -> CliffordRep: ...
    @staticmethod
    def h3(q: int) -> CliffordRep: ...
    @staticmethod
    def h4(q: int) -> CliffordRep: ...
    @staticmethod
    def h5(q: int) -> CliffordRep: ...
    @staticmethod
    def h6(q: int) -> CliffordRep: ...

    # Sqrt gates and daggers
    @staticmethod
    def sx(q: int) -> CliffordRep: ...
    @staticmethod
    def sxdg(q: int) -> CliffordRep: ...
    @staticmethod
    def sy(q: int) -> CliffordRep: ...
    @staticmethod
    def sydg(q: int) -> CliffordRep: ...
    @staticmethod
    def sz(q: int) -> CliffordRep: ...
    @staticmethod
    def szdg(q: int) -> CliffordRep: ...

    # Face gates
    @staticmethod
    def f(q: int) -> CliffordRep: ...
    @staticmethod
    def fdg(q: int) -> CliffordRep: ...
    @staticmethod
    def f2(q: int) -> CliffordRep: ...
    @staticmethod
    def f2dg(q: int) -> CliffordRep: ...
    @staticmethod
    def f3(q: int) -> CliffordRep: ...
    @staticmethod
    def f3dg(q: int) -> CliffordRep: ...
    @staticmethod
    def f4(q: int) -> CliffordRep: ...
    @staticmethod
    def f4dg(q: int) -> CliffordRep: ...

    # Two-qubit gates
    @staticmethod
    def cx(c: int, t: int) -> CliffordRep: ...
    @staticmethod
    def cy(c: int, t: int) -> CliffordRep: ...
    @staticmethod
    def cz(a: int, b: int) -> CliffordRep: ...
    @staticmethod
    def swap(a: int, b: int) -> CliffordRep: ...
    @staticmethod
    def sxx(a: int, b: int) -> CliffordRep: ...
    @staticmethod
    def sxxdg(a: int, b: int) -> CliffordRep: ...
    @staticmethod
    def syy(a: int, b: int) -> CliffordRep: ...
    @staticmethod
    def syydg(a: int, b: int) -> CliffordRep: ...
    @staticmethod
    def szz(a: int, b: int) -> CliffordRep: ...
    @staticmethod
    def szzdg(a: int, b: int) -> CliffordRep: ...
    @staticmethod
    def iswap(a: int, b: int) -> CliffordRep: ...
    @staticmethod
    def g(a: int, b: int) -> CliffordRep: ...

    # Enumeration and random sampling
    @staticmethod
    def single_qubit_cliffords(qubit: int) -> list[CliffordRep]: ...
    @staticmethod
    def random_single_qubit(qubit: int) -> CliffordRep: ...
    @staticmethod
    def random(num_qubits: int, depth: int | None = None) -> CliffordRep: ...
    @staticmethod
    def from_pauli_string(pauli: PauliString) -> CliffordRep: ...

    # Operations
    def compose(self, other: CliffordRep) -> CliffordRep: ...
    def inverse(self) -> CliffordRep: ...
    def extended_to(self, n: int) -> CliffordRep: ...
    def apply(self, pauli: PauliString) -> PauliString: ...
    def apply_to_group(self, group: PauliStabilizerGroup) -> PauliStabilizerGroup: ...
    def is_valid(self) -> bool: ...
    def num_qubits(self) -> int: ...
    def x_image(self, qubit: int) -> PauliString: ...
    def z_image(self, qubit: int) -> PauliString: ...
    def __mul__(self, other: CliffordRep) -> CliffordRep: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

class PauliPropRs:
    """Pauli propagator (Rust implementation)."""

    ...

# =============================================================================
# Clifford Gate Constructors
# =============================================================================

# Hadamard variants
def H(q: int) -> CliffordRep: ...
def H2(q: int) -> CliffordRep: ...
def H3(q: int) -> CliffordRep: ...
def H4(q: int) -> CliffordRep: ...
def H5(q: int) -> CliffordRep: ...
def H6(q: int) -> CliffordRep: ...

# Sqrt gates and daggers
def SX(q: int) -> CliffordRep: ...
def SXdg(q: int) -> CliffordRep: ...
def SY(q: int) -> CliffordRep: ...
def SYdg(q: int) -> CliffordRep: ...
def SZ(q: int) -> CliffordRep: ...
def SZdg(q: int) -> CliffordRep: ...

# Face gates
def F(q: int) -> CliffordRep: ...
def Fdg(q: int) -> CliffordRep: ...
def F2(q: int) -> CliffordRep: ...
def F2dg(q: int) -> CliffordRep: ...
def F3(q: int) -> CliffordRep: ...
def F3dg(q: int) -> CliffordRep: ...
def F4(q: int) -> CliffordRep: ...
def F4dg(q: int) -> CliffordRep: ...

# Two-qubit gates
def CX(c: int, t: int) -> CliffordRep: ...
def CY(c: int, t: int) -> CliffordRep: ...
def CZ(a: int, b: int) -> CliffordRep: ...
def SWAP(a: int, b: int) -> CliffordRep: ...
def SXX(a: int, b: int) -> CliffordRep: ...
def SXXdg(a: int, b: int) -> CliffordRep: ...
def SYY(a: int, b: int) -> CliffordRep: ...
def SYYdg(a: int, b: int) -> CliffordRep: ...
def SZZ(a: int, b: int) -> CliffordRep: ...
def SZZdg(a: int, b: int) -> CliffordRep: ...
def ISWAP(a: int, b: int) -> CliffordRep: ...
def G(a: int, b: int) -> CliffordRep: ...

# =============================================================================
# Graph Module
# =============================================================================
class Graph:
    """Graph data structure for MWPM and other algorithms."""

    ...

class graph:
    """Graph algorithms module."""

    Graph = Graph
    # Additional graph functions would go here

# =============================================================================
# LLVM/IR Modules
# =============================================================================
class ir:
    """LLVM IR generation module."""

    ...

class binding:
    """LLVM binding generation module."""

    ...

class llvm:
    """LLVM namespace module."""

    ...

# =============================================================================
# Namespace Modules
# =============================================================================
class quantum:
    """Quantum simulation namespace."""

    state_vector: Callable[..., StateVectorEngineBuilder]
    sparse_stab: Callable[..., SparseStabEngineBuilder]
    stabilizer: Callable[..., StabilizerEngineBuilder]
    clifford_rz: Callable[..., CliffordRzEngineBuilder]
    density_matrix: Callable[..., DensityMatrixEngineBuilder]
    coin_toss: Callable[..., CoinTossEngineBuilder]
    StateVectorEngineBuilder: type[StateVectorEngineBuilder]
    SparseStabEngineBuilder: type[SparseStabEngineBuilder]
    StabilizerEngineBuilder: type[StabilizerEngineBuilder]
    CliffordRzEngineBuilder: type[CliffordRzEngineBuilder]
    DensityMatrixEngineBuilder: type[DensityMatrixEngineBuilder]
    CoinTossEngineBuilder: type[CoinTossEngineBuilder]

class noise:
    """Noise model namespace."""

    general_noise: Callable[..., GeneralNoiseModelBuilder]
    depolarizing_noise: Callable[..., DepolarizingNoiseModelBuilder]
    biased_depolarizing_noise: Callable[..., BiasedDepolarizingNoiseModelBuilder]
    GeneralNoiseModelBuilder: type[GeneralNoiseModelBuilder]
    DepolarizingNoiseModelBuilder: type[DepolarizingNoiseModelBuilder]
    BiasedDepolarizingNoiseModelBuilder: type[BiasedDepolarizingNoiseModelBuilder]

# =============================================================================
# Factory Functions
# =============================================================================
def sim(**kwargs: object) -> object:
    """Create a simulation engine with the specified configuration."""
    ...

def qasm_engine(**kwargs: object) -> QasmEngineBuilder:
    """Create a QASM engine builder."""
    ...

def qis_engine(**kwargs: object) -> QisEngineBuilder:
    """Create a QIS engine builder."""
    ...

def phir_json_engine(**kwargs: object) -> PhirJsonEngineBuilder:
    """Create a PHIR JSON engine builder."""
    ...

def state_vector(**kwargs: object) -> StateVectorEngineBuilder:
    """Create a state vector engine builder."""
    ...

def sparse_stab(**kwargs: object) -> SparseStabEngineBuilder:
    """Create a sparse stabilizer engine builder."""
    ...

def stabilizer(**kwargs: object) -> StabilizerEngineBuilder:
    """Create a stabilizer engine builder."""
    ...

def clifford_rz(**kwargs: object) -> CliffordRzEngineBuilder:
    """Create a Clifford+RZ engine builder."""
    ...

def density_matrix(**kwargs: object) -> DensityMatrixEngineBuilder:
    """Create a density matrix engine builder."""
    ...

def coin_toss(**kwargs: object) -> CoinTossEngineBuilder:
    """Create a coin toss engine builder."""
    ...

def general_noise(**kwargs: object) -> GeneralNoiseModelBuilder:
    """Create a general noise model builder."""
    ...

def depolarizing_noise(p: float) -> DepolarizingNoiseModelBuilder:
    """Create a depolarizing noise model builder."""
    ...

def biased_depolarizing_noise(px: float, py: float, pz: float) -> BiasedDepolarizingNoiseModelBuilder:
    """Create a biased depolarizing noise model builder."""
    ...

def qis_helios_interface(**kwargs: object) -> QisInterfaceBuilder:
    """Create a QIS Helios interface builder."""
    ...

def qis_selene_helios_interface(**kwargs: object) -> QisInterfaceBuilder:
    """Create a QIS Selene-Helios interface builder."""
    ...

# =============================================================================
# HUGR Compilation
# =============================================================================
def compile_hugr_to_qis(hugr_bytes: bytes, output_path: str | None = None) -> str:
    """Compile HUGR bytes to QIS (LLVM IR with quantum instructions)."""
    ...

def get_compilation_backends() -> dict[str, object]:
    """Get information about available compilation backends."""
    ...

# =============================================================================
# WASM
# =============================================================================
class WasmError(Exception): ...

class WasmForeignObject:
    """WebAssembly foreign object for hybrid quantum/classical computation.

    Provides WebAssembly execution capabilities using the Rust Wasmtime runtime.
    WASM modules can be loaded from files (.wasm or .wat) or directly from bytes.

    For clearer code, prefer using the explicit classmethods:
    - `WasmForeignObject.from_file()` - Load from a file path
    - `WasmForeignObject.from_bytes()` - Load from binary bytes in memory

    Example:
        >>> wasm = WasmForeignObject.from_file("math.wasm")
        >>> wasm.init()
        >>> result = wasm.exec("add", [5, 3])
        >>> print(result)  # 8
    """

    def __init__(
        self,
        file: str | os.PathLike[str] | bytes,
        timeout: float | None = None,
        memory_size: int | None = None,
    ) -> None:
        """Create a WebAssembly foreign object.

        Args:
            file: Path to WASM file (str or pathlib.Path) or WASM bytes (bytes)
            timeout: Optional timeout in seconds (default: 1.0 second)
            memory_size: Optional maximum memory size in bytes per linear memory
                        (default: None = unlimited)

        Raises:
            FileNotFoundError: If file path doesn't exist
            RuntimeError: If WASM compilation fails
        """
        ...

    @staticmethod
    def from_file(
        path: str | os.PathLike[str],
        timeout: float | None = None,
        memory_size: int | None = None,
    ) -> WasmForeignObject:
        """Create a WebAssembly foreign object from a file.

        Loads a WebAssembly module from a .wasm (binary) or .wat (text) file.

        Args:
            path: Path to the WASM file (str or pathlib.Path)
            timeout: Optional timeout in seconds for function execution (default: 1.0)
            memory_size: Optional maximum memory size in bytes per linear memory
                        (default: None = unlimited)

        Returns:
            New WebAssembly foreign object instance

        Raises:
            FileNotFoundError: If the file doesn't exist
            RuntimeError: If WASM compilation fails

        Example:
            >>> wasm = WasmForeignObject.from_file("math.wasm")
            >>> wasm = WasmForeignObject.from_file("math.wasm", timeout=5.0)
        """
        ...

    @staticmethod
    def from_bytes(
        data: bytes,
        timeout: float | None = None,
        memory_size: int | None = None,
    ) -> WasmForeignObject:
        """Create a WebAssembly foreign object from bytes.

        Loads a WebAssembly module directly from binary bytes in memory.
        Useful for downloaded, embedded, or programmatically generated WASM.

        Args:
            data: WASM binary as bytes
            timeout: Optional timeout in seconds for function execution (default: 1.0)
            memory_size: Optional maximum memory size in bytes per linear memory
                        (default: None = unlimited)

        Returns:
            New WebAssembly foreign object instance

        Raises:
            RuntimeError: If WASM compilation fails

        Example:
            >>> with open("math.wasm", "rb") as f:
            ...     wasm_bytes = f.read()
            >>> wasm = WasmForeignObject.from_bytes(wasm_bytes)
        """
        ...

    def init(self) -> None:
        """Initialize the WASM module.

        Must be called before using the object. Creates a new instance
        and calls the 'init' function in the WASM module.

        Raises:
            RuntimeError: If init function is missing or execution fails
        """
        ...

    def shot_reinit(self) -> None:
        """Reset variables before each shot.

        Calls the 'shot_reinit' function in the WASM module if it exists.
        This is a no-op if the function doesn't exist.

        Raises:
            RuntimeError: If shot_reinit function exists but execution fails
        """
        ...

    def new_instance(self) -> None:
        """Create a new WASM instance.

        Resets the object's internal state by creating a fresh instance.

        Raises:
            RuntimeError: If instance creation fails
        """
        ...

    def get_funcs(self) -> list[str]:
        """Get list of exported function names.

        Returns:
            List of function names exported by the WASM module
        """
        ...

    def exec(self, func_name: str, args: list[int]) -> int | tuple[int, ...]:
        """Execute a WASM function.

        Args:
            func_name: Name of the function to execute
            args: List of integer arguments (i64)

        Returns:
            Single integer for functions with one return value,
            or tuple for multiple return values

        Raises:
            RuntimeError: If function not found or execution fails
        """
        ...

    @property
    def wasm_bytes(self) -> bytes:
        """Get the WebAssembly binary bytes."""
        ...

    def teardown(self) -> None:
        """Cleanup resources.

        Stops the epoch increment thread. Called automatically when
        the object is dropped, but can be called explicitly.
        """
        ...

    def to_dict(self) -> dict[str, object]:
        """Serialize to dictionary for pickling.

        Returns:
            Dictionary containing 'fobj_class', 'wasm_bytes', 'timeout', and 'memory_size'
        """
        ...

    @staticmethod
    def from_dict(wasmtime_dict: dict[str, object]) -> WasmForeignObject:
        """Deserialize from dictionary (for pickling).

        Args:
            wasmtime_dict: Dictionary containing 'fobj_class', 'wasm_bytes',
                          and optionally 'timeout' and 'memory_size'

        Returns:
            New instance created from the dictionary
        """
        ...

    def __getstate__(self) -> dict[str, object]:
        """Support for pickle serialization."""
        ...

    def __setstate__(self, state: dict[str, object]) -> None:
        """Support for pickle deserialization."""
        ...

# =============================================================================
# Decoder Types
# =============================================================================

class decoders:
    """Decoder submodule for quantum error correction."""

    class BpResult:
        """Result from belief propagation decoders.

        Attributes:
            decoding: The decoded error vector.
            converged: Whether the decoder converged.
            iterations: Number of iterations performed.
        """

        @property
        def decoding(self) -> list[int]: ...
        @property
        def converged(self) -> bool: ...
        @property
        def iterations(self) -> int: ...
        def to_list(self) -> list[int]: ...
        def __repr__(self) -> str: ...
        def __len__(self) -> int: ...
        def __getitem__(self, idx: int) -> int: ...

    class CheckMatrix:
        """Dense check matrix for MWPM decoders."""

        def __init__(self, data: list[list[int]]) -> None: ...
        @property
        def rows(self) -> int: ...
        @property
        def cols(self) -> int: ...
        def __repr__(self) -> str: ...

    class SparseMatrix:
        """Sparse parity check matrix for LDPC decoders."""

        def __init__(self, data: list[list[int]]) -> None: ...
        @property
        def rows(self) -> int: ...
        @property
        def cols(self) -> int: ...
        def __repr__(self) -> str: ...

    class MwpmResult:
        """Result from MWPM decoders."""

        @property
        def correction(self) -> list[int]: ...
        def __repr__(self) -> str: ...

    class PyMatchingDecoder:
        """PyMatching MWPM decoder."""

        def __init__(
            self,
            check_matrix: decoders.CheckMatrix,
            weights: list[float] | None = ...,
        ) -> None: ...
        def decode(self, syndrome: list[int]) -> decoders.MwpmResult: ...
        def __repr__(self) -> str: ...

    class FusionBlossomDecoder:
        """Fusion Blossom MWPM decoder."""

        def __init__(
            self,
            check_matrix: decoders.CheckMatrix,
            weights: list[float] | None = ...,
        ) -> None: ...
        def decode(self, syndrome: list[int]) -> decoders.MwpmResult: ...
        def __repr__(self) -> str: ...

    class BpOsdBuilder:
        """Builder for BP+OSD decoder.

        Belief Propagation with Ordered Statistics Decoding post-processing.

        Args:
            pcm: Sparse parity check matrix.
            error_rate: Channel error probability.

        Example:
            >>> from pecos_rslib.decoders import BpOsdBuilder, SparseMatrix
            >>> H = SparseMatrix([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]])
            >>> decoder = BpOsdBuilder(H, error_rate=0.01).osd_method("osd_cs").osd_order(7).build()
            >>> result = decoder.decode([0, 0, 0])
        """

        def __init__(self, pcm: decoders.SparseMatrix, error_rate: float) -> None: ...
        def max_iter(self, val: int) -> decoders.BpOsdBuilder:
            """Set maximum BP iterations (default: 100)."""
            ...

        def bp_method(self, val: str) -> decoders.BpOsdBuilder:
            """Set BP algorithm: "product_sum" or "minimum_sum" (default: "product_sum")."""
            ...

        def schedule(self, val: str) -> decoders.BpOsdBuilder:
            """Set update schedule: "parallel" or "serial" (default: "parallel")."""
            ...

        def osd_method(self, val: str) -> decoders.BpOsdBuilder:
            """Set OSD variant: "off", "osd0", "osd_e", "osd_cs" (default: "osd0")."""
            ...

        def osd_order(self, val: int) -> decoders.BpOsdBuilder:
            """Set OSD order parameter (default: 0)."""
            ...

        def build(self) -> decoders.BpOsdDecoder:
            """Build the BP+OSD decoder."""
            ...

        def __repr__(self) -> str: ...

    class BpOsdDecoder:
        """BP+OSD decoder for LDPC codes.

        Created via ``BpOsdBuilder(...).build()``.
        """

        def decode(self, syndrome: list[int]) -> decoders.BpResult: ...
        def __repr__(self) -> str: ...

    class BpLsdBuilder:
        """Builder for BP+LSD decoder.

        Belief Propagation with Localized Statistics Decoding.

        Args:
            pcm: Sparse parity check matrix.
            error_rate: Channel error probability.

        Example:
            >>> from pecos_rslib.decoders import BpLsdBuilder, SparseMatrix
            >>> H = SparseMatrix([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]])
            >>> decoder = BpLsdBuilder(H, error_rate=0.01).lsd_order(2).build()
            >>> result = decoder.decode([0, 0, 0])
        """

        def __init__(self, pcm: decoders.SparseMatrix, error_rate: float) -> None: ...
        def max_iter(self, val: int) -> decoders.BpLsdBuilder:
            """Set maximum BP iterations (default: 100)."""
            ...

        def bp_method(self, val: str) -> decoders.BpLsdBuilder:
            """Set BP algorithm: "product_sum" or "minimum_sum" (default: "product_sum")."""
            ...

        def schedule(self, val: str) -> decoders.BpLsdBuilder:
            """Set update schedule: "parallel" or "serial" (default: "parallel")."""
            ...

        def lsd_order(self, val: int) -> decoders.BpLsdBuilder:
            """Set LSD order parameter (default: 0)."""
            ...

        def build(self) -> decoders.BpLsdDecoder:
            """Build the BP+LSD decoder."""
            ...

        def __repr__(self) -> str: ...

    class BpLsdDecoder:
        """BP+LSD decoder for LDPC codes.

        Created via ``BpLsdBuilder(...).build()``.
        """

        def decode(self, syndrome: list[int]) -> decoders.BpResult: ...
        def __repr__(self) -> str: ...

    class UnionFindBuilder:
        """Builder for Union-Find decoder.

        Cluster-based decoder using the Union-Find data structure.

        Args:
            pcm: Sparse parity check matrix.

        Example:
            >>> from pecos_rslib.decoders import UnionFindBuilder, SparseMatrix
            >>> H = SparseMatrix([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]])
            >>> decoder = UnionFindBuilder(H).method("peeling").build()
            >>> result = decoder.decode([0, 0, 0])
        """

        def __init__(self, pcm: decoders.SparseMatrix) -> None: ...
        def method(self, val: str) -> decoders.UnionFindBuilder:
            """Set decoding method: "inversion" or "peeling" (default: "inversion")."""
            ...

        def build(self) -> decoders.UnionFindDecoder:
            """Build the Union-Find decoder."""
            ...

        def __repr__(self) -> str: ...

    class UnionFindDecoder:
        """Union-Find decoder for LDPC codes.

        Created via ``UnionFindBuilder(...).build()``.
        """

        def decode(
            self,
            syndrome: list[int],
            llrs: list[float] | None = ...,
            bits_per_step: int = ...,
        ) -> decoders.BpResult: ...
        def __repr__(self) -> str: ...

    class TesseractResult:
        """Result from Tesseract decoder."""

        @property
        def correction(self) -> list[int]: ...
        @property
        def weight(self) -> float: ...
        def __repr__(self) -> str: ...

    class TesseractDecoder:
        """Tesseract decoder."""

        def __init__(self, dem_string: str) -> None: ...
        def decode(self, syndrome: list[int]) -> decoders.TesseractResult: ...
        def __repr__(self) -> str: ...

    class RelayBpBuilder:
        """Builder for Relay BP ensemble decoder.

        Configures and constructs a RelayBpDecoder for qLDPC codes. Uses an
        ensemble of min-sum BP decoders with randomized damping parameters
        (relay strategy) to improve convergence on codes where standard BP fails.

        Args:
            check_matrix: Parity check matrix as list of lists.
            error_priors: Prior error probabilities for each bit.

        Example:
            >>> from pecos_rslib.decoders import RelayBpBuilder
            >>> H = [[1, 1, 0], [0, 1, 1]]
            >>> decoder = RelayBpBuilder(H, [0.003, 0.003, 0.003]).seed(42).build()
            >>> result = decoder.decode([1, 0])
            >>> result.converged
            True
        """

        def __init__(self, check_matrix: list[list[int]], error_priors: list[float]) -> None: ...
        def max_iter(self, val: int) -> decoders.RelayBpBuilder:
            """Set maximum BP iterations (default: 200)."""
            ...

        def alpha(self, val: float | None) -> decoders.RelayBpBuilder:
            """Set min-sum scaling factor (None = no scaling)."""
            ...

        def gamma0(self, val: float | None) -> decoders.RelayBpBuilder:
            """Set initial damping factor (None = disabled)."""
            ...

        def pre_iter(self, val: int) -> decoders.RelayBpBuilder:
            """Set number of pre-relay BP iterations (default: 80)."""
            ...

        def num_sets(self, val: int) -> decoders.RelayBpBuilder:
            """Set number of relay sets/legs (default: 300)."""
            ...

        def set_max_iter(self, val: int) -> decoders.RelayBpBuilder:
            """Set max iterations per relay set (default: 60)."""
            ...

        def seed(self, val: int) -> decoders.RelayBpBuilder:
            """Set random seed for relay parameter sampling (default: 0)."""
            ...

        def stopping(self, val: str) -> decoders.RelayBpBuilder:
            """Set stopping criterion (default: "n_conv_1")."""
            ...

        def build(self) -> decoders.RelayBpDecoder:
            """Build the Relay BP decoder."""
            ...

        def __repr__(self) -> str: ...

    class RelayBpDecoder:
        """Relay BP ensemble decoder for qLDPC codes.

        Created via ``RelayBpBuilder(...).build()``.
        """

        def decode(self, syndrome: list[int]) -> decoders.BpResult:
            """Decode a syndrome vector.

            Args:
                syndrome: Syndrome vector (length = number of checks).

            Returns:
                BpResult with decoding, convergence status, and iteration count.
            """
            ...

        @property
        def check_count(self) -> int:
            """Number of checks (rows in check matrix)."""
            ...

        @property
        def bit_count(self) -> int:
            """Number of bits (columns in check matrix)."""
            ...

        def __repr__(self) -> str: ...

    class MinSumBpBuilder:
        """Builder for min-sum BP decoder.

        Configures and constructs a MinSumBpDecoder for qLDPC codes. Standard
        min-sum belief propagation -- simpler and faster than RelayBpDecoder
        for codes where plain BP converges.

        Args:
            check_matrix: Parity check matrix as list of lists.
            error_priors: Prior error probabilities for each bit.

        Example:
            >>> from pecos_rslib.decoders import MinSumBpBuilder
            >>> H = [[1, 1, 0], [0, 1, 1]]
            >>> decoder = MinSumBpBuilder(H, [0.003, 0.003, 0.003]).max_iter(100).build()
            >>> result = decoder.decode([1, 0])
            >>> result.converged
            True
        """

        def __init__(self, check_matrix: list[list[int]], error_priors: list[float]) -> None: ...
        def max_iter(self, val: int) -> decoders.MinSumBpBuilder:
            """Set maximum BP iterations (default: 200)."""
            ...

        def alpha(self, val: float | None) -> decoders.MinSumBpBuilder:
            """Set min-sum scaling factor (None = no scaling)."""
            ...

        def gamma0(self, val: float | None) -> decoders.MinSumBpBuilder:
            """Set initial damping factor (None = disabled)."""
            ...

        def build(self) -> decoders.MinSumBpDecoder:
            """Build the min-sum BP decoder."""
            ...

        def __repr__(self) -> str: ...

    class MinSumBpDecoder:
        """Min-sum BP decoder for qLDPC codes.

        Created via ``MinSumBpBuilder(...).build()``.
        """

        def decode(self, syndrome: list[int]) -> decoders.BpResult:
            """Decode a syndrome vector.

            Args:
                syndrome: Syndrome vector (length = number of checks).

            Returns:
                BpResult with decoding, convergence status, and iteration count.
            """
            ...

        @property
        def check_count(self) -> int:
            """Number of checks (rows in check matrix)."""
            ...

        @property
        def bit_count(self) -> int:
            """Number of bits (columns in check matrix)."""
            ...

        def __repr__(self) -> str: ...

# =============================================================================
# Utilities
# =============================================================================
def adjust_tableau_string(tableau: str) -> str:
    """Adjust tableau string format."""
    ...

# =============================================================================
# Version
# =============================================================================
__version__: str
