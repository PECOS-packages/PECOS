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
    def __neg__(self) -> Scalar: ...
    def __pos__(self) -> Scalar: ...
    def __abs__(self) -> Scalar: ...

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
    """Fixed-width integer type with explicit bit width.

    A Rust-backed binary integer type for efficient fixed-width arithmetic.
    Supports both signed and unsigned operations on fixed-width integers.

    Examples:
        >>> b = BitInt(8, 5)    # 8-bit integer with value 5
        >>> b = BitInt("1010")  # 4-bit integer from binary string (value 10)
        >>> b = BitInt(8)       # 8-bit integer with value 0
    """

    @property
    def size(self) -> int:
        """Number of bits in this integer."""
        ...

    @property
    def dtype(self) -> type:
        """Data type (default: i64)."""
        ...

    @overload
    def __init__(
        self,
        binary_str: str,
        value: int = 0,
        signed: bool | None = None,
        dtype: type | None = None,
    ) -> None:
        """Create from binary string (e.g., '1010').

        When created from binary string, defaults to unsigned unless
        signed=True or dtype=pc.i64 etc. is specified.
        """
        ...

    @overload
    def __init__(
        self,
        size: int,
        value: int = 0,
        signed: bool | None = None,
        dtype: type | None = None,
    ) -> None:
        """Create from size and value.

        When created from size and value, defaults to signed
        unless signed=False or dtype=pc.u64 etc. is specified.
        """
        ...

    def __init__(
        self,
        size: str | int,
        value: int = 0,
        signed: bool | None = None,
        dtype: type | None = None,
    ) -> None: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    def __int__(self) -> int: ...
    def __len__(self) -> int: ...
    def __hash__(self) -> int: ...
    def to_binary_str(
        self, reverse_bits: bool = False, separator: str | None = None
    ) -> str:
        """Get binary string with configurable bit ordering.

        Args:
            reverse_bits: If True, reverse bit order (LSB on left instead of right).
                          If False (default), use standard notation (MSB on left).
            separator: Optional separator between bits (e.g., " " or "_").

        Returns:
            Binary string representation.

        Examples:
            >>> b = BitInt("1010")  # value 10
            >>> b.to_binary_str()  # Standard: MSB first
            "1010"
            >>> b.to_binary_str(reverse_bits=True)  # Reversed: LSB first
            "0101"
            >>> b.to_binary_str(separator=" ")
            "1 0 1 0"
        """
        ...
    # Indexing
    def __getitem__(self, index: int) -> int:
        """Get bit at index (0 = LSB)."""
        ...

    def __setitem__(self, index: int, value: int) -> None:
        """Set bit at index (0 = LSB)."""
        ...
    # Comparison operators
    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __lt__(self, other: BitInt | int | str) -> bool: ...
    def __le__(self, other: BitInt | int | str) -> bool: ...
    def __gt__(self, other: BitInt | int | str) -> bool: ...
    def __ge__(self, other: BitInt | int | str) -> bool: ...

    # Bitwise operators
    def __and__(self, other: BitInt | int | str) -> BitInt: ...
    def __rand__(self, other: BitInt | int | str) -> BitInt: ...
    def __or__(self, other: BitInt | int | str) -> BitInt: ...
    def __ror__(self, other: BitInt | int | str) -> BitInt: ...
    def __xor__(self, other: BitInt | int | str) -> BitInt: ...
    def __rxor__(self, other: BitInt | int | str) -> BitInt: ...
    def __invert__(self) -> BitInt: ...
    def __lshift__(self, other: BitInt | int) -> BitInt: ...
    def __rlshift__(self, other: int) -> BitInt: ...
    def __rshift__(self, other: BitInt | int) -> BitInt: ...
    def __rrshift__(self, other: int) -> BitInt: ...

    # Arithmetic operators
    def __add__(self, other: BitInt | int | str) -> BitInt: ...
    def __radd__(self, other: BitInt | int | str) -> BitInt: ...
    def __sub__(self, other: BitInt | int | str) -> BitInt: ...
    def __rsub__(self, other: BitInt | int | str) -> BitInt: ...
    def __mul__(self, other: BitInt | int | str) -> BitInt: ...
    def __rmul__(self, other: BitInt | int | str) -> BitInt: ...
    def __floordiv__(self, other: BitInt | int | str) -> BitInt: ...
    def __rfloordiv__(self, other: BitInt | int | str) -> BitInt: ...
    def __mod__(self, other: BitInt | int | str) -> BitInt: ...
    def __rmod__(self, other: BitInt | int | str) -> BitInt: ...
    def __neg__(self) -> BitInt: ...

    # BinArray-compatible methods
    def set(self, other: BitInt | int | str) -> None:
        """Set value from another BitInt, int, or binary string."""
        ...

    def set_clip(self, other: BitInt | int | str) -> None:
        """Set value, clipping to size (BinArray compatibility)."""
        ...

    def clamp(self, other: BitInt | int | str) -> None:
        """Alias for set_clip (BinArray compatibility)."""
        ...

    def num_bits(self) -> int:
        """Return number of bits (alias for size property)."""
        ...

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
        data: (
            Sequence[int | float | complex] | Sequence[Sequence[int | float | complex]]
        ),
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
    def __add__(
        self, other: Array[Scalar] | Scalar | int | float | complex
    ) -> Array[_ScalarT]: ...
    def __radd__(
        self, other: Array[Scalar] | Scalar | int | float | complex
    ) -> Array[_ScalarT]: ...
    def __sub__(
        self, other: Array[Scalar] | Scalar | int | float | complex
    ) -> Array[_ScalarT]: ...
    def __rsub__(
        self, other: Array[Scalar] | Scalar | int | float | complex
    ) -> Array[_ScalarT]: ...
    def __mul__(
        self, other: Array[Scalar] | Scalar | int | float | complex
    ) -> Array[_ScalarT]: ...
    def __rmul__(
        self, other: Array[Scalar] | Scalar | int | float | complex
    ) -> Array[_ScalarT]: ...
    def __truediv__(
        self, other: Array[Scalar] | Scalar | int | float | complex
    ) -> Array[ScalarF64]: ...
    def __rtruediv__(
        self, other: Array[Scalar] | Scalar | int | float | complex
    ) -> Array[ScalarF64]: ...
    def __floordiv__(
        self, other: Array[Scalar] | Scalar | int | float
    ) -> Array[_ScalarT]: ...
    def __rfloordiv__(
        self, other: Array[Scalar] | Scalar | int | float
    ) -> Array[_ScalarT]: ...
    def __mod__(
        self, other: Array[Scalar] | Scalar | int | float
    ) -> Array[_ScalarT]: ...
    def __rmod__(
        self, other: Array[Scalar] | Scalar | int | float
    ) -> Array[_ScalarT]: ...
    def __pow__(
        self, other: Array[Scalar] | Scalar | int | float
    ) -> Array[ScalarF64]: ...
    def __rpow__(
        self, other: Array[Scalar] | Scalar | int | float
    ) -> Array[ScalarF64]: ...
    def __neg__(self) -> Array[_ScalarT]: ...
    def __pos__(self) -> Array[_ScalarT]: ...
    def __abs__(self) -> Array[_ScalarT]: ...

    # Comparison
    def __eq__(self, other: object) -> Array[ScalarU8]: ...  # type: ignore[override]
    def __ne__(self, other: object) -> Array[ScalarU8]: ...  # type: ignore[override]
    def __lt__(
        self, other: Array[Scalar] | Scalar | int | float
    ) -> Array[ScalarU8]: ...
    def __le__(
        self, other: Array[Scalar] | Scalar | int | float
    ) -> Array[ScalarU8]: ...
    def __gt__(
        self, other: Array[Scalar] | Scalar | int | float
    ) -> Array[ScalarU8]: ...
    def __ge__(
        self, other: Array[Scalar] | Scalar | int | float
    ) -> Array[ScalarU8]: ...

    # Methods
    def reshape(self, *shape: int) -> Array[_ScalarT]: ...
    def flatten(self) -> Array[_ScalarT]: ...
    def ravel(self) -> Array[_ScalarT]: ...
    def transpose(self, *axes: int) -> Array[_ScalarT]: ...
    def sum(self, axis: int | None = None) -> Scalar | Array[_ScalarT]: ...
    def mean(self, axis: int | None = None) -> ScalarF64 | Array[ScalarF64]: ...
    def std(
        self, axis: int | None = None, ddof: int = 0
    ) -> ScalarF64 | Array[ScalarF64]: ...
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
def zeros(
    shape: int | tuple[int, ...], dtype: type[Scalar] | DType | None = None
) -> Array[Scalar]: ...
def ones(
    shape: int | tuple[int, ...], dtype: type[Scalar] | DType | None = None
) -> Array[Scalar]: ...
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
def delete(
    arr: Array[Scalar], indices: int | Sequence[int], axis: int | None = None
) -> Array[Scalar]: ...

# =============================================================================
# Mathematical Functions
# =============================================================================
def mean(a: Array[Scalar], axis: int | None = None) -> ScalarF64 | Array[ScalarF64]: ...
def std(
    a: Array[Scalar], axis: int | None = None, ddof: int = 0
) -> ScalarF64 | Array[ScalarF64]: ...
def sum(
    a: Array[Scalar], axis: int | None = None
) -> Scalar | Array[Scalar]: ...  # noqa: A001
def max(
    a: Array[Scalar], axis: int | None = None
) -> Scalar | Array[Scalar]: ...  # noqa: A001
def min(
    a: Array[Scalar], axis: int | None = None
) -> Scalar | Array[Scalar]: ...  # noqa: A001
def power(
    x: Array[Scalar] | Scalar | float, y: Array[Scalar] | Scalar | float
) -> Array[ScalarF64] | ScalarF64: ...
def sqrt(x: Array[Scalar] | Scalar | float) -> Array[ScalarF64] | ScalarF64: ...
def exp(x: Array[Scalar] | Scalar | float) -> Array[ScalarF64] | ScalarF64: ...
def ln(x: Array[Scalar] | Scalar | float) -> Array[ScalarF64] | ScalarF64: ...
def log(
    x: Array[Scalar] | Scalar | float, base: float | None = None
) -> Array[ScalarF64] | ScalarF64: ...
def abs(x: Array[Scalar] | Scalar | float) -> Array[Scalar] | Scalar: ...  # noqa: A001
def floor(x: Array[Scalar] | Scalar | float) -> Array[ScalarF64] | ScalarF64: ...
def ceil(x: Array[Scalar] | Scalar | float) -> Array[ScalarF64] | ScalarF64: ...
def round(
    x: Array[Scalar] | Scalar | float, decimals: int = 0
) -> Array[ScalarF64] | ScalarF64: ...  # noqa: A001

# Trigonometric functions
def cos(x: Array[Scalar] | Scalar | float) -> Array[ScalarF64] | ScalarF64: ...
def sin(x: Array[Scalar] | Scalar | float) -> Array[ScalarF64] | ScalarF64: ...
def tan(x: Array[Scalar] | Scalar | float) -> Array[ScalarF64] | ScalarF64: ...
def acos(x: Array[Scalar] | Scalar | float) -> Array[ScalarF64] | ScalarF64: ...
def asin(x: Array[Scalar] | Scalar | float) -> Array[ScalarF64] | ScalarF64: ...
def atan(x: Array[Scalar] | Scalar | float) -> Array[ScalarF64] | ScalarF64: ...
def atan2(
    y: Array[Scalar] | Scalar | float, x: Array[Scalar] | Scalar | float
) -> Array[ScalarF64] | ScalarF64: ...

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
def all(
    a: Array[Scalar], axis: int | None = None
) -> bool | Array[ScalarU8]: ...  # noqa: A001
def any(
    a: Array[Scalar], axis: int | None = None
) -> bool | Array[ScalarU8]: ...  # noqa: A001
def where(
    condition: Array[Scalar],
    x: Array[Scalar] | Scalar | float,
    y: Array[Scalar] | Scalar | float,
) -> Array[Scalar]: ...

# Constants
inf: float
nan: float

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
        def pdf(
            x: float | Array[Scalar], loc: float = 0.0, scale: float = 1.0
        ) -> float | Array[ScalarF64]: ...
        @staticmethod
        def cdf(
            x: float | Array[Scalar], loc: float = 0.0, scale: float = 1.0
        ) -> float | Array[ScalarF64]: ...
        @staticmethod
        def ppf(
            q: float | Array[Scalar], loc: float = 0.0, scale: float = 1.0
        ) -> float | Array[ScalarF64]: ...
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
class SparseSim:
    """Sparse stabilizer simulator."""

    def __init__(self, num_qubits: int) -> None: ...
    @property
    def num_qubits(self) -> int: ...
    def __repr__(self) -> str: ...
    # Gate methods would go here

class SparseSimCpp:
    """C++ sparse simulator bindings."""

    def __init__(self, num_qubits: int) -> None: ...
    @property
    def num_qubits(self) -> int: ...

class StateVec:
    """Rust state vector simulator."""

    def __init__(self, num_qubits: int) -> None: ...
    @property
    def num_qubits(self) -> int: ...

class Qulacs:
    """Rust Qulacs state vector simulator."""

    def __init__(self, num_qubits: int, *, seed: int | None = None) -> None: ...
    @property
    def num_qubits(self) -> int: ...

class CoinToss:
    """Coin toss simulator for random measurement outcomes."""

    def __init__(
        self, num_qubits: int, prob: float = 0.5, seed: int | None = None
    ) -> None: ...
    @property
    def num_qubits(self) -> int: ...

class QuestStateVec:
    """QuEST state vector simulator."""

    def __init__(self, num_qubits: int) -> None: ...
    @property
    def num_qubits(self) -> int: ...

class QuestDensityMatrix:
    """QuEST density matrix simulator."""

    def __init__(self, num_qubits: int) -> None: ...
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

class SparseStabilizerEngineBuilder:
    """Builder for sparse stabilizer engines."""

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

class PauliPropRs:
    """Pauli propagator (Rust implementation)."""

    ...

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
    sparse_stabilizer: Callable[..., SparseStabilizerEngineBuilder]
    sparse_stab: Callable[..., SparseStabilizerEngineBuilder]
    StateVectorEngineBuilder: type[StateVectorEngineBuilder]
    SparseStabilizerEngineBuilder: type[SparseStabilizerEngineBuilder]

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

def sparse_stabilizer(**kwargs: object) -> SparseStabilizerEngineBuilder:
    """Create a sparse stabilizer engine builder."""
    ...

def sparse_stab(**kwargs: object) -> SparseStabilizerEngineBuilder:
    """Create a sparse stabilizer engine builder (alias)."""
    ...

def general_noise(**kwargs: object) -> GeneralNoiseModelBuilder:
    """Create a general noise model builder."""
    ...

def depolarizing_noise(p: float) -> DepolarizingNoiseModelBuilder:
    """Create a depolarizing noise model builder."""
    ...

def biased_depolarizing_noise(
    px: float, py: float, pz: float
) -> BiasedDepolarizingNoiseModelBuilder:
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
def compile_hugr_to_llvm(hugr_bytes: bytes, output_path: str | None = None) -> str:
    """Compile HUGR bytes to LLVM IR."""
    ...

def compile_hugr_to_llvm_rust(hugr_bytes: bytes, output_path: str | None = None) -> str:
    """Compile HUGR bytes to LLVM IR (Rust backend)."""
    ...

def check_rust_hugr_availability() -> tuple[bool, str]:
    """Check if Rust HUGR backend is available."""
    ...

def get_compilation_backends() -> dict[str, object]:
    """Get information about available compilation backends."""
    ...

RUST_HUGR_AVAILABLE: bool
HUGR_LLVM_PIPELINE_AVAILABLE: bool

# =============================================================================
# WASM
# =============================================================================
class WasmForeignObject:
    """WASM foreign object wrapper."""

    ...

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
