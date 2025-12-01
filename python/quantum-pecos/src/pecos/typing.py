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

"""Common type definitions used throughout PECOS.

This module provides:
- Numeric type aliases (Integer, Float, Complex, etc.) for type hints
- Runtime type tuples (INTEGER_TYPES, FLOAT_TYPES, etc.) for isinstance checks
- JSON-like types for gate parameters
- Protocol definitions for PECOS interfaces
- Generic Array type for dtype-parameterized arrays
- PhirModel re-export for PHIR program handling
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Generic, Protocol, TypeAlias, TypedDict, TypeVar

import _pecos_rslib as prs

# Import external PHIR model with consistent naming
from phir.model import PHIRModel as PhirModel

# Type variable for dtype (used with Array[DType])
DType = TypeVar("DType")

# =============================================================================
# Numeric Type Aliases
# =============================================================================
# These are analogous to NumPy's typing module (numpy.integer, numpy.floating, etc.)
#
# Type Hierarchy:
#     Numeric
#     ├── Integer
#     │   ├── SignedInteger (i8, i16, i32, i64)
#     │   └── UnsignedInteger (u8, u16, u32, u64)
#     └── Float (f32, f64)
#
#     Inexact
#     ├── Float (f32, f64)
#     └── Complex (complex64, complex128)

# Runtime Type Tuples (for isinstance checks)
# These are tuples of actual types that can be used with isinstance()

#: Tuple of all signed integer scalar types
SIGNED_INTEGER_TYPES: tuple[type, ...] = (prs.i8, prs.i16, prs.i32, prs.i64)

#: Tuple of all unsigned integer scalar types
UNSIGNED_INTEGER_TYPES: tuple[type, ...] = (prs.u8, prs.u16, prs.u32, prs.u64)

#: Tuple of all integer scalar types (signed and unsigned)
INTEGER_TYPES: tuple[type, ...] = SIGNED_INTEGER_TYPES + UNSIGNED_INTEGER_TYPES

#: Tuple of all floating-point scalar types
FLOAT_TYPES: tuple[type, ...] = (prs.f32, prs.f64)

#: Tuple of all complex scalar types
COMPLEX_TYPES: tuple[type, ...] = (prs.complex64, prs.complex128)

#: Tuple of all numeric scalar types (integer and float, excludes complex)
NUMERIC_TYPES: tuple[type, ...] = INTEGER_TYPES + FLOAT_TYPES

#: Tuple of all inexact scalar types (float and complex)
INEXACT_TYPES: tuple[type, ...] = FLOAT_TYPES + COMPLEX_TYPES

# Type Aliases (for static type checking)
# These work with static type checkers like mypy and pyright.

if TYPE_CHECKING:
    #: Type alias for signed integer scalar types (i8, i16, i32, i64)
    SignedInteger: TypeAlias = type[prs.i8 | prs.i16 | prs.i32 | prs.i64]

    #: Type alias for unsigned integer scalar types (u8, u16, u32, u64)
    UnsignedInteger: TypeAlias = type[prs.u8 | prs.u16 | prs.u32 | prs.u64]

    #: Type alias for all integer scalar types
    Integer: TypeAlias = SignedInteger | UnsignedInteger

    #: Type alias for floating-point scalar types (f32, f64)
    Float: TypeAlias = type[prs.f32 | prs.f64]

    #: Type alias for complex scalar types (complex64, complex128)
    Complex: TypeAlias = type[prs.complex64 | prs.complex128]

    #: Type alias for numeric types (integer or float)
    Numeric: TypeAlias = Integer | Float

    #: Type alias for inexact types (float or complex)
    Inexact: TypeAlias = Float | Complex

else:
    # At runtime, these are the type tuples themselves
    # This allows isinstance(x, Integer) to work at runtime
    SignedInteger = SIGNED_INTEGER_TYPES
    UnsignedInteger = UNSIGNED_INTEGER_TYPES
    Integer = INTEGER_TYPES
    Float = FLOAT_TYPES
    Complex = COMPLEX_TYPES
    Numeric = NUMERIC_TYPES
    Inexact = INEXACT_TYPES

# JSON-like types for gate parameters and metadata
JSONValue = str | int | float | bool | dict[str, "JSONValue"] | list["JSONValue"] | None
JSONDict = dict[str, JSONValue]

# Gate parameter type - used for **params in various gate operations
GateParams = JSONDict

# Simulator gate parameters - these are passed to simulator gate functions
SimulatorGateParams = JSONDict

# Simulator initialization parameters
SimulatorInitParams = (
    JSONDict  # Parameters for simulator initialization (e.g., MPS config)
)

# QECC parameter types
QECCParams = JSONDict  # Parameters for QECC initialization
QECCGateParams = JSONDict  # Parameters for QECC gate operations
QECCInstrParams = JSONDict  # Parameters for QECC instruction operations


# Error model parameter types
class ErrorParams(TypedDict, total=False):
    """Type definition for error parameters."""

    p: float
    p1: float
    p2: float
    p2_mem: float | None
    p_meas: float | tuple[float, ...]
    p_init: float
    scale: float
    noiseless_qubits: set[int]


# Threshold calculation types
class ThresholdResult(TypedDict):
    """Type definition for threshold calculation results."""

    distance: int | list[int]
    error_rates: list[float]
    logical_rates: list[float]
    time_rates: list[float] | None


# Fault tolerance checking types
class SpacetimeLocation(TypedDict):
    """Type definition for spacetime location in fault tolerance checking."""

    tick: int
    location: tuple[int, ...]
    before: bool
    symbol: str
    metadata: dict[str, int | str | bool]


class FaultDict(TypedDict, total=False):
    """Type definition for fault dictionary."""

    faults: list[tuple[int, ...]]
    locations: list[tuple[int, ...]]
    symbols: list[str]


# Stabilizer verification types
class StabilizerCheckDict(TypedDict, total=False):
    """Type definition for stabilizer check dictionary."""

    X: set[int]
    Y: set[int]
    Z: set[int]


class StabilizerVerificationResult(TypedDict):
    """Type definition for stabilizer verification results."""

    stabilizers: list[StabilizerCheckDict]
    destabilizers: list[StabilizerCheckDict]
    logicals_x: list[StabilizerCheckDict]
    logicals_z: list[StabilizerCheckDict]
    distance: int | None


# Circuit execution output types
class OutputDict(TypedDict, total=False):
    """Type definition for output dictionary used in circuit execution."""

    # Common keys based on codebase usage
    syndrome: set[int]
    measurements: dict[str, int | list[int]]
    classical_registers: dict[str, int]


# Logical operator types
LogicalOperator = dict[
    str,
    set[int],
]  # Maps Pauli operator ('X', 'Y', 'Z') to qubit indices

# Gate location types
Location = int | tuple[int, ...]  # Single qubit or multi-qubit gate location
LocationSet = (
    set[Location] | list[Location] | tuple[Location, ...]
)  # Collection of locations


class LogicalOpInfo(TypedDict):
    """Information about a logical operator."""

    X: set[int]
    Z: set[int]
    equiv_ops: tuple[str, ...]


# Graph protocol types
# Node identifiers can be any hashable type (str, int, tuple, etc.)
Node = object
# Edges are represented as tuples of two nodes
Edge = tuple[Node, Node]
# Paths are lists of nodes
Path = list[Node]


class GraphProtocol(Protocol):
    """Protocol for graph objects used in decoder precomputation and algorithms.

    This protocol defines the interface that graph implementations must provide
    to be compatible with PECOS decoders and graph algorithms.
    """

    def nodes(self) -> list[Node]:
        """Return list of nodes in the graph.

        Returns:
            List of node identifiers in the graph.
        """
        ...

    def add_edge(
        self,
        a: Node,
        b: Node,
        weight: float | None = None,
        **kwargs: object,
    ) -> None:
        """Add an edge between nodes a and b.

        Args:
            a: First node identifier.
            b: Second node identifier.
            weight: Optional edge weight.
            **kwargs: Additional edge attributes.
        """
        ...

    def single_source_shortest_path(self, source: Node) -> dict[Node, Path]:
        """Compute shortest paths from source to all other nodes.

        Args:
            source: Source node identifier.

        Returns:
            Dictionary mapping target nodes to paths (list of nodes from source to target).
        """
        ...


# =============================================================================
# Generic Array Type
# =============================================================================


class Array(Generic[DType]):
    """Generic type for Array with dtype parameter support.

    This is a typing stub that enables generic type annotations for Array.
    At runtime, use the actual Array from _pecos_rslib.

    Type Parameters:
        DType: The dtype of the array (from _pecos_rslib.dtypes)

    Examples:
        >>> from pecos.typing import Array
        >>> from _pecos_rslib import dtypes
        >>>
        >>> def get_state_vector() -> Array[dtypes.complex128]:
        ...     return array([1 + 0j, 0 + 0j], dtype=dtypes.complex128)
        ...
        >>> def multiply_floats(
        ...     a: Array[dtypes.f64], b: Array[dtypes.f64]
        ... ) -> Array[dtypes.f64]:
        ...     return a * b

    Note:
        This is a type hint only. At runtime, import Array from _pecos_rslib:
        >>> from _pecos_rslib import Array  # Runtime usage
        >>> from pecos.typing import Array  # Type hints only
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
    "COMPLEX_TYPES",
    "FLOAT_TYPES",
    "INEXACT_TYPES",
    "INTEGER_TYPES",
    "NUMERIC_TYPES",
    "SIGNED_INTEGER_TYPES",
    "UNSIGNED_INTEGER_TYPES",
    "Array",
    "Complex",
    "DType",
    "Edge",
    "ErrorParams",
    "FaultDict",
    "Float",
    "GateParams",
    "GraphProtocol",
    "Inexact",
    "Integer",
    "JSONDict",
    "JSONValue",
    "Location",
    "LocationSet",
    "LogicalOpInfo",
    "LogicalOperator",
    "Node",
    "Numeric",
    "OutputDict",
    "Path",
    "PhirModel",
    "QECCGateParams",
    "QECCInstrParams",
    "QECCParams",
    "SignedInteger",
    "SimulatorGateParams",
    "SimulatorInitParams",
    "SpacetimeLocation",
    "StabilizerCheckDict",
    "StabilizerVerificationResult",
    "ThresholdResult",
    "UnsignedInteger",
]
