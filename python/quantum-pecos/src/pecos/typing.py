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

"""Common type definitions used throughout PECOS.

This module provides:
- Numeric type aliases (Integer, Float, Complex, etc.) for type hints
- Runtime type tuples (INTEGER_TYPES, FLOAT_TYPES, etc.) for isinstance checks
- JSON-like types for gate parameters
- Protocol definitions for PECOS interfaces
- Generic Array type for dtype-parameterized arrays
- Compiled program types (CompiledHugr, CompiledQasm, etc.) for type annotations
- PhirModel re-export for PHIR program handling
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Generic, Literal, Protocol, TypeAlias, TypedDict, TypeVar

import pecos_rslib as prs
from phir.model import (
    Barrier as _Barrier,
)
from phir.model import (
    Comment as _Comment,
)
from phir.model import (
    COp as _COp,
)
from phir.model import (
    CVarDefine as _CVarDefine,
)
from phir.model import (
    ExportVar as _ExportVar,
)
from phir.model import (
    FFCall as _FFCall,
)
from phir.model import (
    IfBlock as _IfBlock,
)
from phir.model import (
    MOpType as _MOpType,
)
from phir.model import (
    Op as _Op,
)
from phir.model import (
    PHIRModel as _PHIRModel,
)
from phir.model import (
    QOp as _QOp,
)
from phir.model import (
    QParBlock as _QParBlock,
)
from phir.model import (
    QVarDefine as _QVarDefine,
)
from phir.model import (
    SeqBlock as _SeqBlock,
)
from pydantic import model_validator


class _PecosCVarDefine(_CVarDefine):
    """CVarDefine extended with 8-bit and 16-bit integer data types.

    The upstream PHIR spec's ``CVarDefine`` only accepts ``i32``, ``i64``,
    ``u32``, ``u64``. PECOS additionally supports ``i8``, ``u8``, ``i16``,
    ``u16`` for classical registers. The parent class's ``check_size``
    validator covers 32 and 64-bit cases; this subclass adds the matching
    check for 8-bit and 16-bit sizes.
    """

    data_type: Literal["i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64"]  # type: ignore[assignment]

    @model_validator(mode="after")
    def _check_size_small(self) -> _PecosCVarDefine:
        """Check that ``size`` fits within 8-bit and 16-bit data types."""
        msg = "`size` is greater than what `data_type` can handle"
        if self.size:
            match self.data_type:
                case "i8" | "u8":
                    if self.size > 8:
                        raise ValueError(msg)
                case "i16" | "u16":
                    if self.size > 16:
                        raise ValueError(msg)
        return self


_PecosDataMgmt: TypeAlias = _PecosCVarDefine | _QVarDefine | _ExportVar


class ResultCOp(_Op):
    """PECOS-specific ``Result`` classical operation.

    Copies the value of internal classical registers to external result
    variables, creating the destination variable if needed. Used by the
    PECOS ``HybridEngine`` to transmit measurement bits between the inner
    and outer classical interpreters.

    Example:
        ``{"cop": "Result", "args": ["m"], "returns": ["c"]}``

    This operation is a PECOS extension, not part of the upstream PHIR
    specification.
    """

    cop: Literal["Result"]
    args: list[str]
    returns: list[str]


_PecosOpType: TypeAlias = _FFCall | _COp | ResultCOp | _QOp | _MOpType | _Barrier


class _PecosSeqBlock(_SeqBlock):
    """SeqBlock extended with PECOS-specific classical operations."""

    ops: list[_PecosOpType | _PecosBlockType]  # type: ignore[assignment]


class _PecosIfBlock(_IfBlock):
    """IfBlock extended with PECOS-specific classical operations."""

    true_branch: list[_PecosOpType | _PecosBlockType]  # type: ignore[assignment]
    false_branch: list[_PecosOpType | _PecosBlockType] | None = None  # type: ignore[assignment]


_PecosBlockType: TypeAlias = _PecosSeqBlock | _QParBlock | _PecosIfBlock
_PecosCmd: TypeAlias = _PecosDataMgmt | _PecosOpType | _PecosBlockType | _Comment


class PhirModel(_PHIRModel):
    """PHIR model extended with PECOS-specific classical operations.

    Adds support for the ``Result`` cop used by PECOS ``HybridEngine`` to
    map internal measurement registers to external result variables. Fully
    backwards-compatible with upstream PHIR programs.

    The upstream ``phir.model.PHIRModel`` rejects programs containing
    ``Result`` cops because ``Result`` is not in the PHIR specification.
    Use this class (or ``pecos.typing.PhirModel``) when validating
    programs that may contain PECOS extensions.
    """

    ops: list[_PecosCmd]  # type: ignore[assignment]


_PecosSeqBlock.model_rebuild()
_PecosIfBlock.model_rebuild()
PhirModel.model_rebuild()

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
JSONValue: TypeAlias = str | int | float | bool | dict[str, "JSONValue"] | list["JSONValue"] | None
JSONDict: TypeAlias = dict[str, JSONValue]

#: Gate parameter type - used for **params in various gate operations
GateParams: TypeAlias = JSONDict

#: Simulator gate parameters - passed to simulator gate functions
SimulatorGateParams: TypeAlias = JSONDict

#: Simulator initialization parameters (e.g., MPS config)
SimulatorInitParams: TypeAlias = JSONDict

#: Parameters for QECC initialization
QECCParams: TypeAlias = JSONDict
#: Parameters for QECC gate operations
QECCGateParams: TypeAlias = JSONDict
#: Parameters for QECC instruction operations
QECCInstrParams: TypeAlias = JSONDict


# Error model parameter types
class ErrorParams(TypedDict, total=False):
    """Type definition for error parameters."""

    p: float
    p1: float
    p2: float
    p2_mem: float | None
    p_meas: float | tuple[float, ...]
    p_prep: float
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


#: Logical operator type - maps Pauli operator ('X', 'Y', 'Z') to qubit indices
LogicalOperator: TypeAlias = dict[str, set[int]]

#: Single qubit or multi-qubit gate location
Location: TypeAlias = int | tuple[int, ...]
#: Collection of gate locations
LocationSet: TypeAlias = set[Location] | list[Location] | tuple[Location, ...]


class LogicalOpInfo(TypedDict):
    """Information about a logical operator."""

    X: set[int]
    Z: set[int]
    equiv_ops: tuple[str, ...]


# Graph protocol types
#: Node identifier - can be any hashable type (str, int, tuple, etc.)
Node: TypeAlias = object
#: Edge represented as tuple of two nodes
Edge: TypeAlias = tuple[Node, Node]
#: Path represented as list of nodes
Path: TypeAlias = list[Node]


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
# Compiled Program Types (from pecos_rslib)
# =============================================================================
# These are the low-level Rust program types that the simulator accepts.
# Users typically use the Python wrapper classes (pecos.Qasm, pecos.Hugr, etc.)
# which internally convert to these types.

if TYPE_CHECKING:
    import pecos_rslib.programs as programs_rs

    #: Compiled HUGR program type from pecos_rslib
    CompiledHugr: TypeAlias = programs_rs.Hugr
    #: Compiled PHIR JSON program type from pecos_rslib
    CompiledPhirJson: TypeAlias = programs_rs.PhirJson
    #: Compiled QASM program type from pecos_rslib
    CompiledQasm: TypeAlias = programs_rs.Qasm
    #: Compiled QIS program type from pecos_rslib
    CompiledQis: TypeAlias = programs_rs.Qis
    #: Compiled WASM program type from pecos_rslib
    CompiledWasm: TypeAlias = programs_rs.Wasm
    #: Compiled WAT program type from pecos_rslib
    CompiledWat: TypeAlias = programs_rs.Wat

    #: Union type for any compiled program that can be passed to the simulator
    CompiledProgram: TypeAlias = (
        CompiledHugr | CompiledQasm | CompiledQis | CompiledPhirJson | CompiledWasm | CompiledWat
    )


# =============================================================================
# Generic Array Type
# =============================================================================


class Array(Generic[DType]):
    """Generic type for Array with dtype parameter support.

    This is a typing stub that enables generic type annotations for Array.
    At runtime, use the actual Array from pecos_rslib.

    Type Parameters:
        DType: The dtype of the array (from pecos_rslib.dtypes)

    Examples:
        >>> from pecos.typing import Array
        >>> from pecos_rslib import dtypes
        >>>
        >>> def get_state_vector() -> Array[dtypes.complex128]:
        ...     return array([1 + 0j, 0 + 0j], dtype=dtypes.complex128)
        ...
        >>> def multiply_floats(a: Array[dtypes.f64], b: Array[dtypes.f64]) -> Array[dtypes.f64]:
        ...     return a * b

    Note:
        This is a type hint only. At runtime, import Array from pecos_rslib:
        >>> from pecos_rslib import Array  # Runtime usage
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
    "CompiledHugr",
    "CompiledPhirJson",
    "CompiledProgram",
    "CompiledQasm",
    "CompiledQis",
    "CompiledWasm",
    "CompiledWat",
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
