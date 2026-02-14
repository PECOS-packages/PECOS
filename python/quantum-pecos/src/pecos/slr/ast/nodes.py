# Copyright 2026 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""AST node definitions for SLR programs.

This module defines the Abstract Syntax Tree (AST) nodes for representing
SLR quantum programs. The AST provides:

- Clean separation of syntax from semantics
- Uniform interface for analysis and transformation
- Visitor pattern support for traversal
- Integration with QAlloc for qubit slot management

All nodes are immutable frozen dataclasses for safety and hashability.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pecos.slr.ast.visitor import AstVisitor, T


# =============================================================================
# Source Location
# =============================================================================


@dataclass(frozen=True, kw_only=True)
class SourceLocation:
    """Source location for error reporting.

    Tracks where in the source a node originated for error messages.
    """

    line: int
    column: int
    file: str | None = None

    def __str__(self) -> str:
        if self.file:
            return f"{self.file}:{self.line}:{self.column}"
        return f"{self.line}:{self.column}"


# =============================================================================
# Base Node
# =============================================================================


@dataclass(frozen=True, kw_only=True)
class AstNode(ABC):
    """Base class for all AST nodes.

    All AST nodes are immutable frozen dataclasses that support:
    - Visitor pattern via accept()
    - Child traversal via children()
    - Optional source location tracking
    """

    location: SourceLocation | None = field(default=None, compare=False, repr=False)

    @abstractmethod
    def accept(self, visitor: AstVisitor[T]) -> T:
        """Accept a visitor for traversal."""
        ...

    def children(self) -> Sequence[AstNode]:
        """Return child nodes for traversal. Override in subclasses."""
        return ()


# =============================================================================
# Enums
# =============================================================================


class GateKind(Enum):
    """All supported quantum gate types."""

    # Single-qubit Paulis
    X = auto()
    Y = auto()
    Z = auto()

    # Hadamard
    H = auto()

    # Phase gates
    S = auto()
    Sdg = auto()
    T = auto()
    Tdg = auto()

    # Square root gates
    SX = auto()
    SY = auto()
    SZ = auto()
    SXdg = auto()
    SYdg = auto()
    SZdg = auto()

    # Rotation gates (parameterized)
    RX = auto()
    RY = auto()
    RZ = auto()

    # Two-qubit Clifford gates
    CX = auto()
    CY = auto()
    CZ = auto()
    CH = auto()

    # Two-qubit rotation gates
    SXX = auto()
    SYY = auto()
    SZZ = auto()
    SXXdg = auto()
    SYYdg = auto()
    SZZdg = auto()
    RZZ = auto()

    # Controlled rotation gates (parameterized)
    CRX = auto()
    CRY = auto()
    CRZ = auto()

    # Face rotations
    F = auto()
    Fdg = auto()
    F4 = auto()
    F4dg = auto()

    @property
    def arity(self) -> int:
        """Number of qubit arguments required."""
        two_qubit = {
            GateKind.CX,
            GateKind.CY,
            GateKind.CZ,
            GateKind.CH,
            GateKind.SXX,
            GateKind.SYY,
            GateKind.SZZ,
            GateKind.SXXdg,
            GateKind.SYYdg,
            GateKind.SZZdg,
            GateKind.RZZ,
            GateKind.CRX,
            GateKind.CRY,
            GateKind.CRZ,
        }
        return 2 if self in two_qubit else 1

    @property
    def is_parameterized(self) -> bool:
        """Whether this gate takes angle parameters."""
        return self in {
            GateKind.RX,
            GateKind.RY,
            GateKind.RZ,
            GateKind.RZZ,
            GateKind.CRX,
            GateKind.CRY,
            GateKind.CRZ,
        }


class BinaryOp(Enum):
    """Binary operators for expressions."""

    # Arithmetic
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()

    # Comparison
    EQ = auto()
    NE = auto()
    LT = auto()
    LE = auto()
    GT = auto()
    GE = auto()

    # Logical
    AND = auto()
    OR = auto()
    XOR = auto()

    # Bitwise
    LSHIFT = auto()
    RSHIFT = auto()


class UnaryOp(Enum):
    """Unary operators for expressions."""

    NOT = auto()
    NEG = auto()


# =============================================================================
# References
# =============================================================================


@dataclass(frozen=True, kw_only=True)
class SlotRef(AstNode):
    """Reference to a qubit slot in an allocator.

    This is the AST equivalent of QubitRef - it identifies a specific
    slot in a named allocator.
    """

    allocator: str
    index: int

    def accept(self, visitor: AstVisitor[T]) -> T:
        return visitor.visit_slot_ref(self)

    def __str__(self) -> str:
        return f"{self.allocator}[{self.index}]"


@dataclass(frozen=True, kw_only=True)
class BitRef(AstNode):
    """Reference to a classical bit in a register."""

    register: str
    index: int

    def accept(self, visitor: AstVisitor[T]) -> T:
        return visitor.visit_bit_ref(self)

    def __str__(self) -> str:
        return f"{self.register}[{self.index}]"


# =============================================================================
# Expressions
# =============================================================================


@dataclass(frozen=True, kw_only=True)
class Expression(AstNode, ABC):
    """Base class for all expressions."""


@dataclass(frozen=True, kw_only=True)
class LiteralExpr(Expression):
    """Literal value (int, float, bool)."""

    value: int | float | bool

    def accept(self, visitor: AstVisitor[T]) -> T:
        return visitor.visit_literal(self)


@dataclass(frozen=True, kw_only=True)
class VarExpr(Expression):
    """Variable reference."""

    name: str

    def accept(self, visitor: AstVisitor[T]) -> T:
        return visitor.visit_var(self)


@dataclass(frozen=True, kw_only=True)
class BitExpr(Expression):
    """Bit reference as expression (for conditions)."""

    ref: BitRef

    def accept(self, visitor: AstVisitor[T]) -> T:
        return visitor.visit_bit_expr(self)

    def children(self) -> Sequence[AstNode]:
        return (self.ref,)


@dataclass(frozen=True, kw_only=True)
class BinaryExpr(Expression):
    """Binary operation."""

    op: BinaryOp
    left: Expression
    right: Expression

    def accept(self, visitor: AstVisitor[T]) -> T:
        return visitor.visit_binary(self)

    def children(self) -> Sequence[AstNode]:
        return (self.left, self.right)


@dataclass(frozen=True, kw_only=True)
class UnaryExpr(Expression):
    """Unary operation."""

    op: UnaryOp
    operand: Expression

    def accept(self, visitor: AstVisitor[T]) -> T:
        return visitor.visit_unary(self)

    def children(self) -> Sequence[AstNode]:
        return (self.operand,)


# =============================================================================
# Type Expressions
# =============================================================================


@dataclass(frozen=True, kw_only=True)
class TypeExpr(AstNode, ABC):
    """Base class for type expressions."""


@dataclass(frozen=True, kw_only=True)
class QubitTypeExpr(TypeExpr):
    """Single qubit type."""

    def accept(self, visitor: AstVisitor[T]) -> T:
        return visitor.visit_qubit_type(self)


@dataclass(frozen=True, kw_only=True)
class BitTypeExpr(TypeExpr):
    """Single classical bit type."""

    def accept(self, visitor: AstVisitor[T]) -> T:
        return visitor.visit_bit_type(self)


@dataclass(frozen=True, kw_only=True)
class ArrayTypeExpr(TypeExpr):
    """Array type with element type and size."""

    element: TypeExpr
    size: int

    def accept(self, visitor: AstVisitor[T]) -> T:
        return visitor.visit_array_type(self)

    def children(self) -> Sequence[AstNode]:
        return (self.element,)


@dataclass(frozen=True, kw_only=True)
class AllocatorTypeExpr(TypeExpr):
    """Qubit allocator type with capacity."""

    capacity: int

    def accept(self, visitor: AstVisitor[T]) -> T:
        return visitor.visit_allocator_type(self)


# =============================================================================
# Declarations
# =============================================================================


@dataclass(frozen=True, kw_only=True)
class Declaration(AstNode, ABC):
    """Base class for all declarations."""


@dataclass(frozen=True, kw_only=True)
class AllocatorDecl(Declaration):
    """Qubit allocator declaration.

    Represents a QAlloc in the AST. Can have a parent for hierarchical
    allocation.
    """

    name: str
    capacity: int
    parent: str | None = None  # Name of parent allocator

    def accept(self, visitor: AstVisitor[T]) -> T:
        return visitor.visit_allocator_decl(self)


@dataclass(frozen=True, kw_only=True)
class RegisterDecl(Declaration):
    """Classical register declaration."""

    name: str
    size: int
    is_result: bool = True

    def accept(self, visitor: AstVisitor[T]) -> T:
        return visitor.visit_register_decl(self)


# =============================================================================
# Statements
# =============================================================================


@dataclass(frozen=True, kw_only=True)
class Statement(AstNode, ABC):
    """Base class for all statements."""


@dataclass(frozen=True, kw_only=True)
class GateOp(Statement):
    """Quantum gate application."""

    gate: GateKind
    targets: tuple[SlotRef, ...]
    params: tuple[Expression, ...] = ()

    def accept(self, visitor: AstVisitor[T]) -> T:
        return visitor.visit_gate(self)

    def children(self) -> Sequence[AstNode]:
        return (*self.targets, *self.params)


@dataclass(frozen=True, kw_only=True)
class PrepareOp(Statement):
    """Prepare qubit slots (unprepared -> prepared).

    Represents the prepare() operation from QAlloc.
    If slots is None, all slots in the allocator are prepared.
    """

    allocator: str
    slots: tuple[int, ...] | None = None  # None means prepare_all

    def accept(self, visitor: AstVisitor[T]) -> T:
        return visitor.visit_prepare(self)


@dataclass(frozen=True, kw_only=True)
class MeasureOp(Statement):
    """Measure qubit slots.

    After measurement, slots transition to unprepared state.
    """

    targets: tuple[SlotRef, ...]
    results: tuple[BitRef, ...] = ()

    def accept(self, visitor: AstVisitor[T]) -> T:
        return visitor.visit_measure(self)

    def children(self) -> Sequence[AstNode]:
        return (*self.targets, *self.results)


@dataclass(frozen=True, kw_only=True)
class AssignOp(Statement):
    """Classical assignment."""

    target: BitRef | str  # Variable name or bit reference
    value: Expression

    def accept(self, visitor: AstVisitor[T]) -> T:
        return visitor.visit_assign(self)

    def children(self) -> Sequence[AstNode]:
        nodes: list[AstNode] = []
        if isinstance(self.target, BitRef):
            nodes.append(self.target)
        nodes.append(self.value)
        return nodes


@dataclass(frozen=True, kw_only=True)
class BarrierOp(Statement):
    """Synchronization barrier."""

    allocators: tuple[str, ...] = ()

    def accept(self, visitor: AstVisitor[T]) -> T:
        return visitor.visit_barrier(self)


@dataclass(frozen=True, kw_only=True)
class CommentOp(Statement):
    """Comment in generated code."""

    text: str

    def accept(self, visitor: AstVisitor[T]) -> T:
        return visitor.visit_comment(self)


@dataclass(frozen=True, kw_only=True)
class ReturnOp(Statement):
    """Return statement."""

    values: tuple[Expression | str, ...] = ()  # Can be variable names

    def accept(self, visitor: AstVisitor[T]) -> T:
        return visitor.visit_return(self)

    def children(self) -> Sequence[AstNode]:
        return tuple(v for v in self.values if isinstance(v, AstNode))


@dataclass(frozen=True, kw_only=True)
class PermuteOp(Statement):
    """Permute qubit register assignments.

    Swaps the qubit assignments between two registers or allocators.
    After Permute(a, b), 'a' refers to what was 'b' and vice versa.

    This is a logical permutation that affects register naming, not
    physical qubit movement.
    """

    sources: tuple[str, ...]  # Initial register/allocator names
    targets: tuple[str, ...]  # Final register/allocator names
    add_comment: bool = True  # Whether to add a comment in generated code

    def accept(self, visitor: AstVisitor[T]) -> T:
        return visitor.visit_permute(self)


# =============================================================================
# Control Flow
# =============================================================================


@dataclass(frozen=True, kw_only=True)
class IfStmt(Statement):
    """Conditional execution."""

    condition: Expression
    then_body: tuple[Statement, ...]
    else_body: tuple[Statement, ...] = ()

    def accept(self, visitor: AstVisitor[T]) -> T:
        return visitor.visit_if(self)

    def children(self) -> Sequence[AstNode]:
        return (self.condition, *self.then_body, *self.else_body)


@dataclass(frozen=True, kw_only=True)
class WhileStmt(Statement):
    """While loop."""

    condition: Expression
    body: tuple[Statement, ...]

    def accept(self, visitor: AstVisitor[T]) -> T:
        return visitor.visit_while(self)

    def children(self) -> Sequence[AstNode]:
        return (self.condition, *self.body)


@dataclass(frozen=True, kw_only=True)
class ForStmt(Statement):
    """For loop with iteration variable."""

    variable: str
    start: Expression
    stop: Expression
    step: Expression | None = None
    body: tuple[Statement, ...] = ()

    def accept(self, visitor: AstVisitor[T]) -> T:
        return visitor.visit_for(self)

    def children(self) -> Sequence[AstNode]:
        nodes: list[AstNode] = [self.start, self.stop]
        if self.step:
            nodes.append(self.step)
        nodes.extend(self.body)
        return nodes


@dataclass(frozen=True, kw_only=True)
class RepeatStmt(Statement):
    """Repeat N times."""

    count: int
    body: tuple[Statement, ...]

    def accept(self, visitor: AstVisitor[T]) -> T:
        return visitor.visit_repeat(self)

    def children(self) -> Sequence[AstNode]:
        return self.body


@dataclass(frozen=True, kw_only=True)
class ParallelBlock(Statement):
    """Parallel execution hint."""

    body: tuple[Statement, ...]

    def accept(self, visitor: AstVisitor[T]) -> T:
        return visitor.visit_parallel(self)

    def children(self) -> Sequence[AstNode]:
        return self.body


# =============================================================================
# Program
# =============================================================================


@dataclass(frozen=True, kw_only=True)
class Program(AstNode):
    """Root node representing an SLR program.

    A program consists of:
    - An optional base allocator (required in strict mode)
    - Declarations (allocators, registers)
    - Body statements
    - Optional return type specification
    """

    name: str
    declarations: tuple[Declaration, ...] = ()
    body: tuple[Statement, ...] = ()
    returns: tuple[TypeExpr, ...] = ()
    allocator: AllocatorDecl | None = None  # Base allocator

    def accept(self, visitor: AstVisitor[T]) -> T:
        return visitor.visit_program(self)

    def children(self) -> Sequence[AstNode]:
        nodes: list[AstNode] = []
        if self.allocator:
            nodes.append(self.allocator)
        nodes.extend(self.declarations)
        nodes.extend(self.body)
        nodes.extend(self.returns)
        return nodes

    def get_allocator(self, name: str) -> AllocatorDecl | None:
        """Find an allocator declaration by name."""
        if self.allocator and self.allocator.name == name:
            return self.allocator
        for decl in self.declarations:
            if isinstance(decl, AllocatorDecl) and decl.name == name:
                return decl
        return None

    def get_register(self, name: str) -> RegisterDecl | None:
        """Find a register declaration by name."""
        for decl in self.declarations:
            if isinstance(decl, RegisterDecl) and decl.name == name:
                return decl
        return None
