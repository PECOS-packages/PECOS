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

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pecos.slr.angle import Angle


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
class AstNode:
    """Base class for all AST nodes.

    All AST nodes are immutable frozen dataclasses that support:
    - Visitor traversal via `BaseVisitor` (centralized dispatch)
    - Child traversal via children()
    - Optional source location tracking
    """

    location: SourceLocation | None = field(default=None, compare=False, repr=False)

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


class ResourceEffect(Enum):
    """Effect declared by a `BlockDecl` input on the outer scope's binding."""

    LIVE_PRESERVED = auto()  # caller binding survives the call unchanged
    CONSUMED = auto()  # caller binding is invalidated by the call
    PRODUCED = auto()  # callee writes; caller's binding is rebound from return
    DROPPED = auto()  # callee discards; caller binding is invalidated
    # Reset-reused scratch ancilla: the block resets the input at entry and
    # measures it at exit, depending on no incoming state. The caller's slot is
    # a flatten-path naming vehicle only; in Guppy the block allocates the qubit
    # internally so the same outer slot can feed a subsequent BlockCall (the
    # `consumed` model would kill it).
    SCRATCH = auto()


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

    def __str__(self) -> str:
        return f"{self.allocator}[{self.index}]"


@dataclass(frozen=True, kw_only=True)
class BitRef(AstNode):
    """Reference to a classical bit in a register."""

    register: str
    index: int

    def __str__(self) -> str:
        return f"{self.register}[{self.index}]"


# =============================================================================
# Expressions
# =============================================================================


@dataclass(frozen=True, kw_only=True)
class Expression(AstNode):
    """Base class for all expressions."""


@dataclass(frozen=True, kw_only=True)
class LiteralExpr(Expression):
    """Literal value (int, float, bool, or a typed rotation `Angle`)."""

    value: int | float | bool | Angle


@dataclass(frozen=True, kw_only=True)
class VarExpr(Expression):
    """Variable reference."""

    name: str


@dataclass(frozen=True, kw_only=True)
class BitExpr(Expression):
    """Bit reference as expression (for conditions)."""

    ref: BitRef

    def children(self) -> Sequence[AstNode]:
        return (self.ref,)


@dataclass(frozen=True, kw_only=True)
class BinaryExpr(Expression):
    """Binary operation."""

    op: BinaryOp
    left: Expression
    right: Expression

    def children(self) -> Sequence[AstNode]:
        return (self.left, self.right)


@dataclass(frozen=True, kw_only=True)
class UnaryExpr(Expression):
    """Unary operation."""

    op: UnaryOp
    operand: Expression

    def children(self) -> Sequence[AstNode]:
        return (self.operand,)


# =============================================================================
# Type Expressions
# =============================================================================


@dataclass(frozen=True, kw_only=True)
class TypeExpr(AstNode):
    """Base class for type expressions."""


@dataclass(frozen=True, kw_only=True)
class QubitTypeExpr(TypeExpr):
    """Single qubit type."""


@dataclass(frozen=True, kw_only=True)
class BitTypeExpr(TypeExpr):
    """Single classical bit type."""


@dataclass(frozen=True, kw_only=True)
class ArrayTypeExpr(TypeExpr):
    """Array type with element type and size."""

    element: TypeExpr
    size: int

    def children(self) -> Sequence[AstNode]:
        return (self.element,)


@dataclass(frozen=True, kw_only=True)
class AllocatorTypeExpr(TypeExpr):
    """Qubit allocator type with capacity."""

    capacity: int


# =============================================================================
# Declarations
# =============================================================================


@dataclass(frozen=True, kw_only=True)
class Declaration(AstNode):
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


@dataclass(frozen=True, kw_only=True)
class RegisterDecl(Declaration):
    """Classical register declaration."""

    name: str
    size: int


# =============================================================================
# Statements
# =============================================================================


@dataclass(frozen=True, kw_only=True)
class Statement(AstNode):
    """Base class for all statements."""


@dataclass(frozen=True, kw_only=True)
class GateOp(Statement):
    """Quantum gate application."""

    gate: GateKind
    targets: tuple[SlotRef, ...]
    params: tuple[Expression, ...] = ()

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
    # Canonical prep basis / target eigenstate, one of
    # {PZ,PNZ,PX,PNX,PY,PNY} (|0>,|1>,|+>,|->,|+i>,|-i>). Set by
    # `_convert_prep` from the SLR gate symbol (PZ default). Carried
    # on the AST so codegens lower the correct reset+Clifford tail;
    # MUST be preserved through block substitution (else a non-PZ
    # prep inside a BlockCall body silently reverts to PZ -- a
    # soundness-critical case).
    basis: str = "PZ"


@dataclass(frozen=True, kw_only=True)
class MeasureOp(Statement):
    """Measure qubit slots.

    After measurement, slots transition to unprepared state.
    """

    targets: tuple[SlotRef, ...]
    results: tuple[BitRef, ...] = ()

    def children(self) -> Sequence[AstNode]:
        return (*self.targets, *self.results)


@dataclass(frozen=True, kw_only=True)
class AssignOp(Statement):
    """Classical assignment."""

    target: BitRef | str  # Variable name or bit reference
    value: Expression

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


@dataclass(frozen=True, kw_only=True)
class CommentOp(Statement):
    """Comment in generated code."""

    text: str


@dataclass(frozen=True, kw_only=True)
class ReturnOp(Statement):
    """Return statement."""

    values: tuple[Expression | str, ...] = ()  # Can be variable names
    # Per-value provenance, parallel to `values`: "quantum" (a QReg --
    # no classical record), "classical" (a CReg -- must be recorded),
    # or "expr". Set by `_convert_return` from the SLR object type;
    # `()` means unknown (e.g. a directly-constructed ReturnOp), in
    # which case a backend treats a bare-name value as classical
    # (the fail-loud-safe default). A bare `values` string cannot be
    # disambiguated CReg-vs-QReg by name alone (a returned inline
    # CReg can collide with a declared QReg name -- the
    # name-collision bug), so provenance is carried here, not guessed.
    value_kinds: tuple[str, ...] = ()

    def children(self) -> Sequence[AstNode]:
        return tuple(v for v in self.values if isinstance(v, AstNode))


@dataclass(frozen=True, kw_only=True)
class PrintOp(Statement):
    """Emit an intermediate streamed value at the call site.

    Lowers to Guppy's `result(name, value)`. Scope-orthogonal side-effect:
    does not allocate, does not modify the result-register set used for
    return-shape computation.

    `value` is either a `BitRef` (single bit) or a register name string
    (whole-CReg emission). `tag` is the resolved tag (the SLR-side
    conversion derives the default from the value's name when the user
    did not pass `tag=` explicitly). `namespace` is the tag prefix; the
    full emitted Guppy tag is `f"{namespace}.{tag}"`.
    """

    value: BitRef | str
    tag: str
    namespace: str = "result"

    def children(self) -> Sequence[AstNode]:
        if isinstance(self.value, AstNode):
            return (self.value,)
        return ()


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
    # Whole-register swap `Permute(a, b)` -> comment `; Permutation:
    # a <-> b`; else per-element `; Permutation: a[0] -> b[1], ...`
    # (the comment is rendered at codegen from the post-substitution
    # sources/targets, mirroring the legacy gen_qir format).
    whole_register: bool = False


# =============================================================================
# Control Flow
# =============================================================================


@dataclass(frozen=True, kw_only=True)
class IfStmt(Statement):
    """Conditional execution."""

    condition: Expression
    then_body: tuple[Statement, ...]
    else_body: tuple[Statement, ...] = ()

    def children(self) -> Sequence[AstNode]:
        return (self.condition, *self.then_body, *self.else_body)


@dataclass(frozen=True, kw_only=True)
class WhileStmt(Statement):
    """While loop."""

    condition: Expression
    body: tuple[Statement, ...]

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

    def children(self) -> Sequence[AstNode]:
        return self.body


@dataclass(frozen=True, kw_only=True)
class ParallelBlock(Statement):
    """Parallel execution hint."""

    body: tuple[Statement, ...]

    def children(self) -> Sequence[AstNode]:
        return self.body


# =============================================================================
# Reusable block declarations
# =============================================================================


@dataclass(frozen=True, kw_only=True)
class BlockInput(AstNode):
    """One declared input parameter to a `BlockDecl`."""

    name: str
    effect: ResourceEffect
    type_expr: TypeExpr

    def children(self) -> Sequence[AstNode]:
        return (self.type_expr,)


@dataclass(frozen=True, kw_only=True)
class BlockDecl(AstNode):
    """Reusable Block declaration that lowers to a top-level Guppy function.

    Non-Guppy codegens inline `body` at every `BlockCall` site.
    """

    name: str
    inputs: tuple[BlockInput, ...]
    body: tuple[Statement, ...]
    return_op: ReturnOp | None = None

    def children(self) -> Sequence[AstNode]:
        nodes: list[AstNode] = list(self.inputs)
        nodes.extend(self.body)
        if self.return_op is not None:
            nodes.append(self.return_op)
        return nodes


# ---- BlockCall argument types (typed sum type) ----


@dataclass(frozen=True, kw_only=True)
class BlockArg(AstNode):
    """Base class for `BlockCall` argument bindings.

    Each BlockInput on the callee is bound to exactly one BlockArg at the
    caller, describing what outer-scope state the input refers to.
    """


@dataclass(frozen=True, kw_only=True)
class AllocatorArg(BlockArg):
    """Whole-allocator binding: every slot of an outer-scope allocator."""

    name: str


@dataclass(frozen=True, kw_only=True)
class SingleQubitArg(BlockArg):
    """Single-qubit slot binding."""

    slot: SlotRef

    def children(self) -> Sequence[AstNode]:
        return (self.slot,)


@dataclass(frozen=True, kw_only=True)
class SingleBitArg(BlockArg):
    """Single classical-bit binding (write-back via array[bool, 1] proxy in emitter)."""

    bit: BitRef

    def children(self) -> Sequence[AstNode]:
        return (self.bit,)


@dataclass(frozen=True, kw_only=True)
class QubitBundleArg(BlockArg):
    """Non-contiguous bundle of qubit slots packed into a single array[qubit, N]."""

    slots: tuple[SlotRef, ...]

    def children(self) -> Sequence[AstNode]:
        return self.slots


@dataclass(frozen=True, kw_only=True)
class BitBundleArg(BlockArg):
    """Non-contiguous bundle of classical bits packed into a single array[bool, N]."""

    bits: tuple[BitRef, ...]

    def children(self) -> Sequence[AstNode]:
        return self.bits


@dataclass(frozen=True, kw_only=True)
class BlockCall(Statement):
    """Invoke a `BlockDecl` from the outer scope.

    `arg_bindings` lists outer-scope bindings (typed `BlockArg`) in the same
    order as the callee's declared inputs (one per input).
    `out_bindings` lists outer-scope bindings that receive the callee's
    outputs (`live_preserved`/`produced` inputs + explicit `Return` values,
    in declaration order then return order). Empty for callees that return
    nothing.
    """

    callee: str
    arg_bindings: tuple[BlockArg, ...]
    out_bindings: tuple[BlockArg, ...] = ()

    def children(self) -> Sequence[AstNode]:
        return (*self.arg_bindings, *self.out_bindings)


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
    block_decls: tuple[BlockDecl, ...] = ()

    def children(self) -> Sequence[AstNode]:
        nodes: list[AstNode] = []
        if self.allocator:
            nodes.append(self.allocator)
        nodes.extend(self.declarations)
        nodes.extend(self.block_decls)
        nodes.extend(self.body)
        nodes.extend(self.returns)
        return nodes

    def get_block_decl(self, name: str) -> BlockDecl | None:
        """Find a BlockDecl by name."""
        for decl in self.block_decls:
            if decl.name == name:
                return decl
        return None

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
