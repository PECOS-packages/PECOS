# SLR Abstract Syntax Tree (AST) Proposal

## Status

**Draft** - Ready for review

---

## Motivation

### Current State

SLR currently uses Python classes directly as both the syntax and the runtime representation:

```python
prog = Main(
    q := QReg("q", 2),
    c := CReg("c", 2),
    qb.H(q[0]),
    qb.CX(q[0], q[1]),
    qb.Measure(q) > c,
)
```

This approach has drawbacks:

1. **Mixed concerns**: Representation classes also contain execution logic
2. **Difficult analysis**: No clean separation for static analysis passes
3. **Inconsistent structure**: Different node types have different interfaces
4. **Hard to transform**: Modifying programs requires understanding implementation details
5. **Code generation complexity**: Generators work directly with heterogeneous objects

### Benefits of a Formal AST

1. **Clean separation**: Syntax representation separate from semantics
2. **Uniform interface**: All nodes share a common base with predictable structure
3. **Easy traversal**: Visitor pattern for analysis and transformation
4. **Better tooling**: Linting, formatting, refactoring tools
5. **Simpler code gen**: One AST → multiple targets (QASM, Guppy, HUGR, etc.)
6. **Integration with QAlloc**: Clean representation of allocator hierarchy and slot states

---

## Design Principles

### 1. Immutable Data Structures

AST nodes should be immutable dataclasses. This enables:
- Safe sharing and caching
- Easy equality comparison
- Predictable behavior in analysis passes

### 2. Type Safety

Use Python's type system with generics and protocols:
- All nodes have precise types
- Analysis results are strongly typed
- IDE support for navigation and refactoring

### 3. Location Tracking

Every node can optionally track source location for error reporting:
```python
@dataclass(frozen=True)
class SourceLocation:
    line: int
    column: int
    file: str | None = None
```

### 4. Visitor Pattern

Support the visitor pattern for analysis and transformation:
```python
class ASTVisitor(Protocol[T]):
    def visit_program(self, node: Program) -> T: ...
    def visit_gate(self, node: GateOp) -> T: ...
    # etc.
```

### 5. Bidirectional Conversion

The AST should support:
- Building from current SLR objects (`from_slr()`)
- Converting back for execution (`to_slr()`)
- Direct construction for new code

---

## AST Node Hierarchy

### Base Types

```python
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TypeVar, Generic, Protocol, Sequence

@dataclass(frozen=True)
class SourceLocation:
    """Source location for error reporting."""
    line: int
    column: int
    file: str | None = None

@dataclass(frozen=True)
class ASTNode(ABC):
    """Base class for all AST nodes."""
    location: SourceLocation | None = field(default=None, compare=False)

    @abstractmethod
    def accept(self, visitor: ASTVisitor[T]) -> T:
        """Accept a visitor for traversal."""
        ...

    def children(self) -> Sequence[ASTNode]:
        """Return child nodes for traversal."""
        return ()
```

### Program Structure

```python
@dataclass(frozen=True)
class Program(ASTNode):
    """Root node representing an SLR program."""
    name: str
    allocator: AllocatorDecl | None  # Base allocator (required in strict mode)
    declarations: tuple[Declaration, ...]
    body: tuple[Statement, ...]
    returns: tuple[TypeExpr, ...] = ()

    def accept(self, visitor: ASTVisitor[T]) -> T:
        return visitor.visit_program(self)

    def children(self) -> Sequence[ASTNode]:
        nodes = []
        if self.allocator:
            nodes.append(self.allocator)
        nodes.extend(self.declarations)
        nodes.extend(self.body)
        return nodes
```

### Declarations

```python
class Declaration(ASTNode, ABC):
    """Base for all declarations."""
    pass

@dataclass(frozen=True)
class AllocatorDecl(Declaration):
    """Qubit allocator declaration."""
    name: str
    capacity: int
    parent: str | None = None  # Name of parent allocator

    def accept(self, visitor: ASTVisitor[T]) -> T:
        return visitor.visit_allocator_decl(self)

@dataclass(frozen=True)
class RegisterDecl(Declaration):
    """Classical register declaration."""
    name: str
    size: int
    is_result: bool = True

    def accept(self, visitor: ASTVisitor[T]) -> T:
        return visitor.visit_register_decl(self)
```

### Statements

```python
class Statement(ASTNode, ABC):
    """Base for all statements."""
    pass

@dataclass(frozen=True)
class GateOp(Statement):
    """Quantum gate application."""
    gate: GateKind
    targets: tuple[SlotRef, ...]
    params: tuple[Expression, ...] = ()

    def accept(self, visitor: ASTVisitor[T]) -> T:
        return visitor.visit_gate(self)

    def children(self) -> Sequence[ASTNode]:
        return (*self.targets, *self.params)

@dataclass(frozen=True)
class PrepareOp(Statement):
    """Prepare qubit slots (unprepared -> prepared)."""
    allocator: str
    slots: tuple[int, ...] | None = None  # None means all slots

    def accept(self, visitor: ASTVisitor[T]) -> T:
        return visitor.visit_prepare(self)

@dataclass(frozen=True)
class MeasureOp(Statement):
    """Measure qubit slots."""
    targets: tuple[SlotRef, ...]
    results: tuple[BitRef, ...] = ()

    def accept(self, visitor: ASTVisitor[T]) -> T:
        return visitor.visit_measure(self)

    def children(self) -> Sequence[ASTNode]:
        return (*self.targets, *self.results)

@dataclass(frozen=True)
class AssignOp(Statement):
    """Classical assignment."""
    target: BitRef | str  # Variable or bit reference
    value: Expression

    def accept(self, visitor: ASTVisitor[T]) -> T:
        return visitor.visit_assign(self)

    def children(self) -> Sequence[ASTNode]:
        nodes = [self.value]
        if isinstance(self.target, ASTNode):
            nodes.insert(0, self.target)
        return nodes

@dataclass(frozen=True)
class BarrierOp(Statement):
    """Synchronization barrier."""
    allocators: tuple[str, ...] = ()

    def accept(self, visitor: ASTVisitor[T]) -> T:
        return visitor.visit_barrier(self)

@dataclass(frozen=True)
class CommentOp(Statement):
    """Comment in generated code."""
    text: str

    def accept(self, visitor: ASTVisitor[T]) -> T:
        return visitor.visit_comment(self)

@dataclass(frozen=True)
class ReturnOp(Statement):
    """Return statement."""
    values: tuple[Expression, ...]

    def accept(self, visitor: ASTVisitor[T]) -> T:
        return visitor.visit_return(self)

    def children(self) -> Sequence[ASTNode]:
        return self.values
```

### Control Flow

```python
@dataclass(frozen=True)
class IfStmt(Statement):
    """Conditional execution."""
    condition: Expression
    then_body: tuple[Statement, ...]
    else_body: tuple[Statement, ...] = ()

    def accept(self, visitor: ASTVisitor[T]) -> T:
        return visitor.visit_if(self)

    def children(self) -> Sequence[ASTNode]:
        return (self.condition, *self.then_body, *self.else_body)

@dataclass(frozen=True)
class WhileStmt(Statement):
    """While loop."""
    condition: Expression
    body: tuple[Statement, ...]

    def accept(self, visitor: ASTVisitor[T]) -> T:
        return visitor.visit_while(self)

    def children(self) -> Sequence[ASTNode]:
        return (self.condition, *self.body)

@dataclass(frozen=True)
class ForStmt(Statement):
    """For loop with iteration variable."""
    variable: str
    start: Expression
    stop: Expression
    step: Expression | None = None
    body: tuple[Statement, ...]

    def accept(self, visitor: ASTVisitor[T]) -> T:
        return visitor.visit_for(self)

    def children(self) -> Sequence[ASTNode]:
        nodes = [self.start, self.stop]
        if self.step:
            nodes.append(self.step)
        nodes.extend(self.body)
        return nodes

@dataclass(frozen=True)
class RepeatStmt(Statement):
    """Repeat N times."""
    count: int
    body: tuple[Statement, ...]

    def accept(self, visitor: ASTVisitor[T]) -> T:
        return visitor.visit_repeat(self)

    def children(self) -> Sequence[ASTNode]:
        return self.body

@dataclass(frozen=True)
class ParallelBlock(Statement):
    """Parallel execution hint."""
    body: tuple[Statement, ...]

    def accept(self, visitor: ASTVisitor[T]) -> T:
        return visitor.visit_parallel(self)

    def children(self) -> Sequence[ASTNode]:
        return self.body
```

### References

```python
@dataclass(frozen=True)
class SlotRef(ASTNode):
    """Reference to a qubit slot in an allocator."""
    allocator: str
    index: int

    def accept(self, visitor: ASTVisitor[T]) -> T:
        return visitor.visit_slot_ref(self)

    def __str__(self) -> str:
        return f"{self.allocator}[{self.index}]"

@dataclass(frozen=True)
class BitRef(ASTNode):
    """Reference to a classical bit in a register."""
    register: str
    index: int

    def accept(self, visitor: ASTVisitor[T]) -> T:
        return visitor.visit_bit_ref(self)

    def __str__(self) -> str:
        return f"{self.register}[{self.index}]"
```

### Expressions

```python
class Expression(ASTNode, ABC):
    """Base for all expressions."""
    pass

@dataclass(frozen=True)
class LiteralExpr(Expression):
    """Literal value (int, float, bool)."""
    value: int | float | bool

    def accept(self, visitor: ASTVisitor[T]) -> T:
        return visitor.visit_literal(self)

@dataclass(frozen=True)
class VarExpr(Expression):
    """Variable reference."""
    name: str

    def accept(self, visitor: ASTVisitor[T]) -> T:
        return visitor.visit_var(self)

@dataclass(frozen=True)
class BitExpr(Expression):
    """Bit reference as expression (for conditions)."""
    ref: BitRef

    def accept(self, visitor: ASTVisitor[T]) -> T:
        return visitor.visit_bit_expr(self)

    def children(self) -> Sequence[ASTNode]:
        return (self.ref,)

@dataclass(frozen=True)
class BinaryExpr(Expression):
    """Binary operation."""
    op: BinaryOp
    left: Expression
    right: Expression

    def accept(self, visitor: ASTVisitor[T]) -> T:
        return visitor.visit_binary(self)

    def children(self) -> Sequence[ASTNode]:
        return (self.left, self.right)

@dataclass(frozen=True)
class UnaryExpr(Expression):
    """Unary operation."""
    op: UnaryOp
    operand: Expression

    def accept(self, visitor: ASTVisitor[T]) -> T:
        return visitor.visit_unary(self)

    def children(self) -> Sequence[ASTNode]:
        return (self.operand,)

class BinaryOp(Enum):
    """Binary operators."""
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
    """Unary operators."""
    NOT = auto()
    NEG = auto()
```

### Gate Kinds

```python
class GateKind(Enum):
    """All supported gate types."""
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
    # Two-qubit gates
    CX = auto()
    CY = auto()
    CZ = auto()
    CH = auto()
    # Two-qubit rotations
    SXX = auto()
    SYY = auto()
    SZZ = auto()
    SXXdg = auto()
    SYYdg = auto()
    SZZdg = auto()
    RZZ = auto()
    # Face rotations
    F = auto()
    Fdg = auto()
    F4 = auto()
    F4dg = auto()

    @property
    def arity(self) -> int:
        """Number of qubit arguments."""
        two_qubit = {
            GateKind.CX, GateKind.CY, GateKind.CZ, GateKind.CH,
            GateKind.SXX, GateKind.SYY, GateKind.SZZ,
            GateKind.SXXdg, GateKind.SYYdg, GateKind.SZZdg,
            GateKind.RZZ,
        }
        return 2 if self in two_qubit else 1

    @property
    def is_parameterized(self) -> bool:
        """Whether this gate takes angle parameters."""
        return self in {GateKind.RX, GateKind.RY, GateKind.RZ, GateKind.RZZ}
```

### Type Expressions

```python
@dataclass(frozen=True)
class TypeExpr(ASTNode):
    """Type expression for return types and declarations."""
    pass

@dataclass(frozen=True)
class QubitType(TypeExpr):
    """Single qubit type."""
    def accept(self, visitor: ASTVisitor[T]) -> T:
        return visitor.visit_qubit_type(self)

@dataclass(frozen=True)
class BitType(TypeExpr):
    """Single classical bit type."""
    def accept(self, visitor: ASTVisitor[T]) -> T:
        return visitor.visit_bit_type(self)

@dataclass(frozen=True)
class ArrayType(TypeExpr):
    """Array type with element type and size."""
    element: TypeExpr
    size: int

    def accept(self, visitor: ASTVisitor[T]) -> T:
        return visitor.visit_array_type(self)

    def children(self) -> Sequence[ASTNode]:
        return (self.element,)

@dataclass(frozen=True)
class AllocatorType(TypeExpr):
    """Qubit allocator type with capacity."""
    capacity: int

    def accept(self, visitor: ASTVisitor[T]) -> T:
        return visitor.visit_allocator_type(self)
```

---

## Visitor Protocol

```python
from typing import TypeVar, Protocol

T = TypeVar('T')

class ASTVisitor(Protocol[T]):
    """Protocol for AST visitors."""

    # Program structure
    def visit_program(self, node: Program) -> T: ...
    def visit_allocator_decl(self, node: AllocatorDecl) -> T: ...
    def visit_register_decl(self, node: RegisterDecl) -> T: ...

    # Statements
    def visit_gate(self, node: GateOp) -> T: ...
    def visit_prepare(self, node: PrepareOp) -> T: ...
    def visit_measure(self, node: MeasureOp) -> T: ...
    def visit_assign(self, node: AssignOp) -> T: ...
    def visit_barrier(self, node: BarrierOp) -> T: ...
    def visit_comment(self, node: CommentOp) -> T: ...
    def visit_return(self, node: ReturnOp) -> T: ...

    # Control flow
    def visit_if(self, node: IfStmt) -> T: ...
    def visit_while(self, node: WhileStmt) -> T: ...
    def visit_for(self, node: ForStmt) -> T: ...
    def visit_repeat(self, node: RepeatStmt) -> T: ...
    def visit_parallel(self, node: ParallelBlock) -> T: ...

    # References
    def visit_slot_ref(self, node: SlotRef) -> T: ...
    def visit_bit_ref(self, node: BitRef) -> T: ...

    # Expressions
    def visit_literal(self, node: LiteralExpr) -> T: ...
    def visit_var(self, node: VarExpr) -> T: ...
    def visit_bit_expr(self, node: BitExpr) -> T: ...
    def visit_binary(self, node: BinaryExpr) -> T: ...
    def visit_unary(self, node: UnaryExpr) -> T: ...

    # Types
    def visit_qubit_type(self, node: QubitType) -> T: ...
    def visit_bit_type(self, node: BitType) -> T: ...
    def visit_array_type(self, node: ArrayType) -> T: ...
    def visit_allocator_type(self, node: AllocatorType) -> T: ...


class BaseVisitor(Generic[T]):
    """Base visitor with default traversal behavior."""

    def visit(self, node: ASTNode) -> T:
        """Dispatch to appropriate visit method."""
        return node.accept(self)

    def visit_children(self, node: ASTNode) -> list[T]:
        """Visit all children and collect results."""
        return [self.visit(child) for child in node.children()]

    # Default implementations that just visit children
    def visit_program(self, node: Program) -> T:
        self.visit_children(node)
        return self.default_result()

    # ... etc for all node types ...

    def default_result(self) -> T:
        """Default result when no specific handling."""
        return None  # type: ignore
```

---

## Example: AST for QEC Program

```python
# The SLR code:
# def main():
#     base = QAlloc(17)
#     data = base.child(9)
#     ancilla = base.child(8)
#     data.prepare_all()
#     ancilla.prepare_all()
#     H(data[0])
#     CX(data[0], data[1])
#     Measure(ancilla) > syndrome

# As AST:
program = Program(
    name="main",
    allocator=AllocatorDecl(name="base", capacity=17),
    declarations=(
        AllocatorDecl(name="data", capacity=9, parent="base"),
        AllocatorDecl(name="ancilla", capacity=8, parent="base"),
        RegisterDecl(name="syndrome", size=8),
    ),
    body=(
        PrepareOp(allocator="data"),
        PrepareOp(allocator="ancilla"),
        GateOp(
            gate=GateKind.H,
            targets=(SlotRef("data", 0),),
        ),
        GateOp(
            gate=GateKind.CX,
            targets=(SlotRef("data", 0), SlotRef("data", 1)),
        ),
        MeasureOp(
            targets=tuple(SlotRef("ancilla", i) for i in range(8)),
            results=tuple(BitRef("syndrome", i) for i in range(8)),
        ),
    ),
)
```

---

## Integration with QAlloc

The AST is designed to work seamlessly with the QAlloc system:

### Allocator Hierarchy in AST

```python
# Parent-child relationships are explicit
AllocatorDecl(name="base", capacity=100)
AllocatorDecl(name="data", capacity=7, parent="base")
AllocatorDecl(name="ancilla", capacity=6, parent="base")
```

### Slot State Validation

The `QubitStateValidator` can work directly on the AST:

```python
class ASTStateValidator(BaseVisitor[None]):
    """Validate qubit states on AST."""

    def __init__(self):
        self.slot_states: dict[tuple[str, int], SlotState] = {}
        self.violations: list[StateViolation] = []

    def visit_prepare(self, node: PrepareOp) -> None:
        # Mark slots as prepared
        if node.slots is None:
            # prepare_all - need allocator capacity
            ...
        else:
            for slot in node.slots:
                self.slot_states[(node.allocator, slot)] = SlotState.PREPARED

    def visit_gate(self, node: GateOp) -> None:
        # Validate all targets are prepared
        for target in node.targets:
            state = self.slot_states.get(
                (target.allocator, target.index),
                SlotState.UNPREPARED
            )
            if state == SlotState.UNPREPARED:
                self.violations.append(StateViolation(...))

    def visit_measure(self, node: MeasureOp) -> None:
        # Mark slots as unprepared
        for target in node.targets:
            self.slot_states[(target.allocator, target.index)] = SlotState.UNPREPARED
```

---

## Conversion Functions

### From Current SLR to AST

```python
def slr_to_ast(block: SLRBlock) -> Program:
    """Convert current SLR block to AST."""
    converter = SLRToASTConverter()
    return converter.convert(block)

class SLRToASTConverter:
    """Converts SLR objects to AST nodes."""

    def convert(self, block: SLRBlock) -> Program:
        declarations = []
        body = []

        # Convert variables
        for var in block.vars:
            declarations.append(self.convert_var(var))

        # Convert operations
        for op in block.ops:
            body.append(self.convert_op(op))

        return Program(
            name=getattr(block, "block_name", "main"),
            allocator=None,  # Legacy mode - no base allocator
            declarations=tuple(declarations),
            body=tuple(body),
        )

    def convert_var(self, var) -> Declaration:
        if isinstance(var, QReg):
            # Convert to legacy allocator-like declaration
            return AllocatorDecl(name=var.sym, capacity=var.size)
        elif isinstance(var, CReg):
            return RegisterDecl(name=var.sym, size=var.size, is_result=var.result)
        elif isinstance(var, QAlloc):
            return AllocatorDecl(
                name=var.name,
                capacity=var.capacity,
                parent=var.parent.name if var.parent else None,
            )
        # ... etc

    def convert_op(self, op) -> Statement:
        op_name = type(op).__name__

        if op_name in GATE_NAMES:
            return self.convert_gate(op)
        elif op_name == "Measure":
            return self.convert_measure(op)
        elif op_name in ("Prep", "Init", "Reset"):
            return self.convert_prepare(op)
        elif op_name == "If":
            return self.convert_if(op)
        # ... etc
```

### From AST to Current SLR

```python
def ast_to_slr(program: Program) -> SLRBlock:
    """Convert AST to current SLR objects for execution."""
    converter = ASTToSLRConverter()
    return converter.convert(program)
```

---

## Code Generation from AST

Each target gets a visitor:

```python
class QASMGenerator(BaseVisitor[str]):
    """Generate QASM from AST."""

    def visit_program(self, node: Program) -> str:
        lines = ["OPENQASM 2.0;", 'include "qelib1.inc";', ""]

        # Declarations
        for decl in node.declarations:
            lines.append(self.visit(decl))

        lines.append("")

        # Body
        for stmt in node.body:
            lines.append(self.visit(stmt))

        return "\n".join(lines)

    def visit_allocator_decl(self, node: AllocatorDecl) -> str:
        return f"qreg {node.name}[{node.capacity}];"

    def visit_register_decl(self, node: RegisterDecl) -> str:
        return f"creg {node.name}[{node.size}];"

    def visit_gate(self, node: GateOp) -> str:
        gate_name = node.gate.name.lower()
        targets = ", ".join(str(t) for t in node.targets)
        if node.params:
            params = ", ".join(str(p) for p in node.params)
            return f"{gate_name}({params}) {targets};"
        return f"{gate_name} {targets};"

    # ... etc


class GuppyGenerator(BaseVisitor[str]):
    """Generate Guppy code from AST."""
    # Similar structure, different output format


class HUGRGenerator(BaseVisitor[HUGRNode]):
    """Generate HUGR from AST."""
    # Returns HUGR IR nodes instead of strings
```

---

## Implementation Plan

### Phase 1: Core AST Nodes

1. Define all node dataclasses in `pecos/slr/ast/nodes.py`
2. Define visitor protocol in `pecos/slr/ast/visitor.py`
3. Implement `BaseVisitor` with default traversal

### Phase 2: Conversion

1. Implement `SLRToASTConverter` for current SLR → AST
2. Implement `ASTToSLRConverter` for AST → current SLR
3. Add tests for round-trip conversion

### Phase 3: Analysis

1. Port `QubitStateValidator` to work on AST
2. Add more analysis passes (unused variables, unreachable code, etc.)
3. Add pretty-printer for debugging

### Phase 4: Code Generation

1. Migrate QASM generator to use AST
2. Migrate Guppy generator to use AST
3. Add new generators as needed

### Phase 5: DSL Integration

1. Consider new DSL syntax that builds AST directly
2. Add builder pattern for programmatic AST construction
3. Integration with IDE tooling

---

## Open Questions

1. **Span tracking**: Should we track full spans (start + end) or just start locations?

2. **Error recovery**: Should AST support partial/invalid trees for better error reporting?

3. **Macro expansion**: How to handle QEC library blocks (Steane, etc.) that expand to multiple operations?

4. **Metadata**: What additional metadata should nodes carry (e.g., optimization hints)?

5. **Serialization**: Should AST be serializable to JSON/protobuf for tooling?

---

## Summary

The SLR AST provides:

- **Clean structure**: Immutable, typed nodes with uniform interface
- **Easy analysis**: Visitor pattern for traversal and transformation
- **QAlloc integration**: First-class support for allocator hierarchy and slot states
- **Multi-target codegen**: One AST → QASM, Guppy, HUGR, etc.
- **Backward compatibility**: Conversion to/from current SLR objects
