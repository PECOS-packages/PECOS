"""Intermediate Representation for Guppy code generation.

This module provides an IR that allows us to analyze and transform code
before generating the final Guppy output.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar


class ResourceState(Enum):
    """State of a quantum resource."""

    AVAILABLE = "available"
    CONSUMED = "consumed"
    BORROWED = "borrowed"


@dataclass
class VariableInfo:
    """Information about a variable."""

    name: str
    original_name: str  # Before renaming
    var_type: str  # "quantum", "classical", etc.
    size: int | None = None
    is_array: bool = False
    is_unpacked: bool = False
    unpacked_names: list[str] = field(default_factory=list)
    state: ResourceState = ResourceState.AVAILABLE
    is_struct: bool = False
    struct_info: dict | None = None
    is_struct_field: bool = False
    struct_name: str | None = None
    field_name: str | None = None


@dataclass
class ScopeContext:
    """Context for a scope (function, block, etc.)."""

    parent: ScopeContext | None = None
    variables: dict[str, VariableInfo] = field(default_factory=dict)
    unpacked_arrays: dict[str, list[str]] = field(default_factory=dict)
    consumed_resources: set[str] = field(default_factory=set)
    refreshed_arrays: dict[str, str] = field(
        default_factory=dict,
    )  # original_name -> fresh_name

    def lookup_variable(self, name: str) -> VariableInfo | None:
        """Look up a variable in this scope or parent scopes."""
        if name in self.variables:
            return self.variables[name]

        # Check if this variable was refreshed by a function call
        if name in self.refreshed_arrays:
            fresh_name = self.refreshed_arrays[name]
            if fresh_name in self.variables:
                return self.variables[fresh_name]

        if self.parent:
            return self.parent.lookup_variable(name)
        return None

    def add_variable(self, var_info: VariableInfo) -> None:
        """Add a variable to this scope."""
        self.variables[var_info.name] = var_info

    def mark_consumed(self, name: str) -> None:
        """Mark a resource as consumed."""
        self.consumed_resources.add(name)
        var = self.lookup_variable(name)
        if var:
            var.state = ResourceState.CONSUMED


class IRNode(ABC):
    """Base class for all IR nodes."""

    @abstractmethod
    def analyze(self, context: ScopeContext) -> None:
        """Analyze this node for resource usage, unpacking needs, etc."""

    @abstractmethod
    def render(self, context: ScopeContext) -> list[str]:
        """Render this node to Guppy code lines."""


@dataclass
class ArrayAccess(IRNode):
    """Represents array[index] access."""

    array_name: str = None  # Optional for backwards compatibility
    array: IRNode = None  # Can be a FieldAccess for struct.field[index]
    index: int | str | IRNode = None
    force_array_syntax: bool = False  # If True, never use unpacked names

    def __post_init__(self):
        """Initialize ArrayAccess, supporting both old and new API."""
        # Support both old and new API
        if self.array_name and not self.array:
            self.array = VariableRef(self.array_name)

    def analyze(self, context: ScopeContext) -> None:
        """Mark that this array needs element access."""
        if self.array:
            self.array.analyze(context)

    def render(self, context: ScopeContext) -> list[str]:
        """Render array access, using unpacked name if available."""
        # Handle old API
        if self.array_name and not self.force_array_syntax:
            var = context.lookup_variable(self.array_name)
            if (
                var
                and var.is_unpacked
                and isinstance(self.index, int)
                and self.index < len(var.unpacked_names)
            ):
                return [var.unpacked_names[self.index]]

        # Render array if it's an IRNode (e.g., FieldAccess)
        if self.array:
            array_code = self.array.render(context)
            array_str = array_code[0] if len(array_code) == 1 else "???"
        else:
            array_str = self.array_name

        # Render index if it's an IRNode
        if isinstance(self.index, IRNode):
            index_code = self.index.render(context)
            if len(index_code) == 1:
                return [f"{array_str}[{index_code[0]}]"]
            # Complex index expression - shouldn't happen usually
            return [f"{array_str}[{' '.join(index_code)}]"]

        return [f"{array_str}[{self.index}]"]


@dataclass
class FieldAccess(IRNode):
    """Access to a struct field: obj.field"""

    obj: IRNode
    field: str

    def analyze(self, context: ScopeContext) -> None:
        """Analyze the object being accessed."""
        self.obj.analyze(context)

    def render(self, context: ScopeContext) -> list[str]:
        """Render field access as obj.field."""
        obj_code = self.obj.render(context)
        obj_str = obj_code[0] if len(obj_code) == 1 else "???"
        return [f"{obj_str}.{self.field}"]


@dataclass
class VariableRef(IRNode):
    """Reference to a variable."""

    name: str

    def analyze(self, context: ScopeContext) -> None:
        """Check variable exists."""
        # Nothing to analyze for a simple variable reference

    def render(self, context: ScopeContext) -> list[str]:
        """Render variable reference."""
        var = context.lookup_variable(self.name)

        if var:
            return [var.name]  # Use potentially renamed name
        return [self.name]


@dataclass
class Literal(IRNode):
    """Literal value."""

    value: Any

    def analyze(self, context: ScopeContext) -> None:
        """Analyze literal - no-op as literals don't need analysis."""
        # Nothing to analyze for literals

    def render(self, context: ScopeContext) -> list[str]:
        _ = context  # Context not needed for literal rendering
        if isinstance(self.value, bool):
            return ["True" if self.value else "False"]
        if isinstance(self.value, str):
            return [f'"{self.value}"']
        return [str(self.value)]


@dataclass
class Statement(IRNode):
    """Base class for statements."""


@dataclass
class Expression(IRNode):
    """Base class for expressions."""


@dataclass
class BinaryOp(Expression):
    """Binary operation: left op right."""

    left: IRNode
    op: str
    right: IRNode
    needs_parens: bool = False  # Track if this expression needs parentheses

    # Operator precedence (higher number = higher precedence)
    PRECEDENCE: ClassVar[dict[str, int]] = {
        "or": 1,
        "|": 1,
        "and": 2,
        "&": 2,
        "^": 3,
        "==": 4,
        "!=": 4,
        "<": 4,
        ">": 4,
        "<=": 4,
        ">=": 4,
        "+": 5,
        "-": 5,
        "*": 6,
        "/": 6,
        "//": 6,
        "%": 6,
        "**": 7,
    }

    def analyze(self, context: ScopeContext) -> None:
        """Analyze both operands."""
        self.left.analyze(context)
        self.right.analyze(context)

    def _needs_parens(self, child: IRNode, *, is_right: bool = False) -> bool:
        """Check if child expression needs parentheses."""
        if not isinstance(child, BinaryOp):
            return False

        child_prec = self.PRECEDENCE.get(child.op, 10)
        self_prec = self.PRECEDENCE.get(self.op, 10)

        # Lower precedence needs parens
        if child_prec < self_prec:
            return True
        # Same precedence: check associativity (left-to-right)
        # For operators like -, /, we need parens on the right
        return child_prec == self_prec and is_right and self.op in ["-", "/", "//", "%"]

    def render(self, context: ScopeContext) -> list[str]:
        """Render binary operation with proper precedence handling."""
        # Render children
        left_code = self.left.render(context)
        right_code = self.right.render(context)

        # Add parentheses if needed for children
        left_str = left_code[0] if len(left_code) == 1 else " ".join(left_code)
        right_str = right_code[0] if len(right_code) == 1 else " ".join(right_code)

        if self._needs_parens(self.left):
            left_str = f"({left_str})"
        if self._needs_parens(self.right, is_right=True):
            right_str = f"({right_str})"

        result = f"{left_str} {self.op} {right_str}"

        # Add parentheses if this expression was marked as needing them
        if self.needs_parens:
            result = f"({result})"

        return [result]


@dataclass
class UnaryOp(Expression):
    """Unary operation: op operand."""

    op: str
    operand: IRNode

    def analyze(self, context: ScopeContext) -> None:
        """Analyze the operand."""
        self.operand.analyze(context)

    def render(self, context: ScopeContext) -> list[str]:
        """Render unary operation."""
        operand_code = self.operand.render(context)
        if len(operand_code) == 1:
            return [f"{self.op} {operand_code[0]}"]
        # For complex expressions, use parentheses
        return [f"{self.op} ({' '.join(operand_code)})"]


@dataclass
class Assignment(Statement):
    """Assignment statement: target = value."""

    target: IRNode
    value: IRNode

    def analyze(self, context: ScopeContext) -> None:
        """Analyze both target and value."""
        self.target.analyze(context)
        self.value.analyze(context)

    def render(self, context: ScopeContext) -> list[str]:
        target_code = self.target.render(context)
        value_code = self.value.render(context)
        if len(target_code) == 1 and len(value_code) == 1:
            return [f"{target_code[0]} = {value_code[0]}"]
        # Handle multi-line expressions
        result = value_code[:-1]  # All but last line
        result.append(f"{target_code[0]} = {value_code[-1]}")
        return result


@dataclass
class FunctionCall(Expression):
    """Function call expression."""

    func_name: str
    args: list[IRNode]

    def analyze(self, context: ScopeContext) -> None:
        for arg in self.args:
            arg.analyze(context)

    def render(self, context: ScopeContext) -> list[str]:
        arg_strs = []
        for arg in self.args:
            arg_code = arg.render(context)
            arg_strs.append(arg_code[0] if len(arg_code) == 1 else "???")
        return [f"{self.func_name}({', '.join(arg_strs)})"]


@dataclass
class MethodCall(Expression):
    """Method call: obj.method(args)."""

    obj: IRNode
    method: str
    args: list[IRNode]

    def analyze(self, context: ScopeContext) -> None:
        self.obj.analyze(context)
        for arg in self.args:
            arg.analyze(context)

    def render(self, context: ScopeContext) -> list[str]:
        obj_code = self.obj.render(context)
        arg_strs = []
        for arg in self.args:
            arg_code = arg.render(context)
            arg_strs.append(arg_code[0] if len(arg_code) == 1 else "???")

        obj_str = obj_code[0] if len(obj_code) == 1 else "???"
        return [f"{obj_str}.{self.method}({', '.join(arg_strs)})"]


@dataclass
class Measurement(Statement):
    """Measurement operation."""

    qubit: IRNode
    target: IRNode | None = None

    def analyze(self, context: ScopeContext) -> None:
        self.qubit.analyze(context)
        if self.target:
            self.target.analyze(context)

        # Mark qubit as consumed if it's a simple reference
        if isinstance(self.qubit, VariableRef):
            context.mark_consumed(self.qubit.name)
        elif isinstance(self.qubit, ArrayAccess):
            # Track that this array element is consumed
            pass

    def render(self, context: ScopeContext) -> list[str]:
        qubit_code = self.qubit.render(context)
        qubit_str = qubit_code[0] if len(qubit_code) == 1 else "???"

        if self.target:
            target_code = self.target.render(context)
            target_str = target_code[0] if len(target_code) == 1 else "???"
            return [f"{target_str} = quantum.measure({qubit_str})"]
        return [f"quantum.measure({qubit_str})"]


@dataclass
class ArrayUnpack(Statement):
    """Array unpacking: a, b, c = array."""

    targets: list[str]
    source: str

    def analyze(self, context: ScopeContext) -> None:
        # Mark the array as unpacked
        var = context.lookup_variable(self.source)
        if var:
            var.is_unpacked = True
            var.unpacked_names = self.targets

    def render(self, context: ScopeContext) -> list[str]:
        _ = context  # Context not needed for unpacking
        if len(self.targets) == 1:
            # Special syntax for single element
            return [f"{self.targets[0]}, = {self.source}"]
        return [f"{', '.join(self.targets)} = {self.source}"]


@dataclass
class Comment(Statement):
    """Comment line."""

    text: str

    def analyze(self, context: ScopeContext) -> None:
        """Analyze comment - no-op."""
        # Nothing to analyze for comments

    def render(self, context: ScopeContext) -> list[str]:
        """Render comment line."""
        _ = context  # Context not needed for comments
        if self.text:
            return [f"# {self.text}"]
        return []  # Don't render empty comments


@dataclass
class ReturnStatement(Statement):
    """Return statement."""

    value: IRNode | None = None

    def analyze(self, context: ScopeContext) -> None:
        """Analyze return value if present."""
        if self.value:
            self.value.analyze(context)

    def render(self, context: ScopeContext) -> list[str]:
        """Render return statement."""
        if self.value:
            value_code = self.value.render(context)
            return [f"return {value_code[0]}"]
        return ["return"]


@dataclass
class TupleExpression(Expression):
    """Tuple expression for multiple returns."""

    elements: list[IRNode]

    def analyze(self, context: ScopeContext) -> None:
        """Analyze all elements."""
        for elem in self.elements:
            elem.analyze(context)

    def render(self, context: ScopeContext) -> list[str]:
        """Render tuple expression."""
        elem_codes = [elem.render(context)[0] for elem in self.elements]
        return [", ".join(elem_codes)]  # No parentheses needed for tuple returns


@dataclass
class Block(IRNode):
    """Block of statements."""

    statements: list[Statement] = field(default_factory=list)

    def analyze(self, context: ScopeContext) -> None:
        for stmt in self.statements:
            stmt.analyze(context)

    def render(self, context: ScopeContext) -> list[str]:
        lines = []
        for stmt in self.statements:
            lines.extend(stmt.render(context))
        return lines


@dataclass
class IfStatement(Statement):
    """If statement with optional else."""

    condition: IRNode
    then_block: Block
    else_block: Block | None = None

    def analyze(self, context: ScopeContext) -> None:
        self.condition.analyze(context)

        # Create new scope for then block
        then_context = ScopeContext(parent=context)
        self.then_block.analyze(then_context)

        if self.else_block:
            # Create new scope for else block
            else_context = ScopeContext(parent=context)
            self.else_block.analyze(else_context)

    def render(self, context: ScopeContext) -> list[str]:
        lines = []

        # Render condition
        cond_code = self.condition.render(context)
        cond_str = cond_code[0] if len(cond_code) == 1 else "???"
        lines.append(f"if {cond_str}:")

        # Render then block (indented)
        then_lines = self.then_block.render(context)
        if then_lines:
            lines.extend(f"    {line}" for line in then_lines)
        else:
            lines.append("    pass")

        # Render else block if present
        if self.else_block:
            lines.append("else:")
            else_lines = self.else_block.render(context)
            if else_lines:
                lines.extend(f"    {line}" for line in else_lines)
            else:
                lines.append("    pass")

        return lines


@dataclass
class WhileStatement(Statement):
    """While loop statement."""

    condition: IRNode
    body: Block

    def analyze(self, context: ScopeContext) -> None:
        self.condition.analyze(context)
        # Create new scope for loop body
        loop_context = ScopeContext(parent=context)
        self.body.analyze(loop_context)

    def render(self, context: ScopeContext) -> list[str]:
        lines = []

        # Render condition
        cond_code = self.condition.render(context)
        cond_str = cond_code[0] if len(cond_code) == 1 else "???"
        lines.append(f"while {cond_str}:")

        # Render body (indented)
        body_lines = self.body.render(context)
        if body_lines:
            lines.extend(f"    {line}" for line in body_lines)
        else:
            lines.append("    pass")

        return lines


@dataclass
class ForStatement(Statement):
    """For loop statement."""

    loop_var: str
    iterable: IRNode
    body: Block

    def analyze(self, context: ScopeContext) -> None:
        # Analyze iterable
        self.iterable.analyze(context)

        # Create new scope for loop body with loop variable
        loop_context = ScopeContext(parent=context)
        # Add loop variable to context (simplified - would need type info)
        self.body.analyze(loop_context)

    def render(self, context: ScopeContext) -> list[str]:
        lines = []

        # Render iterable
        iter_code = self.iterable.render(context)
        iter_str = iter_code[0] if len(iter_code) == 1 else "???"
        lines.append(f"for {self.loop_var} in {iter_str}:")

        # Render body (indented)
        body_lines = self.body.render(context)
        if body_lines:
            lines.extend(f"    {line}" for line in body_lines)
        else:
            lines.append("    pass")

        return lines


@dataclass
class Function(IRNode):
    """Function definition."""

    name: str
    params: list[tuple[str, str]]  # [(name, type), ...]
    return_type: str
    body: Block
    decorators: list[str] = field(default_factory=list)

    def analyze(self, context: ScopeContext) -> None:
        # Create new scope for function
        func_context = ScopeContext(parent=context)

        # Add parameters to scope
        for param_name, param_type in self.params:
            var_info = VariableInfo(
                name=param_name,
                original_name=param_name,
                var_type=param_type,
            )
            func_context.add_variable(var_info)

        # Analyze body
        self.body.analyze(func_context)

    def render(self, context: ScopeContext) -> list[str]:
        lines = []

        # Decorators
        lines.extend(f"@{decorator}" for decorator in self.decorators)

        # Function signature
        param_strs = []
        for name, ptype in self.params:
            param_strs.append(f"{name}: {ptype}")

        lines.append(f"def {self.name}({', '.join(param_strs)}) -> {self.return_type}:")

        # Body
        body_lines = self.body.render(context)
        if body_lines:
            lines.extend(f"    {line}" for line in body_lines)
        else:
            lines.append("    pass")

        return lines


@dataclass
class Module(IRNode):
    """Module containing imports and definitions."""

    imports: list[str] = field(default_factory=list)
    functions: list[Function] = field(default_factory=list)
    refreshed_arrays: dict[str, set[str]] = field(
        default_factory=dict,
    )  # function_name -> set of refreshed array names

    def analyze(self, context: ScopeContext) -> None:
        for func in self.functions:
            func.analyze(context)

    def render(self, context: ScopeContext) -> list[str]:
        lines = []

        # Imports
        lines.extend(self.imports)

        if self.imports and self.functions:
            lines.append("")  # Blank line between imports and code

        # Functions
        for i, func in enumerate(self.functions):
            if i > 0:
                lines.append("")  # Blank line between functions
            lines.extend(func.render(context))

        return lines
