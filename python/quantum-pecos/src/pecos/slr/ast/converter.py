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

"""Converter from SLR representation to AST representation.

This module provides utilities to convert SLR programs (Main, Block, etc.)
into the structured AST representation for analysis and transformation.

Example:
    from pecos.slr import Main, QReg
    from pecos.slr.qeclib import qubit as qb
    from pecos.slr.ast import SLRToAST

    prog = Main(
        q := QReg("q", 2),
        qb.Prep(q[0]),
        qb.H(q[0]),
    )

    converter = SLRToAST()
    ast = converter.convert(prog)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pecos.slr.ast.nodes import (
    AllocatorDecl,
    ArrayTypeExpr,
    AssignOp,
    BarrierOp,
    BinaryExpr,
    BinaryOp,
    BitExpr,
    BitRef,
    BitTypeExpr,
    CommentOp,
    Expression,
    ForStmt,
    GateKind,
    GateOp,
    IfStmt,
    LiteralExpr,
    MeasureOp,
    ParallelBlock,
    PermuteOp,
    PrepareOp,
    Program,
    QubitTypeExpr,
    RegisterDecl,
    RepeatStmt,
    ReturnOp,
    SlotRef,
    Statement,
    TypeExpr,
    UnaryExpr,
    UnaryOp,
    VarExpr,
    WhileStmt,
)

if TYPE_CHECKING:
    from pecos.slr.block import Block
    from pecos.slr.main import Main


# Mapping from SLR gate class names to AST GateKind
GATE_KIND_MAP: dict[str, GateKind] = {
    # Single-qubit Paulis
    "X": GateKind.X,
    "Y": GateKind.Y,
    "Z": GateKind.Z,
    # Hadamard
    "H": GateKind.H,
    # Phase gates
    "S": GateKind.S,
    "Sdg": GateKind.Sdg,
    "T": GateKind.T,
    "Tdg": GateKind.Tdg,
    # Square root gates
    "SX": GateKind.SX,
    "SY": GateKind.SY,
    "SZ": GateKind.SZ,
    "SXdg": GateKind.SXdg,
    "SYdg": GateKind.SYdg,
    "SZdg": GateKind.SZdg,
    # Rotation gates
    "RX": GateKind.RX,
    "RY": GateKind.RY,
    "RZ": GateKind.RZ,
    # Two-qubit gates
    "CX": GateKind.CX,
    "CNOT": GateKind.CX,  # Alias
    "CY": GateKind.CY,
    "CZ": GateKind.CZ,
    "CH": GateKind.CH,
    # Two-qubit rotation gates
    "SXX": GateKind.SXX,
    "SYY": GateKind.SYY,
    "SZZ": GateKind.SZZ,
    "SXXdg": GateKind.SXXdg,
    "SYYdg": GateKind.SYYdg,
    "SZZdg": GateKind.SZZdg,
    "RZZ": GateKind.RZZ,
    # Face rotations
    "F": GateKind.F,
    "Fdg": GateKind.Fdg,
    "F4": GateKind.F4,
    "F4dg": GateKind.F4dg,
}

# Mapping from SLR BinOp class names to AST BinaryOp
BINARY_OP_MAP: dict[str, BinaryOp] = {
    "PLUS": BinaryOp.ADD,
    "MINUS": BinaryOp.SUB,
    "MUL": BinaryOp.MUL,
    "DIV": BinaryOp.DIV,
    "EQUIV": BinaryOp.EQ,
    "NEQUIV": BinaryOp.NE,
    "LT": BinaryOp.LT,
    "LE": BinaryOp.LE,
    "GT": BinaryOp.GT,
    "GE": BinaryOp.GE,
    "AND": BinaryOp.AND,
    "OR": BinaryOp.OR,
    "XOR": BinaryOp.XOR,
    "LSHIFT": BinaryOp.LSHIFT,
    "RSHIFT": BinaryOp.RSHIFT,
}

# Mapping from SLR UnaryOp class names to AST UnaryOp
UNARY_OP_MAP: dict[str, UnaryOp] = {
    "NOT": UnaryOp.NOT,
    "NEG": UnaryOp.NEG,
}


class SlrToAst:
    """Converter from SLR representation to AST.

    Converts SLR Main/Block programs into the structured AST format.
    """

    def __init__(self) -> None:
        """Initialize the converter."""
        self._position = 0  # Track position for source locations

    def convert(self, block: Main | Block) -> Program:
        """Convert an SLR Main/Block to an AST Program.

        Args:
            block: The SLR block to convert.

        Returns:
            An AST Program node.
        """
        self._position = 0

        # Get the block name
        name = getattr(block, "block_name", block.__class__.__name__)

        # Convert declarations from vars
        declarations = self._convert_declarations(block)

        # Convert body statements
        body = self._convert_statements(block.ops)

        # Convert return types if present
        returns = self._convert_return_types(block)

        # Check for base allocator (for new-style allocator-based blocks)
        allocator = None
        for var in block.vars:
            if hasattr(var, "_capacity"):  # QAlloc detection
                allocator = AllocatorDecl(
                    name=var.name,
                    capacity=var._capacity,
                    parent=var._parent.name if var._parent else None,
                )
                break

        return Program(
            name=name,
            declarations=tuple(declarations),
            body=tuple(body),
            returns=returns,
            allocator=allocator,
        )

    def _convert_declarations(self, block: Block) -> list:
        """Convert block variables to AST declarations.

        This scans both block.vars (for traditional QReg/CReg declarations)
        and block.ops (for QAllocs created with walrus operator).
        """
        declarations = []
        seen_names: set[str] = set()

        # First, scan block.vars for declarations
        for var in block.vars:
            decl = self._convert_var_to_declaration(var)
            if decl is not None:
                declarations.append(decl)
                seen_names.add(decl.name)

        # Also scan block.ops for QAllocs (they end up in ops when using walrus operator)
        for op in block.ops:
            if op.__class__.__name__ == "QAlloc":
                # Only add if not already seen (avoid duplicates)
                if op.name not in seen_names:
                    decl = self._convert_var_to_declaration(op)
                    if decl is not None:
                        declarations.append(decl)
                        seen_names.add(decl.name)

        return declarations

    def _convert_return_types(self, block: Block) -> tuple[TypeExpr, ...]:
        """Convert block return type annotations to AST TypeExpr nodes.

        Args:
            block: The SLR block to extract return types from.

        Returns:
            Tuple of AST TypeExpr nodes representing return types.
        """
        # Import here to avoid circular imports
        from pecos.slr.types import ArrayType, ReturnNotSet

        # Check if block has return type annotation
        block_returns = getattr(block.__class__, "block_returns", ReturnNotSet)
        if block_returns is ReturnNotSet:
            return ()

        # Handle None (explicitly no return)
        if block_returns is None:
            return ()

        # Convert single return type
        if isinstance(block_returns, ArrayType):
            return (self._convert_array_type(block_returns),)

        # Convert tuple of return types
        if isinstance(block_returns, tuple):
            return tuple(self._convert_array_type(rt) for rt in block_returns)

        return ()

    def _convert_array_type(self, array_type: Any) -> ArrayTypeExpr:
        """Convert an SLR ArrayType to an AST ArrayTypeExpr.

        Args:
            array_type: The SLR ArrayType to convert.

        Returns:
            An AST ArrayTypeExpr node.
        """
        # Determine element type based on elem_type.name
        elem_name = array_type.elem_type.name
        if elem_name == "Qubit":
            element = QubitTypeExpr()
        elif elem_name == "Bit":
            element = BitTypeExpr()
        else:
            # Default to Qubit for unknown types
            element = QubitTypeExpr()

        return ArrayTypeExpr(element=element, size=array_type.size)

    def _convert_var_to_declaration(self, var: Any):
        """Convert a single variable to an AST declaration."""
        var_class = var.__class__.__name__

        if var_class == "QReg":
            # QReg maps to AllocatorDecl
            return AllocatorDecl(name=var.sym, capacity=var.size)

        if var_class == "CReg":
            # CReg maps to RegisterDecl
            return RegisterDecl(
                name=var.sym,
                size=var.size,
                is_result=getattr(var, "result", True),
            )

        if var_class == "QAlloc":
            # QAlloc maps to AllocatorDecl
            return AllocatorDecl(
                name=var.name,
                capacity=var._capacity,
                parent=var._parent.name if var._parent else None,
            )

        # Unknown variable type - skip
        return None

    def _convert_statements(self, ops: list) -> list[Statement]:
        """Convert a list of SLR operations to AST statements."""
        statements = []
        for op in ops:
            stmt = self._convert_statement(op)
            if stmt is not None:
                statements.append(stmt)
        return statements

    def _convert_statement(self, op: Any) -> Statement | None:
        """Convert a single SLR operation to an AST statement."""
        op_class = op.__class__.__name__

        # Gate operations
        if hasattr(op, "is_qgate") and op.is_qgate:
            return self._convert_gate(op)

        # Control flow
        if op_class == "If":
            return self._convert_if(op)

        if op_class == "While":
            return self._convert_while(op)

        if op_class == "For":
            return self._convert_for(op)

        if op_class == "Repeat":
            return self._convert_repeat(op)

        if op_class == "Parallel":
            return self._convert_parallel(op)

        # Misc statements
        if op_class == "Barrier":
            return self._convert_barrier(op)

        if op_class == "Comment":
            return CommentOp(text=op.txt)

        if op_class == "Return":
            return self._convert_return(op)

        if op_class == "Permute":
            return self._convert_permute(op)

        # Assignment operations
        if op_class == "SET":
            return self._convert_assignment(op)

        # Nested blocks
        if hasattr(op, "ops"):
            # This is a nested block - convert its statements
            # For now, we flatten it
            stmts = self._convert_statements(op.ops)
            if len(stmts) == 1:
                return stmts[0]
            # Multiple statements - wrap in parallel? Or just return first?
            # For now, return None and handle nested blocks elsewhere
            return None

        return None

    def _convert_gate(self, gate: Any) -> Statement:
        """Convert an SLR gate to an AST GateOp, PrepareOp, or MeasureOp."""
        gate_name = gate.sym

        # Handle special operations
        if gate_name == "Prep":
            return self._convert_prep(gate)

        if gate_name == "Measure":
            return self._convert_measure(gate)

        # Regular gate
        gate_kind = GATE_KIND_MAP.get(gate_name)
        if gate_kind is None:
            # Unknown gate - use a fallback or raise
            msg = f"Unknown gate type: {gate_name}"
            raise ValueError(msg)

        # Convert targets
        targets = tuple(self._convert_qubit_ref(q) for q in gate.qargs)

        # Convert parameters if present
        params: tuple = ()
        if gate.params:
            params = tuple(self._convert_expression(p) for p in gate.params)

        return GateOp(gate=gate_kind, targets=targets, params=params)

    def _convert_prep(self, gate: Any) -> PrepareOp:
        """Convert an SLR Prep gate to an AST PrepareOp."""
        if not gate.qargs:
            msg = "Prep gate has no qubit arguments"
            raise ValueError(msg)

        # Get allocator name and slot indices
        first_qubit = gate.qargs[0]
        allocator = first_qubit.reg.sym if hasattr(first_qubit.reg, "sym") else str(first_qubit.reg)

        slots = tuple(q.index for q in gate.qargs)

        return PrepareOp(allocator=allocator, slots=slots)

    def _convert_measure(self, gate: Any) -> MeasureOp:
        """Convert an SLR Measure gate to an AST MeasureOp."""
        targets = tuple(self._convert_qubit_ref(q) for q in gate.qargs)

        # Convert classical output bits if present
        results: tuple = ()
        if gate.cout:
            results = tuple(self._convert_bit_ref(b) for b in gate.cout)

        return MeasureOp(targets=targets, results=results)

    def _convert_qubit_ref(self, qubit: Any) -> SlotRef:
        """Convert an SLR Qubit/QubitRef to an AST SlotRef."""
        # Handle both old Qubit (has .reg) and new QubitRef (has .alloc)
        if hasattr(qubit, "reg"):
            allocator = qubit.reg.sym if hasattr(qubit.reg, "sym") else str(qubit.reg)
        elif hasattr(qubit, "alloc"):
            allocator = qubit.alloc.name
        else:
            allocator = str(qubit)

        return SlotRef(allocator=allocator, index=qubit.index)

    def _convert_bit_ref(self, bit: Any) -> BitRef:
        """Convert an SLR Bit to an AST BitRef."""
        register = bit.reg.sym if hasattr(bit.reg, "sym") else str(bit.reg)
        return BitRef(register=register, index=bit.index)

    def _convert_if(self, op: Any) -> IfStmt:
        """Convert an SLR If block to an AST IfStmt."""
        condition = self._convert_expression(op.cond)
        then_body = tuple(self._convert_statements(op.ops))

        else_body: tuple = ()
        if op.else_block is not None:
            else_body = tuple(self._convert_statements(op.else_block.ops))

        return IfStmt(condition=condition, then_body=then_body, else_body=else_body)

    def _convert_while(self, op: Any) -> WhileStmt:
        """Convert an SLR While block to an AST WhileStmt."""
        condition = self._convert_expression(op.cond)
        body = tuple(self._convert_statements(op.ops))

        return WhileStmt(condition=condition, body=body)

    def _convert_for(self, op: Any) -> ForStmt:
        """Convert an SLR For block to an AST ForStmt."""
        variable = str(op.var) if not isinstance(op.var, str) else op.var

        # Handle range-based or explicit start/stop
        if op.iterable is not None:
            # Range object
            if isinstance(op.iterable, range):
                start = LiteralExpr(value=op.iterable.start)
                stop = LiteralExpr(value=op.iterable.stop)
                step_val = op.iterable.step
                step = LiteralExpr(value=step_val) if step_val != 1 else None
            else:
                # Some other iterable - convert bounds
                start = LiteralExpr(value=0)
                stop = LiteralExpr(value=len(op.iterable))
                step = None
        else:
            start = self._convert_expression(op.start)
            stop = self._convert_expression(op.stop)
            step = self._convert_expression(op.step) if op.step is not None and op.step != 1 else None

        body = tuple(self._convert_statements(op.ops))

        return ForStmt(variable=variable, start=start, stop=stop, step=step, body=body)

    def _convert_repeat(self, op: Any) -> RepeatStmt:
        """Convert an SLR Repeat block to an AST RepeatStmt."""
        count = op.cond  # Repeat uses cond to store the count
        body = tuple(self._convert_statements(op.ops))

        return RepeatStmt(count=count, body=body)

    def _convert_parallel(self, op: Any) -> ParallelBlock:
        """Convert an SLR Parallel block to an AST ParallelBlock."""
        body = tuple(self._convert_statements(op.ops))
        return ParallelBlock(body=body)

    def _convert_barrier(self, op: Any) -> BarrierOp:
        """Convert an SLR Barrier to an AST BarrierOp."""
        allocators: tuple = ()
        if op.qregs:
            allocators = tuple(
                q.sym if hasattr(q, "sym") else str(q)
                for q in op.qregs
                if not hasattr(q, "index")  # Skip individual qubits
            )
        return BarrierOp(allocators=allocators)

    def _convert_return(self, op: Any) -> ReturnOp:
        """Convert an SLR Return to an AST ReturnOp."""
        values: list = []
        for var in op.return_vars:
            if isinstance(var, str):
                values.append(var)
            elif hasattr(var, "sym"):
                values.append(var.sym)
            else:
                # Try to convert as expression
                values.append(self._convert_expression(var))

        return ReturnOp(values=tuple(values))

    def _convert_permute(self, op: Any) -> PermuteOp:
        """Convert an SLR Permute to an AST PermuteOp."""
        # Extract register/allocator names from sources (elems_i)
        sources: list[str] = []
        elems_i = op.elems_i
        if hasattr(elems_i, "__iter__") and not isinstance(elems_i, str):
            for elem in elems_i:
                if hasattr(elem, "sym"):
                    sources.append(elem.sym)
                elif hasattr(elem, "name"):
                    sources.append(elem.name)
                else:
                    sources.append(str(elem))
        elif hasattr(elems_i, "sym"):
            sources.append(elems_i.sym)
        elif hasattr(elems_i, "name"):
            sources.append(elems_i.name)
        else:
            sources.append(str(elems_i))

        # Extract register/allocator names from targets (elems_f)
        targets: list[str] = []
        elems_f = op.elems_f
        if hasattr(elems_f, "__iter__") and not isinstance(elems_f, str):
            for elem in elems_f:
                if hasattr(elem, "sym"):
                    targets.append(elem.sym)
                elif hasattr(elem, "name"):
                    targets.append(elem.name)
                else:
                    targets.append(str(elem))
        elif hasattr(elems_f, "sym"):
            targets.append(elems_f.sym)
        elif hasattr(elems_f, "name"):
            targets.append(elems_f.name)
        else:
            targets.append(str(elems_f))

        add_comment = getattr(op, "comment", True)

        return PermuteOp(
            sources=tuple(sources),
            targets=tuple(targets),
            add_comment=add_comment,
        )

    def _convert_assignment(self, op: Any) -> AssignOp:
        """Convert an SLR SET operation to an AST AssignOp."""
        # Target
        target = op.left
        if hasattr(target, "reg") and hasattr(target, "index"):
            # It's a Bit reference
            target = self._convert_bit_ref(target)
        elif hasattr(target, "sym"):
            target = target.sym
        else:
            target = str(target)

        # Value
        value = self._convert_expression(op.right)

        return AssignOp(target=target, value=value)

    def _convert_expression(self, expr: Any) -> Expression:
        """Convert an SLR expression to an AST Expression."""
        if expr is None:
            return LiteralExpr(value=0)

        # Literal values
        if isinstance(expr, bool | int | float):
            return LiteralExpr(value=expr)

        expr_class = expr.__class__.__name__

        # Binary operations
        if expr_class in BINARY_OP_MAP:
            return BinaryExpr(
                op=BINARY_OP_MAP[expr_class],
                left=self._convert_expression(expr.left),
                right=self._convert_expression(expr.right),
            )

        # Unary operations
        if expr_class in UNARY_OP_MAP:
            return UnaryExpr(
                op=UNARY_OP_MAP[expr_class],
                operand=self._convert_expression(expr.value),
            )

        # Bit reference as expression
        if hasattr(expr, "reg") and hasattr(expr, "index"):
            if expr.__class__.__name__ == "Bit":
                return BitExpr(ref=self._convert_bit_ref(expr))

        # Variable reference
        if hasattr(expr, "sym"):
            return VarExpr(name=expr.sym)

        # String variable name
        if isinstance(expr, str):
            return VarExpr(name=expr)

        # Tuple handling for parameter access
        if isinstance(expr, tuple):
            if len(expr) == 1:
                return self._convert_expression(expr[0])
            # Multiple values - return first for now
            return self._convert_expression(expr[0])

        # Fallback - try to get a value
        if hasattr(expr, "value"):
            return LiteralExpr(value=expr.value)

        # Unknown expression type - convert to string var
        return VarExpr(name=str(expr))


def slr_to_ast(block: Main | Block) -> Program:
    """Convert an SLR Main/Block to an AST Program.

    Convenience function for simple conversions.

    Args:
        block: The SLR block to convert.

    Returns:
        An AST Program node.
    """
    converter = SlrToAst()
    return converter.convert(block)
