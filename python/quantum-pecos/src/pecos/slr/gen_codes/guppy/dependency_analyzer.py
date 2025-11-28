"""Dependency analyzer for SLR blocks."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pecos.slr import Block


@dataclass
class BlockDependency:
    """Represents dependencies for a block."""

    block_type: str
    constructor_params: dict[str, Any]  # parameter name -> value/type
    used_variables: set[str]  # Set of variable names used in operations
    nested_blocks: list[BlockDependency]  # Dependencies of nested blocks


class DependencyAnalyzer:
    """Analyzes SLR blocks to determine parameter dependencies."""

    def __init__(self):
        self.analyzed_blocks = {}  # Cache of analyzed block types

    def analyze_block(self, block: Block) -> BlockDependency:
        """Analyze a block to determine its dependencies."""
        block_type = type(block).__name__

        # Get constructor parameters
        constructor_params = self._get_constructor_params(block)

        # Find used variables in operations
        used_variables = set()
        nested_blocks = []

        if hasattr(block, "ops"):
            for op in block.ops:
                # Collect variables from operations
                self._collect_variables_from_op(op, used_variables)

                # If it's a nested block, analyze it too
                if hasattr(op, "ops") and hasattr(op, "vars"):
                    nested_dep = self.analyze_block(op)
                    nested_blocks.append(nested_dep)
                    # Add nested block's used variables
                    used_variables.update(nested_dep.used_variables)

        return BlockDependency(
            block_type=block_type,
            constructor_params=constructor_params,
            used_variables=used_variables,
            nested_blocks=nested_blocks,
        )

    def _get_constructor_params(self, block: Block) -> dict[str, Any]:
        """Extract constructor parameters from a block instance."""
        params = {}

        # Get the constructor signature
        sig = inspect.signature(type(block).__init__)

        # Try to match parameters with instance attributes
        for param_name in sig.parameters:
            if param_name == "self":
                continue

            # Common patterns for how parameters are stored
            if hasattr(block, param_name):
                params[param_name] = getattr(block, param_name)
            elif hasattr(block, f"_{param_name}"):
                params[param_name] = getattr(block, f"_{param_name}")
            # Try to infer from operations
            elif param_name in ["data", "qubits", "q"]:
                # Look for quantum registers
                params[param_name] = self._find_qreg_in_ops(block)
            elif param_name in ["ancilla", "a"]:
                # Look for ancilla qubits
                params[param_name] = self._find_ancilla_in_ops(block)
            elif param_name in ["init_bit", "init", "bit", "c"]:
                # Look for classical bits
                params[param_name] = self._find_bit_in_ops(block)

        return params

    def _collect_variables_from_op(self, op, used_vars: set[str]):
        """Collect variable names used in an operation."""
        # Check quantum arguments
        if hasattr(op, "qargs"):
            for qarg in op.qargs:
                if hasattr(qarg, "reg") and hasattr(qarg.reg, "sym"):
                    used_vars.add(qarg.reg.sym)
                elif hasattr(qarg, "sym"):
                    # Direct QReg object
                    used_vars.add(qarg.sym)
                elif isinstance(qarg, tuple):
                    # Handle tuples of qubits
                    for q in qarg:
                        if hasattr(q, "reg") and hasattr(q.reg, "sym"):
                            used_vars.add(q.reg.sym)

        # Check classical arguments
        if hasattr(op, "cargs"):
            for carg in op.cargs:
                if hasattr(carg, "reg") and hasattr(carg.reg, "sym"):
                    used_vars.add(carg.reg.sym)

        # Check output bits (for measurements)
        if hasattr(op, "cout") and op.cout:
            for cout in op.cout:
                if hasattr(cout, "reg") and hasattr(cout.reg, "sym"):
                    used_vars.add(cout.reg.sym)
                elif hasattr(cout, "sym"):
                    # Direct CReg reference
                    used_vars.add(cout.sym)

        # Check condition (for If blocks)
        if hasattr(op, "cond"):
            self._collect_variables_from_expr(op.cond, used_vars)

        # Check expressions (for classical operations)
        if hasattr(op, "left"):
            self._collect_variables_from_expr(op.left, used_vars)
        if hasattr(op, "right"):
            self._collect_variables_from_expr(op.right, used_vars)

    def _collect_variables_from_expr(self, expr, used_vars: set[str]):
        """Collect variable names from expressions."""
        if hasattr(expr, "reg") and hasattr(expr.reg, "sym"):
            used_vars.add(expr.reg.sym)
        elif hasattr(expr, "left") and hasattr(expr, "right"):
            # Binary operation
            self._collect_variables_from_expr(expr.left, used_vars)
            self._collect_variables_from_expr(expr.right, used_vars)
        elif hasattr(expr, "args"):
            # Function call or similar
            for arg in expr.args:
                self._collect_variables_from_expr(arg, used_vars)

    def _find_qreg_in_ops(self, block):
        """Try to find quantum register used in operations."""
        if hasattr(block, "ops"):
            for op in block.ops:
                if hasattr(op, "qargs") and op.qargs:
                    qarg = op.qargs[0]
                    if hasattr(qarg, "reg"):
                        return qarg.reg
        return None

    def _find_ancilla_in_ops(self, block):
        """Try to find ancilla qubit used in operations."""
        # Look for single qubit operations that might be ancilla
        if hasattr(block, "ops"):
            for op in block.ops:
                if type(op).__name__ == "Prep" and hasattr(op, "qargs"):
                    # Prep operations often reset ancillas
                    for qarg in op.qargs:
                        if hasattr(qarg, "index") and not hasattr(qarg, "size"):
                            # Single qubit
                            return qarg
        return None

    def _find_bit_in_ops(self, block):
        """Try to find classical bit used in operations."""
        if hasattr(block, "ops"):
            for op in block.ops:
                if hasattr(op, "cout") and op.cout:
                    # Measurement output
                    return op.cout[0]
        return None

    def get_required_parameters(
        self,
        block: Block,
        parent_context: dict[str, Any],
    ) -> list[tuple[str, str]]:
        """Get the parameters required for a block function.

        Args:
            block: The block to analyze
            parent_context: Dictionary mapping variable names to their types/values

        Returns:
            List of (param_name, param_type) tuples
        """
        dep = self.analyze_block(block)

        # Collect all used variables
        all_used = dep.used_variables.copy()

        # Map to parameter types
        params = []
        for var_name in sorted(all_used):
            if var_name in parent_context:
                var_info = parent_context[var_name]
                if hasattr(var_info, "__class__"):
                    var_type = var_info.__class__.__name__
                    if var_type == "QReg":
                        size = var_info.size if hasattr(var_info, "size") else 1
                        params.append((var_name, f"array[quantum.qubit, {size}]"))
                    elif var_type == "CReg":
                        size = var_info.size if hasattr(var_info, "size") else 1
                        params.append((var_name, f"array[bool, {size}]"))
                    else:
                        params.append((var_name, var_type))

        return params
