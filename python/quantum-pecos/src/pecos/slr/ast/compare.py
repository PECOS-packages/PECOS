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

"""AST comparison and diff utilities.

This module provides functions to compare AST nodes for structural equality
and identify differences between programs.

Example:
    >>> from pecos.slr import Main, QReg
    >>> from pecos.slr.qeclib import qubit as qb
    >>> from pecos.slr.ast import slr_to_ast
    >>> from pecos.slr.ast.compare import ast_equal, compare_ast
    >>>
    >>> prog1 = Main(q := QReg("q", 2), qb.H(q[0]), qb.CX(q[0], q[1]))
    >>> prog2 = Main(q := QReg("q", 2), qb.H(q[0]), qb.CX(q[0], q[1]))
    >>> ast1, ast2 = slr_to_ast(prog1), slr_to_ast(prog2)
    >>> ast_equal(ast1, ast2)  # True
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from typing import Any

from pecos.slr.ast.nodes import AstNode, Program, SourceLocation


@dataclass
class AstDiff:
    """Result of comparing two AST programs.

    Attributes:
        equal: Whether the programs are structurally equal.
        differences: List of human-readable difference descriptions.
        path: Current path in the AST being compared (internal use).
    """

    equal: bool
    differences: list[str]

    def __str__(self) -> str:
        if self.equal:
            return "ASTs are equal"
        result = f"ASTs differ ({len(self.differences)} difference(s)):\n"
        for diff in self.differences:
            result += f"  - {diff}\n"
        return result.rstrip()

    def __bool__(self) -> bool:
        """Returns True if ASTs are equal."""
        return self.equal


class AstComparator:
    """Compare two AST structures for equality."""

    def __init__(self, *, ignore_location: bool = True, ignore_name: bool = False):
        """Initialize comparator.

        Args:
            ignore_location: If True, ignore SourceLocation differences.
            ignore_name: If True, ignore program name differences.
        """
        self._ignore_location = ignore_location
        self._ignore_name = ignore_name
        self._differences: list[str] = []
        self._path: list[str] = []

    def compare(self, a: Program, b: Program) -> AstDiff:
        """Compare two programs.

        Args:
            a: First program.
            b: Second program.

        Returns:
            AstDiff with comparison results.
        """
        self._differences = []
        self._path = []
        self._compare_nodes(a, b)
        return AstDiff(
            equal=len(self._differences) == 0,
            differences=self._differences,
        )

    def _current_path(self) -> str:
        """Get current path as string."""
        return ".".join(self._path) if self._path else "root"

    def _add_diff(self, message: str) -> None:
        """Add a difference with current path."""
        self._differences.append(f"{self._current_path()}: {message}")

    def _compare_nodes(self, a: Any, b: Any) -> bool:
        """Compare two values recursively.

        Returns:
            True if equal, False otherwise.
        """
        # Handle None
        if a is None and b is None:
            return True
        if a is None or b is None:
            self._add_diff(f"one is None, other is {type(b if a is None else a).__name__}")
            return False

        # Handle different types
        if type(a) is not type(b):
            self._add_diff(f"type mismatch: {type(a).__name__} vs {type(b).__name__}")
            return False

        # Handle primitives
        if isinstance(a, (int, float, bool, str)):
            if a != b:
                self._add_diff(f"value mismatch: {a!r} vs {b!r}")
                return False
            return True

        # Handle enums
        if hasattr(a, "name") and hasattr(a, "value") and not is_dataclass(a):
            if a != b:
                self._add_diff(f"enum mismatch: {a.name} vs {b.name}")
                return False
            return True

        # Handle tuples and lists
        if isinstance(a, (tuple, list)):
            if len(a) != len(b):
                self._add_diff(f"length mismatch: {len(a)} vs {len(b)}")
                return False
            all_equal = True
            for i, (item_a, item_b) in enumerate(zip(a, b)):
                self._path.append(f"[{i}]")
                if not self._compare_nodes(item_a, item_b):
                    all_equal = False
                self._path.pop()
            return all_equal

        # Handle dataclasses (AST nodes)
        if is_dataclass(a):
            all_equal = True
            for f in fields(a):
                # Skip location if configured
                if self._ignore_location and f.name == "location":
                    continue
                # Skip name if configured (for Program)
                if self._ignore_name and f.name == "name" and isinstance(a, Program):
                    continue

                self._path.append(f.name)
                val_a = getattr(a, f.name)
                val_b = getattr(b, f.name)
                if not self._compare_nodes(val_a, val_b):
                    all_equal = False
                self._path.pop()
            return all_equal

        # Unknown type
        self._add_diff(f"cannot compare type: {type(a).__name__}")
        return False


def compare_ast(a: Program, b: Program, *, ignore_location: bool = True, ignore_name: bool = False) -> AstDiff:
    """Compare two AST programs for structural equality.

    Args:
        a: First program to compare.
        b: Second program to compare.
        ignore_location: If True, ignore SourceLocation differences.
        ignore_name: If True, ignore program name differences.

    Returns:
        AstDiff object with comparison results.

    Example:
        >>> diff = compare_ast(ast1, ast2)
        >>> if not diff.equal:
        ...     print(diff)
    """
    comparator = AstComparator(ignore_location=ignore_location, ignore_name=ignore_name)
    return comparator.compare(a, b)


def ast_equal(a: Program, b: Program, *, ignore_location: bool = True, ignore_name: bool = False) -> bool:
    """Check if two AST programs are structurally equal.

    Args:
        a: First program to compare.
        b: Second program to compare.
        ignore_location: If True, ignore SourceLocation differences.
        ignore_name: If True, ignore program name differences.

    Returns:
        True if programs are equal, False otherwise.

    Example:
        >>> ast_equal(ast1, ast2)
        True
    """
    return compare_ast(a, b, ignore_location=ignore_location, ignore_name=ignore_name).equal


def nodes_equal(a: AstNode, b: AstNode, *, ignore_location: bool = True) -> bool:
    """Check if two AST nodes are structurally equal.

    This is a more general version that works with any AST node type,
    not just Program nodes.

    Args:
        a: First node to compare.
        b: Second node to compare.
        ignore_location: If True, ignore SourceLocation differences.

    Returns:
        True if nodes are equal, False otherwise.
    """
    comparator = AstComparator(ignore_location=ignore_location)
    comparator._compare_nodes(a, b)
    return len(comparator._differences) == 0
