"""Enhanced error handling for HUGR compilation failures."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import ClassVar


@dataclass
class ErrorContext:
    """Context information for an error."""

    line_number: int
    line_content: str
    variable_name: str | None = None
    error_type: str | None = None
    suggestion: str | None = None


class HugrErrorHandler:
    """Provides detailed error messages and suggestions for HUGR compilation failures."""

    # Common error patterns and their explanations
    ERROR_PATTERNS: ClassVar[dict[str, dict[str, str]]] = {
        r"PlaceNotUsedError.*Variable\(name='(\w+)'": {
            "type": "PlaceNotUsedError",
            "message": "Quantum register '{var}' was not consumed",
            "suggestion": "Add a measurement for this quantum register or ensure it's consumed in all execution paths",
        },
        r"NotOwnedError.*Variable\(name='(\w+)'": {
            "type": "NotOwnedError",
            "message": "Variable '{var}' is not owned in this context",
            "suggestion": "Ensure the variable is passed with @owned annotation or is properly borrowed",
        },
        r"AlreadyUsedError.*Variable\(name='(\w+)'": {
            "type": "AlreadyUsedError",
            "message": "Variable '{var}' has already been consumed",
            "suggestion": "Quantum resources can only be used once. Check for duplicate measurements or operations",
        },
        r"MoveOutOfSubscriptError": {
            "type": "MoveOutOfSubscriptError",
            "message": "Cannot move out of array subscript",
            "suggestion": (
                "Use array unpacking or measure_array() instead of individual element access after consumption"
            ),
        },
        r"NotCallableError.*'(\w+)'": {
            "type": "NotCallableError",
            "message": "'{var}' is not callable",
            "suggestion": "Check if a variable name conflicts with a function name (e.g., 'result')",
        },
        r"NameError.*name\s+'(\w+)'\s+is\s+not\s+defined": {
            "type": "NameError",
            "message": "Variable '{var}' is not defined",
            "suggestion": "Check variable scoping or if the variable was renamed to avoid conflicts",
        },
        r"TypeError.*missing.*positional argument.*'(\w+)'": {
            "type": "TypeError",
            "message": "Missing required argument '{var}'",
            "suggestion": "Check function signatures and ensure all required parameters are provided",
        },
        r"UnknownSourceError.*obj=<class.*\.(\w+)'>": {
            "type": "UnknownSourceError",
            "message": "Cannot find source location for dynamically generated class '{var}'",
            "suggestion": "This is a known limitation with dynamically generated structs. "
            "The struct definition is correct but lacks source tracking metadata.",
        },
    }

    def __init__(self, guppy_code: str):
        """Initialize with the generated Guppy code."""
        self.code_lines = guppy_code.split("\n")

    def analyze_error(self, error: Exception) -> str:
        """Analyze an error and provide detailed diagnostics."""
        error_str = str(error)
        error_type = type(error).__name__

        # Try to match against known patterns
        for pattern, info in self.ERROR_PATTERNS.items():
            match = re.search(pattern, error_str)
            if match:
                return self._format_known_error(match, info, error_str)

        # Handle specific error types with custom logic
        if "MoveOutOfSubscriptError" in error_str:
            return self._analyze_subscript_error(error_str)

        # Generic error handling
        return self._format_generic_error(error_type, error_str)

    def _format_known_error(self, match: re.Match, info: dict, error_str: str) -> str:
        """Format a known error pattern."""
        var_name = match.group(1) if match.groups() else None

        lines = [
            f"\n{'='*60}",
            f"HUGR Compilation Error: {info['type']}",
            f"{'='*60}\n",
        ]

        if var_name:
            lines.append(f"Problem: {info['message'].format(var=var_name)}")
        else:
            lines.append(f"Problem: {info['message']}")

        lines.append(f"\nSuggestion: {info['suggestion']}")

        # Find relevant code context
        context = self._find_code_context(var_name)
        if context:
            lines.append("\nRelevant code:")
            lines.extend(f"  Line {ctx.line_number}: {ctx.line_content.strip()}" for ctx in context)

        # Add specific examples for common fixes
        if info["type"] == "PlaceNotUsedError" and var_name:
            lines.append("\nExample fix:")
            lines.append("  # Add before the end of the function:")
            lines.append(f"  _ = quantum.measure_array({var_name})")

        elif info["type"] == "MoveOutOfSubscriptError":
            lines.append("\nExample fix:")
            lines.append("  # Instead of accessing elements after measurement:")
            lines.append("  # BAD:  c = measure_array(q); x = q[0]")
            lines.append("  # GOOD: q_0, q_1 = q; c_0 = measure(q_0)")

        lines.append(f"\nOriginal error: {error_str}")
        lines.append(f"{'='*60}\n")

        return "\n".join(lines)

    def _analyze_subscript_error(self, error_str: str) -> str:
        """Analyze MoveOutOfSubscriptError in detail."""
        lines = [
            f"\n{'='*60}",
            "HUGR Compilation Error: MoveOutOfSubscriptError",
            f"{'='*60}\n",
            "Problem: Attempting to access array elements after the array has been consumed",
            "\nThis typically happens when:",
            "  1. You measure an entire array with measure_array()",
            "  2. Then try to access individual elements like array[0]",
            "\nSolution approaches:",
            "  1. Unpack the array before measurements:",
            "     q_0, q_1, q_2 = q  # Unpack at the start",
            "     c_0 = quantum.measure(q_0)  # Use unpacked names",
            "\n  2. Use measure_array() for the entire array:",
            "     c = quantum.measure_array(q)  # Measure all at once",
            "\n  3. Measure individual elements without unpacking:",
            "     c[0] = quantum.measure(q[0])  # Before array is consumed",
        ]

        # Look for array access patterns in the code
        for i, line in enumerate(self.code_lines):
            if (
                "measure_array" in line
                and "["
                in self.code_lines[
                    min(i + 1, len(self.code_lines) - 1) : min(
                        i + 5,
                        len(self.code_lines),
                    )
                ]
            ):
                lines.append(f"\nPotential issue found around line {i+1}:")
                lines.append(f"  {line.strip()}")

        lines.append(f"\nOriginal error: {error_str}")
        lines.append(f"{'='*60}\n")

        return "\n".join(lines)

    def _format_generic_error(self, error_type: str, error_str: str) -> str:
        """Format a generic error with basic diagnostics."""
        lines = [
            f"\n{'='*60}",
            f"HUGR Compilation Error: {error_type}",
            f"{'='*60}\n",
            f"Error details: {error_str}",
            "\nGeneral troubleshooting tips:",
            "  1. Check that all quantum registers are consumed (measured)",
            "  2. Ensure variables don't conflict with reserved names (result, array, quantum)",
            "  3. Verify array operations happen before the array is consumed",
            "  4. Check function parameter types and ownership annotations",
            "\nFor more specific help, please check the error message above.",
            f"{'='*60}\n",
        ]

        return "\n".join(lines)

    def _find_code_context(self, var_name: str | None) -> list[ErrorContext]:
        """Find relevant code lines for a variable."""
        if not var_name:
            return []

        contexts = []
        for i, line in enumerate(self.code_lines):
            if var_name in line:
                contexts.append(
                    ErrorContext(
                        line_number=i + 1,
                        line_content=line,
                        variable_name=var_name,
                    ),
                )

        return contexts
