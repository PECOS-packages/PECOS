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

"""Base classes and result types for code generation.

This module provides the CodegenResult class that bundles generated code
with optional validation and analysis metadata.

Example:
    >>> from pecos.slr.ast.codegen import generate_with_validation
    >>> result = generate_with_validation(ast, target="qasm")
    >>> if result.validation.valid:
    ...     print(result.code)
    ...     print(f"T-count: {result.t_count.t_count}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pecos.slr.ast.analysis import (
        ConnectivityResult,
        DepthResult,
        ParallelismResult,
        ResourceCount,
        TCountResult,
    )
    from pecos.slr.ast.validation import ValidationResult


@dataclass
class CodegenResult:
    """Result of code generation with optional metadata.

    This class bundles the generated code with optional validation
    results and analysis metrics.

    Attributes:
        code: The generated code (type depends on target).
        validation: Validation results, if validation was requested.
        resources: Resource counts (gates, qubits, etc.), if analysis was requested.
        t_count: T-count analysis results, if analysis was requested.
        depth: Circuit depth analysis results, if analysis was requested.
        connectivity: Connectivity analysis results, if analysis was requested.
        parallelism: Parallelism analysis results, if analysis was requested.
        target: The code generation target used.
    """

    code: str | list[str] | Any
    target: str = ""
    validation: ValidationResult | None = None
    resources: ResourceCount | None = None
    t_count: TCountResult | None = None
    depth: DepthResult | None = None
    connectivity: ConnectivityResult | None = None
    parallelism: ParallelismResult | None = None

    @property
    def valid(self) -> bool:
        """Check if validation passed (or wasn't run).

        Returns:
            True if validation passed or wasn't run, False otherwise.
        """
        if self.validation is None:
            return True
        return self.validation.valid

    def __str__(self) -> str:
        """String representation of the result."""
        lines = [f"CodegenResult(target={self.target!r})"]

        if self.validation is not None:
            status = "valid" if self.validation.valid else "invalid"
            lines.append(f"  validation: {status}")
            if not self.validation.valid:
                lines.append(f"    errors: {len(self.validation.errors)}")

        if self.resources is not None:
            lines.append(f"  resources: {self.resources.total_gates} gates, {self.resources.qubit_count} qubits")

        if self.t_count is not None:
            lines.append(f"  t_count: {self.t_count.t_count}")

        if self.depth is not None:
            lines.append(f"  depth: {self.depth.depth}")

        return "\n".join(lines)


@dataclass
class CodegenOptions:
    """Options for code generation.

    Attributes:
        validate: If True, run validation before generating code.
        include_resources: If True, include resource counts.
        include_t_count: If True, include T-count analysis.
        include_depth: If True, include depth analysis.
        include_connectivity: If True, include connectivity analysis.
        include_parallelism: If True, include parallelism analysis.
        include_all_analysis: If True, include all analysis passes.
    """

    validate: bool = False
    include_resources: bool = False
    include_t_count: bool = False
    include_depth: bool = False
    include_connectivity: bool = False
    include_parallelism: bool = False
    include_all_analysis: bool = False

    def should_include_resources(self) -> bool:
        """Check if resource counting should be included."""
        return self.include_resources or self.include_all_analysis

    def should_include_t_count(self) -> bool:
        """Check if T-count should be included."""
        return self.include_t_count or self.include_all_analysis

    def should_include_depth(self) -> bool:
        """Check if depth should be included."""
        return self.include_depth or self.include_all_analysis

    def should_include_connectivity(self) -> bool:
        """Check if connectivity should be included."""
        return self.include_connectivity or self.include_all_analysis

    def should_include_parallelism(self) -> bool:
        """Check if parallelism should be included."""
        return self.include_parallelism or self.include_all_analysis
