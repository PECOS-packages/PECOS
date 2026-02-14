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

"""Optimization pipeline for composing multiple passes.

The pipeline allows chaining multiple optimization passes together and
optionally iterating until a fixed point is reached.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from pecos.slr.ast.optimizations.base import OptimizationPass, OptimizationResult
from pecos.slr.ast.optimizations.gate_cancellation import GateCancellationPass
from pecos.slr.ast.optimizations.identity_removal import IdentityRemovalPass
from pecos.slr.ast.optimizations.inverse_cancellation import InverseCancellationPass
from pecos.slr.ast.optimizations.rotation_merging import RotationMergingPass

if TYPE_CHECKING:
    from pecos.slr.ast.nodes import Program


class OptimizationPipeline:
    """Compose multiple optimization passes into a pipeline.

    Passes are applied in order. The pipeline can optionally iterate
    until no more optimizations are found (fixed-point iteration).

    Example:
        pipeline = OptimizationPipeline([
            GateCancellationPass(),
            InverseCancellationPass(),
            RotationMergingPass(),
            IdentityRemovalPass(),
        ])
        result = pipeline.optimize(program)

    Attributes:
        passes: List of optimization passes to apply.
        max_iterations: Maximum number of iterations (default 10).
        iterate_to_fixed_point: Whether to repeat until no changes (default True).
    """

    def __init__(
        self,
        passes: Sequence[OptimizationPass],
        max_iterations: int = 10,
        iterate_to_fixed_point: bool = True,
    ):
        self.passes = list(passes)
        self.max_iterations = max_iterations
        self.iterate_to_fixed_point = iterate_to_fixed_point

    def optimize(self, program: Program) -> OptimizationResult:
        """Apply all passes to the program.

        Args:
            program: The AST Program to optimize.

        Returns:
            OptimizationResult with the fully optimized program and
            cumulative statistics from all passes.
        """
        current = program
        total_removed = 0
        total_merged = 0
        all_passes: list[str] = []

        for _iteration in range(self.max_iterations):
            iteration_removed = 0
            iteration_merged = 0

            for opt_pass in self.passes:
                result = opt_pass.optimize(current)
                current = result.program
                iteration_removed += result.gates_removed
                iteration_merged += result.gates_merged
                all_passes.extend(result.passes_applied)

            total_removed += iteration_removed
            total_merged += iteration_merged

            if not self.iterate_to_fixed_point:
                break

            if iteration_removed == 0 and iteration_merged == 0:
                # Fixed point reached
                break

        return OptimizationResult(
            program=current,
            gates_removed=total_removed,
            gates_merged=total_merged,
            passes_applied=all_passes,
        )


def optimize(program: Program, level: int = 1) -> OptimizationResult:
    """Optimize a program with the specified optimization level.

    This is a convenience function that creates an appropriate pipeline
    based on the requested optimization level.

    Args:
        program: The AST Program to optimize.
        level: Optimization level (0-3).
            0: No optimization
            1: Gate cancellation and inverse cancellation
            2: Level 1 + rotation merging
            3: Level 2 + identity removal (full optimization)

    Returns:
        OptimizationResult with the optimized program and statistics.

    Example:
        from pecos.slr.ast import slr_to_ast
        from pecos.slr.ast.optimizations import optimize

        ast = slr_to_ast(program)
        result = optimize(ast, level=2)
        print(f"Removed {result.gates_removed} gates")
    """
    if level == 0:
        return OptimizationResult(program=program)

    passes: list[OptimizationPass] = [
        GateCancellationPass(),
        InverseCancellationPass(),
    ]

    if level >= 2:
        passes.append(RotationMergingPass())

    if level >= 3:
        passes.append(IdentityRemovalPass())

    pipeline = OptimizationPipeline(passes)
    return pipeline.optimize(program)


def create_default_pipeline() -> OptimizationPipeline:
    """Create a pipeline with all standard optimizations.

    The default pipeline includes all optimization passes in an order
    that maximizes the number of optimizations found:
    1. Identity removal (removes RX(0), etc.)
    2. Gate cancellation (removes X-X, H-H, etc.)
    3. Inverse cancellation (removes S-Sdg, T-Tdg, etc.)
    4. Rotation merging (combines RX(a)+RX(b), etc.)

    Returns:
        An OptimizationPipeline configured with all standard passes.

    Example:
        from pecos.slr.ast.optimizations import create_default_pipeline

        pipeline = create_default_pipeline()
        result = pipeline.optimize(ast)
    """
    return OptimizationPipeline(
        [
            IdentityRemovalPass(),  # Remove identity first
            GateCancellationPass(),  # Then cancel self-inverse
            InverseCancellationPass(),  # Then cancel inverse pairs
            RotationMergingPass(),  # Finally merge rotations
        ]
    )
