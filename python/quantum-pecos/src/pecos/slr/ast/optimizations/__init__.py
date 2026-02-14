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

"""AST optimization passes for quantum circuit simplification.

This package provides composable optimization passes that transform
AST Programs to remove redundant gates and simplify circuits.

Quick Start:
    from pecos.slr import Main, QReg
    from pecos.slr.qeclib import qubit as qb
    from pecos.slr.ast import slr_to_ast
    from pecos.slr.ast.optimizations import optimize

    # Create a circuit with redundant gates
    prog = Main(
        q := QReg("q", 1),
        qb.X(q[0]),
        qb.X(q[0]),  # Cancels with previous X
    )

    # Convert to AST and optimize
    ast = slr_to_ast(prog)
    result = optimize(ast, level=1)

    print(f"Removed {result.gates_removed} gates")
    # Output: Removed 2 gates

Optimization Levels:
    Level 0: No optimization
    Level 1: Gate cancellation (X-X, H-H) and inverse cancellation (S-Sdg, T-Tdg)
    Level 2: Level 1 + rotation merging (RX(a)+RX(b)=RX(a+b))
    Level 3: Level 2 + identity removal (RX(0) removed)

Individual Passes:
    - GateCancellationPass: Remove consecutive self-inverse gates
    - InverseCancellationPass: Remove consecutive inverse pairs
    - RotationMergingPass: Merge consecutive rotation gates
    - IdentityRemovalPass: Remove identity rotation gates

Custom Pipelines:
    from pecos.slr.ast.optimizations import (
        OptimizationPipeline,
        GateCancellationPass,
        RotationMergingPass,
    )

    # Create a custom pipeline with only specific passes
    pipeline = OptimizationPipeline([
        GateCancellationPass(),
        RotationMergingPass(),
    ])
    result = pipeline.optimize(ast)
"""

from pecos.slr.ast.optimizations.base import (
    OptimizationPass,
    OptimizationResult,
    StatementListOptimizer,
)
from pecos.slr.ast.optimizations.gate_cancellation import GateCancellationPass
from pecos.slr.ast.optimizations.gate_properties import (
    INVERSE_PAIRS,
    ROTATION_GATES,
    SELF_INVERSE_GATES,
    get_inverse,
    is_rotation_gate,
    is_self_inverse,
    targets_match,
)
from pecos.slr.ast.optimizations.identity_removal import IdentityRemovalPass
from pecos.slr.ast.optimizations.inverse_cancellation import InverseCancellationPass
from pecos.slr.ast.optimizations.pipeline import (
    OptimizationPipeline,
    create_default_pipeline,
    optimize,
)
from pecos.slr.ast.optimizations.rotation_merging import RotationMergingPass

__all__ = [
    # Base classes
    "OptimizationPass",
    "OptimizationResult",
    "StatementListOptimizer",
    # Passes
    "GateCancellationPass",
    "InverseCancellationPass",
    "RotationMergingPass",
    "IdentityRemovalPass",
    # Pipeline
    "OptimizationPipeline",
    "optimize",
    "create_default_pipeline",
    # Gate properties
    "SELF_INVERSE_GATES",
    "INVERSE_PAIRS",
    "ROTATION_GATES",
    "is_self_inverse",
    "get_inverse",
    "is_rotation_gate",
    "targets_match",
]
