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

"""Tests for optimization pipeline."""

import math

from pecos.slr import Main, QReg
from pecos.slr.ast import slr_to_ast
from pecos.slr.ast.nodes import GateKind, GateOp, LiteralExpr
from pecos.slr.ast.optimizations import (
    GateCancellationPass,
    IdentityRemovalPass,
    InverseCancellationPass,
    OptimizationPipeline,
    RotationMergingPass,
    create_default_pipeline,
    optimize,
)
from pecos.slr.qeclib import qubit as qb


class TestOptimizeLevels:
    """Tests for optimize() function with different levels."""

    def test_level_0_no_optimization(self) -> None:
        """Level 0 performs no optimization."""
        prog = Main(
            q := QReg("q", 1),
            qb.X(q[0]),
            qb.X(q[0]),
        )

        ast = slr_to_ast(prog)
        result = optimize(ast, level=0)

        assert len(result.program.body) == 2
        assert result.gates_removed == 0
        assert result.gates_merged == 0

    def test_level_1_gate_cancellation(self) -> None:
        """Level 1 performs gate cancellation."""
        prog = Main(
            q := QReg("q", 1),
            qb.X(q[0]),
            qb.X(q[0]),
        )

        ast = slr_to_ast(prog)
        result = optimize(ast, level=1)

        assert len(result.program.body) == 0
        assert result.gates_removed == 2

    def test_level_1_inverse_cancellation(self) -> None:
        """Level 1 performs inverse cancellation."""
        prog = Main(
            q := QReg("q", 1),
            qb.SZ(q[0]),
            qb.SZdg(q[0]),
        )

        ast = slr_to_ast(prog)
        result = optimize(ast, level=1)

        assert len(result.program.body) == 0
        assert result.gates_removed == 2

    def test_level_2_rotation_merging(self) -> None:
        """Level 2 adds rotation merging."""
        prog = Main(
            q := QReg("q", 1),
            qb.RZ[0.5](q[0]),
            qb.RZ[0.3](q[0]),
        )

        ast = slr_to_ast(prog)
        result = optimize(ast, level=2)

        assert len(result.program.body) == 1
        assert result.gates_merged == 1

    def test_level_3_identity_removal(self) -> None:
        """Level 3 adds identity removal."""
        prog = Main(
            q := QReg("q", 1),
            qb.RZ[0](q[0]),
        )

        ast = slr_to_ast(prog)
        result = optimize(ast, level=3)

        assert len(result.program.body) == 0
        assert result.gates_removed == 1


class TestOptimizationPipeline:
    """Tests for OptimizationPipeline class."""

    def test_custom_pipeline(self) -> None:
        """Custom pipeline with specific passes."""
        pipeline = OptimizationPipeline(
            [
                GateCancellationPass(),
            ],
        )

        prog = Main(
            q := QReg("q", 1),
            qb.X(q[0]),
            qb.X(q[0]),
        )

        ast = slr_to_ast(prog)
        result = pipeline.optimize(ast)

        assert len(result.program.body) == 0
        assert result.gates_removed == 2

    def test_pipeline_fixed_point(self) -> None:
        """Pipeline iterates to fixed point."""
        # RZ(0.5) + RZ(-0.5) -> RZ(0) -> removed
        pipeline = OptimizationPipeline(
            [
                RotationMergingPass(),
                IdentityRemovalPass(),
            ],
        )

        prog = Main(
            q := QReg("q", 1),
            qb.RZ[0.5](q[0]),
            qb.RZ[-0.5](q[0]),
        )

        ast = slr_to_ast(prog)
        result = pipeline.optimize(ast)

        assert len(result.program.body) == 0
        assert result.gates_merged == 1
        assert result.gates_removed == 1

    def test_pipeline_no_fixed_point(self) -> None:
        """Pipeline with iterate_to_fixed_point=False runs once."""
        pipeline = OptimizationPipeline(
            [
                GateCancellationPass(),
            ],
            iterate_to_fixed_point=False,
        )

        # Four X gates: first pass removes 2, second would remove 2 more
        prog = Main(
            q := QReg("q", 1),
            qb.X(q[0]),
            qb.X(q[0]),
            qb.X(q[0]),
            qb.X(q[0]),
        )

        ast = slr_to_ast(prog)
        result = pipeline.optimize(ast)

        # Without fixed point, only first pairs cancel
        assert len(result.program.body) == 0  # Both pairs cancel in one scan
        assert result.gates_removed == 4

    def test_pipeline_max_iterations(self) -> None:
        """Pipeline respects max_iterations."""
        pipeline = OptimizationPipeline(
            [
                GateCancellationPass(),
            ],
            max_iterations=1,
            iterate_to_fixed_point=True,
        )

        prog = Main(
            q := QReg("q", 1),
            qb.X(q[0]),
            qb.X(q[0]),
        )

        ast = slr_to_ast(prog)
        result = pipeline.optimize(ast)

        assert len(result.program.body) == 0
        assert result.gates_removed == 2


class TestCreateDefaultPipeline:
    """Tests for create_default_pipeline()."""

    def test_default_pipeline_all_optimizations(self) -> None:
        """Default pipeline applies all optimizations."""
        pipeline = create_default_pipeline()

        # Circuit with multiple optimization opportunities
        prog = Main(
            q := QReg("q", 1),
            qb.RZ[0](q[0]),  # Identity removal
            qb.X(q[0]),  # Gate cancellation
            qb.X(q[0]),
            qb.SZ(q[0]),  # Inverse cancellation
            qb.SZdg(q[0]),
            qb.RZ[0.5](q[0]),  # Rotation merging
            qb.RZ[0.3](q[0]),
        )

        ast = slr_to_ast(prog)
        result = pipeline.optimize(ast)

        # Only merged RZ(0.8) should remain
        assert len(result.program.body) == 1
        gate = result.program.body[0]
        assert isinstance(gate, GateOp)
        assert gate.gate == GateKind.RZ
        assert isinstance(gate.params[0], LiteralExpr)
        assert abs(gate.params[0].value - 0.8) < 1e-10


class TestPipelinePassTracking:
    """Tests for pass tracking in results."""

    def test_passes_applied_tracked(self) -> None:
        """Applied passes are tracked in result."""
        pipeline = OptimizationPipeline(
            [
                GateCancellationPass(),
                InverseCancellationPass(),
            ],
        )

        prog = Main(
            q := QReg("q", 1),
            qb.X(q[0]),
            qb.X(q[0]),
        )

        ast = slr_to_ast(prog)
        result = pipeline.optimize(ast)

        assert "gate_cancellation" in result.passes_applied
        assert "inverse_cancellation" in result.passes_applied

    def test_total_optimizations(self) -> None:
        """total_optimizations property works correctly."""
        prog = Main(
            q := QReg("q", 1),
            qb.X(q[0]),
            qb.X(q[0]),
            qb.RZ[0.5](q[0]),
            qb.RZ[0.3](q[0]),
        )

        ast = slr_to_ast(prog)
        result = optimize(ast, level=2)

        assert result.gates_removed == 2
        assert result.gates_merged == 1
        assert result.total_optimizations == 3


class TestComplexOptimization:
    """Tests for complex optimization scenarios."""

    def test_bell_state_no_optimization_needed(self) -> None:
        """Bell state circuit has no redundant gates."""
        prog = Main(
            q := QReg("q", 2),
            qb.H(q[0]),
            qb.CX(q[0], q[1]),
        )

        ast = slr_to_ast(prog)
        result = optimize(ast, level=3)

        assert len(result.program.body) == 2
        assert result.total_optimizations == 0

    def test_redundant_syndrome_extraction(self) -> None:
        """Redundant syndrome extraction gates can be optimized."""
        prog = Main(
            q := QReg("q", 2),
            # Redundant CX gates
            qb.CX(q[0], q[1]),
            qb.CX(q[0], q[1]),
        )

        ast = slr_to_ast(prog)
        result = optimize(ast, level=1)

        assert len(result.program.body) == 0
        assert result.gates_removed == 2

    def test_rotation_to_identity_chain(self) -> None:
        """Chain of rotations that sum to identity."""
        prog = Main(
            q := QReg("q", 1),
            qb.RZ[math.pi](q[0]),
            qb.RZ[math.pi](q[0]),  # Sum is 2*pi = identity
        )

        ast = slr_to_ast(prog)
        result = optimize(ast, level=3)

        assert len(result.program.body) == 0
        # One merge, then one removal
        assert result.gates_merged == 1
        assert result.gates_removed == 1
