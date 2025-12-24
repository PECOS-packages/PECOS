# Copyright 2025 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Tests for the ParallelOptimizer transformation pass."""

import pytest
from pecos.qeclib import qubit as qb
from pecos.slr import Block, CReg, If, Main, Parallel, QReg, Repeat
from pecos.slr.transforms import ParallelOptimizer


def test_basic_parallel_optimization() -> None:
    """Test basic optimization of independent operations."""
    optimizer = ParallelOptimizer()

    prog = Main(
        q := QReg("q", 4),
        _c := CReg("m", 4),
        Parallel(
            qb.H(q[0]),
            qb.X(q[1]),
            qb.H(q[2]),
            qb.X(q[3]),
        ),
    )

    optimized = optimizer.transform(prog)

    # Should have one Parallel block transformed into a Block with nested Parallels
    assert len(optimized.ops) == 1
    assert isinstance(optimized.ops[0], Block)

    # Should have grouped H gates and X gates
    inner_block = optimized.ops[0]
    assert len(inner_block.ops) == 2
    assert all(isinstance(op, Parallel) for op in inner_block.ops)

    # First group should be H gates
    h_group = inner_block.ops[0]
    assert len(h_group.ops) == 2
    assert all(isinstance(op, qb.H) for op in h_group.ops)

    # Second group should be X gates
    x_group = inner_block.ops[1]
    assert len(x_group.ops) == 2
    assert all(isinstance(op, qb.X) for op in x_group.ops)


def test_bell_state_optimization() -> None:
    """Test optimization of multiple Bell state preparations."""
    optimizer = ParallelOptimizer()

    prog = Main(
        q := QReg("q", 6),
        c := CReg("m", 6),
        Parallel(
            Block(  # Bell pair 1
                qb.H(q[0]),
                qb.CX(q[0], q[1]),
                qb.Measure(q[0]) > c[0],
                qb.Measure(q[1]) > c[1],
            ),
            Block(  # Bell pair 2
                qb.H(q[2]),
                qb.CX(q[2], q[3]),
                qb.Measure(q[2]) > c[2],
                qb.Measure(q[3]) > c[3],
            ),
            Block(  # Bell pair 3
                qb.H(q[4]),
                qb.CX(q[4], q[5]),
                qb.Measure(q[4]) > c[4],
                qb.Measure(q[5]) > c[5],
            ),
        ),
    )

    optimized = optimizer.transform(prog)

    # Should have transformed the Parallel block
    assert len(optimized.ops) == 1
    assert isinstance(optimized.ops[0], Block)

    # Should have three groups: H gates, CX gates, Measurements
    inner_block = optimized.ops[0]
    assert len(inner_block.ops) >= 3

    # First group should be H gates
    assert isinstance(inner_block.ops[0], Parallel)
    assert len(inner_block.ops[0].ops) == 3
    assert all(isinstance(op, qb.H) for op in inner_block.ops[0].ops)

    # Second group should be CX gates
    assert isinstance(inner_block.ops[1], Parallel)
    assert len(inner_block.ops[1].ops) == 3
    assert all(isinstance(op, qb.CX) for op in inner_block.ops[1].ops)


def test_dependent_operations() -> None:
    """Test that dependent operations are not reordered incorrectly."""
    optimizer = ParallelOptimizer()

    prog = Main(
        q := QReg("q", 2),
        Parallel(
            qb.H(q[0]),
            qb.CX(q[0], q[1]),  # Depends on H(q[0])
            qb.X(q[1]),  # Depends on CX
        ),
    )

    optimized = optimizer.transform(prog)

    # Operations should maintain dependency order
    assert len(optimized.ops) == 1
    inner_block = optimized.ops[0]

    # Should have 3 operations (H, CX, X) in order due to dependencies
    assert len(inner_block.ops) == 3


def test_parallel_with_control_flow() -> None:
    """Test that Parallel blocks with control flow are not optimized."""
    optimizer = ParallelOptimizer()

    prog = Main(
        q := QReg("q", 4),
        c := CReg("m", 4),
        Parallel(
            qb.H(q[0]),
            If(c[0] == 1).Then(qb.X(q[1])),
            qb.H(q[2]),
        ),
    )

    optimized = optimizer.transform(prog)

    # Should not optimize due to control flow
    assert len(optimized.ops) == 1
    assert isinstance(optimized.ops[0], Parallel)
    assert len(optimized.ops[0].ops) == 3


def test_nested_parallel_blocks() -> None:
    """Test optimization of nested Parallel blocks."""
    optimizer = ParallelOptimizer()

    prog = Main(
        q := QReg("q", 4),
        Parallel(
            Parallel(
                qb.H(q[0]),
                qb.H(q[1]),
            ),
            Parallel(
                qb.X(q[2]),
                qb.X(q[3]),
            ),
        ),
    )

    optimized = optimizer.transform(prog)

    # Should optimize inner Parallels first, then outer
    assert len(optimized.ops) == 1
    assert isinstance(optimized.ops[0], Block)


def test_parallel_with_repeat() -> None:
    """Test that Parallel blocks with Repeat are not optimized."""
    optimizer = ParallelOptimizer()

    prog = Main(
        q := QReg("q", 2),
        Parallel(
            qb.H(q[0]),
            Repeat(3).block(qb.X(q[1])),
        ),
    )

    optimized = optimizer.transform(prog)

    # Should not optimize due to Repeat
    assert len(optimized.ops) == 1
    assert isinstance(optimized.ops[0], Parallel)


def test_mixed_gate_types() -> None:
    """Test optimization with various gate types."""
    optimizer = ParallelOptimizer()

    prog = Main(
        q := QReg("q", 8),
        Parallel(
            qb.H(q[0]),
            qb.X(q[1]),
            qb.Y(q[2]),
            qb.Z(q[3]),
            qb.SZ(q[4]),
            qb.T(q[5]),
            qb.H(q[6]),
            qb.X(q[7]),
        ),
    )

    optimized = optimizer.transform(prog)

    # Should group gates by type
    assert len(optimized.ops) == 1
    assert isinstance(optimized.ops[0], Block)

    # Should have multiple groups for different gate types
    inner_block = optimized.ops[0]
    assert len(inner_block.ops) >= 2  # At least H gates and other gates


def test_empty_parallel() -> None:
    """Test handling of empty Parallel blocks."""
    optimizer = ParallelOptimizer()

    prog = Main(
        _q := QReg("q", 2),
        Parallel(),
    )

    optimized = optimizer.transform(prog)

    # Empty parallel should remain unchanged
    assert len(optimized.ops) == 1
    assert isinstance(optimized.ops[0], Parallel)
    assert len(optimized.ops[0].ops) == 0


def test_single_operation_parallel() -> None:
    """Test Parallel block with single operation."""
    optimizer = ParallelOptimizer()

    prog = Main(
        q := QReg("q", 1),
        Parallel(qb.H(q[0])),
    )

    optimized = optimizer.transform(prog)

    # Single operation should remain in Parallel
    assert len(optimized.ops) == 1
    assert isinstance(optimized.ops[0], Parallel)
    assert len(optimized.ops[0].ops) == 1


def test_classical_operation_barrier() -> None:
    """Test that classical operations act as barriers (future enhancement)."""
    optimizer = ParallelOptimizer()

    prog = Main(
        q := QReg("q", 2),
        c := CReg("m", 2),
        Parallel(
            qb.H(q[0]),
            qb.Measure(q[0]) > c[0],  # Classical operation
            qb.H(q[1]),  # Could be reordered if we handle classical ops
        ),
    )

    optimized = optimizer.transform(prog)

    # Current implementation treats all operations uniformly
    # Future enhancement could handle classical operations specially
    assert len(optimized.ops) == 1


def test_complex_nested_structure() -> None:
    """Test complex nested structure with mixed blocks."""
    optimizer = ParallelOptimizer()

    prog = Main(
        q := QReg("q", 4),
        _c := CReg("m", 4),
        Block(
            Parallel(
                qb.H(q[0]),
                qb.H(q[1]),
            ),
            Block(
                Parallel(
                    qb.X(q[2]),
                    qb.X(q[3]),
                ),
            ),
        ),
    )

    optimized = optimizer.transform(prog)

    # Should optimize each Parallel block independently
    assert len(optimized.ops) == 1
    assert isinstance(optimized.ops[0], Block)

    outer_block = optimized.ops[0]
    assert len(outer_block.ops) == 2

    # First should be optimized H gates
    # Second should be a Block containing optimized X gates


def test_preserves_main_attributes() -> None:
    """Test that Main block attributes are preserved."""
    optimizer = ParallelOptimizer()

    prog = Main(
        q := QReg("q", 2),
        _c := CReg("m", 2),
        Parallel(
            qb.H(q[0]),
            qb.H(q[1]),
        ),
    )

    optimized = optimizer.transform(prog)

    # Should preserve vars
    assert hasattr(optimized, "vars")
    # Vars include the QReg and CReg declarations
    vars_dict = {var.sym: var for var in optimized.vars}
    assert "q" in vars_dict
    assert "m" in vars_dict  # CReg("m", 2)


@pytest.mark.parametrize(
    ("gate_type", "expected_groups"),
    [
        ([qb.H, qb.H, qb.H], 1),  # All same type
        ([qb.H, qb.X, qb.H], 2),  # Mixed types
        ([qb.H, qb.X, qb.Y, qb.Z], 4),  # All different
    ],
)
def test_gate_grouping(gate_type: list, expected_groups: int) -> None:
    """Test that gates are grouped correctly by type."""
    optimizer = ParallelOptimizer()

    prog = Main(
        q := QReg("q", len(gate_type)),
        Parallel(*[gate(q[i]) for i, gate in enumerate(gate_type)]),
    )

    optimized = optimizer.transform(prog)

    assert len(optimized.ops) == 1
    if expected_groups == 1:
        # Single group stays as Parallel
        assert isinstance(optimized.ops[0], Parallel)
    else:
        # Multiple groups become Block with Parallels
        assert isinstance(optimized.ops[0], Block)
        assert len(optimized.ops[0].ops) == expected_groups


# Control flow edge case tests


def test_nested_if_in_parallel() -> None:
    """Test Parallel containing nested If blocks."""
    optimizer = ParallelOptimizer()

    prog = Main(
        q := QReg("q", 4),
        c := CReg("m", 4),
        Parallel(
            qb.H(q[0]),
            Block(
                If(c[0] == 1).Then(
                    qb.X(q[1]),
                ),
            ),
            qb.H(q[2]),
        ),
    )

    optimized = optimizer.transform(prog)

    # Should not optimize due to control flow in nested block
    assert len(optimized.ops) == 1
    assert isinstance(optimized.ops[0], Parallel)


def test_parallel_in_if_block() -> None:
    """Test If block containing Parallel - should optimize inner Parallel."""
    optimizer = ParallelOptimizer()

    prog = Main(
        q := QReg("q", 4),
        c := CReg("m", 4),
        If(c[0] == 1).Then(
            Parallel(
                qb.H(q[0]),
                qb.H(q[1]),
                qb.X(q[2]),
                qb.X(q[3]),
            ),
        ),
    )

    optimized = optimizer.transform(prog)

    # Should optimize the Parallel inside the If
    assert len(optimized.ops) == 1
    assert isinstance(optimized.ops[0], If)

    # The Then block should contain optimized structure
    if_block = optimized.ops[0]
    assert len(if_block.ops) == 1
    assert isinstance(if_block.ops[0], Block)  # Optimized parallel


def test_if_else_with_parallel() -> None:
    """Test If-Else structure with Parallel blocks."""
    optimizer = ParallelOptimizer()

    prog = Main(
        q := QReg("q", 2),
        c := CReg("m", 2),
        If(c[0] == 1)
        .Then(
            Parallel(qb.H(q[0]), qb.H(q[1])),
        )
        .Else(
            Parallel(qb.X(q[0]), qb.X(q[1])),
        ),
    )

    optimized = optimizer.transform(prog)

    # Both branches should be optimized
    assert len(optimized.ops) == 1
    assert isinstance(optimized.ops[0], If)

    # Then branch should have optimized Parallel
    if_block = optimized.ops[0]
    assert len(if_block.ops) == 1
    # Single type group stays as Parallel
    assert isinstance(if_block.ops[0], Parallel)

    # Else branch should also be optimized
    else_block = if_block.else_block
    assert len(else_block.ops) == 1
    assert isinstance(else_block.ops[0], Parallel)


def test_repeat_with_nested_parallel() -> None:
    """Test Repeat containing Parallel blocks."""
    optimizer = ParallelOptimizer()

    prog = Main(
        q := QReg("q", 2),
        Repeat(3).block(
            Parallel(
                qb.H(q[0]),
                qb.X(q[1]),
            ),
        ),
    )

    optimized = optimizer.transform(prog)

    # Should optimize Parallel inside Repeat
    assert len(optimized.ops) == 1
    assert isinstance(optimized.ops[0], Repeat)

    # The repeated block should contain optimized structure
    repeat_block = optimized.ops[0]
    assert len(repeat_block.ops) == 1
    assert isinstance(repeat_block.ops[0], Block)  # Optimized parallel


def test_mixed_control_flow_and_parallel() -> None:
    """Test complex mix of control flow and parallel blocks."""
    optimizer = ParallelOptimizer()

    prog = Main(
        q := QReg("q", 6),
        c := CReg("m", 6),
        Parallel(
            qb.H(q[0]),
            qb.H(q[1]),
        ),
        If(c[0] == 1).Then(
            Parallel(
                qb.X(q[2]),
                qb.Y(q[3]),
            ),
        ),
        Repeat(2).block(
            Parallel(
                qb.Z(q[4]),
                qb.SZ(q[5]),
            ),
        ),
    )

    optimized = optimizer.transform(prog)

    # Should have 3 operations: optimized Parallel, If, Repeat
    assert len(optimized.ops) == 3

    # First should be optimized Parallel (single type so stays Parallel)
    assert isinstance(optimized.ops[0], Parallel)

    # Second should be If with optimized inner Parallel
    assert isinstance(optimized.ops[1], If)

    # Third should be Repeat with optimized inner Parallel
    assert isinstance(optimized.ops[2], Repeat)


def test_deeply_nested_control_and_parallel() -> None:
    """Test deeply nested structure with control flow and parallel blocks."""
    optimizer = ParallelOptimizer()

    prog = Main(
        q := QReg("q", 4),
        c := CReg("m", 4),
        Block(
            If(c[0] == 1).Then(
                Block(
                    Parallel(
                        Block(
                            qb.H(q[0]),
                            If(c[1] == 1).Then(qb.X(q[1])),
                        ),
                        qb.H(q[2]),
                    ),
                ),
            ),
        ),
    )

    optimized = optimizer.transform(prog)

    # Navigate to the inner Parallel
    outer_block = optimized.ops[0]
    assert isinstance(outer_block, Block)

    if_block = outer_block.ops[0]
    assert isinstance(if_block, If)

    # If's then operations are in if_block.ops
    inner_block = if_block.ops[0]
    assert isinstance(inner_block, Block)

    # Inner Parallel should not be optimized due to control flow
    parallel = inner_block.ops[0]
    assert isinstance(parallel, Parallel)


def test_barrier_behavior() -> None:
    """Test that barriers could act as optimization boundaries (future enhancement)."""
    optimizer = ParallelOptimizer()

    from pecos.slr import Barrier

    prog = Main(
        q := QReg("q", 4),
        Parallel(
            qb.H(q[0]),
            qb.H(q[1]),
            Barrier(q[0], q[1]),  # Could act as barrier in future
            qb.X(q[2]),
            qb.X(q[3]),
        ),
    )

    optimized = optimizer.transform(prog)

    # Current implementation treats Barrier as regular operation
    # Future enhancement could use it as optimization boundary
    assert len(optimized.ops) == 1


def test_measurement_dependencies() -> None:
    """Test handling of measurement dependencies."""
    optimizer = ParallelOptimizer()

    prog = Main(
        q := QReg("q", 4),
        c := CReg("m", 4),
        Parallel(
            qb.H(q[0]),
            qb.Measure(q[0]) > c[0],
            If(c[0] == 1).Then(qb.X(q[1])),  # Depends on measurement
            qb.H(q[2]),  # Independent
        ),
    )

    optimized = optimizer.transform(prog)

    # Should not optimize due to control flow
    assert len(optimized.ops) == 1
    assert isinstance(optimized.ops[0], Parallel)
