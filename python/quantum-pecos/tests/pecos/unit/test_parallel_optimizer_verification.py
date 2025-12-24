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

"""Verification tests showing exact transformations performed by ParallelOptimizer."""

from pecos.qeclib import qubit as qb
from pecos.slr import Block, Main, Parallel, QReg
from pecos.slr.transforms import ParallelOptimizer


def test_exact_bell_state_transformation() -> None:
    """Test the exact transformation described in the documentation."""
    optimizer = ParallelOptimizer()

    # Documentation of the transformation logic
    #     Block(H(q[0]), CX(q[0], q[1])),
    #     Block(H(q[2]), CX(q[2], q[3])),
    #     Block(H(q[4]), CX(q[4], q[5]))
    # )
    prog = Main(
        q := QReg("q", 6),
        Parallel(
            Block(
                qb.H(q[0]),
                qb.CX(q[0], q[1]),
            ),
            Block(
                qb.H(q[2]),
                qb.CX(q[2], q[3]),
            ),
            Block(
                qb.H(q[4]),
                qb.CX(q[4], q[5]),
            ),
        ),
    )

    optimized = optimizer.transform(prog)

    # After optimization:
    # Block(
    #     Parallel(H(q[0]), H(q[2]), H(q[4])),        # All H gates
    #     Parallel(CX(q[0],q[1]), CX(q[2],q[3]), CX(q[4],q[5]))  # All CX gates
    # )

    # Verify structure
    assert len(optimized.ops) == 1
    outer_block = optimized.ops[0]
    assert isinstance(outer_block, Block)

    # Should have exactly 2 groups
    assert len(outer_block.ops) == 2

    # First group: All H gates in parallel
    h_group = outer_block.ops[0]
    assert isinstance(h_group, Parallel)
    assert len(h_group.ops) == 3
    assert all(isinstance(op, qb.H) for op in h_group.ops)

    # Check specific qubits for H gates
    h_qubits = [op.qargs[0].index for op in h_group.ops]
    assert sorted(h_qubits) == [0, 2, 4]

    # Second group: All CX gates in parallel
    cx_group = outer_block.ops[1]
    assert isinstance(cx_group, Parallel)
    assert len(cx_group.ops) == 3
    assert all(isinstance(op, qb.CX) for op in cx_group.ops)

    # Check specific qubits for CX gates
    cx_pairs = [(op.qargs[0].index, op.qargs[1].index) for op in cx_group.ops]
    assert sorted(cx_pairs) == [(0, 1), (2, 3), (4, 5)]


def test_visual_transformation_output() -> None:
    """Test the structure of the optimized output for three Bell pairs."""
    optimizer = ParallelOptimizer()

    # Create three independent Bell pairs
    prog = Main(
        q := QReg("q", 6),
        Parallel(
            Block(qb.H(q[0]), qb.CX(q[0], q[1])),
            Block(qb.H(q[2]), qb.CX(q[2], q[3])),
            Block(qb.H(q[4]), qb.CX(q[4], q[5])),
        ),
    )

    optimized = optimizer.transform(prog)

    # Verify the optimized structure
    assert len(optimized.ops) == 1
    outer_block = optimized.ops[0]
    assert isinstance(outer_block, Block)

    # Should have exactly 2 Parallel groups (one for H gates, one for CX gates)
    assert (
        len(outer_block.ops) == 2
    ), f"Expected 2 parallel groups, got {len(outer_block.ops)}"

    # First group should be Parallel with 3 H gates
    first_group = outer_block.ops[0]
    assert isinstance(first_group, Parallel), "First group should be Parallel"
    assert len(first_group.ops) == 3, "First group should have 3 H gates"

    # Check all operations in first group are H gates on even qubits
    for i, op in enumerate(first_group.ops):
        assert (
            type(op).__name__ == "H"
        ), f"Operation {i} in first group should be H gate"
        assert op.qargs[0].index == i * 2, f"H gate {i} should be on qubit {i * 2}"

    # Second group should be Parallel with 3 CX gates
    second_group = outer_block.ops[1]
    assert isinstance(second_group, Parallel), "Second group should be Parallel"
    assert len(second_group.ops) == 3, "Second group should have 3 CX gates"

    # Check all operations in second group are CX gates with correct qubit pairs
    for i, op in enumerate(second_group.ops):
        assert (
            type(op).__name__ == "CX"
        ), f"Operation {i} in second group should be CX gate"
        assert (
            op.qargs[0].index == i * 2
        ), f"CX gate {i} control should be on qubit {i * 2}"
        assert (
            op.qargs[1].index == i * 2 + 1
        ), f"CX gate {i} target should be on qubit {i * 2 + 1}"

    # The transformation successfully converts:
    # Main(
    #   Parallel(
    #     Block(H(q[0]), CX(q[0], q[1])),
    #     Block(H(q[2]), CX(q[2], q[3])),
    #     Block(H(q[4]), CX(q[4], q[5]))
    #   )
    # )
    # Into:
    # Main(
    #   Block(
    #     Parallel(H(q[0]), H(q[2]), H(q[4])),
    #     Parallel(CX(q[0], q[1]), CX(q[2], q[3]), CX(q[4], q[5]))
    #   )
    # )


def test_mixed_gates_transformation() -> None:
    """Test transformation with different gate types to show grouping."""
    optimizer = ParallelOptimizer()

    prog = Main(
        q := QReg("q", 8),
        Parallel(
            qb.H(q[0]),
            qb.X(q[1]),
            qb.H(q[2]),
            qb.Y(q[3]),
            qb.H(q[4]),
            qb.Z(q[5]),
            qb.X(q[6]),
            qb.Y(q[7]),
        ),
    )

    optimized = optimizer.transform(prog)

    # Should group by gate type
    assert len(optimized.ops) == 1
    outer_block = optimized.ops[0]
    assert isinstance(outer_block, Block)

    # Find groups and verify ordering
    groups = outer_block.ops
    gate_types = []
    gate_counts = []

    for group in groups:
        if isinstance(group, Parallel):
            # Multiple gates of same type
            types_in_group = {type(op).__name__ for op in group.ops}
            assert len(types_in_group) == 1
            gate_types.append(next(iter(types_in_group)))
            gate_counts.append(len(group.ops))
        else:
            # Single gate (not wrapped in Parallel)
            gate_types.append(type(group).__name__)
            gate_counts.append(1)

    # H gates should come first, then X, then Y, then Z (based on the ordering in _group_operations)
    assert gate_types == ["H", "X", "Y", "Z"]

    # Verify gate counts
    assert gate_counts[0] == 3  # 3 H gates
    assert gate_counts[1] == 2  # 2 X gates
    assert gate_counts[2] == 2  # 2 Y gates
    assert gate_counts[3] == 1  # 1 Z gate


def test_dependent_operations_not_reordered() -> None:
    """Test that dependent operations maintain their order."""
    optimizer = ParallelOptimizer()

    prog = Main(
        q := QReg("q", 3),
        Parallel(
            qb.H(q[0]),
            qb.CX(q[0], q[1]),  # Depends on H(q[0])
            qb.CX(q[1], q[2]),  # Depends on CX(q[0], q[1])
        ),
    )

    optimized = optimizer.transform(prog)

    # Due to dependencies, operations should maintain order
    assert len(optimized.ops) == 1
    outer_block = optimized.ops[0]
    assert isinstance(outer_block, Block)

    # Should have 3 separate operations (no parallelization possible due to dependencies)
    assert len(outer_block.ops) == 3

    # First should be H
    assert isinstance(outer_block.ops[0], qb.H)

    # Then CX operations in order
    assert isinstance(outer_block.ops[1], qb.CX)
    assert outer_block.ops[1].qargs[0].index == 0
    assert outer_block.ops[1].qargs[1].index == 1

    assert isinstance(outer_block.ops[2], qb.CX)
    assert outer_block.ops[2].qargs[0].index == 1
    assert outer_block.ops[2].qargs[1].index == 2
