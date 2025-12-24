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

"""Example demonstrating the ParallelOptimizer transformation."""

from pecos.qeclib import qubit as qb
from pecos.slr import Block, CReg, Main, Parallel, QReg, SlrConverter
from pecos.slr.transforms import ParallelOptimizer


def test_parallel_optimization_example() -> None:
    """Example showing how ParallelOptimizer transforms Bell state preparations."""
    # Create a program with three Bell state preparations in parallel
    prog = Main(
        q := QReg("q", 6),
        c := CReg("m", 6),
        Parallel(
            Block(  # Bell pair 1
                qb.H(q[0]),
                qb.CX(q[0], q[1]),
            ),
            Block(  # Bell pair 2
                qb.H(q[2]),
                qb.CX(q[2], q[3]),
            ),
            Block(  # Bell pair 3
                qb.H(q[4]),
                qb.CX(q[4], q[5]),
            ),
        ),
        qb.Measure(q) > c,
    )

    # Generate QASM without optimization
    # qasm_unoptimized = SlrConverter(prog).qasm()
    # print("=== QASM without optimization ===")
    # print(qasm_unoptimized)
    # print()

    # Apply the ParallelOptimizer transformation
    optimizer = ParallelOptimizer()
    optimized_prog = optimizer.transform(prog)

    # Generate QASM with optimization
    qasm_optimized = SlrConverter(optimized_prog).qasm()
    # print("=== QASM with optimization ===")
    # print(qasm_optimized)

    # The optimizer has transformed the structure to:
    # Block(
    #     Parallel(H(q[0]), H(q[2]), H(q[4])),  # All H gates in parallel
    #     Parallel(CX(q[0], q[1]), CX(q[2], q[3]), CX(q[4], q[5])),  # All CX gates in parallel
    # )

    # Verify the operations are grouped by type
    assert "h q[0]" in qasm_optimized
    assert "h q[2]" in qasm_optimized
    assert "h q[4]" in qasm_optimized
    # These should appear before the CX gates
    h_positions = [
        qasm_optimized.index("h q[0]"),
        qasm_optimized.index("h q[2]"),
        qasm_optimized.index("h q[4]"),
    ]
    cx_positions = [
        qasm_optimized.index("cx q[0]"),
        qasm_optimized.index("cx q[2]"),
        qasm_optimized.index("cx q[4]"),
    ]
    assert all(h < cx for h in h_positions for cx in cx_positions)
