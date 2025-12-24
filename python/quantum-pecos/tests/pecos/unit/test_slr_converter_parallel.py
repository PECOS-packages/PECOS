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

"""Tests for SlrConverter with ParallelOptimizer integration."""

from pecos.qeclib import qubit as qb
from pecos.slr import Block, CReg, Main, Parallel, QReg, SlrConverter


def test_slr_converter_without_optimization() -> None:
    """Test SlrConverter without optimization."""
    prog = Main(
        q := QReg("q", 4),
        Parallel(
            qb.H(q[0]),
            qb.X(q[1]),
            qb.H(q[2]),
            qb.X(q[3]),
        ),
    )

    # Explicitly disable optimization
    qasm = SlrConverter(prog, optimize_parallel=False).qasm()

    # Operations should appear in original order
    ops = [
        line.strip()
        for line in qasm.split("\n")
        if line.strip() and not line.startswith(("OPENQASM", "include", "qreg", "creg"))
    ]
    assert ops == ["h q[0];", "x q[1];", "h q[2];", "x q[3];"]


def test_slr_converter_with_optimization() -> None:
    """Test SlrConverter with optimization (default behavior)."""
    prog = Main(
        q := QReg("q", 4),
        Parallel(
            qb.H(q[0]),
            qb.X(q[1]),
            qb.H(q[2]),
            qb.X(q[3]),
        ),
    )

    # Default: optimization enabled
    qasm = SlrConverter(prog).qasm()

    # Operations should be reordered by type (H gates first, then X gates)
    ops = [
        line.strip()
        for line in qasm.split("\n")
        if line.strip() and not line.startswith(("OPENQASM", "include", "qreg", "creg"))
    ]
    assert ops == ["h q[0];", "h q[2];", "x q[1];", "x q[3];"]


def test_slr_converter_optimization_preserves_semantics() -> None:
    """Test that optimization preserves the semantic meaning of the circuit."""
    prog = Main(
        q := QReg("q", 4),
        c := CReg("m", 4),
        Parallel(
            Block(
                qb.H(q[0]),
                qb.CX(q[0], q[1]),
            ),
            Block(
                qb.H(q[2]),
                qb.CX(q[2], q[3]),
            ),
        ),
        qb.Measure(q) > c,
    )

    # Generate both versions
    qasm_unopt = SlrConverter(prog, optimize_parallel=False).qasm()
    qasm_opt = SlrConverter(prog).qasm()

    # Both should have the same gates
    for gate in [
        "h q[0]",
        "h q[2]",
        "cx q[0], q[1]",
        "cx q[2], q[3]",
        "measure q -> m",
    ]:
        assert gate in qasm_unopt
        assert gate in qasm_opt

    # Optimized version should have H gates before CX gates
    assert qasm_opt.index("h q[0]") < qasm_opt.index("cx q[0], q[1]")
    assert qasm_opt.index("h q[2]") < qasm_opt.index("cx q[2], q[3]")
    assert qasm_opt.index("h q[0]") < qasm_opt.index(
        "cx q[2], q[3]",
    )  # Cross-pair ordering


def test_slr_converter_no_parallel_blocks() -> None:
    """Test that programs without Parallel blocks are unaffected by optimization setting."""
    prog = Main(
        q := QReg("q", 4),
        qb.H(q[0]),
        qb.CX(q[0], q[1]),
        qb.H(q[2]),
        qb.CX(q[2], q[3]),
    )

    # Both should produce identical output since there are no Parallel blocks
    qasm_default = SlrConverter(prog).qasm()
    qasm_no_opt = SlrConverter(prog, optimize_parallel=False).qasm()

    assert qasm_default == qasm_no_opt
