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

"""QASM regression tests for control flow structures."""

from collections.abc import Callable

from pecos.qeclib import qubit as qb
from pecos.slr import Block, CReg, If, Main, Parallel, QReg, Repeat


def test_phys_teleport(compare_qasm: Callable[..., None]) -> None:
    """Test basic physical teleportation circuit QASM regression."""
    prog = Main(
        q := QReg("q", 2),
        c := CReg("m", 2),
        qb.H(q[0]),
        qb.CX(q[0], q[1]),
        qb.Measure(q) > c,
    )

    compare_qasm(prog, filename="phys.teleport")


def test_phys_tele_block_block(compare_qasm: Callable[..., None]) -> None:
    """Test teleportation with nested block structure QASM regression."""
    prog = Main(
        q := QReg("q", 2),
        c := CReg("m", 2),
        qb.H(q[0]),
        qb.CX(q[0], q[1]),
        qb.Measure(q) > c,
        Block(
            qb.H(q[0]),
            Block(
                qb.H(q[1]),
            ),
        ),
    )

    compare_qasm(prog, filename="phys.tele_block_block")


def test_phys_tele_if(compare_qasm: Callable[..., None]) -> None:
    """Test teleportation with conditional statements QASM regression."""
    prog = Main(
        q := QReg("q", 2),
        c := CReg("m", 2),
        qb.H(q[0]),
        qb.CX(q[0], q[1]),
        qb.Measure(q) > c,
        If(c == 0).Then(
            qb.H(q[0]),
        ),
    )

    compare_qasm(prog, filename="phys.tele_if")


def test_phys_tele_if_block_block(compare_qasm: Callable[..., None]) -> None:
    """Test teleportation with conditional and nested blocks QASM regression."""
    prog = Main(
        q := QReg("q", 2),
        c := CReg("m", 2),
        qb.H(q[0]),
        qb.CX(q[0], q[1]),
        qb.Measure(q) > c,
        If(c == 0).Then(
            qb.H(q[0]),
            Block(
                qb.H(q[1]),
            ),
        ),
    )

    compare_qasm(prog, filename="phys.tele_if_block_block")


def test_phys_tele_block_telep_block(compare_qasm: Callable[..., None]) -> None:
    """Test complex teleportation with multiple nested blocks QASM regression."""
    prog = Main(
        q := QReg("q", 2),
        c := CReg("m", 2),
        c2 := CReg("m2", 2),
        qb.H(q[0]),
        qb.CX(q[0], q[1]),
        qb.Measure(q) > c,
        Block(
            qb.Prep(q),
            qb.H(q[0]),
            qb.CX(q[0], q[1]),
            qb.Measure(q) > c2,
            Block(
                qb.H(q[0]),
            ),
        ),
    )

    compare_qasm(prog, filename="phys.tele_block_telep_block")


def test_phys_repeat(compare_qasm: Callable[..., None]) -> None:
    """Test teleportation with repeat blocks QASM regression."""
    prog = Main(
        q := QReg("q", 2),
        c := CReg("m", 2),
        Repeat(3).block(
            qb.H(q[0]),
            qb.CX(q[0], q[1]),
            qb.Measure(q) > c,
        ),
    )

    compare_qasm(prog, filename="phys.tele_repeat")


def test_phys_parallel() -> None:
    """Test parallel block QASM generation."""
    from pecos.slr import SlrConverter

    prog = Main(
        q := QReg("q", 4),
        c := CReg("m", 4),
        Parallel(
            qb.H(q[0]),
            qb.H(q[1]),
            qb.X(q[2]),
            qb.Y(q[3]),
        ),
        qb.Measure(q) > c,
    )

    qasm = SlrConverter(prog).qasm()

    # Verify all operations are present in the generated QASM
    assert "h q[0]" in qasm
    assert "h q[1]" in qasm
    assert "x q[2]" in qasm
    assert "y q[3]" in qasm
    # QASM generator uses compact notation for register-wide measurements
    assert "measure q -> m;" in qasm


def test_phys_nested_parallel() -> None:
    """Test nested parallel blocks QASM generation."""
    from pecos.slr import SlrConverter

    prog = Main(
        q := QReg("q", 4),
        c := CReg("m", 4),
        Parallel(
            qb.H(q[0]),
            Block(
                qb.X(q[1]),
                qb.Y(q[2]),
            ),
            qb.Z(q[3]),
        ),
        qb.Measure(q) > c,
    )

    qasm = SlrConverter(prog).qasm()

    # Verify all operations are present
    assert "h q[0]" in qasm
    assert "x q[1]" in qasm
    assert "y q[2]" in qasm
    assert "z q[3]" in qasm
    assert "qreg q[4]" in qasm
    assert "creg m[4]" in qasm


def test_phys_parallel_in_if() -> None:
    """Test parallel block inside conditional QASM generation."""
    from pecos.slr import SlrConverter

    prog = Main(
        q := QReg("q", 4),
        c := CReg("m", 4),
        qb.H(q[0]),
        qb.Measure(q[0]) > c[0],
        If(c[0] == 1).Then(
            Parallel(
                qb.X(q[1]),
                qb.Y(q[2]),
                qb.Z(q[3]),
            ),
        ),
        qb.Measure(q[1:4]) > c[1:4],
    )

    qasm = SlrConverter(prog).qasm()

    # Verify the conditional structure
    assert "h q[0]" in qasm
    assert "measure q[0] -> m[0]" in qasm
    # QASM generator applies conditions to individual operations
    assert "if(m[0] == 1) x q[1]" in qasm
    assert "if(m[0] == 1) y q[2]" in qasm
    assert "if(m[0] == 1) z q[3]" in qasm
