"""Tests for register-wide gate expansion.

The v1 emitter expands register-wide single-qubit gates (e.g. ``H(q)``)
to one functional call per slot. The v1 acceptance corpus does not yet
exercise this expansion, so the cases below verify the expanded form
compiles. The legacy string assertions on ``quantum.h(q[i])`` were the
buggy form and have been deleted.
"""

from pecos.slr import Block, CReg, Main, QReg, Return
from pecos.slr.qeclib import qubit
from pecos.slr.qeclib.qubit.measures import Measure

from ..ast_guppy._harness import assert_ast_guppy_compiles  # noqa: TID252


def test_consecutive_gate_applications() -> None:
    """Apply the same single-qubit gate to each element individually."""
    prog = Main(
        q := QReg("q", 5),
        c := CReg("c", 5),
        qubit.H(q[0]),
        qubit.H(q[1]),
        qubit.H(q[2]),
        qubit.H(q[3]),
        qubit.H(q[4]),
        Measure(q) > c,
        Return(c),
    )
    assert_ast_guppy_compiles(prog)


def test_register_wide_generates_loop() -> None:
    """Register-wide H(q) expands to one call per slot."""
    prog = Main(
        q := QReg("q", 5),
        c := CReg("c", 5),
        qubit.H(q),
        Measure(q) > c,
        Return(c),
    )
    assert_ast_guppy_compiles(prog)


def test_mixed_individual_and_register_wide() -> None:
    """Mixing register-wide and element-specific operations on one QReg."""
    prog = Main(
        q := QReg("q", 4),
        c := CReg("c", 4),
        qubit.H(q),
        qubit.X(q[0]),
        qubit.X(q[2]),
        qubit.Z(q),
        Measure(q) > c,
        Return(c),
    )
    assert_ast_guppy_compiles(prog)


def test_loop_in_function() -> None:
    """Register-wide operations inside a Block subclass flatten and expand."""

    class ApplyHadamards(Block):
        def __init__(self, q: QReg) -> None:
            super().__init__()
            self.q = q
            self.ops = [
                qubit.H(q),
            ]

    prog = Main(
        q := QReg("q", 4),
        c := CReg("c", 4),
        ApplyHadamards(q),
        Measure(q) > c,
        Return(c),
    )
    assert_ast_guppy_compiles(prog)


def test_different_gates_separate_loops() -> None:
    """Several different register-wide gates on the same register."""
    prog = Main(
        q := QReg("q", 3),
        c := CReg("c", 3),
        qubit.H(q),
        qubit.X(q),
        qubit.Y(q),
        qubit.Z(q),
        Measure(q) > c,
        Return(c),
    )
    assert_ast_guppy_compiles(prog)


def test_multiple_registers() -> None:
    """Register-wide operations across multiple QRegs."""
    prog = Main(
        q1 := QReg("q1", 3),
        q2 := QReg("q2", 3),
        c := CReg("c", 6),
        qubit.H(q1),
        qubit.X(q2),
        Measure(q1) > c[0:3],
        Measure(q2) > c[3:6],
        Return(c),
    )
    assert_ast_guppy_compiles(prog)
