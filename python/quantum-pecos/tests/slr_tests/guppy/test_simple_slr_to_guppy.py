"""Simple SLR-to-Guppy translation tests.

The straightforward Bell / GHZ / two-qubit-gate / multi-op tests in the
legacy file are covered by the v1 acceptance corpus
(``tests/slr_tests/ast_guppy/test_v1_acceptance.py``). What survives
here is the measure-then-PZ cycle invoked multiple times via a
Block subclass, which is in v1 scope but not part of the acceptance
set.
"""

from pecos.slr import Block, CReg, Main, QReg, Return
from pecos.slr.qeclib import qubit as qb
from pecos.slr.qeclib.qubit.measures import Measure
from pecos.slr.qeclib.qubit.preps import PZ

from ..ast_guppy._harness import assert_ast_guppy_compiles  # noqa: TID252


def test_simple_explicit_reset_in_loop() -> None:
    """A Block performing measure + PZ is invoked three times in a row."""

    class ResetQubit(Block):
        def __init__(self, q: QReg, c: CReg) -> None:
            super().__init__()
            self.q = q
            self.c = c
            self.ops = [
                Measure(q[0]) > c[0],
                PZ(q[0]),
            ]

    prog = Main(
        q := QReg("q", 1),
        c := CReg("c", 3),
        ResetQubit(q, c[0:1]),
        ResetQubit(q, c[1:2]),
        ResetQubit(q, c[2:3]),
        Return(c),
    )
    assert_ast_guppy_compiles(prog)


def test_simple_measurement_then_reset() -> None:
    """Measure-then-reset followed by a root-level X on the freshly-prepped slot."""

    class MeasureAndReset(Block):
        def __init__(self, q: QReg, c: CReg) -> None:
            super().__init__()
            self.q = q
            self.c = c
            self.ops = [
                Measure(q[0]) > c[0],
                PZ(q[0]),
            ]

    prog = Main(
        q := QReg("q", 1),
        c := CReg("c", 1),
        MeasureAndReset(q, c),
        qb.X(q[0]),
        Return(c),
    )
    assert_ast_guppy_compiles(prog)
