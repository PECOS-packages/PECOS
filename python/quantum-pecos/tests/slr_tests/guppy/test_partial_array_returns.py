"""Tests for partial array patterns through Block flattening.

The two-round-stabilizer pattern from the legacy file used the same
ancilla register across two rounds without an intervening ``PZ``.
The v1 AST emitter rejects use-after-measurement, so that case is
deleted; the remaining tests cover Block-flattening + partial-
measurement patterns that v1 supports but that are not part of the v1
acceptance set in ``tests/slr_tests/ast_guppy/test_v1_acceptance.py``.
"""

from pecos.slr import Block, CReg, Main, QReg, Return
from pecos.slr.qeclib import qubit
from pecos.slr.qeclib.qubit.measures import Measure

from ..ast_guppy._harness import assert_ast_guppy_compiles  # noqa: TID252


def test_block_with_partial_measurements() -> None:
    """Block measures ancillas; data qubits are consumed at the root level."""

    class MeasureAncillas(Block):
        def __init__(self, data: QReg, ancilla: QReg, syndrome: CReg) -> None:
            super().__init__()
            self.data = data
            self.ancilla = ancilla
            self.syndrome = syndrome
            self.ops = [
                qubit.CX(data[0], ancilla[0]),
                qubit.CX(data[1], ancilla[1]),
                Measure(ancilla[0]) > syndrome[0],
                Measure(ancilla[1]) > syndrome[1],
            ]

    prog = Main(
        data := QReg("data", 2),
        ancilla := QReg("ancilla", 2),
        syndrome := CReg("syndrome", 2),
        final := CReg("final", 2),
        MeasureAncillas(data, ancilla, syndrome),
        Measure(data) > final,
        Return(syndrome, final),
    )
    assert_ast_guppy_compiles(prog)


def test_partial_array_operations() -> None:
    """Block measures odd indices for discard, root consumes the even ones."""

    class SelectEvenQubits(Block):
        def __init__(self, q: QReg) -> None:
            super().__init__()
            self.q = q
            self.ops = [
                qubit.H(q[0]),
                qubit.H(q[1]),
                qubit.H(q[2]),
                qubit.H(q[3]),
                Measure(q[1]),
                Measure(q[3]),
            ]

    prog = Main(
        q := QReg("q", 4),
        result := CReg("result", 2),
        SelectEvenQubits(q),
        Measure(q[0]) > result[0],
        Measure(q[2]) > result[1],
        Return(result),
    )
    assert_ast_guppy_compiles(prog)


def test_multiple_blocks_with_measurements() -> None:
    """Block consumes one slot per QReg; root consumes the rest."""

    class SplitAndMeasure(Block):
        def __init__(self, a: QReg, b: QReg, results: CReg) -> None:
            super().__init__()
            self.a = a
            self.b = b
            self.results = results
            self.ops = [
                Measure(a[0]) > results[0],
                Measure(b[0]) > results[1],
            ]

    prog = Main(
        a := QReg("a", 2),
        b := QReg("b", 2),
        results := CReg("results", 4),
        SplitAndMeasure(a, b, results[0:2]),
        Measure(a[1]) > results[2],
        Measure(b[1]) > results[3],
        Return(results),
    )
    assert_ast_guppy_compiles(prog)


def test_all_qubits_consumed() -> None:
    """Block consumes every qubit in the QReg; root has no remainder."""

    class MeasureAll(Block):
        def __init__(self, q: QReg, c: CReg) -> None:
            super().__init__()
            self.q = q
            self.c = c
            self.ops = [
                Measure(q[0]) > c[0],
                Measure(q[1]) > c[1],
            ]

    prog = Main(
        q := QReg("q", 2),
        c := CReg("c", 2),
        MeasureAll(q, c),
        Return(c),
    )
    assert_ast_guppy_compiles(prog)
