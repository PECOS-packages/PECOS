"""Realistic measurement patterns built from v1 supported features.

The straightforward measurement coverage (full-register, partial,
selective, conditionals on measurement results) is in
``tests/slr_tests/ast_guppy/test_v1_acceptance.py``. Surviving here is
the QEC syndrome-extraction pattern, which combines Block flattening,
register-wide measurement, and If-driven Pauli corrections in one
realistic shape.
"""

from pecos.slr import Block, CReg, If, Main, QReg, Return
from pecos.slr.qeclib import qubit
from pecos.slr.qeclib.qubit.measures import Measure

from ..ast_guppy._harness import assert_ast_guppy_compiles  # noqa: TID252


class TestComplexPatterns:
    """Test complex measurement patterns from real QEC code."""

    def test_syndrome_extraction_pattern(self) -> None:
        """Syndrome extraction Block + per-bit If corrections."""

        class ExtractSyndrome(Block):
            def __init__(self, data: QReg, ancilla: QReg, syndrome: CReg) -> None:
                super().__init__()
                self.data = data
                self.ancilla = ancilla
                self.syndrome = syndrome
                self.ops = [
                    qubit.H(ancilla[0]),
                    qubit.CX(data[0], ancilla[0]),
                    qubit.CX(data[1], ancilla[0]),
                    qubit.CX(data[2], ancilla[0]),
                    qubit.H(ancilla[0]),
                    qubit.H(ancilla[1]),
                    qubit.CX(data[3], ancilla[1]),
                    qubit.CX(data[4], ancilla[1]),
                    qubit.CX(data[5], ancilla[1]),
                    qubit.H(ancilla[1]),
                    Measure(ancilla) > syndrome,
                ]

        prog = Main(
            data := QReg("data", 7),
            ancilla := QReg("ancilla", 2),
            syndrome := CReg("syndrome", 2),
            ExtractSyndrome(data, ancilla, syndrome),
            If(syndrome[0]).Then(
                qubit.X(data[0]),
            ),
            If(syndrome[1]).Then(
                qubit.X(data[3]),
            ),
            Return(syndrome),
        )
        assert_ast_guppy_compiles(prog)
