"""Tests for partial array consumption patterns.

The straight-line measurement variants are covered by the v1 acceptance
corpus (``tests/slr_tests/ast_guppy/test_v1_acceptance.py``). The cases
here exercise Block flattening + measurement patterns and the empty-
Main / no-measurement edge cases that the acceptance corpus does not
cover.
"""

import pytest
from pecos.slr import Block, CReg, Main, QReg, Return, SlrConverter
from pecos.slr.qeclib import qubit
from pecos.slr.qeclib.qubit.measures import Measure

from ..ast_guppy._harness import assert_ast_guppy_compiles  # noqa: TID252


class TestPartialConsumption:
    """Test cases for partial quantum array consumption."""

    def test_measure_ancillas_preserve_data(self) -> None:
        """Block measures every ancilla; data is consumed at root after a gate."""

        class MeasureAncillas(Block):
            def __init__(self, data: QReg, ancilla: QReg, syndrome: CReg) -> None:
                super().__init__()
                self.data = data
                self.ancilla = ancilla
                self.syndrome = syndrome
                self.ops = [
                    Measure(ancilla[0]) > syndrome[0],
                    Measure(ancilla[1]) > syndrome[1],
                    Measure(ancilla[2]) > syndrome[2],
                    Measure(ancilla[3]) > syndrome[3],
                ]

        prog = Main(
            data := QReg("data", 7),
            ancilla := QReg("ancilla", 4),
            syndrome := CReg("syndrome", 4),
            data_result := CReg("data_result", 7),
            qubit.H(data[0]),
            qubit.CX(data[0], ancilla[0]),
            MeasureAncillas(data, ancilla, syndrome),
            qubit.X(data[0]),
            Measure(data) > data_result,
            Return(syndrome, data_result),
        )
        assert_ast_guppy_compiles(prog)

    def test_consume_subset_of_qubits(self) -> None:
        """Block consumes a subset of slots; root consumes the rest."""

        class MeasureFirstHalf(Block):
            def __init__(self, qubits: QReg, results: CReg) -> None:
                super().__init__()
                self.qubits = qubits
                self.results = results
                self.ops = [
                    Measure(qubits[0]) > results[0],
                    Measure(qubits[1]) > results[1],
                    Measure(qubits[2]) > results[2],
                ]

        prog = Main(
            q := QReg("q", 6),
            c_first := CReg("c_first", 3),
            c_second := CReg("c_second", 3),
            qubit.H(q[0]),
            qubit.CX(q[0], q[1]),
            MeasureFirstHalf(q, c_first),
            qubit.H(q[3]),
            Measure(q[3]) > c_second[0],
            Measure(q[4]) > c_second[1],
            Measure(q[5]) > c_second[2],
            Return(c_first, c_second),
        )
        assert_ast_guppy_compiles(prog)

    def test_block_operations_flattened(self) -> None:
        """Stabilizer-style Block + post-block gate on data qubits."""

        class StabilizerMeasurement(Block):
            def __init__(self, data: QReg, ancilla: QReg, syndrome: CReg) -> None:
                super().__init__()
                self.data = data
                self.ancilla = ancilla
                self.syndrome = syndrome
                self.ops = [
                    qubit.H(ancilla[0]),
                    qubit.CX(data[0], ancilla[0]),
                    qubit.CX(data[1], ancilla[0]),
                    qubit.H(ancilla[0]),
                    Measure(ancilla[0]) > syndrome[0],
                ]

        prog = Main(
            data := QReg("data", 2),
            ancilla := QReg("ancilla", 1),
            syndrome := CReg("syndrome", 1),
            final := CReg("final", 2),
            StabilizerMeasurement(data, ancilla, syndrome),
            qubit.Z(data[0]),
            Measure(data) > final,
            Return(syndrome, final),
        )
        assert_ast_guppy_compiles(prog)

    def test_single_element_block(self) -> None:
        """Block with a single statement still compiles after flattening."""

        class MeasureSingle(Block):
            def __init__(self, q: QReg, c: CReg) -> None:
                super().__init__()
                self.q = q
                self.c = c
                self.ops = [
                    Measure(q[0]) > c[0],
                ]

        prog = Main(
            single := QReg("single", 1),
            result := CReg("result", 1),
            MeasureSingle(single, result),
            Return(result),
        )
        assert_ast_guppy_compiles(prog)

    @pytest.mark.optional_dependency
    def test_hugr_compilation(self) -> None:
        """Test that patterns compile to HUGR."""
        prog = Main(
            q := QReg("q", 3),
            c := CReg("c", 3),
            # Individual measurements
            Measure(q[0]) > c[0],
            Measure(q[1]) > c[1],
            Measure(q[2]) > c[2],
            Return(c),
        )

        # This should compile without errors
        try:
            hugr = SlrConverter(prog).hugr()
            assert hugr is not None
        except ImportError as e:
            pytest.fail(f"HUGR compilation failed: {e}")


class TestEdgeCases:
    """Test edge cases and corner cases for the v1 emitter."""

    def test_empty_block(self) -> None:
        """A Block with an empty op list flattens to nothing."""

        class DoNothing(Block):
            def __init__(self, q: QReg) -> None:
                super().__init__()
                self.q = q
                self.ops = []

        prog = Main(
            q := QReg("q", 2),
            c := CReg("c", 2),
            DoNothing(q),
            Measure(q) > c,
            Return(c),
        )
        assert_ast_guppy_compiles(prog)

    def test_unmeasured_qubits_returned(self) -> None:
        """Main with no measurements: live qubits are discarded at exit."""
        prog = Main(
            q := QReg("q", 2),
            qubit.H(q[0]),
            qubit.CX(q[0], q[1]),
        )
        assert_ast_guppy_compiles(prog)
