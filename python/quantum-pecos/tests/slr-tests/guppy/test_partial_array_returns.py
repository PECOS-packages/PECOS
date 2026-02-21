"""Tests for partial array patterns in Guppy code generation.

Note: The AST codegen flattens Block subclasses into the main function
rather than generating nested functions. These tests verify that blocks
are correctly flattened and all operations are included.
"""

from pecos.slr import Block, CReg, Main, QReg, SlrConverter
from pecos.slr.qeclib import qubit
from pecos.slr.qeclib.qubit.measures import Measure


def test_block_with_partial_measurements() -> None:
    """Test that blocks with partial measurements are flattened correctly."""

    class MeasureAncillas(Block):
        """Measure ancilla qubits, return data qubits."""

        def __init__(self, data: QReg, ancilla: QReg, syndrome: CReg) -> None:
            super().__init__()
            self.data = data
            self.ancilla = ancilla
            self.syndrome = syndrome
            self.ops = [
                # Entangle for syndrome extraction
                qubit.CX(data[0], ancilla[0]),
                qubit.CX(data[1], ancilla[1]),
                # Measure only ancillas
                Measure(ancilla[0]) > syndrome[0],
                Measure(ancilla[1]) > syndrome[1],
                # data qubits remain unmeasured
            ]

    prog = Main(
        data := QReg("data", 2),
        ancilla := QReg("ancilla", 2),
        syndrome := CReg("syndrome", 2),
        final := CReg("final", 2),
        MeasureAncillas(data, ancilla, syndrome),
        # Continue using data qubits
        Measure(data) > final,
    )

    guppy = SlrConverter(prog).guppy()

    # AST codegen flattens blocks into main
    # Check that all operations from the block are present
    assert "quantum.cx(data[0], ancilla[0])" in guppy
    assert "quantum.cx(data[1], ancilla[1])" in guppy
    assert "syndrome_0 = quantum.measure(ancilla[0])" in guppy
    assert "syndrome_1 = quantum.measure(ancilla[1])" in guppy

    # Final measurements should also be present
    assert "final_0 = quantum.measure(data[0])" in guppy
    assert "final_1 = quantum.measure(data[1])" in guppy


def test_partial_array_operations() -> None:
    """Test operations on subsets of arrays."""

    class SelectEvenQubits(Block):
        """Process array, measure odd indices."""

        def __init__(self, q: QReg) -> None:
            super().__init__()
            self.q = q
            self.ops = [
                # Apply gates to all
                qubit.H(q[0]),
                qubit.H(q[1]),
                qubit.H(q[2]),
                qubit.H(q[3]),
                # Measure odd indices
                Measure(q[1]),  # Discard
                Measure(q[3]),  # Discard
                # q[0] and q[2] remain
            ]

    prog = Main(
        q := QReg("q", 4),
        result := CReg("result", 2),
        SelectEvenQubits(q),
        Measure(q[0]) > result[0],
        Measure(q[2]) > result[1],
    )

    guppy = SlrConverter(prog).guppy()

    # AST codegen flattens blocks
    # Check that H gates are applied to all qubits
    assert "quantum.h(q[0])" in guppy
    assert "quantum.h(q[1])" in guppy
    assert "quantum.h(q[2])" in guppy
    assert "quantum.h(q[3])" in guppy

    # Check measurements
    assert "quantum.measure(q[1])" in guppy
    assert "quantum.measure(q[3])" in guppy
    assert "result_0 = quantum.measure(q[0])" in guppy
    assert "result_1 = quantum.measure(q[2])" in guppy


def test_multiple_blocks_with_measurements() -> None:
    """Test multiple blocks with different measurement patterns."""

    class SplitAndMeasure(Block):
        """Split two arrays, measure half of each."""

        def __init__(self, a: QReg, b: QReg, results: CReg) -> None:
            super().__init__()
            self.a = a
            self.b = b
            self.results = results
            self.ops = [
                # Measure first half of each array
                Measure(a[0]) > results[0],
                Measure(b[0]) > results[1],
                # a[1] and b[1] remain
            ]

    prog = Main(
        a := QReg("a", 2),
        b := QReg("b", 2),
        results := CReg("results", 4),
        SplitAndMeasure(a, b, results[0:2]),
        # Use remaining qubits
        Measure(a[1]) > results[2],
        Measure(b[1]) > results[3],
    )

    guppy = SlrConverter(prog).guppy()

    # AST codegen flattens blocks
    # Check measurements from block
    assert "results_0 = quantum.measure(a[0])" in guppy
    assert "results_1 = quantum.measure(b[0])" in guppy

    # Check remaining measurements
    assert "results_2 = quantum.measure(a[1])" in guppy
    assert "results_3 = quantum.measure(b[1])" in guppy


def test_all_qubits_consumed() -> None:
    """Test that blocks consuming all qubits work correctly."""

    class MeasureAll(Block):
        """Measure all input qubits."""

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
    )

    guppy = SlrConverter(prog).guppy()

    # AST codegen flattens blocks
    assert "c_0 = quantum.measure(q[0])" in guppy
    assert "c_1 = quantum.measure(q[1])" in guppy


def test_qec_pattern_flattened() -> None:
    """Test realistic QEC pattern is correctly flattened."""

    class StabilizerRound(Block):
        """Perform one round of stabilizer measurements."""

        def __init__(self, data: QReg, ancilla: QReg, syndrome: CReg) -> None:
            super().__init__()
            self.data = data
            self.ancilla = ancilla
            self.syndrome = syndrome
            self.ops = [
                # Syndrome extraction
                qubit.H(ancilla[0]),
                qubit.CX(data[0], ancilla[0]),
                qubit.CX(data[1], ancilla[0]),
                qubit.H(ancilla[0]),
                qubit.H(ancilla[1]),
                qubit.CX(data[1], ancilla[1]),
                qubit.CX(data[2], ancilla[1]),
                qubit.H(ancilla[1]),
                # Measure ancillas only
                Measure(ancilla) > syndrome,
                # Data qubits preserved
            ]

    prog = Main(
        data := QReg("data", 3),
        ancilla := QReg("ancilla", 2),
        syndrome1 := CReg("syndrome1", 2),
        syndrome2 := CReg("syndrome2", 2),
        final := CReg("final", 3),
        # First round
        StabilizerRound(data, ancilla, syndrome1),
        # Second round (same block used twice)
        StabilizerRound(data, ancilla, syndrome2),
        # Final measurement
        Measure(data) > final,
    )

    guppy = SlrConverter(prog).guppy()

    # AST codegen flattens blocks (operations appear twice for two rounds)
    assert "quantum.h(ancilla[0])" in guppy
    assert "quantum.cx(data[0], ancilla[0])" in guppy
    assert "quantum.cx(data[1], ancilla[0])" in guppy

    # Syndrome measurements for both rounds
    assert "syndrome1_0 = quantum.measure(ancilla[0])" in guppy
    assert "syndrome1_1 = quantum.measure(ancilla[1])" in guppy
    assert "syndrome2_0 = quantum.measure(ancilla[0])" in guppy
    assert "syndrome2_1 = quantum.measure(ancilla[1])" in guppy

    # Final measurements
    assert "final_0 = quantum.measure(data[0])" in guppy
    assert "final_1 = quantum.measure(data[1])" in guppy
    assert "final_2 = quantum.measure(data[2])" in guppy
