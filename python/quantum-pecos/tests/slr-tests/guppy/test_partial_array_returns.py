"""Tests for functions returning partial arrays."""

from pecos.qeclib import qubit
from pecos.qeclib.qubit.measures import Measure
from pecos.slr import Block, CReg, Main, QReg, SlrConverter


def test_function_returns_unconsumed_qubits() -> None:
    """Test that functions properly return unconsumed qubits."""

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

    # Check function signature
    assert "-> array[quantum.qubit, 2]:" in guppy
    # Array may be unpacked for element access, then reconstructed for return
    assert "return data" in guppy or "return array(data_" in guppy

    # Check function call captures return
    assert "data = test_partial_array_returns_measure_ancillas" in guppy


def test_partial_array_return() -> None:
    """Test function that returns subset of input array."""

    class SelectEvenQubits(Block):
        """Process array, return only even-indexed qubits."""

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
        # This is the current behavior - still references original array
        # TODO: Should use returned partial array
        Measure(q[0]) > result[0],
        Measure(q[2]) > result[1],
    )

    guppy = SlrConverter(prog).guppy()

    # Check function returns partial array
    assert "-> array[quantum.qubit, 2]:" in guppy
    assert "return array(" in guppy

    # The function should return array with q[0] and q[2]
    # After unpacking, returns array(q_0, q_2)
    assert (
        "array(q[0], q[2])" in guppy
        or "array(_q_0, _q_2)" in guppy
        or "array(q_0, q_2)" in guppy
    )


def test_multiple_partial_returns() -> None:
    """Test function returning multiple partial arrays."""

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

    # Function should return both partial arrays
    assert "-> tuple[array[quantum.qubit, 1], array[quantum.qubit, 1]]:" in guppy

    # Should construct and return both arrays
    assert "return " in guppy
    # Should see array construction for remaining qubits


def test_no_return_when_all_consumed() -> None:
    """Test that functions consuming all qubits return None."""

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

    # Function should return None
    assert "-> None:" in guppy
    # When all qubits are consumed, no explicit return statement needed


def test_qec_pattern_with_partial_returns() -> None:
    """Test realistic QEC pattern using partial returns."""

    class StabilizerRound(Block):
        """Perform one round of stabilizer measurements."""

        def __init__(self, data: QReg, ancilla: QReg, syndrome: CReg) -> None:
            super().__init__()
            self.data = data
            self.ancilla = ancilla
            self.syndrome = syndrome
            self.ops = [
                # Reset ancillas
                # Note: Reset operation might not be available, using fresh qubits instead
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
        # Second round
        StabilizerRound(data, ancilla, syndrome2),
        # Final measurement
        Measure(data) > final,
    )

    guppy = SlrConverter(prog).guppy()

    # Function should be generated
    assert "stabilizer_round" in guppy
    # Function parameters should include data array
    assert "data: array[quantum.qubit, 3]" in guppy

    # Should return data array since ancilla is consumed
    assert "-> array[quantum.qubit, 3]:" in guppy
    # Array may be unpacked for element access, then reconstructed for return
    assert "return data" in guppy or "return array(data_" in guppy

    # Main should capture returned data
    assert "data = test_partial_array_returns_stabilizer_round(ancilla, data" in guppy
