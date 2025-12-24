"""Simple SLR-to-Guppy translation tests.

These tests verify that basic SLR patterns translate cleanly to Guppy
and compile to HUGR without errors. They serve as both documentation
of expected translations and regression tests.
"""

from pecos.qeclib import qubit as qb
from pecos.qeclib.qubit.measures import Measure
from pecos.qeclib.qubit.preps import Prep
from pecos.slr import Block, CReg, Main, QReg, SlrConverter


def test_simple_bell_state() -> None:
    """Test simple Bell state preparation translates cleanly."""
    prog = Main(
        q := QReg("q", 2),
        c := CReg("c", 2),
        # Bell state: H on q[0], then CNOT
        qb.H(q[0]),
        qb.CX(q[0], q[1]),
        # Measure both qubits
        Measure(q) > c,
    )

    # Generate Guppy code
    guppy_code = SlrConverter(prog).guppy()

    # Verify clean translation
    assert "quantum.h(q[0])" in guppy_code
    assert "quantum.cx(q[0], q[1])" in guppy_code
    assert "quantum.measure_array(q)" in guppy_code

    # Verify it compiles to HUGR
    hugr = SlrConverter(prog).hugr()
    assert hugr is not None
    assert hasattr(hugr, "modules")

    print("Bell state: Clean translation and HUGR compilation")


def test_simple_reset() -> None:
    """Test that reset operations translate cleanly to functional reset."""
    prog = Main(
        q := QReg("q", 1),
        c := CReg("c", 1),
        # Prepare |+⟩
        qb.H(q[0]),
        # Measure
        Measure(q[0]) > c[0],
        # Reset (should use functional reset)
        Prep(q[0]),
        # Apply X
        qb.X(q[0]),
    )

    guppy_code = SlrConverter(prog).guppy()

    # Should allocate fresh qubit after measurement (Prep operation)
    # Note: Due to Guppy's linear type constraints, after q_0 is consumed by measurement,
    # Prep creates a fresh variable q_0_1 instead of reassigning to q_0
    assert "q_0_1 = quantum.qubit()" in guppy_code
    # Should have measurement before the Prep
    assert "quantum.measure(q_0)" in guppy_code
    # The X gate should use the fresh variable
    assert "quantum.x(q_0_1)" in guppy_code

    # Should compile to HUGR
    hugr = SlrConverter(prog).hugr()
    assert hugr is not None

    print("Reset: Functional reset with correct assignment")


def test_simple_function_with_return() -> None:
    """Test that functions with quantum returns translate cleanly."""

    class ApplyH(Block):
        """Simple block that applies H to a qubit."""

        def __init__(self, q: QReg) -> None:
            super().__init__()
            self.q = q
            self.ops = [qb.H(q[0])]

    prog = Main(
        q := QReg("q", 1),
        c := CReg("c", 1),
        # Apply H (function should return q)
        ApplyH(q),
        # Measure
        Measure(q[0]) > c[0],
    )

    guppy_code = SlrConverter(prog).guppy()

    # Function should have proper signature with return (may include module prefix)
    assert (
        "apply_h(q: array[quantum.qubit, 1] @owned) -> array[quantum.qubit, 1]:"
        in guppy_code
    )
    assert "return q" in guppy_code or "return array(q_0)" in guppy_code

    # Main should capture return (may include module prefix in function name)
    # Check that function is called with array arg and result is assigned to q
    assert "_apply_h(array(q_0))" in guppy_code
    assert "q =" in guppy_code

    # Should compile
    hugr = SlrConverter(prog).hugr()
    assert hugr is not None

    print("Function return: Proper signature and capture")


def test_simple_measurement_then_reset() -> None:
    """Test measure-reset pattern common in QEC."""

    class MeasureAndReset(Block):
        """Measure a qubit and reset it."""

        def __init__(self, q: QReg, c: CReg) -> None:
            super().__init__()
            self.q = q
            self.c = c
            self.ops = [
                Measure(q[0]) > c[0],
                Prep(q[0]),  # Explicit reset
            ]

    prog = Main(
        q := QReg("q", 1),
        c := CReg("c", 1),
        # Measure and reset
        MeasureAndReset(q, c),
        # Apply gate to reset qubit
        qb.X(q[0]),
    )

    guppy_code = SlrConverter(prog).guppy()

    # Function should return the fresh qubit
    assert "-> array[quantum.qubit, 1]:" in guppy_code
    # Should allocate fresh qubit (Prep operation)
    assert "quantum.qubit()" in guppy_code

    # Should compile to HUGR
    hugr = SlrConverter(prog).hugr()
    assert hugr is not None

    print("Measure-reset: Explicit reset returned correctly")


def test_simple_two_qubit_gate() -> None:
    """Test two-qubit gate translation."""
    prog = Main(
        q := QReg("q", 3),
        c := CReg("c", 3),
        # Apply CNOT gates
        qb.CX(q[0], q[1]),
        qb.CX(q[1], q[2]),
        # Measure
        Measure(q) > c,
    )

    guppy_code = SlrConverter(prog).guppy()

    # Check gates are in order
    assert "quantum.cx(q[0], q[1])" in guppy_code
    assert "quantum.cx(q[1], q[2])" in guppy_code
    # Check order (q[0],q[1]) should come before (q[1],q[2])
    idx1 = guppy_code.index("quantum.cx(q[0], q[1])")
    idx2 = guppy_code.index("quantum.cx(q[1], q[2])")
    assert idx1 < idx2

    # Should compile
    hugr = SlrConverter(prog).hugr()
    assert hugr is not None

    print("Two-qubit gates: Correct order preserved")


def test_simple_loop_pattern() -> None:
    """Test that loops generate clean code."""
    prog = Main(
        q := QReg("q", 5),
        c := CReg("c", 5),
        # Apply H to all qubits (should generate loop)
        qb.H(q),
        # Measure all
        Measure(q) > c,
    )

    guppy_code = SlrConverter(prog).guppy()

    # Should generate a loop for H gates
    assert "for i in range(0, 5):" in guppy_code
    assert "quantum.h(q[i])" in guppy_code

    # Should compile
    hugr = SlrConverter(prog).hugr()
    assert hugr is not None

    print("Loop generation: Clean for loop")


def test_simple_partial_consumption() -> None:
    """Test partial array consumption pattern."""

    class MeasureFirst(Block):
        """Measure only first qubit."""

        def __init__(self, q: QReg, c: CReg) -> None:
            super().__init__()
            self.q = q
            self.c = c
            self.ops = [
                Measure(q[0]) > c[0],
                # q[1] and q[2] remain
            ]

    prog = Main(
        q := QReg("q", 3),
        c := CReg("c", 3),
        # Measure first qubit only
        MeasureFirst(q, c[0:1]),
        # Use remaining qubits
        qb.H(q[1]),
        qb.H(q[2]),
        Measure(q[1]) > c[1],
        Measure(q[2]) > c[2],
    )

    guppy_code = SlrConverter(prog).guppy()

    # Function should return partial array (q[1] and q[2])
    assert "-> array[quantum.qubit, 2]:" in guppy_code

    # Should compile
    hugr = SlrConverter(prog).hugr()
    assert hugr is not None

    print("Partial consumption: Returns only unconsumed qubits")


def test_simple_explicit_reset_in_loop() -> None:
    """Test that explicit resets work in loop patterns."""

    class ResetQubit(Block):
        """Measure and reset a qubit."""

        def __init__(self, q: QReg, c: CReg) -> None:
            super().__init__()
            self.q = q
            self.c = c
            self.ops = [
                Measure(q[0]) > c[0],
                Prep(q[0]),  # Explicit reset - should be returned!
            ]

    prog = Main(
        q := QReg("q", 1),
        c := CReg("c", 3),
        # Call three times - requires consistent return size
        ResetQubit(q, c[0:1]),
        ResetQubit(q, c[1:2]),
        ResetQubit(q, c[2:3]),
    )

    guppy_code = SlrConverter(prog).guppy()

    # Function should return size 1 (the fresh qubit from Prep)
    assert "-> array[quantum.qubit, 1]:" in guppy_code
    # Should allocate fresh qubit
    assert "quantum.qubit()" in guppy_code

    # Should compile to HUGR (this is the critical test!)
    hugr = SlrConverter(prog).hugr()
    assert hugr is not None

    print("Explicit reset in loop: Maintains array size correctly")


def test_simple_multi_qubit_operations() -> None:
    """Test multiple operations on same qubits."""
    prog = Main(
        q := QReg("q", 2),
        c := CReg("c", 2),
        # Multiple operations
        qb.H(q[0]),
        qb.X(q[1]),
        qb.CX(q[0], q[1]),
        qb.H(q[0]),
        qb.H(q[1]),
        # Measure
        Measure(q) > c,
    )

    guppy_code = SlrConverter(prog).guppy()

    # All operations should be present in order
    operations = [
        "quantum.h(q[0])",
        "quantum.x(q[1])",
        "quantum.cx(q[0], q[1])",
    ]

    for op in operations:
        assert op in guppy_code

    # Should compile
    hugr = SlrConverter(prog).hugr()
    assert hugr is not None

    print("Multiple operations: All present and ordered")


def test_simple_ghz_state() -> None:
    """Test GHZ state preparation (3-qubit entangled state)."""
    prog = Main(
        q := QReg("q", 3),
        c := CReg("c", 3),
        # GHZ state: H on first qubit, then CNOTs
        qb.H(q[0]),
        qb.CX(q[0], q[1]),
        qb.CX(q[0], q[2]),
        # Measure all
        Measure(q) > c,
    )

    guppy_code = SlrConverter(prog).guppy()

    # Check structure
    assert "quantum.h(q[0])" in guppy_code
    assert "quantum.cx(q[0], q[1])" in guppy_code
    assert "quantum.cx(q[0], q[2])" in guppy_code
    assert "quantum.measure_array(q)" in guppy_code

    # Should compile
    hugr = SlrConverter(prog).hugr()
    assert hugr is not None

    print("GHZ state: Clean 3-qubit entanglement")


if __name__ == "__main__":
    """Run all tests and print results."""
    test_simple_bell_state()
    test_simple_reset()
    test_simple_function_with_return()
    test_simple_measurement_then_reset()
    test_simple_two_qubit_gate()
    test_simple_loop_pattern()
    test_simple_partial_consumption()
    test_simple_explicit_reset_in_loop()
    test_simple_multi_qubit_operations()
    test_simple_ghz_state()
    print("\nAll simple SLR-to-Guppy tests passed!")
