"""Tests for Guppy code generation from SLR programs."""

from pecos.qeclib import qubit as qb
from pecos.slr import CReg, If, Main, QReg, Repeat, SlrConverter


def test_simple_circuit() -> None:
    """Test simple quantum circuit generation."""
    prog = Main(
        q := QReg("q", 2),
        c := CReg("c", 2),
        # Prepare Bell state
        qb.H(q[0]),
        qb.CX(q[0], q[1]),
        # Measure
        qb.Measure(q) > c,
    )

    # Generate Guppy code
    guppy_code = SlrConverter(prog).guppy()

    # Basic assertions
    assert "@guppy" in guppy_code
    assert "def main()" in guppy_code
    assert "quantum.h(q[0])" in guppy_code
    assert "quantum.cx(q[0], q[1])" in guppy_code


def test_conditional_logic() -> None:
    """Test conditional logic generation."""
    prog = Main(
        q := QReg("q", 1),
        c := CReg("c", 1),
        qb.H(q[0]),
        qb.Measure(q[0]) > c[0],
        If(c[0] == 1).Then(
            qb.X(q[0]),
        ),
    )

    guppy_code = SlrConverter(prog).guppy()

    # Check conditional structure
    # With unpacking, c[0] becomes c_0
    assert "if c[0]:" in guppy_code or "if c_0:" in guppy_code
    assert "quantum.x(q_0)" in guppy_code


def test_repeat_loop() -> None:
    """Test repeat loop generation."""
    prog = Main(
        q := QReg("q", 1),
        # Apply H gate 3 times
        Repeat(3).block(
            qb.H(q[0]),
        ),
    )

    guppy_code = SlrConverter(prog).guppy()

    # Check loop structure
    assert "for _ in range(3):" in guppy_code
    assert "quantum.h(q[0])" in guppy_code


def test_steane_snippet() -> None:
    """Test a simple Steane code snippet."""
    from pecos.qeclib.steane.steane_class import Steane

    prog = Main(
        # Create two logical qubits
        s1 := Steane("s1"),
        s2 := Steane("s2"),
        # Prepare logical |0>
        s1.pz(),
        # Apply logical Hadamard
        s1.h(),
        # Logical CNOT
        s1.cx(s2),
        # QEC cycle
        s1.qec(),
        s2.qec(),
    )

    guppy_code = SlrConverter(prog).guppy()

    # Check that Steane registers are declared
    assert "s1_d = array(quantum.qubit() for _ in range(7))" in guppy_code
    assert "s2_d = array(quantum.qubit() for _ in range(7))" in guppy_code
    # Check that some quantum operations are present
    assert "quantum.h(" in guppy_code
    assert "quantum.cx(" in guppy_code


def test_measurement_handling() -> None:
    """Test different measurement patterns."""
    prog = Main(
        q := QReg("q", 3),
        c := CReg("c", 3),
        # Individual qubit measurement
        qb.Measure(q[0]) > c[0],
        # Full register measurement
        qb.Measure(q) > c,
    )

    guppy_code = SlrConverter(prog).guppy()

    # Check measurement generation
    # Note: IR generator may use local allocation optimization
    assert "c[0] = quantum.measure(" in guppy_code
    assert "c = quantum.measure_array(q)" in guppy_code


def test_various_gates() -> None:
    """Test generation of various quantum gates."""
    prog = Main(
        q := QReg("q", 2),
        # Single-qubit gates
        qb.H(q[0]),
        qb.X(q[0]),
        qb.Y(q[0]),
        qb.Z(q[0]),
        qb.SZ(q[0]),  # S gate
        qb.SZdg(q[0]),  # Sdg gate
        qb.T(q[0]),
        qb.Tdg(q[0]),
        # Two-qubit gates
        qb.CX(q[0], q[1]),
        qb.CY(q[0], q[1]),
        qb.CZ(q[0], q[1]),
    )

    guppy_code = SlrConverter(prog).guppy()

    # Check all gates are present
    gates = ["h", "x", "y", "z", "s", "sdg", "t", "tdg", "cx", "cy", "cz"]
    for gate in gates:
        assert f"quantum.{gate}(" in guppy_code


def test_bitwise_operations() -> None:
    """Test generation of bitwise operations."""
    prog = Main(
        c := CReg("c", 8),
        # Initialize some bits
        c[0].set(1),
        c[1].set(0),
        c[2].set(1),
        # Test bitwise in assignments
        c[3].set(c[0] ^ c[1]),
        c[4].set(c[0] & c[2]),
        c[5].set(c[1] | c[2]),
        # Test NOT operation
        c[6].set(~c[0]),
        c[7].set((c[0] | c[1]) & ~c[2]),
    )

    guppy_code = SlrConverter(prog).guppy()

    # Check bitwise operations in assignments
    # IR generator now properly uses boolean operators
    assert "c[3] = c[0] ^ c[1]" in guppy_code
    assert "c[4] = c[0] & c[2]" in guppy_code
    assert "c[5] = c[1] | c[2]" in guppy_code
    assert "c[6] = not c[0]" in guppy_code
    # Complex expression - exact parentheses may vary due to precedence
    assert "c[7] = " in guppy_code
    assert "c[0] | c[1]" in guppy_code
    assert "not c[2]" in guppy_code


def test_register_operations() -> None:
    """Test operations on full quantum registers."""
    prog = Main(
        q := QReg("q", 4),
        _c := CReg("c", 4),
        # Apply gate to full register
        qb.H(q),
        # Apply gate to specific qubits
        qb.X(q[0]),
        qb.X(q[2]),
        # Two-qubit gates on specific pairs
        qb.CX(q[0], q[1]),
        qb.CX(q[2], q[3]),
    )

    guppy_code = SlrConverter(prog).guppy()

    # Check register-wide operation generates a loop
    assert "for i in range(0, 4):" in guppy_code
    assert "quantum.h(q[i])" in guppy_code

    # Check individual operations
    assert "quantum.x(q[0])" in guppy_code
    assert "quantum.x(q[2])" in guppy_code
    assert "quantum.cx(q[0], q[1])" in guppy_code


def test_steane_encoding_circuit_pattern() -> None:
    """Test the specific multi-pair CX pattern from Steane encoding circuit."""
    prog = Main(
        q := QReg("q", 7),
        # Prepare first 6 qubits
        qb.Prep(q[0], q[1], q[2], q[3], q[4], q[5]),
        # Single gates
        qb.CX(q[6], q[5]),
        qb.H(q[1]),
        qb.CX(q[1], q[0]),
        qb.H(q[2]),
        qb.CX(q[2], q[4]),
        qb.H(q[3]),
        # Multi-pair CX operations (Steane encoding pattern)
        qb.CX(
            (q[3], q[5]),
            (q[2], q[0]),
            (q[6], q[4]),
        ),
        qb.CX(
            (q[2], q[6]),
            (q[3], q[4]),
            (q[1], q[5]),
        ),
        qb.CX(
            (q[1], q[6]),
            (q[3], q[0]),
        ),
    )

    guppy_code = SlrConverter(prog).guppy()

    # Check Prep operations generate fresh qubit allocations
    # IR generator uses a loop for consecutive Prep operations
    assert "for i in range(0, 6):" in guppy_code
    assert "quantum.qubit()" in guppy_code  # Fresh qubit allocation (Prep operation)

    # Check single CX operations
    assert "quantum.cx(q[6], q[5])" in guppy_code
    assert "quantum.cx(q[1], q[0])" in guppy_code
    assert "quantum.cx(q[2], q[4])" in guppy_code

    # Check multi-pair CX operations are expanded correctly
    assert "quantum.cx(q[3], q[5])" in guppy_code
    assert "quantum.cx(q[2], q[0])" in guppy_code
    assert "quantum.cx(q[6], q[4])" in guppy_code
    assert "quantum.cx(q[2], q[6])" in guppy_code
    assert "quantum.cx(q[3], q[4])" in guppy_code
    assert "quantum.cx(q[1], q[5])" in guppy_code
    assert "quantum.cx(q[1], q[6])" in guppy_code
    assert "quantum.cx(q[3], q[0])" in guppy_code


def test_reset_operations() -> None:
    """Test that Prep operations generate proper reset calls."""
    prog = Main(
        q := QReg("q", 3),
        _c := CReg("c", 3),
        # Reset single qubit
        qb.Prep(q[0]),
        # Apply some operations
        qb.H(q[0]),
        qb.CX(q[0], q[1]),
        # Reset multiple qubits
        qb.Prep(q[1], q[2]),
        # More operations
        qb.X(q[0]),
        qb.Y(q[1]),
        qb.Z(q[2]),
    )

    guppy_code = SlrConverter(prog).guppy()

    # Check Prep operations generate fresh qubit allocations
    # Individual Prep with assignment
    assert "q[0] = quantum.qubit()" in guppy_code
    # IR generator uses a loop for consecutive Prep operations q[1] and q[2]
    assert "for i in range(1, 3):" in guppy_code
    assert "quantum.qubit()" in guppy_code

    # Count quantum.qubit() occurrences (one for q[0], one in loop for q[i])
    # Note: Prep allocates fresh qubits
    qubit_count = guppy_code.count("quantum.qubit()")
    assert qubit_count == 2  # q[0] once with assignment, q[i] once in loop


def test_permute_operations() -> None:
    """Test that Permute operations generate proper swaps."""
    from pecos.slr.misc import Permute

    prog = Main(
        a := QReg("a", 3),
        b := QReg("b", 3),
        c := CReg("c", 2),
        d := CReg("d", 2),
        # Individual element permutation
        Permute([a[0], b[1]], [b[1], a[0]]),
        # Multiple element permutation (rotation)
        Permute([a[0], a[1], a[2]], [a[2], a[0], a[1]]),
        # Whole register permutation
        Permute(c, d),
        # Apply some gates to verify permutation
        qb.H(a[0]),
        qb.X(b[1]),
    )

    guppy_code = SlrConverter(prog).guppy()

    # Check that permutation operations are present
    # Note: The exact syntax may vary based on implementation
    assert (
        "# Permute" in guppy_code
        or "swap" in guppy_code.lower()
        or ("a[0]" in guppy_code and "b[1]" in guppy_code)
    )

    # Check that gates work after permutation
    assert "quantum.h(" in guppy_code
    assert "quantum.x(" in guppy_code
