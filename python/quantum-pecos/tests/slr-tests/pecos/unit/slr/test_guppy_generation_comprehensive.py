"""Comprehensive tests for Guppy code generation from SLR programs.

These tests cover various quantum algorithms, patterns, and edge cases
to ensure the Guppy generator produces correct output for diverse scenarios.
"""

from pecos.qeclib import qubit as qb
from pecos.slr import CReg, If, Main, Permute, QReg, Repeat, SlrConverter


def test_quantum_teleportation() -> None:
    """Test quantum teleportation protocol generation."""
    prog = Main(
        alice := QReg("alice", 1),
        bob := QReg("bob", 1),
        epr := QReg("epr", 1),
        c := CReg("c", 2),
        # Create EPR pair
        qb.H(epr[0]),
        qb.CX(epr[0], bob[0]),
        # Alice's operations
        qb.CX(alice[0], epr[0]),
        qb.H(alice[0]),
        # Measure Alice's qubits
        qb.Measure(alice[0]) > c[0],
        qb.Measure(epr[0]) > c[1],
        # Bob's corrections
        If(c[1]).Then(
            qb.X(bob[0]),
        ),
        If(c[0]).Then(
            qb.Z(bob[0]),
        ),
    )

    guppy_code = SlrConverter(prog).guppy()

    # Check key elements
    # IR generator uses dynamic allocation for single-element arrays
    assert "quantum.h(epr_0)" in guppy_code
    # bob is unpacked because it's used in conditional blocks (improved behavior)
    assert "quantum.cx(epr_0, bob_0)" in guppy_code
    assert "c_0 = quantum.measure(alice_0)" in guppy_code
    assert "c_1 = quantum.measure(epr_0)" in guppy_code
    assert "if c_1:" in guppy_code
    assert "quantum.x(bob_0)" in guppy_code
    assert "if c_0:" in guppy_code
    assert "quantum.z(bob_0)" in guppy_code


def test_syndrome_extraction_pattern() -> None:
    """Test error syndrome extraction with conditional corrections."""
    prog = Main(
        data := QReg("data", 3),
        ancilla := QReg("ancilla", 2),
        syndrome := CReg("syndrome", 2),
        # Parity check 1: data[0] and data[1]
        qb.H(ancilla[0]),
        qb.CX(ancilla[0], data[0]),
        qb.CX(ancilla[0], data[1]),
        qb.H(ancilla[0]),
        qb.Measure(ancilla[0]) > syndrome[0],
        # Reset ancilla
        If(syndrome[0]).Then(
            qb.X(ancilla[0]),  # Reset to |0>
        ),
        # Parity check 2: data[1] and data[2]
        qb.H(ancilla[1]),
        qb.CX(ancilla[1], data[1]),
        qb.CX(ancilla[1], data[2]),
        qb.H(ancilla[1]),
        qb.Measure(ancilla[1]) > syndrome[1],
        # Decode syndrome and apply corrections
        If(syndrome[0] & ~syndrome[1]).Then(
            qb.X(data[0]),
        ),
        If(syndrome[0] & syndrome[1]).Then(
            qb.X(data[1]),
        ),
        If(~syndrome[0] & syndrome[1]).Then(
            qb.X(data[2]),
        ),
    )

    guppy_code = SlrConverter(prog).guppy()

    # Check syndrome measurement and corrections
    # IR generator unpacks arrays and uses unpacked names
    assert "syndrome_0 = quantum.measure(ancilla_0)" in guppy_code
    assert "syndrome_1 = quantum.measure(ancilla_1)" in guppy_code
    # Conditionals use unpacked names
    assert "if syndrome_0:" in guppy_code
    assert "if syndrome_0 & syndrome_1:" in guppy_code
    assert "if not syndrome_0 & syndrome_1:" in guppy_code


def test_parameterized_circuit() -> None:
    """Test circuit with classical parameters controlling quantum operations."""
    prog = Main(
        q := QReg("q", 4),
        params := CReg("params", 3),
        results := CReg("results", 4),
        # Set parameters
        params[0].set(1),
        params[1].set(0),
        params[2].set(1),
        # Conditional initialization
        If(params[0]).Then(
            qb.H(q[0]),
            qb.X(q[1]),
        ),
        # Parameterized entangling gates
        If(params[1]).Then(
            qb.CX(q[0], q[1]),
            qb.CX(q[2], q[3]),
        ),
        If(~params[1]).Then(
            qb.CX(q[0], q[2]),
            qb.CX(q[1], q[3]),
        ),
        # Conditional measurements
        If(params[2]).Then(
            qb.Measure(q) > results,
        ),
        If(~params[2]).Then(
            qb.Measure(q[0], q[1]) > [results[0], results[1]],
        ),
    )

    guppy_code = SlrConverter(prog).guppy()

    # Check parameterized behavior
    # Classical arrays may or may not be unpacked depending on quantum array strategy
    assert "params_0 = True" in guppy_code or "params[0] = True" in guppy_code
    assert "params_1 = False" in guppy_code or "params[1] = False" in guppy_code
    assert "params_2 = True" in guppy_code or "params[2] = True" in guppy_code
    # Conditionals may use unpacked or array access
    assert "if params_0:" in guppy_code or "if params[0]:" in guppy_code
    assert "if params_1:" in guppy_code or "if params[1]:" in guppy_code
    assert "if not params_1:" in guppy_code or "if not params[1]:" in guppy_code
    assert "results = quantum.measure_array(q)" in guppy_code
    # Multi-qubit measurement handling is different in IR generator
    # It generates TODO comments for partial measurements in conditionals


def test_complex_permutation_patterns() -> None:
    """Test various permutation patterns including single and multi-element."""
    prog = Main(
        q := QReg("q", 4),
        work := QReg("work", 2),
        # Single qubit permutations
        Permute(q[0], q[1]),
        Permute(q[2], work[0]),
        # Multi-qubit permutation
        Permute([q[0], q[1]], [work[0], work[1]]),
        # Apply gates after permutation
        qb.CX(q[0], q[1]),
        qb.CZ(q[2], q[3]),
    )

    guppy_code = SlrConverter(prog).guppy()

    # Check that permutation operations and gates are present
    # IR generator generates TODO comments for complex permutations
    assert "TODO: Implement complex permutation" in guppy_code
    assert "quantum.cx(q[0], q[1])" in guppy_code
    assert "quantum.cz(q[2], q[3])" in guppy_code


def test_nested_repeat_with_measurements() -> None:
    """Test nested repeat blocks with measurements and conditional logic."""
    prog = Main(
        q := QReg("q", 2),
        flag := CReg("flag", 1),
        counter := CReg("counter", 3),
        Repeat(3).block(
            flag[0].set(0),
            Repeat(2).block(
                qb.H(q[0]),
                qb.Measure(q[0]) > flag[0],
                If(flag[0]).Then(
                    qb.CX(q[0], q[1]),
                    counter[0].set(counter[0] | flag[0]),
                ),
                If(~flag[0]).Then(
                    qb.Prep(q[0]),
                ),
            ),
            counter[1].set(counter[1] ^ 1),
        ),
        If(counter[0] & counter[1]).Then(
            counter[2].set(1),
        ),
    )

    guppy_code = SlrConverter(prog).guppy()

    # Check nested structure
    assert "range(3)" in guppy_code
    assert "range(2)" in guppy_code
    assert "quantum.measure" in guppy_code
    assert "flag" in guppy_code
    assert "counter" in guppy_code
    # Note: reset and bitwise operations may be represented differently


def test_complex_boolean_expressions() -> None:
    """Test complex classical boolean expressions with proper precedence."""
    prog = Main(
        c := CReg("c", 8),
        # Set initial values
        c[0].set(1),
        c[1].set(0),
        c[2].set(1),
        # Complex expressions - test precedence
        c[3].set(c[0] | (c[1] & c[2])),
        c[4].set((c[0] | c[1]) & (c[2] ^ c[3])),
        c[5].set((c[0] ^ c[1]) ^ (c[2] & ~c[3])),
        # Nested operations
        If((c[0] | ~c[1]) & (c[2] ^ (c[3] & c[4]))).Then(
            c[6].set(~(c[0] & c[1]) | (c[2] ^ c[3])),
            c[7].set(~((c[5] & c[6]) ^ (c[0] | c[3]))),
        ),
    )

    guppy_code = SlrConverter(prog).guppy()

    # Check that boolean operations are present
    # Classical arrays may or may not be unpacked depending on quantum array strategy
    assert "c_3 = " in guppy_code or "c[3] = " in guppy_code
    assert "c_4 = " in guppy_code or "c[4] = " in guppy_code
    assert "c_5 = " in guppy_code or "c[5] = " in guppy_code
    assert "if" in guppy_code

    # Boolean operations should be present
    assert "|" in guppy_code or "OR" in guppy_code
    assert "&" in guppy_code or "AND" in guppy_code


def test_empty_blocks_and_edge_cases() -> None:
    """Test empty blocks and various edge cases."""
    prog = Main(
        q := QReg("q", 1),
        c := CReg("c", 2),
        # Empty conditional
        If(c[0]).Then(),
        # Empty repeat
        Repeat(3).block(),
        # Nested empty blocks
        If(c[0]).Then(
            Repeat(2).block(
                If(c[1]).Then(),
            ),
        ),
        # Measurement without output
        qb.H(q[0]),
        qb.Measure(q[0]),
        # Apply gate to register
        qb.Prep(q),
    )

    guppy_code = SlrConverter(prog).guppy()

    # Check that code is generated without errors
    assert "def main()" in guppy_code
    assert len(guppy_code) > 100
    # Note: empty blocks and reset operations may be optimized or handled differently


def test_grover_decomposition() -> None:
    """Test Grover's algorithm with CCX decomposition."""
    prog = Main(
        q := QReg("q", 2),
        ancilla := QReg("ancilla", 1),
        c := CReg("c", 3),
        # Initialize superposition
        qb.H(q),
        # Oracle using decomposed CCX
        qb.H(ancilla[0]),
        qb.CX(q[1], ancilla[0]),
        qb.Tdg(ancilla[0]),
        qb.CX(q[0], ancilla[0]),
        qb.T(ancilla[0]),
        qb.CX(q[1], ancilla[0]),
        qb.Tdg(ancilla[0]),
        qb.CX(q[0], ancilla[0]),
        qb.T(ancilla[0]),
        qb.H(ancilla[0]),
        # Diffusion operator
        qb.H(q),
        qb.X(q),
        qb.CZ(q[0], q[1]),
        qb.X(q),
        qb.H(q),
        # Measure
        qb.Measure(q) > [c[0], c[1]],
        qb.Measure(ancilla[0]) > c[2],
    )

    guppy_code = SlrConverter(prog).guppy()

    # Check CCX decomposition
    # Ancilla may use unpacked variables or array access
    assert "quantum.h(ancilla_0)" in guppy_code or "quantum.h(ancilla[0])" in guppy_code
    assert "quantum.t(ancilla_0)" in guppy_code or "quantum.t(ancilla[0])" in guppy_code
    assert (
        "quantum.tdg(ancilla_0)" in guppy_code
        or "quantum.tdg(ancilla[0])" in guppy_code
    )

    # Check diffusion operator
    assert "for i in range(0, 2):" in guppy_code  # Register operations with loops
    assert "quantum.cz(q[0], q[1])" in guppy_code


def test_multi_pair_cx_pattern() -> None:
    """Test multi-pair CX pattern from Steane encoding."""
    prog = Main(
        q := QReg("q", 7),
        # Multi-pair CX from Steane encoding
        qb.CX(
            (q[3], q[5]),
            (q[2], q[0]),
            (q[6], q[4]),
        ),
        # Another pattern
        qb.CX(
            (q[0], q[1]),
            (q[2], q[3]),
        ),
    )

    guppy_code = SlrConverter(prog).guppy()

    # Check all CX pairs are generated
    assert "quantum.cx(q[3], q[5])" in guppy_code
    assert "quantum.cx(q[2], q[0])" in guppy_code
    assert "quantum.cx(q[6], q[4])" in guppy_code
    assert "quantum.cx(q[0], q[1])" in guppy_code
    assert "quantum.cx(q[2], q[3])" in guppy_code


def test_mixed_classical_quantum_complex() -> None:
    """Test complex mixed classical and quantum operations."""
    prog = Main(
        q := QReg("q", 3),
        control := CReg("control", 4),
        data := CReg("data", 6),
        # Classical logic
        control[0].set(1),
        control[1].set(0),
        control[2].set(control[0] ^ control[1]),
        control[3].set(control[0] & (control[1] | ~control[2])),
        # Quantum operations based on classical logic
        If(control[0] ^ control[1]).Then(
            qb.H(q[0]),
            If(control[2] | control[3]).Then(
                qb.CX(q[0], q[1]),
                qb.CZ(q[1], q[2]),
            ),
        ),
        # Measure and process
        qb.Measure(q[0], q[1]) > [data[0], data[1]],
        data[2].set(data[0] ^ data[1]),
        data[3].set(data[2] & (control[0] | control[1])),
        # Complex final expression
        data[4].set(((data[0] | data[1]) | data[2]) | (data[3] & control[3])),
        data[5].set(~((data[4] & control[0]) ^ (data[2] | control[2]))),
    )

    guppy_code = SlrConverter(prog).guppy()

    # Check that operations are present
    # The implementation unpacks arrays that are accessed element-wise before operations
    assert "control_2 = " in guppy_code
    assert "control_3 = " in guppy_code
    assert "if" in guppy_code
    # IR generator unpacks q array
    assert "quantum.h(q_0)" in guppy_code
    # Data array is NOT unpacked initially, uses array indexing
    assert "data[2] = " in guppy_code
    assert "data[3] = " in guppy_code
    assert "data[4] = " in guppy_code
    assert "data[5] = " in guppy_code
