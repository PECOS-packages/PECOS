"""Tests for measurement unrolling with permutations in both QASM and QIR generation."""

import re

import pytest
from pecos.slr import CReg, Main, Permute, QReg, Return, SlrConverter
from pecos.slr.qeclib import qubit


def create_measurement_unrolling_program() -> tuple:
    """Create a program with permutations and register-wide measurements."""
    a = QReg("a", 3)
    b = QReg("b", 3)
    c = QReg("c", 3)
    m = CReg("m", 3)

    return Main(
        a,
        b,
        c,
        m,
        # Initial gates
        qubit.H(a),
        qubit.X(b[1]),
        # First permutation
        Permute(
            [a[0], b[1], c[2]],
            [c[2], a[0], b[1]],
        ),
        # Gates after first permutation
        qubit.CX(a[0], b[0]),  # Should be CX(c[2], b[0])
        qubit.Z(b[1]),  # Should be Z(a[0])
        # Second permutation
        Permute(a, c),
        # Gates after second permutation
        qubit.H(a[1]),  # Should be H(c[1])
        qubit.CX(c[0], b[2]),  # Should be CX(a[0], b[2])
        # Register-wide measurement - should be unrolled correctly
        qubit.Measure(a) > m,
        Return(m),
    )


def test_measurement_unrolling_qasm() -> None:
    """Test measurement unrolling with permutations in QASM generation."""
    prog = create_measurement_unrolling_program()

    # Print the program structure for debugging
    print("\nProgram structure:")
    print(f"Operations: {[type(op).__name__ for op in prog.ops]}")

    # Get the last non-Return operation (should be the Measure operation)
    measure_op = prog.ops[-2]
    print(f"\nMeasure operation: {type(measure_op).__name__}")
    print(f"qargs: {measure_op.qargs}")
    print(f"cout: {measure_op.cout}")

    # Generate QASM using SlrConverter
    qasm = SlrConverter(prog).qasm()

    # Print the QASM for debugging
    print("\nQASM output:")
    print(qasm)

    # Verify that the register-wide measurement is unrolled correctly
    # After permutation composition:
    # First perm: a[0] -> c[2], b[1] -> a[0], c[2] -> b[1]
    # Second perm (a <-> c swap): compose with first
    # Result: a[0] -> a[2], a[1] -> c[1], a[2] -> c[2]
    assert "measure a[2] -> m[0];" in qasm, f"Expected 'measure a[2] -> m[0];' not found in QASM:\n{qasm}"
    assert "measure c[1] -> m[1];" in qasm, f"Expected 'measure c[1] -> m[1];' not found in QASM:\n{qasm}"
    assert "measure c[2] -> m[2];" in qasm, f"Expected 'measure c[2] -> m[2];' not found in QASM:\n{qasm}"

    # Verify that running QASM generation twice produces consistent results
    qasm2 = SlrConverter(prog).qasm()
    assert qasm == qasm2, "QASM generation is not deterministic"


@pytest.mark.optional_dependency
def test_measurement_unrolling_qir() -> None:
    """Element-wise then whole-register Permute compose through measures.

    A Permute is realized as a static logical relabel; the
    element-wise `[a[0], b[1], c[2]] -> [c[2], a[0], b[1]]` composes
    with the later whole-register `Permute(a, c)`, and the register
    measurement unrolls onto the relabelled qubits. Pinned from the
    actual emitted QIR (qubit indices deterministic: a=0-2, b=3-5,
    c=6-8).
    """
    prog = create_measurement_unrolling_program()
    qir = SlrConverter(prog).qir()

    assert "; Permutation: a[0] -> c[2], b[1] -> a[0], c[2] -> b[1]" in qir, qir
    assert "; Permutation: a <-> c" in qir, qir
    mz = re.findall(r"call void @__quantum__qis__mz__body\(%Qubit\* inttoptr \(i64 (\d+) to %Qubit\*\), %Result\*", qir)
    # Measure(a) after both permutes -> a relabelled onto c's qubits
    # plus the element cycle: realized as q6, q7, q4.
    assert mz == ["6", "7", "4"], qir

    assert qir == SlrConverter(prog).qir(), "QIR generation is not deterministic"
