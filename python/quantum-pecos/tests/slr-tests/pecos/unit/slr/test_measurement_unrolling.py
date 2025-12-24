"""Tests for measurement unrolling with permutations in both QASM and QIR generation."""

import pytest
from pecos.qeclib import qubit
from pecos.slr import CReg, Main, Permute, QReg, SlrConverter


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
    )


def test_measurement_unrolling_qasm() -> None:
    """Test measurement unrolling with permutations in QASM generation."""
    prog = create_measurement_unrolling_program()

    # Print the program structure for debugging
    print("\nProgram structure:")
    print(f"Operations: {[type(op).__name__ for op in prog.ops]}")

    # Get the last operation (should be the Measure operation)
    measure_op = prog.ops[-1]
    print(f"\nMeasure operation: {type(measure_op).__name__}")
    print(f"qargs: {measure_op.qargs}")
    print(f"cout: {measure_op.cout}")

    # Generate QASM using SlrConverter
    qasm = SlrConverter(prog).qasm()

    # Print the QASM for debugging
    print("\nQASM output:")
    print(qasm)

    # Verify that the register-wide measurement is unrolled correctly
    # After permutations:
    # a[0] -> c[0]
    # a[1] -> c[1]
    # a[2] -> c[2]
    assert (
        "measure c[0] -> m[0];" in qasm
    ), f"Expected 'measure c[0] -> m[0];' not found in QASM:\n{qasm}"
    assert (
        "measure c[1] -> m[1];" in qasm
    ), f"Expected 'measure c[1] -> m[1];' not found in QASM:\n{qasm}"
    assert (
        "measure c[2] -> m[2];" in qasm
    ), f"Expected 'measure c[2] -> m[2];' not found in QASM:\n{qasm}"

    # Verify that running QASM generation twice produces consistent results
    qasm2 = SlrConverter(prog).qasm()
    assert qasm == qasm2, "QASM generation is not deterministic"


@pytest.mark.optional_dependency
def test_measurement_unrolling_qir() -> None:
    """Test measurement unrolling with permutations in QIR generation."""
    prog = create_measurement_unrolling_program()
    qir = SlrConverter(prog).qir()

    # Print the QIR for debugging
    print("\nQIR output:")
    print(qir)

    # Verify that the QIR contains comments about the permutations
    assert (
        "; Permutation: a[0] -> c[2], b[1] -> a[0], c[2] -> b[1]" in qir
    ), f"Expected permutation comment not found in QIR:\n{qir}"
    assert (
        "; Permutation: a <-> c" in qir
    ), f"Expected permutation comment not found in QIR:\n{qir}"

    # Verify that the QIR contains the correct quantum operations after permutations
    # H gates should be applied to a[0], a[1], a[2] (qubits 0, 1, 2) initially
    assert (
        "call void @__quantum__qis__h__body(%Qubit* inttoptr (i64 0 to %Qubit*))" in qir
    ), f"Expected H gate on a[0] not found in QIR:\n{qir}"
    assert (
        "call void @__quantum__qis__h__body(%Qubit* inttoptr (i64 1 to %Qubit*))" in qir
    ), f"Expected H gate on a[1] not found in QIR:\n{qir}"
    assert (
        "call void @__quantum__qis__h__body(%Qubit* inttoptr (i64 2 to %Qubit*))" in qir
    ), f"Expected H gate on a[2] not found in QIR:\n{qir}"

    # X gate should be applied to b[1] (qubit 4) initially
    assert (
        "call void @__quantum__qis__x__body(%Qubit* inttoptr (i64 4 to %Qubit*))" in qir
    ), f"Expected X gate on b[1] not found in QIR:\n{qir}"

    # After first permutation:
    # CNOT gate should be applied to c[2] (qubit 8) and b[0] (qubit 3)
    assert (
        "call void @__quantum__qis__cnot__body("
        "%Qubit* inttoptr (i64 8 to %Qubit*), %Qubit* inttoptr (i64 3 to %Qubit*))"
        in qir
    ), f"Expected CNOT gate on permuted qubits not found in QIR:\n{qir}"

    # Z gate should be applied to a[0] (qubit 0) after first permutation
    assert (
        "call void @__quantum__qis__z__body(%Qubit* inttoptr (i64 0 to %Qubit*))" in qir
    ), f"Expected Z gate on permuted qubit not found in QIR:\n{qir}"

    # After second permutation:
    # H gate should be applied to c[1] (qubit 7) after both permutations
    assert (
        "call void @__quantum__qis__h__body(%Qubit* inttoptr (i64 7 to %Qubit*))" in qir
    ), f"Expected H gate on permuted qubit not found in QIR:\n{qir}"

    # CNOT gate should be applied to a[0] (qubit 0) and b[2] (qubit 5) after both permutations
    assert (
        "call void @__quantum__qis__cnot__body("
        "%Qubit* inttoptr (i64 0 to %Qubit*), %Qubit* inttoptr (i64 5 to %Qubit*))"
        in qir
    ), f"Expected CNOT gate on permuted qubits not found in QIR:\n{qir}"

    # Verify that the QIR contains the correct measurements after permutations
    # Register-wide measurement of a should be unrolled to individual measurements
    # a[0] should be measured as c[0] (qubit 2) after both permutations
    assert (
        "call void @mz_to_creg_bit(%Qubit* inttoptr (i64 2 to %Qubit*), i1* %m, i64 0)"
        in qir
    ), f"Expected measurement of a[0] as c[0] not found in QIR:\n{qir}"

    # a[1] should be measured as c[1] (qubit 7) after both permutations
    assert (
        "call void @mz_to_creg_bit(%Qubit* inttoptr (i64 7 to %Qubit*), i1* %m, i64 1)"
        in qir
    ), f"Expected measurement of a[1] as c[1] not found in QIR:\n{qir}"

    # a[2] should be measured as c[2] (qubit 8) after both permutations
    assert (
        "call void @mz_to_creg_bit(%Qubit* inttoptr (i64 8 to %Qubit*), i1* %m, i64 2)"
        in qir
    ), f"Expected measurement of a[2] as c[2] not found in QIR:\n{qir}"

    # Verify that running QIR generation twice produces consistent results
    qir2 = SlrConverter(prog).qir()
    assert qir == qir2, "QIR generation is not deterministic"
