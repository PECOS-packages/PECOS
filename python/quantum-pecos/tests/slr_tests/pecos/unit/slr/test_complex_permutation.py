"""Tests for complex permutation scenarios in both QASM and QIR generation."""

import re

import pytest
from pecos.slr import CReg, If, Main, Permute, QReg, Return, SlrConverter
from pecos.slr.qeclib import qubit

# QASM Tests


def test_complex_permutation_circuit() -> None:
    """Test a more complex circuit with multiple permutations at different stages."""
    prog = Main(
        a := QReg("a", 3),
        b := QReg("b", 3),
        c := QReg("c", 3),
        # Initial operations - Layer 1
        qubit.H(a[0]),  # Hadamard on a[0]
        qubit.X(b[1]),  # X gate on b[1]
        qubit.Z(c[2]),  # Z gate on c[2]
        # First permutation: rotate registers
        Permute(
            [a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2]],
            [b[0], b[1], b[2], c[0], c[1], c[2], a[0], a[1], a[2]],
        ),
        # Operations after first permutation - Layer 2
        # a[0] -> b[0], b[1] -> c[1], c[2] -> a[2]
        qubit.X(a[0]),  # X gate on a[0] -> should become X on b[0]
        qubit.Y(b[1]),  # Y gate on b[1] -> should become Y on c[1]
        qubit.Z(c[2]),  # Z gate on c[2] -> should become Z on a[2]
    )

    qasm = SlrConverter(prog).qasm()

    # Check that the permutation was applied correctly
    assert "h a[0];" in qasm.lower()  # Initial operation
    assert "x b[1];" in qasm.lower()  # Initial operation
    assert "z c[2];" in qasm.lower()  # Initial operation

    assert "x b[0];" in qasm.lower()  # After first permutation
    assert "y c[1];" in qasm.lower()  # After first permutation
    assert "z a[2];" in qasm.lower()  # After first permutation


def test_multiple_permutations_qasm() -> None:
    """Test multiple sequential permutations in QASM generation."""
    # Create a program with multiple sequential permutations
    a = QReg("a", 3)
    b = QReg("b", 3)

    prog = Main(
        a,
        b,
        # First permutation
        Permute(
            [a[0], a[1]],
            [a[1], a[0]],
        ),
        # Apply an operation
        qubit.H(a[0]),  # Should become H(a[1]) after first permutation
        # Second permutation
        Permute(
            [a[1], b[0]],
            [b[0], a[1]],
        ),
        # Apply another operation
        qubit.H(a[0]),  # Should still be H(a[1]) after first permutation only
        qubit.X(a[1]),  # Should become X(b[0]) after both permutations
    )

    qasm = SlrConverter(prog).qasm()

    # Print the QASM for debugging
    print("\nQASM Output:")
    print(qasm)

    # Verify that the QASM contains the correct permuted operations
    assert "h a[1];" in qasm  # First H gate
    # The second H gate is applied to a[0] which is mapped to b[0] after both permutations
    assert "h b[0];" in qasm  # Second H gate
    # The X gate is applied to a[1] which is mapped to a[0] after both permutations
    assert "x a[0];" in qasm  # X gate after both permutations


def test_permutation_with_conditional_qasm() -> None:
    """Test permutation with conditional operations in QASM generation."""
    # Create a program with permutation and conditional operations
    a = QReg("a", 2)
    b = CReg("b", 2)

    prog = Main(
        a,
        b,
        # Set a classical bit
        b[0].set(1),
        # Apply a permutation
        Permute(
            [a[0], a[1], b[0], b[1]],
            [a[1], a[0], b[1], b[0]],
        ),
        # Apply a conditional operation
        # After permutation: b[0] -> b[1], a[0] -> a[1]
        # So the condition should be on b[1] and the operation should be on a[1]
        If(b[0] == 1).Then(qubit.X(a[0])),
        Return(b),
    )

    qasm = SlrConverter(prog).qasm()

    # Print the QASM for debugging
    print("\nQASM Output:")
    print(qasm)

    # Verify that the QASM contains the correct permuted operations
    assert "b[0] = 1;" in qasm  # The classical bit assignment happens before permutation
    # The condition and operation should both be permuted
    assert "if(b[1] == 1) x a[1];" in qasm


# QIR Tests


# A Permute is realized as a static logical relabel consulted at
# every qubit/classical-bit lowering (mirroring the Guppy linearity
# tracker's `.permute()`; QIR/Selene have no runtime permute
# intrinsic). These pin the realized targeting from the actual
# emitted QIR (qubit indices deterministic in declaration order).


@pytest.mark.optional_dependency
def test_multiple_permutations_qir() -> None:
    """Two sequential element-wise QReg Permutes compose correctly."""
    a = QReg("a", 3)
    b = QReg("b", 3)

    prog = Main(
        a,
        b,
        Permute([a[0], a[1]], [a[1], a[0]]),
        qubit.H(a[0]),  # a[0]->a[1] = q1
        Permute([a[1], b[0]], [b[0], a[1]]),
        qubit.H(a[0]),  # a[0] still ->a[1] = q1
        qubit.X(a[1]),  # composed: a[1]->b[0] = q3
    )

    qir = SlrConverter(prog).qir()

    assert "; Permutation: a[0] -> a[1], a[1] -> a[0]" in qir, qir
    assert "; Permutation: a[1] -> b[0], b[0] -> a[1]" in qir, qir
    # Qubits: a=0,1,2 b=3,4,5. Both H(a[0]) -> q1; X(a[1]) -> b[0]=q3.
    h = re.findall(r"call void @__quantum__qis__h__body\(%Qubit\* inttoptr \(i64 (\d+) ", qir)
    x = re.findall(r"call void @__quantum__qis__x__body\(%Qubit\* inttoptr \(i64 (\d+) ", qir)
    assert h == ["1", "1"], qir
    assert x == ["3"], qir

    assert qir == SlrConverter(prog).qir(), "QIR generation is not deterministic"


@pytest.mark.optional_dependency
def test_permutation_with_conditional_qir() -> None:
    """Element-wise QReg/CReg Permute is realized around control flow."""
    a = QReg("a", 2)
    b = CReg("b", 2)

    prog = Main(
        a,
        b,
        b[0].set(1),  # before the permute -> b[0]'s slot
        Permute(
            [a[0], a[1], b[0], b[1]],
            [a[1], a[0], b[1], b[0]],
        ),
        If(b[0] == 1).Then(qubit.X(a[0])),  # a[0]->a[1] = q1
    )

    qir = SlrConverter(prog).qir()

    assert "; Permutation: a[0] -> a[1], a[1] -> a[0], b[0] -> b[1], b[1] -> b[0]" in qir, qir
    # b[0].set(1) is BEFORE the permute -> b[0] slot (index 0).
    assert re.search(
        r"%(\.\d+) = getelementptr \[2 x i1\], \[2 x i1\]\* %b, i64 0, i64 0\n\s*store i1 1, i1\* %\1",
        qir,
    ), qir
    # X(a[0]) after the permute -> a[1] = q1.
    assert re.findall(r"call void @__quantum__qis__x__body\(%Qubit\* inttoptr \(i64 (\d+) ", qir) == ["1"], qir

    assert qir == SlrConverter(prog).qir(), "QIR generation is not deterministic"
