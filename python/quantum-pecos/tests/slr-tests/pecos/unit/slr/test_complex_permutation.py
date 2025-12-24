"""Tests for complex permutation scenarios in both QASM and QIR generation."""

import re

import pytest
from pecos.qeclib import qubit
from pecos.slr import CReg, If, Main, Permute, QReg, SlrConverter

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
    )

    qasm = SlrConverter(prog).qasm()

    # Print the QASM for debugging
    print("\nQASM Output:")
    print(qasm)

    # Verify that the QASM contains the correct permuted operations
    assert (
        "b[0] = 1;" in qasm
    )  # The classical bit assignment happens before permutation
    # The condition and operation should both be permuted
    assert "if(b[1] == 1) x a[1];" in qasm


# QIR Tests


@pytest.mark.optional_dependency
def test_multiple_permutations_qir() -> None:
    """Test multiple sequential permutations in QIR generation."""
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

    qir = SlrConverter(prog).qir()

    # Verify that the QIR contains comments about the permutations
    assert "Permutation: a[0] -> a[1], a[1] -> a[0]" in qir
    assert "Permutation: a[1] -> b[0], b[0] -> a[1]" in qir

    # Extract the quantum operations
    h_calls = re.findall(
        r"call void @__quantum__qis__h__body\(%Qubit\* inttoptr \(i64 (\d+) to %Qubit\*\)\)",
        qir,
    )
    x_calls = re.findall(
        r"call void @__quantum__qis__x__body\(%Qubit\* inttoptr \(i64 (\d+) to %Qubit\*\)\)",
        qir,
    )

    # We should have at least two H calls and one X call
    assert len(h_calls) >= 2, f"Expected at least 2 H gate calls, found {len(h_calls)}"
    assert len(x_calls) >= 1, f"Expected at least 1 X gate call, found {len(x_calls)}"


@pytest.mark.optional_dependency
def test_permutation_with_conditional_qir() -> None:
    """Test permutation with conditional operations in QIR generation."""
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
    )

    qir = SlrConverter(prog).qir()

    # Verify that the QIR contains a comment about the permutation
    assert "Permutation: a[0] -> a[1], a[1] -> a[0], b[0] -> b[1], b[1] -> b[0]" in qir

    # Extract the set_creg_bit call
    set_creg_calls = re.findall(
        r"call void @set_creg_bit\(i1\* %(\w+), i64 (\d+), i1 1\)",
        qir,
    )

    # We should have at least one set_creg_bit call
    assert len(set_creg_calls) >= 1, "No set_creg_bit call found"

    # Get the register and index
    reg_name, index = set_creg_calls[0]

    # Verify that the set_creg_bit call is setting b[0] (not permuted, as it happens before the permutation)
    assert reg_name == "b", f"set_creg_bit applied to register {reg_name}, expected b"
    assert index == "0", f"set_creg_bit applied to index {index}, expected 0"

    # Extract the conditional X operation
    # In QIR, conditionals use __quantum__rt__array_get_element_ptr_1d to access the condition
    # and then branch based on the condition
    # This is a simplified check that just verifies an X gate is called somewhere after the condition check
    x_calls = re.findall(
        r"call void @__quantum__qis__x__body\(%Qubit\* inttoptr \(i64 (\d+) to %Qubit\*\)\)",
        qir,
    )

    # We should have at least one X call
    assert len(x_calls) >= 1, "No X gate call found"
