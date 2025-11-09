"""Tests for whole register permutation functionality in both QASM and QIR generation."""

import re

import pytest
from pecos.qeclib import qubit
from pecos.slr import CReg, Main, Permute, QReg, SlrConverter

# Test fixtures


def create_whole_register_permutation_program() -> tuple:
    """Create a program with permutation of whole registers."""
    a = CReg("a", 5)
    b = CReg("b", 5)

    return Main(
        a,
        b,
        Permute(
            a,
            b,
        ),
        b[2].set(1),  # After permutation, this still refers to b[2]
        a[3].set(0),  # After permutation, this still refers to a[3]
    )


def create_mixed_permutation_program() -> tuple:
    """Create a program with both whole register and element permutations."""
    a = QReg("a", 3)
    b = QReg("b", 3)
    c = QReg("c", 3)

    return Main(
        a,
        b,
        c,
        # First permute specific elements
        Permute(
            [a[0], c[1]],
            [c[1], a[0]],
        ),
        # Then permute whole registers a and b
        Permute(
            a,
            b,
        ),
        # Apply gates to see the effect of permutations
        qubit.H(a[0]),  # Should apply to c[1] after both permutations
        qubit.X(b[1]),  # Should apply to a[1] after the whole register permutation
        qubit.Z(c[2]),  # Should apply to c[2] since it's not permuted
    )


# QASM Tests


def test_whole_register_permutation_qasm() -> None:
    """Test permutation of whole registers in QASM generation."""
    prog = create_whole_register_permutation_program()
    qasm = SlrConverter(prog).qasm()

    # Print the QASM for debugging
    print("\nQASM output:")
    print(qasm)

    # Verify the permutation comment is correct
    assert (
        "// Permutation: a <-> b" in qasm or "// Permuting: a <-> b" in qasm
    ), f"Expected permutation comment not found in QASM:\n{qasm}"

    # Verify the XOR swap operations are generated
    assert "a = a ^ b;" in qasm, f"Expected 'a = a ^ b;' not found in QASM:\n{qasm}"
    assert "b = b ^ a;" in qasm, f"Expected 'b = b ^ a;' not found in QASM:\n{qasm}"
    assert "a = a ^ b;" in qasm, f"Expected 'a = a ^ b;' not found in QASM:\n{qasm}"

    # Verify the temporary bit approach is NOT used for whole register permutations
    assert (
        "creg _bit_swap[1];" not in qasm
    ), f"Unexpected 'creg _bit_swap[1];' found in QASM:\n{qasm}"

    # For classical registers, we're using XOR swap, which swaps the values, not the references.
    # For bit-level operations, the permutation is applied, so b[2].set(1) becomes b[2] = 1;
    # For register-level operations, the original register name is used, so a[3].set(0) becomes a[3] = 0;
    assert "b[2] = 1;" in qasm, f"Expected 'b[2] = 1;' not found in QASM:\n{qasm}"
    assert "a[3] = 0;" in qasm, f"Expected 'a[3] = 0;' not found in QASM:\n{qasm}"


def test_mixed_permutation_qasm() -> None:
    """Test mixed whole register and element permutations in QASM generation."""
    prog = create_mixed_permutation_program()
    qasm = SlrConverter(prog).qasm()

    # Verify the permutation comments are correct
    assert (
        "// Permutation: a <-> b" in qasm or "// Permuting: a <-> b" in qasm
    ), f"Expected permutation comment not found in QASM:\n{qasm}"
    assert (
        "// Permutation: a[0] -> c[1], c[1] -> a[0]" in qasm
    ), f"Expected permutation comment not found in QASM:\n{qasm}"

    # For QRegs, we're using the permutation map approach, not XOR swap
    # So we shouldn't see XOR operations for QRegs
    assert "a = a ^ b;" not in qasm, f"Unexpected XOR operation found in QASM:\n{qasm}"

    # Verify the operations after the permutation
    # For quantum registers, we're using the permutation map approach
    # So H(a[0]) should become H(c[1]) after both permutations
    # X(b[1]) should become X(a[1]) after the whole register permutation
    # Z(c[2]) remains Z(c[2]) since it's not permuted
    assert "h c[1]" in qasm, f"Expected 'h c[1]' not found in QASM:\n{qasm}"
    assert "x a[1]" in qasm, f"Expected 'x a[1]' not found in QASM:\n{qasm}"
    assert "z c[2]" in qasm, f"Expected 'z c[2]' not found in QASM:\n{qasm}"


# QIR Tests


@pytest.mark.optional_dependency
def test_whole_register_permutation_qir() -> None:
    """Test permutation of whole registers in QIR generation."""
    prog = create_whole_register_permutation_program()
    qir = SlrConverter(prog).qir()

    # Verify the permutation comment is present
    assert (
        "; Permutation: a <-> b" in qir
    ), f"Expected permutation comment not found in QIR:\n{qir}"

    # Verify the XOR operations are present
    assert "xor" in qir, f"Expected XOR operations not found in QIR:\n{qir}"

    # Verify the temporary bit approach is NOT used for whole register permutations
    assert (
        "_bit_swap = call i1* @create_creg(i64 1)" not in qir
    ), f"Unexpected '_bit_swap = call i1* @create_creg(i64 1)' found in QIR:\n{qir}"
    assert (
        "call i1 @get_creg_bit" not in qir
    ), f"Unexpected 'call i1 @get_creg_bit' found in QIR:\n{qir}"

    # Verify the set operations are present
    set_pattern = r"call void @set_creg_bit\(i1\* %(\w+), i64 (\d+), i1 (\d+)\)"
    set_calls = re.findall(set_pattern, qir)

    # Verify the operations after the permutation
    # Since we're swapping values, not references, the operations should still refer to the original registers
    b2_set = False
    a3_set = False
    for reg, idx, val in set_calls:
        if reg == "b" and idx == "2" and val == "1":
            b2_set = True
        if reg == "a" and idx == "3" and val == "0":
            a3_set = True

    assert b2_set, f"Expected set_creg_bit(b, 2, 1) not found in QIR:\n{qir}"
    assert a3_set, f"Expected set_creg_bit(a, 3, 0) not found in QIR:\n{qir}"


@pytest.mark.optional_dependency
def test_mixed_permutation_qir() -> None:
    """Test mixed whole register and element permutations in QIR generation."""
    prog = create_mixed_permutation_program()
    qir = SlrConverter(prog).qir()

    # Print the QIR for debugging
    print("\nQIR output:")
    print(qir)

    # Verify the permutation comments are present
    assert (
        "; Permutation: a[0] -> c[1], c[1] -> a[0]" in qir
    ), f"Expected permutation comment not found in QIR:\n{qir}"
    assert (
        "; Permutation: a <-> b" in qir
    ), f"Expected permutation comment not found in QIR:\n{qir}"

    # Verify that the QIR contains the correct quantum operations after permutations
    # H gate should be applied to c[1] (after both permutations)
    # The exact qubit index depends on how QIR allocates qubits, but we can check for the pattern
    h_gate_pattern = r"call void @__quantum__qis__h__body\(%Qubit\* inttoptr \(i64 (\d+) to %Qubit\*\)\)"
    h_gates = re.findall(h_gate_pattern, qir)
    assert len(h_gates) >= 1, f"Expected at least one H gate, found {len(h_gates)}"

    # X gate should be applied to a[1] (after the whole register permutation)
    x_gate_pattern = r"call void @__quantum__qis__x__body\(%Qubit\* inttoptr \(i64 (\d+) to %Qubit\*\)\)"
    x_gates = re.findall(x_gate_pattern, qir)
    assert len(x_gates) >= 1, f"Expected at least one X gate, found {len(x_gates)}"

    # Z gate should be applied to c[2] (which is not permuted)
    z_gate_pattern = r"call void @__quantum__qis__z__body\(%Qubit\* inttoptr \(i64 (\d+) to %Qubit\*\)\)"
    z_gates = re.findall(z_gate_pattern, qir)
    assert len(z_gates) >= 1, f"Expected at least one Z gate, found {len(z_gates)}"

    # Verify that the qubit indices for the gates are different
    # This ensures that the permutations are being applied correctly
    all_gates = h_gates + x_gates + z_gates
    assert len(set(all_gates)) == len(
        all_gates,
    ), f"Expected all gates to be applied to different qubits, found duplicates: {all_gates}"
