"""Tests for basic permutation functionality in both QASM and QIR generation."""

import re

import pytest
from pecos.slr import CReg, Main, Permute, SlrConverter

# Test fixtures


def create_basic_permutation_program() -> tuple:
    """Create a basic program with permutation of classical registers."""
    a = CReg("a", 2)
    b = CReg("b", 2)

    prog = Main(
        a,
        b,
        Permute(
            [a[0], b[1]],
            [b[1], a[0]],
        ),
        a[0].set(1),  # Should become b[1] = 1 after permutation
    )

    return prog, a, b


def create_same_register_permutation_program() -> tuple:
    """Create a program with permutation within the same register."""
    a = CReg("a", 3)

    prog = Main(
        a,
        Permute(
            [a[0], a[1], a[2]],
            [a[2], a[0], a[1]],
        ),
        a[0].set(1),  # Should become a[2] = 1
        a[1].set(0),  # Should become a[0] = 0
        a[2].set(1),  # Should become a[1] = 1
    )

    return prog, a


# QASM Tests


def test_permutation_consistency_for_bits_in_qasm() -> None:
    """Test that permutation is consistent across multiple QASM generations."""
    prog = Main(
        a := CReg("a", 2),
        b := CReg("b", 2),
        Permute(
            [a[0], b[1]],
            [b[1], a[0]],
        ),
        a[0].set(1),
    )

    qasm1 = SlrConverter(prog).qasm()
    qasm2 = SlrConverter(prog).qasm()

    # Print the QASM for debugging
    # print("\nQASM output:")
    # print(qasm1)

    assert qasm1 == qasm2
    assert "a[0] = 1;" in qasm1

    # Verify that the bit permutation is using the temporary bit approach, not XOR swap
    assert "creg _bit_swap[1];" in qasm1
    assert "_bit_swap[0] = a[0];" in qasm1
    assert "a[0] = b[1];" in qasm1
    assert "b[1] = _bit_swap[0];" in qasm1
    assert "a[0] = a[0] ^ b[1];" not in qasm1  # Make sure XOR swap is not used


def test_basic_permutation_qasm(basic_permutation_program: tuple) -> None:
    """Test basic permutation functionality in QASM generation."""
    prog, _, _ = basic_permutation_program

    # Generate QASM
    qasm = SlrConverter(prog).qasm()

    # Print the QASM for debugging
    # print("\nQASM output:")
    # print(qasm)

    # Verify that the QASM contains the correct permuted operation
    # For classical bit permutations, operations still refer to the original bit names
    assert "a[0] = 1;" in qasm

    # Verify that the bit permutation is using the temporary bit approach, not XOR swap
    assert "creg _bit_swap[1];" in qasm
    assert "_bit_swap[0] = a[0];" in qasm
    assert "a[0] = b[1];" in qasm
    assert "b[1] = _bit_swap[0];" in qasm
    assert "a[0] = a[0] ^ b[1];" not in qasm  # Make sure XOR swap is not used

    # Verify that running QASM generation twice produces consistent results
    qasm2 = SlrConverter(prog).qasm()
    assert qasm == qasm2, "QASM generation is not deterministic"


def test_same_register_permutation_qasm(
    same_register_permutation_program: tuple,
) -> None:
    """Test permutation of elements within the same register in QASM."""
    prog, _ = same_register_permutation_program

    qasm = SlrConverter(prog).qasm()

    # Print the QASM for debugging
    # print("\nQASM output:")
    # print(qasm)

    # For classical bit permutations, operations still refer to the original bit names
    assert "a[0] = 1;" in qasm
    assert "a[1] = 0;" in qasm
    assert "a[2] = 1;" in qasm

    # Verify that the bit permutation is using the temporary bit approach, not XOR swap
    assert "creg _bit_swap[1];" in qasm
    assert "_bit_swap[0] = a[0];" in qasm
    assert "a[0] = a[2];" in qasm  # Part of the cycle
    assert "a[2] = a[1];" in qasm  # Part of the cycle
    assert "a[1] = _bit_swap[0];" in qasm  # Completing the cycle
    assert "a[0] = a[0] ^ a[1];" not in qasm  # Make sure XOR swap is not used


# QIR Tests


@pytest.mark.optional_dependency
def test_basic_permutation_qir(basic_permutation_program: tuple) -> None:
    """Test basic permutation functionality in QIR generation."""
    prog, _, _ = basic_permutation_program

    # Generate QIR
    qir = SlrConverter(prog).qir()

    # Print the QIR for debugging
    # print("\nQIR output:")
    # print(qir)

    # Verify that the QIR contains a comment about the permutation
    assert "Permutation: a[0] -> b[1], b[1] -> a[0]" in qir

    # Extract the register and index used in the set_creg_bit call
    # This should be setting a[0] (register %a, index 0) since the permutation
    # is not being applied to the operations in the QIR generator
    set_creg_calls = re.findall(
        r"call void @set_creg_bit\(i1\* %(\w+), i64 (\d+), i1 1\)",
        qir,
    )

    # We should have at least one set_creg_bit call
    assert len(set_creg_calls) >= 1, "No set_creg_bit call found"

    # Get the register and index
    reg_name, index = set_creg_calls[0]

    # Verify that the set_creg_bit call is setting a[0] since the permutation
    # is not being applied to the operations in the QIR generator
    assert reg_name == "a", f"set_creg_bit applied to register {reg_name}, expected a"
    assert index == "0", f"set_creg_bit applied to index {index}, expected 0"

    # Verify that running QIR generation twice produces consistent results
    qir2 = SlrConverter(prog).qir()
    assert qir == qir2, "QIR generation is not deterministic"


@pytest.mark.optional_dependency
def test_same_register_permutation_qir(
    same_register_permutation_program: tuple,
) -> None:
    """Test permutation of elements within the same register in QIR."""
    prog, _ = same_register_permutation_program

    qir = SlrConverter(prog).qir()

    # Print the QIR for debugging
    # print("\nQIR output:")
    # print(qir)

    # Verify that the QIR contains a comment about the permutation
    assert "Permutation: a[0] -> a[2], a[1] -> a[0], a[2] -> a[1]" in qir

    # Extract the register and indices used in the set_creg_bit calls
    set_creg_calls = re.findall(
        r"call void @set_creg_bit\(i1\* %(\w+), i64 (\d+), i1 (\d+)\)",
        qir,
    )

    # We should have at least three set_creg_bit calls
    assert (
        len(set_creg_calls) >= 3
    ), f"Expected at least 3 set_creg_bit calls, found {len(set_creg_calls)}"

    # Create a dictionary to store the values set for each index
    set_values = {}
    for reg_name, index, value in set_creg_calls:
        assert (
            reg_name == "a"
        ), f"set_creg_bit applied to register {reg_name}, expected a"
        set_values[int(index)] = int(value)

    # Verify that the set_creg_bit calls are setting the correct values
    # Since the permutation is not being applied to the operations in the QIR generator,
    # we expect the original operations to be executed
    assert set_values.get(0) == 1, "a[0] should be set to 1"
    assert set_values.get(1) == 0, "a[1] should be set to 0"
    assert set_values.get(2) == 1, "a[2] should be set to 1"
