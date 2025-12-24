"""Tests for classical register permutation functionality."""

import re

import pecos.slr
import pytest
from pecos.slr.slr_converter import SlrConverter


def create_creg_permutation_program() -> tuple:
    """Create a program with permutation of whole classical registers followed by both bit and register operations."""
    a = pecos.slr.CReg("a", size=1)
    b = pecos.slr.CReg("b", size=1)

    return pecos.slr.Main(
        a,
        b,
        pecos.slr.Permute(a, b),
        a[0].set(1),  # Bit-level operation
        a.set(1),  # Register-level operation
    )


def test_creg_permutation_qasm() -> None:
    """Test permutation of whole classical registers followed by both bit and register operations in QASM."""
    prog = create_creg_permutation_program()
    qasm = SlrConverter(prog).qasm()

    # Print the QASM for debugging
    # print("\nQASM output:")
    # print(qasm)

    # Verify the XOR swap operations are generated
    assert "a = a ^ b;" in qasm, f"Expected 'a = a ^ b;' not found in QASM:\n{qasm}"
    assert "b = b ^ a;" in qasm, f"Expected 'b = b ^ a;' not found in QASM:\n{qasm}"
    assert "a = a ^ b;" in qasm, f"Expected 'a = a ^ b;' not found in QASM:\n{qasm}"

    # Verify the temporary bit approach is NOT used for whole register permutations
    assert (
        "creg _bit_swap[1];" not in qasm
    ), f"Unexpected 'creg _bit_swap[1];' found in QASM:\n{qasm}"

    # Verify the permutation comment is correct
    assert (
        "// Permutation: a <-> b" in qasm
    ), f"Expected permutation comment not found in QASM:\n{qasm}"

    # Verify the operations after the permutation
    # For classical bit permutations, we're physically moving the values,
    # Since we're not updating the permutation map for classical register permutations,
    # both bit-level and register-level operations should still refer to the original registers.
    assert "a[0] = 1;" in qasm, f"Expected 'a[0] = 1;' not found in QASM:\n{qasm}"
    assert "a = 1;" in qasm, f"Expected 'a = 1;' not found in QASM:\n{qasm}"

    # Verify that running QASM generation twice produces consistent results
    qasm2 = SlrConverter(prog).qasm()
    assert qasm == qasm2, "QASM generation is not deterministic"


@pytest.mark.optional_dependency
def test_creg_permutation_qir() -> None:
    """Test permutation of whole classical registers followed by both bit and register operations in QIR."""
    prog = create_creg_permutation_program()
    qir = SlrConverter(prog).qir()

    # Print the QIR for debugging
    # print("\nQIR output:")
    # print(qir)

    # Verify that the QIR contains a comment about the permutation
    assert (
        "Permutation: a <-> b" in qir
    ), "Expected permutation comment not found in QIR"

    # Verify that the XOR operations are present
    assert "xor" in qir, "Expected XOR operations not found in QIR"

    # Verify the temporary bit approach is NOT used for whole register permutations
    assert (
        "_bit_swap = call i1* @create_creg(i64 1)" not in qir
    ), "Unexpected '_bit_swap = call i1* @create_creg(i64 1)' found in QIR"

    # Extract the register and index used in the set_creg_bit call for the bit-level operation
    set_creg_bit_calls = re.findall(
        r"call void @set_creg_bit\(i1\* %(\w+), i64 (\d+), i1 1\)",
        qir,
    )
    assert (
        len(set_creg_bit_calls) >= 1
    ), "No set_creg_bit call found for bit-level operation"

    # Get the register and index for the bit-level operation
    reg_name, index = set_creg_bit_calls[0]

    # In QIR, unlike QASM, the permutation is not applied to bit-level operations
    # So a[0].set(1) still refers to register a, index 0
    assert reg_name == "a", f"set_creg_bit applied to register {reg_name}, expected a"
    assert index == "0", f"set_creg_bit applied to index {index}, expected 0"

    # Extract the register used in the set_creg call for the register-level operation
    set_creg_calls = re.findall(
        r"call void @set_creg_to_int\(i1\* %(\w+), i64 1\)",
        qir,
    )
    assert (
        len(set_creg_calls) >= 1
    ), "No set_creg_to_int call found for register-level operation"

    # Get the register for the register-level operation
    reg_name = set_creg_calls[0]

    # For register-level operations, the original register name is used
    # So a.set(1) still refers to register a
    assert (
        reg_name == "a"
    ), f"set_creg_to_int applied to register {reg_name}, expected a"

    # Verify that running QIR generation twice produces consistent results
    qir2 = SlrConverter(prog).qir()
    assert qir == qir2, "QIR generation is not deterministic"
