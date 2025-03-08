"""Tests for basic permutation functionality in both QASM and QIR generation."""

from pecos.slr import Main, QReg, CReg, Permute, SlrConverter
from pecos.qeclib import qubit as Q
import re
import pytest

# Test fixtures

def create_basic_permutation_program():
    """Create a basic program with permutation of classical registers."""
    a = CReg("a", 2)
    b = CReg("b", 2)
    
    prog = Main(
        a, b,
        Permute(
            [a[0], b[1]],
            [b[1], a[0]],
        ),
        a[0].set(1)  # Should become b[1] = 1 after permutation
    )
    
    return prog, a, b

def create_same_register_permutation_program():
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

def test_permutation_consistency_for_bits_in_qasm():
    """Test that permutation is consistent across multiple QASM generations."""
    prog = Main(
        a := CReg("a", 2),
        b := CReg("b", 2),

        Permute(
            [a[0], b[1]],
            [b[1], a[0]],
        ),
        a[0].set(1)
    )

    qasm1 = SlrConverter(prog).qasm()
    qasm2 = SlrConverter(prog).qasm()

    assert(qasm1 == qasm2)
    assert("b[1] = 1;" in qasm1)

def test_basic_permutation_qasm(basic_permutation_program):
    """Test basic permutation functionality in QASM generation."""
    prog, _, _ = basic_permutation_program
    
    # Generate QASM
    qasm = SlrConverter(prog).qasm()
    
    # Verify that the QASM contains the correct permuted operation
    assert "b[1] = 1;" in qasm
    
    # Verify that running QASM generation twice produces consistent results
    qasm2 = SlrConverter(prog).qasm()
    assert qasm == qasm2, "QASM generation is not deterministic"


def test_same_register_permutation_qasm(same_register_permutation_program):
    """Test permutation of elements within the same register in QASM."""
    prog, _ = same_register_permutation_program
    
    qasm = SlrConverter(prog).qasm()
    
    assert "a[2] = 1;" in qasm
    assert "a[0] = 0;" in qasm
    assert "a[1] = 1;" in qasm


# QIR Tests

@pytest.mark.optional_dependency
def test_basic_permutation_qir(basic_permutation_program):
    """Test basic permutation functionality in QIR generation."""
    prog, _, _ = basic_permutation_program
    
    # Generate QIR
    qir = SlrConverter(prog).qir()
    
    # Verify that the QIR contains a comment about the permutation
    assert "Permutation: a[0] -> b[1], b[1] -> a[0]" in qir
    
    # Extract the register and index used in the set_creg_bit call
    # This should be setting b[1] (register %b, index 1) after permutation
    set_creg_calls = re.findall(r'call void @set_creg_bit\(i1\* %(\w+), i64 (\d+), i1 1\)', qir)
    
    # We should have at least one set_creg_bit call
    assert len(set_creg_calls) >= 1, "No set_creg_bit call found"
    
    # Get the register and index
    reg_name, index = set_creg_calls[0]
    
    # Verify that the set_creg_bit call is setting b[1] after permutation
    assert reg_name == "b", f"set_creg_bit applied to register {reg_name}, expected b"
    assert index == "1", f"set_creg_bit applied to index {index}, expected 1"
    
    # Verify that running QIR generation twice produces consistent results
    qir2 = SlrConverter(prog).qir()
    assert qir == qir2, "QIR generation is not deterministic"


@pytest.mark.optional_dependency
def test_same_register_permutation_qir(same_register_permutation_program):
    """Test permutation of elements within the same register in QIR."""
    prog, _ = same_register_permutation_program
    
    qir = SlrConverter(prog).qir()
    
    # Verify that the QIR contains a comment about the permutation
    assert "Permutation: a[0] -> a[2], a[1] -> a[0], a[2] -> a[1]" in qir
    
    # Extract the register and indices used in the set_creg_bit calls
    set_creg_calls = re.findall(r'call void @set_creg_bit\(i1\* %(\w+), i64 (\d+), i1 (\d+)\)', qir)
    
    # We should have at least three set_creg_bit calls
    assert len(set_creg_calls) >= 3, f"Expected at least 3 set_creg_bit calls, found {len(set_creg_calls)}"
    
    # Create a dictionary to store the values set for each index
    set_values = {}
    for reg_name, index, value in set_creg_calls:
        assert reg_name == "a", f"set_creg_bit applied to register {reg_name}, expected a"
        set_values[int(index)] = int(value)
    
    # Verify that the set_creg_bit calls are setting the correct values after permutation
    assert set_values.get(2) == 1, "a[2] should be set to 1"
    assert set_values.get(0) == 0, "a[0] should be set to 0"
    assert set_values.get(1) == 1, "a[1] should be set to 1" 