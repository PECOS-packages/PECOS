"""Tests for whole register permutation functionality in both QASM and QIR generation."""

from pecos.slr import Main, QReg, CReg, Permute, SlrConverter
from pecos.qeclib import qubit as Q
import re
import pytest

# Test fixtures

def create_whole_register_permutation_program():
    """Create a program with permutation of whole registers."""
    a = CReg("a", 5)
    b = CReg("b", 5)
    
    prog = Main(
        a, b,
        Permute(
            a,
            b
        ),
        b[2].set(1),  # Should become a[2] = 1 after permutation
        a[3].set(0),  # Should become b[3] = 0 after permutation
    )
    
    return prog

def create_mixed_permutation_program():
    """Create a program with both whole register and element permutations."""
    a = QReg("a", 3)
    b = QReg("b", 3)
    c = QReg("c", 3)
    
    prog = Main(
        a, b, c,
        # Permute whole registers a and b
        Permute(
            a,
            b
        ),
        # Then permute specific elements
        Permute(
            [a[0], c[1]],
            [c[1], a[0]]
        ),
        # Apply gates to see the effect of permutations
        Q.H(a[0]),  # Should apply to c[1]
        Q.X(b[1]),  # Should apply to a[1]
        Q.Z(c[2]),  # Should apply to c[2]
    )
    
    return prog

# QASM Tests

def test_whole_register_permutation_qasm():
    """Test permutation of whole registers in QASM generation."""
    prog = create_whole_register_permutation_program()
    qasm = SlrConverter(prog).qasm()
    
    # Verify the permutation comment is correct
    assert "// Permutation: a <-> b" in qasm or "// Permuting: a <-> b" in qasm, f"Expected permutation comment not found in QASM:\n{qasm}"
    
    # Verify the operations are correctly permuted
    # After permutation, b[2].set(1) becomes a[2] = 1, and a[3].set(0) becomes b[3] = 0
    assert "a[2] = 1" in qasm, f"Expected 'a[2] = 1' not found in QASM:\n{qasm}"
    assert "b[3] = 0" in qasm, f"Expected 'b[3] = 0' not found in QASM:\n{qasm}"

def test_mixed_permutation_qasm():
    """Test mixed whole register and element permutations in QASM generation."""
    prog = create_mixed_permutation_program()
    qasm = SlrConverter(prog).qasm()
    
    # Verify the permutation comments are correct
    assert "// Permutation: a <-> b" in qasm or "// Permuting: a <-> b" in qasm, f"Expected permutation comment not found in QASM:\n{qasm}"
    assert "// Permutation: a[0] -> c[1], c[1] -> a[0]" in qasm, f"Expected permutation comment not found in QASM:\n{qasm}"
    
    # Verify the operations are correctly permuted
    # After permutation, a and b are swapped, then a[0] and c[1] are swapped
    # So H(a[0]) becomes H(b[0]) (since a[0] is now mapped to b[0])
    # X(b[1]) becomes X(a[1]) (since b[1] is now mapped to a[1])
    # Z(c[2]) remains Z(c[2]) (since c[2] is not permuted)
    assert "h b[0]" in qasm, f"Expected 'h b[0]' not found in QASM:\n{qasm}"
    assert "x a[1]" in qasm, f"Expected 'x a[1]' not found in QASM:\n{qasm}"
    assert "z c[2]" in qasm, f"Expected 'z c[2]' not found in QASM:\n{qasm}"

# QIR Tests

@pytest.mark.optional_dependency
def test_whole_register_permutation_qir():
    """Test permutation of whole registers in QIR generation."""
    prog = create_whole_register_permutation_program()
    qir = SlrConverter(prog).qir()
    
    # Extract the set operations from the QIR
    set_pattern = r'call void @set_creg_bit\(i1\* %(\w+), i64 (\d+), i1 (\d+)\)'
    set_calls = re.findall(set_pattern, qir)
    
    # Verify the QIR contains the expected operations
    assert len(set_calls) > 0, f"No set operations found in QIR:\n{qir}"
    
    # Verify the permutation comment is present
    assert "; Permutation: a <-> b" in qir, f"Expected permutation comment not found in QIR:\n{qir}"
    
    # Verify the operations are correctly permuted
    # After permutation, b[2].set(1) becomes a[2] = 1, and a[3].set(0) becomes b[3] = 0
    # In QIR, this should be set_creg_bit(a, 2, 1) and set_creg_bit(b, 3, 0)
    a2_set = False
    b3_set = False
    for reg, idx, val in set_calls:
        if reg == 'a' and idx == '2' and val == '1':
            a2_set = True
        if reg == 'b' and idx == '3' and val == '0':
            b3_set = True
    
    assert a2_set, f"Expected set_creg_bit(a, 2, 1) not found in QIR:\n{qir}"
    assert b3_set, f"Expected set_creg_bit(b, 3, 0) not found in QIR:\n{qir}"

@pytest.mark.optional_dependency
def test_mixed_permutation_qir():
    """Test mixed whole register and element permutations in QIR generation."""
    prog = create_mixed_permutation_program()
    qir = SlrConverter(prog).qir()
    
    # Verify the permutation comments are present
    assert "; Permutation: a <-> b" in qir, f"Expected permutation comment not found in QIR:\n{qir}"
    assert "; Permutation: a[0] -> c[1], c[1] -> a[0]" in qir, f"Expected permutation comment not found in QIR:\n{qir}"
    
    # Check if the QIR contains the expected gate calls
    assert "call void @__quantum__qis__h__body" in qir, f"H gate call not found in QIR:\n{qir}"
    assert "call void @__quantum__qis__x__body" in qir, f"X gate call not found in QIR:\n{qir}"
    assert "call void @__quantum__qis__z__body" in qir, f"Z gate call not found in QIR:\n{qir}" 