"""Tests for whole register permutation functionality in both QASM and QIR generation."""

import re

import pytest
from pecos.slr import CReg, Main, Permute, QReg, Return, SlrConverter
from pecos.slr.qeclib import qubit

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
        Return(a, b),
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
    assert "creg _bit_swap[1];" not in qasm, f"Unexpected 'creg _bit_swap[1];' found in QASM:\n{qasm}"

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


# A Permute is realized as a static logical relabel consulted at
# every qubit/classical-bit lowering (the bespoke
# @set_creg_bit/@get_creg_bit/creg-xor helpers the old tests pinned
# were removed by the static CReg model; classical writes are `store i1` into the
# relabelled register's `[N x i1]` buffer). Works for whole-register
# (QReg + CReg) and element-wise; pinned from the actual emitted QIR.


@pytest.mark.optional_dependency
def test_whole_register_permutation_qir() -> None:
    """Whole-register CReg Permute is realized (a <-> b)."""
    prog = create_whole_register_permutation_program()
    qir = SlrConverter(prog).qir()

    assert "; Permutation: a <-> b" in qir, qir
    # create_whole_register_permutation_program: CReg a,b(5);
    # Permute(a, b); b[2].set(1); a[3].set(0). After the swap, b[2]
    # resolves to a[2] and a[3] resolves to b[3].
    assert re.search(
        r"%(\.\d+) = getelementptr \[5 x i1\], (?:ptr|\[5 x i1\]\*) %a, i64 0, i64 2\n\s*store i1 1, (?:ptr|i1\*) %\1",
        qir,
    ), f"b[2].set(1) should target a[2] after swap:\n{qir}"
    assert re.search(
        r"%(\.\d+) = getelementptr \[5 x i1\], (?:ptr|\[5 x i1\]\*) %b, i64 0, i64 3\n\s*store i1 0, (?:ptr|i1\*) %\1",
        qir,
    ), f"a[3].set(0) should target b[3] after swap:\n{qir}"

    assert qir == SlrConverter(prog).qir(), "QIR generation is not deterministic"


@pytest.mark.optional_dependency
def test_mixed_permutation_qir() -> None:
    """Element-wise then whole-register QReg Permute compose correctly."""
    prog = create_mixed_permutation_program()
    qir = SlrConverter(prog).qir()

    assert "; Permutation: a[0] -> c[1], c[1] -> a[0]" in qir, qir
    assert "; Permutation: a <-> b" in qir, qir
    # QRegs a=0-2, b=3-5, c=6-8. First Permute([a[0],c[1]],[c[1],a[0]])
    # then whole Permute(a, b). Realized: H(a[0])->q3, X(b[1])->q1,
    # Z(c[2])->q8.
    assert re.findall(r"call void @__quantum__qis__h__body\(%Qubit\* inttoptr \(i64 (\d+) ", qir) == ["3"], qir
    assert re.findall(r"call void @__quantum__qis__x__body\(%Qubit\* inttoptr \(i64 (\d+) ", qir) == ["1"], qir
    assert re.findall(r"call void @__quantum__qis__z__body\(%Qubit\* inttoptr \(i64 (\d+) ", qir) == ["8"], qir

    assert qir == SlrConverter(prog).qir(), "QIR generation is not deterministic"
