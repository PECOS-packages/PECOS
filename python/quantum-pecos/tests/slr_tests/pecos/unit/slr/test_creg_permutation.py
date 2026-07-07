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
        pecos.slr.Return(a, b),
    )


def test_creg_permutation_qasm() -> None:
    """Test permutation of whole classical registers followed by both bit and register operations in QASM."""
    prog = create_creg_permutation_program()
    qasm = SlrConverter(prog).qasm()

    # Print the QASM for debugging
    print("\nQASM output:")
    print(qasm)

    # Verify the XOR swap operations are generated
    assert "a = a ^ b;" in qasm, f"Expected 'a = a ^ b;' not found in QASM:\n{qasm}"
    assert "b = b ^ a;" in qasm, f"Expected 'b = b ^ a;' not found in QASM:\n{qasm}"
    assert "a = a ^ b;" in qasm, f"Expected 'a = a ^ b;' not found in QASM:\n{qasm}"

    # Verify the temporary bit approach is NOT used for whole register permutations
    assert "creg _bit_swap[1];" not in qasm, f"Unexpected 'creg _bit_swap[1];' found in QASM:\n{qasm}"

    # Verify the permutation comment is correct
    assert "// Permutation: a <-> b" in qasm, f"Expected permutation comment not found in QASM:\n{qasm}"

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
    """Whole-register CReg Permute IS realized in QIR (static CReg model).

    A Permute is realized as a static logical relabel consulted
    at every classical-bit lowering (the bespoke
    @set_creg_bit/@set_creg_to_int/creg-xor helpers the old test
    pinned were removed by the static CReg model). `Permute(a, b)` relabels a[0] <->
    b[0], so the subsequent `a[0].set(1)` writes b[0]'s storage.
    """
    prog = create_creg_permutation_program()
    qir = SlrConverter(prog).qir()

    assert "; Permutation: a <-> b" in qir, f"Expected whole-register permutation comment in QIR:\n{qir}"
    # a[0].set(1) after Permute(a, b) -> stores into register b's
    # `[1 x i1]` buffer (a[0] now resolves to b[0]); register a's
    # buffer is never the store target for that set.
    assert re.search(r"store i1 1, (?:ptr|i1\*) %\.\d+", qir), qir
    b_slot = re.search(r"%(\.\d+) = getelementptr \[1 x i1\], (?:ptr|\[1 x i1\]\*) %b, i64 0, i64 0", qir)
    assert b_slot, f"Expected a[0].set(1) to target register b's buffer (relabelled):\n{qir}"

    assert qir == SlrConverter(prog).qir(), "QIR generation is not deterministic"
