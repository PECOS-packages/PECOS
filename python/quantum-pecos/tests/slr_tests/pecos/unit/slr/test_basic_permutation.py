"""Tests for basic permutation functionality in both QASM and QIR generation."""

import re

import pytest
from pecos.slr import CReg, Main, Permute, Return, SlrConverter

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
        Return(a, b),
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
        Return(a),
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
        Return(a, b),
    )

    qasm1 = SlrConverter(prog).qasm()
    qasm2 = SlrConverter(prog).qasm()

    # Print the QASM for debugging
    print("\nQASM output:")
    print(qasm1)

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
    from pecos.slr.gen_codes.gen_qasm import QASMGenerator
    from pecos.slr.slr_converter import SlrConverter

    # Create a custom QASM generator to debug the permutation map
    generator = QASMGenerator(_internal=True)
    generator.generate_block(prog)
    qasm = generator.get_output()

    # Print the QASM for debugging
    print("\nQASM output:")
    print(qasm)

    # Print the permutation map
    print("\nPermutation map:")
    print(generator.permutation_map)

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
    # Remove version comments for comparison as they might differ
    qasm_lines = [line for line in qasm.split("\n") if not line.startswith("// Generated using:")]
    qasm2_lines = [line for line in qasm2.split("\n") if not line.startswith("// Generated using:")]
    assert "\n".join(qasm_lines) == "\n".join(
        qasm2_lines,
    ), "QASM generation is not deterministic"


def test_same_register_permutation_qasm(
    same_register_permutation_program: tuple,
) -> None:
    """Test permutation of elements within the same register in QASM."""
    prog, _ = same_register_permutation_program

    qasm = SlrConverter(prog).qasm()

    # Print the QASM for debugging
    print("\nQASM output:")
    print(qasm)

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


# A Permute is realized as a static logical relabel consulted at
# every classical-bit lowering (mirroring the Guppy linearity
# tracker's `.permute()`). The bespoke @set_creg_bit helpers the old
# tests pinned were removed by the static CReg model; bit writes are now `store i1`
# into the relabelled register's `[N x i1]` buffer.


@pytest.mark.optional_dependency
def test_basic_permutation_qir(basic_permutation_program: tuple) -> None:
    """Element-wise CReg Permute is realized (a[0] <-> b[1])."""
    prog, _, _ = basic_permutation_program
    qir = SlrConverter(prog).qir()

    assert "; Permutation: a[0] -> b[1], b[1] -> a[0]" in qir, qir
    # a[0].set(1) after the swap writes b[1]'s slot, not a[0]'s.
    m = re.search(r"%(\.\d+) = getelementptr \[2 x i1\], \[2 x i1\]\* %b, i64 0, i64 1\n\s*store i1 1, i1\* %\1", qir)
    assert m, f"Expected a[0].set(1) to store into b[1] (relabelled):\n{qir}"

    assert qir == SlrConverter(prog).qir(), "QIR generation is not deterministic"


@pytest.mark.optional_dependency
def test_same_register_permutation_qir(
    same_register_permutation_program: tuple,
) -> None:
    """Element-wise same-CReg cycle Permute is realized."""
    prog, _ = same_register_permutation_program
    qir = SlrConverter(prog).qir()

    assert "; Permutation: a[0] -> a[2], a[1] -> a[0], a[2] -> a[1]" in qir, qir

    # a[0].set(1)->a[2], a[1].set(0)->a[0], a[2].set(1)->a[1].
    def _stored(slot: int, val: int) -> bool:
        pat = (
            rf"%(\.\d+) = getelementptr \[3 x i1\], \[3 x i1\]\* %a, i64 0, i64 {slot}\n"
            rf"\s*store i1 {val}, i1\* %\1"
        )
        return bool(
            re.search(
                pat,
                qir,
            ),
        )

    assert _stored(2, 1), qir  # a[0].set(1) -> a[2]
    assert _stored(0, 0), qir  # a[1].set(0) -> a[0]
    assert _stored(1, 1), qir  # a[2].set(1) -> a[1]

    assert qir == SlrConverter(prog).qir(), "QIR generation is not deterministic"
