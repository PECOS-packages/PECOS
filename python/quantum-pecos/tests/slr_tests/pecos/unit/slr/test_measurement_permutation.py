"""Tests for measurement with permutation functionality in both QASM and QIR generation."""

import re

import pytest
from pecos.slr import SlrConverter

# QASM Tests


def test_individual_measurement_permutation_qasm(
    individual_measurement_program: tuple,
) -> None:
    """Test individual measurements with permutations in QASM generation."""
    prog, _, _, _, _ = individual_measurement_program

    # Generate QASM
    qasm = SlrConverter(prog).qasm()

    # Print the QASM for debugging
    print("\nQASM output:")
    print(qasm)

    # Verify that the QASM contains the correct permuted measurements
    # After permutation: a[0] -> b[0], m[0] -> n[0]
    # For classical bit permutations, operations still refer to the original bit names
    # For quantum registers, we still use the permutation map approach
    assert "measure b[0] -> m[0];" in qasm
    assert "measure a[1] -> m[1];" in qasm

    # Verify that the bit permutation is using the temporary bit approach, not XOR swap
    assert "creg _bit_swap[1];" in qasm
    assert "_bit_swap[0] = m[0];" in qasm
    assert "m[0] = n[0];" in qasm
    assert "n[0] = _bit_swap[0];" in qasm
    assert "m[0] = m[0] ^ n[0];" not in qasm  # Make sure XOR swap is not used

    # Verify that running QASM generation twice produces consistent results
    qasm2 = SlrConverter(prog).qasm()
    assert qasm == qasm2, "QASM generation is not deterministic"


def test_register_measurement_permutation_qasm(
    register_measurement_program: tuple,
) -> None:
    """Test register-wide measurements with permutations in QASM generation."""
    prog, _, _, _, _ = register_measurement_program

    # Generate QASM
    qasm = SlrConverter(prog).qasm()

    # Print the QASM for debugging
    print("\nQASM output:")
    print(qasm)

    # Register-wide measurements are now unrolled correctly with permutations
    # The expected behavior is:
    assert "measure b[0] -> m[0];" in qasm, f"Expected 'measure b[0] -> m[0];' not found in QASM:\n{qasm}"
    assert "measure a[1] -> m[1];" in qasm, f"Expected 'measure a[1] -> m[1];' not found in QASM:\n{qasm}"

    # Verify that running QASM generation twice produces consistent results
    qasm2 = SlrConverter(prog).qasm()
    assert qasm == qasm2, "QASM generation is not deterministic"


# QIR Tests


# A Permute is realized as a static logical relabel consulted at
# every qubit/classical-bit lowering (the bespoke
# @create_creg/@set_creg_bit/@mz_to_creg_bit helpers the old tests
# pinned were removed by the static CReg model; measurement is the standard 2-arg
# `@__quantum__qis__mz__body(%Qubit*, %Result*)` + read_result +
# store-into-creg-buffer). Pin the realized measurement targeting.


def _mz_then_store(qir: str) -> list[tuple[str, str, str]]:
    """(qubit_idx, creg_name, creg_idx) for each measure+store."""
    pat = (
        r"call void @__quantum__qis__mz__body\(%Qubit\* inttoptr \(i64 (\d+) to %Qubit\*\), "
        r"%Result\* inttoptr \(i64 (\d+) to %Result\*\)\)\n"
        r"\s*%(?:\.\d+) = call i1 @__quantum__rt__read_result\(%Result\* inttoptr \(i64 \2 to %Result\*\)\)\n"
        r"\s*%(\.\d+) = getelementptr \[\d+ x i1\], (?:ptr|\[\d+ x i1\]\*) %(\w+), i64 0, i64 (\d+)"
    )
    # groups: 1=qubit, 2=result idx, 3=gep var, 4=creg name, 5=creg idx
    return [(m[0], m[3], m[4]) for m in re.findall(pat, qir)]


@pytest.mark.optional_dependency
def test_individual_measurement_permutation_qir(
    individual_measurement_program: tuple,
) -> None:
    """Element-wise QReg+CReg Permute is realized through measurements."""
    prog, _, _, _, _ = individual_measurement_program
    qir = SlrConverter(prog).qir()

    assert "; Permutation: a[0] -> b[0], b[0] -> a[0]" in qir, qir
    assert "; Permutation: m[0] -> n[0], n[0] -> m[0]" in qir, qir
    # Qubits a=0,1 b=2,3. Measure(a[0]) -> b[0]=q2, result -> m[0]
    # which is relabelled to n[0]. Measure(a[1]) -> q1 (unpermuted),
    # result -> m[1] (unpermuted).
    assert _mz_then_store(qir) == [("2", "n", "0"), ("1", "m", "1")], qir

    assert qir == SlrConverter(prog).qir(), "QIR generation is not deterministic"


@pytest.mark.optional_dependency
def test_register_measurement_permutation_qir(
    register_measurement_program: tuple,
) -> None:
    """Element-wise Permute is realized through a register-wide measure."""
    prog, _, _, _, _ = register_measurement_program
    qir = SlrConverter(prog).qir()

    assert "; Permutation: a[0] -> b[0], b[0] -> a[0]" in qir, qir
    assert "; Permutation: m[0] -> n[0], n[0] -> m[0]" in qir, qir
    # Measure(a) unrolls: a[0]->b[0]=q2 (-> m[0] relabelled to n[0]),
    # a[1]->q1 (-> m[1] unpermuted).
    assert _mz_then_store(qir) == [("2", "n", "0"), ("1", "m", "1")], qir

    assert qir == SlrConverter(prog).qir(), "QIR generation is not deterministic"
