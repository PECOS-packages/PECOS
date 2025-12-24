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
    assert (
        "measure b[0] -> m[0];" in qasm
    ), f"Expected 'measure b[0] -> m[0];' not found in QASM:\n{qasm}"
    assert (
        "measure a[1] -> m[1];" in qasm
    ), f"Expected 'measure a[1] -> m[1];' not found in QASM:\n{qasm}"

    # Verify that running QASM generation twice produces consistent results
    qasm2 = SlrConverter(prog).qasm()
    assert qasm == qasm2, "QASM generation is not deterministic"


# QIR Tests


@pytest.mark.optional_dependency
def test_individual_measurement_permutation_qir(
    individual_measurement_program: tuple,
) -> None:
    """Test individual measurements with permutations in QIR generation."""
    prog, _, _, _, _ = individual_measurement_program

    # Generate QIR
    qir = SlrConverter(prog).qir()

    # Print the QIR for debugging
    print("\nQIR output:")
    print(qir)

    # Verify that the QIR contains comments about the permutations
    assert (
        "; Permutation: a[0] -> b[0], b[0] -> a[0]" in qir
    ), f"Expected permutation comment not found in QIR:\n{qir}"
    assert (
        "; Permutation: m[0] -> n[0], n[0] -> m[0]" in qir
    ), f"Expected permutation comment not found in QIR:\n{qir}"

    # Verify that the QIR contains the correct classical bit permutation using a temporary bit
    assert (
        "%_bit_swap = call i1* @create_creg(i64 1)" in qir
    ), f"Expected temporary bit creation not found in QIR:\n{qir}"

    # Verify that the QIR contains the correct quantum operations after permutation
    # H gate should be applied to b[0] after permutation
    h_gate_pattern = r"call void @__quantum__qis__h__body\(%Qubit\* inttoptr \(i64 (\d+) to %Qubit\*\)\)"
    h_gates = re.findall(h_gate_pattern, qir)
    assert len(h_gates) >= 1, f"Expected at least one H gate, found {len(h_gates)}"

    # CX gate should be applied to b[0] and a[0] after permutation
    cx_gate_pattern = (
        r"call void @__quantum__qis__cnot__body\("
        r"%Qubit\* inttoptr \(i64 (\d+) to %Qubit\*\), "
        r"%Qubit\* inttoptr \(i64 (\d+) to %Qubit\*\)\)"
    )
    cx_gates = re.findall(cx_gate_pattern, qir)
    assert len(cx_gates) >= 1, f"Expected at least one CX gate, found {len(cx_gates)}"

    # Extract the measurement operations
    # In QIR, measurements are done with mz_to_creg_bit
    mz_to_creg_pattern = (
        r"call void @mz_to_creg_bit\("
        r"%Qubit\* inttoptr \(i64 (\d+) to %Qubit\*\), "
        r"i1\* %(\w+), i64 (\d+)\)"
    )
    mz_to_creg_calls = re.findall(mz_to_creg_pattern, qir)

    # We should have at least two measurement calls (one for each qubit in register a)
    assert (
        len(mz_to_creg_calls) >= 2
    ), f"Expected at least 2 measurement calls, found {len(mz_to_creg_calls)}"

    # Create a dictionary to store the measurements
    measurements = {}
    for qubit_idx, creg_name, creg_idx in mz_to_creg_calls:
        if creg_name == "m":
            measurements[int(creg_idx)] = (creg_name, int(qubit_idx))

    # Verify that the correct qubits are measured into the correct classical bits
    assert (
        0 in measurements
    ), f"Expected measurement to m[0], found measurements to {list(measurements.keys())}"
    assert (
        1 in measurements
    ), f"Expected measurement to m[1], found measurements to {list(measurements.keys())}"

    # Verify that different qubits are measured into different classical bits
    measured_qubits = [idx for _, idx in measurements.values()]
    assert len(set(measured_qubits)) == len(
        measured_qubits,
    ), f"Expected all measurements to be from different qubits, found duplicates: {measured_qubits}"

    # Verify that running QIR generation twice produces consistent results
    qir2 = SlrConverter(prog).qir()
    assert qir == qir2, "QIR generation is not deterministic"


@pytest.mark.optional_dependency
def test_register_measurement_permutation_qir(
    register_measurement_program: tuple,
) -> None:
    """Test register-wide measurements with permutations in QIR generation."""
    prog, _, _, _, _ = register_measurement_program

    # Generate QIR
    qir = SlrConverter(prog).qir()

    # Print the QIR for debugging
    print("\nQIR output:")
    print(qir)

    # Verify that the QIR contains comments about the permutations
    assert (
        "; Permutation: a[0] -> b[0], b[0] -> a[0]" in qir
    ), f"Expected permutation comment not found in QIR:\n{qir}"
    assert (
        "; Permutation: m[0] -> n[0], n[0] -> m[0]" in qir
    ), f"Expected permutation comment not found in QIR:\n{qir}"

    # Verify that the QIR contains the correct classical bit permutation using a temporary bit
    assert (
        "%_bit_swap = call i1* @create_creg(i64 1)" in qir
    ), f"Expected temporary bit creation not found in QIR:\n{qir}"
    assert (
        "call void @set_creg_bit(i1* %_bit_swap, i64 0, i1 %.4)" in qir
    ), f"Expected temporary bit assignment not found in QIR:\n{qir}"
    assert (
        "call void @set_creg_bit(i1* %m, i64 0, i1 %.6)" in qir
    ), f"Expected bit assignment not found in QIR:\n{qir}"
    assert (
        "call void @set_creg_bit(i1* %n, i64 0, i1 %.8)" in qir
    ), f"Expected bit assignment not found in QIR:\n{qir}"

    # Verify that the QIR contains the correct quantum operations after permutation
    # H gate should be applied to b[0] (qubit 2) after permutation
    assert (
        "call void @__quantum__qis__h__body(%Qubit* inttoptr (i64 2 to %Qubit*))" in qir
    ), f"Expected H gate on permuted qubit not found in QIR:\n{qir}"

    # CNOT gate should be applied to b[0] (qubit 2) and a[0] (qubit 0) after permutation
    assert (
        "call void @__quantum__qis__cnot__body("
        "%Qubit* inttoptr (i64 2 to %Qubit*), %Qubit* inttoptr (i64 0 to %Qubit*))"
        in qir
    ), f"Expected CNOT gate on permuted qubits not found in QIR:\n{qir}"

    # Verify that the QIR contains the correct measurements after permutation
    # a[0] should be measured as b[0] (qubit 2) after permutation
    assert (
        "call void @mz_to_creg_bit(%Qubit* inttoptr (i64 2 to %Qubit*), i1* %m, i64 0)"
        in qir
    ), f"Expected measurement of permuted qubit not found in QIR:\n{qir}"

    # a[1] should be measured as a[1] (qubit 1) since it's not permuted
    assert (
        "call void @mz_to_creg_bit(%Qubit* inttoptr (i64 1 to %Qubit*), i1* %m, i64 1)"
        in qir
    ), f"Expected measurement of non-permuted qubit not found in QIR:\n{qir}"

    # Verify that running QIR generation twice produces consistent results
    qir2 = SlrConverter(prog).qir()
    assert qir == qir2, "QIR generation is not deterministic"
