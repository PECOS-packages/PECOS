"""Tests for quantum permutation functionality in both QASM and QIR generation."""

import re

import pytest
from pecos.qeclib import qubit
from pecos.slr import CReg, Main, Permute, QReg, SlrConverter

# QASM Tests


def test_permutation_consistency_with_multiple_calls() -> None:
    """Test that multiple calls to qasm() produce the same result."""
    prog = Main(
        a := QReg("a", 2),
        b := QReg("b", 2),
        Permute(
            [a[0], a[1], b[0], b[1]],
            [b[0], b[1], a[0], a[1]],
        ),
        qubit.H(a[0]),  # Should become H b[0];
        qubit.X(a[1]),  # Should become X b[1];
        qubit.Z(b[0]),  # Should become Z a[0];
        qubit.Y(b[1]),  # Should become Y a[1];
    )

    qasm1 = SlrConverter(prog).qasm()
    qasm2 = SlrConverter(prog).qasm()
    qasm3 = SlrConverter(prog).qasm()

    assert qasm1 == qasm2
    assert qasm2 == qasm3

    # Check that the permutation was applied correctly
    assert "h b[0];" in qasm1.lower()
    assert "x b[1];" in qasm1.lower()
    assert "z a[0];" in qasm1.lower()
    assert "y a[1];" in qasm1.lower()


def test_quantum_permutation_qasm(quantum_permutation_program: tuple) -> None:
    """Test permutation with quantum gates in QASM generation."""
    prog, _, _ = quantum_permutation_program

    # Generate QASM
    qasm = SlrConverter(prog).qasm()

    # Verify that the QASM contains the correct permuted quantum operations
    assert "h b[0];" in qasm
    assert "cx b[0], a[1];" in qasm

    # Verify that running QASM generation twice produces consistent results
    qasm2 = SlrConverter(prog).qasm()
    assert qasm == qasm2, "QASM generation is not deterministic"


# QIR Tests


@pytest.mark.optional_dependency
def test_quantum_permutation_qir(quantum_permutation_program: tuple) -> None:
    """Test permutation with quantum gates in QIR generation."""
    prog, _, _ = quantum_permutation_program

    # Generate QIR
    qir = SlrConverter(prog).qir()

    # Print the QIR for analysis
    # print("\nQIR Output for quantum_permutation_qir:")
    # print(qir)

    # Verify that the QIR contains a comment about the permutation
    assert "Permutation: a[0] -> b[0], b[0] -> a[0]" in qir

    # Extract the qubit indices used in the H and CNOT operations
    h_calls = re.findall(
        r"call void @__quantum__qis__h__body\(%Qubit\* inttoptr \(i64 (\d+) to %Qubit\*\)\)",
        qir,
    )
    cnot_calls = re.findall(
        r"call void @__quantum__qis__cnot__body\("
        r"%Qubit\* inttoptr \(i64 (\d+) to %Qubit\*\), "
        r"%Qubit\* inttoptr \(i64 (\d+) to %Qubit\*\)\)",
        qir,
    )

    # print(f"H calls found: {h_calls}")
    # print(f"CNOT calls found: {cnot_calls}")

    # We should have at least one H call and one CNOT call
    assert len(h_calls) >= 1, "No H gate call found"
    assert len(cnot_calls) >= 1, "No CNOT gate call found"

    # Get the qubit indices
    h_qubit = int(h_calls[0])
    cnot_control, _cnot_target = map(int, cnot_calls[0])

    # Verify that the H and CNOT operations are applied to the correct qubits after permutation
    # The exact indices will depend on how qubits are allocated in the QIR generator
    # We can't assert the exact indices without knowing the allocation strategy
    # But we can verify that the CNOT control qubit is the same as the H qubit
    assert (
        h_qubit == cnot_control
    ), f"H applied to qubit {h_qubit}, but CNOT control is qubit {cnot_control}"

    # Verify that running QIR generation twice produces consistent results
    qir2 = SlrConverter(prog).qir()
    assert qir == qir2, "QIR generation is not deterministic"


@pytest.mark.optional_dependency
def test_permutation_with_bell_circuit_qir() -> None:
    """Test permutation functionality with a Bell circuit in QIR generation."""
    # Create a program with permutations and a Bell circuit
    a = QReg("a", 2)
    b = QReg("b", 2)
    m = CReg("m", 2)
    n = CReg("n", 2)

    prog = Main(
        a,
        b,
        m,
        n,
        # Permute quantum registers
        Permute(
            [a[0], b[1]],
            [b[1], a[0]],
        ),
        # Permute classical registers
        Permute(
            [m[0], n[0]],
            [n[0], m[0]],
        ),
        # Apply H gate to a[0] - should be applied to b[1] after permutation
        qubit.H(a[0]),
        # Apply CX gate from a[0] to a[1] - should be from b[1] to a[1] after permutation
        qubit.CX(a[0], a[1]),
        # Measure individual qubits to individual bits
        qubit.Measure(a[0]) > m[0],
        qubit.Measure(a[1]) > m[1],
    )

    # Generate QIR
    qir = SlrConverter(prog).qir()

    # Print the QIR for analysis
    # print("\nQIR Output for bell_circuit_qir:")
    # print(qir)

    # Verify that the QIR contains comments about the permutations
    assert "Permutation: a[0] -> b[1], b[1] -> a[0]" in qir
    assert "Permutation: m[0] -> n[0], n[0] -> m[0]" in qir

    # Extract the quantum operations
    h_calls = re.findall(
        r"call void @__quantum__qis__h__body\(%Qubit\* inttoptr \(i64 (\d+) to %Qubit\*\)\)",
        qir,
    )
    cx_calls = re.findall(
        r"call void @__quantum__qis__cnot__body\("
        r"%Qubit\* inttoptr \(i64 (\d+) to %Qubit\*\), "
        r"%Qubit\* inttoptr \(i64 (\d+) to %Qubit\*\)\)",
        qir,
    )

    # print(f"H calls found: {h_calls}")
    # print(f"CX calls found: {cx_calls}")

    # We should have at least one H call and one CX call
    assert len(h_calls) >= 1, "No H gate call found"
    assert len(cx_calls) >= 1, "No CX gate call found"

    # Extract the measurement operations
    mz_calls = re.findall(
        r"call %Result\* @__quantum__qis__mz__body\(%Qubit\* inttoptr \(i64 (\d+) to %Qubit\*\)\)",
        qir,
    )
    mz_to_creg_calls = re.findall(
        r"call void @mz_to_creg_bit\(%Qubit\* inttoptr \(i64 (\d+) to %Qubit\*\), i1\* %(\w+), i64 (\d+)\)",
        qir,
    )

    # print(f"MZ calls found: {mz_calls}")
    # print(f"MZ to creg calls found: {mz_to_creg_calls}")

    # We should have at least two measurement calls (one for each qubit)
    assert len(mz_calls) + len(mz_to_creg_calls) >= 2, (
        f"Expected at least 2 measurement calls, found {len(mz_calls)} mz calls "
        f"and {len(mz_to_creg_calls)} mz_to_creg calls"
    )


@pytest.mark.optional_dependency
def test_comprehensive_qir_verification() -> None:
    """Test comprehensive verification of QIR generation with permutations."""
    # Create a program with a variety of operations to test permutation effects
    a = QReg("a", 2)
    b = QReg("b", 2)
    c = QReg("c", 2)
    m = CReg("m", 2)
    n = CReg("n", 2)

    prog = Main(
        a,
        b,
        c,
        m,
        n,
        # Apply some initial gates to track qubit allocation
        qubit.H(a[0]),  # Track as "original a[0]"
        qubit.X(a[1]),  # Track as "original a[1]"
        qubit.Y(b[0]),  # Track as "original b[0]"
        qubit.Z(b[1]),  # Track as "original b[1]"
        # First permutation: swap a[0] and b[0]
        Permute(
            [a[0], b[0]],
            [b[0], a[0]],
        ),
        # Apply gates after first permutation
        qubit.H(a[0]),  # Should apply to "original b[0]"
        qubit.X(b[0]),  # Should apply to "original a[0]"
        # Second permutation: swap a[1] and b[1]
        Permute(
            [a[1], b[1]],
            [b[1], a[1]],
        ),
        # Apply gates after second permutation
        qubit.Y(a[1]),  # Should apply to "original b[1]"
        qubit.Z(b[1]),  # Should apply to "original a[1]"
        # Apply some two-qubit gates to test cross-register operations
        qubit.CX(a[0], b[1]),  # Should be CX from "original b[0]" to "original a[1]"
        # Measure qubits to classical bits
        qubit.Measure(a[0]) > m[0],  # Should measure "original b[0]" to m[0]
        qubit.Measure(b[1]) > n[0],  # Should measure "original a[1]" to n[0]
    )

    # Generate QIR
    qir = SlrConverter(prog).qir()

    # Print the QIR for analysis
    # print("\nQIR Output for comprehensive_qir_verification:")
    # print(qir)

    # Extract all gate operations to track qubit allocation
    h_calls = re.findall(
        r"call void @__quantum__qis__h__body\(%Qubit\* inttoptr \(i64 (\d+) to %Qubit\*\)\)",
        qir,
    )
    x_calls = re.findall(
        r"call void @__quantum__qis__x__body\(%Qubit\* inttoptr \(i64 (\d+) to %Qubit\*\)\)",
        qir,
    )
    y_calls = re.findall(
        r"call void @__quantum__qis__y__body\(%Qubit\* inttoptr \(i64 (\d+) to %Qubit\*\)\)",
        qir,
    )
    z_calls = re.findall(
        r"call void @__quantum__qis__z__body\(%Qubit\* inttoptr \(i64 (\d+) to %Qubit\*\)\)",
        qir,
    )
    cx_calls = re.findall(
        r"call void @__quantum__qis__cnot__body\("
        r"%Qubit\* inttoptr \(i64 (\d+) to %Qubit\*\), "
        r"%Qubit\* inttoptr \(i64 (\d+) to %Qubit\*\)\)",
        qir,
    )
    mz_to_creg_calls = re.findall(
        r"call void @mz_to_creg_bit\(%Qubit\* inttoptr \(i64 (\d+) to %Qubit\*\), i1\* %(\w+), i64 (\d+)\)",
        qir,
    )

    # print(f"H calls: {h_calls}")
    # print(f"X calls: {x_calls}")
    # print(f"Y calls: {y_calls}")
    # print(f"Z calls: {z_calls}")
    # print(f"CX calls: {cx_calls}")
    # print(f"MZ to creg calls: {mz_to_creg_calls}")

    # Based on the initial gates, we can infer the qubit allocation:
    # The first H call should be for "original a[0]"
    # The first X call should be for "original a[1]"
    # The first Y call should be for "original b[0]"
    # The first Z call should be for "original b[1]"
    if (
        len(h_calls) >= 1
        and len(x_calls) >= 1
        and len(y_calls) >= 1
        and len(z_calls) >= 1
    ):
        original_a0 = int(h_calls[0])
        original_a1 = int(x_calls[0])
        original_b0 = int(y_calls[0])
        original_b1 = int(z_calls[0])

        # print("Inferred qubit allocation:")
        # print(f"  original a[0] -> physical qubit {original_a0}")
        # print(f"  original a[1] -> physical qubit {original_a1}")
        # print(f"  original b[0] -> physical qubit {original_b0}")
        # print(f"  original b[1] -> physical qubit {original_b1}")

        # Now we can verify that the gates after permutations are applied to the correct qubits
        # The second H call should be for "original b[0]"
        # The second X call should be for "original a[0]"
        if len(h_calls) >= 2 and len(x_calls) >= 2:
            assert int(h_calls[1]) == original_b0, (
                f"Second H gate should be applied to original b[0] "
                f"(physical qubit {original_b0}), but was applied to physical qubit {h_calls[1]}"
            )
            assert int(x_calls[1]) == original_a0, (
                f"Second X gate should be applied to original a[0] "
                f"(physical qubit {original_a0}), but was applied to physical qubit {x_calls[1]}"
            )

        # The second Y call should be for "original b[1]"
        # The second Z call should be for "original a[1]"
        if len(y_calls) >= 2 and len(z_calls) >= 2:
            assert int(y_calls[1]) == original_b1, (
                f"Second Y gate should be applied to original b[1] "
                f"(physical qubit {original_b1}), but was applied to physical qubit {y_calls[1]}"
            )
            assert int(z_calls[1]) == original_a1, (
                f"Second Z gate should be applied to original a[1] "
                f"(physical qubit {original_a1}), but was applied to physical qubit {z_calls[1]}"
            )

        # The CX gate should be from "original b[0]" to "original a[1]"
        if len(cx_calls) >= 1:
            cx_control, cx_target = map(int, cx_calls[0])
            assert (
                cx_control == original_b0
            ), f"CX control should be original b[0] (physical qubit {original_b0}), but was physical qubit {cx_control}"
            assert (
                cx_target == original_a1
            ), f"CX target should be original a[1] (physical qubit {original_a1}), but was physical qubit {cx_target}"

        # The measurements should be from "original b[0]" to m[0] and from "original a[1]" to n[0]
        if len(mz_to_creg_calls) >= 2:
            mz1_qubit, mz1_reg, mz1_idx = mz_to_creg_calls[0]
            mz2_qubit, mz2_reg, mz2_idx = mz_to_creg_calls[1]

            # Check if either measurement matches our expectations
            b0_to_m0 = (
                int(mz1_qubit) == original_b0 and mz1_reg == "m" and int(mz1_idx) == 0
            ) or (
                int(mz2_qubit) == original_b0 and mz2_reg == "m" and int(mz2_idx) == 0
            )
            a1_to_n0 = (
                int(mz1_qubit) == original_a1 and mz1_reg == "n" and int(mz1_idx) == 0
            ) or (
                int(mz2_qubit) == original_a1 and mz2_reg == "n" and int(mz2_idx) == 0
            )

            assert b0_to_m0, (
                f"Expected measurement from original b[0] (physical qubit {original_b0}) to m[0], "
                f"but found measurements: {mz_to_creg_calls}"
            )
            assert a1_to_n0, (
                f"Expected measurement from original a[1] (physical qubit {original_a1}) to n[0], "
                f"but found measurements: {mz_to_creg_calls}"
            )


@pytest.mark.optional_dependency
def test_rotation_gates_with_permutation() -> None:
    """Test that permutations work correctly with rotation gates in QIR generation."""
    # Create a program with rotation gates and permutations
    a = QReg("a", 2)
    b = QReg("b", 2)

    prog = Main(
        a,
        b,
        # Apply initial gates to track qubit allocation
        qubit.RX[0.1](a[0]),  # Track as "original a[0]"
        qubit.RY[0.2](a[1]),  # Track as "original a[1]"
        qubit.RZ[0.3](b[0]),  # Track as "original b[0]"
        qubit.SZ(b[1]),  # Track as "original b[1]"
        # Apply permutation
        Permute(
            [a[0], b[0]],
            [b[0], a[0]],
        ),
        # Apply gates after permutation
        qubit.RX[0.4](a[0]),  # Should apply to "original b[0]"
        qubit.RY[0.5](b[0]),  # Should apply to "original a[0]"
        qubit.T(a[1]),  # Should apply to "original a[1]"
        qubit.Tdg(b[1]),  # Should apply to "original b[1]"
    )

    # Generate QIR
    qir = SlrConverter(prog).qir()

    # Print the QIR for analysis
    # print("\nQIR Output for rotation_gates_with_permutation:")
    # print(qir)

    # Extract all gate operations to track qubit allocation
    rx_calls = re.findall(
        r"call void @__quantum__qis__rx__body\(double (0x[0-9a-f]+), %Qubit\* inttoptr \(i64 (\d+) to %Qubit\*\)\)",
        qir,
    )
    ry_calls = re.findall(
        r"call void @__quantum__qis__ry__body\(double (0x[0-9a-f]+), %Qubit\* inttoptr \(i64 (\d+) to %Qubit\*\)\)",
        qir,
    )
    rz_calls = re.findall(
        r"call void @__quantum__qis__rz__body\(double (0x[0-9a-f]+), %Qubit\* inttoptr \(i64 (\d+) to %Qubit\*\)\)",
        qir,
    )
    s_calls = re.findall(
        r"call void @__quantum__qis__s__body\(%Qubit\* inttoptr \(i64 (\d+) to %Qubit\*\)\)",
        qir,
    )
    t_calls = re.findall(
        r"call void @__quantum__qis__t__body\(%Qubit\* inttoptr \(i64 (\d+) to %Qubit\*\)\)",
        qir,
    )
    tdg_calls = re.findall(
        r"call void @__quantum__qis__t__adj\(%Qubit\* inttoptr \(i64 (\d+) to %Qubit\*\)\)",
        qir,
    )

    # print(f"Rx calls: {rx_calls}")
    # print(f"Ry calls: {ry_calls}")
    # print(f"Rz calls: {rz_calls}")
    # print(f"S calls: {s_calls}")
    # print(f"T calls: {t_calls}")
    # print(f"Tdg calls: {tdg_calls}")

    # Based on the initial gates, we can infer the qubit allocation:
    if (
        len(rx_calls) >= 1
        and len(ry_calls) >= 1
        and len(rz_calls) >= 1
        and len(s_calls) >= 1
    ):
        # Extract the qubit indices from the first calls
        original_a0 = int(rx_calls[0][1])
        original_a1 = int(ry_calls[0][1])
        original_b0 = int(rz_calls[0][1])
        original_b1 = int(s_calls[0])

        # print("Inferred qubit allocation:")
        # print(f"  original a[0] -> physical qubit {original_a0}")
        # print(f"  original a[1] -> physical qubit {original_a1}")
        # print(f"  original b[0] -> physical qubit {original_b0}")
        # print(f"  original b[1] -> physical qubit {original_b1}")

        # Now we can verify that the gates after permutations are applied to the correct qubits
        if len(rx_calls) >= 2 and len(ry_calls) >= 2:
            assert int(rx_calls[1][1]) == original_b0, (
                f"Second Rx gate should be applied to original b[0] "
                f"(physical qubit {original_b0}), but was applied to physical qubit {rx_calls[1][1]}"
            )
            assert int(ry_calls[1][1]) == original_a0, (
                f"Second Ry gate should be applied to original a[0] "
                f"(physical qubit {original_a0}), but was applied to physical qubit {ry_calls[1][1]}"
            )

        if len(t_calls) >= 1 and len(tdg_calls) >= 1:
            assert int(t_calls[0]) == original_a1, (
                f"T gate should be applied to original a[1] "
                f"(physical qubit {original_a1}), but was applied to physical qubit {t_calls[0]}"
            )
            assert int(tdg_calls[0]) == original_b1, (
                f"Tdg gate should be applied to original b[1] "
                f"(physical qubit {original_b1}), but was applied to physical qubit {tdg_calls[0]}"
            )
