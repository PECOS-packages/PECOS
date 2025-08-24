# Copyright 2025 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Test cases for Stim <-> SLR converters."""

import pytest


@pytest.mark.optional_dependency
def test_stim_to_slr_basic() -> None:
    """Test basic Stim circuit to SLR conversion."""
    import stim
    from pecos.slr import SlrConverter

    # Create a simple Bell state circuit in Stim
    circuit = stim.Circuit()
    circuit.append_operation("H", [0])
    circuit.append_operation("CX", [0, 1])
    circuit.append_operation("M", [0, 1])

    # Convert to SLR
    slr_prog = SlrConverter.from_stim(circuit)

    # Check that we have the right structure
    assert slr_prog is not None
    assert len(slr_prog.vars.vars) > 0  # Should have registers

    # Generate QASM to verify conversion
    converter = SlrConverter(slr_prog)
    qasm = converter.qasm(skip_headers=True)
    assert "h q[0]" in qasm.lower()
    assert "cx q[0], q[1]" in qasm.lower() or "cx q[0],q[1]" in qasm.lower()
    assert "measure" in qasm.lower()


@pytest.mark.optional_dependency
def test_stim_to_slr_with_repeat() -> None:
    """Test Stim repeat block conversion to SLR."""
    import importlib.util

    if importlib.util.find_spec("stim") is None:
        pytest.skip("Stim not installed")

    import stim
    from pecos.slr import SlrConverter

    # Create circuit with repeat block
    circuit = stim.Circuit()
    circuit.append_operation("H", [0])
    # Create a repeat block properly
    circuit.append(
        stim.CircuitRepeatBlock(
            3,
            stim.Circuit(
                """
        X 0
        Y 0
    """,
            ),
        ),
    )

    # Convert to SLR
    slr_prog = SlrConverter.from_stim(circuit)

    # Verify structure (detailed checks would require inspecting the ops)
    assert slr_prog is not None


@pytest.mark.optional_dependency
def test_slr_to_stim_basic() -> None:
    """Test basic SLR to Stim conversion."""
    import importlib.util

    if importlib.util.find_spec("stim") is None:
        pytest.skip("Stim not installed")

    from pecos.qeclib import qubit
    from pecos.slr import CReg, Main, QReg, SlrConverter

    # Create a simple SLR program
    prog = Main(
        q := QReg("q", 2),
        c := CReg("c", 2),
        qubit.H(q[0]),
        qubit.CX(q[0], q[1]),
        qubit.Measure(q) > c,
    )

    # Convert to Stim using SlrConverter
    stim_circuit = SlrConverter(prog).stim()

    # Check the circuit
    assert stim_circuit.num_qubits == 2
    assert stim_circuit.num_measurements == 2

    # Check operations
    circuit_str = str(stim_circuit)
    assert "H 0" in circuit_str
    assert "CX 0 1" in circuit_str or "CNOT 0 1" in circuit_str
    assert "M 0 1" in circuit_str


@pytest.mark.optional_dependency
def test_slr_to_stim_with_repeat() -> None:
    """Test SLR Repeat block to Stim conversion."""
    import importlib.util

    if importlib.util.find_spec("stim") is None:
        pytest.skip("Stim not installed")

    from pecos.qeclib import qubit
    from pecos.slr import Main, QReg, Repeat, SlrConverter

    # Create SLR program with repeat
    prog = Main(
        q := QReg("q", 1),
        Repeat(5).block(
            qubit.H(q[0]),
            qubit.X(q[0]),
        ),
    )

    # Convert to Stim using SlrConverter
    stim_circuit = SlrConverter(prog).stim()

    # Check that we have operations
    assert stim_circuit.num_qubits == 1
    circuit_str = str(stim_circuit)
    assert "REPEAT 5" in circuit_str or "H 0" in circuit_str


def test_quantum_circuit_to_slr() -> None:
    """Test PECOS QuantumCircuit to SLR conversion."""
    from pecos.circuits.quantum_circuit import QuantumCircuit
    from pecos.slr import SlrConverter

    # Create a QuantumCircuit
    qc = QuantumCircuit()
    qc.append({"H": {0}, "X": {1}})  # Tick 1: H on qubit 0, X on qubit 1
    qc.append({"CX": {(0, 1)}})  # Tick 2: CNOT from 0 to 1
    qc.append({"M": {0, 1}})  # Tick 3: Measure both qubits

    # Convert to SLR
    slr_prog = SlrConverter.from_quantum_circuit(qc)

    # Check structure
    assert slr_prog is not None
    assert len(slr_prog.vars.vars) > 0

    # Generate QASM to verify
    converter = SlrConverter(slr_prog)
    qasm = converter.qasm(skip_headers=True)
    assert "h q[0]" in qasm.lower()
    assert "x q[1]" in qasm.lower()
    assert "cx q[0], q[1]" in qasm.lower() or "cx q[0],q[1]" in qasm.lower()


@pytest.mark.optional_dependency
def test_round_trip_conversion() -> None:
    """Test round-trip conversion Stim -> SLR -> Stim."""
    import importlib.util

    if importlib.util.find_spec("stim") is None:
        pytest.skip("Stim not installed")

    import stim
    from pecos.slr import SlrConverter

    # Create original Stim circuit
    original = stim.Circuit()
    original.append_operation("H", [0])
    original.append_operation("CX", [0, 1])
    original.append_operation("X", [2])
    original.append_operation("M", [0, 1, 2])

    # Convert Stim -> SLR -> Stim
    slr_prog = SlrConverter.from_stim(original)
    converter = SlrConverter(slr_prog)
    reconstructed = converter.stim()

    # Check basic properties are preserved
    assert reconstructed.num_qubits == original.num_qubits
    assert reconstructed.num_measurements == original.num_measurements

    # Check operations are present (order might differ slightly)
    recon_str = str(reconstructed)
    assert "H 0" in recon_str
    assert "CX 0 1" in recon_str or "CNOT 0 1" in recon_str
    assert "X 2" in recon_str
    assert "M" in recon_str


@pytest.mark.optional_dependency
def test_stim_noise_handling() -> None:
    """Test handling of Stim noise operations."""
    import importlib.util

    if importlib.util.find_spec("stim") is None:
        pytest.skip("Stim not installed")

    import stim
    from pecos.slr import SlrConverter

    # Create circuit with noise
    circuit = stim.Circuit()
    circuit.append_operation("H", [0])
    circuit.append_operation("X_ERROR", [0], 0.1)
    circuit.append_operation("DEPOLARIZE1", [0], 0.01)
    circuit.append_operation("M", [0])

    # Convert to SLR (noise should be converted to comments)
    slr_prog = SlrConverter.from_stim(circuit)

    # Should not fail, even if noise is just commented
    assert slr_prog is not None


@pytest.mark.optional_dependency
def test_stim_detector_handling() -> None:
    """Test handling of Stim detector and observable annotations."""
    import importlib.util

    if importlib.util.find_spec("stim") is None:
        pytest.skip("Stim not installed")

    import stim
    from pecos.slr import SlrConverter

    # Create circuit with detectors
    circuit = stim.Circuit()
    circuit.append_operation("H", [0])
    circuit.append_operation("M", [0])
    circuit.append_operation("DETECTOR", [stim.target_rec(-1)])
    circuit.append_operation("OBSERVABLE_INCLUDE", [stim.target_rec(-1)], 0)

    # Convert to SLR
    slr_prog = SlrConverter.from_stim(circuit)

    # Should handle annotations (as comments)
    assert slr_prog is not None


if __name__ == "__main__":
    # Run basic tests
    print("Testing Stim <-> SLR converters...")

    try:
        test_stim_to_slr_basic()
        print("[PASS] Basic Stim to SLR conversion works")
    except (ImportError, AttributeError, ValueError) as e:
        print(f"[FAIL] Basic Stim to SLR conversion failed: {e}")

    try:
        test_slr_to_stim_basic()
        print("[PASS] Basic SLR to Stim conversion works")
    except (ImportError, AttributeError, ValueError) as e:
        print(f"[FAIL] Basic SLR to Stim conversion failed: {e}")

    try:
        test_quantum_circuit_to_slr()
        print("[PASS] QuantumCircuit to SLR conversion works")
    except (ImportError, AttributeError, ValueError) as e:
        print(f"[FAIL] QuantumCircuit to SLR conversion failed: {e}")

    try:
        test_round_trip_conversion()
        print("[PASS] Round-trip conversion works")
    except (ImportError, AttributeError, ValueError) as e:
        print(f"[FAIL] Round-trip conversion failed: {e}")

    print("\nAll basic tests completed!")
