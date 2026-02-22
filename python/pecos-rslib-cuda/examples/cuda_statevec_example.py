#!/usr/bin/env python3
# Copyright 2026 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.

"""Example of using CudaStateVec for GPU-accelerated quantum simulation.

This example demonstrates:
- Creating a CudaStateVec simulator
- Applying various quantum gates
- Creating and measuring Bell states
- Using rotation gates

Requirements:
- NVIDIA GPU with CUDA support
- cuQuantum SDK installed
- pecos-rslib-cuda package
"""

import math
import sys


def check_cuda_available():
    """Check if CUDA/cuQuantum is available."""
    try:
        from pecos_rslib_cuda import is_cuquantum_available

        if not is_cuquantum_available():
            print("cuQuantum SDK is not available.")
            print("Please install cuQuantum SDK from NVIDIA.")
            return False
        return True
    except ImportError:
        print("pecos-rslib-cuda is not installed.")
        print("Please build and install it: cd python/pecos-rslib-cuda && maturin develop")
        return False


def bell_state_example():
    """Demonstrate Bell state creation and measurement."""
    print("=" * 60)
    print("Bell State Example with CudaStateVec")
    print("=" * 60)

    from pecos.simulators import CudaStateVec

    # Create a 2-qubit simulator with a fixed seed for reproducibility
    sim = CudaStateVec(2, seed=42)
    print(f"Created CudaStateVec simulator with {sim.num_qubits} qubits")

    # Create Bell state |Phi+> = (|00> + |11>) / sqrt(2)
    print("\nCreating Bell state |Phi+>:")
    print("  1. Apply Hadamard to qubit 0")
    sim.run_gate("H", [0])

    print("  2. Apply CNOT with control=0, target=1")
    sim.run_gate("CX", [(0, 1)])

    # Measure both qubits
    print("\nMeasuring both qubits...")
    results = sim.run_gate("Measure", [0, 1])
    print(f"Measurement results: {results}")

    # Run multiple shots to see correlation
    print("\nRunning 100 shots to verify Bell state correlation:")
    correlations = {"00": 0, "01": 0, "10": 0, "11": 0}
    for _i in range(100):
        sim = CudaStateVec(2)  # New simulator for each shot
        sim.run_gate("H", [0])
        sim.run_gate("CX", [(0, 1)])
        results = sim.run_gate("Measure", [0, 1])

        # Extract measurement values
        if isinstance(results, dict):
            vals = list(results.values())
            if len(vals) == 2:
                key = f"{vals[0]}{vals[1]}"
                correlations[key] = correlations.get(key, 0) + 1

    print(f"  |00>: {correlations.get('00', 0)} times")
    print(f"  |01>: {correlations.get('01', 0)} times (should be ~0)")
    print(f"  |10>: {correlations.get('10', 0)} times (should be ~0)")
    print(f"  |11>: {correlations.get('11', 0)} times")
    print("  Note: |00> and |11> should be roughly equal, |01> and |10> should be ~0")


def rotation_gates_example():
    """Demonstrate rotation gates."""
    print("\n" + "=" * 60)
    print("Rotation Gates Example")
    print("=" * 60)

    from pecos.simulators import CudaStateVec

    sim = CudaStateVec(3, seed=123)
    print(f"Created CudaStateVec simulator with {sim.num_qubits} qubits")

    # Apply various rotation gates
    print("\nApplying rotation gates:")

    print("  RX(pi/4) on qubit 0")
    sim.run_gate("RX", [0], angles=(math.pi / 4,))

    print("  RY(pi/2) on qubit 1")
    sim.run_gate("RY", [1], angles=(math.pi / 2,))

    print("  RZ(pi) on qubit 2")
    sim.run_gate("RZ", [2], angles=(math.pi,))

    # Two-qubit rotation
    print("  RZZ(pi/4) on qubits (0, 1)")
    sim.run_gate("RZZ", [(0, 1)], angles=(math.pi / 4,))

    # Measure
    print("\nMeasuring all qubits...")
    results = sim.run_gate("Measure", [0, 1, 2])
    print(f"Measurement results: {results}")


def clifford_gates_example():
    """Demonstrate Clifford gates."""
    print("\n" + "=" * 60)
    print("Clifford Gates Example")
    print("=" * 60)

    from pecos.simulators import CudaStateVec

    sim = CudaStateVec(4)
    print(f"Created CudaStateVec simulator with {sim.num_qubits} qubits")

    print("\nApplying Clifford gates:")

    # Single-qubit Cliffords
    print("  Pauli X on qubit 0")
    sim.run_gate("X", [0])

    print("  Pauli Y on qubit 1")
    sim.run_gate("Y", [1])

    print("  Pauli Z on qubit 2")
    sim.run_gate("Z", [2])

    print("  Hadamard on qubit 3")
    sim.run_gate("H", [3])

    print("  S gate on qubit 0")
    sim.run_gate("S", [0])

    print("  sqrt(X) on qubit 1")
    sim.run_gate("SX", [1])

    print("  sqrt(Y) on qubit 2")
    sim.run_gate("SY", [2])

    # Two-qubit Cliffords
    print("  CX (CNOT) on (0, 1)")
    sim.run_gate("CX", [(0, 1)])

    print("  CZ on (2, 3)")
    sim.run_gate("CZ", [(2, 3)])

    print("  SWAP on (1, 2)")
    sim.run_gate("SWAP", [(1, 2)])

    print("  sqrt(ZZ) on (0, 3)")
    sim.run_gate("SZZ", [(0, 3)])

    # Measure
    print("\nMeasuring all qubits...")
    results = sim.run_gate("Measure", [0, 1, 2, 3])
    print(f"Measurement results: {results}")


def quantum_simulator_backend_example():
    """Demonstrate using CudaStateVec as a QuantumSimulator backend."""
    print("\n" + "=" * 60)
    print("QuantumSimulator Backend Example")
    print("=" * 60)

    from pecos.simulators.quantum_simulator import QuantumSimulator

    # Create QuantumSimulator with CudaStateVec backend
    qsim = QuantumSimulator(backend="CudaStateVec", seed=42)
    qsim.init(4)
    print(f"Created QuantumSimulator with CudaStateVec backend, {qsim.num_qubits} qubits")

    # Use the run method with QOp-style operations
    print("\nThis backend can be used with HybridEngine for PHIR execution.")
    print("See the PECOS documentation for HybridEngine examples.")


def main():
    """Run all examples."""
    print("CudaStateVec (Rust cuQuantum Bindings) Examples")
    print("=" * 60)

    if not check_cuda_available():
        sys.exit(1)

    bell_state_example()
    rotation_gates_example()
    clifford_gates_example()
    quantum_simulator_backend_example()

    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
