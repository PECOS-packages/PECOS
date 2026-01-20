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

"""Example of using CudaStabilizer for GPU-accelerated stabilizer simulation.

CudaStabilizer is designed for:
- Large-scale Clifford circuit simulation (1000s of qubits)
- Quantum error correction simulations
- Stabilizer state preparation and measurement

Note: CudaStabilizer only supports Clifford gates (no T gates or arbitrary rotations).

Requirements:
- NVIDIA GPU with CUDA support
- cuQuantum SDK installed
- pecos-rslib-cuda package
"""

import sys
import time


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


def basic_stabilizer_example():
    """Demonstrate basic stabilizer operations."""
    print("=" * 60)
    print("Basic Stabilizer Example")
    print("=" * 60)

    from pecos.simulators import CudaStabilizer

    # Create a stabilizer simulator - can handle many more qubits than state vector!
    sim = CudaStabilizer(10, seed=42)
    print(f"Created CudaStabilizer with {sim.num_qubits} qubits")

    # Apply Clifford gates
    print("\nApplying Clifford gates:")

    print("  Hadamard on qubits 0-4")
    sim.run_gate("H", [0, 1, 2, 3, 4])

    print("  CNOT chain: 0->1->2->3->4")
    sim.run_gate("CX", [(0, 1)])
    sim.run_gate("CX", [(1, 2)])
    sim.run_gate("CX", [(2, 3)])
    sim.run_gate("CX", [(3, 4)])

    print("  S gates on qubits 5-9")
    sim.run_gate("S", [5, 6, 7, 8, 9])

    print("  CZ gates")
    sim.run_gate("CZ", [(5, 6), (7, 8)])

    # Measure
    print("\nMeasuring qubits 0-4...")
    results = sim.run_gate("Measure", [0, 1, 2, 3, 4])
    print(f"Measurement results: {results}")


def large_scale_example():
    """Demonstrate large-scale stabilizer simulation."""
    print("\n" + "=" * 60)
    print("Large-Scale Stabilizer Simulation")
    print("=" * 60)

    from pecos.simulators import CudaStabilizer

    # Create a large stabilizer simulator
    num_qubits = 500
    sim = CudaStabilizer(num_qubits)
    print(f"Created CudaStabilizer with {sim.num_qubits} qubits")

    # Time a series of operations
    print(f"\nApplying gates to {num_qubits} qubits...")

    start = time.perf_counter()

    # Apply Hadamard to all qubits
    for i in range(0, num_qubits, 50):
        batch = list(range(i, min(i + 50, num_qubits)))
        sim.run_gate("H", batch)

    # Apply CNOT chain
    for i in range(num_qubits - 1):
        sim.run_gate("CX", [(i, i + 1)])

    elapsed = time.perf_counter() - start
    print(f"Gate application took {elapsed:.3f} seconds")

    # Measure a subset
    print("\nMeasuring first 10 qubits...")
    results = sim.run_gate("Measure", list(range(10)))
    print(f"Measurement results: {results}")


def ghz_state_example():
    """Create and verify a GHZ state."""
    print("\n" + "=" * 60)
    print("GHZ State Example")
    print("=" * 60)

    from pecos.simulators import CudaStabilizer

    num_qubits = 20
    sim = CudaStabilizer(num_qubits, seed=123)
    print(f"Creating {num_qubits}-qubit GHZ state...")

    # Create GHZ state: H on first qubit, then CNOT chain
    sim.run_gate("H", [0])
    for i in range(num_qubits - 1):
        sim.run_gate("CX", [(i, i + 1)])

    print("GHZ state created: |0...0> + |1...1>")

    # Measure all qubits - should get all 0s or all 1s
    print("\nMeasuring GHZ state (should be all 0s or all 1s)...")
    results = sim.run_gate("Measure", list(range(num_qubits)))

    values = list(results.values())
    if all(v == values[0] for v in values):
        print(f"SUCCESS: All qubits measured as {values[0]}")
    else:
        print(f"Results: {values}")


def surface_code_syndrome_example():
    """Demonstrate a simple surface code syndrome extraction."""
    print("\n" + "=" * 60)
    print("Surface Code Syndrome Extraction Example")
    print("=" * 60)

    from pecos.simulators import CudaStabilizer

    # Simple 3x3 surface code layout (9 data qubits + 8 ancillas)
    # Data qubits: 0-8, Ancilla qubits: 9-16
    num_qubits = 17
    sim = CudaStabilizer(num_qubits)
    print(f"Created simulator for surface code with {num_qubits} qubits")

    # Initialize data qubits to |0>
    print("\nInitializing data qubits to |0>...")
    sim.run_gate("Init +Z", list(range(9)))

    # Simple X-stabilizer measurement (one plaquette)
    # Ancilla 9 checks data qubits 0, 1, 3, 4
    print("\nMeasuring X-stabilizer (plaquette 0,1,3,4):")
    ancilla = 9
    data_qubits = [0, 1, 3, 4]

    print(f"  H on ancilla {ancilla}")
    sim.run_gate("H", [ancilla])

    for d in data_qubits:
        print(f"  CX({ancilla}, {d})")
        sim.run_gate("CX", [(ancilla, d)])

    print(f"  H on ancilla {ancilla}")
    sim.run_gate("H", [ancilla])

    print(f"  Measure ancilla {ancilla}")
    result = sim.run_gate("Measure", [ancilla])
    print(f"  Syndrome: {result}")

    # For |0> data state, X-stabilizer should give +1 (measurement = 0)
    if ancilla in result:
        if result[ancilla] == 0:
            print("  No error detected (expected for |0> state)")
        else:
            print("  Error detected!")


def non_clifford_error_example():
    """Demonstrate that non-Clifford gates raise an error."""
    print("\n" + "=" * 60)
    print("Non-Clifford Gate Error Example")
    print("=" * 60)

    from pecos.simulators import CudaStabilizer

    sim = CudaStabilizer(2)
    print("CudaStabilizer only supports Clifford gates.")
    print("Attempting to apply T gate (non-Clifford)...")

    try:
        sim.run_gate("T", [0])
        print("ERROR: T gate should have raised an exception!")
    except ValueError as e:
        print(f"Correctly raised ValueError: {e}")
        print("\nFor non-Clifford gates, use CudaStateVec instead.")


def main():
    """Run all examples."""
    print("CudaStabilizer (Rust cuQuantum Bindings) Examples")
    print("=" * 60)

    if not check_cuda_available():
        sys.exit(1)

    basic_stabilizer_example()
    large_scale_example()
    ghz_state_example()
    surface_code_syndrome_example()
    non_clifford_error_example()

    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
