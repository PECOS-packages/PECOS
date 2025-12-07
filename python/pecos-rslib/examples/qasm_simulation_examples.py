#!/usr/bin/env python3
"""QASM Simulation API Example

This example demonstrates the PECOS QASM simulation API with various
noise models and quantum engines.
"""

import time
from collections import Counter

from pecos_rslib import (
    biased_depolarizing_noise,
    depolarizing_noise,
    qasm_engine,
    sparse_stabilizer,
    state_vector,
)
from pecos_rslib.programs import QasmProgram


def example_bell_state() -> None:
    """Example: Create and measure a Bell state."""
    print("\n=== Bell State Example ===")

    qasm = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[2];
    creg c[2];
    h q[0];
    cx q[0], q[1];
    measure q -> c;
    """

    # Run without noise
    results = qasm_engine().program(QasmProgram.from_string(qasm)).to_sim().run(1000)
    results_dict = results.to_dict()
    counts = Counter(results_dict["c"])

    print("Bell state measurements (no noise):")
    for outcome, count in sorted(counts.items()):
        print(f"  |{outcome:02b}⟩: {count} times")

    # Run with depolarizing noise
    noise = (
        depolarizing_noise()
        .with_prep_probability(0.001)
        .with_meas_probability(0.002)
        .with_p1_probability(0.02)
        .with_p2_probability(0.02)
    )
    results_noisy = (
        qasm_engine()
        .program(QasmProgram.from_string(qasm))
        .to_sim()
        .seed(42)
        .noise(noise)
        .run(1000)
    )
    results_noisy_dict = results_noisy.to_dict()
    counts_noisy = Counter(results_noisy_dict["c"])

    print("\nBell state measurements (2% depolarizing noise):")
    for outcome, count in sorted(counts_noisy.items()):
        print(f"  |{outcome:02b}⟩: {count} times")


def example_ghz_state() -> None:
    """Example: Create and measure a 3-qubit GHZ state."""
    print("\n=== GHZ State Example ===")

    qasm = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[3];
    creg c[3];
    h q[0];
    cx q[0], q[1];
    cx q[1], q[2];
    measure q -> c;
    """

    # Run with custom depolarizing noise
    noise = (
        depolarizing_noise()
        .with_prep_probability(0.001)  # Low preparation error
        .with_meas_probability(0.005)  # Moderate measurement error
        .with_p1_probability(0.001)  # Low single-qubit gate error
        .with_p2_probability(0.01)
    )  # Higher two-qubit gate error

    # Different ways to specify quantum engine:
    # 1. Using builder function (recommended)
    #    .quantum_engine(sparse_stabilizer())
    # 2. Using builder class
    #    .quantum_engine(SparseStabilizerBuilder())
    # 3. Using string (backward compatibility)
    #    .quantum_engine("sparsestabilizer")

    results = (
        qasm_engine()
        .program(QasmProgram.from_string(qasm))
        .to_sim()
        .seed(42)
        .noise(noise)
        .quantum_engine(sparse_stabilizer())
        .run(1000)
    )

    results_dict = results.to_dict()
    counts = Counter(results_dict["c"])
    print("GHZ state measurements (custom noise):")
    for outcome, count in sorted(counts.items()):
        print(f"  |{outcome:03b}⟩: {count} times")


def example_biased_depolarizing() -> None:
    """Example: Demonstrate biased depolarizing noise."""
    print("\n=== Biased Depolarizing Example ===")

    qasm = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[2];
    creg c[2];
    x q[0];
    x q[1];
    measure q -> c;
    """

    # Perfect measurements
    results_ideal = (
        qasm_engine().program(QasmProgram.from_string(qasm)).to_sim().run(1000)
    )
    results_ideal_dict = results_ideal.to_dict()
    ideal_counts = Counter(results_ideal_dict["c"])

    # Biased depolarizing noise
    noise = (
        biased_depolarizing_noise()
        .with_prep_probability(0.1)
        .with_meas_0_probability(0.1)
        .with_meas_1_probability(0.1)
        .with_p1_probability(0.1)
        .with_p2_probability(0.1)
    )

    results_biased = (
        qasm_engine()
        .program(QasmProgram.from_string(qasm))
        .to_sim()
        .seed(42)
        .noise(noise)
        .run(1000)
    )
    results_biased_dict = results_biased.to_dict()
    biased_counts = Counter(results_biased_dict["c"])

    print("Preparing |11⟩ state:")
    print(f"  Ideal: {ideal_counts}")
    print(f"  Biased depolarizing: {biased_counts}")
    print("  (Notice the errors introduced by biased depolarizing noise)")


def example_quantum_engines() -> None:
    """Example: Compare different quantum engines."""
    print("\n=== Quantum Engine Comparison ===")

    # Circuit with non-Clifford gates
    qasm = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[2];
    creg c[2];
    h q[0];
    rz(0.5) q[0];
    cx q[0], q[1];
    measure q -> c;
    """

    # State vector engine (can handle arbitrary gates)
    try:
        results_sv = (
            qasm_engine()
            .program(QasmProgram.from_string(qasm))
            .to_sim()
            .seed(42)
            .quantum_engine(state_vector())
            .run(100)
        )
        sv_dict = results_sv.to_dict()
        sv_counts = Counter(sv_dict["c"])
        print(f"StateVector engine: {dict(sv_counts)}")
    except (ValueError, RuntimeError, KeyError) as e:
        print(f"StateVector engine error: {e}")

    # Sparse stabilizer engine (efficient for Clifford circuits)
    # This will fail for non-Clifford gates like rz(0.5)
    try:
        results_stab = (
            qasm_engine()
            .program(QasmProgram.from_string(qasm))
            .to_sim()
            .seed(42)
            .quantum_engine(sparse_stabilizer())
            .run(100)
        )
        stab_dict = results_stab.to_dict()
        stab_counts = Counter(stab_dict["c"])
        print(f"SparseStabilizer engine: {dict(stab_counts)}")
    except (ValueError, RuntimeError):
        print(
            "SparseStabilizer engine error: Expected - cannot handle non-Clifford gates",
        )


def example_builder_pattern() -> None:
    """Example: Using the builder pattern for reusable simulations."""
    print("\n=== Builder Pattern Example ===")

    qasm = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[2];
    creg c[2];
    h q[0];
    cx q[0], q[1];
    measure q -> c;
    """

    # Build once, run multiple times with different shot counts
    noise = (
        depolarizing_noise()
        .with_prep_probability(0.01)
        .with_meas_probability(0.01)
        .with_p1_probability(0.01)
        .with_p2_probability(0.01)
    )

    sim = (
        qasm_engine()
        .program(QasmProgram.from_string(qasm))
        .to_sim()
        .seed(42)
        .noise(noise)
        .quantum_engine(sparse_stabilizer())
        .workers(4)
        .build()
    )

    print("Running same circuit with different shot counts:")
    for shots in [10, 100, 1000]:
        results = sim.run(shots)
        results_dict = results.to_dict()
        counts = Counter(results_dict["c"])
        print(f"  {shots} shots: {dict(counts)}")

    # Or run directly without building
    noise_biased = (
        biased_depolarizing_noise()
        .with_prep_probability(0.005)
        .with_meas_0_probability(0.005)
        .with_meas_1_probability(0.005)
        .with_p1_probability(0.005)
        .with_p2_probability(0.005)
    )

    results = (
        qasm_engine()
        .program(QasmProgram.from_string(qasm))
        .to_sim()
        .noise(noise_biased)
        .run(500)
    )

    results_dict = results.to_dict()
    counts = Counter(results_dict["c"])
    print(f"\nDirect run with biased depolarizing noise: {dict(counts)}")


def example_large_register() -> None:
    """Example: Handling large quantum registers (>64 qubits)."""
    print("\n=== Large Register Example ===")

    # Create a circuit with 70 qubits
    qasm = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[70];
    creg c[70];

    // Create a pattern
    x q[0];
    x q[10];
    x q[20];
    x q[30];
    x q[40];
    x q[50];
    x q[60];
    x q[69];

    measure q -> c;
    """

    results = qasm_engine().program(QasmProgram.from_string(qasm)).to_sim().run(10)
    results_dict = results.to_dict()

    print("Large register measurements (70 qubits):")
    for i, value in enumerate(results_dict["c"][:5]):  # Show first 5
        # Convert to binary string for large values
        binary = bin(value)[2:].zfill(70)
        set_bits = [i for i, bit in enumerate(reversed(binary)) if bit == "1"]
        print(f"  Shot {i}: bits {set_bits} are set")
    print(f"  ... ({len(results_dict['c'])} total shots)")


def example_parallel_execution() -> None:
    """Example: Parallel execution with multiple workers."""
    print("\n=== Parallel Execution Example ===")

    qasm = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[5];
    creg c[5];

    // Random circuit
    h q[0];
    h q[1];
    h q[2];
    cx q[0], q[3];
    cx q[1], q[4];
    cx q[2], q[3];
    h q[3];
    h q[4];

    measure q -> c;
    """

    noise = (
        depolarizing_noise()
        .with_prep_probability(0.001)
        .with_meas_probability(0.001)
        .with_p1_probability(0.001)
        .with_p2_probability(0.001)
    )

    # Single worker
    start = time.time()
    (
        qasm_engine()
        .program(QasmProgram.from_string(qasm))
        .to_sim()
        .seed(42)
        .noise(noise)
        .workers(1)
        .run(10000)
    )
    single_time = time.time() - start

    # Multiple workers
    start = time.time()
    (
        qasm_engine()
        .program(QasmProgram.from_string(qasm))
        .to_sim()
        .seed(42)
        .noise(noise)
        .workers(4)
        .run(10000)
    )
    parallel_time = time.time() - start

    print("Execution time comparison (10,000 shots):")
    print(f"  Single worker: {single_time:.3f}s")
    print(f"  4 workers: {parallel_time:.3f}s")
    print(f"  Speedup: {single_time/parallel_time:.2f}x")

    # Note: Results may differ slightly between single and multi-worker runs
    # due to different random number generation patterns


if __name__ == "__main__":
    print("PECOS QASM Simulation API Examples")
    print("==================================")

    example_bell_state()
    example_ghz_state()
    example_biased_depolarizing()
    example_quantum_engines()
    example_builder_pattern()
    example_large_register()
    example_parallel_execution()

    print("\nAll examples completed successfully!")
