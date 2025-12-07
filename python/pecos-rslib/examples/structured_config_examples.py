"""Examples demonstrating the structured configuration approach for PECOS QASM simulations.

This file shows how to use the Rust-native GeneralNoiseModelBuilder with fluent method chaining
to configure quantum simulations with better type safety and IDE support compared to
the legacy dictionary-based approach.
"""

from collections import Counter

from pecos_rslib import (
    biased_depolarizing_noise,
    depolarizing_noise,
    general_noise,
    sim,
)
from pecos_rslib.quantum import state_vector


def example_basic_noise_builder() -> None:
    """Example 1: Basic usage of general_noise() function."""
    print("\n=== Example 1: Direct general_noise() function ===")

    # Create a simple Bell state circuit
    qasm = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[2];
    creg c[2];
    h q[0];
    cx q[0], q[1];
    measure q -> c;
    """

    # Create and configure noise using functional API with fluent chaining
    noise = (
        general_noise()
        .with_seed(42)
        .with_p1_probability(0.001)  # Single-qubit gate error
        .with_p2_probability(0.01)  # Two-qubit gate error
        .with_meas_0_probability(0.002)  # 0->1 measurement flip
        .with_meas_1_probability(0.002)  # 1->0 measurement flip
    )

    # Use noise directly with .noise()
    results = sim(qasm).noise(noise).run(1000)

    # Analyze results
    counts = Counter(results["c"])
    print(f"Bell state measurement results: {dict(counts)}")
    print("Expected: mostly 0 (|00>) and 3 (|11>) with some errors")
    print("Note: Using functional API for maximum performance")


def example_advanced_noise_builder() -> None:
    """Example 2: Advanced general_noise() with detailed noise configuration."""
    print("\n=== Example 2: Advanced general_noise() ===")

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

    # Build complex noise model
    noise = (
        general_noise()
        # Global parameters
        .with_seed(42)
        .with_scale(1.2)  # Scale all error rates by 1.2
        .with_noiseless_gate("H")  # H gates have no noise
        # Single-qubit gate noise with Pauli distribution
        .with_average_p1_probability(0.001)  # Average error (converted to total)
        .with_p1_pauli_model(
            {
                "X": 0.5,  # 50% X errors
                "Y": 0.3,  # 30% Y errors
                "Z": 0.2,  # 20% Z errors
            },
        )
        # Two-qubit gate noise
        .with_average_p2_probability(0.008)  # Average error (converted to total)
        # Preparation and measurement noise
        .with_prep_probability(0.001)
        .with_meas_0_probability(0.002)
        .with_meas_1_probability(0.003)  # Asymmetric measurement error
    )

    results = sim(qasm).noise(noise).run(1000)
    counts = Counter(results["c"])
    print(f"GHZ-like state results: {dict(counts)}")
    print("Expected: mostly 0 (|000>) and 7 (|111>) with errors")


def example_direct_configuration() -> None:
    """Example 3: Using direct method chaining for complete simulation setup."""
    print("\n=== Example 3: Direct Method Chaining ===")

    qasm = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[4];
    creg c[4];
    h q[0];
    h q[2];
    cx q[0], q[1];
    cx q[2], q[3];
    measure q -> c;
    """

    # Create noise using functional API
    noise = general_noise().with_p1_probability(0.001).with_p2_probability(0.01)

    # Configure entire simulation with method chaining
    simulation = (
        sim(qasm)
        .seed(42)
        .auto_workers()  # Automatically use all CPU cores
        .noise(noise)
        .quantum(
            state_vector(),
        )  # Use state_vector() instead of undefined QuantumEngine
        .run(100)
    )

    results_100 = simulation
    results_1000 = (
        sim(qasm).seed(42).auto_workers().noise(noise).quantum(state_vector()).run(1000)
    )

    print("First run (100 shots):")
    print(f"  Sample results: {results_100['c'][:5]}")
    print("  Format: binary strings of length 4")

    print("Second run (1000 shots):")
    counts = Counter(results_1000["c"])
    print(f"  Most common states: {counts.most_common(4)}")


def example_builder_vs_direct() -> None:
    """Example 4: Comparing different ways to configure noise."""
    print("\n=== Example 4: Different Noise Configuration Methods ===")

    qasm = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[2];
    creg c[2];
    h q[0];
    cx q[0], q[1];
    measure q -> c;
    """

    # APPROACH 1: Using general_noise() with method chaining
    print("Using general_noise() with method chaining:")
    noise_via_builder = (
        general_noise()
        .with_p1_probability(0.001)
        .with_p2_probability(0.01)
        .with_meas_0_probability(0.002)
        .with_meas_1_probability(0.002)
        .with_noiseless_gate("H")
        .with_p1_pauli_model({"X": 0.5, "Y": 0.3, "Z": 0.2})
    )

    results_builder = (
        sim(qasm)
        .seed(42)
        .workers(4)
        .noise(noise_via_builder)
        .quantum(state_vector())
        .run(100)
    )
    print(f"  Results type: {type(results_builder['c'][0])} (integers)")

    # APPROACH 2: Using another configuration with same parameters
    print("\nUsing equivalent configuration:")
    noise_equivalent = (
        general_noise()
        .with_seed(42)
        .with_p1_probability(0.001)
        .with_p2_probability(0.01)
        .with_meas_0_probability(0.002)
        .with_meas_1_probability(0.002)
        .set_noiseless_gates(["H"])
        .with_p1_pauli_model({"X": 0.5, "Y": 0.3, "Z": 0.2})
    )

    results_equivalent = sim(qasm).seed(42).workers(4).noise(noise_equivalent).run(100)
    print(f"  Results type: {type(results_equivalent['c'][0])} (integers)")
    print(f"  Results match: {results_builder['c'] == results_equivalent['c']}")


def example_different_noise_models() -> None:
    """Example 5: Using different built-in noise models."""
    print("\n=== Example 5: Different Noise Models ===")

    qasm = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[1];
    creg c[1];
    x q[0];
    measure q[0] -> c[0];
    """

    # Test different noise models
    noise_models = [
        ("No noise", None),
        ("Depolarizing", depolarizing_noise().with_probability(0.1)),
        (
            "Custom depolarizing",
            depolarizing_noise()
            .with_prep_probability(0.01)
            .with_meas_probability(0.05)
            .with_p1_probability(0.02)
            .with_p2_probability(0.03),
        ),
        ("Biased depolarizing", biased_depolarizing_noise().with_probability(0.1)),
        (
            "General",
            general_noise().with_meas_1_probability(0.1),  # 10% chance to flip 1->0
        ),
    ]

    for name, noise in noise_models:
        results = sim(qasm).seed(42).noise(noise).run(1000)
        errors = sum(1 for val in results["c"] if val == 0)
        print(f"{name:20} - Errors: {errors}/1000 ({errors/10:.1f}%)")


def example_ion_trap_noise() -> None:
    """Example 6: Realistic ion trap noise model."""
    print("\n=== Example 6: Ion Trap Noise Model ===")

    qasm = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[5];
    creg c[5];

    // Create W state
    x q[0];
    h q[1];
    cx q[1], q[2];
    cx q[2], q[3];
    cx q[3], q[4];
    cx q[0], q[1];
    h q[1];

    measure q -> c;
    """

    # Realistic ion trap noise parameters
    noise = (
        general_noise()
        .with_seed(42)
        # Ion trap typical parameters
        .with_prep_probability(0.001)  # State prep error
        # Single-qubit gates (typically very good)
        .with_p1_probability(0.0001)
        # Two-qubit gates (main error source)
        .with_p2_probability(0.003)
        # Measurement (asymmetric for ions)
        .with_meas_0_probability(0.001)  # Dark state error
        .with_meas_1_probability(0.005)  # Bright state error
    )

    results = sim(qasm).noise(noise).run(1000)
    counts = Counter(results["c"])
    print("W-state preparation results (top 5):")
    for state, count in counts.most_common(5):
        binary = format(state, "05b")
        print(f"  |{binary}> : {count}")


def main() -> None:
    """Run all examples."""
    print("PECOS Structured Configuration Examples")
    print("=" * 50)

    example_basic_noise_builder()
    example_advanced_noise_builder()
    example_direct_configuration()
    example_builder_vs_direct()
    example_different_noise_models()
    example_ion_trap_noise()

    print("\n" + "=" * 50)
    print("Examples completed!")


if __name__ == "__main__":
    main()
