#!/usr/bin/env python3
"""Noise Model Builder Examples.

This example demonstrates how to use GeneralNoiseModelBuilder to create
various quantum noise models for QASM simulations.
"""

from collections import Counter

from pecos.rslib import GeneralNoiseModelBuilder, qasm_engine
from pecos.rslib.programs import QasmProgram


def simple_noise_example() -> None:
    """Basic noise model with uniform error probabilities."""
    print("\n=== Simple Noise Model ===")

    # Bell state circuit
    qasm = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[2];
    creg c[2];
    h q[0];
    cx q[0], q[1];
    measure q -> c;
    """

    # Simple uniform noise
    noise = GeneralNoiseModelBuilder().with_seed(42).with_p1_probability(0.001).with_p2_probability(0.01)

    results = qasm_engine().program(QasmProgram.from_string(qasm)).to_sim().noise(noise).run(1000)
    results_dict = results.to_dict()
    counts = Counter(results_dict["c"])

    print(f"Bell state results: {dict(counts)}")
    print("Expected: Mostly 0 (|00>) and 3 (|11>) with small error rates")


def hardware_realistic_noise() -> None:
    """Noise model based on typical superconducting qubit parameters."""
    print("\n=== Hardware-Realistic Noise ===")

    # GHZ state circuit
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

    # Realistic superconducting qubit noise
    noise = (
        GeneralNoiseModelBuilder()
        .with_seed(42)
        # Gate errors (two-qubit much worse)
        .with_average_p1_probability(0.0001)  # 0.01%
        .with_average_p2_probability(0.001)  # 0.1%
        # Measurement is often the dominant error
        .with_prep_probability(0.001)
        .with_meas_0_probability(0.01)  # 1% false positive
        .with_meas_1_probability(0.005)
    )  # 0.5% false negative

    results = qasm_engine().program(QasmProgram.from_string(qasm)).to_sim().noise(noise).run(1000)
    results_dict = results.to_dict()
    counts = Counter(results_dict["c"])

    print("GHZ state results (top 5):")
    for state, count in counts.most_common(5):
        binary = format(state, "03b")
        print(f"  |{binary}>: {count}")


def biased_noise_example() -> None:
    """Noise model with biased Pauli errors (more dephasing than bit flips)."""
    print("\n=== Biased Noise Model ===")

    # Simple circuit to show phase vs bit errors
    qasm = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[1];
    creg c[1];
    h q[0];
    t q[0];
    t q[0];
    t q[0];
    t q[0];  // Multiple T gates accumulate phase errors
    h q[0];
    measure q[0] -> c[0];
    """

    # Biased noise (more Z errors)
    noise = (
        GeneralNoiseModelBuilder()
        .with_seed(42)
        .with_average_p1_probability(0.01)  # Higher error for visibility
        .with_p1_pauli_model(
            {
                "X": 0.1,  # 10% bit flips
                "Y": 0.1,  # 10% Y errors
                "Z": 0.8,  # 80% phase errors (dominant)
            },
        )
    )

    results = qasm_engine().program(QasmProgram.from_string(qasm)).to_sim().noise(noise).run(1000)
    results_dict = results.to_dict()
    errors = sum(1 for val in results_dict["c"] if val == 1)

    print(f"Circuit should measure |0>, but got {errors} errors out of 1000")
    print("With biased noise (80% Z errors), phase errors accumulate")


def ion_trap_noise() -> None:
    """Noise model for ion trap quantum computers."""
    print("\n=== Ion Trap Noise Model ===")

    qasm = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[2];
    creg c[2];
    h q[0];
    cx q[0], q[1];
    measure q -> c;
    """

    # Ion trap characteristics
    noise = (
        GeneralNoiseModelBuilder()
        .with_seed(42)
        # Excellent single-qubit gates
        .with_average_p1_probability(0.00001)  # 0.001% error
        # Two-qubit gates are limiting factor
        .with_average_p2_probability(0.003)  # 0.3% error
        # Asymmetric measurement
        .with_meas_0_probability(0.001)  # Dark state error
        .with_meas_1_probability(0.005)
    )  # Bright state error

    results = qasm_engine().program(QasmProgram.from_string(qasm)).to_sim().noise(noise).run(1000)
    results_dict = results.to_dict()
    counts = Counter(results_dict["c"])

    print(f"Ion trap Bell state: {dict(counts)}")
    print("Note: Two-qubit gate errors dominate in ion traps")


def noiseless_gates_example() -> None:
    """Demonstrate making specific gates noiseless."""
    print("\n=== Noiseless Gates Example ===")

    qasm = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[2];
    creg c[2];
    h q[0];      // This will be noiseless
    x q[0];      // This will have noise
    cx q[0], q[1];
    measure q -> c;
    """

    # Make H gates perfect (e.g., if they're virtual gates)
    noise = (
        GeneralNoiseModelBuilder()
        .with_seed(42)
        .with_p1_probability(0.01)  # High error for visibility
        .with_p2_probability(0.01)
        .with_noiseless_gate("H")
    )  # H gates have no error

    results = qasm_engine().program(QasmProgram.from_string(qasm)).to_sim().noise(noise).run(1000)
    results_dict = results.to_dict()
    counts = Counter(results_dict["c"])

    print(f"Results with noiseless H: {dict(counts)}")
    print("H gate is perfect, but X and CX gates have 1% error rate")


def scaled_noise_example() -> None:
    """Use scaling to easily adjust overall noise levels."""
    print("\n=== Scaled Noise Example ===")

    qasm = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[2];
    creg c[2];
    h q[0];
    cx q[0], q[1];
    measure q -> c;
    """

    # Base noise model
    base_noise = (
        GeneralNoiseModelBuilder()
        .with_seed(42)
        .with_p1_probability(0.001)
        .with_p2_probability(0.01)
        .with_meas_0_probability(0.002)
        .with_meas_1_probability(0.002)
    )

    # Same model scaled up 3x
    scaled_noise = (
        GeneralNoiseModelBuilder()
        .with_seed(42)
        .with_p1_probability(0.001)
        .with_p2_probability(0.01)
        .with_meas_0_probability(0.002)
        .with_meas_1_probability(0.002)
        .with_scale(3.0)
    )  # Triple all error rates!

    # Run both
    results_base = qasm_engine().program(QasmProgram.from_string(qasm)).to_sim().noise(base_noise).run(1000)
    results_scaled = qasm_engine().program(QasmProgram.from_string(qasm)).to_sim().noise(scaled_noise).run(1000)

    # Count errors (anything not 0 or 3)
    results_base_dict = results_base.to_dict()
    results_scaled_dict = results_scaled.to_dict()
    errors_base = sum(1 for val in results_base_dict["c"] if val not in [0, 3])
    errors_scaled = sum(1 for val in results_scaled_dict["c"] if val not in [0, 3])

    print(f"Base noise errors: {errors_base}/1000")
    print(f"3x scaled noise errors: {errors_scaled}/1000")
    print("Scaling makes it easy to test noise sensitivity")


def comprehensive_example() -> None:
    """Complete example using many builder features."""
    print("\n=== Comprehensive Noise Model ===")

    # 4-qubit GHZ circuit
    qasm = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[4];
    creg c[4];
    h q[0];
    cx q[0], q[1];
    cx q[1], q[2];
    cx q[2], q[3];
    barrier q;
    measure q -> c;
    """

    # Kitchen sink noise model
    noise = (
        GeneralNoiseModelBuilder()
        # Reproducibility
        .with_seed(42)
        # Global scaling
        .with_scale(1.2)  # 20% worse than nominal
        # Make Hadamard noiseless
        .with_noiseless_gate("h")
        # State preparation
        .with_prep_probability(0.001)
        # Single-qubit with custom Pauli
        .with_average_p1_probability(0.0001)
        .with_p1_pauli_model(
            {
                "X": 0.2,
                "Y": 0.2,
                "Z": 0.6,  # More dephasing
            },
        )
        # Two-qubit gates
        .with_average_p2_probability(0.001)
        # Measurement errors
        .with_meas_0_probability(0.002)
        .with_meas_1_probability(0.005)
    )

    results = qasm_engine().program(QasmProgram.from_string(qasm)).to_sim().noise(noise).run(1000)
    results_dict = results.to_dict()
    counts = Counter(results_dict["c"])

    print("4-qubit GHZ results (top 8):")
    for state, count in counts.most_common(8):
        binary = format(state, "04b")
        print(f"  |{binary}>: {count}")

    print("\nFeatures demonstrated:")
    print("- Seed for reproducibility")
    print("- Global scaling factor")
    print("- Noiseless H gates")
    print("- Custom Pauli distributions")
    print("- Asymmetric measurement errors")


def main() -> None:
    """Run all examples."""
    print("GeneralNoiseModelBuilder Examples")
    print("=" * 50)

    simple_noise_example()
    hardware_realistic_noise()
    biased_noise_example()
    ion_trap_noise()
    noiseless_gates_example()
    scaled_noise_example()
    comprehensive_example()

    print("\n" + "=" * 50)
    print("Examples completed!")
    print(
        "\nFor more details, see the Noise Model Builders guide in the documentation.",
    )


if __name__ == "__main__":
    main()
