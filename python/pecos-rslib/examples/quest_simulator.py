#!/usr/bin/env python3
"""Test script for QuEST simulators exposed to Python via pecos-rslib"""

import math

from _pecos_rslib import QuestDensityMatrix, QuestStateVec


def test_quest_statevec() -> None:
    """Test the QuEST state vector simulator"""
    print("Testing QuEST State Vector Simulator")
    print("=" * 40)

    # Create a 2-qubit state vector simulator
    sim = QuestStateVec(2)
    print(f"Created simulator: {sim}")
    print(f"Number of qubits: {sim.num_qubits()}")

    # Test initial state |00⟩
    print("\nInitial state |00⟩:")
    prob00 = sim.probability(0b00)
    print(f"  Probability of |00⟩: {prob00:.4f}")
    amp00 = sim.get_amplitude(0b00)
    print(f"  Amplitude of |00⟩: {amp00[0]:.4f} + {amp00[1]:.4f}i")

    # Apply Hadamard to qubit 0
    print("\nApplying H(0)...")
    sim.run_1q_gate("H", 0)

    # Check probabilities after H
    print("After H(0):")
    for i in range(4):
        prob = sim.probability(i)
        state = f"|{i:02b}⟩"
        print(f"  Probability of {state}: {prob:.4f}")

    # Apply CNOT(0, 1) to create Bell state
    print("\nApplying CNOT(0, 1)...")
    sim.run_2q_gate("CX", (0, 1))

    # Check Bell state
    print("Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2:")
    for i in range(4):
        prob = sim.probability(i)
        amp = sim.get_amplitude(i)
        state = f"|{i:02b}⟩"
        print(f"  {state}: prob={prob:.4f}, amp=({amp[0]:.4f}, {amp[1]:.4f})")

    # Test measurement
    print("\nPerforming measurements:")
    for _ in range(5):
        sim.reset()
        sim.run_1q_gate("H", 0)
        sim.run_2q_gate("CX", (0, 1))

        result0 = sim.run_1q_gate("MZ", 0)
        result1 = sim.run_1q_gate("MZ", 1)
        print(f"  Measured: qubit 0 = {result0}, qubit 1 = {result1}")

    # Test rotation gates
    print("\nTesting rotation gates:")
    sim.reset()
    sim.run_1q_gate("RX", 0, {"angle": math.pi / 4})
    prob0 = sim.probability(0)
    prob1 = sim.probability(1)
    print(f"  After RX(π/4) on |0⟩: P(|0⟩)={prob0:.4f}, P(|1⟩)={prob1:.4f}")

    sim.reset()
    sim.run_1q_gate("RY", 0, {"angle": math.pi / 2})
    amp0 = sim.get_amplitude(0)
    amp1 = sim.get_amplitude(1)
    print("  After RY(π/2) on |0⟩:")
    print(f"    |0⟩ amplitude: ({amp0[0]:.4f}, {amp0[1]:.4f})")
    print(f"    |1⟩ amplitude: ({amp1[0]:.4f}, {amp1[1]:.4f})")


def test_quest_density_matrix() -> None:
    """Test the QuEST density matrix simulator"""
    print("\n\nTesting QuEST Density Matrix Simulator")
    print("=" * 40)

    # Create a 2-qubit density matrix simulator
    sim = QuestDensityMatrix(2)
    print(f"Created simulator: {sim}")
    print(f"Number of qubits: {sim.num_qubits()}")

    # Test initial state |00⟩⟨00|
    print("\nInitial state |00⟩⟨00|:")
    prob00 = sim.probability(0b00)
    print(f"  Probability of |00⟩: {prob00:.4f}")

    # Apply gates to create mixed state
    print("\nApplying H(0) and X(1)...")
    sim.run_1q_gate("H", 0)
    sim.run_1q_gate("X", 1)

    # Check probabilities
    print("After H(0) and X(1):")
    for i in range(4):
        prob = sim.probability(i)
        state = f"|{i:02b}⟩"
        print(f"  Probability of {state}: {prob:.4f}")

    # Test two-qubit gates
    print("\nResetting and creating entangled state...")
    sim.reset()
    sim.run_1q_gate("H", 0)
    sim.run_2q_gate("CX", (0, 1))

    print("After H(0) and CNOT(0,1):")
    for i in range(4):
        prob = sim.probability(i)
        state = f"|{i:02b}⟩"
        print(f"  Probability of {state}: {prob:.4f}")

    # Test measurement
    print("\nPerforming measurement on qubit 0:")
    result = sim.run_1q_gate("MZ", 0)
    print(f"  Measured: {result}")

    print("\nState after measurement:")
    for i in range(4):
        prob = sim.probability(i)
        state = f"|{i:02b}⟩"
        print(f"  Probability of {state}: {prob:.4f}")


if __name__ == "__main__":
    test_quest_statevec()
    test_quest_density_matrix()
    print("\nAll tests completed successfully!")
