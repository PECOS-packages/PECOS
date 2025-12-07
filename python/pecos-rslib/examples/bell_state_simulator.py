#!/usr/bin/env python3
# Copyright 2025 The PECOS Developers
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

"""Example of running a Bell state experiment using the StateVecEngine."""

import collections
import os
import sys

# Add the parent directory to the path to import pecos_rslib
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pecos_rslib import ByteMessage, StateVecEngine


def run_bell_state_experiment() -> None:
    """Run a Bell state experiment using the StateVecEngine."""
    print("==== Bell State Experiment with Simulator ====")

    # Create a Bell state circuit
    builder = ByteMessage.quantum_operations_builder()

    # Add gates to create a Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2
    print("Building Bell state circuit...")
    builder.add_h(0)  # Hadamard on qubit 0
    builder.add_cx(0, 1)  # CNOT with control=0, target=1
    builder.add_measurement(0, 0)  # Measure qubit 0
    builder.add_measurement(1, 1)  # Measure qubit 1

    bell_circuit = builder.build()
    print("Circuit built successfully")

    # Create a simulator with 2 qubits
    simulator = StateVecEngine(2)
    print("Created state vector simulator with 2 qubits")

    # Run the circuit once and check results
    print("\nRunning circuit once...")
    simulator.reset()
    result_message = simulator.process(bell_circuit)
    results = result_message.measurement_results()

    print("Measurement results:")
    for result_id, outcome in results:
        print(f"  Qubit {result_id}: {outcome}")

    # Run the circuit multiple times to verify Bell state correlations
    num_shots = 100
    print(f"\nRunning circuit for {num_shots} shots...")
    all_results = simulator.run_circuit_with_shots(bell_circuit, num_shots)

    # Analyze the results for correlation
    correlations = {"00": 0, "01": 0, "10": 0, "11": 0}

    for shot_results in all_results:
        q0_result = None
        q1_result = None

        for result_id, outcome in shot_results:
            if result_id == 0:
                q0_result = outcome
            elif result_id == 1:
                q1_result = outcome

        # Form a key like "00", "01", "10", or "11"
        if q0_result is not None and q1_result is not None:
            key = f"{q0_result}{q1_result}"
            correlations[key] += 1

    print("\nCorrelation statistics:")
    for outcome, count in correlations.items():
        percentage = (count / num_shots) * 100
        print(f"  {outcome}: {count} times ({percentage:.1f}%)")

    # Check if we have the expected correlations for a Bell state
    # In a Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2, we expect only 00 and 11 outcomes
    correlated_outcomes = correlations["00"] + correlations["11"]
    anticorrelated_outcomes = correlations["01"] + correlations["10"]

    print("\nCorrelation analysis:")
    print(
        f"  Correlated outcomes (00 or 11): {correlated_outcomes} ({correlated_outcomes / num_shots * 100:.1f}%)",
    )
    print(
        f"  Anti-correlated outcomes (01 or 10): {anticorrelated_outcomes} ({anticorrelated_outcomes / num_shots * 100:.1f}%)",
    )

    if correlated_outcomes > 0.95 * num_shots:
        print(
            "\nSuccess! The qubits are highly correlated, as expected in a Bell state.",
        )
    elif anticorrelated_outcomes > 0.95 * num_shots:
        print(
            "\nInteresting! The qubits are anti-correlated, which is another valid Bell state.",
        )
    else:
        print(
            "\nUnexpected result: The qubits don't show the strong correlation expected in a Bell state.",
        )

    print("\n==== End of Bell State Experiment ====")


def run_custom_experiment() -> None:
    """Run a custom quantum experiment using the StateVecEngine."""
    print("\n==== Custom Quantum Experiment ====")

    # Create a simulator with 3 qubits
    simulator = StateVecEngine(3)
    print("Created state vector simulator with 3 qubits")

    # Create a GHZ state circuit: |GHZ⟩ = (|000⟩ + |111⟩)/√2
    builder = ByteMessage.quantum_operations_builder()

    print("Building GHZ state circuit...")
    # Apply H to qubit 0
    builder.add_h(0)

    # Apply CNOT from 0 to 1
    builder.add_cx(0, 1)

    # Apply CNOT from 1 to 2
    builder.add_cx(1, 2)

    # Measure all qubits
    builder.add_measurement(0, 0)
    builder.add_measurement(1, 1)
    builder.add_measurement(2, 2)

    ghz_circuit = builder.build()

    # Run the circuit multiple times
    num_shots = 100
    print(f"Running GHZ circuit for {num_shots} shots...")
    all_results = simulator.run_circuit_with_shots(ghz_circuit, num_shots)

    # Analyze the results
    outcome_counts = collections.Counter()

    for shot_results in all_results:
        results_dict = {result_id: outcome for result_id, outcome in shot_results}
        outcome_str = "".join(str(results_dict.get(i, "X")) for i in range(3))
        outcome_counts[outcome_str] += 1

    print("\nOutcome statistics:")
    for outcome, count in sorted(outcome_counts.items()):
        percentage = (count / num_shots) * 100
        print(f"  |{outcome}⟩: {count} times ({percentage:.1f}%)")

    # Check if we have the expected correlations for a GHZ state
    # In a GHZ state (|000⟩ + |111⟩)/√2, we expect only 000 and 111 outcomes
    expected_outcomes = outcome_counts.get("000", 0) + outcome_counts.get("111", 0)
    unexpected_outcomes = num_shots - expected_outcomes

    print("\nGHZ state analysis:")
    print(
        f"  Expected outcomes (000 or 111): {expected_outcomes} ({expected_outcomes / num_shots * 100:.1f}%)",
    )
    print(
        f"  Unexpected outcomes: {unexpected_outcomes} ({unexpected_outcomes / num_shots * 100:.1f}%)",
    )

    print("\n==== End of Custom Experiment ====")


if __name__ == "__main__":
    run_bell_state_experiment()
    run_custom_experiment()
