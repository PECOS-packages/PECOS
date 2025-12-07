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

"""Tests for the StateVecEngine Python bindings."""

from pecos_rslib import ByteMessage, StateVecEngine


def test_simulator_creation() -> None:
    """Test creating a StateVecEngine."""
    simulator = StateVecEngine(2)
    assert simulator is not None


def test_bell_state_correlations() -> None:
    """Test that measurements in a Bell state are correlated."""
    # Create a Bell state circuit
    builder = ByteMessage.quantum_operations_builder()
    builder.add_h(0)
    builder.add_cx(0, 1)
    builder.add_measurement(0, 0)
    builder.add_measurement(1, 1)
    bell_circuit = builder.build()

    # Create a simulator with 2 qubits
    simulator = StateVecEngine(2)

    # Run the circuit multiple times
    num_shots = 50
    all_results = simulator.run_circuit_with_shots(bell_circuit, num_shots)

    # Check that we have the expected number of results
    assert len(all_results) == num_shots

    # Analyze the results for correlation
    non_correlated_count = 0

    for shot_results in all_results:
        assert len(shot_results) == 2

        q0_result = None
        q1_result = None

        for result_id, outcome in shot_results:
            if result_id == 0:
                q0_result = outcome
            elif result_id == 1:
                q1_result = outcome

        # In a Bell state, qubits should have the same measurement outcome
        if q0_result != q1_result:
            non_correlated_count += 1

    # We expect almost all measurements to be correlated
    # Allow for a small margin of error (5%)
    assert (
        non_correlated_count <= 0.05 * num_shots
    ), f"Expected high correlation in Bell state, but got {non_correlated_count}/{num_shots} non-correlated results"


def test_simulator_reset() -> None:
    """Test resetting the simulator state."""
    # Create a simple circuit: X on qubit 0, measure qubit 0
    builder = ByteMessage.quantum_operations_builder()
    builder.add_x(0)
    builder.add_measurement(0, 0)
    circuit = builder.build()

    # Create a simulator with 1 qubit
    simulator = StateVecEngine(1)

    # Run the circuit
    simulator.reset()
    result1 = simulator.process(circuit)
    measurements1 = result1.measurement_results()

    # Qubit 0 should be in state |1⟩ after X gate
    assert len(measurements1) == 1
    assert measurements1[0][0] == 0  # result_id = 0
    assert measurements1[0][1] == 1  # outcome = 1

    # Reset and run again without X gate
    simulator.reset()

    # Create a circuit with just measurement
    builder = ByteMessage.quantum_operations_builder()
    builder.add_measurement(0, 0)
    measure_circuit = builder.build()

    # Run the circuit
    result2 = simulator.process(measure_circuit)
    measurements2 = result2.measurement_results()

    # Qubit 0 should be in state |0⟩ after reset
    assert len(measurements2) == 1
    assert measurements2[0][0] == 0  # result_id = 0
    assert measurements2[0][1] == 0  # outcome = 0
