"""Test dynamic circuit execution with Guppy programs.

This test suite validates that dynamic circuits - where conditionals depend on
mid-circuit measurement results - work correctly.

The execution model runs LLVM on a worker thread that pauses when measurement
results are needed, allowing proper back-and-forth between the classical
control engine and quantum system.
"""

import pytest
from guppylang import guppy
from guppylang.std.quantum import cx, h, measure, qubit, x
from pecos import Guppy, sim
from pecos_rslib import state_vector


class TestDynamicCircuitExecution:
    """Test cases for dynamic circuit execution."""

    def test_conditional_x_gate_deterministic(self) -> None:
        """Test that conditional X gate based on measurement works correctly.

        This test creates a qubit in |0>, measures it (getting False), and
        conditionally applies X to a second qubit. Without dynamic execution,
        this might not work correctly because the entire LLVM program would
        run before any quantum simulation.
        """

        @guppy
        def conditional_x_from_zero() -> bool:
            q1 = qubit()  # |0>
            q2 = qubit()  # |0>

            # Measure first qubit - should always be False (|0>)
            result1 = measure(q1)

            # Apply X to second qubit only if first was True
            # Since first is always False, X should NOT be applied
            if result1:
                x(q2)

            return measure(q2)  # Should always be False

        # Run the circuit
        results = sim(Guppy(conditional_x_from_zero)).qubits(2).quantum(state_vector()).seed(42).run(100)

        # Extract the return value (last measurement in each shot)
        # Results format: [[m1, m2], [m1, m2], ...] where m2 is the return value
        measurements = results.get("measurements", [])
        return_values = [shot[-1] for shot in measurements]

        # All results should be False since q1 is |0>, so X is never applied to q2
        ones_count = sum(1 for m in return_values if m)
        assert ones_count == 0, f"Conditional X from |0> should never trigger, but got {ones_count}/100 ones"

    def test_conditional_x_gate_from_one(self) -> None:
        """Test conditional X when source qubit is |1>."""

        @guppy
        def conditional_x_from_one() -> bool:
            q1 = qubit()
            q2 = qubit()

            # Flip first qubit to |1>
            x(q1)

            # Measure first qubit - should always be True (|1>)
            result1 = measure(q1)

            # Apply X to second qubit only if first was True
            # Since first is always True, X SHOULD be applied
            if result1:
                x(q2)

            return measure(q2)  # Should always be True

        # Run the circuit
        results = sim(Guppy(conditional_x_from_one)).qubits(2).quantum(state_vector()).seed(42).run(100)

        # Extract measurements
        measurements = results.get("measurements", [])
        if not measurements and "measurement_0" in results:
            measurements = results["measurement_0"]

        # All results should be True since q1 is |1>, so X is always applied to q2
        ones_count = sum(1 for m in measurements if m)
        assert ones_count == 100, f"Conditional X from |1> should always trigger, but got {ones_count}/100 ones"

    def test_measurement_feedback_entanglement(self) -> None:
        """Test that measurement feedback creates correct correlations.

        This test puts a qubit in superposition, measures it, and conditionally
        applies X to a second qubit based on the result. The second qubit should
        always match the first qubit's measurement result.
        """

        @guppy
        def measurement_feedback() -> tuple[bool, bool]:
            q1 = qubit()
            q2 = qubit()

            # Put first qubit in superposition
            h(q1)

            # Measure first qubit
            result1 = measure(q1)

            # Apply X to second qubit if first measured True
            # This should make q2 always match the measurement of q1
            if result1:
                x(q2)

            return result1, measure(q2)

        # Run the circuit
        results = sim(Guppy(measurement_feedback)).qubits(2).quantum(state_vector()).seed(42).run(100)

        # Extract measurements - should have two measurements per shot
        # Need to decode the results
        measurements = []
        if "measurement_0" in results and "measurement_1" in results:
            m0 = results["measurement_0"]
            m1 = results["measurement_1"]
            measurements = list(zip(m0, m1, strict=False))
        elif "measurements" in results:
            measurements = results["measurements"]

        # Both measurements should always match
        mismatches = sum(1 for (a, b) in measurements if a != b)
        assert (
            mismatches == 0
        ), f"Measurement feedback should create perfect correlation, but got {mismatches}/100 mismatches"

    def test_teleportation_like_protocol(self) -> None:
        """Test a simplified teleportation-like protocol with measurement feedback.

        This test demonstrates a protocol where:
        1. Create Bell pair between q1 and q2
        2. Prepare q0 in a known state (|1>)
        3. Do Bell measurement on q0 and q1
        4. Apply corrections to q2 based on measurement results
        5. Verify q2 ends up in the original state of q0
        """

        @guppy
        def teleport_one() -> bool:
            # Qubit to teleport (we'll set it to |1>)
            q0 = qubit()
            x(q0)  # Set to |1>

            # Create Bell pair
            q1 = qubit()
            q2 = qubit()
            h(q1)
            cx(q1, q2)

            # Bell measurement on q0 and q1
            cx(q0, q1)
            h(q0)
            m0 = measure(q0)
            m1 = measure(q1)

            # Apply corrections based on measurement results
            if m1:
                x(q2)
            if m0:
                # Z correction - for |1> state, Z has no observable effect on measurement
                # but we include it for completeness
                pass

            # Measure final state - should be |1>
            return measure(q2)

        # Run the circuit
        results = sim(Guppy(teleport_one)).qubits(3).quantum(state_vector()).seed(42).run(100)

        # Extract the return value (last measurement in each shot)
        # Results format: [[m0, m1, m2], ...] where m2 is the return value
        measurements = results.get("measurements", [])
        return_values = [shot[-1] for shot in measurements]

        # The teleported state should be |1>, so we expect all True
        ones_count = sum(1 for m in return_values if m)
        assert ones_count > 95, f"Teleportation of |1> should succeed with high probability, got {ones_count}/100 ones"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
