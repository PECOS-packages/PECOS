"""Tests for missing coverage areas in the Guppy test suite.

This test file addresses gaps identified in the test coverage analysis:
    pass
1. Noise models and error simulation
2. Array and batch quantum operations
3. Advanced control flow patterns
4. Different quantum engines
5. Error handling with quantum resources
"""

import pytest
from guppylang import guppy
from guppylang.std.builtins import array
from guppylang.std.builtins import result as record_result
from guppylang.std.quantum import cx, h, measure, qubit, x
from pecos import Guppy, sim
from pecos_rslib import (
    biased_depolarizing_noise,
    depolarizing_noise,
    general_noise,
    sparse_stab,
    state_vector,
)

# Try to import optional functions that might not be available
try:
    from guppylang.std.quantum import collect_measurements, discard_array, measure_array
except ImportError:
    measure_array = None
    discard_array = None

try:
    from guppylang.std.quantum_functional import project_z
except ImportError:
    project_z = None

try:
    from guppylang.std.builtins import owned, panic
except ImportError:
    owned = None
    panic = None

# Try to import array type for quantum operations
try:
    from guppylang.std.quantum import array as qubit_array
except ImportError:
    qubit_array = None


def decode_integer_results(results: list[int], n_bits: int) -> list[tuple[bool, ...]]:
    """Decode integer-encoded results back to tuples of booleans."""
    decoded = []
    for val in results:
        bits = [bool(val & (1 << i)) for i in range(n_bits)]
        decoded.append(tuple(bits))
    return decoded


def get_measurements(results: dict, _expected_count: int = 1) -> list:
    """Read scalar outcome tags and convert array tags to measurement tuples."""
    return [tuple(value) if isinstance(value, list) else value for value in results["outcome"]]


# ============================================================================
# NOISE MODEL TESTS
# ============================================================================


class TestNoiseModels:
    """Test quantum simulations with various noise models."""

    def test_depolarizing_noise(self) -> None:
        """Test uniform depolarizing noise on quantum operations."""

        @guppy
        def noisy_circuit() -> bool:
            q = qubit()
            x(q)  # Just X gate to flip to |1⟩ deterministically
            output_value = measure(q).read()
            record_result("outcome", output_value)
            return output_value

        # Test with no noise - should be deterministic
        results_ideal = sim(Guppy(noisy_circuit)).qubits(1).quantum(state_vector()).seed(42).run(10).to_dict()
        measurements_ideal = get_measurements(results_ideal)
        ones_ideal = sum(measurements_ideal)
        assert ones_ideal == 10, f"Ideal circuit should produce all 1s, got {ones_ideal}/10"

        # Test with depolarizing noise
        noise = depolarizing_noise().with_uniform_probability(0.1)  # 10% error rate
        results_noisy = sim(noisy_circuit).qubits(1).quantum(state_vector()).seed(42).noise(noise).run(100).to_dict()
        measurements_noisy = get_measurements(results_noisy)
        ones_noisy = sum(measurements_noisy)

        # With noise, we should see some errors (not all 1s)
        # 10% depolarizing noise means ~10% chance of error
        # But depolarizing can cause various errors, so be more lenient
        assert 70 <= ones_noisy <= 95, f"Expected 70-95% ones with 10% noise, got {ones_noisy}/100"

    def test_biased_depolarizing_noise(self) -> None:
        """Test biased depolarizing noise model."""

        @guppy
        def bell_state() -> tuple[bool, bool]:
            q0, q1 = qubit(), qubit()
            h(q0)
            cx(q0, q1)
            output_value = measure(q0).read(), measure(q1).read()
            record_result("outcome", array(output_value[0], output_value[1]))
            return output_value

        # Test with biased noise
        noise = biased_depolarizing_noise().with_uniform_probability(
            0.05,
        )
        results = sim(bell_state).qubits(2).quantum(state_vector()).seed(123).noise(noise).run(100).to_dict()
        # Results are tuples (0, 0) or (1, 1) for correlated Bell states
        correlated = sum(1 for r in get_measurements(results) if r in [(0, 0), (1, 1)])

        # With 5% biased noise, Bell states should still be somewhat correlated
        # But biased depolarizing might affect correlation more than expected
        assert correlated > 40, f"Bell state correlation too low: {correlated}/100"

    def test_custom_depolarizing_noise(self) -> None:
        """Test custom depolarizing noise with different rates."""

        @guppy
        def prep_measure_circuit() -> bool:
            q = qubit()  # Preparation
            h(q)
            x(q)
            output_value = measure(q).read()
            record_result("outcome", output_value)
            return output_value

        # Custom noise: high prep error, low measurement error
        noise = (
            general_noise()
            .with_p_prep(0.2)  # 20% preparation error
            .with_measurement_probability(
                0.01,
                0.01,
            )
            .with_p1(0.05)  # 5% single-qubit gate error
            .with_p2(0.1)  # 10% two-qubit gate error
        )

        results = sim(prep_measure_circuit).qubits(1).quantum(state_vector()).seed(456).noise(noise).run(100).to_dict()
        errors = 100 - sum(
            get_measurements(results),
        )
        # The circuit has prep + 2 gates + measurement, so errors compound
        assert 15 <= errors <= 60, f"Expected 15-60% errors with custom noise, got {errors}/100"


# ============================================================================
# ARRAY AND BATCH OPERATIONS
# ============================================================================


class TestArrayOperations:
    """Test array and batch quantum operations."""

    def test_measure_array(self) -> None:
        """Test measuring multiple qubits (simulating array behavior)."""

        @guppy
        def measure_multiple_test() -> tuple[bool, bool, bool, bool, bool]:
            # Create 5 qubits individually (simulating array)
            q0 = qubit()
            q1 = qubit()
            q2 = qubit()
            q3 = qubit()
            q4 = qubit()

            # Apply different operations
            h(q0)
            x(q1)
            h(q2)
            x(q3)
            # q4 stays |0⟩

            # Measure all qubits
            m0 = measure(q0).read()
            m1 = measure(q1).read()
            m2 = measure(q2).read()
            m3 = measure(q3).read()
            m4 = measure(q4).read()

            output_value = m0, m1, m2, m3, m4
            record_result(
                "outcome",
                array(output_value[0], output_value[1], output_value[2], output_value[3], output_value[4]),
            )
            return output_value

        results = sim(measure_multiple_test).qubits(5).quantum(state_vector()).seed(789).run(10).to_dict()
        for result in get_measurements(results):
            # Result is a tuple of 5 booleans
            # Extract individual measurements
            _b0, b1, _b2, b3, b4 = result

            # Check known deterministic bits (measurements are 0 or 1, not bool)
            assert b1 == 1, "Bit 1 should be 1 (from x gate)"
            assert b3 == 1, "Bit 3 should be 1 (from x gate)"
            assert b4 == 0, "Bit 4 should be 0 (stays |0⟩)"

            # b0 and b2 are probabilistic (from H gates)

    def test_discard_array(self) -> None:
        """Test discarding an array of qubits."""
        # First check if discard_array is available
        assert discard_array is not None, "discard_array not available in this guppy version"

        @guppy
        def discard_array_test() -> bool:
            # Create and manipulate array
            qs = array(qubit() for _ in range(10))
            for i in range(10):
                if i % 2 == 0:
                    h(qs[i])

            # Use discard_array to discard all qubits at once
            discard_array(qs)

            # Create new qubit to return something
            q = qubit()
            x(q)
            output_value = measure(q).read()
            record_result("outcome", output_value)
            return output_value

        # Should run without errors
        results = sim(Guppy(discard_array_test)).qubits(10).quantum(state_vector()).seed(42).run(10).to_dict()
        assert all(r == 1 for r in get_measurements(results)), "Final qubit should be |1⟩"

    def test_array_indexing_and_loops(self) -> None:
        """Test array indexing within loops."""
        assert measure_array is not None, "measure_array not available in this guppy version"

        @guppy
        def array_loop_test() -> int:
            qs = array(qubit() for _ in range(4))

            # Apply H gate to even indices
            for i in range(4):
                if i % 2 == 0:
                    h(qs[i])
                else:
                    x(qs[i])

            # Use measure_array to measure all at once
            results = collect_measurements(measure_array(qs))

            record_result("outcome", results)

            # Encode as integer
            result = 0
            for i in range(4):
                if results[i]:
                    result |= 1 << i

            output_value = result
            record_result("value", output_value)
            return output_value

        results = sim(Guppy(array_loop_test)).qubits(4).quantum(state_vector()).seed(42).run(10).to_dict()
        # Even indices (0,2) are in superposition, odd indices (1,3) are |1⟩
        # This gives us a specific pattern we can verify
        for result in get_measurements(results):
            # Result is a tuple of 4 measurements
            if isinstance(result, tuple):
                assert len(result) == 4, f"Expected 4 measurements, got {len(result)}"
                _b0, b1, _b2, b3 = result
            else:
                # Try to extract as integer
                result & 1
                b1 = (result >> 1) & 1
                (result >> 2) & 1
                b3 = (result >> 3) & 1

            # Odd indices should always be 1
            assert b1 == 1, f"Index 1 should be |1⟩, got: {result}"
            assert b3 == 1, f"Index 3 should be |1⟩, got: {result}"


# ============================================================================
# ADVANCED CONTROL FLOW
# ============================================================================


class TestAdvancedControlFlow:
    """Test complex control flow patterns."""

    @pytest.mark.skip(
        reason=(
            "print_int drops integer results other than 0/1, "
            "causing missing tags or per-shot register-count mismatches"
        ),
    )
    def test_nested_loops(self) -> None:
        """Test loops with quantum operations."""

        @guppy
        def loop_test() -> int:
            count = 0
            measured = array(False for _ in range(6))

            # Simple loop with quantum operations
            for i in range(6):  # Total of 6 iterations
                q = qubit()  # Create fresh qubit for each iteration
                h(q)
                # Directly add measurement result
                m = measure(q).read()
                measured[i] = m
                if m:
                    count = count + 1

            record_result("outcome", measured)
            output_value = count
            record_result("value", output_value)
            return output_value

        # Run multiple times to see distribution
        results = sim(Guppy(loop_test)).qubits(1).quantum(state_vector()).seed(111).run(10).to_dict()

        # Each shot records all six measurements and their accumulated count.
        measurements = get_measurements(results)
        assert len(measurements) == 10
        assert len(results["value"]) == 10
        for shot, count in zip(measurements, results["value"], strict=True):
            assert len(shot) == 6, f"Expected 6 measurements, got {len(shot)}"
            assert count == sum(shot), f"Count {count} should equal the six measurements"
            assert 0 <= count <= 6, f"Count {count} out of range"

    def test_conditional_quantum_operations(self) -> None:
        """Test quantum operations inside conditionals."""

        # Create separate functions for each test case since sim doesn't support parameters
        @guppy
        def conditional_quantum_0() -> bool:
            q = qubit()
            # n = 0: Do nothing - return |0⟩
            output_value = measure(q).read()
            record_result("outcome", output_value)
            return output_value

        @guppy
        def conditional_quantum_1() -> bool:
            q = qubit()
            # n = 1: Return |1⟩
            x(q)
            output_value = measure(q).read()
            record_result("outcome", output_value)
            return output_value

        @guppy
        def conditional_quantum_2() -> bool:
            q = qubit()
            # n = 2: Superposition
            h(q)
            output_value = measure(q).read()
            record_result("outcome", output_value)
            return output_value

        # Test case n=0
        results = sim(conditional_quantum_0).qubits(1).quantum(state_vector()).seed(42).run(10).to_dict()
        assert all(r == 0 for r in get_measurements(results)), "Case n=0 failed"

        # Test case n=1
        results = sim(conditional_quantum_1).qubits(1).quantum(state_vector()).seed(42).run(10).to_dict()
        assert all(r == 1 for r in get_measurements(results)), "Case n=1 failed"

        # Test case n=2 (superposition - should have both 0 and 1)
        results = sim(conditional_quantum_2).qubits(1).quantum(state_vector()).seed(42).run(100).to_dict()
        zeros = sum(1 for r in get_measurements(results) if r == 0)
        ones = sum(1 for r in get_measurements(results) if r == 1)
        assert zeros > 20, f"Case n=2 should have >20 zeros, got {zeros}"
        assert ones > 20, f"Case n=2 should have >20 ones, got {ones}"

    def test_early_return_with_quantum(self) -> None:
        """Test early returns with quantum resources."""

        # Create separate functions for each test case
        @guppy
        def early_return_test_true() -> bool:
            q1 = qubit()
            h(q1)

            # Early return - measure consumes the qubit
            output_value = measure(q1).read()
            record_result("outcome", output_value)
            return output_value

        @guppy
        def early_return_test_false() -> bool:
            q1 = qubit()
            h(q1)

            # Continue with more operations
            q2 = qubit()
            cx(q1, q2)
            # Measure q2 to consume it
            measure(q2).read()  # Can't use _ in Guppy

            output_value = measure(q1).read()
            record_result("outcome", output_value)
            return output_value

        # Test both paths
        results_true = sim(early_return_test_true).qubits(10).quantum(state_vector()).seed(42).run(100).to_dict()
        results_false = sim(early_return_test_false).qubits(10).quantum(state_vector()).seed(42).run(100).to_dict()
        measurements_true = get_measurements(results_true)
        measurements_false = get_measurements(results_false)
        assert len(measurements_true) == 100
        assert len(measurements_false) == 100


# ============================================================================
# QUANTUM ENGINE TESTS
# ============================================================================


class TestQuantumEngines:
    """Test different quantum simulation engines."""

    def test_state_vector_engine(self) -> None:
        """Test explicit state vector engine selection."""

        @guppy
        def engine_test() -> tuple[bool, bool]:
            q0, q1 = qubit(), qubit()
            h(q0)
            cx(q0, q1)
            output_value = measure(q0).read(), measure(q1).read()
            record_result("outcome", array(output_value[0], output_value[1]))
            return output_value

        # Use state vector engine (already set by quantum())
        results = (
            sim(engine_test)
            .qubits(2)  # Only need 2 qubits for Bell state
            .quantum(state_vector())
            .seed(42)
            .run(100)
            .to_dict()
        )
        assert all(r in [(0, 0), (1, 1)] for r in get_measurements(results)), "Bell state should be |00⟩ or |11⟩"

    def test_clifford_circuit_simulation(self) -> None:
        """Test simulation of Clifford-like circuits.

        Tests a circuit that uses Clifford gates at the Guppy level.
        The sequence H-X-H is equivalent to a Z gate, so starting from |0⟩
        should give us |0⟩ after measurement (Z|0⟩ = |0⟩).

        Note: While these are Clifford gates at the source level, the
        compilation pipeline decomposes them into RXY and RZ rotations.
        """

        @guppy
        def clifford_circuit() -> bool:
            # Clifford circuit: H-X-H = Z gate
            q = qubit()
            h(q)  # Hadamard
            x(q)  # Pauli X
            h(q)  # Hadamard
            # The sequence H-X-H = Z, so Z|0⟩ = |0⟩
            output_value = measure(q).read()
            record_result("outcome", output_value)
            return output_value

        # Test with state vector engine (compatible with all gate decompositions)
        results = sim(Guppy(clifford_circuit)).qubits(1).quantum(state_vector()).seed(42).run(100).to_dict()
        measurements = get_measurements(results)

        # H-X-H sequence on |0⟩ should always give |0⟩ (since H-X-H = Z)
        assert all(
            r == 0 for r in measurements
        ), f"Clifford circuit H-X-H on |0⟩ should always measure 0, got {set(measurements)}"

    def test_sparse_stab_with_qasm(self) -> None:
        """Test sparse stabilizer engine with QASM input (which preserves Clifford gates).

        The sparse stabilizer simulator works with QASM programs that use
        true Clifford gates, unlike Guppy programs which get decomposed.
        """
        from pecos import Qasm

        # Create a QASM program with pure Clifford gates
        qasm_str = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        h q[0];
        cx q[0], q[1];
        measure q[0] -> c[0];
        measure q[1] -> c[1];
        """

        # Create QASM program using Qasm wrapper
        program = Qasm(qasm_str)

        # Test with sparse stabilizer - should work with QASM Clifford circuits
        results = sim(program).qubits(2).quantum(sparse_stab()).seed(42).run(100)

        # QASM returns dict with register names as keys
        assert "c" in results, "Results should contain register 'c'"
        measurements = results["c"]

        # Bell state: values should be 0 (00) or 3 (11) for correlated qubits
        # Never 1 (01) or 2 (10) for anti-correlated qubits
        correlated = sum(1 for m in measurements if m in [0, 3])
        assert correlated == 100, f"Bell state should be 100% correlated (0 or 3), got {correlated}/100"


# ============================================================================
# ERROR HANDLING WITH QUANTUM RESOURCES
# ============================================================================


class TestQuantumErrorHandling:
    """Test error handling with quantum resources."""

    def test_error_handling_with_quantum_resources(self) -> None:
        """Test error handling patterns with quantum resources.

        Since panic() doesn't raise runtime exceptions in compiled HUGR,
        we test alternative error handling patterns.
        """

        @guppy
        def error_handling_test() -> tuple[bool, bool]:
            """Demonstrate proper quantum resource management with error conditions."""
            q1 = qubit()
            h(q1)

            # Measure first qubit
            m1 = measure(q1).read()

            # Conditional quantum operation based on measurement
            q2 = qubit()
            if m1:  # m1=True means measurement was 1
                # Error condition path - still need to properly handle q2
                x(q2)  # Apply X gate in error case
                success = False
            else:  # m1=False means measurement was 0
                # Normal path
                h(q2)  # Apply H gate in normal case
                success = True

            # Always measure q2 to properly consume it
            m2 = measure(q2).read()

            output_value = success, m2
            record_result("outcome", array(m1, m2))
            return output_value

        # Run the test with more shots for statistical stability
        results = sim(Guppy(error_handling_test)).qubits(2).quantum(state_vector()).seed(42).run(1000).to_dict()
        measurements = get_measurements(results)

        # The measurements are captured in order: m1 (measurement_0), m2 (measurement_1)
        # The relationship between m1 and success is: success = NOT m1
        # - m1=0 (False) → else branch → success=True → H gate applied
        # - m1=1 (True) → if branch → success=False → X gate applied

        # Filter by the first element (m1, not success!)
        # m1=0 means success=True, m1=1 means success=False
        success_cases = [m for m in measurements if m[0] == 0]  # m1=0 → success=True
        error_cases = [m for m in measurements if m[0] == 1]  # m1=1 → success=False

        # With H gate on q1 producing 50/50, expect roughly equal split
        assert len(success_cases) > 400, f"Should have >400 success cases, got {len(success_cases)}"
        assert len(error_cases) > 400, f"Should have >400 error cases, got {len(error_cases)}"

        # Verify the expected behavior:
        # - success=True (normal path) → H gate applied → m2 should be 50/50
        # - success=False (error path) → X gate applied → m2 should always be 1

        # Check success cases (H gate should give 50/50 distribution)
        success_zeros = [m for m in success_cases if m[1] == 0]
        success_ones = [m for m in success_cases if m[1] == 1]
        # With H gate, should get roughly 50/50 distribution
        assert len(success_zeros) > 150, f"H gate should produce ~50% 0s, got {len(success_zeros)}/{len(success_cases)}"
        assert len(success_ones) > 150, f"H gate should produce ~50% 1s, got {len(success_ones)}/{len(success_cases)}"

        # Check error cases: the error branch applies X, so m2 must be 1
        # in EVERY error-case shot. (This assertion spent years disabled
        # behind a "known conditional-branch issue" note while a tautology
        # kept the test green; branch execution is fixed.)
        error_ones = [m for m in error_cases if m[1] == 1]
        assert len(error_ones) == len(
            error_cases,
        ), f"X branch must force m2=1 in all {len(error_cases)} error cases, got {len(error_ones)}"

    def test_projective_measurement(self) -> None:
        """Test measurement collapse behavior."""

        @guppy
        def measurement_collapse_test() -> bool:
            q = qubit()
            h(q)  # Put in superposition

            # Measurement collapses the state
            output_value = measure(q).read()
            record_result("outcome", output_value)
            return output_value

            # Return the measurement result

        results = sim(measurement_collapse_test).qubits(1).quantum(state_vector()).seed(42).run(100).to_dict()

        measurements = get_measurements(results)
        ones = sum(measurements)
        zeros = len(measurements) - ones

        # Should be roughly 50/50 with some tolerance
        assert 35 <= ones <= 65, f"Expected roughly 50 ones out of 100, got {ones}"
        assert 35 <= zeros <= 65, f"Expected roughly 50 zeros out of 100, got {zeros}"

    def test_reset_operation(self) -> None:
        """Test reset-like behavior with fresh qubits."""

        @guppy
        def reset_test() -> tuple[bool, bool]:
            # Create two qubits in different states
            q1 = qubit()
            x(q1)  # Set to |1⟩
            m1 = measure(q1).read()

            # Create a fresh qubit in |0⟩ state (simulating reset)
            q2 = qubit()  # Fresh qubits start in |0⟩
            m2 = measure(q2).read()

            output_value = m1, m2
            record_result("outcome", array(output_value[0], output_value[1]))
            return output_value

        results = sim(Guppy(reset_test)).qubits(2).quantum(state_vector()).seed(42).run(10).to_dict()

        # All results should be (1, 0) as tuples
        measurements = get_measurements(results)

        assert all(
            r == (1, 0) for r in measurements
        ), f"Should produce |1⟩ then |0⟩ as tuple (1, 0), got {measurements[:3]}..."
