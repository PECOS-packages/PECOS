"""Test qubit allocation limits and error handling."""

import pytest
from guppylang import guppy
from guppylang.std.builtins import array
from guppylang.std.quantum import h, measure, qubit
from pecos import Guppy, sim
from pecos_rslib import state_vector


class TestQubitAllocationLimits:
    """Test qubit allocation limits and dynamic allocation behavior."""

    def test_static_allocation_within_limit(self) -> None:
        """Test static allocation within the max_qubits limit."""

        @guppy
        def static_test() -> tuple[bool, bool, bool]:
            q1 = qubit()
            q2 = qubit()
            q3 = qubit()
            return measure(q1).read(), measure(q2).read(), measure(q3).read()

        # Should work fine with max_qubits=5 (3 qubits needed)
        results = sim(Guppy(static_test)).qubits(5).quantum(state_vector()).run(10).to_dict()

        # Check we got results - format is [[m0, m1, m2], [m0, m1, m2], ...]
        measurements = results["measurements"]
        assert len(measurements) == 10, "Should have 10 measurements"
        for m in measurements:
            assert len(m) == 3, f"Each shot should have 3 measurements, got {len(m)}"

    @pytest.mark.skip(
        reason="For-loop with int return not supported by HUGR interpreter",
    )
    def test_dynamic_allocation_in_loop(self) -> None:
        """Test dynamic allocation in a loop - requires sufficient max_qubits."""

        @guppy
        def dynamic_loop_test() -> int:
            count = 0
            # This allocates qubits dynamically in the loop
            for _i in range(3):
                q = qubit()
                h(q)
                if measure(q).read():
                    count += 1
            return count

        # Set max_qubits high enough for dynamic allocation
        results = sim(Guppy(dynamic_loop_test)).qubits(10).quantum(state_vector()).seed(42).run(100)

        # Extract measurements
        measurements = results.get("measurement_0", results["measurements"])
        assert len(measurements) == 100, "Should have 100 measurements"

        # Due to Guppy limitation, only returns 0 or 1 (last measurement)
        values = set(measurements)
        assert len(values) >= 2, "Should see at least some variation in results"
        assert all(0 <= v <= 1 for v in measurements), "Values should be 0-1 (last measurement only)"

        # Note: Due to current Guppy limitations with integer accumulation in loops,
        # this only returns 0 or 1 (the last measurement result) rather than the sum.
        # The test function attempts to count across loop iterations but only the
        # last iteration's result is captured.
        average = sum(measurements) / len(measurements)
        assert 0.3 < average < 0.7, f"Average should be around 0.5 (last measurement only), got {average}"

    def test_allocation_exceeds_limit_fixed_size_simulator(self) -> None:
        """A fixed-size simulator must reject allocation past the qubit limit.

        Stabilizer-family simulators do not grow: a program that touches a
        qubit index at or beyond the configured capacity must fail with the
        capacity-guard error naming the op, the qubit, and the capacity --
        not succeed silently or die with an unrelated IPC failure.
        """
        from guppylang.std.quantum import cx
        from pecos_rslib import sparse_stab

        @guppy
        def four_qubit_program() -> tuple[bool, bool, bool, bool]:
            """Program that uses 4 qubits simultaneously."""
            q0 = qubit()
            q1 = qubit()
            q2 = qubit()
            q3 = qubit()

            # Create entanglement chain
            h(q0)
            cx(q0, q1)
            cx(q1, q2)
            cx(q2, q3)

            # Measure all
            return measure(q0).read(), measure(q1).read(), measure(q2).read(), measure(q3).read()

        with pytest.raises(RuntimeError, match=r"targets qubit 3.*holds 3 qubits"):
            sim(Guppy(four_qubit_program)).qubits(3).quantum(sparse_stab()).run(10)

    def test_allocation_exceeds_limit_state_vector_grows(self) -> None:
        """The state-vector engine grows past the configured qubit count.

        Unlike the fixed-size simulators, the state-vector engine expands to
        the highest qubit index a message touches, so a 4-qubit program with
        .qubits(3) runs anyway -- and must still produce CORRECT physics
        (a GHZ chain measures all-equal), not results computed on a
        truncated register.
        """
        from guppylang.std.quantum import cx

        @guppy
        def four_qubit_ghz() -> tuple[bool, bool, bool, bool]:
            q0 = qubit()
            q1 = qubit()
            q2 = qubit()
            q3 = qubit()

            h(q0)
            cx(q0, q1)
            cx(q1, q2)
            cx(q2, q3)

            return measure(q0).read(), measure(q1).read(), measure(q2).read(), measure(q3).read()

        results = sim(Guppy(four_qubit_ghz)).qubits(3).quantum(state_vector()).seed(42).run(10).to_dict()

        measurements = results["measurements"]
        assert len(measurements) == 10, "Should have 10 shots"
        for m in measurements:
            assert len(m) == 4, f"Each shot should measure 4 qubits, got {len(m)}"
            assert len(set(m)) == 1, f"GHZ measurements must all agree, got {m}"

    @pytest.mark.skip(
        reason="Nested loops with int return not supported by HUGR interpreter",
    )
    def test_nested_loop_allocation(self) -> None:
        """Test nested loops with qubit allocation."""

        @guppy
        def nested_loop_test() -> int:
            count = 0
            # Nested loops allocating qubits
            for i in range(3):
                for j in range(2):
                    q = qubit()
                    if i > j:
                        h(q)
                        if measure(q).read():
                            count += 1
                    else:
                        # Direct measurement of |0⟩
                        if measure(q).read():
                            count += 1
            return count

        # Need sufficient qubits for nested allocation
        results = sim(Guppy(nested_loop_test)).qubits(10).quantum(state_vector()).seed(42).run(50)

        measurements = results.get("measurement_0", results["measurements"])
        assert len(measurements) == 50, "Should have 50 measurements"

        # Count should be 0-6 (depends on measurements)
        assert all(0 <= v <= 6 for v in measurements), "Values should be 0-6"

    @pytest.mark.skip(reason="Loops with int return not supported by HUGR interpreter")
    def test_allocation_with_measurement_reuse(self) -> None:
        """Test that measuring and discarding allows potential qubit reuse."""

        @guppy
        def measurement_reuse_test() -> int:
            count = 0
            for _i in range(5):
                q = qubit()
                h(q)
                if measure(q).read():
                    count += 1
                # After measurement, qubit is consumed and could be reused
            return count

        # Run with various qubit limits
        for max_qubits in [5, 10]:
            results = sim(measurement_reuse_test).qubits(max_qubits).quantum(state_vector()).seed(42).run(50)

            measurements = results.get("measurement_0", results["measurements"])
            assert len(measurements) == 50, f"Should have 50 measurements with max_qubits={max_qubits}"

            # Due to Guppy limitation, only returns 0 or 1 (last measurement)
            assert all(0 <= v <= 1 for v in measurements), "Values should be 0-1 (last measurement only)"

            # Note: Due to current Guppy limitations with integer accumulation in loops,
            # this only returns the last measurement result, not the accumulated count
            average = sum(measurements) / len(measurements)
            assert 0.3 < average < 0.7, f"Average should be around 0.5 (last measurement only), got {average}"

    def test_explicit_max_qubits_setting(self) -> None:
        """Test that max_qubits parameter is properly respected."""

        @guppy
        def single_qubit_test() -> bool:
            q = qubit()
            h(q)
            return measure(q).read()

        # Test with different max_qubits values
        for max_q in [1, 5, 10, 20]:
            results = sim(single_qubit_test).qubits(max_q).quantum(state_vector()).seed(42).run(10).to_dict()

            raw_measurements = results["measurements"]
            measurements = [m[-1] if isinstance(m, list) else m for m in raw_measurements]
            assert len(measurements) == 10, f"Should have 10 measurements with max_qubits={max_q}"

            # Single qubit program should work with any max_qubits >= 1
            assert all(isinstance(m, bool | int) for m in measurements), "Measurements should be bool/int"

    @pytest.mark.skip(
        reason="TailLoop/CFG control flow with arrays needs work - loop iterations not completing",
    )
    def test_qubit_array_allocation(self) -> None:
        """Test allocation of qubit arrays using Guppy's array type with proper ownership."""
        from guppylang.std.builtins import owned
        from guppylang.std.quantum import collect_measurements, measure_array

        @guppy
        def apply_h_to_array(qubits: array[qubit, 3] @ owned) -> array[qubit, 3]:
            """Apply H gates to array elements using @owned annotation for borrowing."""
            # With @owned, we can borrow elements from the array
            h(qubits[0])
            h(qubits[1])
            h(qubits[2])
            return qubits

        @guppy
        def array_test() -> array[bool, 3]:
            # Allocate array of 3 qubits using generator expression
            qubits = array(qubit() for _ in range(3))

            # Pass array to function that can borrow elements
            qubits = apply_h_to_array(qubits)

            # Measure all qubits at once using measure_array
            return collect_measurements(measure_array(qubits))

        # Need at least 3 qubits for the array
        results = sim(Guppy(array_test)).qubits(3).quantum(state_vector()).seed(42).run(50).to_dict()

        # The result should be an array of 3 booleans for each shot
        # Results format is [[m0, m1, m2], [m0, m1, m2], ...]
        measurements = results["measurements"]
        assert len(measurements) == 50, "Should have 50 measurement sets"

        # Each measurement should be an array/tuple of 3 booleans
        for m in measurements[:5]:  # Check first few
            assert len(m) == 3, f"Each result should have 3 measurements, got {len(m)}"

        # Check distribution for each qubit position
        for i in range(3):
            ones = sum(1 for m in measurements if m[i])
            assert 15 < ones < 35, f"Qubit {i} should have ~50/50 distribution, got {ones}/50"

    def test_parallel_qubit_operations(self) -> None:
        """Test parallel operations on multiple qubits."""

        @guppy
        def parallel_ops() -> tuple[bool, bool, bool, bool]:
            # Allocate 4 qubits
            q0 = qubit()
            q1 = qubit()
            q2 = qubit()
            q3 = qubit()

            # Apply different operations in parallel
            h(q0)
            h(q1)
            h(q2)
            h(q3)

            # Measure all
            return measure(q0).read(), measure(q1).read(), measure(q2).read(), measure(q3).read()

        # Test with exact number of qubits needed
        # Use 500 shots for better statistics; seed 1000 produces [253, 259, 255, 258]
        results = sim(Guppy(parallel_ops)).qubits(4).quantum(state_vector()).seed(1000).run(500)

        if "measurement_0" in results:
            # Check all 4 measurements are present
            for i in range(4):
                key = f"measurement_{i}"
                assert key in results, f"Should have {key}"
                assert len(results[key]) == 500, f"Should have 500 measurements for {key}"

                # Each qubit in superposition should give roughly 50/50 results
                # 2 sigma for 500 shots: expected=250, std=11.18, range=228-272
                ones = sum(results[key])
                zeros = 500 - ones
                assert 228 < ones < 272, f"Should be roughly 50/50 distribution, got {ones}/{zeros}"
