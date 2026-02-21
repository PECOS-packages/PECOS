"""Test the sim builder pattern API.

This test demonstrates the sim() API builder pattern for quantum simulations.
"""

from pathlib import Path

import pytest
from guppylang import guppy
from guppylang.std.quantum import cx, h, measure, qubit
from pecos import Guppy, sim
from pecos_rslib import state_vector


def decode_integer_results(results: list[int], n_bits: int) -> list[tuple[bool, ...]]:
    """Decode integer-encoded results back to tuples of booleans."""
    decoded = []
    for val in results:
        bits = [bool(val & (1 << i)) for i in range(n_bits)]
        decoded.append(tuple(bits))
    return decoded


class TestGuppySimBuilder:
    """Test the sim builder pattern."""

    @guppy
    def bell_state() -> tuple[bool, bool]:
        """Create a Bell state."""
        q0, q1 = qubit(), qubit()
        h(q0)
        cx(q0, q1)
        return measure(q0), measure(q1)

    @guppy
    def single_qubit() -> bool:
        """Single qubit in superposition."""
        q = qubit()
        h(q)
        return measure(q)

    def test_basic_build_and_run(self) -> None:
        """Test basic build() and run() pattern."""
        # Build once
        # Run multiple times with same configuration
        results1 = sim(self.bell_state).qubits(10).quantum(state_vector()).run(10)
        results2 = sim(self.bell_state).qubits(10).quantum(state_vector()).run(10)

        # Check format has measurement results
        # Bell state returns tuple, so we should have measurement_0 and measurement_0
        if "measurement_0" in results1 and "measurement_0" in results1:
            # New format with individual measurement keys
            assert len(results1["measurement_0"]) == 10
            assert len(results1["measurement_0"]) == 10
            assert len(results2["measurement_0"]) == 10
            assert len(results2["measurement_0"]) == 10
        else:
            # Fallback to old format
            measurements1 = results1.get("measurements", results1.get("result", []))
            measurements2 = results2.get("measurements", results2.get("result", []))
            assert len(measurements1) == 10
            assert len(measurements2) == 10

    def test_direct_run(self) -> None:
        """Test direct run() without explicit build()."""
        results = sim(self.single_qubit).qubits(10).quantum(state_vector()).run(10).to_dict()

        # Check that we have measurement results
        raw_measurements = results.get("measurements", [])
        measurements = [m[-1] if isinstance(m, list) else m for m in raw_measurements]
        assert len(measurements) == 10
        assert all(r in [0, 1] for r in measurements)

    def test_builder_methods(self) -> None:
        """Test the builder pattern methods of the sim API."""
        builder = (
            sim(self.bell_state)
            .qubits(2)
            .quantum(state_vector())
            .seed(42)
            .workers(2)
            .verbose(True)
            .debug(False)
            .optimize(True)
        )
        sim_obj = builder.build()
        results = sim_obj.run(100)

        measurements = results.get(
            "measurements",
            results.get("measurement_0", results.get("result", [])),
        )
        assert measurements is not None
        assert len(measurements) > 0
        assert len(measurements) == 100  # 100 shots, each with integer-encoded 2 qubits

    def test_seeded_reproducibility(self) -> None:
        """Test that seeded runs are reproducible."""
        # Run with same seed twice
        results1 = sim(self.single_qubit).qubits(10).quantum(state_vector()).seed(12345).run(100)
        results2 = sim(self.single_qubit).qubits(10).quantum(state_vector()).seed(12345).run(100)
        measurements1 = results1.get(
            "measurements",
            results1.get("measurement_0", results1.get("result", [])),
        )
        measurements2 = results2.get(
            "measurements",
            results2.get("measurement_0", results2.get("result", [])),
        )
        assert measurements1 == measurements2

    def test_config_dict(self) -> None:
        """Test configuration via dictionary."""
        # Test seed configuration (most commonly used)
        results = sim(self.bell_state).qubits(10).quantum(state_vector()).seed(42).run(50)
        if "measurement_0" in results:
            assert len(results["measurement_0"]) == 50
            assert len(results["measurement_1"]) == 50
        else:
            measurements = results.get("measurements", results.get("result", []))
            assert len(measurements) == 50

    def test_bell_state_correlation(self) -> None:
        """Test that Bell state results are correlated."""
        results = sim(self.bell_state).qubits(10).quantum(state_vector()).seed(42).run(1000).to_dict()

        # Measurements format is [[m0, m1], [m0, m1], ...]
        raw_measurements = results.get("measurements", [])
        correlated = sum(1 for m in raw_measurements if m[0] == m[1])
        assert correlated == len(
            raw_measurements,
        ), "Bell state should be 100% correlated"

    def test_keep_intermediate_files(self) -> None:
        """Test keeping intermediate compilation files."""
        import shutil

        sim_obj = sim(self.single_qubit).qubits(10).quantum(state_vector()).keep_intermediate_files(True).build()
        assert sim_obj.temp_dir is not None
        assert Path(sim_obj.temp_dir).exists()

        # Check that intermediate files exist
        temp_path = Path(sim_obj.temp_dir)
        ll_files = list(temp_path.glob("*.ll"))
        hugr_files = list(temp_path.glob("*.hugr"))

        assert len(ll_files) > 0, "Should have created LLVM IR file"
        assert len(hugr_files) > 0, "Should have created HUGR file"

        # Run simulation
        results = sim_obj.run(10).to_dict()
        raw_measurements = results.get("measurements", [])
        measurements = [m[-1] if isinstance(m, list) else m for m in raw_measurements]
        assert len(measurements) == 10

        # Files should still exist after run
        assert Path(sim_obj.temp_dir).exists()
        assert ll_files[0].exists()
        assert hugr_files[0].exists()

        # Manually clean up
        shutil.rmtree(sim_obj.temp_dir, ignore_errors=True)


def test_api_demonstration() -> None:
    """Demonstrate the builder pattern API."""
    try:
        from guppylang import guppy
        from guppylang.std.quantum import h, measure, qubit
    except ImportError:
        pytest.skip("Guppy not available")
        return

    @guppy
    def demo_circuit() -> bool:
        """Demo circuit that creates superposition and measures."""
        q = qubit()
        h(q)
        return measure(q)

    # Show builder pattern
    sim_obj = sim(demo_circuit).qubits(10).quantum(state_vector()).seed(42).verbose(True).build()
    results = sim_obj.run(100)
    results.get(
        "measurements",
        results.get("measurement_0", results.get("result", [])),
    )

    # print("\n3. Running 1000 shots with a new builder...")
    # Need to create a new builder since the previous one is consumed
    results = sim(Guppy(demo_circuit)).qubits(10).quantum(state_vector()).seed(42).run(1000)
    results.get(
        "measurements",
        results.get("measurement_0", results.get("result", [])),
    )
    results = sim(Guppy(demo_circuit)).qubits(10).quantum(state_vector()).seed(123).run(50)
    results.get(
        "measurements",
        results.get("measurement_0", results.get("result", [])),
    )


def test_simulation_reset() -> None:
    """Test that reset() returns the simulator and allows re-running."""
    try:
        from guppylang import guppy
        from guppylang.std.quantum import h, measure, qubit
    except ImportError:
        pytest.skip("Guppy not available")
        return

    @guppy
    def superposition() -> bool:
        """Create superposition and measure."""
        q = qubit()
        h(q)
        return measure(q)

    # Build a simulation
    sim_obj = sim(superposition).qubits(10).quantum(state_vector()).seed(42).build()

    # Run once
    results1 = sim_obj.run(100)
    assert len(results1) == 100

    # Reset should return the same simulator object
    returned_sim = sim_obj.reset()
    assert returned_sim is sim_obj

    # Run again after reset - should work and give results
    results2 = sim_obj.run(100)
    assert len(results2) == 100

    # With the same seed, reset should give reproducible results
    sim_obj_seeded = sim(superposition).qubits(10).quantum(state_vector()).seed(42).build()
    results_a = sim_obj_seeded.run(50)
    sim_obj_seeded.reset()
    results_b = sim_obj_seeded.run(50)

    # Both runs should produce the same results since we reset with the same seed
    assert results_a.to_dict() == results_b.to_dict()


def test_reset_returns_to_zero_state() -> None:
    """Test that reset() returns the quantum state to all |0⟩."""
    try:
        from guppylang import guppy
        from guppylang.std.quantum import measure, qubit
    except ImportError:
        pytest.skip("Guppy not available")
        return

    @guppy
    def measure_without_gates() -> bool:
        """Measure a qubit without any gates - should always be 0 in |0⟩ state."""
        q = qubit()
        return measure(q)

    # Build a simulation
    sim_obj = sim(measure_without_gates).qubits(10).quantum(state_vector()).seed(42).build()

    # First run - all measurements should be 0 since qubit starts in |0⟩
    results1 = sim_obj.run(100)
    measurements1 = results1.to_dict().get("measurements", [])
    assert all(
        m == 0 or m == [0] or m == (0,) or m is False for m in measurements1
    ), f"Expected all measurements to be 0, got: {measurements1[:5]}..."

    # Reset the simulator
    sim_obj.reset()

    # After reset, qubit should be back in |0⟩, so all measurements should still be 0
    results2 = sim_obj.run(100)
    measurements2 = results2.to_dict().get("measurements", [])
    assert all(
        m == 0 or m == [0] or m == (0,) or m is False for m in measurements2
    ), f"After reset, expected all measurements to be 0, got: {measurements2[:5]}..."
